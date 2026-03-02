import math
import sys
import time
import torch
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RunConfig
from dataloader import build_dataloaders
from manifest import generate_run_id, create_manifest, complete_manifest, fail_manifest
from model import GPT, ModelConfig
from tokenizer import Tokenizer
from tracker import Tracker
from utils import get_device, set_seed, write_json


def get_lr(step: int, training: dict) -> float:
    lr       = training['learning_rate']
    warmup   = training.get('warmup_steps', 200)
    max_steps = training['max_steps']
    min_lr   = lr * 0.1

    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    decay  = (step - warmup) / max(max_steps - warmup, 1)
    coeff  = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (lr - min_lr)


def build_optimizer(model: GPT, training: dict):
    return torch.optim.AdamW(
        model.parameters(),
        lr=training['learning_rate'],
        betas=(training.get('beta1', 0.9), training.get('beta2', 0.95)),
        weight_decay=training.get('weight_decay', 0.1),
    )


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_steps, device):
    model.eval()
    results = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        it     = iter(loader)
        losses = []
        for _ in range(eval_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def run_training(config_path: str):
    run_cfg = RunConfig.load(config_path)
    run_cfg.validate()

    env    = run_cfg.environment
    device = get_device(env.get('device', 'cpu'))
    set_seed(env.get('seed', 42))

    run_id  = generate_run_id(run_cfg.run_name)
    run_dir = Path('runs/train') / run_id
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve + save config before training loop starts
    resolved = run_cfg.resolve(run_id)
    write_json(run_dir / 'resolved_run.json', resolved)

    lineage = {
        'tok_id':       run_cfg.tokenizer['tok_id'],
        'ds_id':        run_cfg.dataset['ds_id'],
        'parent_run_id': None,
    }
    create_manifest(run_dir, run_id, 'train', resolved, lineage)

    # Build data
    train_loader, val_loader = build_dataloaders(
        dataset_cfg={**run_cfg.dataset, 'max_seq_len': run_cfg.dataset['max_seq_len']},
        training_cfg={**run_cfg.training, 'num_workers': env.get('num_workers', 0)},
    )

    # Build model
    model_cfg = ModelConfig.from_dict(run_cfg.model)
    model     = GPT(model_cfg).to(device)
    if env.get('dtype') == 'bfloat16':
        model = model.to(torch.bfloat16)

    optimizer = build_optimizer(model, run_cfg.training)
    tracker   = Tracker(run_cfg.observe, run_dir)
    tracker.log_config(resolved)

    training      = run_cfg.training
    max_steps     = training['max_steps']
    eval_interval = training['eval_interval']
    eval_steps    = training['eval_steps']
    ckpt_interval = training.get('checkpoint_interval', 1000)
    grad_clip     = training.get('grad_clip', 1.0)
    log_interval  = run_cfg.observe.get('log_interval', 10)

    print(f"Run ID:  {run_id}")
    print(f"Run dir: runs/train/{run_id}")
    print(f"Device:  {device}  |  Params: {model.num_parameters()/1e6:.1f}M  |  Steps: {max_steps}")
    
    # Print model dimensions summary
    B = training['batch_size']
    T = run_cfg.model['block_size']
    C = run_cfg.model['n_embd']
    V = run_cfg.model['vocab_size']
    
    print(f"\n{'='*70}")
    print(f"Model Dimensions Summary")
    print(f"{'='*70}")
    print(f"  B (Batch size):        {B:>8,}")
    print(f"  T (Block size):        {T:>8,}")
    print(f"  C (Embedding dim):     {C:>8,}")
    print(f"  V (Vocab size):        {V:>8,}")
    print(f"  {'─'*66}")
    print(f"  B × T × C:             {B*T*C:>8,}  (activations per batch)")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    train_iter    = iter(train_loader)
    loss          = torch.tensor(0.0)
    
    # Training metrics tracking
    training_start_time = time.time()
    tokens_per_batch = training['batch_size'] * run_cfg.dataset['max_seq_len']
    total_tokens_trained = 0

    model.train()
    try:
        for step in range(max_steps):
            lr = get_lr(step, training)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            # Track tokens processed
            total_tokens_trained += tokens_per_batch

            if step % log_interval == 0:
                print(f"step {step:>5}: loss {loss.item():.4f}  lr {lr:.2e}")
                tracker.log_metric('Loss/train_step', loss.item(), step)
                tracker.log_metric('LR', lr, step)

            if step % eval_interval == 0 or step == max_steps - 1:
                ev = estimate_loss(model, train_loader, val_loader, eval_steps, device)
                print(f"         eval  train={ev['train']:.4f}  val={ev['val']:.4f}")
                tracker.log_metric('Loss/train_eval', ev['train'], step)
                tracker.log_metric('Loss/val',        ev['val'],   step)

                if ev['val'] < best_val_loss:
                    best_val_loss = ev['val']
                    model.save_checkpoint(ckpt_dir / 'best.pt', step, optimizer, best_val_loss)
                    print(f"         -> best.pt  val={best_val_loss:.4f}")

            if step > 0 and step % ckpt_interval == 0:
                model.save_checkpoint(ckpt_dir / 'latest.pt', step, optimizer, loss.item())

        model.save_checkpoint(ckpt_dir / 'latest.pt', max_steps, optimizer, loss.item())

        # Calculate training metrics
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        tokens_per_second = total_tokens_trained / total_training_time if total_training_time > 0 else 0
        
        summary = {
            'best_val_loss': best_val_loss,
            'final_loss':    loss.item(),
            'steps':         max_steps,
            'total_tokens_trained': total_tokens_trained,
            'total_training_time_seconds': round(total_training_time, 2),
            'tokens_per_second': round(tokens_per_second, 2),
        }
        
        # Update resolved config with training metrics
        resolved['_resolved']['total_tokens_trained'] = total_tokens_trained
        resolved['_resolved']['total_training_time_seconds'] = round(total_training_time, 2)
        resolved['_resolved']['tokens_per_second'] = round(tokens_per_second, 2)
        resolved['_resolved']['training_completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        write_json(run_dir / 'resolved_run.json', resolved)
        
        tracker.finish(summary)
        complete_manifest(run_dir, summary)
        
        # Print training statistics
        print(f"\n{'='*70}")
        print(f"Training Complete: {run_id}")
        print(f"{'='*70}")
        print(f"Total tokens trained:  {total_tokens_trained:,}")
        print(f"Total training time:   {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        print(f"Training speed:        {tokens_per_second:,.2f} tokens/second")
        print(f"Best validation loss:  {best_val_loss:.4f}")
        print(f"Final training loss:   {loss.item():.4f}")
        print(f"{'='*70}")

    except Exception as e:
        fail_manifest(run_dir, str(e))
        raise


if __name__ == '__main__':
    run_training(sys.argv[1])
