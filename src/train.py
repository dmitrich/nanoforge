import math
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RunConfig
from dataloader import build_dataloaders
from manifest import generate_run_id, create_manifest, complete_manifest, fail_manifest
from model import GPT, ModelConfig
from tokenizer import Tokenizer
from observability import (
    CHECKPOINT_EVENT_NAME,
    TRAINING_EVAL_SPAN_NAME,
    TRAINING_ROOT_SPAN_NAME,
    build_common_tags,
    build_common_trace_metadata,
    flush_langfuse,
    get_langfuse_client,
)
from tracker import Tracker
from utils import get_device, set_seed, write_json


def get_lr(step: int, training: dict) -> float:
    lr = training['learning_rate']

    if training.get('scheduler', 'cosine') == 'const':
        return lr

    warmup    = training.get('warmup_steps', 200)
    max_steps = training['max_steps']
    min_lr    = lr * 0.1

    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    decay = (step - warmup) / max(max_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
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
    langfuse = get_langfuse_client(enabled=run_cfg.observability.get('langfuse', False))
    train_trace = langfuse.trace(
        id=run_id,
        name=TRAINING_ROOT_SPAN_NAME,
        session_id=run_id,
        input={"config_path": config_path, "resolved": resolved},
        metadata=build_common_trace_metadata(
            experiment_id=run_id,
            run_type="training",
            source_run_id=run_id,
            extra={
                "dataset.id": run_cfg.dataset["ds_id"],
                "tokenizer.id": run_cfg.tokenizer["tok_id"],
                "device": device,
                "model.architecture": run_cfg.model["architecture"],
                "model.layers": run_cfg.model["n_layer"],
                "model.heads": run_cfg.model["n_head"],
                "model.embedding_dim": run_cfg.model["n_embd"],
                "training.max_steps": run_cfg.training["max_steps"],
                "training.batch_size": run_cfg.training["batch_size"],
                "training.eval_interval": run_cfg.training["eval_interval"],
                "training.scheduler": run_cfg.training.get("scheduler", "cosine"),
                "training.learning_rate": run_cfg.training["learning_rate"],
                "observe.log_interval": run_cfg.observability.get("log_interval", 10),
            },
        ),
        tags=build_common_tags("training", run_id, run_cfg.meta.get("run_name")),
    )
    training_span = langfuse.span(
        trace_id=train_trace.id,
        name=TRAINING_ROOT_SPAN_NAME,
        input={"config_path": config_path},
        metadata={
            "run_dir": str(run_dir),
            "checkpoint_dir": str(ckpt_dir),
            "seed": env.get("seed", 42),
            "dtype": env.get("dtype", "float32"),
        },
    )

    # Build data
    train_loader, val_loader = build_dataloaders(
        dataset_cfg=run_cfg.dataset,
        training_cfg={**run_cfg.training, 'num_workers': env.get('num_workers', 0)},
    )

    # Build model
    model_cfg = ModelConfig.from_dict(run_cfg.model)
    model     = GPT(model_cfg).to(device)
    if env.get('dtype') == 'bfloat16':
        model = model.to(torch.bfloat16)

    optimizer = build_optimizer(model, run_cfg.training)
    tracker   = Tracker(run_cfg.observability, run_dir)
    tracker.log_config(resolved)
    if run_cfg.observability.get('tensorboard', True):
        print(f"TensorBoard is on — run: tensorboard --logdir runs/train")

    training      = run_cfg.training
    max_steps     = training['max_steps']
    eval_interval = training['eval_interval']
    eval_steps    = training['eval_steps']
    ckpt_interval = training.get('checkpoint_interval', 1000)
    grad_clip     = training.get('grad_clip', 1.0)
    log_interval  = run_cfg.observability.get('log_interval', 10)

    scheduler = training.get('scheduler', 'cosine')
    lr        = training['learning_rate']
    if scheduler == 'const':
        lr_desc = f"LR: {lr:.2e} (constant)"
    else:
        lr_desc = f"LR: {lr:.2e} → {lr*0.1:.2e}  warmup: {training.get('warmup_steps', 200)} steps"

    print(f"Run ID:  {run_id}")
    print(f"Run dir: runs/train/{run_id}")
    print(f"Device:  {device}  |  Params: {model.num_parameters()/1e6:.1f}M  |  Steps: {max_steps}")
    print(f"Optimizer: AdamW  |  Scheduler: {scheduler}  |  {lr_desc}")
    
    # Print model dimensions summary
    B = training['batch_size']
    T = run_cfg.model['block_size']
    C = run_cfg.model['n_embd']
    V = run_cfg.model['vocab_size']
    L = run_cfg.model['n_layer']
    H = run_cfg.model['n_head']

    dtype_str      = env.get('dtype', 'float32')
    bytes_per_elem = 2 if dtype_str in ('bfloat16', 'float16') else 4
    kv_cache_mb    = L * 2 * T * C * bytes_per_elem / (1024 * 1024)
    n_params       = model.num_parameters()

    print(f"\n{'='*70}")
    print(f"Model Dimensions Summary")
    print(f"{'='*70}")
    print(f"  {'Dim':<6}  {'Meaning':<28}  {'Value':>10}")
    print(f"  {'─'*50}")
    print(f"  {'B':<6}  {'Batch size':<28}  {B:>10,}")
    print(f"  {'T':<6}  {'Block size (tokens)':<28}  {T:>10,}")
    print(f"  {'C':<6}  {'Embedding dim':<28}  {C:>10,}")
    print(f"  {'L':<6}  {'Transformer blocks':<28}  {L:>10,}")
    print(f"  {'H':<6}  {'Attention heads':<28}  {H:>10,}")
    print(f"  {'V':<6}  {'Vocab size':<28}  {V:>10,}")
    print(f"  {'─'*50}")
    print(f"  {'B×T×C':<6}  {'Activations per batch':<28}  {B*T*C:>10,}")
    print(f"  {'KV':<6}  {f'KV cache per seq ({dtype_str})':<28}  {kv_cache_mb:>9.1f} MB")
    print(f"  {'N':<6}  {'Total parameters':<28}  {n_params:>10,}")
    print(f"  {'':6}  2VC + TC + L(12C² + 10C) + V + 2C")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    train_iter    = iter(train_loader)
    loss          = torch.tensor(0.0)
    
    # Training metrics tracking
    training_start_time = time.time()
    tokens_per_batch = training['batch_size'] * run_cfg.model['block_size']
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

            should_log  = (step % log_interval == 0)
            should_eval = (step % eval_interval == 0 or step == max_steps - 1)

            ev = None
            if should_eval:
                ev = estimate_loss(model, train_loader, val_loader, eval_steps, device)
                tracker.log_metric('Loss/train_eval', ev['train'], step)
                tracker.log_metric('Loss/val',        ev['val'],   step)
                eval_span = langfuse.span(
                    trace_id=train_trace.id,
                    parent_observation_id=training_span.id,
                    name=TRAINING_EVAL_SPAN_NAME,
                    input={"step": step, "eval_steps": eval_steps, "lr": lr},
                    metadata={
                        "best_val_loss_before": best_val_loss,
                        "tokens_trained": total_tokens_trained,
                    },
                )
                if ev['val'] < best_val_loss:
                    best_val_loss = ev['val']
                    model.save_checkpoint(ckpt_dir / 'best.safetensors', step, optimizer, best_val_loss)
                    langfuse.span(
                        trace_id=train_trace.id,
                        parent_observation_id=training_span.id,
                        name=CHECKPOINT_EVENT_NAME,
                        input={"step": step},
                        output={"path": str(ckpt_dir / 'best.safetensors')},
                        metadata={
                            "kind": "best",
                            "val_loss": ev["val"],
                            "train_loss": ev["train"],
                            "tokens_trained": total_tokens_trained,
                        },
                    ).end()
                eval_span.end(output=ev)

            if should_log or should_eval:
                line = f"step {step:>5}: loss {loss.item():.4f}  lr {lr:.2e}"
                if ev is not None:
                    line += f"  | train={ev['train']:.4f}  val={ev['val']:.4f}"
                print(line)
                if should_log:
                    tracker.log_metric('Loss/train_step', loss.item(), step)
                    tracker.log_metric('LR', lr, step)

            if step > 0 and step % ckpt_interval == 0:
                model.save_checkpoint(ckpt_dir / 'latest.safetensors', step, optimizer, loss.item())
                langfuse.span(
                    trace_id=train_trace.id,
                    parent_observation_id=training_span.id,
                    name=CHECKPOINT_EVENT_NAME,
                    input={"step": step},
                    output={"path": str(ckpt_dir / 'latest.safetensors')},
                    metadata={
                        "kind": "latest",
                        "loss": loss.item(),
                        "tokens_trained": total_tokens_trained,
                    },
                ).end()

        model.save_checkpoint(ckpt_dir / 'latest.safetensors', max_steps, optimizer, loss.item())
        langfuse.span(
            trace_id=train_trace.id,
            parent_observation_id=training_span.id,
            name=CHECKPOINT_EVENT_NAME,
            input={"step": max_steps},
            output={"path": str(ckpt_dir / 'latest.safetensors')},
            metadata={
                "kind": "latest",
                "loss": loss.item(),
                "final": True,
                "tokens_trained": total_tokens_trained,
            },
        ).end()

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
        training_span.end(output=summary)
        train_trace.update(output=summary)
        flush_langfuse()
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
        training_span.end(
            level="ERROR",
            status_message=str(e),
            output={"error": str(e)},
        )
        train_trace.update(output={"error": str(e)})
        flush_langfuse()
        fail_manifest(run_dir, str(e))
        raise


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/train.json'
    run_training(config_path)
