import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import InferConfig
from manifest import generate_run_id, create_manifest, complete_manifest
from model import GPT
from tokenizer import Tokenizer
from utils import append_jsonl, get_device, write_json


def _find_latest_train_run() -> Path:
    runs = [p for p in Path('runs/train').iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.stat().st_mtime)[-1]


def run_inference(config_path: str = None, force_interactive: bool = False):
    if config_path is None:
        # No config: synthesise a minimal InferConfig from the latest training run
        run_dir  = _find_latest_train_run()
        resolved = json.load(open(run_dir / 'resolved_run.json'))

        cfg = InferConfig(
            meta={'infer_name': 'infer'},
            source={'run_id': run_dir.name, 'checkpoint': 'best'},
            model=resolved['model'],
            tokenizer=resolved['tokenizer'],
            generation={'max_new_tokens': 200, 'temperature': 1.0, 'top_k': 0},
            input={'prompts': ['\n'], 'interactive': False, 'evals': False},
        )
    else:
        cfg = InferConfig.load(config_path)

    infer_id  = generate_run_id(cfg.infer_name)
    infer_dir = Path('runs/infer') / infer_id
    infer_dir.mkdir(parents=True, exist_ok=True)

    resolved_infer = cfg.resolve(infer_id)
    write_json(infer_dir / 'resolved_infer.json', resolved_infer)
    create_manifest(infer_dir, infer_id, 'infer', resolved_infer, lineage={
        'from_run': cfg.source.get('run_id'),
    })

    # Load model
    ckpt_path     = cfg.checkpoint_path
    model, _ckpt  = GPT.from_checkpoint(ckpt_path)
    device        = get_device(cfg.model.get('device', 'mps'))
    model         = model.to(device).eval()

    # Load tokenizer
    tok_id    = cfg.tokenizer['tok_id']
    tokenizer = Tokenizer.load(tok_id)

    gen        = cfg.generation
    max_new    = gen.get('max_new_tokens', 200)
    temperature = gen.get('temperature', 1.0)
    top_k      = gen.get('top_k', 0)
    stop_strings = gen.get('stop_tokens', [])
    
    # Convert stop strings to token IDs
    stop_token_ids = []
    if stop_strings:
        stop_token_ids = tokenizer.get_stop_token_ids(stop_strings)
        print(f"Stop tokens: {stop_strings} -> IDs: {stop_token_ids}")
    
    # Check for interactive mode
    interactive = force_interactive or cfg.input.get('interactive', False)
    evals_enabled = cfg.input.get('evals', False)
    
    print(f"Infer ID: {infer_id}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Mode: {'Interactive' if interactive else 'Batch'}")
    
    # Handle evals mode (placeholder for future implementation)
    if evals_enabled:
        print(f"Evals: Enabled (evals.json)")
        print(f"Note: Evals functionality not yet implemented")
        # TODO: Load and process evals.json
        # evals_path = Path('evals.json')
        # if evals_path.exists():
        #     with open(evals_path) as f:
        #         evals_data = json.load(f)
        #     # Process evals using deepeval format
        return
    
    if interactive:
        # Interactive mode: prompt user for input
        print(f"\nInteractive Mode - Enter prompts (Ctrl+C or 'quit' to exit)")
        print(f"Generation settings: max_tokens={max_new}, temperature={temperature}, top_k={top_k}")
        print('=' * 70)
        
        try:
            while True:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    print("Empty prompt, skipping...")
                    continue
                
                tokens = tokenizer.encode(prompt)
                idx    = torch.tensor([tokens], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model.generate(idx, max_new, temperature=temperature, top_k=top_k, stop_token_ids=stop_token_ids)
                text = tokenizer.decode(out[0].tolist())
                
                print(f"\nGenerated:")
                print('-' * 70)
                print(text)
                print('-' * 70)
                
                # Save to file
                append_jsonl(infer_dir / 'generations.jsonl', {'prompt': prompt, 'generated': text})
        
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
        
        summary = {'mode': 'interactive', 'checkpoint': ckpt_path, 'infer_id': infer_id}
        write_json(infer_dir / 'summary.json', summary)
        complete_manifest(infer_dir, summary)
        print(f"\nInference complete. Run: {infer_id}")
    
    else:
        # Batch mode: use prompts from config
        prompts = cfg.input.get('prompts', ['\n'])
        print(f"Generating {len(prompts)} prompt(s), max_new_tokens={max_new}\n")
        print('=' * 70)

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            idx    = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(idx, max_new, temperature=temperature, top_k=top_k, stop_token_ids=stop_token_ids)
            text = tokenizer.decode(out[0].tolist())
            print(text)
            print('-' * 70)
            append_jsonl(infer_dir / 'generations.jsonl', {'prompt': prompt, 'generated': text})

        summary = {'mode': 'batch', 'num_prompts': len(prompts), 'checkpoint': ckpt_path, 'infer_id': infer_id}
        write_json(infer_dir / 'summary.json', summary)
        complete_manifest(infer_dir, summary)
        print(f"\nInference complete. Run: {infer_id}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='nanoforge inference')
    parser.add_argument('config', nargs='?', default=None,
                        help='path to infer config JSON (default: config/infer.json when --i is set)')
    parser.add_argument('--i', action='store_true', dest='interactive',
                        help='force interactive mode, overrides config setting')
    args = parser.parse_args()

    config_path = args.config
    if args.interactive and config_path is None:
        config_path = 'config/infer.json'

    run_inference(config_path, force_interactive=args.interactive)
