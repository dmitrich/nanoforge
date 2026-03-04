import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import write_json, append_jsonl, read_jsonl, hash_dict

REGISTRY = Path('runs/registry.jsonl')


def generate_run_id(run_name: str) -> str:
    ts = datetime.now().strftime('%m%d_%H%M%S')
    return f"{run_name}_{ts}"


def create_manifest(run_dir, run_id, run_type, config, lineage=None):
    run_dir = Path(run_dir)
    manifest = {
        'run_id':      run_id,
        'run_type':    run_type,
        'run_name':    config.get('meta', {}).get('run_name', run_name_from_id(run_id)),
        'created_at':  datetime.now().isoformat(),
        'status':      'running',
        'config_hash': hash_dict(config),
        'lineage':     lineage or {},
        'outputs': {
            'best_checkpoint':  'checkpoints/best.safetensors',
            'latest_checkpoint': 'checkpoints/latest.safetensors',
            'resolved_config':  'resolved_run.json',
            'metrics':          'metrics.jsonl',
        },
        'summary': {},
    }
    write_json(run_dir / 'manifest.json', manifest)
    append_jsonl(REGISTRY, {
        'run_id':   run_id,
        'run_type': run_type,
        'run_dir':  str(run_dir),
    })
    return manifest


def run_name_from_id(run_id):
    parts = run_id.rsplit('_', 2)
    return parts[0] if len(parts) >= 3 else run_id


def complete_manifest(run_dir, summary: dict):
    path = Path(run_dir) / 'manifest.json'
    with open(path) as f:
        m = json.load(f)
    m['status']       = 'completed'
    m['completed_at'] = datetime.now().isoformat()
    m['summary']      = summary
    write_json(path, m)


def fail_manifest(run_dir, error: str):
    path = Path(run_dir) / 'manifest.json'
    with open(path) as f:
        m = json.load(f)
    m['status']    = 'failed'
    m['failed_at'] = datetime.now().isoformat()
    m['error']     = str(error)
    write_json(path, m)


def rebuild_registry():
    entries = []
    for run_type in ('train', 'infer'):
        base = Path('runs') / run_type
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            mf = run_dir / 'manifest.json'
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                entries.append({
                    'run_id':   m['run_id'],
                    'run_type': run_type,
                    'run_dir':  str(run_dir),
                })
    with open(REGISTRY, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')
    print(f"Registry rebuilt: {len(entries)} runs")


def list_runs(run_type=None):
    if not REGISTRY.exists():
        print("No runs found.")
        return
    runs = read_jsonl(REGISTRY)
    if run_type:
        runs = [r for r in runs if r['run_type'] == run_type]
    print(f"{'run_id':<45} {'type':<8} run_dir")
    print('-' * 80)
    for r in runs:
        print(f"{r['run_id']:<45} {r['run_type']:<8} {r['run_dir']}")


def verify_run(run_dir):
    run_dir = Path(run_dir)
    mf_path = run_dir / 'manifest.json'
    with open(mf_path) as f:
        m = json.load(f)
    resolved_path = run_dir / 'resolved_run.json'
    with open(resolved_path) as f:
        resolved = json.load(f)
    current_hash = hash_dict(resolved)
    stored_hash  = m.get('config_hash', '')
    ok = current_hash == stored_hash
    print(f"Run:    {m['run_id']}")
    print(f"Status: {m['status']}")
    print(f"Hash:   {'OK' if ok else f'MISMATCH (stored={stored_hash}, current={current_hash})'}")
    for name, rel in m.get('outputs', {}).items():
        p = run_dir / rel
        print(f"  {name}: {'OK' if p.exists() else 'MISSING'} ({rel})")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'list'
    if cmd == 'list':
        list_runs()
    elif cmd == 'rebuild':
        rebuild_registry()
    elif cmd == 'verify':
        verify_run(sys.argv[2])
