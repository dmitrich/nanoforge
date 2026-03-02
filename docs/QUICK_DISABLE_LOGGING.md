# Quick Guide: Disable metrics.jsonl

## TL;DR

Add this to your config file to disable metrics.jsonl:

```json
{
  "observe": {
    "disable_jsonl": true
  }
}
```

## All Options

```json
{
  "observe": {
    "log_interval": 10,
    "disable_tensorboard": false,  // true = no TensorBoard
    "disable_jsonl": false          // true = no metrics.jsonl
  }
}
```

## Example Configs

### Disable metrics.jsonl only
```bash
python src/train.py configs/run_tensorboard_only.json
```

### Disable TensorBoard only
```bash
python src/train.py configs/run_jsonl_only.json
```

### Disable both (maximum speed)
```bash
python src/train.py configs/run_no_logging.json
```

## What Gets Disabled

| Config | TensorBoard | metrics.jsonl | Speed Gain |
|--------|-------------|---------------|------------|
| `disable_jsonl: true` | ✅ | ❌ | +1% |
| `disable_tensorboard: true` | ❌ | ✅ | +1-2% |
| Both `true` | ❌ | ❌ | +2-3% |

## Full Documentation

See `docs/disabling_logging.md` for complete details.
