# Disabling Logging: Configuration Guide

## Overview

You can now disable TensorBoard and/or metrics.jsonl logging through configuration files. This is useful for:
- Maximum training speed
- Reducing disk I/O
- Minimizing storage usage
- Benchmarking performance

## Configuration Options

Add these flags to the `observe` section of your training config:

```json
{
  "observe": {
    "log_interval": 10,
    "disable_tensorboard": false,  // Set to true to disable TensorBoard
    "disable_jsonl": false          // Set to true to disable metrics.jsonl
  }
}
```

## Option 1: Disable metrics.jsonl Only

**Use case**: You only want TensorBoard visualization, not text logs

**Config**:
```json
{
  "observe": {
    "log_interval": 10,
    "disable_tensorboard": false,
    "disable_jsonl": true
  }
}
```

**Result**:
- ✅ TensorBoard works (`tb/events.out.tfevents.*` created)
- ❌ No `metrics.jsonl` file
- ✅ Slightly faster (~1% speedup)
- ⚠️ Can't use scripts to analyze metrics

**Example config**: `configs/run_tensorboard_only.json`

**Run**:
```bash
python src/train.py configs/run_tensorboard_only.json
```

## Option 2: Disable TensorBoard Only

**Use case**: You only want text logs for scripting, not visualization

**Config**:
```json
{
  "observe": {
    "log_interval": 10,
    "disable_tensorboard": true,
    "disable_jsonl": false
  }
}
```

**Result**:
- ❌ No TensorBoard files (`tb/` directory empty or not created)
- ✅ `metrics.jsonl` created
- ✅ Slightly faster (~1-2% speedup)
- ⚠️ Can't use TensorBoard visualization

**Example config**: `configs/run_jsonl_only.json`

**Run**:
```bash
python src/train.py configs/run_jsonl_only.json
```

## Option 3: Disable Both (Maximum Speed)

**Use case**: Benchmarking, throwaway experiments, maximum speed

**Config**:
```json
{
  "observe": {
    "log_interval": 100,
    "disable_tensorboard": true,
    "disable_jsonl": true
  }
}
```

**Result**:
- ❌ No TensorBoard files
- ❌ No `metrics.jsonl`
- ✅ Fastest training (~2-3% speedup)
- ⚠️ No monitoring capability at all
- ⚠️ Only console output and final summary

**Example config**: `configs/run_no_logging.json`

**Run**:
```bash
python src/train.py configs/run_no_logging.json
```

## Option 4: Default (Both Enabled)

**Use case**: Normal training with full monitoring

**Config**:
```json
{
  "observe": {
    "log_interval": 10,
    "disable_tensorboard": false,
    "disable_jsonl": false
  }
}
```

Or simply omit the flags (defaults to false):
```json
{
  "observe": {
    "log_interval": 10
  }
}
```

**Result**:
- ✅ TensorBoard works
- ✅ `metrics.jsonl` created
- ✅ Full monitoring capability
- ⚠️ Slightly slower (~2-3% overhead)

**Example config**: `configs/run_exp001.json`

## Performance Impact

| Configuration | TensorBoard | metrics.jsonl | Speed Impact | Use Case |
|--------------|-------------|---------------|--------------|----------|
| **Both enabled** (default) | ✅ | ✅ | Baseline | Normal training |
| **Disable JSONL** | ✅ | ❌ | +1% | TensorBoard only |
| **Disable TensorBoard** | ❌ | ✅ | +1-2% | Scripting only |
| **Disable both** | ❌ | ❌ | +2-3% | Maximum speed |

**Note**: Impact is minimal. Only disable if you're optimizing for absolute maximum speed.

## What You Lose

### Without metrics.jsonl

❌ **Can't**:
- Parse metrics with `jq`
- Write analysis scripts easily
- Export to CSV/other formats
- Process metrics programmatically

✅ **Can still**:
- View in TensorBoard
- See console output
- Check final summary
- Access manifest.json results

### Without TensorBoard

❌ **Can't**:
- View interactive plots
- Compare runs visually
- Monitor training in real-time
- Use TensorBoard features

✅ **Can still**:
- Parse metrics.jsonl
- Write custom plots
- See console output
- Check final summary

### Without Both

❌ **Can't**:
- Monitor training progress
- Analyze metrics after training
- Debug training dynamics
- Compare runs easily

✅ **Can still**:
- See console output during training
- Check final summary at end
- View manifest.json results
- Use checkpoints for inference

## Example Configurations

### Maximum Speed (No Logging)

`configs/run_no_logging.json`:
```json
{
  "training": {
    "eval_interval": 999999,
    "eval_steps": 1,
    "checkpoint_interval": 999999
  },
  "observe": {
    "log_interval": 100,
    "disable_tensorboard": true,
    "disable_jsonl": true
  }
}
```

**Expected speedup**: ~30% faster than default

### TensorBoard Only

`configs/run_tensorboard_only.json`:
```json
{
  "training": {
    "eval_interval": 100,
    "eval_steps": 10,
    "checkpoint_interval": 500
  },
  "observe": {
    "log_interval": 10,
    "disable_tensorboard": false,
    "disable_jsonl": true
  }
}
```

**Use when**: You prefer visual monitoring over scripting

### JSONL Only

`configs/run_jsonl_only.json`:
```json
{
  "training": {
    "eval_interval": 100,
    "eval_steps": 10,
    "checkpoint_interval": 500
  },
  "observe": {
    "log_interval": 10,
    "disable_tensorboard": true,
    "disable_jsonl": false
  }
}
```

**Use when**: You prefer scripting over visual monitoring

## Verification

### Check if metrics.jsonl is created

```bash
# Run training
python src/train.py configs/run_tensorboard_only.json

# Check for metrics.jsonl
ls runs/train/run_tensorboard_only_*/metrics.jsonl
# Should show: No such file or directory (if disabled)
```

### Check if TensorBoard files are created

```bash
# Run training
python src/train.py configs/run_jsonl_only.json

# Check for TensorBoard files
ls runs/train/run_jsonl_only_*/tb/
# Should be empty or not exist (if disabled)
```

### Check console output

Both disabled:
```bash
python src/train.py configs/run_no_logging.json

# You'll still see:
# - Initial setup messages
# - Console logs every log_interval steps
# - Final training summary
```

## Recommendations

### For Development
**Use**: Both enabled (default)
- Need full monitoring
- Want to debug issues
- Compare experiments

### For Hyperparameter Search
**Use**: JSONL only (`disable_tensorboard: true`)
- Need programmatic access
- Don't need visualization
- Running many experiments

### For Benchmarking
**Use**: Both disabled
- Only care about final speed
- Don't need monitoring
- Throwaway experiments

### For Production Training
**Use**: Both enabled (default)
- Need full audit trail
- Want to monitor progress
- May need to debug later

## Migration Guide

### Updating Existing Configs

Add to your existing config files:

```json
{
  "observe": {
    "log_interval": 10,
    "disable_tensorboard": false,  // Add this line
    "disable_jsonl": false          // Add this line
  }
}
```

### Backward Compatibility

If you don't add these flags:
- **Default behavior**: Both enabled (same as before)
- **No breaking changes**: Existing configs work as-is
- **Opt-in**: Only disable if you explicitly set to `true`

## Troubleshooting

### Issue: metrics.jsonl still created even with disable_jsonl: true

**Check**:
1. Config file has correct syntax
2. Using latest `src/tracker.py`
3. No typos in flag name

**Verify config is loaded**:
```bash
jq '.observe' configs/your_config.json
# Should show: "disable_jsonl": true
```

### Issue: TensorBoard still creates files even with disable_tensorboard: true

**Check**:
1. Config file has correct syntax
2. Using latest `src/tracker.py`
3. No typos in flag name

**Verify**:
```bash
jq '.observe' configs/your_config.json
# Should show: "disable_tensorboard": true
```

### Issue: Want to disable but keep some monitoring

**Solution**: Use higher `log_interval` instead of disabling:

```json
{
  "observe": {
    "log_interval": 100,  // Log 10x less often
    "disable_tensorboard": false,
    "disable_jsonl": false
  }
}
```

This reduces overhead while keeping monitoring capability.

## Summary

### Quick Reference

**Disable metrics.jsonl**:
```json
{"observe": {"disable_jsonl": true}}
```

**Disable TensorBoard**:
```json
{"observe": {"disable_tensorboard": true}}
```

**Disable both**:
```json
{"observe": {"disable_tensorboard": true, "disable_jsonl": true}}
```

### Example Configs

- `configs/run_no_logging.json` - Both disabled
- `configs/run_tensorboard_only.json` - JSONL disabled
- `configs/run_jsonl_only.json` - TensorBoard disabled
- `configs/run_exp001.json` - Both enabled (default)

### When to Disable

**Disable metrics.jsonl when**:
- You only use TensorBoard
- You want slightly faster training
- You don't write analysis scripts

**Disable TensorBoard when**:
- You only use scripts for analysis
- You want slightly faster training
- You don't need visualization

**Disable both when**:
- Benchmarking performance
- Maximum speed is critical
- Throwaway experiments
- You only need final results

**Keep both enabled when**:
- Normal training
- Need full monitoring
- Want flexibility
- Production models
