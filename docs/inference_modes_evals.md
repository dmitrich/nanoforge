# Inference Modes

## Overview

The inference script supports three modes:
1. **Batch mode**: Generate from predefined prompts (default)
2. **Interactive mode**: Prompt user for input in real-time
3. **Evals mode**: Run evaluations from evals.json (placeholder for future)

## Configuration

Add these fields to the `input` section of your inference config:

```json
{
  "input": {
    "interactive": false,  // true = interactive mode
    "evals": false,        // true = evals mode
    "prompts": [...]       // used in batch mode
  }
}
```

## Mode 1: Batch Mode (Default)

**Use case**: Generate text from a predefined list of prompts

**Config**: `configs/infer_batch.json`

```json
{
  "input": {
    "mode": "batch",
    "interactive": false,
    "evals": false,
    "prompts": [
      "Once upon a time",
      "The little girl",
      "In a magical forest"
    ]
  }
}
```

**Usage**:
```bash
python src/infer.py configs/infer_batch.json
```

**Output**:
```
Infer ID: infer_batch_0301_143022
Checkpoint: runs/train/run_exp001_0301_101611/checkpoints/best.pt
Mode: Batch
Generating 3 prompt(s), max_new_tokens=200

======================================================================
Once upon a time there was a little girl named Lily...
----------------------------------------------------------------------
The little girl went to the park with her mom...
----------------------------------------------------------------------
In a magical forest, there lived many animals...
----------------------------------------------------------------------

Inference complete. Run: infer_batch_0301_143022
```

**Features**:
- Processes all prompts automatically
- Saves results to `generations.jsonl`
- No user interaction required
- Good for batch processing

## Mode 2: Interactive Mode

**Use case**: Chat with the model, enter prompts dynamically

**Config**: `configs/infer_interactive.json`

```json
{
  "input": {
    "mode": "interactive",
    "interactive": true,
    "evals": false
  }
}
```

**Usage**:
```bash
python src/infer.py configs/infer_interactive.json
```

**Output**:
```
Infer ID: infer_interactive_0301_143530
Checkpoint: runs/train/run_exp001_0301_101611/checkpoints/best.pt
Mode: Interactive

Interactive Mode - Enter prompts (Ctrl+C or 'quit' to exit)
Generation settings: max_tokens=200, temperature=0.8, top_k=40
======================================================================

Prompt: Once upon a time

Generated:
----------------------------------------------------------------------
Once upon a time there was a little girl named Lily. She loved to 
play outside in the sunshine...
----------------------------------------------------------------------

Prompt: Tell me a story about a cat

Generated:
----------------------------------------------------------------------
There was a fluffy cat named Whiskers. Whiskers liked to chase mice...
----------------------------------------------------------------------

Prompt: quit

Exiting interactive mode...

Inference complete. Run: infer_interactive_0301_143530
```

**Features**:
- Enter prompts one at a time
- See results immediately
- Continue until you type 'quit', 'exit', 'q', or press Ctrl+C
- All generations saved to `generations.jsonl`
- Good for experimentation and demos

**Commands**:
- Type any text to generate
- Type `quit`, `exit`, or `q` to exit
- Press `Ctrl+C` to exit
- Empty prompt is skipped

## Mode 3: Evals Mode (Placeholder)

**Use case**: Run structured evaluations using deepeval format

**Config**: `configs/infer_evals.json`

```json
{
  "input": {
    "mode": "evals",
    "interactive": false,
    "evals": true,
    "evals_file": "evals.json"
  }
}
```

**Status**: ⚠️ **Not yet implemented** - placeholder for future functionality

**Usage**:
```bash
python src/infer.py configs/infer_evals.json
```

**Output**:
```
Infer ID: infer_evals_0301_144015
Checkpoint: runs/train/run_exp001_0301_101611/checkpoints/best.pt
Mode: Batch
Evals: Enabled (evals.json)
Note: Evals functionality not yet implemented

Inference complete. Run: infer_evals_0301_144015
```

**Future functionality**:
- Load test cases from `evals.json`
- Run inference on each test case
- Compute metrics (coherence, fluency, etc.)
- Generate evaluation report
- Integration with deepeval framework

## evals.json Format

Template for future deepeval integration:

```json
{
  "test_cases": [
    {
      "id": "test_001",
      "input": "Once upon a time",
      "expected_output": null,
      "metrics": ["coherence", "fluency"],
      "metadata": {
        "category": "story_generation",
        "difficulty": "easy"
      }
    }
  ],
  "config": {
    "framework": "deepeval",
    "version": "1.0",
    "description": "Evaluation test cases"
  }
}
```

**Fields**:
- `id`: Unique test case identifier
- `input`: Prompt to generate from
- `expected_output`: Expected generation (optional, for comparison)
- `metrics`: List of metrics to compute
- `metadata`: Additional information about the test case

## Comparison

| Feature | Batch | Interactive | Evals |
|---------|-------|-------------|-------|
| **Prompts** | Predefined | User input | From file |
| **User interaction** | None | Required | None |
| **Use case** | Automation | Experimentation | Evaluation |
| **Output** | All at once | One at a time | Report |
| **Status** | ✅ Implemented | ✅ Implemented | ⚠️ Placeholder |

## Examples

### Example 1: Quick Test (Interactive)

```bash
# Start interactive mode
python src/infer.py configs/infer_interactive.json

# Try different prompts
Prompt: Hello, my name is
Prompt: Once upon a time in a land far away
Prompt: The quick brown fox
Prompt: quit
```

### Example 2: Batch Generation

```bash
# Generate from multiple prompts
python src/infer.py configs/infer_batch.json

# Results saved to runs/infer/infer_batch_*/generations.jsonl
cat runs/infer/infer_batch_*/generations.jsonl | jq .
```

### Example 3: Custom Batch Config

Create `configs/my_prompts.json`:
```json
{
  "meta": {"infer_name": "my_prompts"},
  "source": {"run_id": null, "checkpoint": "best"},
  "model": {"$from_run": true},
  "tokenizer": {"$from_run": true},
  "generation": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_k": 50
  },
  "input": {
    "interactive": false,
    "evals": false,
    "prompts": [
      "My custom prompt 1",
      "My custom prompt 2"
    ]
  }
}
```

Run:
```bash
python src/infer.py configs/my_prompts.json
```

## Configuration Reference

### input.interactive

**Type**: boolean  
**Default**: false  
**Values**:
- `true`: Interactive mode (prompt user for input)
- `false`: Batch mode (use prompts from config)

**Example**:
```json
{"input": {"interactive": true}}
```

### input.evals

**Type**: boolean  
**Default**: false  
**Values**:
- `true`: Evals mode (load from evals.json)
- `false`: Normal mode

**Example**:
```json
{"input": {"evals": true}}
```

**Note**: Currently a placeholder. When enabled, shows a message and exits.

### input.evals_file

**Type**: string  
**Default**: "evals.json"  
**Description**: Path to evals file (for future use)

**Example**:
```json
{"input": {"evals": true, "evals_file": "my_evals.json"}}
```

### input.prompts

**Type**: array of strings  
**Default**: ["\n"]  
**Description**: List of prompts for batch mode

**Example**:
```json
{
  "input": {
    "prompts": [
      "Once upon a time",
      "The little girl",
      "In a magical forest"
    ]
  }
}
```

## Output Files

All modes save results to the same location:

```
runs/infer/<infer_id>/
├── manifest.json          # Run metadata
├── resolved_infer.json    # Full config
├── generations.jsonl      # Generated texts
└── summary.json          # Summary stats
```

**generations.jsonl format**:
```json
{"prompt": "Once upon a time", "generated": "Once upon a time there was..."}
{"prompt": "The little girl", "generated": "The little girl went to..."}
```

## Tips

### Interactive Mode Tips

1. **Quick exit**: Type `q` or `quit`
2. **Force exit**: Press `Ctrl+C`
3. **Empty prompts**: Just press Enter to skip
4. **Long prompts**: Paste multi-line text (will be treated as one prompt)

### Batch Mode Tips

1. **Many prompts**: Put them in a separate file and load via config
2. **Reuse prompts**: Save config for reproducibility
3. **Different settings**: Create multiple configs with different generation params

### Evals Mode Tips (Future)

1. **Test cases**: Organize by category in evals.json
2. **Metrics**: Choose appropriate metrics for your use case
3. **Baseline**: Run evals on multiple checkpoints to compare

## Troubleshooting

### Issue: Interactive mode not accepting input

**Solution**: Make sure you're running in a terminal, not redirecting stdin

### Issue: Batch mode not finding prompts

**Check**: `input.prompts` is defined in config

### Issue: Evals mode does nothing

**Expected**: Evals functionality not yet implemented. Shows message and exits.

## Summary

**Quick reference**:

```bash
# Interactive mode
python src/infer.py configs/infer_interactive.json

# Batch mode
python src/infer.py configs/infer_batch.json

# Evals mode (placeholder)
python src/infer.py configs/infer_evals.json
```

**Config flags**:
```json
{
  "input": {
    "interactive": false,  // true for interactive mode
    "evals": false,        // true for evals mode (not implemented)
    "prompts": [...]       // for batch mode
  }
}
```
