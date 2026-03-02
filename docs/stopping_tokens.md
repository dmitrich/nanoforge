# Stopping Tokens in Inference

## Overview

Stopping tokens allow generation to terminate early when specific tokens are encountered, rather than always generating the maximum number of tokens. This is useful for:
- Stopping at end-of-sequence markers (`</s>`)
- Stopping at paragraph breaks (`\n\n`)
- Stopping at custom delimiters
- Preventing unnecessary generation

## Configuration

Add `stop_tokens` to the `generation` section of your inference config:

```json
{
  "generation": {
    "max_new_tokens": 200,
    "temperature": 0.8,
    "top_k": 40,
    "stop_tokens": ["</s>", "\n\n", "END"]
  }
}
```

## How It Works

1. **Token Encoding**: Stop strings are converted to token IDs using the tokenizer
2. **Generation Loop**: After each token is generated, it's checked against stop token IDs
3. **Early Termination**: If a stop token is generated, generation stops immediately
4. **Fallback**: If no stop token is encountered, generation continues until `max_new_tokens`

## Examples

### Example 1: Stop at End-of-Sequence

```json
{
  "generation": {
    "max_new_tokens": 200,
    "stop_tokens": ["</s>"]
  }
}
```

**Behavior**:
- Generates up to 200 tokens
- Stops immediately if `</s>` token is generated
- Useful for models trained with explicit end markers

### Example 2: Stop at Paragraph Break

```json
{
  "generation": {
    "max_new_tokens": 500,
    "stop_tokens": ["\n\n"]
  }
}
```

**Behavior**:
- Generates up to 500 tokens
- Stops at double newline (paragraph break)
- Useful for generating single paragraphs

### Example 3: Multiple Stop Tokens

```json
{
  "generation": {
    "max_new_tokens": 300,
    "stop_tokens": ["</s>", "\n\n", "THE END", "###"]
  }
}
```

**Behavior**:
- Stops at any of the specified tokens
- First encountered stop token triggers termination
- Useful for flexible stopping conditions

### Example 4: No Stop Tokens (Default)

```json
{
  "generation": {
    "max_new_tokens": 200,
    "stop_tokens": []
  }
}
```

Or omit the field entirely:
```json
{
  "generation": {
    "max_new_tokens": 200
  }
}
```

**Behavior**:
- Always generates exactly `max_new_tokens` tokens
- No early stopping
- Original behavior

## Common Stop Tokens

### For Story Generation

```json
"stop_tokens": ["</s>", "\n\n", "The End"]
```

Stops at:
- End-of-sequence marker
- Paragraph breaks
- Story ending phrase

### For Dialogue

```json
"stop_tokens": ["</s>", "\n\n", "User:", "Assistant:"]
```

Stops at:
- End-of-sequence marker
- Paragraph breaks
- Turn-taking markers

### For Code Generation

```json
"stop_tokens": ["</s>", "\n\n", "```", "# End"]
```

Stops at:
- End-of-sequence marker
- Blank lines
- Code block delimiters
- Comment markers

### For Q&A

```json
"stop_tokens": ["</s>", "\n\n", "Question:", "Q:"]
```

Stops at:
- End-of-sequence marker
- Paragraph breaks
- Next question markers

## Technical Details

### Single-Token Stops

When a stop string encodes to a single token, stopping is exact:

```python
# "</s>" encodes to token ID 3
stop_tokens = ["</s>"]
# Generation stops immediately when token 3 is generated
```

### Multi-Token Stops

When a stop string encodes to multiple tokens, only the first token is used:

```python
# "The End" might encode to [123, 456]
stop_tokens = ["The End"]
# Generation stops when token 123 is generated
# (Approximation - may stop earlier than intended)
```

**Note**: For precise multi-token stopping, consider using single-token markers or implementing sequence matching (future enhancement).

### Token ID Lookup

Stop strings are converted to token IDs at inference start:

```python
tokenizer = Tokenizer.load('tok_bpe_8k')
stop_strings = ["</s>", "\n\n"]
stop_token_ids = tokenizer.get_stop_token_ids(stop_strings)
# Example output: [3, 198]
```

This is printed at inference start:
```
Stop tokens: ['</s>', '\n\n'] -> IDs: [3, 198]
```

## Configuration Examples

### Minimal Config

```json
{
  "meta": {"infer_name": "infer_with_stops"},
  "source": {"run_id": null, "checkpoint": "best"},
  "model": {"$from_run": true},
  "tokenizer": {"$from_run": true},
  "generation": {
    "max_new_tokens": 200,
    "temperature": 0.8,
    "top_k": 40,
    "stop_tokens": ["</s>"]
  },
  "input": {
    "interactive": false,
    "evals": false,
    "prompts": ["Once upon a time"]
  }
}
```

### Interactive with Stops

```json
{
  "meta": {"infer_name": "infer_interactive_stops"},
  "source": {"run_id": null, "checkpoint": "best"},
  "model": {"$from_run": true},
  "tokenizer": {"$from_run": true},
  "generation": {
    "max_new_tokens": 500,
    "temperature": 0.7,
    "top_k": 50,
    "stop_tokens": ["</s>", "\n\n", "THE END"]
  },
  "input": {
    "interactive": true,
    "evals": false
  }
}
```

## Output Examples

### Without Stop Tokens

```
Prompt: Once upon a time

Generated:
Once upon a time there was a little girl named Lily. She loved to play 
outside in the sunshine. One day she found a big red ball in the park. 
She kicked it very high into the sky. The ball went up and up and up. 
Then it came back down. Lily caught it and smiled. She was very happy. 
She played with the ball all day long. When the sun went down, she went 
home. Her mom gave her dinner. Then she went to bed. She dreamed about 
the red ball. The next day she went back to the park. She found the ball 
again. She played with it some more. She had so much fun. She loved that 
ball very much. It was her favorite toy. She played with it every day...
[continues for 200 tokens]
```

### With Stop Tokens (`["</s>", "\n\n"]`)

```
Prompt: Once upon a time

Generated:
Once upon a time there was a little girl named Lily. She loved to play 
outside in the sunshine. One day she found a big red ball in the park. 
She kicked it very high into the sky. The ball went up and up and up. 
Then it came back down. Lily caught it and smiled. She was very happy.

[stopped at \n\n after ~60 tokens]
```

## Troubleshooting

### Issue: Generation doesn't stop at expected token

**Possible causes**:
1. Stop string encodes to multiple tokens (only first token is checked)
2. Model never generates the stop token
3. Stop token ID is incorrect

**Solutions**:
1. Use single-token stop markers (e.g., `</s>` instead of `The End`)
2. Check token IDs in output: `Stop tokens: ['</s>'] -> IDs: [3]`
3. Verify tokenizer has the stop token in vocabulary

### Issue: Generation stops too early

**Possible causes**:
1. Stop token appears naturally in text
2. Multi-token stop string matches on first token only

**Solutions**:
1. Use more specific stop tokens
2. Remove ambiguous stop tokens
3. Increase `max_new_tokens` to allow longer generation

### Issue: Stop tokens not working in interactive mode

**Check**:
1. Config has `stop_tokens` in `generation` section
2. Console shows: `Stop tokens: [...] -> IDs: [...]`
3. Tokenizer loaded correctly

### Issue: Want to stop on multiple consecutive tokens

**Current limitation**: Only single-token stopping is supported

**Workaround**: Use single-token markers or wait for sequence matching feature

## Best Practices

### 1. Use Special Tokens

Prefer special tokens from tokenizer config:
```json
"stop_tokens": ["</s>"]  // Good - explicit end marker
```

Rather than natural language:
```json
"stop_tokens": ["The End"]  // Less reliable - may appear in text
```

### 2. Combine with max_new_tokens

Always set a reasonable `max_new_tokens` as fallback:
```json
{
  "max_new_tokens": 500,  // Fallback limit
  "stop_tokens": ["</s>"]  // Preferred stopping
}
```

### 3. Test Stop Tokens

Check what token IDs your stop strings map to:
```python
from src.tokenizer import Tokenizer
tok = Tokenizer.load('tok_bpe_8k')
print(tok.get_stop_token_ids(["</s>", "\n\n"]))
```

### 4. Use Multiple Stop Tokens

Provide alternatives for robustness:
```json
"stop_tokens": ["</s>", "\n\n", "###"]
```

### 5. Monitor Generation Length

Check if stop tokens are actually being hit:
- Short generations → stop tokens working
- Always max length → stop tokens not encountered

## Implementation Details

### Model Changes

The `GPT.generate()` method now accepts `stop_token_ids`:

```python
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, stop_token_ids=None):
    if stop_token_ids is None:
        stop_token_ids = []
    
    for _ in range(max_new_tokens):
        # ... generation logic ...
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Check for stop tokens
        if stop_token_ids and idx_next.item() in stop_token_ids:
            break
    
    return idx
```

### Tokenizer Changes

Added methods to convert stop strings to token IDs:

```python
def get_stop_token_ids(self, stop_strings):
    """Convert stop strings to token IDs."""
    stop_ids = []
    for stop_str in stop_strings:
        tokens = self.encode(stop_str)
        if len(tokens) >= 1:
            stop_ids.append(tokens[0])
    return stop_ids
```

### Inference Changes

Stop tokens are loaded from config and passed to generation:

```python
stop_strings = gen.get('stop_tokens', [])
stop_token_ids = tokenizer.get_stop_token_ids(stop_strings)
out = model.generate(idx, max_new, temperature, top_k, stop_token_ids)
```

## Future Enhancements

Potential improvements:

1. **Sequence Matching**: Stop on multi-token sequences
2. **Regex Stops**: Stop on pattern matches
3. **Conditional Stops**: Stop based on context
4. **Stop Statistics**: Track which stop token was hit
5. **Partial Matching**: Stop on token subsequences

## Summary

**Quick reference**:

```json
{
  "generation": {
    "max_new_tokens": 200,
    "stop_tokens": ["</s>", "\n\n"]
  }
}
```

**Behavior**:
- Generates up to `max_new_tokens` tokens
- Stops early if any `stop_tokens` is generated
- Falls back to `max_new_tokens` if no stop token encountered

**Common patterns**:
- `["</s>"]` - Stop at end-of-sequence
- `["\n\n"]` - Stop at paragraph break
- `["</s>", "\n\n"]` - Stop at either
- `[]` or omit - No early stopping (default)
