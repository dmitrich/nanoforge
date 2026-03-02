# Learning Rate Schedule Explained

## Question
**"Why is the learning rate changing? I set up LR as 5e-4 as default - please explain the intuition and logic"**

## Answer

The learning rate **changes during training** even though you set `learning_rate: 0.0005` (5e-4) in your config. This is intentional and beneficial for training performance.

## The Schedule Implementation

Your training uses a **Warmup + Cosine Decay** schedule implemented in `src/train.py`:

```python
def get_lr(step: int, training: dict) -> float:
    lr       = training['learning_rate']      # Your configured max LR (5e-4)
    warmup   = training.get('warmup_steps', 200)
    max_steps = training['max_steps']
    min_lr   = lr * 0.1                       # Minimum LR (10% of max)

    # Phase 1: Linear Warmup
    if step < warmup:
        return lr * step / max(warmup, 1)
    
    # Phase 3: After training ends
    if step >= max_steps:
        return min_lr
    
    # Phase 2: Cosine Decay
    decay  = (step - warmup) / max(max_steps - warmup, 1)
    coeff  = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (lr - min_lr)
```

## Three Phases of Learning Rate

### Phase 1: Linear Warmup (Steps 0 → 200)

**What happens**: LR starts at 0 and linearly increases to your configured max_lr (5e-4)

**Formula**: `lr = max_lr × (current_step / warmup_steps)`

**Example with your config** (max_lr=5e-4, warmup_steps=200):
- Step 0: lr = 5e-4 × (0/200) = **0.0**
- Step 50: lr = 5e-4 × (50/200) = **1.25e-4**
- Step 100: lr = 5e-4 × (100/200) = **2.5e-4**
- Step 200: lr = 5e-4 × (200/200) = **5e-4** ← Reaches your configured LR

**Why warmup?**
- **Prevents early training instability**: When weights are randomly initialized, large learning rates can cause the model to diverge or get stuck in bad regions
- **Gradients are noisy early on**: The model hasn't learned anything yet, so gradients point in random directions
- **Allows optimizer momentum to build up**: AdamW's momentum terms need time to accumulate useful statistics

**Intuition**: Think of it like warming up a car engine in cold weather - you don't immediately floor the accelerator, you let it warm up first.

### Phase 2: Cosine Decay (Steps 200 → 1000)

**What happens**: LR smoothly decreases from max_lr (5e-4) to min_lr (5e-5) following a cosine curve

**Formula**: 
```python
progress = (step - warmup_steps) / (max_steps - warmup_steps)
coeff = 0.5 × (1.0 + cos(π × progress))
lr = min_lr + coeff × (max_lr - min_lr)
```

**Example with your config** (max_lr=5e-4, min_lr=5e-5, steps 200→1000):
- Step 200: progress=0.0, coeff=1.0, lr = **5e-4** (max)
- Step 400: progress=0.25, coeff=0.85, lr = **4.33e-4**
- Step 600: progress=0.5, coeff=0.5, lr = **2.75e-4** (halfway)
- Step 800: progress=0.75, coeff=0.15, lr = **1.18e-4**
- Step 1000: progress=1.0, coeff=0.0, lr = **5e-5** (min)

**Why cosine decay?**
- **Smooth convergence**: As training progresses, the model needs smaller updates to fine-tune
- **Escape shallow minima**: Early in training, high LR helps escape bad local minima
- **Settle into good minima**: Late in training, low LR allows precise optimization
- **Better than linear decay**: Cosine curve spends more time at higher LR (for exploration) and smoothly transitions to lower LR (for exploitation)

**Intuition**: Like landing an airplane - you start fast to cover distance, then gradually slow down for a smooth landing.

### Phase 3: Post-Training (Steps ≥ 1000)

**What happens**: LR stays at min_lr (5e-5)

This phase typically doesn't occur during normal training, but ensures the LR doesn't go negative if training somehow continues past max_steps.

## Visual Representation

For your config (lr=5e-4, warmup=200, max_steps=1000):

```
Learning Rate Over Time

5.0e-4 |     ╱‾‾‾‾‾╲
       |    ╱        ╲___
       |   ╱             ╲___
       |  ╱                  ╲___
       | ╱                       ╲___
5.0e-5 |╱                            ‾‾‾
       |_________________________________
       0    200   400   600   800   1000
       └warmup┘└──── cosine decay ─────┘

Phase 1: Linear increase (0 → 5e-4)
Phase 2: Cosine decrease (5e-4 → 5e-5)
```

## Why This Works Better Than Constant LR

### Problem with Constant LR

If you used a constant learning rate of 5e-4 throughout training:

1. **Early training**: Risk of divergence or instability
2. **Late training**: Steps are too large, model oscillates around minimum instead of settling in
3. **Final performance**: Often worse validation loss

### Benefits of Warmup + Cosine Decay

1. **Stable start**: Warmup prevents early divergence
2. **Fast learning**: High LR in middle phase learns quickly
3. **Fine-tuning**: Low LR at end refines the solution
4. **Better convergence**: Typically achieves 5-10% lower final loss
5. **Widely used**: Standard practice in modern deep learning (GPT, BERT, etc.)

## Empirical Evidence

Research shows that learning rate schedules consistently outperform constant learning rates:

- **GPT-2/GPT-3**: Use cosine decay with warmup
- **BERT**: Uses linear warmup + linear decay
- **Vision Transformers**: Use cosine decay
- **Your model**: Following best practices from these successful architectures

## How to Observe This in Your Training

### 1. Console Output
```bash
step     0: loss 10.9876  lr 0.00e+00  ← Warmup starts at 0
step    10: loss 8.5432   lr 2.50e-05  ← Increasing
step   100: loss 7.2341   lr 2.50e-04  ← Halfway through warmup
step   200: loss 6.1234   lr 5.00e-04  ← Max LR reached
step   400: loss 5.4321   lr 4.33e-04  ← Cosine decay begins
step   600: loss 4.8765   lr 2.75e-04  ← Halfway through decay
step   800: loss 4.5432   lr 1.18e-04  ← Near end
step  1000: loss 4.3210   lr 5.00e-05  ← Min LR
```

### 2. TensorBoard
```bash
tensorboard --logdir runs/train
```

Navigate to the "Scalars" tab and look at the "LR" plot. You'll see:
- Sharp linear increase (warmup)
- Smooth cosine curve (decay)

### 3. Metrics JSONL
```bash
cat runs/train/run_*/metrics.jsonl | grep '"name":"LR"' | head -20
```

## Configuration Parameters

In your `configs/run_exp001.json`:

```json
{
  "training": {
    "learning_rate": 0.0005,    // Max LR (5e-4) - reached after warmup
    "warmup_steps": 200,        // Steps to ramp up from 0 to max_lr
    "max_steps": 1000,          // Total training steps
    // min_lr is automatically set to 0.1 × learning_rate = 5e-5
  }
}
```

### Tuning These Parameters

**learning_rate** (max_lr):
- Too high: Training diverges or is unstable
- Too low: Training is slow, may not converge
- Typical range: 1e-4 to 1e-3 for small models
- Your value (5e-4): Good middle ground

**warmup_steps**:
- Too short: Doesn't prevent early instability
- Too long: Wastes training time at low LR
- Typical: 5-10% of max_steps
- Your value (200/1000 = 20%): Slightly conservative, but safe

**max_steps**:
- Determines when cosine decay ends
- Should match your total training budget
- Your value (1000): Good for quick experiments

## How to Configure: Fixed vs. Scheduled Learning Rate

The current implementation uses a scheduled learning rate by default. Here's how to configure different behaviors:

### Option 1: Keep Scheduled LR (Recommended - Current Default)

**No changes needed!** Just configure these parameters in your training config:

```json
{
  "training": {
    "learning_rate": 0.0005,    // Maximum LR
    "warmup_steps": 200,        // Warmup duration
    "max_steps": 1000,          // Total steps (determines decay end)
    "scheduler": "cosine"       // Currently not used, but documents intent
  }
}
```

The schedule is automatically applied in `src/train.py`.

### Option 2: Use Fixed/Constant Learning Rate

To use a constant learning rate throughout training, you have two approaches:

#### Approach A: Modify the Training Config (Recommended)

Add a new parameter to control the schedule behavior. First, update your config:

```json
{
  "training": {
    "learning_rate": 0.0005,
    "warmup_steps": 0,          // Set to 0 to disable warmup
    "max_steps": 1000,
    "scheduler": "constant",    // Add this to document intent
    "use_lr_schedule": false    // Add this flag
  }
}
```

Then modify `src/train.py` to check this flag:

```python
def get_lr(step: int, training: dict) -> float:
    # Check if schedule is disabled
    if not training.get('use_lr_schedule', True):
        return training['learning_rate']  # Return constant LR
    
    # Original scheduled LR code
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
```

#### Approach B: Quick Hack (For Testing Only)

Temporarily replace the `get_lr()` function in `src/train.py`:

```python
def get_lr(step: int, training: dict) -> float:
    return training['learning_rate']  # Always return constant LR
```

**Warning**: This approach:
- Ignores warmup_steps and scheduler config
- Requires code modification (not config-based)
- Not recommended for production experiments

### Option 3: Warmup Only (No Decay)

If you want warmup but no decay (constant LR after warmup):

```python
def get_lr(step: int, training: dict) -> float:
    lr     = training['learning_rate']
    warmup = training.get('warmup_steps', 200)
    
    if step < warmup:
        return lr * step / max(warmup, 1)  # Linear warmup
    else:
        return lr  # Constant after warmup
```

Config:
```json
{
  "training": {
    "learning_rate": 0.0005,
    "warmup_steps": 200,
    "scheduler": "warmup_only"
  }
}
```

### Option 4: Different Schedule Types

You can implement other popular schedules:

#### Linear Decay
```python
def get_lr(step: int, training: dict) -> float:
    lr       = training['learning_rate']
    warmup   = training.get('warmup_steps', 200)
    max_steps = training['max_steps']
    min_lr   = lr * 0.1
    
    if step < warmup:
        return lr * step / max(warmup, 1)
    
    # Linear decay from max_lr to min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return max_lr - progress * (max_lr - min_lr)
```

#### Exponential Decay
```python
def get_lr(step: int, training: dict) -> float:
    lr       = training['learning_rate']
    warmup   = training.get('warmup_steps', 200)
    decay_rate = training.get('lr_decay_rate', 0.95)
    
    if step < warmup:
        return lr * step / max(warmup, 1)
    
    # Exponential decay
    return lr * (decay_rate ** (step - warmup))
```

#### Step Decay
```python
def get_lr(step: int, training: dict) -> float:
    lr         = training['learning_rate']
    warmup     = training.get('warmup_steps', 200)
    decay_step = training.get('lr_decay_step', 300)
    decay_rate = training.get('lr_decay_rate', 0.5)
    
    if step < warmup:
        return lr * step / max(warmup, 1)
    
    # Drop LR by decay_rate every decay_step steps
    num_decays = (step - warmup) // decay_step
    return lr * (decay_rate ** num_decays)
```

### Comparison: Fixed vs. Scheduled LR

| Aspect | Fixed LR | Scheduled LR (Current) |
|--------|----------|------------------------|
| **Simplicity** | ✅ Very simple | ⚠️ More complex |
| **Stability** | ⚠️ Can diverge early | ✅ Warmup prevents divergence |
| **Convergence** | ⚠️ May oscillate at end | ✅ Smooth convergence |
| **Final Loss** | ⚠️ Typically 5-10% worse | ✅ Better final performance |
| **Training Speed** | ⚠️ Slower (need lower LR) | ✅ Faster (can use higher max LR) |
| **Hyperparameter Tuning** | ✅ One parameter (LR) | ⚠️ Three parameters (LR, warmup, decay) |
| **Industry Standard** | ❌ Rarely used | ✅ Standard practice |

### Recommended Configurations

#### For Quick Experiments (Fixed LR)
```json
{
  "training": {
    "learning_rate": 0.0003,     // Lower than scheduled max
    "warmup_steps": 0,
    "use_lr_schedule": false,
    "max_steps": 1000
  }
}
```

#### For Best Performance (Scheduled LR - Current Default)
```json
{
  "training": {
    "learning_rate": 0.0005,     // Can be higher with schedule
    "warmup_steps": 200,
    "use_lr_schedule": true,
    "max_steps": 1000
  }
}
```

#### For Stable Training (Warmup Only)
```json
{
  "training": {
    "learning_rate": 0.0004,
    "warmup_steps": 100,
    "scheduler": "warmup_only",
    "max_steps": 1000
  }
}
```

### When to Use Fixed LR

Consider using a fixed learning rate when:
- **Debugging**: Simplifies troubleshooting
- **Very short training**: < 500 steps where schedule overhead isn't worth it
- **Transfer learning**: Fine-tuning pre-trained models (use low fixed LR like 1e-5)
- **Specific research**: Comparing to baselines that use fixed LR

### When to Use Scheduled LR (Recommended)

Use scheduled learning rate (current default) when:
- **Training from scratch**: Always recommended
- **Long training runs**: > 1000 steps
- **Best performance**: When you need the lowest possible loss
- **Following best practices**: Modern deep learning standard
- **Production models**: When final quality matters

### Implementation Example

Here's a complete implementation that supports both modes:

```python
def get_lr(step: int, training: dict) -> float:
    """
    Get learning rate for current step.
    
    Supports multiple modes via training config:
    - use_lr_schedule=False: Constant LR
    - scheduler="constant": Constant LR
    - scheduler="warmup_only": Warmup then constant
    - scheduler="cosine" (default): Warmup + cosine decay
    - scheduler="linear": Warmup + linear decay
    """
    lr = training['learning_rate']
    
    # Check if schedule is disabled
    if not training.get('use_lr_schedule', True):
        return lr
    
    scheduler = training.get('scheduler', 'cosine')
    
    # Constant LR
    if scheduler == 'constant':
        return lr
    
    # Warmup phase (common to all schedules)
    warmup = training.get('warmup_steps', 200)
    if step < warmup:
        return lr * step / max(warmup, 1)
    
    # Warmup only (constant after warmup)
    if scheduler == 'warmup_only':
        return lr
    
    # Decay phase
    max_steps = training['max_steps']
    min_lr = lr * 0.1
    
    if step >= max_steps:
        return min_lr
    
    progress = (step - warmup) / max(max_steps - warmup, 1)
    
    # Cosine decay (default)
    if scheduler == 'cosine':
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coeff * (lr - min_lr)
    
    # Linear decay
    if scheduler == 'linear':
        return lr - progress * (lr - min_lr)
    
    # Default to cosine
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)
```

### Testing Your Configuration

After changing the LR configuration, verify it works:

```bash
# Run a short training test
python src/train.py configs/your_config.json

# Check the console output - look at the 'lr' values:
# step     0: loss 10.9876  lr 5.00e-04  ← Should be constant if fixed
# step    10: loss 8.5432   lr 5.00e-04  ← Should be constant if fixed
# step    20: loss 7.2341   lr 5.00e-04  ← Should be constant if fixed

# Or view in TensorBoard
tensorboard --logdir runs/train
# Look at the LR plot - should be flat line if fixed
```

### Summary

**Current Implementation**: Scheduled LR (warmup + cosine decay) - **Recommended**

**To Use Fixed LR**: 
1. Add `"use_lr_schedule": false` to your training config, OR
2. Modify `get_lr()` function in `src/train.py`

**Best Practice**: Keep the scheduled LR for better performance, only use fixed LR for debugging or specific research needs.

## Summary

**Your configured `learning_rate: 0.0005` is the MAXIMUM learning rate**, not a constant value.

The actual learning rate follows this path:
1. **Warmup** (steps 0-200): 0 → 5e-4
2. **Cosine Decay** (steps 200-1000): 5e-4 → 5e-5
3. **Minimum** (step 1000+): 5e-5

This schedule is a **feature, not a bug** - it's a best practice that improves training stability and final model performance.

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper using warmup
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) - Cosine annealing
- [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187) - LR schedule ablations
