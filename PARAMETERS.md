# Total Parameter Count Formula

## Formula

```
N = 2VC + TC + L(12C² + 10C) + V + 2C
```

| Symbol | Meaning              |
|--------|----------------------|
| V      | Vocab size           |
| C      | Embedding dim        |
| T      | Block size (tokens)  |
| L      | Transformer blocks   |

---

## Derivation

### 1. Token Embedding
`V × C` — no bias

### 2. Position Embedding
`T × C` — no bias

### 3. Per Transformer Block (× L)

#### Multi-Head Attention
| Component               | Parameters         | Notes                     |
|-------------------------|--------------------|---------------------------|
| K, Q, V projections     | 3 × H × C × (C/H) = 3C² | H heads, head_size = C/H, no bias |
| Output projection       | C × C + C = C² + C | bias=True                 |
| **MHA total**           | **4C² + C**        |                           |

#### Feed-Forward Network
| Component | Parameters       | Notes     |
|-----------|------------------|-----------|
| fc1       | C × 4C + 4C = 4C² + 4C | bias=True |
| fc2       | 4C × C + C = 4C² + C   | bias=True |
| **FFN total** | **8C² + 5C** |       |

#### LayerNorm × 2
`2 × (C + C) = 4C` — weight + bias each

#### Per Block Total
```
(4C² + C) + (8C² + 5C) + 4C = 12C² + 10C
```

### 4. Final LayerNorm
`2C` — weight + bias

### 5. LM Head
`C × V + V` — bias=True

---

## Full Expansion

```
N = VC          (token embedding)
  + TC          (position embedding)
  + L(12C² + 10C)  (transformer blocks)
  + 2C          (final LayerNorm)
  + CV + V      (LM head)

  = 2VC + TC + L(12C² + 10C) + V + 2C
```

---

## Validation

For the default config (L=8, C=1024, V=8000, T=128):

| Term                    | Value         |
|-------------------------|---------------|
| 2VC = 2×8000×1024       | 16,384,000    |
| TC  = 128×1024          | 131,072       |
| L×12C² = 8×12×1024²    | 100,663,296   |
| L×10C  = 8×10×1024      | 81,920        |
| V      = 8000           | 8,000         |
| 2C     = 2×1024         | 2,048         |
| **Total**               | **117,270,336** |

Verified against `model.num_parameters()` → **exact match**.
