# Transient Profile ML

Benchmarks three time-series models — AR, MLP-NN, and LSTM — on three synthetic transient signals of increasing complexity, then animates the training process with Manim.

## Signals

| Signal | Description |
|--------|-------------|
| Decay | Exponentially decaying sine wave |
| Chirp | Sine with linearly increasing frequency |
| Multi-Scale | Low-frequency carrier + high-frequency ripple |

## Models

| Model | Approach |
|-------|----------|
| AR (order 50) | Autoregressive linear baseline |
| MLP-NN (128 hidden, ~17k params) | Pointwise `t → y(t)` |
| LSTM (63 hidden, ~17k params) | Sequential; hidden state tracks phase via TBPTT |

## Files

| File | Purpose |
|------|---------|
| `transient_signals.py` | Generate and plot the three signals |
| `benchmark.py` | Train all models, grid-search LR, save predictions and epoch snapshots |
| `plot_results.py` | Produce success/failure 3×3 grid figures |
| `animate_training.py` | Manim animation of model fits evolving across training epochs |

## Success / Failure Matrix

|  | AR | MLP | LSTM |
|--|:--:|:---:|:----:|
| Decay | ✓ | ✓ | ✓ |
| Chirp | ✗ | ✓ | ✓ |
| Multi-Scale | ✗ | ✗ | ✓ |

AR cannot fit non-stationary signals. MLP cannot capture the phase coherence required by multi-scale signals. LSTM succeeds on all three by maintaining hidden state across time.

## Running

```bash
# 1. Generate signals
python transient_signals.py

# 2. Train models and save snapshots (~1 hour on MPS/GPU)
python benchmark.py

# 3. Plot static figures
python plot_results.py

# 4. Render training animation (high quality)
manim -pqh animate_training.py SuccessGrid
manim -pqh animate_training.py FailureGrid
```

## Dependencies

- Python 3.10+, PyTorch, NumPy, Matplotlib
- [Manim Community](https://www.manim.community/) v0.20+
