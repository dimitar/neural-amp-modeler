# Parametric .nam Export Format

## Concept

Standard NAM WaveNet models derive their conditioning signal from the audio itself (the `_WaveNet` class feeds audio through a separate path to produce a conditioning tensor `c`). For parametric modeling, we bypass that entirely. Instead, a small MLP ("param encoder") maps user-facing knob values (OD1, OD2) into the same conditioning space, and that conditioning tensor is fed to the existing `_LayerArray` internals.

The architecture is **`architecture: "ParametricWaveNet"`** in the .nam JSON. The C++ side needs to recognize this as a distinct architecture.

## Signal Flow

```
knobs (OD1, OD2)  →  normalize to [0,1]  →  ParamEncoder MLP  →  cond (16-dim)  →  expand to (16, L)
                                                                                          ↓
audio input  ───────────────────────────────→  _LayerArray[0]  →  _LayerArray[1]  →  head_scale  →  output
                                               (with FiLM)       (with FiLM)
```

- **Normalization**: `(knob_value - minimum) / (maximum - minimum)` — the .nam file provides min/max per param
- **ParamEncoder**: `Linear(2→16) → Tanh → Linear(16→16) → Tanh` — produces a 16-dim vector
- **Conditioning expansion**: The 16-dim vector is broadcast across the time dimension to produce `(batch, 16, L)`, which is the same shape as standard WaveNet conditioning
- **WaveNet layers**: Standard NAM `_LayerArray` with FiLM conditioning (`conv_pre_film` and `activation_pre_film` enabled). The FiLM layers apply `scale * x + shift` where scale/shift are derived from the conditioning via 1x1 convolutions

## .nam JSON Structure

```json
{
  "version": "0.6.0",
  "architecture": "ParametricWaveNet",
  "config": {
    "param_encoder": {
      "num_params": 2,
      "hidden_size": 16,
      "condition_size": 16,
      "activation": "Tanh",
      "params": [
        {"name": "OD1", "minimum": 2.0, "maximum": 10.0},
        {"name": "OD2", "minimum": 2.0, "maximum": 10.0}
      ]
    },
    "layers": [/* standard _LayerArray.export_config() output, includes FiLM config */],
    "head": null,
    "head_scale": 0.02
  },
  "weights": [/* flat array of all weights */],
  "sample_rate": 48000
}
```

## Weight Packing Order

The `weights` array is a single flat list of floats, packed in this order:

### 1. Param encoder weights (320 floats total)

| Segment | Shape | Count | Notes |
|---------|-------|-------|-------|
| `Linear1.weight` | 16×2 | 32 | Row-major |
| `Linear1.bias` | 16 | 16 | |
| `Linear2.weight` | 16×16 | 256 | Row-major |
| `Linear2.bias` | 16 | 16 | |

### 2. WaveNet weights (~33,730 floats)

| Segment | Notes |
|---------|-------|
| `_LayerArray[0].export_weights()` | Standard NAM weight export order, including FiLM weights |
| `_LayerArray[1].export_weights()` | Same |
| `head_scale` | 1 float, appended at the very end |

The WaveNet weight order within each `_LayerArray` follows NAM's existing `export_weights()` method exactly — so the C++ loader can reuse the existing WaveNet weight-loading code for that portion. It just needs to **read the param encoder weights first** before handing off to the standard WaveNet loader.

## What the C++ Plugin Needs to Do

1. **Parse** the `"ParametricWaveNet"` architecture string and `config.param_encoder` section
2. **Build a small MLP** (2 Linear layers with Tanh activations) from the param encoder config
3. **Load param encoder weights** from the front of the weights array (count = `num_params * hidden_size + hidden_size + hidden_size * condition_size + condition_size`)
4. **Load WaveNet weights** from the remainder using the existing WaveNet loader (the layer configs, FiLM configs, etc. are in standard NAM format)
5. **At runtime**: take OD1/OD2 values from UI knobs → normalize using the min/max from config → run through param encoder MLP → expand to `(condition_size, num_samples)` → feed as conditioning to the WaveNet forward pass
6. **UI**: Two knobs labeled "OD1" and "OD2" with ranges from the `params` array in the config

## Key Detail: FiLM

The model uses FiLM (Feature-wise Linear Modulation) at two insertion points per WaveNet layer: `conv_pre_film` and `activation_pre_film`. This is already supported in NAM's C++ WaveNet implementation. The FiLM config and weights are included in the standard `layers` export — no special handling needed beyond what NAM already does for FiLM.

## Test Export Stats

- Param encoder weights: 320
- WaveNet weights (with FiLM): 33,730
- Total weights: 34,050
- File size: ~454 KB
