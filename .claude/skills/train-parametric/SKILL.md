---
name: train-parametric
description: Train a parametric NAM model from multiple captures with different knob settings. Use when the user says "train parametric", "parametric model", or "multi-knob training".
---

# Train Parametric Model

Train a ParametricWaveNet model using `train_parametric.py`. This produces a single model conditioned on knob parameters (e.g., OD1/OD2) via FiLM layers, learning all capture settings simultaneously.

## Arguments

`/train-parametric` — interactive, will ask the user for details.

## Data Format

`train_parametric.py` expects a **flat directory** of WAV files (NOT JSON configs):

```
<data_dir>/
├── input.wav                    # Shared dry/DI capture signal
├── 1 <amp_name> NAM.wav         # Capture at knob setting A
├── 2 <amp_name> NAM.wav         # Capture at knob setting B
├── ...
└── N <amp_name> NAM.wav         # Capture at knob setting N
```

- `input.wav` must be in the directory root (loaded as `data_dir / "input.wav"`)
- Capture WAVs are numbered; the filename pattern is in `load_data()`: `f"{cap_num} ADA MP-1 NAM.wav"`
- All WAVs must be the same sample rate (default: 48kHz)
- The capture-to-knob mapping is defined in `PARAM_TABLE` in `train_parametric.py`

**This is different from `nam-full` training** which uses JSON config files pointing to WAV paths. Use `/create-training-data` if you need to set up data for either format.

## Workflow

### Step 1: Verify Data Directory

List the data directory and confirm:
- `input.wav` exists
- All numbered capture WAVs are present
- Naming matches the pattern in `load_data()`

```bash
ls "<data_dir>/"
```

### Step 2: Configure train_parametric.py

The following constants at the top of `train_parametric.py` must match your data:

**`PARAM_TABLE`** — maps capture numbers to knob values:
```python
PARAM_TABLE = [
    (1, od1_value, od2_value),
    (2, od1_value, od2_value),
    ...
]
```

**Other constants to check:**
| Constant | Default | What to check |
|----------|---------|---------------|
| `DELAY_SAMPLES` | 10 | Sample delay between input and output |
| `SAMPLE_RATE` | 48000 | Must match your WAV files |
| `TRAIN_SECONDS` | 180 | How many seconds for training (rest is validation) |

**Functions to update if naming/ranges differ:**
- `load_data()` line `wav_path = data_dir / f"{cap_num} ADA MP-1 NAM.wav"` — WAV filename pattern
- `normalize_params()` — if parameter ranges aren't [2, 10]
- `export_nam()` → `param_encoder_config["params"]` — parameter names and min/max for the .nam export

### Step 3: Train

```bash
cd neural-amp-modeler && conda activate nam && python train_parametric.py train --data-dir "<data_directory>"
```

Optional flags:
- `--epochs N` — default 250
- `--lr 0.004` — learning rate
- `--output-dir parametric_output` — output directory
- `--resume path/to/best.ckpt` — fine-tune from checkpoint

Use `run_in_background: true` since training takes 30-60+ minutes on GPU.

### Step 4: Check Results

Monitor via TensorBoard logs or `monitor_training.py`. Key metrics:
- `val_MSE` — checkpoint metric (lower is better)
- `val_ESR` — logged for monitoring (target: < 0.01 mean across captures)
- `train_loss` — MSE + pre-emphasis MSE

### Step 5: Export to .nam

```bash
cd neural-amp-modeler && conda activate nam && python train_parametric.py export --checkpoint parametric_output/parametric_wavenet_model.pt --output parametric_wavenet.nam
```

The `.nam` file will contain:
- `"architecture": "ParametricWaveNet"`
- `config.param_encoder` with parameter names, min/max ranges, MLP dimensions
- Weights: param encoder first, then standard WaveNet weights, then head_scale

### Step 6: Test (Optional)

Inference at specific knob settings:

```bash
cd neural-amp-modeler && conda activate nam && python train_parametric.py infer --checkpoint parametric_output/parametric_wavenet_model.pt --input-wav <input.wav> --output-wav test_out.wav --od1 5 --od2 7
```

## Architecture Defaults

| Setting | Value | Notes |
|---------|-------|-------|
| Channels | 32/16 | Layer array 1 / layer array 2 |
| Condition size | 32 | MLP output → FiLM conditioning dimension |
| Head size | 16/1 | Layer array 1 / layer array 2 |
| FiLM | `activation_pre_film` only | Identity-initialized, 0.1x LR |
| Loss | MSE + pre-emphasis MSE (coef=0.85) | ESR logged but not used for training |
| LR | 4e-3, ExponentialLR gamma=0.993 | 5-epoch linear warmup |
| Gradient clip | 1.0 | Prevents FiLM instability |
| Params | ~101K (large) or ~26K (small) | Adjust channels/condition_size/head_size |

## Key Lessons

- **Verify capture-to-parameter mapping** against source documents — a single mapping error can plateau ESR at 4x the achievable value
- **Check for duplicate parameter assignments** — two captures mapped to the same knob values force conflicting outputs and poison training (one bad capture dragged mean ESR from 0.003 to 0.041)
- **Inspect per-capture ESR** when aggregate metric seems high — isolate the bad data point

## Reference: Existing Data

`ADA MP-1 Captures FULL 1-25/` — 25 captures across a 5x5 OD1/OD2 grid (values 2,4,6,8,10 each), delay 10 samples. Capture files named `{N} ADA MP-1 NAM.wav`.
