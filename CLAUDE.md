# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural Amp Modeler (NAM) is a Python library for **training, reamping, and exporting neural network models** that simulate guitar amplifier tone. Trained models are exported as `.nam` files for real-time playback in the companion [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

- **Python 3.10+**, built on **PyTorch** and **PyTorch Lightning**
- Configurations use **Pydantic v2+** BaseModel classes
- Versioning via **setuptools-scm** (git tags drive version)

## Common Commands

```bash
# Install (editable for development)
pip install -e .

# Verify installation
nam-hello-world

# Run all tests
pytest -v

# Run a single test file
pytest tests/test_nam/test_models/test_wavenet.py -v

# Run a specific test
pytest tests/test_nam/test_models/test_wavenet.py::TestWaveNet::test_forward -v

# Lint (CI uses flake8, not black enforcement)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Build docs (from repo root)
cd docs && make html        # Linux/Mac
cd docs && make.bat html    # Windows
```

**CI matrix**: Python 3.10–3.13 on Ubuntu. CI also clones and builds [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) (C++) for integration tests.

## Architecture

### Model Hierarchy

All models inherit from `BaseNet` (`nam/models/base.py`), which combines `nn.Module`, `InitializableFromConfig` (config-based construction), and `Exportable` (`.nam` export). The four model architectures:

| Architecture | File | Notes |
|---|---|---|
| **WaveNet** | `nam/models/wavenet.py` | Most complex; dilated convolutions, FiLM conditioning, grouped convs, bottlenecks |
| **LSTM** | `nam/models/recurrent.py` | Recurrent approach |
| **ConvNet** | `nam/models/conv_net.py` | Standard convolutional |
| **Linear** | `nam/models/linear.py` | Baseline linear model |

`Sequential` (`nam/models/sequential.py`) composes multiple models in series.

### Model Factory & Registry

`nam/models/factory.py` maintains `_model_net_init_registry` mapping architecture names to `init_from_config` constructors. New architectures can be registered via `factory.register()`. The factory also supports import-based initialization for fully-qualified Python class paths.

### Training Stack

Two trainer paths exist:

1. **Simplified trainer** (`nam/train/core.py`) — Used by the GUI (`nam` CLI) and Google Colab. Hardcodes 48kHz sample rate. Contains extensive Pydantic config models.
2. **Full trainer** (`nam/train/full.py`) — Advanced CLI (`nam-full`) accepting three JSON config files: data, model, and learning. Example configs in `nam_full_configs/`.

Both use `LightningModule` (`nam/train/lightning_module.py`) which wraps a `BaseNet` with loss computation, optimizer setup, and training/validation steps.

### Data Pipeline

`nam/data.py` handles audio I/O and dataset construction. Key concepts:
- `AbstractDataset` base class with `Split.TRAIN` / `Split.VALIDATION`
- Dataset-model compatibility verified via `handshake()` pattern (`nam/_handshake.py`)
- Audio loaded as mono WAV; input/output pairs represent DI signal and amped signal

### Loss Functions

`nam/models/losses.py` provides:
- **ESR** (Error Signal Ratio) — primary metric
- **MSE** and **FFT-based MSE** (frequency domain)
- **Multi-Resolution STFT Loss** via embedded `auraloss` dependency (`nam/_dependencies/auraloss/`)
- Pre-emphasis filtering for loss computation

### Export Format

`.nam` files are JSON containing: version, architecture name, config (hyperparameters), flattened weights array, and metadata (date, loudness, gain). Optional test snapshots (input/output numpy arrays) can verify plugin implementations.

### Extension System

Users can place Python modules in `~/.neural-amp-modeler/extensions/` which are auto-loaded at CLI startup (`nam/cli.py`). Extensions can register custom model architectures via the factory.

## Code Conventions

- **Private imports**: All imports are aliased with underscore prefix (`import torch as _torch`, `from typing import Dict as _Dict`). This is a deliberate convention throughout the codebase — follow it when adding new code.
- **`InitializableFromConfig`**: Models and configurable objects implement `init_from_config(cls, config)` and `parse_config(cls, config)` class methods. New models should follow this pattern.
- **Pre-commit**: Configured for `black` formatting.
- **Test structure**: Tests mirror source layout under `tests/test_nam/`. Model tests are in `tests/test_nam/test_models/`.

## CLI Entry Points

| Command | Function | Purpose |
|---|---|---|
| `nam` | `nam.cli:nam_gui` | GUI trainer (tkinter) |
| `nam-full` | `nam.cli:nam_full` | Full trainer: `nam-full <data.json> <model.json> <learning.json> <outdir>` |
| `nam-hello-world` | `nam.cli:nam_hello_world` | Installation check |

## Parametric WaveNet Training (`train_parametric.py`)

Custom training script for multi-capture parametric amp modeling. Trains a single WaveNet conditioned on knob parameters (OD1, OD2) via FiLM layers, learning all capture settings simultaneously.

### Key Lessons

**Data mapping is critical.** The `PARAM_TABLE` maps capture WAV numbers to knob settings. The original table had OD1/OD2 swapped for 19/25 captures (the RTF source doc uses a different column ordering than assumed). This single error caused ESR to plateau at 0.028 with a 101K-param model. Correcting the mapping dropped ESR to 0.007 with a 26K-param model. **Always verify capture-to-parameter mapping against the source document before training.**

**Duplicate/conflicting captures poison training.** Captures 1 and 6 were both mapped to (OD1=2, OD2=2), forcing the model to learn two different outputs for the same input. Capture 1 ESR was 0.86 (complete failure) while the other 23 captures were 0.002–0.006. One bad data point dragged the mean from 0.003 to 0.041. **Always check for duplicate parameter assignments** and investigate per-capture ESR when the aggregate metric seems too high.

**RTF parsing for ADA MP-1 captures:** The RTF table ordering is OD2 varying slowly (groups of 5: 2,4,6,8,10) with OD1 varying fast within each group (2,4,6,8,10). Capture 6 is OD1=2, OD2=4 (RTF erroneously listed it as (4,4), duplicating capture 7).

### Architecture & Training Defaults

- **Small model (26K params):** channels 16/8, condition_size 16, head_size 8/1 — achieves 0.007 mean ESR on 24 captures in 50 epochs
- **Large model (101K params):** channels 32/16, condition_size 32, head_size 16/1 — marginal improvement over small model, not worth the extra inference cost
- **FiLM conditioning:** Identity-initialized (scale=1, shift=0), trained from start at 0.1x base LR. Freezing FiLM layers is unnecessary with identity init + gradient clipping
- **Loss:** MSE + pre-emphasis MSE (coef=0.85). ESR is logged for monitoring only — it's too noisy for training loss or checkpointing
- **Checkpoint metric:** val_MSE (not val_ESR) — smoother and more stable
- **LR:** 4e-3 with ExponentialLR gamma=0.993, 5-epoch linear warmup
- **DataLoader:** `drop_last=True` on train loader (NAM default)
- **Param encoder:** 2-layer MLP (num_params → condition_size → condition_size) with Tanh activations. No output scaling needed when FiLM is identity-initialized

### Commands

```bash
# Train
python train_parametric.py train --data-dir "ADA MP-1 Captures FULL 1-25"

# Train with custom settings
python train_parametric.py train --data-dir "ADA MP-1 Captures FULL 1-25" --epochs 50 --lr 1e-3

# Fine-tune from checkpoint
python train_parametric.py train --data-dir "ADA MP-1 Captures FULL 1-25" --resume path/to/best.ckpt --lr 1e-3 --epochs 100

# Export to .nam
python train_parametric.py export --checkpoint parametric_output/parametric_wavenet_model.pt

# Inference
python train_parametric.py infer --checkpoint parametric_output/parametric_wavenet_model.pt \
    --input-wav input.wav --output-wav out.wav --od1 5 --od2 7
```

### Export Format

The `.nam` export uses architecture name `ParametricWaveNet`. Weight order: param_encoder weights first (linear1.w, linear1.b, linear2.w, linear2.b), then standard WaveNet weights, then head_scale. The config includes a `param_encoder` section with knob names, ranges, and MLP dimensions. **Requires a custom NAM plugin build** — the standard plugin does not support parametric conditioning.

### Monitoring

`monitor_training.py` polls TensorBoard logs and posts Slack updates every 5 epochs via incoming webhook. Features: rolling average trend with log-scale sparkline, ETA, continue/stop recommendation, and automatic collapse detection (kills training if ESR > 1.0, sustained regression, or NaN).
