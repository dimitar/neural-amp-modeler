---
name: train-single
description: Train a single NAM WaveNet model from a data config. Use when the user says "train", "train a model", or "run training".
---

# Train Single Model

Train a standard NAM WaveNet model from an input/output WAV pair using `nam-full`.

## Arguments

`/train-single [data_config]` — path to the data JSON config. If omitted, ask the user which data config to use.

## Workflow

### Step 1: Identify Data Config

If not provided as an argument, list available data configs and ask the user to pick one:

```bash
find . -path "./.git" -prune -o -name "*.json" -path "*/data/*" -print
```

### Step 2: Determine Output Directory

Create an output directory based on the data config name. For example:
- `data/MyAmp/clean.json` → `output/MyAmp_clean`

Ask the user to confirm or provide a different output path.

### Step 3: Run Training

```bash
cd neural-amp-modeler && conda activate nam && python -m nam.cli <data_config> nam_full_configs/models/wavenet_preemph.json nam_full_configs/learning/default_200.json <output_dir> --no-show
```

**Default configs:**
- **Model:** `wavenet_preemph.json` — standard WaveNet with pre-emphasis loss (weight=1.0, coef=0.85) for better high-frequency accuracy
- **Learning:** `default_200.json` — 200 epochs, GPU, batch size 16

Other available configs:
- `wavenet.json` — standard WaveNet without pre-emphasis
- `default.json` — 100 epochs (faster, slightly lower quality)

Use the defaults unless the user specifies otherwise.

To launch in a visible window so the user can monitor progress:
```bash
cmd.exe /c "start cmd /k \"title <descriptive_name> && C:\Users\dimit\miniconda3\condabin\conda.bat activate nam && cd <repo_dir> && python -m nam.cli <data_config> nam_full_configs/models/wavenet_preemph.json nam_full_configs/learning/default_200.json <output_dir> --no-show\""
```

### Step 4: Check Results

When training completes, check the output directory (NAM creates a timestamped subdirectory per run):

```bash
ls <output_dir>/
```

Expected outputs inside the timestamped directory:
- `model.nam` — the trained model file (this is what the plugin loads)
- `comparison.png` — visual comparison of model output vs target signal
- `config_*.json` — copies of the configs used
- `lightning_logs/` — TensorBoard logs and checkpoints

Read `comparison.png` to visually inspect the model quality and report to the user.

## Notes

- Training uses GPU by default
- 200 epochs takes ~20-40 minutes depending on audio length and GPU
- The output `model.nam` can be loaded directly in the NAM plugin
- ESR < 0.005 is excellent, 0.005-0.01 is very good, 0.01-0.04 is good
