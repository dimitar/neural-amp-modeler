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
cd neural-amp-modeler && conda activate nam && python -m nam.cli nam-full <data_config> nam_full_configs/models/wavenet.json nam_full_configs/learning/default.json <output_dir>
```

Check `nam_full_configs/` for available model and learning configs. Use the defaults unless the user specifies otherwise.

Use `run_in_background: true` for the Bash tool since training takes a long time (typically 30-60 minutes on GPU).

### Step 4: Check Results

When training completes, check the output directory:

```bash
ls <output_dir>/
```

Expected outputs:
- `model.nam` — the trained model file (this is what the plugin loads)
- `comparison.png` — visual comparison of model output vs target signal
- `config_*.json` — copies of the configs used
- `lightning_logs/` — TensorBoard logs and checkpoints

Read `comparison.png` to visually inspect the model quality and report to the user.

## Notes

- Training uses GPU by default
- Training runs for 500 epochs by default
- The output `model.nam` can be loaded directly in the NAM plugin
