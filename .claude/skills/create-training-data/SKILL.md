---
name: create-training-data
description: Create NAM training data config files. Use when the user wants to set up training data, create a data config, or prepare for training.
---

# Create Training Data Config

Create training data configuration for NAM model training. There are two different data formats depending on the training type.

## Arguments

`/create-training-data` — interactive, will ask the user for details.

## Workflow

### Step 1: Determine Training Type

Ask the user whether this is for:
- **Single-model training** (`nam-full`) — one input/output WAV pair → one model
- **Parametric training** (`train_parametric.py`) — multiple captures at different knob settings → one parametric model

---

## Single-Model Data Format

For standard `nam-full` training. Creates a JSON config file pointing to WAV paths.

### Gather Information

Ask the user for:
1. **Output WAV path** — the reamped/processed signal WAV
2. **Delay compensation** — integer sample delay between input and output (0 if already aligned, typically 0-20)
3. **Output config path** — where to save the JSON file

The **input WAV** is resolved automatically:
1. Look in the same directory as the output WAV for a file with "input" in the name
2. If not found, fall back to the parent `data/input.wav`

Confirm the resolved input path with the user before writing.

### Write Config JSON

```json
{
    "_notes": [
        "Dev note: Ensure that tests/test_bin/test_train/test_main.py's data is ",
        "representative of this!"
    ],
    "train": {
        "start_seconds": null,
        "stop_seconds": -9.0,
        "ny": 8192
    },
    "validation": {
        "start_seconds": -9.0,
        "stop_seconds": null,
        "ny": null
    },
    "common": {
        "x_path": "<absolute_path_to_input_wav>",
        "y_path": "<absolute_path_to_output_wav>",
        "delay": <delay_samples>
    }
}
```

Key details:
- `x_path` and `y_path`: Absolute Windows paths with `\\` separators
- `stop_seconds: -9.0` → training uses everything except the last 9 seconds
- `start_seconds: -9.0` → validation uses the last 9 seconds
- `ny: 8192` is the training sample length (receptive field)

There is also a `two_pairs.json` variant where train and validation use separate WAV files — see `nam_full_configs/data/two_pairs.json` for the template.

### Verify

Read back the written file and confirm paths and delay with the user.

---

## Parametric Data Format

For `train_parametric.py`. Uses a **flat directory of WAV files** — no JSON config files.

### Expected Directory Structure

```
<data_dir>/
├── input.wav                    # Shared dry/DI capture signal
├── 1 <amp_name> NAM.wav         # Capture 1 at knob setting A
├── 2 <amp_name> NAM.wav         # Capture 2 at knob setting B
├── 3 <amp_name> NAM.wav         # Capture 3 at knob setting C
└── ...
```

- `input.wav` must be in the data directory (the script loads `data_dir / "input.wav"`)
- Capture WAVs are numbered sequentially; the filename pattern is set in `load_data()` (currently `f"{cap_num} ADA MP-1 NAM.wav"`)
- All WAVs must be the same sample rate (default: 48kHz, set by `SAMPLE_RATE`)

### Gather Information

Ask the user for:
1. **Data directory** — where the WAVs are (or will be placed)
2. **Capture WAV naming pattern** — how the output WAVs are named (update `load_data()` if different from `{num} ADA MP-1 NAM.wav`)
3. **Parameter mapping** — which capture number corresponds to which knob settings
4. **Delay compensation** — sample delay between input and output

### Update train_parametric.py

The parameter-to-capture mapping is defined in `PARAM_TABLE` at the top of `train_parametric.py`:

```python
PARAM_TABLE = [
    (capture_num, param1_value, param2_value),
    ...
]
```

Also check and update if needed:
- `DELAY_SAMPLES` — delay compensation
- `SAMPLE_RATE` — if not 48kHz
- `TRAIN_SECONDS` — how much audio to use for training (rest is validation)
- WAV filename pattern in `load_data()` line: `wav_path = data_dir / f"{cap_num} ADA MP-1 NAM.wav"`
- `normalize_params()` — if parameter ranges differ from [2, 10]
- `export_nam()` → `param_encoder_config["params"]` — parameter names and min/max for the exported .nam

### Verify

List the data directory and confirm all expected WAVs are present and the naming matches.

---

## Input WAV Convention

The input WAV is resolved by looking in the **same directory as the output WAV(s)** for a file with "input" in the name. For parametric training, it must be named exactly `input.wav` in the data directory.
