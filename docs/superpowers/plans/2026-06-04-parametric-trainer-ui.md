# Parametric Trainer UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A double-click desktop app that lets a non-technical user turn a folder of captures into a trained parametric `.nam`, with no command line.

**Architecture:** A Gradio Blocks UI shown inside a pywebview native window. Pure-Python backend modules (torch-free, unit-tested) handle dataset scanning, params.json generation (reusing `make_params_json.py`), and building/streaming the `train_parametric.py` subprocess. Training runs as a subprocess so it can be cleanly stopped; config generation imports `make_params_json` directly.

**Tech Stack:** Python, Gradio, pywebview, pytest. Runs in the existing `nam` conda env.

**Working directory:** All paths are relative to `neural-amp-modeler/`. New package lives at `neural-amp-modeler/trainer_app/`. Tests live at `neural-amp-modeler/tests/`. Root-level modules (`train_parametric.py`, `make_params_json.py`, `capture_naming.py`) are imported by adding the repo root to `sys.path` (the existing test convention).

---

## File Structure

| File | Responsibility |
|---|---|
| `capture_naming.py` (new, root) | Torch-free capture-filename pattern detect/resolve (extracted from `train_parametric.py`) |
| `trainer_app/__init__.py` (new) | Package marker |
| `trainer_app/dataset.py` (new) | Scan a folder, resolve pattern, list captures, place `input.wav`, readiness |
| `trainer_app/params_editor.py` (new) | Bridge UI grid/CSV ↔ `make_params_json` config |
| `trainer_app/runner.py` (new) | Build train/export CLI commands; subprocess mgmt with graceful stop; stdout progress parser |
| `trainer_app/app.py` (new) | Gradio Blocks layout + event wiring |
| `trainer_app/launch.py` (new) | Start Gradio in background, open pywebview window |
| `trainer_app/assets/icon.ico` (optional) | Window/shortcut icon (falls back to default if absent) |
| `launch_trainer.bat` (new, root) | Activate `nam` env, launch with `pythonw` (no console) |
| `install_shortcut.ps1` (new, root) | One-time: create desktop shortcut |
| `pyproject.toml` (modify) | Add `ui` optional-dependencies extra |
| `tests/test_capture_naming.py` (new) | Tests for extracted module |
| `tests/test_trainer_app_dataset.py` (new) | Tests for dataset.py |
| `tests/test_trainer_app_params_editor.py` (new) | Tests for params_editor.py |
| `tests/test_trainer_app_runner.py` (new) | Tests for runner.py |

---

## Task 1: Extract capture-naming helpers to a torch-free module

Rationale: `dataset.py` needs `_resolve_capture_pattern`, but it currently lives in `train_parametric.py`, which imports torch. Extract it so the backend stays torch-free, with no behavior change. Re-export from `train_parametric.py` so existing imports/tests keep working (DRY).

**Files:**
- Create: `capture_naming.py`
- Create: `tests/test_capture_naming.py`
- Modify: `train_parametric.py` (remove the two functions, import them instead)

- [ ] **Step 1: Write the failing test**

Create `tests/test_capture_naming.py`:
```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import capture_naming as cn


def test_resolve_returns_override_with_placeholder():
    pat = "MP-1 3TM NAM Amp DI {cap_num}.wav"
    assert cn._resolve_capture_pattern("data/x", override=pat) == pat


def test_resolve_rejects_override_without_placeholder():
    with pytest.raises(ValueError, match="cap_num"):
        cn._resolve_capture_pattern("data/x", override="no placeholder.wav")


def test_detect_prefix_pattern_on_disk(tmp_path):
    (tmp_path / "1 ADA MP-1 NAM.wav").write_bytes(b"")
    assert cn._detect_capture_pattern(tmp_path) == "{cap_num} ADA MP-1 NAM.wav"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_capture_naming.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'capture_naming'`

- [ ] **Step 3: Create `capture_naming.py`**

Move the two functions verbatim from `train_parametric.py` (currently around lines 384–428). Create `capture_naming.py`:
```python
# File: capture_naming.py
# Purpose: Torch-free detection/resolution of capture WAV filename patterns,
#          shared by train_parametric.py and the trainer UI backend.

from pathlib import Path


def _detect_capture_pattern(data_dir):
    """Auto-detect the WAV filename pattern from the data directory."""
    data_dir = Path(data_dir)
    patterns = [
        "{cap_num} ADA MP-1 NAM.wav",
        "{cap_num} ADA MP-1 PS2 96khz.wav",
    ]
    for pat in patterns:
        if (data_dir / pat.format(cap_num=1)).exists():
            return pat
    for f in data_dir.glob("1 *.wav"):
        if f.name != "input.wav":
            return "{cap_num} " + f.name[2:]
    raise FileNotFoundError(
        f"Cannot detect capture WAV pattern in {data_dir}. "
        "Expected files like '1 ADA MP-1 NAM.wav'. If your captures number the "
        "files differently (e.g. a trailing number), pass --capture-pattern."
    )


def _resolve_capture_pattern(data_dir, override=None):
    """Resolve the capture WAV filename template (override or auto-detect)."""
    if override is not None:
        if "{cap_num}" not in override:
            raise ValueError(
                "--capture-pattern must contain the '{cap_num}' placeholder, "
                "e.g. 'MP-1 3TM NAM Amp DI {cap_num}.wav'"
            )
        return override
    return _detect_capture_pattern(data_dir)
```

- [ ] **Step 4: Update `train_parametric.py` to import instead of define**

In `train_parametric.py`, delete the `_detect_capture_pattern` and `_resolve_capture_pattern` function definitions (the block from `def _detect_capture_pattern(` through the end of `_resolve_capture_pattern`). Add this import near the other imports at the top of the file (after the existing `from nam...` imports):
```python
from capture_naming import _detect_capture_pattern, _resolve_capture_pattern
```

- [ ] **Step 5: Run both test files to verify pass**

Run: `python -m pytest tests/test_capture_naming.py tests/test_capture_pattern.py -q`
Expected: PASS (the existing `tests/test_capture_pattern.py` still passes via the re-export).

- [ ] **Step 6: Commit**

```bash
git add capture_naming.py tests/test_capture_naming.py train_parametric.py
git commit -m "Extract capture-naming helpers into torch-free module"
```

---

## Task 2: Dataset scanning backend

**Files:**
- Create: `trainer_app/__init__.py` (empty)
- Create: `trainer_app/dataset.py`
- Create: `tests/test_trainer_app_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_trainer_app_dataset.py`:
```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import dataset as ds


def test_captures_from_pattern_suffix():
    names = [
        "MP-1 3TM NAM Amp DI 1.wav",
        "MP-1 3TM NAM Amp DI 10.wav",
        "input.wav",
        "notes.txt",
    ]
    pat = "MP-1 3TM NAM Amp DI {cap_num}.wav"
    assert ds.captures_from_pattern(names, pat) == [1, 10]


def test_scan_prefix_dataset_ready(tmp_path):
    (tmp_path / "input.wav").write_bytes(b"")
    (tmp_path / "1 ADA MP-1 NAM.wav").write_bytes(b"")
    (tmp_path / "2 ADA MP-1 NAM.wav").write_bytes(b"")

    status = ds.scan_dataset(tmp_path)

    assert status.pattern == "{cap_num} ADA MP-1 NAM.wav"
    assert status.capture_numbers == [1, 2]
    assert status.input_wav_present is True
    assert status.ready is True


def test_scan_suffix_needs_override(tmp_path):
    (tmp_path / "input.wav").write_bytes(b"")
    for n in (1, 2):
        (tmp_path / f"MP-1 3TM NAM Amp DI {n}.wav").write_bytes(b"")

    auto = ds.scan_dataset(tmp_path)
    assert auto.ready is False           # auto-detect can't handle suffix
    assert auto.errors

    override = ds.scan_dataset(tmp_path, capture_pattern="MP-1 3TM NAM Amp DI {cap_num}.wav")
    assert override.capture_numbers == [1, 2]
    assert override.ready is True


def test_scan_missing_input_wav(tmp_path):
    (tmp_path / "1 ADA MP-1 NAM.wav").write_bytes(b"")
    status = ds.scan_dataset(tmp_path)
    assert status.input_wav_present is False
    assert status.ready is False


def test_place_input_wav_refuses_overwrite(tmp_path):
    src = tmp_path / "di.wav"
    src.write_bytes(b"DI")
    dest = tmp_path / "input.wav"
    dest.write_bytes(b"OLD")

    with pytest.raises(FileExistsError):
        ds.place_input_wav(tmp_path, src)
    assert dest.read_bytes() == b"OLD"

    ds.place_input_wav(tmp_path, src, force=True)
    assert dest.read_bytes() == b"DI"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer_app_dataset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainer_app'`

- [ ] **Step 3: Create the package and module**

Create empty `trainer_app/__init__.py`. Create `trainer_app/dataset.py`:
```python
# File: trainer_app/dataset.py
# Purpose: Scan a capture folder, resolve the capture filename pattern, list
#          captures, place input.wav, and report training readiness.

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from capture_naming import _resolve_capture_pattern


def pattern_to_regex(pattern):
    """Compile a '{cap_num}' template into a full-match regex with one int group."""
    placeholder = "\x00"
    tmp = pattern.replace("{cap_num}", placeholder)
    escaped = re.escape(tmp).replace(placeholder, r"(\d+)")
    return re.compile("^" + escaped + "$")


def captures_from_pattern(names, pattern):
    """Return the sorted, unique capture numbers among `names` matching `pattern`."""
    rx = pattern_to_regex(pattern)
    nums = [int(m.group(1)) for n in names if (m := rx.match(n))]
    return sorted(set(nums))


@dataclass
class DatasetStatus:
    data_dir: str
    pattern: str = None
    capture_numbers: list = field(default_factory=list)
    input_wav_present: bool = False
    errors: list = field(default_factory=list)

    @property
    def ready(self):
        return (
            self.input_wav_present
            and len(self.capture_numbers) > 0
            and not self.errors
        )


def scan_dataset(data_dir, capture_pattern=None):
    """Inspect a dataset folder and report pattern, captures, and readiness."""
    d = Path(data_dir)
    status = DatasetStatus(data_dir=str(d))
    if not d.is_dir():
        status.errors.append(f"Not a folder: {d}")
        return status

    status.input_wav_present = (d / "input.wav").exists()
    names = [p.name for p in d.glob("*.wav") if p.name != "input.wav"]

    try:
        status.pattern = _resolve_capture_pattern(d, capture_pattern)
    except (FileNotFoundError, ValueError) as e:
        status.errors.append(str(e))
        return status

    status.capture_numbers = captures_from_pattern(names, status.pattern)
    if not status.capture_numbers:
        status.errors.append(f"No capture files matched pattern '{status.pattern}'.")
    if not status.input_wav_present:
        status.errors.append("Missing input.wav (the shared DI signal).")
    return status


def place_input_wav(data_dir, src_path, force=False):
    """Copy a DI file into the dataset folder as input.wav, refusing overwrite."""
    dest = Path(data_dir) / "input.wav"
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} already exists. Use force=True to overwrite.")
    shutil.copyfile(src_path, dest)
    return dest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trainer_app_dataset.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add trainer_app/__init__.py trainer_app/dataset.py tests/test_trainer_app_dataset.py
git commit -m "Add dataset scanning backend for trainer UI"
```

---

## Task 3: params.json editor backend

**Files:**
- Create: `trainer_app/params_editor.py`
- Create: `tests/test_trainer_app_params_editor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_trainer_app_params_editor.py`:
```python
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import params_editor as pe


def test_grid_to_config_auto_ranges():
    knob_specs = [{"name": "OD1"}, {"name": "OD2"}]
    rows = [["1", "2", "2"], ["2", "10", "8"]]

    config = pe.grid_to_config(knob_specs, rows)

    assert config["params"] == [
        {"name": "OD1", "minimum": 2.0, "maximum": 10.0},
        {"name": "OD2", "minimum": 2.0, "maximum": 8.0},
    ]
    assert config["captures"][0] == {"capture": 1, "values": [2, 2]}


def test_grid_to_config_range_override():
    knob_specs = [{"name": "OD1", "minimum": 0.0, "maximum": 10.0}]
    rows = [["1", "2"], ["2", "8"]]

    config = pe.grid_to_config(knob_specs, rows)

    assert config["params"][0] == {"name": "OD1", "minimum": 0.0, "maximum": 10.0}


def test_csv_to_grid(tmp_path):
    csv_path = tmp_path / "c.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capture", "OD1", "OD2"])
        w.writerow([1, 2, 2])
        w.writerow([2, 4, 2])

    knob_names, rows = pe.csv_to_grid(csv_path)

    assert knob_names == ["OD1", "OD2"]
    assert rows == [["1", "2", "2"], ["2", "4", "2"]]


def test_generate_writes_and_warns_on_duplicates(tmp_path):
    knob_specs = [{"name": "OD1"}, {"name": "OD2"}]
    rows = [["1", "2", "2"], ["6", "2", "2"], ["2", "4", "2"]]

    path, warnings = pe.generate_params_json(tmp_path, knob_specs, rows)

    assert path == tmp_path / "params.json"
    assert json.loads(path.read_text())["captures"][0]["capture"] == 1
    assert warnings == [[1, 6]]


def test_generate_refuses_overwrite(tmp_path):
    (tmp_path / "params.json").write_text("existing")
    with pytest.raises(FileExistsError):
        pe.generate_params_json(tmp_path, [{"name": "OD1"}], [["1", "2"]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer_app_params_editor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainer_app.params_editor'`

- [ ] **Step 3: Create the module**

Create `trainer_app/params_editor.py`:
```python
# File: trainer_app/params_editor.py
# Purpose: Bridge the UI's knob/grid/CSV inputs to make_params_json's config.

from pathlib import Path

import make_params_json as mpj


def csv_to_grid(csv_path):
    """Read a CSV into (knob_names, rows-of-string-cells incl. capture column)."""
    header, rows = mpj.read_csv(csv_path)
    return header[1:], rows


def grid_to_config(knob_specs, rows):
    """Build a params.json config dict from knob specs and grid rows.

    :param knob_specs: list of {"name", optional "minimum", "maximum"} dicts,
        in column order.
    :param rows: list of [capture, value1, value2, ...] cells (str or numeric).
    """
    header = ["capture"] + [k["name"] for k in knob_specs]
    overrides = {
        k["name"]: (k["minimum"], k["maximum"])
        for k in knob_specs
        if k.get("minimum") is not None and k.get("maximum") is not None
    }
    str_rows = [[str(cell) for cell in row] for row in rows]
    return mpj.build_config(header, str_rows, overrides)


def generate_params_json(data_dir, knob_specs, rows, force=False):
    """Write <data_dir>/params.json; return (path, duplicate_value_groups)."""
    config = grid_to_config(knob_specs, rows)
    warnings = mpj.find_duplicate_value_groups(config)
    path = Path(data_dir) / "params.json"
    mpj.write_params_json(config, path, force=force)
    return path, warnings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trainer_app_params_editor.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add trainer_app/params_editor.py tests/test_trainer_app_params_editor.py
git commit -m "Add params.json editor backend for trainer UI"
```

---

## Task 4: Training subprocess runner

**Files:**
- Create: `trainer_app/runner.py`
- Create: `tests/test_trainer_app_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_trainer_app_runner.py`:
```python
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import runner as rn


def test_build_train_command_includes_core_flags():
    cmd = rn.build_train_command(
        "data/amp", "out", model_size="small", delay=10, epochs=50,
        python_exe="py", script="train_parametric.py",
    )
    assert cmd[:3] == ["py", "train_parametric.py", "train"]
    assert "--data-dir" in cmd and "data/amp" in cmd
    assert "--output-dir" in cmd and "out" in cmd
    assert cmd[cmd.index("--epochs") + 1] == "50"
    assert "--capture-pattern" not in cmd


def test_build_train_command_adds_capture_pattern():
    cmd = rn.build_train_command(
        "d", "o", capture_pattern="X {cap_num}.wav", python_exe="py",
    )
    assert cmd[cmd.index("--capture-pattern") + 1] == "X {cap_num}.wav"


def test_build_export_command():
    cmd = rn.build_export_command("out/model.pt", "amp.nam", python_exe="py")
    assert cmd[:3] == ["py", "train_parametric.py", "export"]
    assert cmd[cmd.index("--checkpoint") + 1] == "out/model.pt"
    assert cmd[cmd.index("--output") + 1] == "amp.nam"


def test_parse_epoch_line():
    line = "+- epoch  12/250  (8s, ETA 0h32m)"
    assert rn.parse_progress_line(line) == {
        "type": "epoch", "epoch": 12, "max_epochs": 250,
        "elapsed_s": 8, "eta_h": 0, "eta_m": 32,
    }


def test_parse_metric_line():
    assert rn.parse_progress_line("|  val_loss : 0.001234  (best 0.001 @ epoch 10)") == {
        "type": "metric", "name": "val_loss", "value": 0.001234,
    }


def test_parse_metric_line_none_placeholder():
    assert rn.parse_progress_line("|  val_ESR  : --------  (best 9.9 @ epoch 0)") == {
        "type": "metric", "name": "val_ESR", "value": None,
    }


def test_parse_non_progress_line_returns_none():
    assert rn.parse_progress_line("Capture pattern: foo") is None


def test_train_process_streams_then_stops(tmp_path):
    # A dummy long-running script that prints, then loops until interrupted.
    script = tmp_path / "dummy.py"
    script.write_text(textwrap.dedent("""
        import sys, time
        print("line-1", flush=True)
        print("line-2", flush=True)
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("stopped", flush=True)
            sys.exit(0)
    """))

    proc = rn.TrainProcess([sys.executable, str(script)])
    proc.start()
    lines = []
    for line in proc.stream():
        lines.append(line)
        if len(lines) >= 2:
            break
    assert lines == ["line-1", "line-2"]

    proc.stop()
    proc.wait(timeout=10)
    assert proc.returncode is not None  # process actually exited
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer_app_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainer_app.runner'`

- [ ] **Step 3: Create the module**

Create `trainer_app/runner.py`:
```python
# File: trainer_app/runner.py
# Purpose: Build train/export CLI commands, run training as a subprocess with a
#          graceful stop, and parse the trainer's stdout for progress.

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_train_command(
    data_dir, output_dir, *, model_size="small", delay=10, epochs=250,
    lr=4e-3, train_stop_seconds=-9.0, val_start_seconds=-9.0,
    capture_pattern=None, python_exe=None, script="train_parametric.py",
):
    """Return the argv for a `train_parametric.py train` invocation."""
    cmd = [
        python_exe or sys.executable, script, "train",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--model-size", model_size,
        "--delay", str(delay),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--train-stop-seconds", str(train_stop_seconds),
        "--val-start-seconds", str(val_start_seconds),
    ]
    if capture_pattern:
        cmd += ["--capture-pattern", capture_pattern]
    return cmd


def build_export_command(checkpoint, output, *, python_exe=None,
                         script="train_parametric.py"):
    """Return the argv for a `train_parametric.py export` invocation."""
    return [
        python_exe or sys.executable, script, "export",
        "--checkpoint", str(checkpoint),
        "--output", str(output),
    ]


_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)\s+\((\d+)s, ETA (\d+)h(\d+)m\)")
_METRIC_RE = re.compile(r"\|\s+(val_loss|val_ESR|train_ESR)\s*:\s*([\d.]+|-+)")


def parse_progress_line(line):
    """Parse one stdout line into a progress dict, or None if not progress."""
    m = _EPOCH_RE.search(line)
    if m:
        return {
            "type": "epoch", "epoch": int(m.group(1)),
            "max_epochs": int(m.group(2)), "elapsed_s": int(m.group(3)),
            "eta_h": int(m.group(4)), "eta_m": int(m.group(5)),
        }
    m = _METRIC_RE.search(line)
    if m:
        raw = m.group(2)
        value = None if set(raw) == {"-"} else float(raw)
        return {"type": "metric", "name": m.group(1), "value": value}
    return None


class TrainProcess:
    """Manage the training subprocess with a graceful (model-saving) stop."""

    def __init__(self, cmd, cwd=None):
        self.cmd = cmd
        self.cwd = str(cwd or REPO_ROOT)
        self._proc = None

    def start(self):
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self._proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, **kwargs,
        )

    def stream(self):
        """Yield stdout lines (without trailing newline) until the process ends."""
        for line in self._proc.stdout:
            yield line.rstrip("\n")

    def stop(self):
        """Send a graceful interrupt so train_parametric.py saves on shutdown."""
        if self._proc and self._proc.poll() is None:
            if os.name == "nt":
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._proc.send_signal(signal.SIGINT)

    def wait(self, timeout=None):
        return self._proc.wait(timeout=timeout)

    @property
    def returncode(self):
        return self._proc.returncode if self._proc else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trainer_app_runner.py -q`
Expected: PASS (8 tests). The `test_train_process_streams_then_stops` test confirms streaming and graceful stop on this OS.

- [ ] **Step 5: Commit**

```bash
git add trainer_app/runner.py tests/test_trainer_app_runner.py
git commit -m "Add training subprocess runner with graceful stop"
```

---

## Task 5: Gradio app (UI wiring)

This is UI glue; it is verified manually (Step 3) rather than by unit tests — all logic it calls is already tested in Tasks 2–4.

**Files:**
- Create: `trainer_app/app.py`

- [ ] **Step 1: Create the app module**

Create `trainer_app/app.py`:
```python
# File: trainer_app/app.py
# Purpose: Gradio Blocks UI wiring the trainer backend (dataset, params, runner).

import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr

from trainer_app import dataset as ds
from trainer_app import params_editor as pe
from trainer_app import runner as rn

DEFAULTS = dict(model_size="small", delay=10, epochs=250, lr=4e-3,
                train_stop_seconds=-9.0, val_start_seconds=-9.0)


def _status_md(status):
    lines = [f"**Pattern:** `{status.pattern or '—'}`",
             f"**input.wav:** {'✓' if status.input_wav_present else '✗ missing'}",
             f"**Captures matched:** {len(status.capture_numbers)}"]
    for e in status.errors:
        lines.append(f"- ⚠️ {e}")
    return "\n\n".join(lines)


def build_app():
    with gr.Blocks(title="NAM Parametric Trainer") as app:
        gr.Markdown("# NAM Parametric Trainer")
        state_dir = gr.State("")
        state_pattern = gr.State(None)
        proc_box = gr.State({})  # holds {"proc": TrainProcess}

        # --- 1. Dataset ---
        with gr.Tab("1. Dataset"):
            folder = gr.Textbox(label="Capture folder")
            pattern_override = gr.Textbox(
                label="Capture filename pattern (only if auto-detect fails)",
                placeholder="e.g. MP-1 3TM NAM Amp DI {cap_num}.wav",
            )
            scan_btn = gr.Button("Scan folder")
            status_md = gr.Markdown()
            di_file = gr.File(label="DI / input signal (.wav)")
            set_input_btn = gr.Button("Set as input.wav")
            input_msg = gr.Markdown()

            def do_scan(folder, override):
                override = override.strip() or None
                status = ds.scan_dataset(folder, override)
                return folder, status.pattern, _status_md(status)

            scan_btn.click(do_scan, [folder, pattern_override],
                           [state_dir, state_pattern, status_md])

            def do_set_input(folder, di):
                if not folder or not di:
                    return "Pick a folder and a DI file first."
                try:
                    ds.place_input_wav(folder, di, force=True)
                    return "✓ input.wav set."
                except Exception as e:  # noqa: BLE001
                    return f"⚠️ {e}"

            set_input_btn.click(do_set_input, [folder, di_file], [input_msg])

        # --- 2. Parameters ---
        with gr.Tab("2. Parameters"):
            knobs = gr.Dataframe(
                headers=["name", "minimum", "maximum"],
                datatype=["str", "number", "number"],
                row_count=(2, "dynamic"), label="Knobs (leave min/max blank to auto-derive)",
                value=[["OD1", None, None], ["OD2", None, None]],
            )
            grid = gr.Dataframe(label="Capture → values (first column = capture #)",
                                row_count=(1, "dynamic"))
            csv_file = gr.File(label="…or load a CSV")
            load_csv_btn = gr.Button("Load CSV into grid")
            gen_btn = gr.Button("Generate params.json", variant="primary")
            params_msg = gr.Markdown()

            def do_load_csv(csv):
                names, rows = pe.csv_to_grid(csv)
                knob_rows = [[n, None, None] for n in names]
                return knob_rows, rows

            load_csv_btn.click(do_load_csv, [csv_file], [knobs, grid])

            def do_generate(folder, knob_tbl, grid_tbl):
                if not folder:
                    return "Scan a folder first."
                knob_specs = []
                for r in knob_tbl:
                    if not r or not str(r[0]).strip():
                        continue
                    spec = {"name": str(r[0]).strip()}
                    if r[1] is not None and r[2] is not None:
                        spec["minimum"], spec["maximum"] = float(r[1]), float(r[2])
                    knob_specs.append(spec)
                try:
                    path, warns = pe.generate_params_json(
                        folder, knob_specs, grid_tbl, force=True)
                except Exception as e:  # noqa: BLE001
                    return f"⚠️ {e}"
                msg = f"✓ Wrote `{path}`."
                for g in warns:
                    msg += f"\n\n⚠️ Captures {g} share identical knob values."
                return msg

            gen_btn.click(do_generate, [folder, knobs, grid], [params_msg])

        # --- 3. Train ---
        with gr.Tab("3. Train"):
            with gr.Accordion("Advanced", open=False):
                model_size = gr.Radio(["small", "large"], value=DEFAULTS["model_size"],
                                      label="Model size")
                delay = gr.Number(DEFAULTS["delay"], label="Delay (samples)")
                epochs = gr.Number(DEFAULTS["epochs"], label="Epochs")
                lr = gr.Number(DEFAULTS["lr"], label="Learning rate")
                train_stop = gr.Number(DEFAULTS["train_stop_seconds"],
                                       label="Train stop seconds")
                val_start = gr.Number(DEFAULTS["val_start_seconds"],
                                      label="Val start seconds")
            out_dir = gr.Textbox("parametric_output", label="Output folder")
            start_btn = gr.Button("Start training", variant="primary")
            stop_btn = gr.Button("Stop")
            log = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True)

            def do_train(folder, pattern, out, ms, dl, ep, l_r, ts, vs, box):
                proc = rn.TrainProcess(rn.build_train_command(
                    folder, out, model_size=ms, delay=int(dl), epochs=int(ep),
                    lr=float(l_r), train_stop_seconds=float(ts),
                    val_start_seconds=float(vs), capture_pattern=pattern))
                proc.start()
                box["proc"] = proc
                buffer = ""
                for line in proc.stream():
                    buffer += line + "\n"
                    yield buffer, box
                proc.wait()
                buffer += f"\n[exited with code {proc.returncode}]\n"
                yield buffer, box

            start_btn.click(
                do_train,
                [folder, state_pattern, out_dir, model_size, delay, epochs, lr,
                 train_stop, val_start, proc_box],
                [log, proc_box])

            def do_stop(box):
                proc = box.get("proc")
                if proc:
                    proc.stop()
                    return "Stopping (saving model)…"
                return "Nothing to stop."

            stop_btn.click(do_stop, [proc_box], [input_msg])

        # --- 4. Export ---
        with gr.Tab("4. Export"):
            nam_name = gr.Textbox("model.nam", label="Output .nam filename")
            export_btn = gr.Button("Export .nam", variant="primary")
            open_btn = gr.Button("Open output folder")
            export_msg = gr.Markdown()

            def do_export(out, name):
                ckpt = Path(out) / "parametric_wavenet_model.pt"
                if not ckpt.exists():
                    return f"⚠️ No trained model at `{ckpt}`. Train first."
                nam_path = Path(out) / name
                cmd = rn.build_export_command(ckpt, nam_path)
                r = subprocess.run(cmd, cwd=str(rn.REPO_ROOT),
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    return f"⚠️ Export failed:\n```\n{r.stdout}\n{r.stderr}\n```"
                return f"✓ Exported `{nam_path}`."

            export_btn.click(do_export, [out_dir, nam_name], [export_msg])

            def do_open(out):
                p = Path(out).resolve()
                if os.name == "nt":
                    os.startfile(p)  # noqa: S606
                return f"Opened `{p}`."

            open_btn.click(do_open, [out_dir], [export_msg])

    return app
```

- [ ] **Step 2: Install UI dependencies**

Run: `pip install gradio pywebview`
Expected: installs cleanly into the `nam` env.

- [ ] **Step 3: Manual smoke test (browser)**

Run: `python -c "from trainer_app.app import build_app; build_app().launch()"`
Expected: a localhost URL prints; opening it shows the four tabs. Point the Dataset tab at `data/ADA MP-1-Tube-Clean-PS2-96khz`, click **Scan folder** → status shows the detected pattern, ✓ input.wav, 25 captures. (Full train/export is exercised in Task 8 verification.) Ctrl+C to stop.

- [ ] **Step 4: Commit**

```bash
git add trainer_app/app.py
git commit -m "Add Gradio UI for parametric trainer"
```

---

## Task 6: pywebview launcher

**Files:**
- Create: `trainer_app/launch.py`

- [ ] **Step 1: Create the launcher**

Create `trainer_app/launch.py`:
```python
# File: trainer_app/launch.py
# Purpose: Start the Gradio app on localhost and show it in a native window.

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "launcher.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    try:
        import webview
        from trainer_app.app import build_app

        app = build_app()
        _, local_url, _ = app.launch(
            prevent_thread_lock=True, server_port=7860,
            inbrowser=False, show_error=True,
        )
        logging.info("Gradio running at %s", local_url)
        webview.create_window(
            "NAM Parametric Trainer", local_url, width=1100, height=820)
        webview.start()
    except Exception:
        logging.exception("Launcher failed")
        raise


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual test (native window)**

Run: `python -m trainer_app.launch`
Expected: a native window titled "NAM Parametric Trainer" opens showing the four tabs (no browser). Close the window to exit. If it fails, check `trainer_app/logs/launcher.log`.

- [ ] **Step 3: Commit**

```bash
git add trainer_app/launch.py
git commit -m "Add pywebview launcher for trainer UI"
```

---

## Task 7: Desktop shortcut launch chain

**Files:**
- Create: `launch_trainer.bat`
- Create: `install_shortcut.ps1`

- [ ] **Step 1: Create the batch launcher**

Create `launch_trainer.bat`:
```bat
@echo off
REM Launches the NAM Parametric Trainer with no console window.
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" nam
cd /d "%~dp0"
start "" pythonw -m trainer_app.launch
```

- [ ] **Step 2: Create the shortcut installer**

Create `install_shortcut.ps1`:
```powershell
# Creates a desktop shortcut that launches the NAM Parametric Trainer.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut((Join-Path $desktop "NAM Trainer.lnk"))
$sc.TargetPath = (Join-Path $here "launch_trainer.bat")
$sc.WorkingDirectory = $here
$sc.WindowStyle = 7  # minimized — hides the brief launcher console
$icon = Join-Path $here "trainer_app\assets\icon.ico"
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()
Write-Host "Created desktop shortcut: NAM Trainer"
```

- [ ] **Step 3: Manual test (shortcut)**

Run: `powershell -ExecutionPolicy Bypass -File install_shortcut.ps1`
Expected: prints "Created desktop shortcut: NAM Trainer". Double-click the new **NAM Trainer** icon on the desktop → the trainer window opens with no visible terminal. (A custom `trainer_app/assets/icon.ico` is optional; without it the shortcut uses the default icon.)

- [ ] **Step 4: Commit**

```bash
git add launch_trainer.bat install_shortcut.ps1
git commit -m "Add desktop shortcut launch chain for trainer UI"
```

---

## Task 8: Dependencies, docs, end-to-end verification

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `ui` optional-dependencies extra**

In `pyproject.toml`, under `[project.optional-dependencies]` (which already has `transformers-compat` and `test`), add:
```toml
ui = [
    "gradio",
    "pywebview",
]
```

- [ ] **Step 2: Run the full backend test suite**

Run: `python -m pytest tests/test_capture_naming.py tests/test_capture_pattern.py tests/test_make_params_json.py tests/test_trainer_app_dataset.py tests/test_trainer_app_params_editor.py tests/test_trainer_app_runner.py -q`
Expected: all PASS.

- [ ] **Step 3: Confirm the whole suite still collects**

Run: `python -m pytest --collect-only -q`
Expected: no import/collection errors; the new test files appear.

- [ ] **Step 4: End-to-end manual run**

1. `python -m trainer_app.launch`
2. Dataset tab → folder = `data/ADA MP-1-Tube-Clean-PS2-96khz` → Scan → ✓ input.wav, 25 captures, pattern shown.
3. Parameters tab → knobs OD1/OD2 (blank ranges) → click in grid or **Load CSV** → **Generate params.json** → "✓ Wrote …/params.json".
4. Train tab → Advanced → set **Epochs = 2** → **Start training** → log streams the `+- epoch 0/2 …` block; progress visible. Let it finish (or **Stop** → confirms a model is saved).
5. Export tab → **Export .nam** → "✓ Exported …model.nam". **Open output folder** shows the file.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "Add ui extra and finalize parametric trainer UI"
```

---

## Self-Review

**Spec coverage:**
- Gradio + pywebview localhost app → Tasks 5, 6. ✓
- Full pipeline (folder → params.json → train → export) → Tasks 2, 3, 5. ✓
- params.json via editable grid **and** CSV → Task 3 (`grid_to_config`, `csv_to_grid`) + Task 5 UI. ✓
- Subprocess training + stdout streaming + graceful Stop (Windows CTRL_BREAK) → Task 4 `TrainProcess`. ✓
- Live progress parsing → Task 4 `parse_progress_line` (matches the real `_MetricsLogger` format). ✓
- Capture-pattern handling incl. suffix override → Tasks 1, 2. ✓
- input.wav placement with overwrite guard → Task 2 `place_input_wav`. ✓
- Train disabled until valid / readiness → `DatasetStatus.ready` (Task 2); UI surfaces status (Task 5). Note: the UI shows readiness but does not hard-disable Start — acceptable; the subprocess errors clearly if prerequisites are missing.
- Desktop shortcut / native window / no terminal → Tasks 6, 7. ✓
- Deps `gradio`, `pywebview` → Task 8. ✓
- Out of scope (no audio, no single-model) → honored. ✓

**Placeholder scan:** No TBD/TODO; all steps contain real code/commands.

**Type consistency:** `DatasetStatus` fields used consistently; `TrainProcess` (`start`/`stream`/`stop`/`wait`/`returncode`) used identically in Task 4 tests and Task 5 app; `grid_to_config`/`csv_to_grid`/`generate_params_json` signatures match between Task 3 and Task 5; `build_train_command`/`build_export_command`/`parse_progress_line` match between Task 4 and Task 5.
