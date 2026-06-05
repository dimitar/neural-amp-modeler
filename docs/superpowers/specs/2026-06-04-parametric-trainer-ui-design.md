# Parametric Trainer UI — Design

**Date:** 2026-06-04
**Status:** Approved (pending spec review)

## Context

Training a parametric NAM model today is entirely command-line: prepare a data
folder, run `make_params_json.py` to build `params.json`, run
`train_parametric.py train …`, then `train_parametric.py export …`. This is fine
for a developer but a non-starter for non-technical users (knob ranges, capture
filename patterns, delay compensation, CLI flags).

The goal is a local desktop app that wraps the existing, already-tested CLI so a
non-technical user can go folder → trained `.nam` without touching a terminal.

Two requirements that came up during design and were explicitly scoped **out**:
- **Listening / real-time audio / cab IR / audio-interface monitoring** belongs
  to a separate future project (a barebones "test-bed" build of the GuitarPlugin
  C++ plugin, which already supports parametric `.nam`, IR loading, and live
  audio). The training UI does **not** play audio; "did it train well?" is
  answered by metrics + a loss curve, and listening happens in the test-bed.
- **Single-model training** already has the upstream `nam` tkinter GUI.

This UI is **Project A**. The test-bed plugin is **Project B** (separate spec;
its foundation should be a *shared reusable core module* extracted from
GuitarPlugin). The two connect only via exported `.nam` files.

## Non-negotiable UX constraint

End users launch the app by **double-clicking a desktop shortcut** that opens a
**native app window** — no terminal, no "start a server" step. A true portable
`.exe` is rejected: the app needs the `nam` conda env (PyTorch + CUDA + the `nam`
package, multiple GB, machine-specific), which is impractical to freeze. Instead
the app runs on the *training machine* (which already has that env) and hides all
launching behind the shortcut.

## Architecture

A Gradio Blocks app served on `localhost`, displayed inside a **pywebview**
native window. The browser-side UI talks to a same-process Python backend. Heavy
training runs as a **subprocess** of `train_parametric.py` (process isolation +
a real Stop button + reuse of the tested CLI). `make_params_json.py` functions
are imported **directly** (no subprocess) for config generation.

```
┌─ pywebview window ─────────────┐
│  Gradio UI (localhost)         │
│        │                       │
│   Python backend (same proc)   │
│   ├─ dataset.py                │
│   ├─ params_editor.py ── imports make_params_json
│   ├─ runner.py ──── subprocess ──▶ train_parametric.py (train / export)
│   └─ app.py / launch.py        │
└────────────────────────────────┘
```

### Modules (small, single-responsibility)

| File | Responsibility | Torch-free / testable |
|---|---|---|
| `trainer_app/dataset.py` | Scan a folder, resolve capture pattern (reuses `_resolve_capture_pattern`), validate, place `input.wav` | yes |
| `trainer_app/params_editor.py` | Bridge UI grid ↔ `make_params_json` config (reuses `build_config`, `find_duplicate_value_groups`, `write_params_json`) | yes |
| `trainer_app/runner.py` | Build the training CLI command from UI state; launch/stream/stop the subprocess; parse stdout for progress/metrics | yes (command-builder + parser) |
| `trainer_app/app.py` | Gradio Blocks layout + event wiring only | UI |
| `trainer_app/launch.py` | Start Gradio in a background thread, open the pywebview window | UI |

## UI flow (single page, top to bottom)

1. **Dataset**
   - Pick capture folder → backend auto-detects the capture pattern and lists
     matched WAVs. If detection fails, an editable pattern field exposing the
     `--capture-pattern` template (`…{cap_num}.wav`).
   - Pick a DI/input file → copied into the folder as `input.wav` (overwrite
     guard; never silently clobber an existing one).
   - Status line: ✓ input.wav · ✓ N captures matched · detected pattern.

2. **Parameters**
   - Define knobs: name, min, max (arbitrary — nothing hardcodes OD1/OD2).
   - Editable grid auto-filled with detected captures as rows × knobs as columns.
   - **Load from CSV** populates the grid (the format `make_params_json.py` takes).
   - **Generate params.json** writes the file via `make_params_json`, surfacing
     the duplicate-knob-values warning (`find_duplicate_value_groups`) inline.

3. **Train**
   - **Advanced** expander (collapsed by default): model size (small/large),
     delay samples, epochs, learning rate, train-stop-seconds, val-start-seconds.
     Sensible defaults so non-technical users never open it.
   - **Start** / **Stop** buttons.
   - Live log stream, a progress bar (parsed `epoch`/`max_epochs`), and a live
     loss/ESR curve.

4. **Export**
   - **Export .nam** (output name + folder) via the `export` subcommand.
   - **Open output folder**.

   (Launching the exported model in the Project-B test-bed will be added when
   that project is built — no stub now.)

## Launch / packaging

- `trainer_app/launch.py` — serve Gradio on a localhost port in a background
  thread, then open a pywebview window (app title + `.ico`; Win11 Edge WebView2).
- `launch_trainer.bat` — activate the `nam` conda env and run the launcher with
  **`pythonw.exe`** (no console window); errors logged to
  `trainer_app/logs/launcher.log`.
- `install_shortcut.ps1` — one-time helper that creates a desktop shortcut (with
  the custom icon) pointing at `launch_trainer.bat`.

End-user experience: double-click desktop icon → trainer window opens. The
conda-env + shortcut setup is a one-time technical step done on the training
machine.

## Error handling

- **Train disabled until the dataset is valid**: `input.wav` present, captures
  matched, every grid cell filled, ranges sane (min < max).
- **Graceful Stop**: send the interrupt the way `tests/test_graceful_shutdown.py`
  already does on Windows (`CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT`) so a
  stopped run still saves a model rather than dying.
- Non-zero subprocess exit and stderr surfaced in the log panel.
- Duplicate knob-value combinations warned before training starts.

## Testing

- Unit tests (no torch, no Gradio) for the backend:
  - `dataset.py`: folder scan, pattern resolution, validation states, `input.wav`
    placement with overwrite guard.
  - `runner.py`: the **CLI-command builder** (UI state → exact `train_parametric.py`
    argv) and the **stdout progress parser** (log line → epoch/metric).
  - `params_editor.py`: grid ↔ config round-trip (mostly delegates to the
    already-tested `make_params_json`).
- Manual e2e: launch the app, run a short training on a small sample folder,
  export a `.nam`.

## Out of scope (YAGNI)

- No in-app audio or offline audition (listening → Project B test-bed).
- No single-model training (upstream `nam` GUI).
- No hosted/remote mode; localhost only.
- New dependencies: `gradio`, `pywebview`.

## Verification

1. `conda activate nam && pip install gradio pywebview`.
2. `pytest tests/test_trainer_app_*.py` → backend units pass.
3. Run `python -m trainer_app.launch` → window opens; walk a small dataset
   through dataset → params → a few epochs → export; confirm a `.nam` is written
   and Stop saves a model.
4. Run `install_shortcut.ps1`, double-click the desktop shortcut, confirm the
   window opens with no terminal.
