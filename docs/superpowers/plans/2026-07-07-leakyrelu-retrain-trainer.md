# LeakyReLU Retrain (Trainer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the parametric ADA MP-1 models with LeakyReLU instead of Tanh so per-sample inference drops the expensive `std::tanh`, and prove — with objective metrics, not just ESR — that the retrained models are tonally equivalent.

**Architecture:** Change the single hardcoded layer activation in `train_parametric.py` from `"Tanh"` to `"LeakyReLU"` (default `negative_slope=0.01`). Everything else in the training recipe stays fixed so activation is the only variable. Add unit tests that pin the activation round-trips through config→export→`.nam`, then retrain and validate against an expanded acceptance gate (per-capture ESR, MRSTFT, and a new aliasing/SNR-A metric) plus an A/B listen. The exported `.nam` is handed to the plugin plan for benchmarking and re-embedding.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Lightning, Pydantic v2, pytest, numpy/scipy for the aliasing analysis, embedded `auraloss` (MRSTFT).

## Global Constraints

- Layer activation string is exactly `"LeakyReLU"` (resolves to `torch.nn.LeakyReLU`, case-sensitive) with `negative_slope=0.01`. Exported `.nam` per-layer entry MUST serialize to `{"type": "LeakyReLU", "negative_slope": 0.01}`.
- Training objective is UNCHANGED: MSE + pre-emphasis MSE (coef 0.85), checkpoint on `val_MSE`. Do not retune loss, LR (4e-3, ExponentialLR gamma 0.993, 5-epoch warmup), batch size (16), or FiLM init.
- Preset is `"small"` — channels `(16, 8)`, `condition_size=16`, `head_size=(8, 1)`. Do not change shape.
- Do NOT change the param-encoder MLP activation (`train_parametric.py:715`, Tanh) — it runs once per knob change, not per sample; no CPU benefit and it is well-behaved.
- Exported `.nam`: architecture `"ParametricWaveNet"`, version `"0.6.0"`.
- Training data dir: `"ADA MP-1 Captures FULL 1-25"` (24 usable captures; verify PARAM_TABLE mapping is intact per CLAUDE.md before trusting ESR).
- Code conventions: underscore-aliased imports (`import numpy as _np`), black formatting, tests mirror source under `tests/test_nam/`.
- Acceptance gate is objective-first: per-capture ESR (not just mean), MRSTFT, and SNR-A aliasing vs the Tanh baseline, then an A/B listen. Aliasing is a HARD gate: `SNR-A(LeakyReLU) >= SNR-A(Tanh) - 3 dB` per test tone (margin adjustable after the baseline is measured).

---

## File Structure

- `train_parametric.py:106` — the one-line activation change (Modify).
- `tests/test_nam/test_models/test_activations.py` — assert `LeakyReLU` default `negative_slope=0.01` (Modify).
- `tests/test_nam/test_models/test_wavenet.py` — assert a LeakyReLU layer exports to `{"type": "LeakyReLU", "negative_slope": 0.01}` (Modify).
- `tools/aliasing_eval.py` — new standalone eval: synth tones → run both models via `infer` → compute SNR-A + summary table (Create).
- `tools/compare_models.py` — new: per-capture ESR + MRSTFT for a `.nam` vs the capture set (Create).
- No changes to `nam/models/_activations.py`, `_layer_array.py`, or `export_nam()` — they already emit the correct string; the tests exist to lock that in.

---

## Task 1: Pin LeakyReLU default slope in the activation unit test

**Files:**
- Test: `tests/test_nam/test_models/test_activations.py`

**Interfaces:**
- Consumes: `get_activation(name, **kwargs)` from `nam.models._activations`.
- Produces: nothing downstream; this is a guard test.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nam/test_models/test_activations.py`:

```python
def test_leakyrelu_default_negative_slope_is_0p01():
    # The retrain relies on the bare string "LeakyReLU" yielding slope 0.01,
    # which must match the a2/"small" preset and the exported .nam.
    import torch as _torch
    act = get_activation("LeakyReLU")
    assert isinstance(act, _torch.nn.LeakyReLU)
    assert act.negative_slope == _pytest.approx(0.01)
```

- [ ] **Step 2: Run it and confirm it passes (this is a characterization test)**

Run: `pytest tests/test_nam/test_models/test_activations.py::test_leakyrelu_default_negative_slope_is_0p01 -v`
Expected: PASS. (If it FAILS, torch changed the default — stop and use the explicit dict form `{"name": "LeakyReLU", "negative_slope": 0.01}` at `train_parametric.py:106` instead of the bare string, then revisit this plan's Task 3.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_nam/test_models/test_activations.py
git commit -m "test: pin LeakyReLU default negative_slope=0.01"
```

---

## Task 2: Pin the exported-config activation string (round-trip guard)

**Files:**
- Test: `tests/test_nam/test_models/test_wavenet.py`

**Interfaces:**
- Consumes: the existing WaveNet/`_LayerArray` construction + `export_config()` helpers already used in this test file (e.g. the layer-list activation cases around lines 84–95 and the `export_config` assertions around lines 128/210/303/402).
- Produces: nothing downstream; guards that `activation="LeakyReLU"` serializes to `{"type": "LeakyReLU", "negative_slope": 0.01}`.

- [ ] **Step 1: Write the failing test**

Add a test that builds a minimal layer array with a LeakyReLU layer and asserts the exported activation entry. Mirror the construction already used by the file's existing `export_config` tests (reuse the same helper/fixture those tests use to build a `_LayerArray`; substitute `activation="LeakyReLU"`):

```python
def test_layerarray_exports_leakyrelu_with_slope():
    # Build the smallest valid layer array using this file's existing helper,
    # with LeakyReLU as the layer activation.
    layer_array = _make_minimal_layer_array(activation="LeakyReLU")  # existing test helper
    config = layer_array.export_config()
    # Every layer's activation entry must carry the explicit slope so the
    # C++ engine reads 0.01 rather than falling back to a default.
    assert config["activation"][0] == {"type": "LeakyReLU", "negative_slope": 0.01}
```

If this file has no reusable `_make_minimal_layer_array` helper, construct the `_LayerArray` inline exactly as the nearest existing `export_config` test does, changing only the activation argument to `"LeakyReLU"`.

- [ ] **Step 2: Run it and confirm it passes**

Run: `pytest tests/test_nam/test_models/test_wavenet.py::test_layerarray_exports_leakyrelu_with_slope -v`
Expected: PASS (the exporter already special-cases `torch.nn.LeakyReLU` at `nam/models/_activations.py:246-247`). If it FAILS on the dict shape, read the actual exported value from the failure and align the assertion — do NOT change the exporter.

- [ ] **Step 3: Commit**

```bash
git add tests/test_nam/test_models/test_wavenet.py
git commit -m "test: guard LeakyReLU layer exports {type, negative_slope} to config"
```

---

## Task 3: Flip the layer activation to LeakyReLU

**Files:**
- Modify: `train_parametric.py:106`

**Interfaces:**
- Consumes: `_MODEL_PRESETS` (`train_parametric.py:52`), `_build_layer_configs` (`:97`).
- Produces: checkpoints/`.nam` whose per-layer activation is LeakyReLU. Both `small` and `large` presets inherit this via `**common`.

- [ ] **Step 1: Make the one-line change**

In `train_parametric.py`, the `common` dict inside `_build_layer_configs`:

```python
    common = {
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "activation": "LeakyReLU",
        "film_params": _FILM_PARAMS,
    }
```

(Was `"activation": "Tanh"`. Leave the param-encoder Tanh at `:715` untouched.)

- [ ] **Step 2: Smoke-verify the model builds and a 1-epoch run starts**

Run (tiny sanity run — kill after it logs epoch 0 starting; do NOT do a full 250-epoch run here):

```bash
python train_parametric.py train --data-dir "ADA MP-1 Captures FULL 1-25" --epochs 1
```

Expected: model constructs without an activation-lookup error, training loop begins, a checkpoint dir `parametric_output/checkpoints/` appears. If it errors on `"LeakyReLU"`, the string spelling is wrong — it must match `torch.nn.LeakyReLU` exactly.

- [ ] **Step 3: Export the 1-epoch model and inspect the activation in the .nam**

```bash
python train_parametric.py export --checkpoint parametric_output/parametric_wavenet_model.pt --output /tmp/leaky_smoke.nam
python -c "import json; c=json.load(open('/tmp/leaky_smoke.nam')); print(c['architecture'], c['version']); print(c['config']['layers'][0]['activation'][0])"
```

Expected: prints `ParametricWaveNet 0.6.0` and `{'type': 'LeakyReLU', 'negative_slope': 0.01}`. This is the trainer half of the engine-parity check (the C++ half is a task in the plugin plan).

- [ ] **Step 4: Commit**

```bash
git add train_parametric.py
git commit -m "feat: use LeakyReLU layer activation for parametric WaveNet"
```

---

## Task 4: Aliasing / SNR-A evaluation tool

**Files:**
- Create: `tools/aliasing_eval.py`

**Interfaces:**
- Consumes: the tested `python train_parametric.py infer --checkpoint … --input-wav … --output-wav … --od1 … --od2 …` CLI (reuses the real inference path rather than re-implementing forward).
- Produces: a printed table of SNR-A (dB) per test tone for a baseline `.nam`/checkpoint and a candidate, plus a PASS/FAIL against the margin. This is the objective aliasing validator required by the acceptance gate.

Rationale: a hard-cornered LeakyReLU aliases more than a smooth Tanh. ESR barely moves on aliasing (low total energy, but inharmonic and audible). SNR-A measures exactly this: steady-state output for a pure input tone `f0`, power at harmonic bins `k*f0` (signal) vs all other in-band power (aliasing + noise).

- [ ] **Step 1: Write the SNR-A analysis as a pure, unit-testable function**

Create `tools/aliasing_eval.py`:

```python
"""Aliasing (SNR-A) evaluation for neural amp models.

Feeds pure tones through a model (via `train_parametric.py infer`) and
measures inharmonic energy. Higher SNR-A dB = less aliasing.
"""
import argparse as _argparse
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
from pathlib import Path as _Path

import numpy as _np
import soundfile as _sf

SR = 96000
TONES_HZ = (1046.5, 2093.0, 3135.9, 4186.0)  # C6..C8-ish; high tones stress aliasing
DURATION_S = 2.0
_HARMONIC_HALFWIDTH_BINS = 2  # bins on each side of a harmonic counted as "signal"


def _synth_tone(freq_hz, sr=SR, seconds=DURATION_S, amp=0.5):
    t = _np.arange(int(sr * seconds)) / sr
    return (amp * _np.sin(2.0 * _np.pi * freq_hz * t)).astype(_np.float32)


def snr_a_db(output, f0_hz, sr=SR):
    """SNR-A in dB for a single-tone response. Steady-state second half only."""
    x = _np.asarray(output, dtype=_np.float64)
    x = x[len(x) // 2:]                       # drop attack/settling
    x = x - x.mean()                          # remove DC (LeakyReLU is not zero-symmetric)
    win = _np.hanning(len(x))
    X = _np.abs(_np.fft.rfft(x * win)) ** 2
    freqs = _np.fft.rfftfreq(len(x), 1.0 / sr)
    nyq = sr / 2.0
    df = freqs[1] - freqs[0]
    signal_mask = _np.zeros_like(X, dtype=bool)
    k = 1
    while k * f0_hz < nyq:
        center = int(round(k * f0_hz / df))
        lo = max(0, center - _HARMONIC_HALFWIDTH_BINS)
        hi = min(len(X), center + _HARMONIC_HALFWIDTH_BINS + 1)
        signal_mask[lo:hi] = True
        k += 1
    signal_mask[0] = False                    # never count DC bin as signal
    signal = X[signal_mask].sum()
    alias = X[~signal_mask].sum()
    if alias <= 0.0:
        return float("inf")
    return 10.0 * _np.log10(signal / alias)


def _run_infer(checkpoint, in_wav, out_wav, od1, od2):
    _subprocess.run(
        [_sys.executable, "train_parametric.py", "infer",
         "--checkpoint", str(checkpoint),
         "--input-wav", str(in_wav), "--output-wav", str(out_wav),
         "--od1", str(od1), "--od2", str(od2)],
        check=True,
    )


def evaluate(checkpoint, od1, od2, tones=TONES_HZ):
    results = {}
    with _tempfile.TemporaryDirectory() as d:
        d = _Path(d)
        for f0 in tones:
            in_wav, out_wav = d / f"in_{int(f0)}.wav", d / f"out_{int(f0)}.wav"
            _sf.write(in_wav, _synth_tone(f0), SR)
            _run_infer(checkpoint, in_wav, out_wav, od1, od2)
            y, _ = _sf.read(out_wav)
            results[f0] = snr_a_db(y, f0)
    return results


def main():
    ap = _argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Tanh checkpoint (.pt) or .nam")
    ap.add_argument("--candidate", required=True, help="LeakyReLU checkpoint (.pt) or .nam")
    ap.add_argument("--od1", type=float, default=5.0)
    ap.add_argument("--od2", type=float, default=7.0)
    ap.add_argument("--margin-db", type=float, default=3.0)
    args = ap.parse_args()

    base = evaluate(args.baseline, args.od1, args.od2)
    cand = evaluate(args.candidate, args.od1, args.od2)

    print(f"{'tone_hz':>8} {'baseline_dB':>12} {'candidate_dB':>13} {'delta_dB':>9} {'verdict':>8}")
    ok = True
    for f0 in TONES_HZ:
        delta = cand[f0] - base[f0]
        passed = delta >= -args.margin_db
        ok = ok and passed
        print(f"{f0:8.1f} {base[f0]:12.2f} {cand[f0]:13.2f} {delta:9.2f} "
              f"{'PASS' if passed else 'FAIL':>8}")
    print(f"\nOverall: {'PASS' if ok else 'FAIL'} "
          f"(candidate SNR-A must be >= baseline - {args.margin_db} dB per tone)")
    _sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write a unit test for the pure metric (no model needed)**

Create `tests/test_tools/test_aliasing_eval.py`:

```python
import numpy as _np
from tools.aliasing_eval import snr_a_db, _synth_tone, SR


def test_clean_tone_has_high_snr_a():
    # A pure sine (no nonlinearity, no aliasing) should score very high.
    y = _synth_tone(2093.0)
    assert snr_a_db(y, 2093.0) > 60.0


def test_added_inharmonic_tone_lowers_snr_a():
    # Injecting energy at a deliberately inharmonic frequency must drop SNR-A.
    f0 = 2093.0
    clean = _synth_tone(f0)
    alias = 0.05 * _synth_tone(f0 * 1.4142)  # irrational multiple -> not a harmonic
    assert snr_a_db(clean + alias, f0) < snr_a_db(clean, f0)
```

- [ ] **Step 3: Run the unit tests**

Run: `pytest tests/test_tools/test_aliasing_eval.py -v`
Expected: PASS. (Ensure `tests/test_tools/__init__.py` exists if the suite needs it, and `soundfile`/`numpy` are installed — both are already NAM deps.)

- [ ] **Step 4: Commit**

```bash
git add tools/aliasing_eval.py tests/test_tools/test_aliasing_eval.py
git commit -m "feat: add SNR-A aliasing evaluation tool for model comparison"
```

---

## Task 5: Per-capture ESR + MRSTFT comparison tool

**Files:**
- Create: `tools/compare_models.py`

**Interfaces:**
- Consumes: the capture WAVs in the data dir, `infer` CLI, `nam.models.losses` (ESR) and the embedded `auraloss` MRSTFT (`nam/_dependencies/auraloss/`).
- Produces: a per-capture ESR table (mean + worst) and an MRSTFT distance for a candidate model vs the captured targets. Catches the CLAUDE.md failure mode where one bad capture hides in the mean.

- [ ] **Step 1: Write the comparison as a function returning per-capture numbers**

Create `tools/compare_models.py` that, for each capture in the data dir: runs `infer` at that capture's (OD1, OD2) from the PARAM_TABLE, loads the model output and the captured target, and computes ESR and MRSTFT. Print a table sorted by worst ESR. Reuse `train_parametric`'s `PARAM_TABLE` and capture-file discovery so mapping matches training exactly:

```python
"""Per-capture ESR + MRSTFT comparison for a candidate .nam/checkpoint."""
import argparse as _argparse
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
from pathlib import Path as _Path

import numpy as _np
import soundfile as _sf
import torch as _torch

from nam.models.losses import esr as _esr
# PARAM_TABLE + capture discovery come from the trainer so mapping is identical.
from train_parametric import PARAM_TABLE  # noqa: E402


def _esr_np(pred, target):
    return float(_esr(_torch.tensor(pred), _torch.tensor(target)))


# ... discover capture pairs (DI input, amped target) for each PARAM_TABLE entry,
# run infer at (od1, od2), align lengths, compute ESR (and MRSTFT via auraloss),
# then print a table sorted by descending ESR with mean + worst summarized.
```

Fill in capture discovery by mirroring how `train_parametric.py` enumerates the data dir (same filename pattern), so this tool cannot drift from the training mapping. Keep MRSTFT optional behind a flag if auraloss import is heavy.

- [ ] **Step 2: Smoke-run against the current Tanh baseline model**

Run (produces the baseline table the LeakyReLU model will be judged against):

```bash
python tools/compare_models.py --checkpoint <tanh_baseline.nam> --data-dir "ADA MP-1 Captures FULL 1-25"
```

Expected: a per-capture ESR table with mean ≈ 0.007 and per-capture 0.002–0.006 (CLAUDE.md baseline). If any capture is an outlier, investigate the PARAM_TABLE mapping BEFORE trusting later comparisons.

- [ ] **Step 3: Commit**

```bash
git add tools/compare_models.py
git commit -m "feat: add per-capture ESR + MRSTFT model comparison tool"
```

---

## Task 6: Measure the Tanh baseline (freeze the reference numbers)

**Files:** none (procedure; produces reference numbers used by the gate).

- [ ] **Step 1: Obtain the current Tanh baseline model**

Use the currently-shipping Tanh checkpoint/`.nam` (the source the embedded `ada_mp1_ss_clean_96k` was exported from), or if unavailable, export it from the last Tanh checkpoint. Record its path.

- [ ] **Step 2: Record baseline SNR-A**

```bash
python tools/aliasing_eval.py --baseline <tanh.nam> --candidate <tanh.nam> --od1 5 --od2 7
```

(Comparing Tanh to itself yields delta 0 and prints the absolute SNR-A per tone.) Save the per-tone dB numbers. These are the reference the LeakyReLU model must stay within 3 dB of.

- [ ] **Step 3: Record baseline per-capture ESR + MRSTFT** (from Task 5 Step 2 output). Save the table.

- [ ] **Step 4: Decide the final margin.** Default 3 dB. If baseline SNR-A is already low (aliasing-prone tones), tighten; if very high, the margin has headroom. Note the chosen margin in the retrain commit message. No code change unless you override `--margin-db`.

---

## Task 7: Full retrain with LeakyReLU

**Files:** none (long-running training; produces the candidate model).

> Cost note: this is the multi-hour training run. Run it on the training machine. Do not wrap it in a loop.

- [ ] **Step 1: Verify capture mapping before training** (CLAUDE.md hard lesson)

Confirm `PARAM_TABLE` has no duplicate (OD1, OD2) assignments and matches the RTF source ordering. A single mismap silently wrecks ESR.

- [ ] **Step 2: Run the full training**

```bash
python train_parametric.py train --data-dir "ADA MP-1 Captures FULL 1-25"
```

Expected: 250 epochs (default), checkpoints in `parametric_output/checkpoints/`, `parametric_output/parametric_wavenet_model.pt` written. Monitor via `monitor_training.py` (auto-kills on ESR>1.0/NaN/regression).

- [ ] **Step 3: Export the candidate `.nam`**

```bash
python train_parametric.py export --checkpoint parametric_output/parametric_wavenet_model.pt --output leaky_candidate.nam
```

- [ ] **Step 4: Do NOT commit binaries.** Record the checkpoint + `.nam` paths for the acceptance tasks. (Model binaries are not committed to this repo.)

---

## Task 8: Acceptance gate — objective metrics

**Files:** none (runs the tools from Tasks 4–5 against the retrained model).

- [ ] **Step 1: Per-capture ESR must hold**

```bash
python tools/compare_models.py --checkpoint leaky_candidate.nam --data-dir "ADA MP-1 Captures FULL 1-25"
```

PASS if: mean ESR within the baseline band (≈0.007) AND no per-capture ESR is a new outlier vs the Tanh baseline table from Task 6. FAIL → investigate before shipping (do not loosen the gate).

- [ ] **Step 2: Aliasing gate (hard)**

```bash
python tools/aliasing_eval.py --baseline <tanh.nam> --candidate leaky_candidate.nam --od1 5 --od2 7 --margin-db 3
```

PASS if: exit code 0 (every tone's SNR-A within margin of baseline). Also spot-check a higher-gain setting (e.g. `--od1 8 --od2 8`) where aliasing is worst. FAIL → the LeakyReLU corner is aliasing audibly; options to escalate to the user: raise `negative_slope` slightly, or reconsider whether `fast_tanh` (plugin plan, no retrain) is the better win.

- [ ] **Step 3: MRSTFT sanity** — confirm MRSTFT distance is comparable to baseline (from Task 5). Report the number; not a hard gate on its own.

- [ ] **Step 4: DC-offset check** — confirm the model output mean is near zero on a silence-to-tone input (LeakyReLU is not zero-symmetric). The plugin's output DC blocker should absorb residual DC, but a large offset is a red flag worth reporting.

---

## Task 9: A/B listen + handoff to the plugin plan

**Files:** none.

- [ ] **Step 1: A/B listen** — render the same DI guitar phrase through the Tanh baseline and the LeakyReLU candidate at clean and driven settings. PASS if no audible tone difference. (This is the final tone gate; objective metrics do not replace it.)

- [ ] **Step 2: Hand the exported `.nam` to the plugin plan** — the plugin plan consumes `leaky_candidate.nam` for the CPU benchmark (LeakyReLU vs `std::tanh`) and, once all gates pass, for re-encryption/re-embedding into `GuitarPlugin/Resources/EmbeddedSources/`. Re-embedding is a plugin-repo task, not part of this plan.

- [ ] **Step 3: Record results** — append the final per-capture ESR / SNR-A / MRSTFT numbers and the A/B verdict to the retrain commit or a short note in `docs/superpowers/` so the decision is auditable.

---

## Self-Review

**Spec coverage (Phase 1b of the design spec):**
- "Change per-layer activation Tanh→LeakyReLU, alpha=0.01, keep recipe fixed" → Task 3 + Global Constraints. ✓
- "Verify exported activation string matches engine key" → Task 3 Step 3 (trainer half) + explicit handoff of the C++ half to the plugin plan. ✓
- "Quality gate: per-capture ESR holds" → Task 5 + Task 8 Step 1. ✓
- "Expanded metrics: aliasing (SNR-A), MRSTFT, per-capture ESR, DC check" (from the metrics discussion) → Tasks 4, 5, 8. ✓
- "Tone gate: A/B listen" → Task 9 Step 1. ✓
- "CPU gate + re-embed" → explicitly handed to the plugin plan (Task 9 Step 2), since both are plugin-repo work. ✓

**Placeholder scan:** Task 5 Step 1 leaves capture-discovery to "mirror `train_parametric.py`" rather than inlining code — this is deliberate (the enumeration must not be duplicated/forked from the trainer, or it will drift from the training mapping; DRY). Every other code step is complete.

**Type consistency:** `snr_a_db`, `_synth_tone`, `evaluate`, `SR`, `TONES_HZ` are used consistently between `tools/aliasing_eval.py` and its test. `--margin-db` default (3.0) matches the Global Constraint. Preset name `"small"` and slope `0.01` are consistent throughout.

**Open item flagged to user:** aliasing hard-gate margin default is 3 dB (Global Constraints / Task 6 Step 4); switch to report-only if the user prefers.
