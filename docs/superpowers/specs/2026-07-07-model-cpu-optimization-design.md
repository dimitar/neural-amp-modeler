# Model CPU Optimization — Design

**Date:** 2026-07-07
**Status:** Approved (pending spec review)
**Spans two repos:**
- `neural-amp-modeler/` (this repo, the **trainer**) — the LeakyReLU retrain (Phase 1b).
- `fictional-octo-happiness/` (the **plugin**) — benchmark harness, `fast_tanh`, SIMD flags, Eigen hygiene, re-embedding (Phases 0, 1a, 2–4).

This doc lives in the trainer repo so it can be committed here and pulled on the
training machine, but the plan is cross-repo. Each phase below is tagged with the
repo it executes in.

## Context

The shipping guitar-amp models are five 96 kHz **ParametricWaveNet** models
(`ada_mp1_*_96k.mleng`, the "a2"/channels-8 shape) running on **MLEngineCore**, a
vendored Eigen-based fork of NeuralAmpModelerCore. The whole amp chain runs mono,
oversampled to a fixed 96 kHz internal rate, with inference on the audio thread.
Vectorization comes entirely from Eigen expression templates — no hand SIMD, no
RTNeural, no ONNX. We want to reduce per-instance CPU without an audible tone
change.

Two prior investigations frame this work:

1. **The `feat-a2-aligned-parametric` "parametric_fast" attempt (a negative
   result we already paid for).** It tried compile-time shape-specialized
   WaveNets (constant kernels/dilations/channels) dispatched by a runtime
   sniffer. It measured **0.99×–0.98×** on Apple Silicon (below the 1.3× gate)
   and was reverted (`4285c23`). Root cause: the specialization only reached the
   once-per-knob param encoder; the per-sample Conv1D/1x1/FiLM hot loops still
   used generic dynamic-size Eigen, which Release builds already vectorize well.
   The revert note left one thread untested: *"restart from compile-time channel
   dimensions in the per-layer kernels."*

2. **The per-sample activation audit (this session).** The dominant per-sample
   transcendental cost is **exact libm `std::tanh`**, run **~240× per output
   sample** at 96 kHz (10 layers × 16ch + 10 layers × 8ch), on every shipping
   model. No gating, no sigmoid, no other per-sample transcendental. This is the
   largest single identified lever and the primary driver of this plan.

## Goals

- Reduce per-instance inference CPU on **both** x86_64 and arm64, measured, with
  a repeatable benchmark and an explicit pass gate per change.
- Keep tone change within a **"tiny if inaudible"** budget: pure-efficiency
  changes need no listening; near-transparent changes (approximations, retrained
  activation, float boundary) ship only after an A/B listen passes.

## Non-goals

- No **eco/lite** mode: no reduction of internal oversampling below 96 kHz and no
  channel-count reduction. (Ruled out by the tone budget.)
- No rebuild of the reverted `parametric_fast` sniffer/dispatch machinery as-was.
- No runtime swap (ONNX/RTNeural/XNNPACK) in this plan; noted only as a fallback
  if Phases 0–3 leave an unacceptable gap.

## Constraints & decisions (from brainstorming)

- **Tone budget:** tiny, only if inaudible. Each near-transparent lever is gated
  by an A/B listen.
- **Platforms:** both x86_64 and arm64 matter; prioritize cross-platform levers.
- **Retraining is acceptable** — this unlocks the activation swap as a real lever
  rather than a future-only default.
- **Measure first:** no baseline numbers exist yet, so Phase 0 (benchmark) is a
  hard prerequisite for every later claim of improvement.

## Key findings (evidence base)

Per-sample cost, a2 / channels-8 shipping models, verified against the actual
embedded `.mleng` configs (not the training presets):

| Where | What runs | Frequency | Implementation |
|---|---|---|---|
| Per-layer activation | `tanh` (all 20 layers) | **per sample** (~240×/sample) | exact `std::tanh` (`activations.h` `ActivationTanh`) |
| Param encoder MLP | 2× `tanh` (16-wide) | **per knob change only** | Eigen `Array::tanh` |
| Gating / sigmoid / exp | — | never | disabled (`gating_mode: none`) |
| FiLM, conv, 1x1, head | affine / GEMM only | per sample | linear, no transcendental |

Two facts that reshaped the plan:

- The shipping models use **Tanh on every layer**, *not* the LeakyReLU of the
  `a2_small` training preset — the deployed models diverged from that preset. So
  LeakyReLU is a genuine change to what ships, not a no-op.
- A **`fast_tanh`** (Padé approximation) and a **LUT** path already exist in
  `activations.h` behind `enable_fast_tanh()` / `enable_lut()`, but **no code in
  the product calls them** (`dsp.cpp:15` even has the swap `#define` commented
  out). They are free to switch on.

Activation cost ordering (cheapest last): `std::tanh` → `fast_tanh` → `leaky_relu`
(a select+multiply, no transcendental, vectorizes trivially in Eigen).

## Plan

Measure-first spine, then the highest-value lever, then the safe free wins, then a
gated structural bet.

### Phase 0 — Benchmark harness *(plugin repo)*

**Prerequisite for everything.** Resurrect the standalone `MLEngineCore`
GoogleTest target (`MLENG_BUILD_TESTS=ON`, kept from the a2 branch) and add a perf
bench that:

- Loads one representative shipping model (`ada_mp1_ss_clean_96k`) and runs N
  blocks of a fixed 96 kHz signal through `mleng::DSP::process`.
- Reports **ns/sample** and **×realtime**, warm-cache, median of many runs.
- Runs on both **arm64** and **x86_64** slices.
- Optionally reports a coarse split (activation vs conv) via a build flag, so we
  can attribute cost.

**Gate:** a change is "worth keeping" only if it beats baseline by a margin
larger than run-to-run noise on **both** arches (set the exact threshold from the
first baseline run; the a2 branch used 1.3× for structural work — reuse that for
Phase 3, use "clearly-above-noise" for the cheaper phases).

### Phase 1 — Activation (highest value)

**1a. Enable `fast_tanh` — no retrain *(plugin repo)*.**
Wire `enable_fast_tanh()` (or the LUT) into engine init. This speeds up **all
five currently-shipped models with no re-capture and no re-embed**, and the
baseline→fast_tanh delta *quantifies the activation's share of total cost* — data
that informs whether 1b is worth it.
- **Tone:** approximation → **A/B listen gate** (null-test the two renders; ensure
  no audible difference on clean and driven settings).
- **Deliverable:** one-line init change + benchmark delta + A/B result.

**1b. Retrain with LeakyReLU — the full win *(trainer repo)*.**
Change the per-layer WaveNet activation from `Tanh` to `LeakyReLU` (start
`alpha=0.01`, matching the `a2_small` preset and the `bench_act.py` finding), in
the model config used by `train_parametric.py`. Retrain the same 24 ADA MP-1
captures, export, then re-encrypt + re-embed in the plugin.

- **Trainer changes:** set layer activation to LeakyReLU in the parametric model
  config; keep everything else (channels 16/8, condition_size 16, head_size 8/1,
  MSE+pre-emphasis loss, val_MSE checkpoint, LR schedule) identical so the only
  variable is the activation. Param-encoder Tanh may stay (once-per-knob, no CPU
  benefit to changing, and it is well-behaved).
- **Export:** `train_parametric.py export`. **Verify the exported `activation`
  string key matches the engine's LeakyReLU key** (`ActivationLeakyReLU` in
  `activations.h`) — a mismatch would fail to load or silently fall back. This is
  an explicit acceptance check, not an assumption.
- **Quality gate:** retrained per-capture ESR must stay in the same band as the
  Tanh models (CLAUDE.md baseline: ~0.007 mean, per-capture 0.002–0.006). If
  LeakyReLU degrades ESR materially, investigate before shipping.
- **Tone gate:** A/B listen on the retrained model vs the current Tanh model.
- **CPU gate:** benchmark the LeakyReLU model vs the `std::tanh` baseline on both
  arches; it should beat both baseline and `fast_tanh`.
- **Re-embed:** encrypt the new `.mleng` into the plugin's `BinaryData` sources
  (`GuitarPlugin/Resources/EmbeddedSources/`) once gates pass.

### Phase 2 — Free tone-identical wins *(plugin repo)*

- **Build/SIMD-flag audit.** Confirm the Release build actually enables wide SIMD
  per arch. On x86_64, if it is compiling to the SSE2 baseline, enabling
  **AVX2/FMA** is a large Eigen win. **Caveat:** AVX2 as a hard baseline crashes
  (illegal instruction) on pre-Haswell CPUs — so this requires either dropping
  old-CPU support or a runtime-dispatched fast path. On arm64, NEON is already
  baseline (this lever is small there). Decide the CPU-floor / dispatch policy
  explicitly before flipping the flag.
- **Eigen hot-loop hygiene.** Targeted sweep of `_Layer::Process`, `conv1d`,
  `film` for missing `.noalias()`, avoidable temporaries, and slicing/layout that
  defeats vectorization. (`.noalias()` and `__restrict__` are already present in
  places — this is a focused pass, not a rewrite.) Tone-identical; benchmark each.

### Phase 3 — Gated structural bet *(plugin repo)*

The untested a2 thread: **compile-time channel dimensions *inside* the Conv1D /
1x1 / FiLM hot loops** (e.g. `Matrix<float, 8, N>` fixed-rows so Eigen fully
unrolls; 8 floats maps to one AVX2 register / two NEON regs). Only pursue if
Phases 0–2 leave an unacceptable gap. **Gate: must clear 1.3× on the hot loop, or
it is dropped** — same discipline that correctly killed the first attempt. Do not
resurrect the runtime sniffer/dispatch; specialize the loop directly.

### Phase 4 — Optional, profile-gated *(plugin repo)*

- **`MLENG_SAMPLE=float`:** removes per-block float↔double converts at the
  resampler boundary (internal math is already float). A/B listen gate.
- **Resampler cost** at 44.1/88.2 kHz (r8brain min-phase): only if the profile
  flags it hot; cheaper resampler is a tone change → A/B.

## Risks & open questions

- **Magnitude is unmeasured.** The activation lever is argued from mechanism
  (~240 `std::tanh`/sample); Phase 0 + Phase 1a confirm the real share before we
  commit to retraining all five models. If 1a shows the activation is a small
  fraction, re-weight toward Phase 2/3.
- **Activation key parity** (trainer export string ↔ engine key) — explicit
  acceptance check in Phase 1b.
- **AVX2 distribution policy** — enabling it changes the minimum supported CPU;
  needs a product decision (floor vs runtime dispatch).
- **LeakyReLU ESR regression** — possible but low risk given `bench_act.py`;
  gated on per-capture ESR.
- **Benchmark representativeness** — one model, warm cache, synthetic signal. Good
  for relative deltas (which is all we need); not an absolute host-CPU figure.

## What we are explicitly not doing (a2 lessons)

- Not rebuilding `parametric_fast`'s sniffer/dispatch as-was (0.99×, broke builds,
  no shipping model matched it).
- Not lowering oversampling or channel count (eco mode; outside tone budget).
- Not swapping the inference runtime in this plan.

## Validation summary

| Phase | Repo | Primary gate | Tone gate |
|---|---|---|---|
| 0 Benchmark | plugin | harness runs on both arches, stable median | — |
| 1a fast_tanh | plugin | benchmark delta > noise | A/B listen |
| 1b LeakyReLU retrain | trainer + plugin | ESR band held; CPU beats baseline+fast_tanh both arches | A/B listen |
| 2 SIMD + hygiene | plugin | benchmark delta > noise, both arches | none (tone-identical) |
| 3 fixed-size hot loop | plugin | ≥1.3× on hot loop or dropped | none (tone-identical) |
| 4 float / resampler | plugin | profile-flagged + benchmark delta | A/B listen |
