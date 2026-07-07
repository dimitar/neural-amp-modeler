"""Train + measure: does the k16 head reduce aliasing vs the k1 baseline?

Trains `small` (k1 baseline) and `small_k16head` on the same captures with
identical settings, then drives both with pure tones and compares SNR-A
(aliasing). Higher SNR-A = less aliasing; a positive k16-minus-k1 delta means
the larger head convolution helped.

Runs on Windows (pure Python, pathlib, no shell built-ins). Uses the GPU if
available. Training quality (ESR) is printed by each training run itself.

Examples (from the repo root):
    python tools/head_kernel_ab.py --data-dir "ADA MP-1 Captures FULL 1-25"
    python tools/head_kernel_ab.py --data-dir "D:/captures/ADA MP-1 Captures FULL 1-25" --epochs 60
    # Re-measure already-trained models without retraining:
    python tools/head_kernel_ab.py --skip-train \
        --baseline-ckpt ab_out/small/parametric_wavenet_model.pt \
        --k16-ckpt      ab_out/small_k16head/parametric_wavenet_model.pt
"""
import argparse as _argparse
import subprocess as _subprocess
import sys as _sys
from pathlib import Path as _Path

import numpy as _np

# tools/ is a sibling of the repo root modules; ensure the root is importable
# whether invoked as `python tools/head_kernel_ab.py` or `python -m tools...`.
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from tools.aliasing import DEFAULT_TONES_HZ, snr_a_db, synth_tone  # noqa: E402

_PRESETS = {"baseline": "small", "k16": "small_k16head"}


def _train(model_size, data_dir, out_dir, epochs):
    """Invoke the tested trainer CLI as a subprocess; return the saved .pt path."""
    cmd = [
        _sys.executable, str(_ROOT / "train_parametric.py"), "train",
        "--data-dir", str(data_dir),
        "--model-size", model_size,
        "--output-dir", str(out_dir),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    print(f"\n=== training {model_size} -> {out_dir} ===", flush=True)
    _subprocess.run(cmd, check=True)
    ckpt = _Path(out_dir) / "parametric_wavenet_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"expected trained model at {ckpt}")
    return ckpt


def _load_model(ckpt_path):
    """Reconstruct a trained ParametricWaveNet from its .pt (mirrors do_infer)."""
    import torch as _torch
    from train_parametric import ParametricWaveNet as _PWN

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    ckpt = _torch.load(ckpt_path, weights_only=False, map_location=device)
    model = _PWN(
        layer_configs=ckpt["layer_configs"],
        head_config=ckpt["head_config"],
        head_scale=ckpt["head_scale"],
        condition_size=ckpt["condition_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model, device, int(ckpt.get("sample_rate", 48_000))


def _run_tone(model, device, sr, freq_hz, od1, od2):
    """Drive the model with one tone at (od1, od2); return the output as numpy."""
    import torch as _torch
    from train_parametric import normalize_params as _norm

    od1n, od2n = _norm(od1, od2)
    params = _torch.tensor([[od1n, od2n]], dtype=_torch.float32, device=device)
    x = _torch.tensor(synth_tone(freq_hz, sr), dtype=_torch.float32, device=device)
    with _torch.no_grad():
        y = model(params, x.unsqueeze(0), pad_start=True).squeeze(0)
    return y.detach().cpu().numpy()


def _measure(ckpt_path, od1, od2, tones):
    model, device, sr = _load_model(ckpt_path)
    return sr, {f0: snr_a_db(_run_tone(model, device, sr, f0, od1, od2), f0, sr)
                for f0 in tones}


def main():
    ap = _argparse.ArgumentParser(description=__doc__,
                                  formatter_class=_argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="captures dir (required unless --skip-train)")
    ap.add_argument("--out-root", default="ab_out", help="where trained models land")
    ap.add_argument("--epochs", type=int, default=None, help="override trainer default")
    ap.add_argument("--od1", type=float, default=8.0, help="drive knob 1 for the tone test")
    ap.add_argument("--od2", type=float, default=8.0, help="drive knob 2 for the tone test")
    ap.add_argument("--skip-train", action="store_true",
                    help="measure existing checkpoints instead of training")
    ap.add_argument("--baseline-ckpt", help="k1 baseline .pt (with --skip-train)")
    ap.add_argument("--k16-ckpt", help="k16 .pt (with --skip-train)")
    args = ap.parse_args()

    if args.skip_train:
        if not (args.baseline_ckpt and args.k16_ckpt):
            ap.error("--skip-train requires --baseline-ckpt and --k16-ckpt")
        baseline_ckpt = _Path(args.baseline_ckpt)
        k16_ckpt = _Path(args.k16_ckpt)
    else:
        if not args.data_dir:
            ap.error("--data-dir is required unless --skip-train")
        out_root = _Path(args.out_root)
        baseline_ckpt = _train(_PRESETS["baseline"], args.data_dir,
                               out_root / "small", args.epochs)
        k16_ckpt = _train(_PRESETS["k16"], args.data_dir,
                          out_root / "small_k16head", args.epochs)

    tones = DEFAULT_TONES_HZ
    print(f"\n=== measuring SNR-A at OD1={args.od1}, OD2={args.od2} ===", flush=True)
    sr_b, base = _measure(baseline_ckpt, args.od1, args.od2, tones)
    sr_k, k16 = _measure(k16_ckpt, args.od1, args.od2, tones)
    if sr_b != sr_k:
        print(f"WARNING: sample-rate mismatch baseline={sr_b} k16={sr_k}")

    print(f"\nSNR-A (dB) at {sr_b} Hz — higher is less aliasing\n")
    print(f"{'tone_hz':>9} {'k1_base':>9} {'k16':>9} {'delta':>8}")
    deltas = []
    for f0 in tones:
        d = k16[f0] - base[f0]
        deltas.append(d)
        print(f"{f0:9.1f} {base[f0]:9.2f} {k16[f0]:9.2f} {d:+8.2f}")
    mean_delta = float(_np.mean(deltas))
    print(f"\nmean delta (k16 - k1): {mean_delta:+.2f} dB")
    if mean_delta > 0:
        print("=> k16 head REDUCES aliasing vs k1 (bigger = better).")
    else:
        print("=> k16 head does NOT reduce aliasing here (bounded-benefit hypothesis "
              "not supported at this setting).")
    print("\nNote: model quality (per-capture / val ESR) is reported by each "
          "training run above — check it held vs the ~0.007 baseline before shipping.")


if __name__ == "__main__":
    main()
