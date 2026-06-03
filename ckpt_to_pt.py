"""Convert a Lightning .ckpt from a parametric_output*/checkpoints/ directory
to the .pt format expected by `train_parametric.py export`. Picks the best
checkpoint by val_loss (lowest first, ties broken by highest epoch).

The resulting .pt embeds the dataset's params.json (knob names and ranges),
so it loads cleanly with the data-driven `do_export` and `do_infer` flows.

Examples:
    # Tube-Clean small preset, 96kHz
    python ckpt_to_pt.py \\
        --ckpt-dir parametric_output_tube_clean/checkpoints \\
        --output parametric_output_tube_clean/parametric_wavenet_model.pt \\
        --data-dir "data/ADA MP-1-Tube-Clean-PS2-96khz"

    # SS-Clean small preset, 96kHz
    python ckpt_to_pt.py \\
        --ckpt-dir parametric_output_ss_clean/checkpoints \\
        --output parametric_output_ss_clean/parametric_wavenet_model.pt \\
        --data-dir "data/ADA-MP-1-SS-Clean-PS2-96khz"
"""

import argparse
import re
import sys
from pathlib import Path

import torch

from train_parametric import (
    HEAD_SCALE,
    _MODEL_PRESETS,
    _build_layer_configs,
    _load_params_config,
)


def _parse_ckpt_name(p):
    m = re.search(r"epoch=(\d+)-val_loss=(\d+\.\d+)\.ckpt$", p.name)
    if not m:
        sys.exit(f"Cannot parse {p.name}")
    return int(m.group(1)), float(m.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True,
                    help="Directory containing best-*.ckpt files")
    ap.add_argument("--output", required=True,
                    help="Output .pt path (refuses to overwrite an existing file)")
    ap.add_argument("--data-dir", required=True,
                    help="Dataset directory containing params.json (knob schema)")
    ap.add_argument("--model-size", choices=list(_MODEL_PRESETS.keys()),
                    default="small",
                    help="Model preset used during training (default: small)")
    ap.add_argument("--sample-rate", type=int, default=96_000,
                    help="Sample rate used during training (default: 96000)")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.output)

    candidates = list(ckpt_dir.glob("best-*.ckpt"))
    if not candidates:
        sys.exit(f"No checkpoints found in {ckpt_dir}")

    ranked = sorted(candidates, key=lambda p: (_parse_ckpt_name(p)[1],
                                                -_parse_ckpt_name(p)[0]))
    print("Available checkpoints (best first):")
    for p in ranked:
        ep, vl = _parse_ckpt_name(p)
        print(f"  epoch={ep}  val_loss={vl}  {p.name}")

    best = ranked[0]
    print(f"\nSelected: {best.name}")

    lit = torch.load(best, weights_only=False)
    state = {
        k.removeprefix("_model."): v
        for k, v in lit["state_dict"].items()
        if k.startswith("_model.")
    }
    if not state:
        sys.exit("No _model.* keys in checkpoint — wrong format?")

    if out_path.exists():
        sys.exit(f"{out_path} already exists — move or delete it first")

    param_config = _load_params_config(args.data_dir)
    preset = _MODEL_PRESETS[args.model_size]
    torch.save(
        {
            "model_state_dict": state,
            "layer_configs": _build_layer_configs(args.model_size),
            "head_config": None,
            "head_scale": HEAD_SCALE,
            "condition_size": preset["condition_size"],
            "receptive_field": None,
            "model_size": args.model_size,
            "sample_rate": args.sample_rate,
            "param_config": param_config,
        },
        out_path,
    )
    print(f"\nWrote {out_path}")
    print(
        f"Knobs: "
        + ", ".join(
            f"{p['name']}=[{p['minimum']}, {p['maximum']}]"
            for p in param_config["params"]
        )
    )


if __name__ == "__main__":
    main()
