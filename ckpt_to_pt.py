"""Convert a Lightning .ckpt from parametric_output/checkpoints/ to the .pt
format expected by `train_parametric.py export`. Picks the best checkpoint by
val_loss (lowest first, ties broken by highest epoch). Assumes the "small"
model preset — edit MODEL_SIZE below if you trained with --model-size large."""

import re
import sys
import torch
from pathlib import Path
from train_parametric import _MODEL_PRESETS, _build_layer_configs, HEAD_SCALE

MODEL_SIZE = "small"
SAMPLE_RATE = 96_000  # Training rate. Edit if captures used a different rate.
CKPT_DIR = Path("parametric_output/checkpoints")
OUT_PATH = Path("parametric_output/parametric_wavenet_model.pt")


def parse(p):
    m = re.search(r"epoch=(\d+)-val_loss=(\d+\.\d+)\.ckpt$", p.name)
    if not m:
        sys.exit(f"Cannot parse {p.name}")
    return int(m.group(1)), float(m.group(2))


candidates = list(CKPT_DIR.glob("best-*.ckpt"))
if not candidates:
    sys.exit(f"No checkpoints found in {CKPT_DIR}")

ranked = sorted(candidates, key=lambda p: (parse(p)[1], -parse(p)[0]))
print("Available checkpoints (best first):")
for p in ranked:
    ep, vl = parse(p)
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

if OUT_PATH.exists():
    sys.exit(f"{OUT_PATH} already exists — move or delete it first")

preset = _MODEL_PRESETS[MODEL_SIZE]
torch.save(
    {
        "model_state_dict": state,
        "layer_configs": _build_layer_configs(MODEL_SIZE),
        "head_config": None,
        "head_scale": HEAD_SCALE,
        "condition_size": preset["condition_size"],
        "receptive_field": None,
        "model_size": MODEL_SIZE,
        "sample_rate": SAMPLE_RATE,
    },
    OUT_PATH,
)
print(f"\nWrote {OUT_PATH}")
