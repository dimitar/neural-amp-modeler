"""
Parametric WaveNet training for ADA MP-1.

Trains a single WaveNet model conditioned on OD1 and OD2 knob settings,
using NAM's WaveNet layer internals with external parametric conditioning.

Usage:
    python train_parametric.py train --data-dir "ADA MP-1 Captures FULL 1-25"
    python train_parametric.py train --data-dir "ADA MP-1 Captures FULL 1-25" --epochs 1
    python train_parametric.py infer --checkpoint parametric_output/parametric_wavenet_model.pt \
        --input-wav "ADA MP-1 Captures FULL 1-25/input.wav" --output-wav out.wav --od1 5 --od2 7
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nam.data import tensor_to_wav, wav_to_tensor
from nam.models.losses import apply_pre_emphasis_filter, esr
from nam.models.wavenet._layer_array import LayerArray as _LayerArray
from nam.models.wavenet._wavenet import _Head

# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 48_000
DELAY_SAMPLES = 10
TRAIN_SECONDS = 180

CONDITION_SIZE = 32
HEAD_SCALE = 0.02
NY = 8192
BATCH_SIZE = 16
LR = 4e-3
LR_GAMMA = 0.993
MAX_EPOCHS = 250
PRE_EMPH_COEF = 0.85

LR_WARMUP_EPOCHS = 5  # Linear LR ramp-up over first N epochs

# (capture_number, OD1, OD2)
# 5x5 grid: OD2 varies by group (2,4,6,8,10), OD1 varies within group.
# Capture 6: RTF listed (4,4) in error; actual settings are OD1=2, OD2=4.
PARAM_TABLE = [
    (1, 2, 2),
    (2, 4, 2),
    (3, 6, 2),
    (4, 8, 2),
    (5, 10, 2),
    (6, 2, 4),
    (7, 4, 4),
    (8, 6, 4),
    (9, 8, 4),
    (10, 10, 4),
    (11, 2, 6),
    (12, 4, 6),
    (13, 6, 6),
    (14, 8, 6),
    (15, 10, 6),
    (16, 2, 8),
    (17, 4, 8),
    (18, 6, 8),
    (19, 8, 8),
    (20, 10, 8),
    (21, 2, 10),
    (22, 4, 10),
    (23, 6, 10),
    (24, 8, 10),
    (25, 10, 10),
]

_FILM_PARAMS = {
    # Scale+shift combined signal before each activation, based on knob params.
    # Single insertion point (not conv_pre + activation_pre) to reduce
    # multiplicative instability during early training.
    "activation_pre_film": {"active": True, "shift": True},
}

LAYER_CONFIGS = [
    {
        "input_size": 1,
        "condition_size": CONDITION_SIZE,
        "head_size": 16,
        "channels": 32,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "activation": "Tanh",
        "film_params": _FILM_PARAMS,
    },
    {
        "input_size": 32,
        "condition_size": CONDITION_SIZE,
        "head_size": 1,
        "channels": 16,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "activation": "Tanh",
        "film_params": _FILM_PARAMS,
    },
]


def normalize_params(od1, od2):
    """Normalize knob values from [2, 10] to [0, 1]."""
    return (od1 - 2) / 8, (od2 - 2) / 8


# ─── Dataset ──────────────────────────────────────────────────────────────────


class ParametricDataset(Dataset):
    """Dataset returning (params, x_segment, y_segment) tuples across all captures."""

    def __init__(self, x, ys, params_list, nx, ny, start=None, stop=None):
        """
        :param x: Shared input signal (N,)
        :param ys: List of output signals [(N,), ...], one per capture
        :param params_list: List of (od1_norm, od2_norm) tuples
        :param nx: Receptive field (input context size)
        :param ny: Output chunk size
        :param start: Start sample index (inclusive)
        :param stop: Stop sample index (exclusive)
        """
        self._params_list = [
            torch.tensor(p, dtype=torch.float32) for p in params_list
        ]
        self._x = x[start:stop]
        self._ys = [y[start:stop] for y in ys]
        self._nx = nx
        self._ny = ny
        n = len(self._x)
        self._chunks_per_capture = max(1, (n - nx + 1) // ny)
        self._num_captures = len(ys)

    def __len__(self):
        return self._chunks_per_capture * self._num_captures

    def __getitem__(self, idx):
        capture_idx = idx // self._chunks_per_capture
        chunk_idx = idx % self._chunks_per_capture
        i = chunk_idx * self._ny
        x_seg = self._x[i : i + self._nx + self._ny - 1]
        y_seg = self._ys[capture_idx][i + self._nx - 1 : i + self._nx - 1 + self._ny]
        return self._params_list[capture_idx], x_seg, y_seg


# ─── Model ────────────────────────────────────────────────────────────────────


class ParametricWaveNet(nn.Module):
    """WaveNet conditioned on external parameters via a learned MLP encoder.

    Bypasses _WaveNet.forward() (which derives conditioning from audio) and
    instead calls _LayerArray.forward(y, c, head_input) directly with
    parameter-derived conditioning.
    """

    def __init__(
        self,
        layer_configs,
        head_config=None,
        head_scale=0.02,
        num_params=2,
        condition_size=16,
    ):
        super().__init__()
        self._param_encoder = nn.Sequential(
            nn.Linear(num_params, condition_size),
            nn.Tanh(),
            nn.Linear(condition_size, condition_size),
            nn.Tanh(),
        )
        configs_with_position = []
        for i, lc in enumerate(layer_configs):
            c = {**lc, "is_first": i == 0, "is_last": i == len(layer_configs) - 1}
            configs_with_position.append(c)
        self._layer_arrays = nn.ModuleList(
            [_LayerArray.init_from_config(lc) for lc in configs_with_position]
        )
        # Initialize FiLM layers to identity: scale=1, shift=0.
        # Without this, random FiLM scales compound across 20 layers and
        # crush the signal to zero.
        self._init_film_identity()

        self._head = None if head_config is None else _Head(**head_config)
        self._head_scale = head_scale

    def _init_film_identity(self):
        """Set all FiLM modules to pass-through (scale=1, shift=0)."""
        with torch.no_grad():
            for la in self._layer_arrays:
                for layer in la._layers:
                    for film in (
                        layer._conv_pre_film,
                        layer._conv_post_film,
                        layer._input_mixin_pre_film,
                        layer._input_mixin_post_film,
                        layer._activation_pre_film,
                        layer._activation_post_film,
                        layer._layer1x1_post_film,
                        layer._head1x1_post_film,
                    ):
                        if film is None:
                            continue
                        nn.init.zeros_(film._film.weight)
                        bias = film._film.bias
                        if film._shift:
                            half = bias.shape[0] // 2
                            bias[:half] = 1.0   # scale
                            bias[half:] = 0.0   # shift
                        else:
                            bias.fill_(1.0)     # scale only

    @property
    def receptive_field(self):
        return 1 + sum(la.receptive_field - 1 for la in self._layer_arrays)

    def forward(self, params, x, pad_start=True):
        """
        :param params: (B, 2) normalized knob parameters
        :param x: (B, L) mono audio input
        :param pad_start: If True, zero-pad input so output length == input length
        :return: (B, L') audio output
        """
        if pad_start:
            x = F.pad(x, (self.receptive_field - 1, 0))

        # Encode parameters to conditioning vector, expand across time
        cond = self._param_encoder(params)  # (B, condition_size)
        cond = cond.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (B, C, L)

        y = x.unsqueeze(1)  # (B, 1, L)

        head_input = None
        for layer_array in self._layer_arrays:
            head_input, y = layer_array(y, cond, head_input=head_input)

        head_input = self._head_scale * head_input
        if self._head is not None:
            head_input = self._head(head_input)
        return head_input.squeeze(1)  # (B, L')


# ─── Lightning Module ─────────────────────────────────────────────────────────


class ParametricLightningModule(pl.LightningModule):
    def __init__(self, model, lr=LR, lr_gamma=LR_GAMMA,
                 warmup_epochs=LR_WARMUP_EPOCHS):
        super().__init__()
        self._model = model
        self._lr = lr
        self._lr_gamma = lr_gamma
        self._warmup_epochs = warmup_epochs

    def forward(self, params, x, **kwargs):
        return self._model(params, x, **kwargs)

    def training_step(self, batch, batch_idx):
        params, x_seg, y_seg = batch
        preds = self._model(params, x_seg, pad_start=False)
        loss_mse = F.mse_loss(preds, y_seg)
        # Pre-emphasis MSE: amplifies high-frequency detail
        preds_pe = apply_pre_emphasis_filter(preds, PRE_EMPH_COEF)
        targets_pe = apply_pre_emphasis_filter(y_seg, PRE_EMPH_COEF)
        loss_pe = F.mse_loss(preds_pe, targets_pe)
        loss = loss_mse + loss_pe
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_MSE", loss_mse)
        self.log("train_ESR", esr(preds, y_seg))
        return loss

    def validation_step(self, batch, batch_idx):
        params, x_seg, y_seg = batch
        preds = self._model(params, x_seg, pad_start=False)
        loss_mse = F.mse_loss(preds, y_seg)
        self.log("val_loss", loss_mse, prog_bar=True)
        self.log("val_ESR", esr(preds, y_seg))
        self.log("val_MSE", loss_mse)
        return loss_mse

    def configure_optimizers(self):
        # Separate param groups: FiLM params get lower LR
        film_param_ids = set()
        film_params = []
        for la in self._model._layer_arrays:
            for layer in la._layers:
                for film in (
                    layer._conv_pre_film,
                    layer._conv_post_film,
                    layer._input_mixin_pre_film,
                    layer._input_mixin_post_film,
                    layer._activation_pre_film,
                    layer._activation_post_film,
                    layer._layer1x1_post_film,
                    layer._head1x1_post_film,
                ):
                    if film is not None:
                        for p in film.parameters():
                            film_param_ids.add(id(p))
                            film_params.append(p)

        base_params = [p for p in self._model.parameters()
                       if id(p) not in film_param_ids]

        optimizer = torch.optim.Adam([
            {"params": base_params, "lr": self._lr},
            {"params": film_params, "lr": self._lr * 0.1},
        ])

        # Exponential decay + linear warmup
        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self._lr_gamma
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=self._warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, exp_scheduler],
            milestones=[self._warmup_epochs],
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ─── Data Loading ─────────────────────────────────────────────────────────────


def load_data(data_dir, nx, ny_train=NY):
    """Load all WAV files, apply delay correction, and create train/val datasets."""
    data_dir = Path(data_dir)

    x = wav_to_tensor(data_dir / "input.wav")
    print(f"Loaded input: {len(x)} samples ({len(x) / SAMPLE_RATE:.1f}s)")

    ys = []
    params_list = []
    for cap_num, od1, od2 in PARAM_TABLE:
        wav_path = data_dir / f"{cap_num} ADA MP-1 NAM.wav"
        y = wav_to_tensor(wav_path)
        # Delay correction: output is 10 samples behind input
        ys.append(y[DELAY_SAMPLES:])
        params_list.append(normalize_params(od1, od2))

    x = x[:-DELAY_SAMPLES]

    for i, y in enumerate(ys):
        assert len(y) == len(x), (
            f"Capture {PARAM_TABLE[i][0]}: length {len(y)} != input length {len(x)}"
        )

    train_stop = TRAIN_SECONDS * SAMPLE_RATE
    val_len = len(x) - train_stop
    print(f"Train: {TRAIN_SECONDS}s ({train_stop} samples)")
    print(f"Val: {val_len / SAMPLE_RATE:.1f}s ({val_len} samples)")

    train_ds = ParametricDataset(x, ys, params_list, nx, ny_train, 0, train_stop)

    # Validation: one chunk per capture covering the full validation segment
    val_ny = val_len - nx + 1
    val_ds = ParametricDataset(x, ys, params_list, nx, val_ny, train_stop, None)

    print(
        f"Train: {len(train_ds)} items "
        f"({train_ds._chunks_per_capture} chunks x {train_ds._num_captures} captures)"
    )
    print(f"Val: {len(val_ds)} items (ny={val_ny})")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


# ─── Training ─────────────────────────────────────────────────────────────────


def do_train(args):
    """Train the parametric WaveNet model."""
    model = ParametricWaveNet(
        layer_configs=LAYER_CONFIGS,
        head_scale=HEAD_SCALE,
        condition_size=CONDITION_SIZE,
    )

    # Resume from checkpoint for fine-tuning
    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)
        state = {
            k.removeprefix("_model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("_model.")
        }
        model.load_state_dict(state)
        print(f"Resumed from: {args.resume}")

    print(f"Receptive field: {model.receptive_field}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, val_loader = load_data(args.data_dir, model.receptive_field)
    lit_model = ParametricLightningModule(model, lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb],
        default_root_dir=str(out_dir),
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # Load best checkpoint weights into model for the standalone .pt save
    if checkpoint_cb.best_model_path:
        best_ckpt = torch.load(checkpoint_cb.best_model_path, weights_only=False)
        state = {
            k.removeprefix("_model."): v
            for k, v in best_ckpt["state_dict"].items()
            if k.startswith("_model.")
        }
        model.load_state_dict(state)
        print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
        print(f"Best val_loss: {checkpoint_cb.best_model_score:.6f}")

    save_path = out_dir / "parametric_wavenet_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "layer_configs": LAYER_CONFIGS,
            "head_config": None,
            "head_scale": HEAD_SCALE,
            "condition_size": CONDITION_SIZE,
            "receptive_field": model.receptive_field,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


# ─── Inference ────────────────────────────────────────────────────────────────


def do_infer(args):
    """Run inference with a trained parametric WaveNet model."""
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model = ParametricWaveNet(
        layer_configs=ckpt["layer_configs"],
        head_config=ckpt["head_config"],
        head_scale=ckpt["head_scale"],
        condition_size=ckpt["condition_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = wav_to_tensor(args.input_wav).to(device)
    od1_n, od2_n = normalize_params(args.od1, args.od2)
    params = torch.tensor([[od1_n, od2_n]], dtype=torch.float32, device=device)

    print(f"Input: {len(x)} samples ({len(x) / SAMPLE_RATE:.1f}s)")
    print(f"OD1={args.od1} ({od1_n:.3f}), OD2={args.od2} ({od2_n:.3f})")

    with torch.no_grad():
        y = model(params, x.unsqueeze(0), pad_start=True).squeeze(0)

    tensor_to_wav(y, args.output_wav, rate=SAMPLE_RATE)
    print(f"Output saved to {args.output_wav}")


# ─── Export to .nam ───────────────────────────────────────────────────────


def _export_param_encoder_weights(param_encoder):
    """Flatten param encoder MLP weights: [linear1.w, linear1.b, linear2.w, linear2.b]"""
    tensors = []
    for module in param_encoder:
        if isinstance(module, nn.Linear):
            tensors.append(module.weight.data.flatten())
            tensors.append(module.bias.data.flatten())
    return torch.cat(tensors)


def _export_wavenet_weights(model):
    """Flatten layer array + head + head_scale weights in NAM export order."""
    tensors = [la.export_weights() for la in model._layer_arrays]
    weights = torch.cat(tensors)
    if model._head is not None:
        weights = torch.cat([weights, model._head.export_weights()])
    weights = torch.cat([weights.cpu(), torch.tensor([model._head_scale])])
    return weights


def export_nam(model, output_path):
    """Export a trained ParametricWaveNet to .nam JSON format.

    Weight order: [param_encoder_weights..., wavenet_weights..., head_scale]
    The C++ loader should read param_encoder weights first (based on
    param_encoder config), then the standard WaveNet weights.
    """
    model.eval()
    model.cpu()

    # Param encoder config
    linear1 = model._param_encoder[0]
    linear2 = model._param_encoder[2]
    param_encoder_config = {
        "num_params": linear1.in_features,
        "hidden_size": linear1.out_features,
        "condition_size": linear2.out_features,
        "activation": "Tanh",
        "params": [
            {"name": "OD1", "minimum": 2.0, "maximum": 10.0},
            {"name": "OD2", "minimum": 2.0, "maximum": 10.0},
        ],
    }

    # WaveNet config (reuses NAM's export_config which handles FiLM)
    layers_config = [la.export_config() for la in model._layer_arrays]
    head_config = None if model._head is None else model._head.export_config()

    # Flatten all weights: param_encoder first, then WaveNet
    param_enc_weights = _export_param_encoder_weights(model._param_encoder)
    wavenet_weights = _export_wavenet_weights(model)
    all_weights = torch.cat([param_enc_weights, wavenet_weights])

    now = datetime.now()
    nam_dict = {
        "version": "0.6.0",
        "metadata": {
            "date": {
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
            },
        },
        "architecture": "ParametricWaveNet",
        "config": {
            "param_encoder": param_encoder_config,
            "layers": layers_config,
            "head": head_config,
            "head_scale": model._head_scale,
        },
        "weights": all_weights.detach().numpy().tolist(),
        "sample_rate": SAMPLE_RATE,
    }

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(nam_dict, f)

    # Summary
    n_enc = param_enc_weights.numel()
    n_wn = wavenet_weights.numel()
    print(f"Exported to {output_path}")
    print(f"  Param encoder weights: {n_enc}")
    print(f"  WaveNet weights:       {n_wn}")
    print(f"  Total weights:         {all_weights.numel()}")
    print(f"  File size:             {output_path.stat().st_size / 1024:.0f} KB")


def do_export(args):
    """Export trained .pt model to .nam format."""
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model = ParametricWaveNet(
        layer_configs=ckpt["layer_configs"],
        head_config=ckpt["head_config"],
        head_scale=ckpt["head_scale"],
        condition_size=ckpt["condition_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    export_nam(model, args.output)
    print(f"\nTo load in C++, parse param_encoder weights first ({model._param_encoder[0].in_features} -> "
          f"{model._param_encoder[2].out_features} MLP with Tanh), "
          f"then standard WaveNet weights.")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Parametric WaveNet training for ADA MP-1"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train the parametric model")
    tp.add_argument(
        "--data-dir",
        required=True,
        help='Directory with input.wav and capture WAVs (e.g. "ADA MP-1 Captures FULL 1-25")',
    )
    tp.add_argument(
        "--output-dir",
        default="parametric_output",
        help="Output directory for checkpoints and model (default: parametric_output)",
    )
    tp.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Max training epochs (default: {MAX_EPOCHS})",
    )
    tp.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate (default: {LR})",
    )
    tp.add_argument(
        "--resume",
        default=None,
        help="Lightning .ckpt to resume from (fine-tuning; resets epoch counter)",
    )

    ep = sub.add_parser("export", help="Export .pt model to .nam format")
    ep.add_argument("--checkpoint", required=True, help="Path to .pt model file")
    ep.add_argument(
        "--output",
        default="parametric_wavenet.nam",
        help="Output .nam file path (default: parametric_wavenet.nam)",
    )

    ip = sub.add_parser("infer", help="Run inference with a trained model")
    ip.add_argument("--checkpoint", required=True, help="Path to .pt model file")
    ip.add_argument("--input-wav", required=True, help="Input WAV file (DI signal)")
    ip.add_argument("--output-wav", required=True, help="Output WAV file path")
    ip.add_argument(
        "--od1", type=float, required=True, help="OD1 knob value (2-10)"
    )
    ip.add_argument(
        "--od2", type=float, required=True, help="OD2 knob value (2-10)"
    )

    args = parser.parse_args()
    if args.command == "train":
        do_train(args)
    elif args.command == "export":
        do_export(args)
    elif args.command == "infer":
        do_infer(args)


if __name__ == "__main__":
    main()
