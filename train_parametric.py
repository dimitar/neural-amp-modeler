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
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
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

HEAD_SCALE = 0.02
NY = 8192
BATCH_SIZE = 16
LR = 4e-3
LR_GAMMA = 0.993
MAX_EPOCHS = 250
PRE_EMPH_COEF = 0.85

LR_WARMUP_EPOCHS = 5  # Linear LR ramp-up over first N epochs

# Model size presets. "small" is the production default per CLAUDE.md
# (~26K params, ~0.007 ESR on 24 captures). "large" (~101K params) gives
# only marginal improvement and is ~3-4x slower at inference.
_MODEL_PRESETS = {
    "small": {
        "channels": (16, 8),
        "condition_size": 16,
        "head_size": (8, 1),
        # Explicit fields that _build_layer_configs was previously hardcoding:
        "kernel_size": 3,
        "head_kernel_size": 1,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "activation": "Tanh",
        "head_scale": 0.02,
    },
    "large": {
        "channels": (32, 16),
        "condition_size": 32,
        "head_size": (16, 1),
        "kernel_size": 3,
        "head_kernel_size": 1,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "activation": "Tanh",
        "head_scale": 0.02,
    },
    # small_k16head: current architecture with head_kernel_size bumped from 1 to 16.
    # Isolates A2's aliasing-reduction variable. Differs from "small" in
    # head_kernel_size only.
    "small_k16head": {
        "channels": (16, 8),
        "condition_size": 16,
        "head_size": (8, 1),
        "kernel_size": 3,
        "head_kernel_size": 16,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "activation": "Tanh",
        "head_scale": 0.02,
    },
}
DEFAULT_MODEL_SIZE = "small"

# A2 upstream constants pinned verbatim from
# upstream NAM v0.13.0 nam/train/_resources/config_model_packed.json (channels_8 submodel).
# Per-layer kernel_sizes: 14×6, 15, 15, 7×6. Dilations reset twice.
_A2_KERNEL_SIZES = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6]
_A2_DILATIONS   = [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239]
_A2_HEAD_KERNEL = 16

# a2_small: full A2 bundle verbatim from upstream channels_8 submodel.
_MODEL_PRESETS["a2_small"] = {
    "channels": (8,),
    "condition_size": 16,
    "head_size": (1,),
    "kernel_sizes": _A2_KERNEL_SIZES,   # per-layer; not a scalar
    "head_kernel_size": _A2_HEAD_KERNEL,
    "dilations": _A2_DILATIONS,
    "activation": "LeakyReLU",
    "activation_kwargs": {"negative_slope": 0.01},
    "head_scale": 0.01,
}

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


def _build_layer_configs(model_size):
    """Build layer_configs list for the given preset name.

    Supports presets with:
    - Scalar "kernel_size" (broadcast to all layers) or per-layer "kernel_sizes" list.
    - String "activation" optionally combined with "activation_kwargs" dict (e.g. for
      LeakyReLU), assembled into a dict that _LayerArray.init_from_config accepts.
    - Variable number of layer arrays: len(channels) arrays, not hardcoded to 2.

    head_kernel_size is stored in the preset for export purposes but is NOT passed to
    _LayerArray.init_from_config (HeadRechannel always uses kernel_size=1 internally).
    """
    p = _MODEL_PRESETS[model_size]
    channels = p["channels"]
    cs = p["condition_size"]
    head_sizes = p["head_size"]

    # Activation: string + optional kwargs → dict form that _LayerArray accepts.
    activation_str = p["activation"]
    activation_kwargs = p.get("activation_kwargs")
    if activation_kwargs:
        activation = {"name": activation_str, **activation_kwargs}
    else:
        activation = activation_str

    # Kernel sizes: prefer per-layer list, fall back to scalar broadcast.
    if "kernel_sizes" in p:
        kernel_field = {"kernel_sizes": p["kernel_sizes"]}
    else:
        kernel_field = {"kernel_size": p["kernel_size"]}

    dilations = p["dilations"]

    layer_configs = []
    for i, (ch, hs) in enumerate(zip(channels, head_sizes)):
        input_size = 1 if i == 0 else channels[i - 1]
        layer_configs.append({
            "input_size": input_size,
            "condition_size": cs,
            "head_size": hs,
            "channels": ch,
            **kernel_field,
            "dilations": dilations,
            "activation": activation,
            # head_kernel_size > 1 backports upstream's aliasing-reduction
            # head conv (see nam/models/wavenet/_layer_array.py). Default 1
            # preserves back-compat for "small"/"large".
            "head_kernel_size": p.get("head_kernel_size", 1),
            "film_params": _FILM_PARAMS,
        })

    return layer_configs


def normalize_params(od1, od2):
    """Normalize knob values from [2, 10] to [0, 1]."""
    return (od1 - 2) / 8, (od2 - 2) / 8


# ─── Dataset ──────────────────────────────────────────────────────────────────


class ParametricDataset(Dataset):
    """Dataset returning (params, x_segment, y_segment, capture_id) tuples across all captures."""

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
        return self._params_list[capture_idx], x_seg, y_seg, capture_idx


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
        # Inter-array signal is layer output x (not trimmed by head_rechannel);
        # only the final array's head_rechannel kernel reduces final output.
        if not self._layer_arrays:
            return 1
        intermediate = sum(
            la._layers_receptive_field - 1 for la in self._layer_arrays[:-1]
        )
        last = self._layer_arrays[-1].receptive_field - 1
        return 1 + intermediate + last

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


# ─── Per-capture metrics ──────────────────────────────────────────────────────


def _compute_per_capture_metrics(errors_by_capture, capture_table):
    """Aggregate per-sample errors by capture, OD1, OD2.

    ESR is computed with pre-emphasis (PRE_EMPH_COEF=0.85), matching training
    loss. Keys are suffixed with `_pe` to distinguish from plain val_ESR.

    Args:
        errors_by_capture: dict[capture_idx, list[float]] of ESR per sample
        capture_table: list of dicts with cap_num, od1, od2 (index = capture_idx)

    Returns:
        dict with keys: per_capture, by_od1, by_od2, by_od1_od2, mean_esr_pe
    """
    per_capture = {}
    by_od1 = {}
    by_od2 = {}
    by_od1_od2 = {}

    capture_means = []
    for cap_idx, samples in errors_by_capture.items():
        if not samples:
            continue
        meta = capture_table[cap_idx]
        cap_num = str(meta["cap_num"])
        od1 = str(meta["od1"])
        od2 = str(meta["od2"])
        bucket = f"{od1}_{od2}"

        mean_esr = float(np.mean(samples))
        max_esr = float(np.max(samples))
        capture_means.append(mean_esr)

        per_capture[cap_num] = {
            "od1": meta["od1"],
            "od2": meta["od2"],
            "esr_pe_mean": mean_esr,
            "esr_pe_max": max_esr,
            "n_samples": len(samples),
        }
        by_od1.setdefault(od1, []).append(mean_esr)
        by_od2.setdefault(od2, []).append(mean_esr)
        by_od1_od2.setdefault(bucket, []).append(mean_esr)

    def _agg(d):
        return {
            k: {"esr_pe_mean": float(np.mean(v)), "esr_pe_max": float(np.max(v)), "n": len(v)}
            for k, v in d.items()
        }

    return {
        "per_capture": per_capture,
        "by_od1": _agg(by_od1),
        "by_od2": _agg(by_od2),
        "by_od1_od2": _agg(by_od1_od2),
        "mean_esr_pe": float(np.mean(capture_means)) if capture_means else 0.0,
    }


# ─── Lightning Module ─────────────────────────────────────────────────────────


class ParametricLightningModule(pl.LightningModule):
    def __init__(self, model, lr=LR, lr_gamma=LR_GAMMA,
                 warmup_epochs=LR_WARMUP_EPOCHS, capture_table=None):
        super().__init__()
        self._model = model
        self._lr = lr
        self._lr_gamma = lr_gamma
        self._warmup_epochs = warmup_epochs
        self._capture_table = capture_table or []
        self._val_errors_by_capture = {}
        self._last_val_metrics = None

    def forward(self, params, x, **kwargs):
        return self._model(params, x, **kwargs)

    def training_step(self, batch, batch_idx):
        params, x_seg, y_seg, _ = batch
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
        params, x_seg, y_seg, capture_id = batch
        preds = self._model(params, x_seg, pad_start=False)
        loss_mse = F.mse_loss(preds, y_seg)
        self.log("val_loss", loss_mse, prog_bar=True)
        self.log("val_ESR", esr(preds, y_seg))
        self.log("val_MSE", loss_mse)

        # Bin per-sample ESR by capture for per-epoch summary
        for i in range(preds.shape[0]):
            cid = int(capture_id[i])
            sample_esr = float(esr(
                apply_pre_emphasis_filter(preds[i:i+1], PRE_EMPH_COEF),
                apply_pre_emphasis_filter(y_seg[i:i+1], PRE_EMPH_COEF),
            ))
            self._val_errors_by_capture.setdefault(cid, []).append(sample_esr)

        return loss_mse

    def on_validation_epoch_end(self):
        if not self._val_errors_by_capture or not self._capture_table:
            self._val_errors_by_capture = {}
            return
        metrics = _compute_per_capture_metrics(
            self._val_errors_by_capture, self._capture_table
        )
        self._last_val_metrics = metrics

        per_cap = list(metrics["per_capture"].items())
        per_cap.sort(key=lambda kv: kv[1]["esr_pe_mean"])
        print(
            f"\n[epoch {self.current_epoch}] "
            f"mean ESR(pe) {metrics['mean_esr_pe']:.5f} | "
            f"best 3: {[(k, round(v['esr_pe_mean'], 5)) for k, v in per_cap[:3]]} | "
            f"worst 3: {[(k, round(v['esr_pe_mean'], 5)) for k, v in per_cap[-3:]]}"
        )
        self._val_errors_by_capture = {}

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


def _detect_capture_pattern(data_dir):
    """Auto-detect the WAV filename pattern from the data directory."""
    data_dir = Path(data_dir)
    # Try known patterns in order
    patterns = [
        "{cap_num} ADA MP-1 NAM.wav",
        "{cap_num} ADA MP-1 PS2 96khz.wav",
    ]
    for pat in patterns:
        test_path = data_dir / pat.format(cap_num=1)
        if test_path.exists():
            return pat
    # Fallback: find any WAV starting with "1 "
    for f in data_dir.glob("1 *.wav"):
        if f.name != "input.wav":
            # Extract pattern by replacing leading "1 " with "{cap_num} "
            return "{cap_num} " + f.name[2:]
    raise FileNotFoundError(
        f"Cannot detect capture WAV pattern in {data_dir}. "
        "Expected files like '1 ADA MP-1 NAM.wav'."
    )


def _resample_tensor(x, from_rate, to_rate):
    """Resample a 1D tensor using linear interpolation."""
    if from_rate == to_rate:
        return x
    ratio = to_rate / from_rate
    new_len = int(len(x) * ratio)
    # Use torch interpolate (expects [batch, channels, length])
    x_3d = x.unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(x_3d, size=new_len, mode="linear", align_corners=False)
    return resampled.squeeze()


def load_data(
    data_dir,
    nx,
    ny_train=NY,
    delay_samples=DELAY_SAMPLES,
    train_stop_seconds=None,
    val_start_seconds=None,
):
    """Load all WAV files, apply delay correction, and create train/val datasets.

    Args:
        train_stop_seconds: Where to stop training data. Negative values are
            relative to end of file (e.g. -9.0 means stop 9s before end).
            None uses TRAIN_SECONDS from start.
        val_start_seconds: Where validation data starts. Negative values are
            relative to end of file. None means immediately after training data.
    """
    data_dir = Path(data_dir)
    cap_pattern = _detect_capture_pattern(data_dir)
    print(f"Capture pattern: {cap_pattern}")

    # Detect sample rates
    input_info = sf.info(str(data_dir / "input.wav"))
    first_cap_path = data_dir / cap_pattern.format(cap_num=PARAM_TABLE[0][0])
    capture_info = sf.info(str(first_cap_path))
    target_rate = capture_info.samplerate
    if target_rate != SAMPLE_RATE:
        raise ValueError(
            f"Capture sample rate {target_rate} Hz != expected sample rate "
            f"{SAMPLE_RATE} Hz. Re-export captures at {SAMPLE_RATE} Hz."
        )
    print(f"Input sample rate: {input_info.samplerate} Hz")
    print(f"Capture sample rate: {target_rate} Hz")

    x = wav_to_tensor(data_dir / "input.wav")
    if input_info.samplerate != target_rate:
        print(
            f"Resampling input from {input_info.samplerate} Hz to {target_rate} Hz..."
        )
        x = _resample_tensor(x, input_info.samplerate, target_rate)
    print(f"Input: {len(x)} samples ({len(x) / target_rate:.1f}s @ {target_rate} Hz)")

    ys = []
    params_list = []
    for cap_num, od1, od2 in PARAM_TABLE:
        wav_path = data_dir / cap_pattern.format(cap_num=cap_num)
        cap_rate = sf.info(str(wav_path)).samplerate
        if cap_rate != SAMPLE_RATE:
            raise ValueError(
                f"Capture {cap_num} sample rate {cap_rate} Hz != expected "
                f"sample rate {SAMPLE_RATE} Hz ({wav_path.name}). "
                f"Re-export captures at {SAMPLE_RATE} Hz."
            )
        y = wav_to_tensor(wav_path)
        # Delay correction: output is behind input by delay_samples
        ys.append(y[delay_samples:])
        params_list.append(normalize_params(od1, od2))

    if delay_samples > 0:
        x = x[:-delay_samples]

    for i, y in enumerate(ys):
        assert len(y) == len(x), (
            f"Capture {PARAM_TABLE[i][0]}: length {len(y)} != input length {len(x)}"
        )

    total_samples = len(x)
    total_seconds = total_samples / target_rate

    # Compute train/val split
    if train_stop_seconds is not None:
        if train_stop_seconds < 0:
            train_stop = total_samples + int(train_stop_seconds * target_rate)
        else:
            train_stop = int(train_stop_seconds * target_rate)
    else:
        train_stop = TRAIN_SECONDS * SAMPLE_RATE

    if val_start_seconds is not None:
        if val_start_seconds < 0:
            val_start = total_samples + int(val_start_seconds * target_rate)
        else:
            val_start = int(val_start_seconds * target_rate)
    else:
        val_start = train_stop

    train_seconds = train_stop / target_rate
    val_seconds = (total_samples - val_start) / target_rate
    print(f"Total: {total_seconds:.1f}s ({total_samples} samples)")
    print(f"Train: {train_seconds:.1f}s ({train_stop} samples)")
    print(f"Val: {val_seconds:.1f}s ({total_samples - val_start} samples)")

    train_ds = ParametricDataset(x, ys, params_list, nx, ny_train, 0, train_stop)

    # Validation: one chunk per capture covering the full validation segment
    val_len = total_samples - val_start
    val_ny = val_len - nx + 1
    val_ds = ParametricDataset(x, ys, params_list, nx, val_ny, val_start, None)

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
    return train_loader, val_loader, target_rate


# ─── Training ─────────────────────────────────────────────────────────────────


class _MetricsLogger(pl.callbacks.Callback):
    """Print compact per-epoch metrics with best-tracking and ETA."""

    def __init__(self):
        super().__init__()
        self._best_val_loss = float("inf")
        self._best_val_esr = float("inf")
        self._best_loss_epoch = -1
        self._best_esr_epoch = -1
        self._epoch_start = None
        self._epoch_durations = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        epoch = trainer.current_epoch

        def _get(key):
            v = m.get(key)
            return float(v) if v is not None else None

        val_loss = _get("val_loss")
        val_esr = _get("val_ESR")
        train_esr = _get("train_ESR_step") or _get("train_ESR")

        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_loss_epoch = epoch
        if val_esr is not None and val_esr < self._best_val_esr:
            self._best_val_esr = val_esr
            self._best_esr_epoch = epoch

        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        self._epoch_durations.append(elapsed)
        recent = self._epoch_durations[-5:]
        avg = sum(recent) / len(recent)
        remaining = max(0, trainer.max_epochs - epoch - 1)
        eta_s = int(avg * remaining)
        eta_h, rem = divmod(eta_s, 3600)
        eta_m = rem // 60

        def _fmt(x):
            return f"{x:.6f}" if x is not None else "--------"

        print(
            f"\n+- epoch {epoch:3d}/{trainer.max_epochs}  "
            f"({elapsed:.0f}s, ETA {eta_h}h{eta_m:02d}m)\n"
            f"|  val_loss : {_fmt(val_loss)}  "
            f"(best {self._best_val_loss:.6f} @ epoch {self._best_loss_epoch})\n"
            f"|  val_ESR  : {_fmt(val_esr)}  "
            f"(best {self._best_val_esr:.6f} @ epoch {self._best_esr_epoch})\n"
            f"|  train_ESR: {_fmt(train_esr)}\n"
            f"+-",
            flush=True,
        )


def _write_per_capture_report(metrics, out_dir, architecture, epochs_trained):
    """Write per-capture ESR report as JSON + markdown table."""
    report = {
        "architecture": architecture,
        "epochs_trained": epochs_trained,
        **metrics,
    }
    json_path = Path(out_dir) / "per_capture_esr.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\nPer-capture ESR report → {json_path}")

    lines = [
        "\n| cap | OD1 | OD2 | ESR(pe) mean |",
        "|----:|----:|----:|-------------:|",
    ]
    rows = sorted(
        metrics["per_capture"].items(),
        key=lambda kv: (kv[1]["od1"], kv[1]["od2"]),
    )
    for cap_num, vals in rows:
        lines.append(
            f"| {cap_num} | {vals['od1']} | {vals['od2']} | {vals['esr_pe_mean']:.5f} |"
        )
    lines.append(
        f"\n**Mean ESR(pe) over {len(rows)} captures:** {metrics['mean_esr_pe']:.5f}"
    )
    print("\n".join(lines))


def do_train(args):
    """Train the parametric WaveNet model."""
    preset = _MODEL_PRESETS[args.model_size]
    layer_configs = _build_layer_configs(args.model_size)
    print(f"Model size: {args.model_size} "
          f"(channels={preset['channels']}, "
          f"condition_size={preset['condition_size']}, "
          f"head_size={preset['head_size']})")
    model = ParametricWaveNet(
        layer_configs=layer_configs,
        head_scale=HEAD_SCALE,
        condition_size=preset["condition_size"],
    )

    # Fine-tune: load weights only, reset epoch counter
    if args.finetune:
        ckpt = torch.load(args.finetune, weights_only=False)
        state = {
            k.removeprefix("_model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("_model.")
        }
        model.load_state_dict(state)
        print(f"Fine-tuning from: {args.finetune}")

    print(f"Receptive field: {model.receptive_field}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, val_loader, target_rate = load_data(
        args.data_dir,
        model.receptive_field,
        delay_samples=args.delay,
        train_stop_seconds=args.train_stop_seconds,
        val_start_seconds=args.val_start_seconds,
    )
    capture_table = [
        {"cap_num": cap_num, "od1": od1, "od2": od2}
        for cap_num, od1, od2 in PARAM_TABLE
    ]
    lit_model = ParametricLightningModule(model, lr=args.lr, capture_table=capture_table)

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
        callbacks=[checkpoint_cb, _MetricsLogger()],
        default_root_dir=str(out_dir),
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    # Resume: restore full state (weights, optimizer, epoch, LR scheduler)
    ckpt_path = args.resume if args.resume else None
    if ckpt_path:
        print(f"Resuming from: {ckpt_path}")
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)

    if lit_model._last_val_metrics is not None:
        _write_per_capture_report(
            lit_model._last_val_metrics,
            out_dir=out_dir,
            architecture=args.model_size,
            epochs_trained=trainer.current_epoch + 1,
        )

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
            "layer_configs": layer_configs,
            "head_config": None,
            "head_scale": HEAD_SCALE,
            "condition_size": preset["condition_size"],
            "receptive_field": model.receptive_field,
            "model_size": args.model_size,
            "sample_rate": target_rate,
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


def export_nam(model, output_path, sample_rate):
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

    # WaveNet config (reuses NAM's export_config which handles FiLM).
    # _LayerArray.export_config() now emits head_kernel_size natively (backported
    # from upstream PR #653) — no augmentation needed.
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
        "sample_rate": sample_rate,
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
    sample_rate = ckpt.get("sample_rate")
    if sample_rate is None:
        if args.sample_rate is None:
            raise SystemExit(
                "Checkpoint has no sample_rate; pass --sample-rate explicitly "
                "(e.g. --sample-rate 96000 for the ADA MP-1 96kHz captures)."
            )
        sample_rate = args.sample_rate
    print(f"Sample rate: {sample_rate} Hz")
    export_nam(model, args.output, sample_rate)
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
        help="Lightning .ckpt to resume from (restores weights, optimizer, epoch, LR)",
    )
    tp.add_argument(
        "--finetune",
        default=None,
        help="Lightning .ckpt to load weights from (resets epoch counter and optimizer)",
    )
    tp.add_argument(
        "--delay",
        type=int,
        default=DELAY_SAMPLES,
        help=f"Output delay in samples (default: {DELAY_SAMPLES})",
    )
    tp.add_argument(
        "--train-stop-seconds",
        type=float,
        default=None,
        help="Where to stop training data. Negative = relative to end (e.g. -9.0). "
        f"Default: {TRAIN_SECONDS}s from start.",
    )
    tp.add_argument(
        "--val-start-seconds",
        type=float,
        default=None,
        help="Where validation data starts. Negative = relative to end (e.g. -9.0). "
        "Default: immediately after training data.",
    )
    tp.add_argument(
        "--model-size",
        choices=list(_MODEL_PRESETS.keys()),
        default=DEFAULT_MODEL_SIZE,
        help=f"Model preset (default: {DEFAULT_MODEL_SIZE}). "
        f"'small' ~26K params, 'large' ~101K params. Must match the preset "
        f"used to train a checkpoint when --resume is used.",
    )

    ep = sub.add_parser("export", help="Export .pt model to .nam format")
    ep.add_argument("--checkpoint", required=True, help="Path to .pt model file")
    ep.add_argument(
        "--output",
        default="parametric_wavenet.nam",
        help="Output .nam file path (default: parametric_wavenet.nam)",
    )
    ep.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate to embed in .nam metadata. Required only for legacy "
             ".pt files that don't store sample_rate (e.g. produced before this "
             "field was added).",
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
