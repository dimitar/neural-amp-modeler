"""Head-kernel-size (aliasing-reduction) integration for the parametric WaveNet.

`small_k16head` is the `small` preset with the head convolution kernel bumped
1 -> 16 (A2's aliasing-reduction variable, isolated — same channels/dilations/
Tanh as `small`). Two things must hold, or the experiment is meaningless:

1. The model must ACTUALLY construct with a kernel-16 head (not silently fall
   back to 1 while the exported JSON claims 16 — the prior "lying export" bug).
2. The exported layer config must carry `head_kernel_size` so the C++ engine
   builds the matching kernel-16 head.

Plus a forward smoke test the original branch never had: the two-layer-array
kernel>1 path must run end to end and preserve output length.
"""
import torch as _torch

import train_parametric as _tp


def _build(size):
    return _tp.ParametricWaveNet(
        _tp._build_layer_configs(size),
        condition_size=_tp._MODEL_PRESETS[size]["condition_size"],
    )


def test_small_k16head_builds_kernel16_head():
    model = _build("small_k16head")
    assert model._layer_arrays[-1]._head_rechannel.kernel_size[0] == 16


def test_small_stays_kernel1_backcompat():
    model = _build("small")
    for la in model._layer_arrays:
        assert la._head_rechannel.kernel_size[0] == 1


def test_export_config_carries_head_kernel_size():
    model = _build("small_k16head")
    assert model._layer_arrays[-1].export_config()["head_kernel_size"] == 16
    # back-compat: small exports 1
    small = _build("small")
    assert small._layer_arrays[-1].export_config()["head_kernel_size"] == 1


def test_small_k16head_forward_runs_and_preserves_length():
    # Guards the two-layer-array kernel>1 length bookkeeping end to end.
    model = _build("small_k16head")
    model.eval()
    B, L = 1, 4096
    params = _torch.zeros(B, 2)
    x = _torch.randn(B, L)
    with _torch.no_grad():
        y = model(params, x, pad_start=True)
    assert y.shape == (B, L)
