"""Tests for per-capture ESR logging in train_parametric."""
import json
import pytest
import torch

from train_parametric import ParametricDataset


def test_paramdataset_returns_capture_id():
    """Dataset must return (params, x, y, capture_id) so validation can bin per capture."""
    n_captures = 3
    nx = 64
    ny = 32
    x = torch.randn(1000)
    ys = [torch.randn(1000) for _ in range(n_captures)]
    params_list = [torch.tensor([0.5, 0.5]) for _ in range(n_captures)]

    ds = ParametricDataset(x, ys, params_list, nx, ny)

    sample = ds[0]
    assert len(sample) == 4, f"Expected 4-tuple (params, x, y, capture_id), got {len(sample)}-tuple"
    _params, _x_seg, _y_seg, capture_id = sample
    assert isinstance(capture_id, int)
    assert 0 <= capture_id < n_captures


def test_compute_per_capture_metrics_groups_correctly():
    """Helper bins per-sample ESR back to captures and aggregates by each param.

    Covers the generalized N-knob schema: capture_table entries carry
    {"capture", "values"}, param_specs labels each value column, and the
    output contains one by_<name> bucket per param plus by_combined.
    For the 2-knob ADA MP-1 case here the buckets reduce to the same numbers
    as the pre-refactor by_od1 / by_od2 / by_od1_od2 contract.
    """
    from train_parametric import _compute_per_capture_metrics

    param_specs = [
        {"name": "od1", "minimum": 2, "maximum": 10},
        {"name": "od2", "minimum": 2, "maximum": 10},
    ]
    capture_table = [
        {"capture": 1, "values": [2, 4]},
        {"capture": 2, "values": [4, 4]},
        {"capture": 3, "values": [2, 6]},
    ]
    errors_by_capture = {
        0: [0.01, 0.02, 0.01],
        1: [0.05, 0.04],
        2: [0.02],
    }

    out = _compute_per_capture_metrics(errors_by_capture, capture_table, param_specs)

    assert out["per_capture"]["1"]["esr_pe_mean"] == pytest.approx(0.0133, abs=1e-3)
    assert out["per_capture"]["1"]["od1"] == 2
    assert out["per_capture"]["1"]["od2"] == 4
    assert out["per_capture"]["2"]["esr_pe_mean"] == pytest.approx(0.045)
    assert out["per_capture"]["3"]["esr_pe_mean"] == pytest.approx(0.02)

    assert out["by_od1"]["2"]["esr_pe_mean"] == pytest.approx(0.01667, abs=1e-3)
    assert out["by_od1"]["2"]["n"] == 2
    assert out["by_od1"]["4"]["esr_pe_mean"] == pytest.approx(0.045)

    assert out["by_od2"]["4"]["esr_pe_mean"] == pytest.approx(0.02917, abs=1e-3)
    assert out["by_od2"]["4"]["n"] == 2

    assert out["by_combined"]["2_4"]["esr_pe_mean"] == pytest.approx(0.0133, abs=1e-3)
    assert out["by_combined"]["2_4"]["n"] == 1
    assert out["by_combined"]["4_4"]["esr_pe_mean"] == pytest.approx(0.045)
    assert out["by_combined"]["2_6"]["esr_pe_mean"] == pytest.approx(0.02)

    assert out["mean_esr_pe"] == pytest.approx((0.0133 + 0.045 + 0.02) / 3, abs=1e-3)


def _write_minimal_params_json(dir_path, captures, names=("od1", "od2")):
    """Write a 2-knob params.json matching the given captures list.

    ``captures`` is a list of (cap_num, value_tuple) pairs. The values must
    match the length of ``names``.
    """
    import json as _json
    cfg = {
        "params": [
            {"name": n, "minimum": 2.0, "maximum": 10.0} for n in names
        ],
        "captures": [
            {"capture": cap_num, "values": list(values)}
            for cap_num, values in captures
        ],
    }
    (dir_path / "params.json").write_text(_json.dumps(cfg))
    return cfg


def test_load_data_rejects_mismatched_sample_rate(tmp_path):
    """load_data must fail loudly if any capture's SR != SAMPLE_RATE."""
    import soundfile as sf
    import numpy as np
    from train_parametric import load_data, SAMPLE_RATE, _load_params_config

    sf.write(str(tmp_path / "input.wav"), np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    sf.write(
        str(tmp_path / "1 ADA MP-1 NAM.wav"),
        np.zeros(44100),
        44100,
    )
    _write_minimal_params_json(tmp_path, [(1, (2, 2))])
    param_config = _load_params_config(tmp_path)

    with pytest.raises((AssertionError, ValueError), match=r"(?i)sample.?rate"):
        load_data(tmp_path, param_config, nx=64)


def test_load_data_rejects_mismatched_sample_rate_on_later_capture(tmp_path):
    """load_data must validate ALL captures, not just the first one."""
    import soundfile as sf
    import numpy as np
    from train_parametric import (
        load_data, SAMPLE_RATE, PARAM_TABLE, _load_params_config,
    )

    pattern = "{cap_num} ADA MP-1 NAM.wav"
    sf.write(str(tmp_path / "input.wav"), np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    # Write ALL captures at correct SR except index 1 (corrupt it)
    for idx, (cap_num, _, _) in enumerate(PARAM_TABLE):
        rate = 44100 if idx == 1 else SAMPLE_RATE
        sf.write(
            str(tmp_path / pattern.format(cap_num=cap_num)),
            np.zeros(rate),
            rate,
        )
    _write_minimal_params_json(
        tmp_path,
        [(cap_num, (od1, od2)) for cap_num, od1, od2 in PARAM_TABLE],
    )
    param_config = _load_params_config(tmp_path)

    with pytest.raises((AssertionError, ValueError), match=r"(?i)sample.?rate"):
        load_data(tmp_path, param_config, nx=64)


@pytest.mark.parametrize("preset_name", ["small", "small_k16head", "a2_small"])
def test_preset_constructs_and_forward_passes(preset_name):
    """Each preset must construct a ParametricWaveNet that runs a forward pass."""
    from train_parametric import (
        _build_layer_configs,
        ParametricWaveNet,
        _MODEL_PRESETS,
    )

    preset = _MODEL_PRESETS[preset_name]
    layer_configs = _build_layer_configs(preset_name)
    model = ParametricWaveNet(
        layer_configs,
        head_config=None,
        head_scale=preset["head_scale"],
        num_params=2,
        condition_size=preset["condition_size"],
    )
    # Forward pass with enough samples to exceed the receptive field
    params = torch.zeros(1, 2)
    x = torch.zeros(1, 4096)
    y = model(params, x, pad_start=True)
    assert y.shape == (1, 4096), f"Expected (1, 4096), got {y.shape}"


def test_a2_constants_match_upstream():
    """Detect silent drift if upstream changes the A2 config."""
    import subprocess
    import json
    from train_parametric import _A2_KERNEL_SIZES, _A2_DILATIONS, _A2_HEAD_KERNEL

    result = subprocess.run(
        ["git", "show", "upstream/main:nam/train/_resources/config_model_packed.json"],
        capture_output=True, text=True, check=True,
        cwd="/Users/jmilo/src/guitar-plugin/neural-amp-modeler",
    )
    cfg = json.loads(result.stdout)
    for sm in cfg["net"]["config"]["submodels"]:
        if sm["name"] == "channels_8":
            la = sm["config"]["layers_configs"][0]
            assert la["kernel_sizes"] == _A2_KERNEL_SIZES, "upstream A2 kernel_sizes drifted"
            assert la["dilations"] == _A2_DILATIONS, "upstream A2 dilations drifted"
            assert la["head"]["kernel_size"] == _A2_HEAD_KERNEL, "upstream A2 head_kernel drifted"
            return
    raise AssertionError("upstream config missing channels_8 submodel")


def test_head_kernel_size_is_actually_applied():
    """Both new presets must construct a model with a kernel-16 head, not kernel-1.

    Pinpoints the bug where head_kernel_size was metadata-only and the trained
    model silently used kernel-1.
    """
    from train_parametric import (
        _build_layer_configs,
        ParametricWaveNet,
        _MODEL_PRESETS,
    )

    for preset_name in ("small_k16head", "a2_small"):
        preset = _MODEL_PRESETS[preset_name]
        model = ParametricWaveNet(
            _build_layer_configs(preset_name),
            head_config=None,
            head_scale=preset["head_scale"],
            num_params=2,
            condition_size=preset["condition_size"],
        )
        # The LAST layer array's head_rechannel is the model's effective output head.
        last_la = model._layer_arrays[-1]
        actual_kernel = last_la._head_rechannel.kernel_size[0]
        assert actual_kernel == 16, (
            f"{preset_name}: expected head_rechannel kernel 16 "
            f"(aliasing fix), got {actual_kernel}"
        )

    # And verify "small" still uses kernel 1 (back-compat).
    small_preset = _MODEL_PRESETS["small"]
    small_model = ParametricWaveNet(
        _build_layer_configs("small"),
        head_config=None,
        head_scale=small_preset["head_scale"],
        num_params=2,
        condition_size=small_preset["condition_size"],
    )
    assert small_model._layer_arrays[-1]._head_rechannel.kernel_size[0] == 1


def test_a2_export_json_includes_new_fields(tmp_path):
    """Exported a2_small .nam JSON must include kernel_sizes (per-layer), head kernel_size
    16, and LeakyReLU activation."""
    from train_parametric import (
        _build_layer_configs,
        ParametricWaveNet,
        _MODEL_PRESETS,
        export_nam,
    )

    preset = _MODEL_PRESETS["a2_small"]
    model = ParametricWaveNet(
        _build_layer_configs("a2_small"),
        head_config=None,
        head_scale=preset["head_scale"],
        num_params=2,
        condition_size=preset["condition_size"],
    )
    out = tmp_path / "a2.nam"
    export_nam(
        model, str(out), sample_rate=48000,
        param_specs=[
            {"name": "OD1", "minimum": 2.0, "maximum": 10.0},
            {"name": "OD2", "minimum": 2.0, "maximum": 10.0},
        ],
    )
    data = json.loads(out.read_text())
    # Export uses "layers" key (not "layers_configs")
    layer = data["config"]["layers"][0]

    # Per-layer kernel_sizes present and correct
    assert "kernel_sizes" in layer, "kernel_sizes not in exported layer config"
    assert layer["kernel_sizes"][:3] == [6, 6, 6], "unexpected kernel_sizes prefix"

    # Head kernel size 16 — native field emitted by _LayerArray.export_config()
    assert layer.get("head_kernel_size") == 16, (
        f"expected head_kernel_size 16, got {layer.get('head_kernel_size')!r}"
    )

    # LeakyReLU activation — export_config serializes each layer's activation
    # as a list of per-layer dicts with "type" key; check the first entry.
    activation = layer["activation"]
    first = activation[0] if isinstance(activation, list) else activation
    if isinstance(first, str):
        act_type = first
    else:
        act_type = first.get("type")
    assert act_type == "LeakyReLU", f"expected LeakyReLU, got {act_type!r}"


def test_small_k16head_export_json(tmp_path):
    """small_k16head must export with kernel_size 3 (per-layer or scalar)
    and head kernel 16 on all layer arrays."""
    from train_parametric import (
        _build_layer_configs,
        ParametricWaveNet,
        _MODEL_PRESETS,
        export_nam,
    )

    preset = _MODEL_PRESETS["small_k16head"]
    model = ParametricWaveNet(
        _build_layer_configs("small_k16head"),
        head_config=None,
        head_scale=preset["head_scale"],
        num_params=2,
        condition_size=preset["condition_size"],
    )
    out = tmp_path / "k16head.nam"
    export_nam(
        model, str(out), sample_rate=48000,
        param_specs=[
            {"name": "OD1", "minimum": 2.0, "maximum": 10.0},
            {"name": "OD2", "minimum": 2.0, "maximum": 10.0},
        ],
    )
    data = json.loads(out.read_text())
    # Export uses "layers" key
    layers = data["config"]["layers"]
    assert len(layers) == 2, f"expected 2 layer arrays, got {len(layers)}"
    for layer in layers:
        assert layer.get("head_kernel_size") == 16, (
            f"expected head_kernel_size 16, got {layer.get('head_kernel_size')!r}"
        )
