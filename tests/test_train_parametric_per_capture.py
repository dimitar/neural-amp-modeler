"""Tests for per-capture ESR logging in train_parametric."""
import json
import math

import pytest
import torch

from train_parametric import ParametricDataset, PARAM_TABLE


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
    params, x_seg, y_seg, capture_id = sample
    assert isinstance(capture_id, int)
    assert 0 <= capture_id < n_captures


def test_compute_per_capture_metrics_groups_correctly():
    """Helper bins per-sample ESR back to captures and aggregates by OD1, OD2."""
    from train_parametric import _compute_per_capture_metrics

    capture_table = [
        {"cap_num": 1, "od1": 2, "od2": 4},
        {"cap_num": 2, "od1": 4, "od2": 4},
        {"cap_num": 3, "od1": 2, "od2": 6},
    ]
    errors_by_capture = {
        0: [0.01, 0.02, 0.01],
        1: [0.05, 0.04],
        2: [0.02],
    }

    out = _compute_per_capture_metrics(errors_by_capture, capture_table)

    assert out["per_capture"]["1"]["esr_mean"] == pytest.approx(0.0133, abs=1e-3)
    assert out["per_capture"]["2"]["esr_mean"] == pytest.approx(0.045)
    assert out["per_capture"]["3"]["esr_mean"] == pytest.approx(0.02)

    assert out["by_od1"]["2"]["esr_mean"] == pytest.approx(0.01667, abs=1e-3)
    assert out["by_od1"]["2"]["n"] == 2
    assert out["by_od1"]["4"]["esr_mean"] == pytest.approx(0.045)

    assert out["by_od2"]["4"]["esr_mean"] == pytest.approx(0.02917, abs=1e-3)
    assert out["by_od2"]["4"]["n"] == 2

    assert out["mean_esr"] == pytest.approx((0.0133 + 0.045 + 0.02) / 3, abs=1e-3)
