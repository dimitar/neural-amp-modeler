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
