import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import progress_view as pv


def test_progress_bar_md_start():
    md = pv.progress_bar_md(0, 250, 0, 32)
    assert "Epoch 1/250" in md
    assert "ETA 0h32m" in md
    assert "█" in md or "░" in md


def test_progress_bar_md_complete():
    md = pv.progress_bar_md(249, 250, 0, 0)
    assert "Epoch 250/250" in md
    assert "100%" in md


def test_loss_figure_none_when_empty():
    assert pv.loss_figure([]) is None
    assert pv.loss_figure([{"epoch": 1, "val_loss": None, "val_ESR": None}]) is None


def test_loss_figure_has_two_lines():
    records = [
        {"epoch": 0, "val_loss": 0.01, "val_ESR": 0.05},
        {"epoch": 1, "val_loss": 0.005, "val_ESR": 0.02},
    ]
    fig = pv.loss_figure(records)
    assert fig is not None
    assert len(fig.axes[0].get_lines()) == 2


def test_loss_figure_one_line_when_single_metric():
    records = [
        {"epoch": 0, "val_loss": 0.01, "val_ESR": None},
        {"epoch": 1, "val_loss": 0.005, "val_ESR": None},
    ]
    fig = pv.loss_figure(records)
    assert fig is not None
    assert len(fig.axes[0].get_lines()) == 1
