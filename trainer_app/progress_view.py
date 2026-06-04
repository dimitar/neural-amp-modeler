# File: trainer_app/progress_view.py
# Purpose: Render training progress (a text progress bar) and a loss/ESR curve
#          for the trainer UI. Kept free of gradio so it can be unit-tested.

import matplotlib

matplotlib.use("Agg")  # headless, thread-safe (server context)
from matplotlib.figure import Figure


def progress_bar_md(epoch, total, eta_h, eta_m):
    """Markdown string: 'Epoch N/total · ETA HhMMm' plus a unicode bar.

    `epoch` is the trainer's 0-indexed epoch; it is displayed 1-indexed so the
    label and the bar percentage agree (Epoch 1/250 … Epoch 250/250 · 100%).
    """
    shown = epoch + 1
    frac = min(1.0, shown / total) if total else 0.0
    filled = int(round(frac * 20))
    bar = "█" * filled + "░" * (20 - filled)
    return (
        f"**Epoch {shown}/{total}** · ETA {eta_h}h{eta_m:02d}m\n\n"
        f"`{bar}` {frac * 100:.0f}%"
    )


def loss_figure(records):
    """Build a matplotlib Figure of val_loss/val_ESR vs epoch (log y).

    :param records: list of {"epoch", "val_loss", "val_ESR"} dicts; values may
        be None until that metric has arrived for the epoch.
    :return: a Figure, or None if there is nothing to plot yet.
    """
    le = [r["epoch"] for r in records if r.get("val_loss") is not None]
    lv = [r["val_loss"] for r in records if r.get("val_loss") is not None]
    ee = [r["epoch"] for r in records if r.get("val_ESR") is not None]
    ev = [r["val_ESR"] for r in records if r.get("val_ESR") is not None]
    if not lv and not ev:
        return None
    fig = Figure(figsize=(6, 3), tight_layout=True)
    ax = fig.add_subplot(111)
    if lv:
        ax.plot(le, lv, marker=".", label="val_loss")
    if ev:
        ax.plot(ee, ev, marker=".", label="val_ESR")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value (log)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig
