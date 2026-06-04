# File: trainer_app/progress_view.py
# Purpose: Render training progress (a text progress bar) and a loss/ESR curve
#          for the trainer UI. Kept free of gradio so it can be unit-tested.

import matplotlib

matplotlib.use("Agg")  # headless, thread-safe (server context)
from matplotlib.figure import Figure


def progress_bar_md(epoch, total, eta_h=None, eta_m=None, step=None, total_steps=None):
    """Markdown training progress.

    The label shows the overall epoch position (1-indexed) and ETA. The bar fills
    over the CURRENT epoch (via step/total_steps) so it visibly advances during a
    long epoch instead of only jumping once per epoch. Without step info (an
    epoch-boundary update) the bar shows the epoch as complete.
    """
    shown = epoch + 1
    overall = min(1.0, shown / total) if total else 0.0
    if step is not None and total_steps:
        frac = min(1.0, step / total_steps)
        tail = f"  (step {step}/{total_steps})"
    else:
        frac = 1.0
        tail = ""
    filled = int(round(frac * 20))
    bar = "█" * filled + "░" * (20 - filled)
    eta = f" · ETA {eta_h}h{eta_m:02d}m" if eta_h is not None and eta_m is not None else ""
    return (
        f"**Epoch {shown}/{total}** ({overall * 100:.0f}% overall){eta}\n\n"
        f"`{bar}` {frac * 100:.0f}% of epoch{tail}"
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
