# File: trainer_app/runner.py
# Purpose: Build train/export CLI commands, run training as a subprocess with a
#          graceful stop, and parse the trainer's stdout for progress.

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_train_command(
    data_dir, output_dir, *, model_size="small", delay=10, epochs=250,
    lr=4e-3, train_stop_seconds=-9.0, val_start_seconds=-9.0,
    capture_pattern=None, python_exe=None, script="train_parametric.py",
):
    """Return the argv for a `train_parametric.py train` invocation."""
    cmd = [
        python_exe or sys.executable, script, "train",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--model-size", model_size,
        "--delay", str(delay),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--train-stop-seconds", str(train_stop_seconds),
        "--val-start-seconds", str(val_start_seconds),
    ]
    if capture_pattern:
        cmd += ["--capture-pattern", capture_pattern]
    return cmd


def build_export_command(checkpoint, output, *, python_exe=None,
                         script="train_parametric.py"):
    """Return the argv for a `train_parametric.py export` invocation."""
    return [
        python_exe or sys.executable, script, "export",
        "--checkpoint", str(checkpoint),
        "--output", str(output),
    ]


_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)\s+\((\d+)s, ETA (\d+)h(\d+)m\)")
_METRIC_RE = re.compile(r"\|\s+(val_loss|val_ESR|train_ESR)\s*:\s*([\d.]+|-+)")


def parse_progress_line(line):
    """Parse one stdout line into a progress dict, or None if not progress."""
    m = _EPOCH_RE.search(line)
    if m:
        return {
            "type": "epoch", "epoch": int(m.group(1)),
            "max_epochs": int(m.group(2)), "elapsed_s": int(m.group(3)),
            "eta_h": int(m.group(4)), "eta_m": int(m.group(5)),
        }
    m = _METRIC_RE.search(line)
    if m:
        raw = m.group(2)
        value = None if set(raw) == {"-"} else float(raw)
        return {"type": "metric", "name": m.group(1), "value": value}
    return None


class TrainProcess:
    """Manage the training subprocess with a graceful (model-saving) stop."""

    def __init__(self, cmd, cwd=None):
        self.cmd = cmd
        self.cwd = str(cwd or REPO_ROOT)
        self._proc = None

    def start(self):
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self._proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, **kwargs,
        )

    def stream(self):
        """Yield stdout lines (without trailing newline) until the process ends."""
        for line in self._proc.stdout:
            yield line.rstrip("\n")

    def stop(self):
        """Send a graceful interrupt so train_parametric.py saves on shutdown."""
        if self._proc and self._proc.poll() is None:
            if os.name == "nt":
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._proc.send_signal(signal.SIGINT)

    def wait(self, timeout=None):
        return self._proc.wait(timeout=timeout)

    @property
    def returncode(self):
        return self._proc.returncode if self._proc else None
