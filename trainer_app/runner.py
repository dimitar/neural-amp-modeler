# File: trainer_app/runner.py
# Purpose: Build train/export CLI commands, run training as a subprocess with a
#          graceful stop, and parse the trainer's stdout for progress.

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_train_command(
    data_dir, output_dir, *, model_size="small", delay=10, epochs=250,
    lr=4e-3, train_stop_seconds=-9.0, val_start_seconds=-9.0,
    capture_pattern=None, resume_ckpt=None, stop_file=None, python_exe=None,
    script="train_parametric.py",
):
    """Return the argv for a `train_parametric.py train` invocation.

    resume_ckpt, when set, passes --resume so Lightning restores the full run
    state (weights, optimizer, epoch, LR schedule) and continues up to --epochs.
    stop_file, when set, passes --stop-file so the run stops gracefully (and
    saves) when that sentinel file appears — how the UI's Stop button works.
    """
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
    if resume_ckpt:
        cmd += ["--resume", str(resume_ckpt)]
    if stop_file:
        cmd += ["--stop-file", str(stop_file)]
    return cmd


def list_checkpoints(output_dir):
    """Return paths of `<output_dir>/checkpoints/*.ckpt`, newest first.

    Training writes Lightning checkpoints here (see train_parametric.py's
    ModelCheckpoint); the Train view lists them so a run can be resumed.
    """
    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    ckpts = sorted(ckpt_dir.glob("*.ckpt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in ckpts]


def build_export_command(checkpoint, output, *, python_exe=None,
                         sample_rate=None, script="train_parametric.py"):
    """Return the argv for a `train_parametric.py export` invocation.

    sample_rate is passed through as --sample-rate (used only as a fallback for
    legacy .pt files that don't embed it; an embedded rate takes precedence).
    """
    cmd = [
        python_exe or sys.executable, script, "export",
        "--checkpoint", str(checkpoint),
        "--output", str(output),
    ]
    if sample_rate:
        cmd += ["--sample-rate", str(int(sample_rate))]
    return cmd


_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)\s+\((\d+)s, ETA (\d+)h(\d+)m\)")
_METRIC_RE = re.compile(r"\|\s+(val_loss|val_ESR|train_ESR)\s*:\s*([\d.]+|--+)")
_STEP_RE = re.compile(r"~step (\d+)/(\d+) epoch (\d+)/(\d+)")


def parse_progress_line(line):
    """Parse one stdout line into a progress dict, or None if not progress."""
    m = _STEP_RE.search(line)
    if m:
        return {
            "type": "step", "step": int(m.group(1)),
            "total_steps": int(m.group(2)), "epoch": int(m.group(3)),
            "max_epochs": int(m.group(4)),
        }
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

    def __init__(self, cmd, cwd=None, stop_file=None):
        self.cmd = cmd
        self.cwd = str(cwd or REPO_ROOT)
        self.stop_file = Path(stop_file) if stop_file else None
        self._proc = None

    def start(self):
        # Clear any stale stop sentinel so this run doesn't stop immediately.
        if self.stop_file:
            try:
                self.stop_file.unlink()
            except FileNotFoundError:
                pass
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        # Force the child to flush stdout line-by-line (PYTHONUNBUFFERED) and emit
        # UTF-8. Decode with errors="replace" so a partial/garbled byte — e.g. left
        # by an abruptly-killed child — can't raise UnicodeDecodeError and crash
        # the streaming reader (which would surface as a red error box in the UI).
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self._proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            bufsize=1, env=env, **kwargs,
        )

    def stream(self):
        """Yield stdout lines (without trailing newline) until the process ends.

        Note: the caller is expected to consume the stream to completion. If the
        caller breaks early, undrained stdout could (with a very verbose child)
        fill the OS pipe buffer; the current trainer's output volume is far below
        that limit.
        """
        if self._proc is None:
            return
        for line in self._proc.stdout:
            yield line.rstrip("\n")

    def stop(self):
        """Request a graceful stop by writing the sentinel file that
        train_parametric.py polls (it then finishes the step and saves).

        We deliberately do NOT send an OS signal: a Windows CTRL_BREAK kills the
        trainer abruptly (no save) and can disrupt the piped log stream, while on
        any platform a KeyboardInterrupt makes Lightning call sys.exit(1) and skip
        the save. The sentinel file avoids all of that.
        """
        if not self._proc or self._proc.poll() is not None:
            return
        if self.stop_file:
            try:
                self.stop_file.write_text("stop")
            except OSError:
                pass

    def wait(self, timeout=None):
        if self._proc is None:
            return None
        return self._proc.wait(timeout=timeout)

    @property
    def returncode(self):
        return self._proc.returncode if self._proc else None
