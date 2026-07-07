import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import runner as rn


def test_build_train_command_includes_core_flags():
    cmd = rn.build_train_command(
        "data/amp", "out", model_size="small", delay=10, epochs=50,
        python_exe="py", script="train_parametric.py",
    )
    assert cmd[:3] == ["py", "train_parametric.py", "train"]
    assert "--data-dir" in cmd and "data/amp" in cmd
    assert "--output-dir" in cmd and "out" in cmd
    assert cmd[cmd.index("--epochs") + 1] == "50"
    assert "--capture-pattern" not in cmd


def test_build_train_command_adds_capture_pattern():
    cmd = rn.build_train_command(
        "d", "o", capture_pattern="X {cap_num}.wav", python_exe="py",
    )
    assert cmd[cmd.index("--capture-pattern") + 1] == "X {cap_num}.wav"


def test_build_train_command_omits_resume_by_default():
    cmd = rn.build_train_command("d", "o", python_exe="py")
    assert "--resume" not in cmd


def test_build_train_command_adds_resume():
    cmd = rn.build_train_command(
        "d", "o", resume_ckpt="out/checkpoints/best.ckpt", python_exe="py")
    assert cmd[cmd.index("--resume") + 1] == "out/checkpoints/best.ckpt"


def test_list_checkpoints_newest_first(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    old = ckpt_dir / "best-01-0.1.ckpt"
    new = ckpt_dir / "best-09-0.02.ckpt"
    old.write_bytes(b"")
    new.write_bytes(b"")
    import os
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    (tmp_path / "notes.txt").write_bytes(b"")  # ignored

    found = rn.list_checkpoints(tmp_path)

    assert found == [str(new), str(old)]


def test_list_checkpoints_missing_dir_returns_empty(tmp_path):
    assert rn.list_checkpoints(tmp_path / "nope") == []


def test_build_export_command():
    cmd = rn.build_export_command("out/model.pt", "amp.nam", python_exe="py")
    assert cmd[:3] == ["py", "train_parametric.py", "export"]
    assert cmd[cmd.index("--checkpoint") + 1] == "out/model.pt"
    assert cmd[cmd.index("--output") + 1] == "amp.nam"


def test_parse_epoch_line():
    line = "+- epoch  12/250  (8s, ETA 0h32m)"
    assert rn.parse_progress_line(line) == {
        "type": "epoch", "epoch": 12, "max_epochs": 250,
        "elapsed_s": 8, "eta_h": 0, "eta_m": 32,
    }


def test_parse_metric_line():
    assert rn.parse_progress_line("|  val_loss : 0.001234  (best 0.001 @ epoch 10)") == {
        "type": "metric", "name": "val_loss", "value": 0.001234,
    }


def test_parse_metric_line_none_placeholder():
    assert rn.parse_progress_line("|  val_ESR  : --------  (best 9.9 @ epoch 0)") == {
        "type": "metric", "name": "val_ESR", "value": None,
    }


def test_parse_non_progress_line_returns_none():
    assert rn.parse_progress_line("Capture pattern: foo") is None


def test_train_process_streams_then_stops(tmp_path):
    # A dummy long-running script that prints, then loops until the stop sentinel
    # file appears (mirroring train_parametric's _StopFileCallback).
    stop_file = tmp_path / "stop"
    script = tmp_path / "dummy.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        stop = Path(r"{stop_file}")
        print("line-1", flush=True)
        print("line-2", flush=True)
        while not stop.exists():
            time.sleep(0.05)
        print("stopped", flush=True)
        sys.exit(0)
    """))

    proc = rn.TrainProcess([sys.executable, str(script)], stop_file=str(stop_file))
    proc.start()
    lines = []
    for line in proc.stream():
        lines.append(line)
        if len(lines) >= 2:
            break
    assert lines == ["line-1", "line-2"]

    proc.stop()  # writes the sentinel file
    proc.wait(timeout=10)
    assert proc.returncode == 0  # exited cleanly via the sentinel


def test_stop_before_start_is_safe():
    proc = rn.TrainProcess(["py", "does-not-exist.py"])
    proc.stop()          # must not raise
    assert proc.returncode is None
    assert proc.wait() is None


def test_nonzero_exit_code_is_reported(tmp_path):
    script = tmp_path / "boom.py"
    script.write_text("import sys; sys.exit(3)")
    proc = rn.TrainProcess([sys.executable, str(script)])
    proc.start()
    list(proc.stream())   # drain to completion
    proc.wait(timeout=10)
    assert proc.returncode == 3
