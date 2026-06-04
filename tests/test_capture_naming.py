import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import capture_naming as cn


def test_resolve_returns_override_with_placeholder():
    pat = "MP-1 3TM NAM Amp DI {cap_num}.wav"
    assert cn._resolve_capture_pattern("data/x", override=pat) == pat


def test_resolve_rejects_override_without_placeholder():
    with pytest.raises(ValueError, match="cap_num"):
        cn._resolve_capture_pattern("data/x", override="no placeholder.wav")


def test_detect_prefix_pattern_on_disk(tmp_path):
    (tmp_path / "1 ADA MP-1 NAM.wav").write_bytes(b"")
    assert cn._detect_capture_pattern(tmp_path) == "{cap_num} ADA MP-1 NAM.wav"


def test_detect_glob_fallback_for_arbitrary_name(tmp_path):
    (tmp_path / "1 My Amp Capture.wav").write_bytes(b"")
    assert cn._detect_capture_pattern(tmp_path) == "{cap_num} My Amp Capture.wav"


def test_detect_raises_when_no_capture_wav(tmp_path):
    with pytest.raises(FileNotFoundError):
        cn._detect_capture_pattern(tmp_path)


def test_detect_trailing_number_pattern(tmp_path):
    for n in (1, 2, 10):
        (tmp_path / f"MP-1 3TM NAM Amp DI {n}.wav").write_bytes(b"")
    assert (
        cn._detect_capture_pattern(tmp_path)
        == "MP-1 3TM NAM Amp DI {cap_num}.wav"
    )


def test_detect_prefers_varying_index_over_constant_digit(tmp_path):
    # The '1' in 'MP-1' is constant across files; the trailing index varies and
    # must win so the constant digit isn't mistaken for the capture slot.
    for n in (1, 2, 3):
        (tmp_path / f"MP-1 Amp {n}.wav").write_bytes(b"")
    assert cn._detect_capture_pattern(tmp_path) == "MP-1 Amp {cap_num}.wav"


def test_detect_ignores_input_wav(tmp_path):
    (tmp_path / "input.wav").write_bytes(b"")
    for n in (1, 2):
        (tmp_path / f"My Amp {n}.wav").write_bytes(b"")
    assert cn._detect_capture_pattern(tmp_path) == "My Amp {cap_num}.wav"
