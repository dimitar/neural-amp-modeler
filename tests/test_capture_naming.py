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
