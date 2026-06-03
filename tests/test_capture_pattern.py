# File: test_capture_pattern.py
# Purpose: Tests for capture WAV filename pattern resolution in
#          train_parametric.py (the --capture-pattern override).

import sys
from pathlib import Path

import pytest

# Add the parent directory to the path so we can import the root-level script.
sys.path.insert(0, str(Path(__file__).parent.parent))

import train_parametric as tp


class TestResolveCapturePattern:
    def test_override_with_placeholder_is_returned_verbatim(self):
        pattern = "MP-1 3TM NAM Amp DI {cap_num}.wav"
        assert tp._resolve_capture_pattern("data/anything", override=pattern) == pattern

    def test_override_missing_placeholder_raises(self):
        with pytest.raises(ValueError, match="cap_num"):
            tp._resolve_capture_pattern("data/anything", override="MP-1 3TM.wav")

    def test_no_override_falls_back_to_autodetect(self, tmp_path):
        # A prefix-numbered file the existing detector recognizes.
        (tmp_path / "1 ADA MP-1 NAM.wav").write_bytes(b"")
        assert (
            tp._resolve_capture_pattern(tmp_path)
            == "{cap_num} ADA MP-1 NAM.wav"
        )
