import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import audio_probe as ap


def _wav(path, rate, n=8):
    sf.write(str(path), np.zeros(n, dtype="float32"), rate)


def test_probe_uniform_rate(tmp_path):
    _wav(tmp_path / "input.wav", 96000)
    _wav(tmp_path / "1 Amp.wav", 96000)
    _wav(tmp_path / "2 Amp.wav", 96000)

    info = ap.probe_sample_rates(tmp_path)

    assert info.training_rate == 96000
    assert info.input_rate == 96000
    assert info.capture_rates == [96000]
    assert info.input_will_resample is False
    assert info.mixed_capture_rates is False
    assert info.errors == []


def test_probe_input_needs_resampling(tmp_path):
    _wav(tmp_path / "input.wav", 48000)
    _wav(tmp_path / "1 Amp.wav", 96000)

    info = ap.probe_sample_rates(tmp_path)

    # The model trains at the capture rate; input.wav is resampled to match.
    assert info.training_rate == 96000
    assert info.input_rate == 48000
    assert info.input_will_resample is True


def test_probe_mixed_capture_rates(tmp_path):
    _wav(tmp_path / "input.wav", 96000)
    _wav(tmp_path / "1 Amp.wav", 96000)
    _wav(tmp_path / "2 Amp.wav", 48000)

    info = ap.probe_sample_rates(tmp_path)

    assert info.mixed_capture_rates is True
    assert info.capture_rates == [48000, 96000]


def test_probe_missing_input_still_reports_training_rate(tmp_path):
    _wav(tmp_path / "1 Amp.wav", 96000)

    info = ap.probe_sample_rates(tmp_path)

    assert info.input_rate is None
    assert info.training_rate == 96000
    assert info.input_will_resample is False


def test_probe_nonexistent_folder_reports_error(tmp_path):
    info = ap.probe_sample_rates(tmp_path / "nope")
    assert info.training_rate is None
    assert info.errors
