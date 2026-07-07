# File: trainer_app/audio_probe.py
# Purpose: Read WAV header sample rates from a capture folder and report the rate
#          the parametric model will train at.
#
# train_parametric.load_data trains at the *capture* sample rate (target_rate)
# and resamples input.wav to match it. This module mirrors that so the Train
# view can tell the user, before starting, what rate the model will train at and
# whether input.wav will be resampled or the captures disagree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import soundfile as _sf

from capture_naming import _resolve_capture_pattern

from trainer_app.dataset import captures_from_pattern


@dataclass
class SampleRateInfo:
    training_rate: int | None = None  # capture rate — what the model trains at
    input_rate: int | None = None
    capture_rates: list[int] = field(default_factory=list)  # distinct, sorted
    errors: list[str] = field(default_factory=list)

    @property
    def input_will_resample(self):
        return (
            self.input_rate is not None
            and self.training_rate is not None
            and self.input_rate != self.training_rate
        )

    @property
    def mixed_capture_rates(self):
        return len(self.capture_rates) > 1


def _rate_of(path, info):
    try:
        return _sf.info(str(path)).samplerate
    except Exception as e:  # noqa: BLE001
        info.errors.append(f"Could not read {path.name}: {e}")
        return None


def probe_sample_rates(data_dir, capture_pattern=None):
    """Read WAV header rates and report the model's training sample rate."""
    d = Path(data_dir)
    info = SampleRateInfo()
    if not d.is_dir():
        info.errors.append(f"Not a folder: {d}")
        return info

    if (d / "input.wav").exists():
        info.input_rate = _rate_of(d / "input.wav", info)

    try:
        pattern = _resolve_capture_pattern(d, capture_pattern)
    except (FileNotFoundError, ValueError) as e:
        info.errors.append(str(e))
        return info

    names = [p.name for p in d.glob("*.wav") if p.name != "input.wav"]
    rates = []
    for num in captures_from_pattern(names, pattern):
        rate = _rate_of(d / pattern.format(cap_num=num), info)
        if rate is not None:
            rates.append(rate)
    if rates:
        info.training_rate = rates[0]  # first capture is the training target
        info.capture_rates = sorted(set(rates))
    return info
