"""Aliasing (SNR-A) measurement — pure numpy, no torch/model deps.

A hard-cornered nonlinearity aliases more than a smooth one; a larger head
convolution is meant to reduce that. SNR-A quantifies it: drive the model with
a pure tone f0, then compare energy at the harmonic bins (k*f0 — the signal the
amp is *supposed* to produce) against all other in-band energy (aliasing + noise).

Higher SNR-A dB = less aliasing. Compare k16 vs k1 on the same tone: a positive
delta means the bigger head kernel reduced aliasing.
"""
import numpy as _np

# High tones stress aliasing (their harmonics reach past Nyquist soonest).
DEFAULT_TONES_HZ = (1046.5, 2093.0, 3135.9, 4186.0)  # ~C6..C8
_HARMONIC_HALFWIDTH_BINS = 2  # bins each side of a harmonic counted as "signal"


def synth_tone(freq_hz, sr, seconds=2.0, amp=0.5):
    """A pure sine at freq_hz, float32."""
    n = int(sr * seconds)
    t = _np.arange(n) / float(sr)
    return (amp * _np.sin(2.0 * _np.pi * freq_hz * t)).astype(_np.float32)


def snr_a_db(output, f0_hz, sr, harmonic_halfwidth_bins=_HARMONIC_HALFWIDTH_BINS):
    """SNR-A (dB) for a single-tone response. Uses the steady-state second half."""
    x = _np.asarray(output, dtype=_np.float64)
    x = x[len(x) // 2:]          # drop attack/settling
    if len(x) < 16:
        return float("nan")
    x = x - x.mean()            # remove DC (LeakyReLU/other nonlins are not zero-symmetric)
    win = _np.hanning(len(x))
    spectrum = _np.abs(_np.fft.rfft(x * win)) ** 2
    freqs = _np.fft.rfftfreq(len(x), 1.0 / sr)
    df = freqs[1] - freqs[0]
    nyquist = sr / 2.0

    signal_mask = _np.zeros_like(spectrum, dtype=bool)
    k = 1
    while k * f0_hz < nyquist:
        center = int(round(k * f0_hz / df))
        lo = max(0, center - harmonic_halfwidth_bins)
        hi = min(len(spectrum), center + harmonic_halfwidth_bins + 1)
        signal_mask[lo:hi] = True
        k += 1
    signal_mask[0] = False       # never count the DC bin as signal

    signal = spectrum[signal_mask].sum()
    alias = spectrum[~signal_mask].sum()
    if alias <= 0.0:
        return float("inf")
    return 10.0 * _np.log10(signal / alias)
