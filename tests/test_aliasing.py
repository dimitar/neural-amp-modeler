"""Unit tests for the pure SNR-A aliasing metric (no torch/model needed)."""
import numpy as _np

from tools.aliasing import snr_a_db, synth_tone

_SR = 48_000


def test_clean_tone_has_high_snr_a():
    # A pure sine has no aliasing -> SNR-A should be very high.
    y = synth_tone(2093.0, _SR)
    assert snr_a_db(y, 2093.0, _SR) > 60.0


def test_added_inharmonic_energy_lowers_snr_a():
    # Injecting energy at an inharmonic (irrational-multiple) frequency, like
    # aliasing would, must drop SNR-A below the clean tone's.
    f0 = 2093.0
    clean = synth_tone(f0, _SR)
    alias = 0.05 * synth_tone(f0 * _np.sqrt(2.0), _SR)  # not a harmonic of f0
    assert snr_a_db(clean + alias, f0, _SR) < snr_a_db(clean, f0, _SR)
