import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import dataset as ds


def test_captures_from_pattern_suffix():
    names = [
        "MP-1 3TM NAM Amp DI 1.wav",
        "MP-1 3TM NAM Amp DI 10.wav",
        "input.wav",
        "notes.txt",
    ]
    pat = "MP-1 3TM NAM Amp DI {cap_num}.wav"
    assert ds.captures_from_pattern(names, pat) == [1, 10]


def test_scan_prefix_dataset_ready(tmp_path):
    (tmp_path / "input.wav").write_bytes(b"")
    (tmp_path / "1 ADA MP-1 NAM.wav").write_bytes(b"")
    (tmp_path / "2 ADA MP-1 NAM.wav").write_bytes(b"")

    status = ds.scan_dataset(tmp_path)

    assert status.pattern == "{cap_num} ADA MP-1 NAM.wav"
    assert status.capture_numbers == [1, 2]
    assert status.input_wav_present is True
    assert status.ready is True


def test_scan_suffix_needs_override(tmp_path):
    (tmp_path / "input.wav").write_bytes(b"")
    for n in (1, 2):
        (tmp_path / f"MP-1 3TM NAM Amp DI {n}.wav").write_bytes(b"")

    auto = ds.scan_dataset(tmp_path)
    assert auto.ready is False           # auto-detect can't handle suffix
    assert auto.errors

    override = ds.scan_dataset(tmp_path, capture_pattern="MP-1 3TM NAM Amp DI {cap_num}.wav")
    assert override.capture_numbers == [1, 2]
    assert override.ready is True


def test_scan_missing_input_wav(tmp_path):
    (tmp_path / "1 ADA MP-1 NAM.wav").write_bytes(b"")
    status = ds.scan_dataset(tmp_path)
    assert status.input_wav_present is False
    assert status.ready is False


def test_place_input_wav_refuses_overwrite(tmp_path):
    src = tmp_path / "di.wav"
    src.write_bytes(b"DI")
    dest = tmp_path / "input.wav"
    dest.write_bytes(b"OLD")

    with pytest.raises(FileExistsError):
        ds.place_input_wav(tmp_path, src)
    assert dest.read_bytes() == b"OLD"

    ds.place_input_wav(tmp_path, src, force=True)
    assert dest.read_bytes() == b"DI"
