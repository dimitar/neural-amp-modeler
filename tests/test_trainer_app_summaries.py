import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import app
from trainer_app import audio_probe as ap
from trainer_app import dataset as ds


def test_rate_md_uniform():
    info = ap.SampleRateInfo(training_rate=96000, input_rate=96000,
                             capture_rates=[96000])
    md = app._rate_md(info)
    assert "96 kHz" in md
    assert "resampled" not in md
    assert "mixed" not in md


def test_rate_md_flags_resampling():
    info = ap.SampleRateInfo(training_rate=96000, input_rate=48000,
                             capture_rates=[96000])
    md = app._rate_md(info)
    assert "resampled to 96 kHz" in md
    assert "48 kHz" in md


def test_rate_md_flags_mixed_captures():
    info = ap.SampleRateInfo(training_rate=96000, input_rate=96000,
                             capture_rates=[48000, 96000])
    md = app._rate_md(info)
    assert "mixed sample rates" in md


def test_rate_md_undetected():
    assert "—" in app._rate_md(ap.SampleRateInfo())


def test_dataset_summary_lists_key_facts():
    status = ds.DatasetStatus(
        data_dir="/x/My Amp", pattern="Amp {cap_num}.wav",
        capture_numbers=[1, 2, 3], input_wav_present=True,
        csv_files=["New ADA.csv"])
    md = app._dataset_summary_md(status)
    assert "Dataset ready" in md
    assert "3 captures" in md
    assert "New ADA.csv" in md


def test_params_summary_includes_warnings():
    md = app._params_summary_md("/x/params.json", 24, ["OD1", "OD2"], [[1, 6]])
    assert "24 captures" in md
    assert "OD1, OD2" in md
    assert "[1, 6]" in md
