import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer_app import params_editor as pe


def test_grid_to_config_auto_ranges():
    knob_specs = [{"name": "OD1"}, {"name": "OD2"}]
    rows = [["1", "2", "2"], ["2", "10", "8"]]

    config = pe.grid_to_config(knob_specs, rows)

    assert config["params"] == [
        {"name": "OD1", "minimum": 2.0, "maximum": 10.0},
        {"name": "OD2", "minimum": 2.0, "maximum": 8.0},
    ]
    assert config["captures"][0] == {"capture": 1, "values": [2, 2]}


def test_grid_to_config_range_override():
    knob_specs = [{"name": "OD1", "minimum": 0.0, "maximum": 10.0}]
    rows = [["1", "2"], ["2", "8"]]

    config = pe.grid_to_config(knob_specs, rows)

    assert config["params"][0] == {"name": "OD1", "minimum": 0.0, "maximum": 10.0}


def test_csv_to_grid(tmp_path):
    csv_path = tmp_path / "c.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capture", "OD1", "OD2"])
        w.writerow([1, 2, 2])
        w.writerow([2, 4, 2])

    knob_names, rows = pe.csv_to_grid(csv_path)

    assert knob_names == ["OD1", "OD2"]
    assert rows == [["1", "2", "2"], ["2", "4", "2"]]


def test_generate_writes_and_warns_on_duplicates(tmp_path):
    knob_specs = [{"name": "OD1"}, {"name": "OD2"}]
    rows = [["1", "2", "2"], ["6", "2", "2"], ["2", "4", "2"]]

    path, warnings = pe.generate_params_json(tmp_path, knob_specs, rows)

    assert path == tmp_path / "params.json"
    assert json.loads(path.read_text())["captures"][0]["capture"] == 1
    assert warnings == [[1, 6]]


def test_generate_refuses_overwrite(tmp_path):
    (tmp_path / "params.json").write_text("existing")
    with pytest.raises(FileExistsError):
        pe.generate_params_json(tmp_path, [{"name": "OD1"}], [["1", "2"]])


def test_generate_force_overwrites(tmp_path):
    (tmp_path / "params.json").write_text("old")
    knob_specs = [{"name": "OD1"}]
    path, warnings = pe.generate_params_json(tmp_path, knob_specs, [["1", "2"]], force=True)
    assert json.loads(path.read_text())["captures"] == [{"capture": 1, "values": [2]}]
    assert warnings == []
