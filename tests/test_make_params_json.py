# File: test_make_params_json.py
# Purpose: Tests for make_params_json.py — CSV -> params.json generator for
#          parametric NAM training.

import json
import sys
from pathlib import Path

import pytest

# Add the parent directory to the path so we can import the root-level script.
sys.path.insert(0, str(Path(__file__).parent.parent))

import make_params_json as mpj


class TestBuildConfig:
    def test_derives_ranges_and_preserves_integer_values(self):
        header = ["capture", "OD1", "OD2"]
        rows = [
            ["1", "2", "2"],
            ["2", "4", "2"],
            ["3", "10", "8"],
        ]

        config = mpj.build_config(header, rows)

        assert config["params"] == [
            {"name": "OD1", "minimum": 2.0, "maximum": 10.0},
            {"name": "OD2", "minimum": 2.0, "maximum": 8.0},
        ]
        assert config["captures"] == [
            {"capture": 1, "values": [2, 2]},
            {"capture": 2, "values": [4, 2]},
            {"capture": 3, "values": [10, 8]},
        ]

    def test_range_override_replaces_derived_range(self):
        header = ["capture", "OD1", "OD2"]
        rows = [["1", "2", "2"], ["2", "8", "6"]]

        config = mpj.build_config(header, rows, range_overrides={"OD1": (0.0, 10.0)})

        od1 = next(p for p in config["params"] if p["name"] == "OD1")
        od2 = next(p for p in config["params"] if p["name"] == "OD2")
        assert od1 == {"name": "OD1", "minimum": 0.0, "maximum": 10.0}
        # OD2 still auto-derived
        assert od2 == {"name": "OD2", "minimum": 2.0, "maximum": 6.0}

    def test_preserves_float_values(self):
        header = ["capture", "Gain"]
        rows = [["1", "0.5"], ["2", "1.0"]]

        config = mpj.build_config(header, rows)

        assert config["captures"][0]["values"] == [0.5]
        assert isinstance(config["captures"][0]["values"][0], float)

    def test_rejects_ragged_row(self):
        header = ["capture", "OD1", "OD2"]
        rows = [["1", "2", "2"], ["2", "4"]]

        with pytest.raises(ValueError, match="capture 2"):
            mpj.build_config(header, rows)

    def test_rejects_duplicate_capture_numbers(self):
        header = ["capture", "OD1"]
        rows = [["1", "2"], ["1", "4"]]

        with pytest.raises(ValueError, match="[Dd]uplicate capture"):
            mpj.build_config(header, rows)

    def test_rejects_non_numeric_value(self):
        header = ["capture", "OD1"]
        rows = [["1", "loud"]]

        with pytest.raises(ValueError):
            mpj.build_config(header, rows)


class TestFindDuplicateValueGroups:
    def test_flags_captures_with_identical_values(self):
        config = {
            "params": [{"name": "OD1", "minimum": 2.0, "maximum": 4.0}],
            "captures": [
                {"capture": 1, "values": [2]},
                {"capture": 2, "values": [4]},
                {"capture": 6, "values": [2]},
            ],
        }

        groups = mpj.find_duplicate_value_groups(config)

        assert groups == [[1, 6]]

    def test_no_duplicates_returns_empty(self):
        config = {
            "params": [{"name": "OD1", "minimum": 2.0, "maximum": 4.0}],
            "captures": [
                {"capture": 1, "values": [2]},
                {"capture": 2, "values": [4]},
            ],
        }

        assert mpj.find_duplicate_value_groups(config) == []


class TestWriteParamsJson:
    def test_writes_valid_json(self, tmp_path):
        config = {"params": [], "captures": []}
        out = tmp_path / "params.json"

        mpj.write_params_json(config, out)

        assert json.loads(out.read_text()) == config

    def test_refuses_overwrite_without_force(self, tmp_path):
        out = tmp_path / "params.json"
        out.write_text("existing")

        with pytest.raises(FileExistsError):
            mpj.write_params_json({"params": [], "captures": []}, out)

        assert out.read_text() == "existing"

    def test_overwrites_with_force(self, tmp_path):
        out = tmp_path / "params.json"
        out.write_text("existing")

        mpj.write_params_json({"params": [], "captures": []}, out, force=True)

        assert json.loads(out.read_text()) == {"params": [], "captures": []}
