# File: trainer_app/params_editor.py
# Purpose: Bridge the UI's knob/grid/CSV inputs to make_params_json's config.

from pathlib import Path

import make_params_json as mpj


def csv_to_grid(csv_path):
    """Return (knob_names, rows) where rows retain the leading capture cell."""
    header, rows = mpj.read_csv(csv_path)
    return header[1:], rows


def grid_to_config(knob_specs, rows):
    """Build a params.json config dict from knob specs and grid rows.

    :param knob_specs: list of {"name", optional "minimum", "maximum"} dicts,
        in column order.
    :param rows: list of [capture, value1, value2, ...] cells (str or numeric).
    Column count must match knob_specs; mismatches are raised by the delegate.
    """
    header = ["capture"] + [k["name"] for k in knob_specs]
    overrides = {
        k["name"]: (k["minimum"], k["maximum"])
        for k in knob_specs
        if k.get("minimum") is not None and k.get("maximum") is not None
    }
    str_rows = [[str(cell) for cell in row] for row in rows]
    return mpj.build_config(header, str_rows, overrides)


def generate_params_json(data_dir, knob_specs, rows, force=False):
    """Write <data_dir>/params.json; return (path, duplicate_value_groups)."""
    config = grid_to_config(knob_specs, rows)
    warnings = mpj.find_duplicate_value_groups(config)
    path = Path(data_dir) / "params.json"
    try:
        mpj.write_params_json(config, path, force=force)
    except FileExistsError:
        raise FileExistsError(f"{path} already exists. Enable overwrite to replace it.")
    return path, warnings
