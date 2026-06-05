# File: make_params_json.py
# Purpose: Generate a params.json for parametric NAM training from a CSV of
#          capture -> knob values.
#
# The parametric trainer (train_parametric.py) reads a params.json from each
# dataset directory describing the knob schema (name + range per knob) and the
# capture-to-knob-value mapping. This script builds that file from a CSV so the
# mapping doesn't have to be hand-edited.
#
# CSV format: first column is the capture (WAV) number; every remaining column
# is one knob, with the header row supplying knob names and column order
# defining the value order. Example:
#
#     capture,OD1,OD2
#     1,2,2
#     2,4,2
#     ...
#
# Usage:
#     python make_params_json.py captures.csv --data-dir "data/My Amp Captures"
#     python make_params_json.py captures.csv --out params.json --range OD1=0:10

import argparse
import csv
import json
import sys
from pathlib import Path


def _parse_number(s):
    """Parse a CSV cell into int (whole numbers) or float (with a decimal)."""
    s = s.strip()
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        raise ValueError(f"Non-numeric value: {s!r}")


def read_csv(path):
    """Read the CSV, returning (header, rows) of stripped string cells."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        records = [
            [cell.strip() for cell in row]
            for row in reader
            if any(cell.strip() for cell in row)  # skip blank lines
        ]
    if not records:
        raise ValueError(f"{path} is empty.")
    return records[0], records[1:]


def build_config(header, rows, range_overrides=None):
    """Build the params.json config dict from a parsed CSV.

    :param header: Header cells; header[0] is the capture column (its label is
        ignored), header[1:] are knob names in order.
    :param rows: Data rows of string cells.
    :param range_overrides: Optional {name: (minimum, maximum)} to override the
        auto-derived range for specific knobs.
    :return: dict with "params" and "captures" keys.
    """
    range_overrides = range_overrides or {}
    param_names = header[1:]
    if not param_names:
        raise ValueError("CSV must have at least one knob column after 'capture'.")

    captures = []
    seen_captures = set()
    columns = {name: [] for name in param_names}

    for row in rows:
        capture = int(row[0])
        if len(row) != len(header):
            raise ValueError(
                f"capture {capture} has {len(row) - 1} values, "
                f"expected {len(param_names)}."
            )
        if capture in seen_captures:
            raise ValueError(f"Duplicate capture number: {capture}")
        seen_captures.add(capture)

        values = [_parse_number(cell) for cell in row[1:]]
        for name, value in zip(param_names, values):
            columns[name].append(value)
        captures.append({"capture": capture, "values": values})

    params = []
    for name in param_names:
        if name in range_overrides:
            minimum, maximum = range_overrides[name]
        else:
            minimum, maximum = min(columns[name]), max(columns[name])
        params.append(
            {"name": name, "minimum": float(minimum), "maximum": float(maximum)}
        )

    return {"params": params, "captures": captures}


def find_duplicate_value_groups(config):
    """Return groups of capture numbers that share identical knob values.

    Duplicate captures map the same input to different outputs and poison
    training (see CLAUDE.md). Returns a list of capture-number lists, one per
    colliding value combination, in order of first appearance.
    """
    by_values = {}
    for entry in config["captures"]:
        key = tuple(entry["values"])
        by_values.setdefault(key, []).append(entry["capture"])
    return [sorted(group) for group in by_values.values() if len(group) > 1]


def write_params_json(config, out_path, force=False):
    """Write config to out_path as pretty JSON, refusing to overwrite."""
    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Pass --force to overwrite."
        )
    out_path.write_text(json.dumps(config, indent=2) + "\n")


def parse_range_overrides(items):
    """Parse --range NAME=MIN:MAX strings into {name: (min, max)}."""
    overrides = {}
    for item in items or []:
        try:
            name, span = item.split("=", 1)
            lo, hi = span.split(":", 1)
            overrides[name] = (float(lo), float(hi))
        except ValueError:
            raise ValueError(
                f"Invalid --range {item!r}; expected NAME=MIN:MAX (e.g. OD1=0:10)."
            )
    return overrides


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate params.json for parametric NAM training from a CSV."
    )
    parser.add_argument("csv", help="CSV of capture,knob1,knob2,... rows.")
    parser.add_argument(
        "--data-dir",
        help="Dataset directory; writes <data-dir>/params.json.",
    )
    parser.add_argument("--out", help="Explicit output path (overrides --data-dir).")
    parser.add_argument(
        "--range",
        action="append",
        metavar="NAME=MIN:MAX",
        help="Override a knob's range (repeatable), e.g. --range OD1=0:10.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing params.json."
    )
    args = parser.parse_args(argv)

    if not args.out and not args.data_dir:
        parser.error("Provide --data-dir or --out for the output location.")
    out_path = Path(args.out) if args.out else Path(args.data_dir) / "params.json"

    header, rows = read_csv(args.csv)
    config = build_config(header, rows, parse_range_overrides(args.range))

    print(f"{len(config['captures'])} captures, {len(config['params'])} knobs:")
    for p in config["params"]:
        print(f"  {p['name']}: [{p['minimum']}, {p['maximum']}]")

    duplicates = find_duplicate_value_groups(config)
    for group in duplicates:
        print(f"  WARNING: captures {group} share identical knob values.")

    write_params_json(config, out_path, force=args.force)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
