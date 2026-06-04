# File: trainer_app/dataset.py
# Purpose: Scan a capture folder, resolve the capture filename pattern, list
#          captures, place input.wav, and report training readiness.

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from capture_naming import _resolve_capture_pattern


def pattern_to_regex(pattern):
    """Compile a '{cap_num}' template into a full-match regex with one int group.

    Assumes the template contains exactly one '{cap_num}' placeholder.
    """
    placeholder = "\x00"
    tmp = pattern.replace("{cap_num}", placeholder)
    escaped = re.escape(tmp).replace(placeholder, r"(\d+)")
    return re.compile("^" + escaped + "$")


def captures_from_pattern(names, pattern):
    """Return the sorted, unique capture numbers among `names` matching `pattern`."""
    rx = pattern_to_regex(pattern)
    nums = [int(m.group(1)) for n in names if (m := rx.match(n))]
    return sorted(set(nums))


@dataclass
class DatasetStatus:
    data_dir: str
    pattern: str | None = None
    capture_numbers: list[int] = field(default_factory=list)
    input_wav_present: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self):
        return (
            self.input_wav_present
            and len(self.capture_numbers) > 0
            and not self.errors
        )


def scan_dataset(data_dir, capture_pattern=None):
    """Inspect a dataset folder and report pattern, captures, and readiness."""
    d = Path(data_dir)
    status = DatasetStatus(data_dir=str(d))
    if not d.is_dir():
        status.errors.append(f"Not a folder: {d}")
        return status

    status.input_wav_present = (d / "input.wav").exists()
    names = [p.name for p in d.glob("*.wav") if p.name != "input.wav"]

    try:
        status.pattern = _resolve_capture_pattern(d, capture_pattern)
    except (FileNotFoundError, ValueError) as e:
        status.errors.append(str(e))
        return status

    status.capture_numbers = captures_from_pattern(names, status.pattern)
    if not status.capture_numbers:
        status.errors.append(f"No capture files matched pattern '{status.pattern}'.")
    if not status.input_wav_present:
        status.errors.append("Missing input.wav (the shared DI signal).")
    return status


def place_input_wav(data_dir, src_path, force=False) -> Path:
    """Copy a DI file into the dataset folder as input.wav, refusing overwrite.

    :returns: the destination Path (<data_dir>/input.wav).
    """
    dest = Path(data_dir) / "input.wav"
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} already exists. Use force=True to overwrite.")
    shutil.copyfile(src_path, dest)
    return dest
