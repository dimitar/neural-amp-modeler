# File: capture_naming.py
# Purpose: Torch-free detection/resolution of capture WAV filename patterns,
#          shared by train_parametric.py and the trainer UI backend.

import re
from pathlib import Path


def _template_to_regex(template):
    """Compile a '{cap_num}' template into a full-match regex with an int slot."""
    placeholder = "\x00"
    tmp = template.replace("{cap_num}", placeholder)
    return re.compile("^" + re.escape(tmp).replace(placeholder, r"\d+") + "$")


def _detect_capture_pattern(data_dir):
    """Auto-detect the capture WAV filename template from the directory.

    Works regardless of where the capture number sits in the name (leading,
    trailing, or in the middle). For each numeric run in each candidate
    filename we form a template with that run replaced by '{cap_num}', then keep
    the template that matches the most files. A constant digit in the name
    (e.g. the '1' in 'MP-1') matches only its own file and so loses to the real,
    varying capture index.
    """
    data_dir = Path(data_dir)
    names = sorted(p.name for p in data_dir.glob("*.wav") if p.name != "input.wav")
    if not names:
        raise FileNotFoundError(
            f"No capture WAV files found in {data_dir}. Expected numbered files "
            "alongside input.wav, e.g. '1 My Amp.wav' or 'My Amp 1.wav'."
        )

    best_count = 0
    best_template = None
    tried = set()
    for name in names:
        for m in re.finditer(r"\d+", name):
            template = name[: m.start()] + "{cap_num}" + name[m.end():]
            if template in tried:
                continue
            tried.add(template)
            rx = _template_to_regex(template)
            count = sum(1 for n in names if rx.match(n))
            if count > best_count:
                best_count = count
                best_template = template

    if best_template is None:
        raise FileNotFoundError(
            f"Could not detect a capture filename pattern in {data_dir}. "
            "Pass an explicit pattern with a '{cap_num}' placeholder, e.g. "
            "'MP-1 3TM NAM Amp DI {cap_num}.wav'."
        )
    return best_template


def _resolve_capture_pattern(data_dir, override=None):
    """Resolve the capture WAV filename template.

    :param override: Explicit template from --capture-pattern. Used verbatim
        when given; must contain the '{cap_num}' placeholder so each capture
        number can be substituted (it may sit anywhere in the name, e.g. a
        trailing index like 'MP-1 3TM NAM Amp DI {cap_num}.wav'). When None,
        falls back to auto-detection.
    """
    if override is not None:
        if "{cap_num}" not in override:
            raise ValueError(
                "--capture-pattern must contain the '{cap_num}' placeholder, "
                "e.g. 'MP-1 3TM NAM Amp DI {cap_num}.wav'"
            )
        return override
    return _detect_capture_pattern(data_dir)
