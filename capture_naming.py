# File: capture_naming.py
# Purpose: Torch-free detection/resolution of capture WAV filename patterns,
#          shared by train_parametric.py and the trainer UI backend.

from pathlib import Path


def _detect_capture_pattern(data_dir):
    """Auto-detect the WAV filename pattern from the data directory."""
    data_dir = Path(data_dir)
    # Try known patterns in order
    patterns = [
        "{cap_num} ADA MP-1 NAM.wav",
        "{cap_num} ADA MP-1 PS2 96khz.wav",
    ]
    for pat in patterns:
        test_path = data_dir / pat.format(cap_num=1)
        if test_path.exists():
            return pat
    # Fallback: find any WAV starting with "1 "
    for f in data_dir.glob("1 *.wav"):
        if f.name != "input.wav":
            # Extract pattern by replacing leading "1 " with "{cap_num} "
            return "{cap_num} " + f.name[2:]
    raise FileNotFoundError(
        f"Cannot detect capture WAV pattern in {data_dir}. "
        "Expected files like '1 ADA MP-1 NAM.wav'. If your captures number the "
        "files differently (e.g. a trailing number), pass --capture-pattern."
    )


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
