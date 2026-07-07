# Auto-detect files on capture-folder selection

Date: 2026-07-07

## Goal

Selecting a capture folder in the trainer UI should immediately detect and
report the captures, `input.wav`, and the params CSV — and auto-load that CSV
into the Parameters grid. This collapses today's manual sequence (browse →
**Scan folder** → **Set as input.wav** → switch tab → **Load CSV**) into a
single folder pick.

## Non-goals (YAGNI)

- Guessing a DI/input file when `input.wav` is absent — keep flagging it missing.
- Auto-loading an existing `params.json`.
- Scanning on every keystroke of the folder textbox.

## Backend — `trainer_app/dataset.py`

Detection stays in the backend module so it remains testable and torch-free.

- Add `csv_files: list[str]` (sorted filenames) to `DatasetStatus`.
- `scan_dataset` globs `*.csv` in the folder and populates `csv_files`.
- A missing CSV is **not** an error (params may be entered by hand). It is
  merely reported.
- `pattern`, `capture_numbers`, `input_wav_present`, and `ready` are unchanged.
  `input.wav` missing stays a flagged error.

## UI — `trainer_app/app.py`

- **Auto-scan trigger:** `do_scan` fires on the Browse button (chained after the
  folder pick) and on the folder textbox's `.submit`/`.blur` (Enter / focus-out),
  so typing a path does not scan per keystroke. The existing button remains,
  relabeled **Rescan**, for when files change on disk after selection.
- **Status line** gains a **Params CSV:** row: the detected filename, `✗ none`,
  or a count when several are present.
- **CSV auto-load across tabs:** `do_scan` also outputs to a new **Params CSV**
  `gr.Dropdown` on the Parameters tab.
  - Exactly one CSV → dropdown value set to it. Setting the value fires the
    dropdown's `.change`, which loads it into `knobs` + `grid`.
  - Multiple CSVs → choices populated, value empty; the user picks one → the
    same `.change` loads it.
  - Zero CSVs → dropdown empty; the existing `csv_file` upload remains the
    manual fallback.

All CSV loading (auto and manual dropdown) routes through one `.change` handler.

### Data flow

```
folder → do_scan → (state_pattern, status_md, csv_dropdown)
csv_dropdown.change → do_load_csv → (knobs, grid)
```

## Testing

Extend `tests/test_trainer_app_dataset.py` for CSV detection: none, exactly one,
and several. UI event wiring in `app.py` is Gradio glue over the tested backend,
so no new UI test harness is added.
