# File: trainer_app/app.py
# Purpose: Gradio Blocks UI wiring the trainer backend (dataset, params, runner).

import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import gradio as gr

from trainer_app import audio_probe as ap
from trainer_app import dataset as ds
from trainer_app import native_dialogs as nd
from trainer_app import params_editor as pe
from trainer_app import runner as rn
from trainer_app.progress_view import loss_figure, progress_bar_md

DEFAULTS = dict(model_size="small", delay=10, epochs=250, lr=4e-3,
                train_stop_seconds=-9.0, val_start_seconds=-9.0)
DEFAULTS_OUT = "parametric_output"
GRID_HEADERS = ["file", "OD1", "OD2"]  # default params-grid columns


def _ckpt_choices(output_dir):
    """(label, value) pairs for the resume dropdown: filename → full path."""
    return [(Path(p).name, p) for p in rn.list_checkpoints(output_dir)]


def _derive_nam_name(capture_folder):
    """Auto-export filename: the capture folder name, spaces → underscores."""
    return Path(capture_folder).name.replace(" ", "_") + ".nam"


def _csv_summary(csv_files):
    if not csv_files:
        return "✗ none"
    if len(csv_files) == 1:
        return f"`{csv_files[0]}`"
    return f"{len(csv_files)} found — pick one on the Parameters tab"


def _khz(rate):
    return f"{rate / 1000:g} kHz"


def _rate_md(info):
    if info.training_rate is None:
        return ("**Training sample rate:** — "
                "(select a capture folder with input.wav + captures to detect)")
    lines = [f"**Training sample rate:** {_khz(info.training_rate)} "
             f"({info.training_rate} Hz — the capture rate)"]
    if info.input_will_resample:
        lines.append(f"- ℹ️ input.wav is {_khz(info.input_rate)} and will be "
                     f"resampled to {_khz(info.training_rate)} during training.")
    if info.mixed_capture_rates:
        rates = ", ".join(_khz(r) for r in info.capture_rates)
        lines.append(f"- ⚠️ Captures have mixed sample rates ({rates}). They "
                     "should all match — re-export them at a single rate.")
    return "\n\n".join(lines)


def _status_md(status):
    lines = [f"**Pattern:** `{status.pattern or '—'}`",
             f"**input.wav:** {'✓' if status.input_wav_present else '✗ missing'}",
             f"**Captures matched:** {len(status.capture_numbers)}",
             f"**Params CSV:** {_csv_summary(status.csv_files)}"]
    for e in status.errors:
        lines.append(f"- ⚠️ {e}")
    return "\n\n".join(lines)


def _dataset_summary_md(status):
    return (f"**✓ Dataset ready** — `{status.data_dir}`\n\n"
            f"{len(status.capture_numbers)} captures · input.wav ✓ · "
            f"pattern `{status.pattern}` · CSV {_csv_summary(status.csv_files)}")


def _params_summary_md(path, n_captures, knob_names, warns):
    lines = [f"**✓ Parameters set** — wrote `{path}`\n\n"
             f"{n_captures} captures · knobs: {', '.join(knob_names)}"]
    for g in warns:
        lines.append(f"⚠️ Captures {g} share identical knob values.")
    return "\n\n".join(lines)


def _show_edit():
    """Visibility updates to reveal an edit group and hide its locked summary."""
    return gr.update(visible=True), gr.update(visible=False)


def build_app():
    with gr.Blocks(title="NAM Parametric Trainer") as app:
        gr.Markdown("# NAM Parametric Trainer")
        state_pattern = gr.State(None)
        proc_box = gr.State({})  # holds {"proc": TrainProcess}
        out_auto = gr.State(DEFAULTS_OUT)  # last auto-filled output folder

        # --- 1. Dataset & Parameters ---
        with gr.Tab("1. Dataset & Parameters"):
            gr.Markdown("## Dataset")
            # Locked summary (shown once the dataset scans ready).
            with gr.Group(visible=False) as dataset_locked:
                dataset_summary = gr.Markdown()
                dataset_edit_btn = gr.Button("✏️ Edit dataset")
            # Editable controls (hidden while locked).
            with gr.Group(visible=True) as dataset_edit:
                with gr.Row():
                    folder = gr.Textbox(label="Capture folder", scale=4)
                    folder_browse = gr.Button("📁 Browse…", scale=1)
                scan_btn = gr.Button("Rescan")
                status_md = gr.Markdown()
                di_file = gr.File(label="DI / input signal (.wav)")
                set_input_btn = gr.Button("Set as input.wav")
                input_msg = gr.Markdown()
                # Captures are auto-detected (every .wav except input.wav); the
                # override is only for odd naming schemes auto-detect can't infer.
                with gr.Accordion("Advanced", open=False):
                    pattern_override = gr.Textbox(
                        label="Capture filename pattern (override auto-detect)",
                        placeholder="e.g. MP-1 3TM NAM Amp DI {cap_num}.wav",
                    )

            def do_set_input(folder, di):
                if not folder or not di:
                    return "Pick a folder and a DI file first."
                try:
                    ds.place_input_wav(folder, di, force=True)
                    return "✓ input.wav set."
                except Exception as e:  # noqa: BLE001
                    return f"⚠️ {e}"

            gr.Markdown("---\n## Parameters")
            # Status stays outside the collapsing groups so it shows in both states.
            params_status = gr.Markdown()
            # Locked summary (shown once params.json is written).
            with gr.Group(visible=False) as params_locked:
                params_summary = gr.Markdown()
                params_edit_btn = gr.Button("✏️ Edit parameters")
            # Editable controls (hidden while locked). Columns are fixed and the
            # file column is read-only (static_columns); search/filter is off.
            with gr.Group(visible=True) as params_edit:
                knobs = gr.Dataframe(
                    headers=["name", "minimum", "maximum"],
                    datatype=["str", "number", "number"],
                    type="array",  # hand handlers list-of-lists, not a pandas frame
                    row_count=2, column_count=(3, "fixed"), show_search="none",
                    label="Knobs (leave min/max blank to auto-derive)",
                    value=[["OD1", None, None], ["OD2", None, None]],
                )
                grid = gr.Dataframe(
                    headers=GRID_HEADERS, type="array", row_count=None,
                    static_columns=[0], show_search="none",
                    label="Capture file → knob values (params.json auto-updates)")
                csv_dropdown = gr.Dropdown(
                    label="Params CSV (auto-detected in the capture folder)",
                    choices=[], interactive=True)
                csv_file = gr.File(label="…or upload a CSV")
                load_csv_btn = gr.Button("Load uploaded CSV into grid")

            def _grid_from_csv(csv_path, pattern):
                # CSV keys rows by capture number; show the resolved filename so
                # the capture→knob mapping can be eyeballed for errors.
                names, rows = pe.csv_to_grid(csv_path)
                if pattern:
                    rows = [[ds.name_for_capture(pattern, r[0])] + list(r[1:])
                            for r in rows]
                knob_rows = [[n, None, None] for n in names]
                return knob_rows, gr.update(value=rows, headers=["file"] + names)

            def do_load_csv(csv, pattern):
                if not csv:
                    return gr.update(), gr.update()
                return _grid_from_csv(csv, pattern)

            load_csv_btn.click(do_load_csv, [csv_file, state_pattern], [knobs, grid])

            def do_load_csv_selected(csv_path, pattern):
                if not csv_path:
                    # New folder with no CSV (or cleared) → reset so stale capture
                    # rows from the previous folder don't linger.
                    return (gr.update(value=[["OD1", None, None], ["OD2", None, None]]),
                            gr.update(value=[], headers=GRID_HEADERS))
                return _grid_from_csv(csv_path, pattern)

            csv_dropdown.change(do_load_csv_selected,
                                [csv_dropdown, state_pattern], [knobs, grid])

            def _num_or_none(x):
                if x is None:
                    return None
                s = str(x).strip()
                if s == "" or s.lower() == "nan":
                    return None
                return float(s)

            def _is_blank(x):
                return x is None or str(x).strip() == "" or str(x).strip().lower() == "nan"

            def _capture_num(cell, pattern):
                """Capture number from a grid's first cell (a filename or a number)."""
                s = str(cell).strip()
                if pattern:
                    n = ds.capture_for_name(pattern, s)
                    if n is not None:
                        return n
                try:
                    return int(float(s))
                except (ValueError, TypeError):
                    return None

            def do_autogen(folder, knob_tbl, grid_tbl, pattern):
                """Write params.json automatically from the grid; lock on success."""
                def editable(msg):  # incomplete/invalid → unlock (show the grid)
                    return (msg, gr.update(visible=True), gr.update(visible=False),
                            gr.update())

                if not folder:
                    return editable("Select a dataset folder first.")
                knob_specs = []
                for r in knob_tbl:
                    if not r or _is_blank(r[0]):
                        continue
                    spec = {"name": str(r[0]).strip()}
                    try:
                        mn, mx = _num_or_none(r[1]), _num_or_none(r[2])
                    except ValueError:
                        return editable(
                            f"⚠️ Knob '{spec['name']}' has a non-numeric min/max.")
                    if mn is not None and mx is not None:
                        spec["minimum"], spec["maximum"] = mn, mx
                    knob_specs.append(spec)
                if not knob_specs:
                    return editable("Add at least one knob (name) first.")
                num_rows = []
                for r in grid_tbl:
                    if not r or _is_blank(r[0]):
                        continue
                    num = _capture_num(r[0], pattern)
                    if num is None:
                        return editable(
                            f"⚠️ Couldn't read a capture number from '{r[0]}'.")
                    num_rows.append([num] + list(r[1:]))
                if not num_rows:
                    return editable("Load a CSV or add capture rows.")
                try:
                    path, warns = pe.generate_params_json(
                        folder, knob_specs, num_rows, force=True)
                except Exception as e:  # noqa: BLE001
                    return editable(f"⚠️ {e}")
                summary = _params_summary_md(
                    path, len(num_rows), [k["name"] for k in knob_specs], warns)
                note = " ⚠️ duplicate knob values — see summary." if warns else ""
                return (f"✓ params.json written ({len(num_rows)} captures).{note}",
                        gr.update(visible=False), gr.update(visible=True), summary)

            autogen_inputs = [folder, knobs, grid, state_pattern]
            autogen_outputs = [params_status, params_edit, params_locked,
                               params_summary]
            knobs.change(do_autogen, autogen_inputs, autogen_outputs)
            grid.change(do_autogen, autogen_inputs, autogen_outputs)
            params_edit_btn.click(_show_edit, None, [params_edit, params_locked])

        # --- 2. Train & Export ---
        with gr.Tab("2. Train & Export"):
            train_rate_md = gr.Markdown(
                "**Training sample rate:** — select a capture folder on the "
                "Dataset tab.")
            with gr.Accordion("Advanced", open=False):
                model_size = gr.Radio(["small", "large"], value=DEFAULTS["model_size"],
                                      label="Model size")
                delay = gr.Number(DEFAULTS["delay"], label="Delay (samples)")
                epochs = gr.Number(DEFAULTS["epochs"], label="Epochs")
                lr = gr.Number(DEFAULTS["lr"], label="Learning rate")
                train_stop = gr.Number(DEFAULTS["train_stop_seconds"],
                                       label="Train stop seconds")
                val_start = gr.Number(DEFAULTS["val_start_seconds"],
                                      label="Val start seconds")
            with gr.Row():
                out_dir = gr.Textbox(DEFAULTS_OUT, label="Output folder", scale=4)
                out_browse = gr.Button("📁 Browse…", scale=1)
            # Open the picker from the selected dataset folder, not the OS default.
            out_browse.click(lambda f: nd.pick_folder(start_dir=f) or gr.update(),
                             [folder], out_dir)
            with gr.Row():
                resume_dd = gr.Dropdown(
                    label="Resume from checkpoint (optional — blank trains from "
                          "scratch)",
                    choices=_ckpt_choices(DEFAULTS_OUT), value=None, scale=4)
                resume_refresh = gr.Button("🔄 Refresh", scale=1)
            gr.Markdown(
                "_Resuming continues the run up to **Epochs** above (raise it to "
                "train longer) and requires the same model size._")

            def do_find_ckpts(out):
                return gr.update(choices=_ckpt_choices(out))

            resume_refresh.click(do_find_ckpts, [out_dir], [resume_dd])
            out_dir.blur(do_find_ckpts, [out_dir], [resume_dd])

            start_btn = gr.Button("Start training", variant="primary")
            stop_btn = gr.Button("Stop")
            stop_msg = gr.Markdown()
            progress_md = gr.Markdown()
            metrics_plot = gr.Plot(label="Loss / ESR (log scale)")
            export_msg = gr.Markdown()
            open_btn = gr.Button("Open output folder")
            with gr.Accordion("Training log", open=False):
                log = gr.Textbox(label="Log", lines=20, max_lines=20,
                                 autoscroll=True, show_label=False)

            def do_train(folder, pattern, out, ms, dl, ep, l_r, ts, vs, resume, box):
                try:
                    stop_file = str(Path(out) / ".nam_stop_requested")
                    proc = rn.TrainProcess(rn.build_train_command(
                        folder, out, model_size=ms, delay=int(dl), epochs=int(ep),
                        lr=float(l_r), train_stop_seconds=float(ts),
                        val_start_seconds=float(vs), capture_pattern=pattern,
                        resume_ckpt=resume or None, stop_file=stop_file),
                        stop_file=stop_file)
                    proc.start()
                except Exception as e:  # noqa: BLE001
                    yield f"⚠️ Could not start training: {e}", "", None, gr.update(), box
                    return
                box["proc"] = proc
                log_lines = deque(maxlen=400)  # bound the streamed log payload
                records, cur, last_fig = [], None, None
                eta_h, eta_m = None, None
                progress = ("⏳ Preparing data and starting training… "
                            "(the first epoch can take a while at 96 kHz)")
                yield "", progress, None, gr.update(), box
                # Guard the whole stream/export so nothing can escape as an
                # (empty) red error box in the UI — errors go to the log instead.
                try:
                    for line in proc.stream():
                        log_lines.append(line)
                        ev = rn.parse_progress_line(line)
                        plot_out = gr.update()
                        if ev and ev["type"] == "step":
                            progress = progress_bar_md(
                                ev["epoch"], ev["max_epochs"], eta_h, eta_m,
                                step=ev["step"], total_steps=ev["total_steps"])
                        elif ev and ev["type"] == "epoch":
                            eta_h, eta_m = ev["eta_h"], ev["eta_m"]
                            cur = {"epoch": ev["epoch"], "val_loss": None,
                                   "val_ESR": None}
                            records.append(cur)
                            progress = progress_bar_md(
                                ev["epoch"], ev["max_epochs"], eta_h, eta_m)
                        elif (ev and ev["type"] == "metric" and ev["value"] is not None
                              and cur is not None
                              and ev["name"] in ("val_loss", "val_ESR")):
                            cur[ev["name"]] = ev["value"]
                            last_fig = loss_figure(records)
                            plot_out = last_fig
                        yield "\n".join(log_lines), progress, plot_out, gr.update(), box
                    proc.wait()
                    log_lines.append(f"[exited with code {proc.returncode}]")
                    fig_out = last_fig if last_fig is not None else gr.update()

                    # Auto-export once the run's model file exists — covers both a
                    # clean finish and a graceful Stop (both save the .pt).
                    ckpt = Path(out) / "parametric_wavenet_model.pt"
                    if not ckpt.exists():
                        yield ("\n".join(log_lines), progress, fig_out,
                               "⚠️ No model file was produced — nothing to export.",
                               box)
                        return
                    nam_path = Path(out) / _derive_nam_name(folder)
                    log_lines.append(f"Exporting {nam_path} …")
                    yield "\n".join(log_lines), progress, fig_out, gr.update(), box
                    # The .pt embeds its sample rate, so export needs no rate here.
                    r = subprocess.run(rn.build_export_command(ckpt, nam_path),
                                       cwd=str(rn.REPO_ROOT), capture_output=True,
                                       text=True)
                    if r.returncode == 0:
                        export_md = f"✓ Exported `{nam_path}`."
                        log_lines.append(export_md)
                    else:
                        export_md = (f"⚠️ Export failed:\n```\n{r.stdout}\n"
                                     f"{r.stderr}\n```")
                        log_lines.append("[export failed]")
                    yield "\n".join(log_lines), progress, fig_out, export_md, box
                except Exception as e:  # noqa: BLE001
                    log_lines.append(f"[trainer UI error: {e}]")
                    yield ("\n".join(log_lines), progress, gr.update(),
                           f"⚠️ {e}", box)

            start_btn.click(
                do_train,
                [folder, state_pattern, out_dir, model_size, delay, epochs, lr,
                 train_stop, val_start, resume_dd, proc_box],
                [log, progress_md, metrics_plot, export_msg, proc_box])

            def do_open(out):
                p = Path(out).resolve()
                try:
                    if os.name == "nt":
                        os.startfile(p)  # noqa: S606
                except OSError as e:  # noqa: BLE001
                    return f"⚠️ Could not open `{p}`: {e}"
                return f"Opened `{p}`."

            open_btn.click(do_open, [out_dir], [export_msg])

            def do_stop(box):
                proc = box.get("proc")
                if not proc:
                    return "Nothing to stop."
                try:
                    proc.stop()
                except Exception as e:  # noqa: BLE001
                    return f"⚠️ Could not signal stop: {e}"
                return "Stopping — the model will save after the current step…"

            stop_btn.click(do_stop, [proc_box], [stop_msg])

        # --- Cross-tab wiring ---
        # Registered here (not inside the Dataset tab) because do_scan outputs to
        # the Params CSV dropdown, which lives on the Parameters tab.
        def do_scan(folder, override, out, out_auto):
            if not folder or not folder.strip():
                # Empty folder → stay editable, unlocked; leave output/resume as-is.
                return (None, "", gr.update(choices=[], value=None),
                        _rate_md(ap.SampleRateInfo()),
                        gr.update(visible=True), gr.update(visible=False),
                        gr.update(), gr.update(), out_auto, gr.update())
            # A Gradio 6 Textbox is None until edited (and a collapsed-accordion
            # child can post None), so normalise before stripping.
            override = (override or "").strip() or None
            status = ds.scan_dataset(folder, override)
            # Dropdown values are full paths (label = filename) so switching to a
            # folder whose CSV has the same name still changes the value and
            # reloads the grid. One CSV → auto-select it (fires the dropdown's
            # .change, which loads the grid); several → leave the user to pick.
            csv_choices = [(name, str(Path(folder) / name))
                           for name in status.csv_files]
            dd = gr.update(choices=csv_choices,
                           value=(csv_choices[0][1] if len(csv_choices) == 1
                                  else None))
            rate = ap.probe_sample_rates(folder, override)
            # Ready → lock into a compact summary; otherwise stay editable.
            edit_vis = gr.update(visible=not status.ready)
            locked_vis = gr.update(visible=status.ready)
            summary = _dataset_summary_md(status) if status.ready else gr.update()
            # Default the output folder to the capture folder, unless the user
            # has changed it away from the last value we auto-filled.
            if Path(folder).is_dir() and (not out or out == out_auto):
                new_out = str(Path(folder))
                out_update, new_auto = gr.update(value=new_out), new_out
            else:
                new_out = out
                out_update, new_auto = gr.update(), out_auto
            # Refresh the resume checkpoints for the effective output folder; the
            # auto-fill above sets out_dir programmatically, which wouldn't fire
            # the field's own blur-based refresh.
            resume_update = gr.update(choices=_ckpt_choices(new_out))
            return (status.pattern, _status_md(status), dd, _rate_md(rate),
                    edit_vis, locked_vis, summary, out_update, new_auto,
                    resume_update)

        scan_outputs = [state_pattern, status_md, csv_dropdown, train_rate_md,
                        dataset_edit, dataset_locked, dataset_summary,
                        out_dir, out_auto, resume_dd]
        scan_inputs = [folder, pattern_override, out_dir, out_auto]
        # Auto-scan when a folder is chosen (Browse) or the path is committed
        # (Enter / focus-out); the Rescan button re-runs it after on-disk changes.
        folder_browse.click(lambda f: nd.pick_folder(start_dir=f) or gr.update(),
                            [folder], folder) \
            .then(do_scan, scan_inputs, scan_outputs)
        folder.submit(do_scan, scan_inputs, scan_outputs)
        folder.blur(do_scan, scan_inputs, scan_outputs)
        scan_btn.click(do_scan, scan_inputs, scan_outputs)
        dataset_edit_btn.click(_show_edit, None, [dataset_edit, dataset_locked])
        # Placing input.wav can complete the dataset → re-scan so it can lock.
        set_input_btn.click(do_set_input, [folder, di_file], [input_msg]) \
            .then(do_scan, scan_inputs, scan_outputs)

    return app
