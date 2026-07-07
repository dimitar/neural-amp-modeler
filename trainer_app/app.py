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

        # --- 1. Dataset ---
        with gr.Tab("1. Dataset"):
            # Locked summary (shown once the dataset scans ready).
            with gr.Group(visible=False) as dataset_locked:
                dataset_summary = gr.Markdown()
                dataset_edit_btn = gr.Button("✏️ Edit dataset")
            # Editable controls (hidden while locked).
            with gr.Group(visible=True) as dataset_edit:
                with gr.Row():
                    folder = gr.Textbox(label="Capture folder", scale=4)
                    folder_browse = gr.Button("📁 Browse…", scale=1)
                pattern_override = gr.Textbox(
                    label="Capture filename pattern (only if auto-detect fails)",
                    placeholder="e.g. MP-1 3TM NAM Amp DI {cap_num}.wav",
                )
                scan_btn = gr.Button("Rescan")
                status_md = gr.Markdown()
                di_file = gr.File(label="DI / input signal (.wav)")
                set_input_btn = gr.Button("Set as input.wav")
                input_msg = gr.Markdown()

            def do_set_input(folder, di):
                if not folder or not di:
                    return "Pick a folder and a DI file first."
                try:
                    ds.place_input_wav(folder, di, force=True)
                    return "✓ input.wav set."
                except Exception as e:  # noqa: BLE001
                    return f"⚠️ {e}"

        # --- 2. Parameters ---
        with gr.Tab("2. Parameters"):
            # Locked summary (shown once params.json is generated).
            with gr.Group(visible=False) as params_locked:
                params_summary = gr.Markdown()
                params_edit_btn = gr.Button("✏️ Edit parameters")
            # Editable controls (hidden while locked).
            with gr.Group(visible=True) as params_edit:
                # Gradio 6: row_count accepts int only (no (min, "dynamic") tuples).
                # Use row_count=None (unlimited) for fully dynamic tables.
                knobs = gr.Dataframe(
                    headers=["name", "minimum", "maximum"],
                    datatype=["str", "number", "number"],
                    type="array",  # hand handlers list-of-lists, not a pandas frame
                    row_count=2,
                    label="Knobs (leave min/max blank to auto-derive)",
                    value=[["OD1", None, None], ["OD2", None, None]],
                )
                grid = gr.Dataframe(
                    label="Capture → values (first column = capture #)",
                    type="array", row_count=None)
                csv_dropdown = gr.Dropdown(
                    label="Params CSV (auto-detected in the capture folder)",
                    choices=[], interactive=True)
                csv_file = gr.File(label="…or upload a CSV")
                load_csv_btn = gr.Button("Load uploaded CSV into grid")
                gen_btn = gr.Button("Generate params.json", variant="primary")
                params_msg = gr.Markdown()

            def _grid_from_csv(csv_path):
                names, rows = pe.csv_to_grid(csv_path)
                knob_rows = [[n, None, None] for n in names]
                return knob_rows, rows

            def do_load_csv(csv):
                return _grid_from_csv(csv)

            load_csv_btn.click(do_load_csv, [csv_file], [knobs, grid])

            def do_load_csv_selected(folder, csv_name):
                if not folder or not csv_name:
                    return gr.update(), gr.update()
                return _grid_from_csv(str(Path(folder) / csv_name))

            csv_dropdown.change(do_load_csv_selected, [folder, csv_dropdown],
                                [knobs, grid])

            def _num_or_none(x):
                if x is None:
                    return None
                s = str(x).strip()
                if s == "" or s.lower() == "nan":
                    return None
                return float(s)

            def _is_blank(x):
                return x is None or str(x).strip() == "" or str(x).strip().lower() == "nan"

            def do_generate(folder, knob_tbl, grid_tbl):
                # Stay editable (no lock) on any validation/write error.
                def editable(msg):
                    return msg, gr.update(), gr.update(), gr.update()

                if not folder:
                    return editable("Scan a folder first.")
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
                rows = [r for r in grid_tbl if r and not _is_blank(r[0])]
                if not rows:
                    return editable(
                        "The capture grid is empty — load a CSV or add rows.")
                try:
                    path, warns = pe.generate_params_json(
                        folder, knob_specs, rows, force=True)
                except Exception as e:  # noqa: BLE001
                    return editable(f"⚠️ {e}")
                # Success → lock: hide the edit group, show the summary.
                summary = _params_summary_md(
                    path, len(rows), [k["name"] for k in knob_specs], warns)
                return (f"✓ Wrote `{path}`.",
                        gr.update(visible=False), gr.update(visible=True), summary)

            gen_btn.click(do_generate, [folder, knobs, grid],
                          [params_msg, params_edit, params_locked, params_summary])
            params_edit_btn.click(_show_edit, None, [params_edit, params_locked])

        # --- 3. Train ---
        with gr.Tab("3. Train"):
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
                out_dir = gr.Textbox("parametric_output", label="Output folder", scale=4)
                out_browse = gr.Button("📁 Browse…", scale=1)
            out_browse.click(lambda: nd.pick_folder() or gr.update(), None, out_dir)
            start_btn = gr.Button("Start training", variant="primary")
            stop_btn = gr.Button("Stop")
            log = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True)
            stop_msg = gr.Markdown()
            progress_md = gr.Markdown()
            metrics_plot = gr.Plot(label="Loss / ESR (log scale)")

            def do_train(folder, pattern, out, ms, dl, ep, l_r, ts, vs, box):
                try:
                    proc = rn.TrainProcess(rn.build_train_command(
                        folder, out, model_size=ms, delay=int(dl), epochs=int(ep),
                        lr=float(l_r), train_stop_seconds=float(ts),
                        val_start_seconds=float(vs), capture_pattern=pattern))
                    proc.start()
                except Exception as e:  # noqa: BLE001
                    yield f"⚠️ Could not start training: {e}", "", None, box
                    return
                box["proc"] = proc
                log_lines = deque(maxlen=400)  # bound the streamed log payload
                records, cur, last_fig = [], None, None
                eta_h, eta_m = None, None
                progress = ("⏳ Preparing data and starting training… "
                            "(the first epoch can take a while at 96 kHz)")
                yield "", progress, None, box
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
                        cur = {"epoch": ev["epoch"], "val_loss": None, "val_ESR": None}
                        records.append(cur)
                        progress = progress_bar_md(
                            ev["epoch"], ev["max_epochs"], eta_h, eta_m)
                    elif (ev and ev["type"] == "metric" and ev["value"] is not None
                          and cur is not None and ev["name"] in ("val_loss", "val_ESR")):
                        cur[ev["name"]] = ev["value"]
                        last_fig = loss_figure(records)
                        plot_out = last_fig
                    yield "\n".join(log_lines), progress, plot_out, box
                proc.wait()
                log_lines.append(f"[exited with code {proc.returncode}]")
                yield ("\n".join(log_lines), progress,
                       last_fig if last_fig is not None else gr.update(), box)

            start_btn.click(
                do_train,
                [folder, state_pattern, out_dir, model_size, delay, epochs, lr,
                 train_stop, val_start, proc_box],
                [log, progress_md, metrics_plot, proc_box])

            def do_stop(box):
                proc = box.get("proc")
                if proc:
                    proc.stop()
                    return "Stopping (saving model)…"
                return "Nothing to stop."

            stop_btn.click(do_stop, [proc_box], [stop_msg])

        # --- 4. Export ---
        with gr.Tab("4. Export"):
            with gr.Row():
                nam_name = gr.Textbox("model.nam", label="Output .nam path", scale=4)
                nam_browse = gr.Button("💾 Save as…", scale=1)
            nam_browse.click(lambda: nd.pick_save_file("model.nam") or gr.update(),
                             None, nam_name)
            sample_rate = gr.Number(
                96000, label="Sample rate (Hz)",
                info="Used only if the trained model file doesn't already store it.")
            export_btn = gr.Button("Export .nam", variant="primary")
            open_btn = gr.Button("Open output folder")
            export_msg = gr.Markdown()

            def do_export(out, name, sr):
                ckpt = Path(out) / "parametric_wavenet_model.pt"
                if not ckpt.exists():
                    return f"⚠️ No trained model at `{ckpt}`. Train first."
                name_path = Path(name)
                nam_path = name_path if name_path.is_absolute() else Path(out) / name
                cmd = rn.build_export_command(
                    ckpt, nam_path, sample_rate=int(sr) if sr else None)
                r = subprocess.run(cmd, cwd=str(rn.REPO_ROOT),
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    return f"⚠️ Export failed:\n```\n{r.stdout}\n{r.stderr}\n```"
                return f"✓ Exported `{nam_path}`."

            export_btn.click(do_export, [out_dir, nam_name, sample_rate], [export_msg])

            def do_open(out):
                p = Path(out).resolve()
                try:
                    if os.name == "nt":
                        os.startfile(p)  # noqa: S606
                except OSError as e:  # noqa: BLE001
                    return f"⚠️ Could not open `{p}`: {e}"
                return f"Opened `{p}`."

            open_btn.click(do_open, [out_dir], [export_msg])

        # --- Cross-tab wiring ---
        # Registered here (not inside the Dataset tab) because do_scan outputs to
        # the Params CSV dropdown, which lives on the Parameters tab.
        def do_scan(folder, override):
            if not folder or not folder.strip():
                # Empty folder → stay editable, unlocked.
                return (None, "", gr.update(choices=[], value=None),
                        _rate_md(ap.SampleRateInfo()),
                        gr.update(visible=True), gr.update(visible=False),
                        gr.update())
            override = override.strip() or None
            status = ds.scan_dataset(folder, override)
            csvs = status.csv_files
            # One CSV → select it (fires the dropdown's .change, which loads the
            # grid). Several → leave unselected so the user picks on the tab.
            dd = gr.update(choices=csvs,
                           value=(csvs[0] if len(csvs) == 1 else None))
            rate = ap.probe_sample_rates(folder, override)
            # Ready → lock into a compact summary; otherwise stay editable.
            edit_vis = gr.update(visible=not status.ready)
            locked_vis = gr.update(visible=status.ready)
            summary = _dataset_summary_md(status) if status.ready else gr.update()
            return (status.pattern, _status_md(status), dd, _rate_md(rate),
                    edit_vis, locked_vis, summary)

        scan_outputs = [state_pattern, status_md, csv_dropdown, train_rate_md,
                        dataset_edit, dataset_locked, dataset_summary]
        scan_inputs = [folder, pattern_override]
        # Auto-scan when a folder is chosen (Browse) or the path is committed
        # (Enter / focus-out); the Rescan button re-runs it after on-disk changes.
        folder_browse.click(lambda: nd.pick_folder() or gr.update(), None, folder) \
            .then(do_scan, scan_inputs, scan_outputs)
        folder.submit(do_scan, scan_inputs, scan_outputs)
        folder.blur(do_scan, scan_inputs, scan_outputs)
        scan_btn.click(do_scan, scan_inputs, scan_outputs)
        dataset_edit_btn.click(_show_edit, None, [dataset_edit, dataset_locked])
        # Placing input.wav can complete the dataset → re-scan so it can lock.
        set_input_btn.click(do_set_input, [folder, di_file], [input_msg]) \
            .then(do_scan, scan_inputs, scan_outputs)

    return app
