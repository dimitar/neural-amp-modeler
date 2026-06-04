# File: trainer_app/app.py
# Purpose: Gradio Blocks UI wiring the trainer backend (dataset, params, runner).

import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr

from trainer_app import dataset as ds
from trainer_app import native_dialogs as nd
from trainer_app import params_editor as pe
from trainer_app import runner as rn
from trainer_app.progress_view import loss_figure, progress_bar_md

DEFAULTS = dict(model_size="small", delay=10, epochs=250, lr=4e-3,
                train_stop_seconds=-9.0, val_start_seconds=-9.0)


def _status_md(status):
    lines = [f"**Pattern:** `{status.pattern or '—'}`",
             f"**input.wav:** {'✓' if status.input_wav_present else '✗ missing'}",
             f"**Captures matched:** {len(status.capture_numbers)}"]
    for e in status.errors:
        lines.append(f"- ⚠️ {e}")
    return "\n\n".join(lines)


def build_app():
    with gr.Blocks(title="NAM Parametric Trainer") as app:
        gr.Markdown("# NAM Parametric Trainer")
        state_pattern = gr.State(None)
        proc_box = gr.State({})  # holds {"proc": TrainProcess}

        # --- 1. Dataset ---
        with gr.Tab("1. Dataset"):
            with gr.Row():
                folder = gr.Textbox(label="Capture folder", scale=4)
                folder_browse = gr.Button("📁 Browse…", scale=1)
            folder_browse.click(lambda: nd.pick_folder() or gr.update(), None, folder)
            pattern_override = gr.Textbox(
                label="Capture filename pattern (only if auto-detect fails)",
                placeholder="e.g. MP-1 3TM NAM Amp DI {cap_num}.wav",
            )
            scan_btn = gr.Button("Scan folder")
            status_md = gr.Markdown()
            di_file = gr.File(label="DI / input signal (.wav)")
            set_input_btn = gr.Button("Set as input.wav")
            input_msg = gr.Markdown()

            def do_scan(folder, override):
                override = override.strip() or None
                status = ds.scan_dataset(folder, override)
                return status.pattern, _status_md(status)

            scan_btn.click(do_scan, [folder, pattern_override],
                           [state_pattern, status_md])

            def do_set_input(folder, di):
                if not folder or not di:
                    return "Pick a folder and a DI file first."
                try:
                    ds.place_input_wav(folder, di, force=True)
                    return "✓ input.wav set."
                except Exception as e:  # noqa: BLE001
                    return f"⚠️ {e}"

            set_input_btn.click(do_set_input, [folder, di_file], [input_msg])

        # --- 2. Parameters ---
        with gr.Tab("2. Parameters"):
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
            grid = gr.Dataframe(label="Capture → values (first column = capture #)",
                                type="array", row_count=None)
            csv_file = gr.File(label="…or load a CSV")
            load_csv_btn = gr.Button("Load CSV into grid")
            gen_btn = gr.Button("Generate params.json", variant="primary")
            params_msg = gr.Markdown()

            def do_load_csv(csv):
                names, rows = pe.csv_to_grid(csv)
                knob_rows = [[n, None, None] for n in names]
                return knob_rows, rows

            load_csv_btn.click(do_load_csv, [csv_file], [knobs, grid])

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
                if not folder:
                    return "Scan a folder first."
                knob_specs = []
                for r in knob_tbl:
                    if not r or _is_blank(r[0]):
                        continue
                    spec = {"name": str(r[0]).strip()}
                    try:
                        mn, mx = _num_or_none(r[1]), _num_or_none(r[2])
                    except ValueError:
                        return f"⚠️ Knob '{spec['name']}' has a non-numeric min/max."
                    if mn is not None and mx is not None:
                        spec["minimum"], spec["maximum"] = mn, mx
                    knob_specs.append(spec)
                if not knob_specs:
                    return "Add at least one knob (name) first."
                rows = [r for r in grid_tbl if r and not _is_blank(r[0])]
                if not rows:
                    return "The capture grid is empty — load a CSV or add rows."
                try:
                    path, warns = pe.generate_params_json(
                        folder, knob_specs, rows, force=True)
                except Exception as e:  # noqa: BLE001
                    return f"⚠️ {e}"
                msg = f"✓ Wrote `{path}`."
                for g in warns:
                    msg += f"\n\n⚠️ Captures {g} share identical knob values."
                return msg

            gen_btn.click(do_generate, [folder, knobs, grid], [params_msg])

        # --- 3. Train ---
        with gr.Tab("3. Train"):
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
                buffer, progress, records, cur, last_fig = "", "", [], None, None
                for line in proc.stream():
                    buffer += line + "\n"
                    ev = rn.parse_progress_line(line)
                    plot_out = gr.update()
                    if ev and ev["type"] == "epoch":
                        cur = {"epoch": ev["epoch"], "val_loss": None, "val_ESR": None}
                        records.append(cur)
                        progress = progress_bar_md(
                            ev["epoch"], ev["max_epochs"], ev["eta_h"], ev["eta_m"])
                    elif (ev and ev["type"] == "metric" and ev["value"] is not None
                          and cur is not None and ev["name"] in ("val_loss", "val_ESR")):
                        cur[ev["name"]] = ev["value"]
                        last_fig = loss_figure(records)
                        plot_out = last_fig
                    yield buffer, progress, plot_out, box
                proc.wait()
                buffer += f"\n[exited with code {proc.returncode}]\n"
                yield buffer, progress, (last_fig if last_fig is not None else gr.update()), box

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
            export_btn = gr.Button("Export .nam", variant="primary")
            open_btn = gr.Button("Open output folder")
            export_msg = gr.Markdown()

            def do_export(out, name):
                ckpt = Path(out) / "parametric_wavenet_model.pt"
                if not ckpt.exists():
                    return f"⚠️ No trained model at `{ckpt}`. Train first."
                name_path = Path(name)
                nam_path = name_path if name_path.is_absolute() else Path(out) / name
                cmd = rn.build_export_command(ckpt, nam_path)
                r = subprocess.run(cmd, cwd=str(rn.REPO_ROOT),
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    return f"⚠️ Export failed:\n```\n{r.stdout}\n{r.stderr}\n```"
                return f"✓ Exported `{nam_path}`."

            export_btn.click(do_export, [out_dir, nam_name], [export_msg])

            def do_open(out):
                p = Path(out).resolve()
                try:
                    if os.name == "nt":
                        os.startfile(p)  # noqa: S606
                except OSError as e:  # noqa: BLE001
                    return f"⚠️ Could not open `{p}`: {e}"
                return f"Opened `{p}`."

            open_btn.click(do_open, [out_dir], [export_msg])

    return app
