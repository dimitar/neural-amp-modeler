"""Monitor parametric WaveNet training and post updates to Slack every 5 epochs."""
import glob
import json
import os
import signal
import subprocess
import time

WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL", "")
LOG_DIR = "parametric_output_small_test/lightning_logs"
MAX_EPOCHS = 50
CHECK_INTERVAL = 30  # seconds between polls
REPORT_EVERY = 5  # epochs between Slack reports
BASELINE_ESR = 0.577  # previous best


def slack(msg):
    subprocess.run(
        ["curl", "-s", "-X", "POST", "-H", "Content-type: application/json",
         "--data", json.dumps({"text": msg}), WEBHOOK],
        capture_output=True,
    )


def read_metrics():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    versions = sorted(glob.glob(os.path.join(LOG_DIR, "version_*")))
    if not versions:
        return None
    ea = EventAccumulator(versions[-1])
    ea.Reload()
    result = {}
    for tag in ["val_ESR", "val_MSE"]:
        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            result[tag] = [(e.wall_time, e.value) for e in events]
    return result


def format_report(metrics, epochs_done):
    esr_vals = [v for _, v in metrics.get("val_ESR", [])]
    mse_vals = [v for _, v in metrics.get("val_MSE", [])]
    wall_times = [t for t, _ in metrics.get("val_ESR", [])]

    current_esr = esr_vals[-1]
    current_mse = mse_vals[-1]
    best_esr = min(esr_vals)
    best_esr_epoch = esr_vals.index(best_esr) + 1

    # ETA
    if len(wall_times) >= 2:
        elapsed = wall_times[-1] - wall_times[0]
        sec_per_epoch = elapsed / (len(wall_times) - 1)
        remaining = MAX_EPOCHS - epochs_done
        eta_min = remaining * sec_per_epoch / 60
        eta_str = f"{eta_min:.0f}min" if eta_min < 120 else f"{eta_min / 60:.1f}hr"
    else:
        eta_str = "calculating..."

    # Rolling averages (5-epoch window) and sparkline
    def rolling_avg(vals, window=5):
        if len(vals) < window:
            return vals[:]
        return [sum(vals[i:i+window]) / window
                for i in range(len(vals) - window + 1)]

    avgs = rolling_avg(esr_vals)

    # Sparkline from rolling averages (log scale, sampled to ~15 points)
    import math
    spark_chars = " ▁▂▃▄▅▆▇█"
    if len(avgs) >= 2:
        log_avgs = [math.log10(max(v, 1e-6)) for v in avgs]
        lo, hi = min(log_avgs), max(log_avgs)
        span = hi - lo if hi > lo else 1e-9
        n_points = min(len(log_avgs), 15)
        step = max(1, len(log_avgs) // n_points)
        sampled = log_avgs[::step]
        spark = "".join(spark_chars[min(8, int((v - lo) / span * 8))]
                        for v in sampled)
        sparkline = f"`{spark}` ({avgs[0]:.4f} -> {avgs[-1]:.4f})"
    else:
        sparkline = "not enough data"

    # Trend: compare current rolling avg window to previous window
    trend_parts = []
    if len(avgs) >= 2:
        recent_change = (avgs[-2] - avgs[-1]) / avgs[-2] * 100
        trend_parts.append(f"5ep avg: {avgs[-1]:.4f} ({recent_change:+.1f}% from prev)")
    if len(avgs) >= 6:
        # Longer-term: compare current avg to 25 epochs ago
        lookback = min(25, len(avgs) - 1)
        old_avg = avgs[-1 - lookback]
        long_change = (old_avg - avgs[-1]) / old_avg * 100
        trend_parts.append(f"vs {lookback}ep ago: {long_change:+.1f}%")
    trend = " | ".join(trend_parts) if trend_parts else "n/a"

    # Analysis
    vs_baseline = (1 - current_esr / BASELINE_ESR) * 100
    if current_esr < 0.02:
        verdict = ":star: Excellent — approaching single-capture quality. Keep going."
    elif current_esr < 0.05:
        verdict = ":rocket: Very good — strong parametric model. Worth continuing."
    elif current_esr < 0.10:
        verdict = ":chart_with_upwards_trend: Good progress — still improving meaningfully."
    elif current_esr < 0.20:
        verdict = ":eyes: Decent but watch for plateau — consider stopping if trend flattens."
    else:
        verdict = ":warning: Still high — early training, patience needed."

    # Check for stall
    if len(esr_vals) >= 10:
        last10 = esr_vals[-10:]
        if max(last10) - min(last10) < 0.005:
            verdict = ":octagonal_sign: Plateau detected — ESR flat for 10 epochs. Consider stopping."

    lines = [
        f"*Parametric WaveNet — Epoch {epochs_done}/{MAX_EPOCHS}*",
        f"",
        f"val_ESR: *{current_esr:.4f}* (best: {best_esr:.4f} @ epoch {best_esr_epoch})",
        f"val_MSE: {current_mse:.6f}",
        f"vs baseline (0.577): *{vs_baseline:+.1f}%*",
        f"Trend: {trend}",
        f"History: {sparkline}",
        f"ETA: {eta_str}",
        f"",
        verdict,
    ]
    return "\n".join(lines)


def find_training_pid():
    """Find the PID of the train_parametric.py process."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             "commandline like '%train_parametric.py%train%'",
             "get", "processid"],
            capture_output=True, text=True, timeout=10,
        )
        pids = [int(line.strip()) for line in result.stdout.strip().split("\n")
                if line.strip().isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


def kill_training():
    """Kill the training process."""
    pid = find_training_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Killed training process (PID {pid})")
            return True
        except OSError:
            pass
    # Fallback: taskkill
    subprocess.run(
        ["taskkill", "/F", "/FI", "WINDOWTITLE eq *train_parametric*"],
        capture_output=True,
    )
    return True


def check_collapse(metrics):
    """Detect model collapse. Returns a reason string or None."""
    esr_vals = [v for _, v in metrics.get("val_ESR", [])]
    mse_vals = [v for _, v in metrics.get("val_MSE", [])]
    if len(esr_vals) < 3:
        return None

    current_esr = esr_vals[-1]
    best_esr = min(esr_vals[:-1]) if len(esr_vals) > 1 else esr_vals[0]

    # ESR exploded above 1.0 (model producing garbage)
    if current_esr > 1.0:
        return f"val_ESR spiked to {current_esr:.3f} (>1.0 = worse than silence)"

    # ESR regressed massively from best (3x worse and above 0.5)
    if current_esr > 0.5 and best_esr < 0.2 and current_esr > best_esr * 3:
        return f"val_ESR regressed to {current_esr:.3f} from best {best_esr:.3f} (3x+ worse)"

    # NaN/Inf in MSE
    if mse_vals and (mse_vals[-1] != mse_vals[-1] or mse_vals[-1] > 1e6):
        return f"val_MSE is {mse_vals[-1]} (NaN/Inf/exploded)"

    # Sustained regression: ESR increasing for 5+ consecutive epochs
    if len(esr_vals) >= 6:
        last6 = esr_vals[-6:]
        if all(last6[i] < last6[i + 1] for i in range(5)):
            return (f"val_ESR rising for 5 consecutive epochs: "
                    f"{last6[0]:.4f} -> {last6[-1]:.4f}")

    return None


def main():
    last_reported = 0
    last_collapse_check = 0
    print("Monitoring started. Polling every 30s, reporting every 5 epochs.")
    print("Collapse detection active — will kill training if model collapses.")
    while True:
        metrics = read_metrics()
        if metrics and "val_ESR" in metrics:
            epochs_done = len(metrics["val_ESR"])

            # Check for collapse every epoch (not just at report intervals)
            if epochs_done > last_collapse_check:
                last_collapse_check = epochs_done
                collapse_reason = check_collapse(metrics)
                if collapse_reason:
                    esr_vals = [v for _, v in metrics["val_ESR"]]
                    msg = (
                        f":rotating_light: *MODEL COLLAPSE DETECTED — Epoch {epochs_done}*\n\n"
                        f"Reason: {collapse_reason}\n"
                        f"Last 5 val_ESR: {[f'{v:.4f}' for v in esr_vals[-5:]]}\n\n"
                        f"Killing training process. Best checkpoint is preserved in "
                        f"parametric_output/checkpoints/"
                    )
                    slack(msg)
                    print(f"[Epoch {epochs_done}] COLLAPSE: {collapse_reason}")
                    kill_training()
                    print("Training killed. Monitor exiting.")
                    break

            # Report at every 5-epoch boundary, and at completion
            next_milestone = ((last_reported // REPORT_EVERY) + 1) * REPORT_EVERY
            if epochs_done >= next_milestone or epochs_done >= MAX_EPOCHS:
                msg = format_report(metrics, epochs_done)
                slack(msg)
                print(f"[Epoch {epochs_done}] Sent Slack update")
                last_reported = epochs_done
            if epochs_done >= MAX_EPOCHS:
                slack(":checkered_flag: *Training complete!* Check parametric_output/ for results.")
                print("Training complete. Monitor exiting.")
                break
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
