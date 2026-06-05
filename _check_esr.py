from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os

versions = sorted(glob.glob(r"C:\Users\dimit\source\guitar-plugin\neural-amp-modeler\parametric_output_ss_clean\lightning_logs\version_*"))
p = versions[-1]
print(f"reading: {os.path.basename(p)}")
ea = EventAccumulator(p)
ea.Reload()
tags = ea.Tags().get("scalars", [])
print(f"available scalar tags: {tags}")

esr = [(e.step, e.value) for e in ea.Scalars("val_ESR")] if "val_ESR" in tags else []
mse = [(e.step, e.value) for e in ea.Scalars("val_MSE")] if "val_MSE" in tags else []

print(f"\nepochs logged: {len(esr)}")
print("latest val_ESR (last 10):")
mse_map = dict(mse)
for s, v in esr[-10:]:
    m = mse_map.get(s, float("nan"))
    print(f"  epoch {s:3d}: ESR={v:.5f}  MSE={m:.6f}")

if esr:
    best_e_step, best_e_val = min(esr, key=lambda t: t[1])
    print(f"\nbest val_ESR: {best_e_val:.5f} at epoch {best_e_step}")
if mse:
    best_m_step, best_m_val = min(mse, key=lambda t: t[1])
    print(f"best val_MSE: {best_m_val:.6f} at epoch {best_m_step}")
