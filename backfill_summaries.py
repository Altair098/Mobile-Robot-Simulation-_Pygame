# backfill_summaries.py
import os, csv, json, numpy as np, sys

root = sys.argv[1]  # e.g., results_batch_all/20250909_075921

def summarize(run_dir):
    logf = os.path.join(run_dir, "log.csv")
    if not os.path.isfile(logf): return False
    with open(logf, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows: return False
    x  = np.array([float(r["x"])  for r in rows])
    y  = np.array([float(r["y"])  for r in rows])
    xd = np.array([float(r["xd"]) for r in rows])
    yd = np.array([float(r["yd"]) for r in rows])
    err_real = np.hypot(x-xd, y-yd).mean()
    # ghosts may be NaN in some ticks—guard:
    def safe_mean(a,b):
        v = np.hypot(a-xd, b-yd)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    xdos = np.array([float(r["x_dos"]) for r in rows])
    ydos = np.array([float(r["y_dos"]) for r in rows])
    xfdi = np.array([float(r["x_fdi"]) for r in rows])
    yfdi = np.array([float(r["y_fdi"]) for r in rows])
    err_dos = safe_mean(xdos, ydos)
    err_fdi = safe_mean(xfdi, yfdi)
    time_in_dt = np.mean([1.0 if r["mode"]=="dt" else 0.0 for r in rows])
    tube_viol = int(np.sum([abs(float(r["d_perp"])) > float(r["r_eff"]) for r in rows]))
    s_final = float(rows[-1]["s"])
    out = dict(s_final=s_final, err_real_mean=float(err_real),
               err_dos_mean=err_dos, err_fdi_mean=err_fdi,
               time_in_dt_ratio=float(time_in_dt), tube_violations=tube_viol)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return True

for pdir in os.listdir(root):
    path_dir = os.path.join(root, pdir)
    if not os.path.isdir(path_dir): continue
    for seed_dir in os.listdir(path_dir):
        run_dir = os.path.join(path_dir, seed_dir)
        if not os.path.isdir(run_dir): continue
        if not os.path.exists(os.path.join(run_dir, "summary.json")):
            summarize(run_dir)
print("Done.")
