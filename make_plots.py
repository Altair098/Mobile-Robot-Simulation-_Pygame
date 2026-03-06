# make_plots.py
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Usage:
  python make_plots.py results_compare/<STAMP>
Requires:
  metrics.csv (from analyze_metrics.py)
Outputs:
  <root>/plots/summary_bars.png
  <root>/plots/cdf_errors.png
"""

KEYS = [
    ("completion", "Completion (s_max)"),
    ("e_mean", "Mean |e| [m]"),
    ("e_max", "Max |e| [m]"),
    ("dt_frac", "Time in DT [%]"),
    ("tts_after_attack", "Stabilize after attack [s]"),
]

def main(root):
    csv_path = os.path.join(root, "metrics.csv")
    if not os.path.isfile(csv_path):
        print("metrics.csv not found. Run analyze_metrics.py first.")
        return
    df = pd.read_csv(csv_path)

    outdir = os.path.join(root, "plots"); os.makedirs(outdir, exist_ok=True)

    # --- BAR SUMMARY (means with 95% CI) ---
    means = df.mean(numeric_only=True)
    stds  = df.std(numeric_only=True, ddof=1)
    n = len(df)
    ci = 1.96 * stds / np.sqrt(max(n,1))

    vals = []; labs = []; errs = []
    for k, lab in KEYS:
        if k in means.index:
            v = means[k]
            if k == "dt_frac": v = 100.0 * v
            vals.append(v); labs.append(lab); errs.append(ci.get(k, np.nan))
    x = np.arange(len(vals))
    plt.figure(figsize=(9,4))
    plt.bar(x, vals, yerr=errs, capsize=3)
    plt.xticks(x, labs, rotation=20, ha='right')
    plt.title("Aggregate Metrics (mean ± 95% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_bars.png"), dpi=180)

    # --- CDF of tracking error (real vs ghosts if present) ---
    # Load per-run logs quickly to stack e for real, dos, fdi
    # (Optional: if too slow, skip or thin)
    es_real, es_dos, es_fdi = [], [], []
    for run in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]):
        logp = os.path.join(root, run, "log.csv")
        if not os.path.isfile(logp): continue
        try:
            L = pd.read_csv(logp)
        except Exception:
            continue
        e = np.sqrt((L["x"]-L["xd"])**2 + (L["y"]-L["yd"])**2).values
        es_real.append(e)
        if "x_dos" in L and "y_dos" in L:
            mask = np.isfinite(L["x_dos"]) & np.isfinite(L["y_dos"])
            if mask.any():
                e_dos = np.sqrt((L["x_dos"][mask]-L["xd"][mask])**2 + (L["y_dos"][mask]-L["yd"][mask])**2).values
                es_dos.append(e_dos)
        if "x_fdi" in L and "y_fdi" in L:
            mask = np.isfinite(L["x_fdi"]) & np.isfinite(L["y_fdi"])
            if mask.any():
                e_fdi = np.sqrt((L["x_fdi"][mask]-L["xd"][mask])**2 + (L["y_fdi"][mask]-L["yd"][mask])**2).values
                es_fdi.append(e_fdi)

    def ecdf(stacked):
        if stacked.size == 0: return np.array([]), np.array([])
        x = np.sort(stacked); y = np.arange(1, len(x)+1)/len(x); return x,y

    e_real = np.concatenate(es_real) if es_real else np.array([])
    e_dos  = np.concatenate(es_dos)  if es_dos  else np.array([])
    e_fdi  = np.concatenate(es_fdi)  if es_fdi  else np.array([])

    xr, yr = ecdf(e_real)
    xd, yd = ecdf(e_dos)
    xf, yf = ecdf(e_fdi)

    plt.figure(figsize=(6,4))
    if xr.size: plt.plot(xr, yr, label="Real (Main/DT)")
    if xd.size: plt.plot(xd, yd, label="Ghost DoS")
    if xf.size: plt.plot(xf, yf, label="Ghost FDI")
    plt.xlabel("|e| [m]"); plt.ylabel("CDF")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "cdf_errors.png"), dpi=180)

    print(f"Saved plots to {outdir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_plots.py <results_root>")
        sys.exit(1)
    main(sys.argv[1])
