# analyze_metrics_nopandas.py
import os, sys, csv, math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# -------------------- small helpers --------------------
def safe_float(x):
    try: return float(x)
    except: return np.nan

def read_log_csv(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    # arrays (real)
    t   = np.array([safe_float(r["t"])  for r in rows])
    x   = np.array([safe_float(r["x"])  for r in rows])
    y   = np.array([safe_float(r["y"])  for r in rows])
    xd  = np.array([safe_float(r["xd"]) for r in rows])
    yd  = np.array([safe_float(r["yd"]) for r in rows])
    # ghosts if present
    xdos = np.array([safe_float(r.get("x_dos", "nan")) for r in rows])
    ydos = np.array([safe_float(r.get("y_dos", "nan")) for r in rows])
    xfdi = np.array([safe_float(r.get("x_fdi", "nan")) for r in rows])
    yfdi = np.array([safe_float(r.get("y_fdi", "nan")) for r in rows])
    mode = np.array([str(r.get("mode","main")) for r in rows])
    atk_dos = np.array([int(float(r.get("attack_dos",0))) for r in rows], dtype=int)
    atk_fdi = np.array([int(float(r.get("attack_fdi",0))) for r in rows], dtype=int)
    return dict(t=t,x=x,y=y,xd=xd,yd=yd,xdos=xdos,ydos=ydos,xfdi=xfdi,yfdi=yfdi,
                mode=mode, atk_dos=atk_dos, atk_fdi=atk_fdi)

def err_norm(x, y, xd, yd):
    return np.sqrt((x-xd)**2 + (y-yd)**2)

def metrics_from_err(e):
    e_f = e[np.isfinite(e)]
    if e_f.size == 0:
        return dict(mean=np.nan, rmse=np.nan, p95=np.nan, mx=np.nan, final=np.nan)
    return dict(
        mean=float(np.mean(e_f)),
        rmse=float(np.sqrt(np.mean(e_f**2))),
        p95=float(np.percentile(e_f, 95)),
        mx=float(np.max(e_f)),
        final=float(e_f[-1]),
    )

# -------------------- main --------------------
def main(batch_dir):
    if not os.path.isdir(batch_dir):
        print(f"ERROR: '{batch_dir}' is not a directory")
        return

    # discover runs
    log_paths = []
    for root, _, files in os.walk(batch_dir):
        if "log.csv" in files:
            log_paths.append(os.path.join(root, "log.csv"))
    if not log_paths:
        print("No runs found (no log.csv files).")
        return

    # outputs
    plots_dir = os.path.join(batch_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    summary_csv = os.path.join(batch_dir, "summary_np.csv")
    per_run_rows = []

    # containers for aggregate plots
    all_e_real = []
    all_e_dos  = []
    all_e_fdi  = []
    per_path_stats = defaultdict(lambda: dict(real=[], dos=[], fdi=[]))

    for lp in log_paths:
        rel = os.path.relpath(lp, batch_dir)
        parts = rel.split(os.sep)
        # try to infer path name and seed from folder structure “…/<path>/<seed>/log.csv”
        path_name = parts[-3] if len(parts) >= 3 else "unknown_path"
        seed_name = parts[-2] if len(parts) >= 2 else "unknown_seed"

        data = read_log_csv(lp)
        t = data["t"]
        # estimate dt for time-in-DT
        dt_est = float(np.nanmean(np.diff(t))) if t.size >= 2 else np.nan

        # errors
        e_real = err_norm(data["x"], data["y"], data["xd"], data["yd"])
        e_dos  = err_norm(data["xdos"], data["ydos"], data["xd"], data["yd"])
        e_fdi  = err_norm(data["xfdi"], data["yfdi"], data["xd"], data["yd"])

        m_real = metrics_from_err(e_real)
        m_dos  = metrics_from_err(e_dos)
        m_fdi  = metrics_from_err(e_fdi)

        # time in DT (mode == 'dt')
        time_in_dt = float(np.nansum((data["mode"] == "dt").astype(float)) * (dt_est if np.isfinite(dt_est) else 0.0))

        per_run_rows.append({
            "rel_path": rel,
            "path": path_name,
            "seed": seed_name,
            "samples": int(len(t)),
            "dt_est": dt_est,
            "time_in_dt": time_in_dt,
            # real
            "real_mean": m_real["mean"], "real_rmse": m_real["rmse"], "real_p95": m_real["p95"],
            "real_max": m_real["mx"], "real_final": m_real["final"],
            # ghosts
            "dos_mean": m_dos["mean"], "dos_rmse": m_dos["rmse"], "dos_p95": m_dos["p95"], "dos_max": m_dos["mx"],
            "fdi_mean": m_fdi["mean"], "fdi_rmse": m_fdi["rmse"], "fdi_p95": m_fdi["p95"], "fdi_max": m_fdi["mx"],
        })

        # collect for aggregates
        if np.isfinite(e_real).any(): all_e_real.append(e_real[np.isfinite(e_real)])
        if np.isfinite(e_dos).any():  all_e_dos.append(e_dos[np.isfinite(e_dos)])
        if np.isfinite(e_fdi).any():  all_e_fdi.append(e_fdi[np.isfinite(e_fdi)])

        per_path_stats[path_name]["real"].append(m_real["rmse"])
        if np.isfinite(m_dos["rmse"]): per_path_stats[path_name]["dos"].append(m_dos["rmse"])
        if np.isfinite(m_fdi["rmse"]): per_path_stats[path_name]["fdi"].append(m_fdi["rmse"])

    # write summary CSV (no pandas)
    with open(summary_csv, "w", newline="") as f:
        cols = list(per_run_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(per_run_rows)
    print(f"[ok] wrote {len(per_run_rows)} rows → {summary_csv}")

    # ---------- aggregate plots ----------
    def plot_cdf(arrays, labels, outpath, title="CDF of |error|"):
        plt.figure(figsize=(7,5))
        for a, lab in zip(arrays, labels):
            if not a: continue
            x = np.concatenate(a)
            x = x[np.isfinite(x)]
            if x.size == 0: continue
            xx = np.sort(x)
            yy = np.linspace(0,1,len(xx))
            plt.plot(xx, yy, label=lab)
        plt.xlabel("|e| [m]"); plt.ylabel("CDF"); plt.grid(True, alpha=0.3)
        plt.legend(); plt.title(title)
        plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()

    def plot_bar_per_path(stats_dict, outpath, title="RMSE per path"):
        paths = sorted(stats_dict.keys())
        rmse_real = [np.nanmean(stats_dict[p]["real"]) if stats_dict[p]["real"] else np.nan for p in paths]
        rmse_dos  = [np.nanmean(stats_dict[p]["dos"])  if stats_dict[p]["dos"]  else np.nan for p in paths]
        rmse_fdi  = [np.nanmean(stats_dict[p]["fdi"])  if stats_dict[p]["fdi"]  else np.nan for p in paths]
        x = np.arange(len(paths))
        w = 0.25
        plt.figure(figsize=(10,5))
        plt.bar(x - w, rmse_real, width=w, label="Real (DT rescue)")
        plt.bar(x,       rmse_dos, width=w, label="Ghost DoS")
        plt.bar(x + w, rmse_fdi, width=w, label="Ghost FDI")
        plt.xticks(x, paths, rotation=15)
        plt.ylabel("RMSE |e| [m]"); plt.title(title)
        plt.legend(); plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()

    plot_cdf([all_e_real, all_e_dos, all_e_fdi],
             ["real (DT rescue)", "ghost DoS", "ghost FDI"],
             os.path.join(plots_dir, "cdf_error.png"))

    plot_bar_per_path(per_path_stats, os.path.join(plots_dir, "rmse_per_path.png"))

    print(f"[ok] saved plots → {plots_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics_nopandas.py <batch_dir>")
        sys.exit(1)
    main(sys.argv[1])
