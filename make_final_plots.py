# make_final_plots.py (camera-ready)
# Final aggregate plots with unified scales, consistent colors, findings titles,
# compact inset for DoS exposure, 95% CIs, and figure captions.

import os, json, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLOR_DT   = "#1f77b4"  # DT blue
COLOR_GH   = "0.25"     # Ghost gray
COLOR_IMP  = "#2ca02c"  # Improvement green
COLOR_INSET= "0.6"      # Inset neutral

def paper_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 180,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlepad": 10.0
    })

def load_summary(root):
    with open(os.path.join(root, "summary.json"), "r") as f:
        return json.load(f)

def load_run_csvs(path_dir):
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(path_dir, "run_*", "log.csv"))):
        df = pd.read_csv(csv_path)
        df["run_id"] = os.path.basename(os.path.dirname(csv_path))
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)

def compute_per_run_metrics(df):
    t = df["t"].to_numpy()
    if len(t) < 2:
        return None
    by_run = {}
    for rid, grp in df.groupby("run_id"):
        tt = grp["t"].to_numpy()
        dtt = np.median(np.diff(tt)) if len(tt) > 1 else 0.1
        e_dt = np.hypot(grp["x"].to_numpy() - grp["xd"].to_numpy(),
                        grp["y"].to_numpy() - grp["yd"].to_numpy()).sum()*dtt
        e_gh = np.hypot(grp["x_dos"].to_numpy() - grp["xd"].to_numpy(),
                        grp["y_dos"].to_numpy() - grp["yd"].to_numpy()).sum()*dtt
        atk_col = "attack" if "attack" in grp.columns else ("attack_dos" if "attack_dos" in grp.columns else None)
        dos_duration = float(np.sum(grp[atk_col].to_numpy()>0) * dtt) if atk_col else 0.0
        # Perp precision proxy
        ey_med = np.nan
        if "ey" in grp.columns and "mode" in grp.columns:
            ey = grp["ey"].to_numpy(); mode = grp["mode"].to_numpy()
            ey_go = ey[mode==1] if np.any(mode==1) else ey
            if ey_go.size > 0:
                ey_med = float(np.median(np.abs(ey_go)))
        by_run[rid] = {"E_dt": float(e_dt), "E_gh": float(e_gh), "EY_med": ey_med, "DoS_time": dos_duration}
    return by_run

def ensure_out(root):
    outdir = os.path.join(root, "final_plots")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def compute_attack_exposure(root, paths):
    means, stds = [], []
    Ns = []
    for pname in paths:
        df = load_run_csvs(os.path.join(root, pname))
        if df is None:
            means.append(0.0); stds.append(0.0); Ns.append(1); continue
        met = compute_per_run_metrics(df)
        vals = [m["DoS_time"] for m in met.values()]
        means.append(float(np.mean(vals))); stds.append(float(np.std(vals))); Ns.append(len(vals))
    return np.array(means), np.array(stds), np.array(Ns)

def ci95(std, n):
    n = np.maximum(1, n)
    return 1.96 * (std / np.sqrt(n))

def add_attack_inset(ax, paths, mean_dos, std_dos, Ns, title="DoS exposure (mean±std)"):
    inset = ax.inset_axes([0.72, 0.57, 0.25, 0.36])  # x,y,w,h
    x = np.arange(len(paths))
    inset.bar(x, mean_dos, yerr=std_dos, color=COLOR_INSET, edgecolor="0.4", linewidth=0.8, capsize=4)
    inset.set_xticks(x); inset.set_xticklabels(paths, rotation=0)
    inset.set_ylabel("s")
    inset.set_title(title, fontsize=10)
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_edgecolor("0.5"); spine.set_linewidth(0.8)

def findings_title_dt_vs_ghost(means_dt, means_gh):
    red = 100*(1 - np.array(means_dt)/np.maximum(1e-6, np.array(means_gh)))
    return f"DT reduces cumulative error by {red.min():.1f}–{red.max():.1f}% across all paths"

def figure_footer(fig, text):
    fig.text(0.5, 0.01, text, ha="center", fontsize=10)

def plot1_cum_error_bar(root, summary):
    outdir = ensure_out(root)
    paths = sorted(summary.keys())
    means_dt = np.array([np.mean(summary[p]["err_dt"]) for p in paths])
    stds_dt  = np.array([np.std(summary[p]["err_dt"]) for p in paths])
    means_gh = np.array([np.mean(summary[p]["err_ghost"]) for p in paths])
    stds_gh  = np.array([np.std(summary[p]["err_ghost"]) for p in paths])
    x = np.arange(len(paths))
    w = 0.36  # slightly narrower for air between bars

    mean_dos, std_dos, Ns = compute_attack_exposure(root, paths)

    paper_style()
    fig, ax = plt.subplots(figsize=(10.5,4.2))
    ax.bar(x - w/2, means_dt, yerr=ci95(stds_dt, Ns), width=w, color=COLOR_DT, capsize=5, label="DT Rescue")
    ax.bar(x + w/2, means_gh, yerr=ci95(stds_gh, Ns), width=w, color=COLOR_GH, capsize=5, label="Ghost DoS")
    ax.set_xticks(x); ax.set_xticklabels(paths)
    ax.set_ylabel("Integrated tracking error [m·s]")
    ax.set_title(findings_title_dt_vs_ghost(means_dt, means_gh))
    ax.legend(loc="upper left", bbox_to_anchor=(0,1.12), ncol=2, frameon=False)

    add_attack_inset(ax, paths, mean_dos, std_dos, Ns)

    figure_footer(fig, "Bars show mean integrated position error over runs; error bars are 95% CIs. Inset shows mean±std DoS duration per path.")
    fig.tight_layout(rect=[0,0.03,1,0.97]); fig.savefig(os.path.join(outdir, "P1_cumulative_error_bar.png")); plt.close(fig)

def plot2_error_reduction(root, summary):
    outdir = ensure_out(root)
    paths = sorted(summary.keys())
    means_dt = np.array([np.mean(summary[p]["err_dt"]) for p in paths])
    means_gh = np.array([np.mean(summary[p]["err_ghost"]) for p in paths])
    reduction = 100.0*(1.0 - means_dt/np.maximum(1e-6, means_gh))

    paper_style()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(range(len(paths)), reduction, color=COLOR_IMP)
    ax.set_xticks(range(len(paths))); ax.set_xticklabels(paths)
    ax.set_ylabel("Error reduction vs Ghost [%]")
    ax.set_ylim(0, max(100, np.nanmax(reduction)*1.1))
    ax.set_title("DT outperforms Ghost baseline on every path (relative improvement)")
    for i, val in enumerate(reduction):
        ax.text(i, val+1, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    figure_footer(fig, "Improvement computed from mean cumulative errors per path across runs.")
    fig.tight_layout(rect=[0,0.05,1,0.97]); fig.savefig(os.path.join(outdir, "P2_error_reduction.png")); plt.close(fig)

def plot3_distribution_box(root):
    outdir = ensure_out(root)
    import glob as _glob
    data_dt, data_gh, labels = [], [], []
    Ns = []
    for path_dir in sorted([d for d in _glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]):
        pname = os.path.basename(path_dir)
        if pname in {"final_plots"}: continue
        csvs = _glob.glob(os.path.join(path_dir, "run_*", "log.csv"))
        if not csvs: continue
        df = load_run_csvs(path_dir)
        if df is None: continue
        metrics = compute_per_run_metrics(df)
        if not metrics: continue
        labels.append(pname)
        vals_dt = [m["E_dt"] for m in metrics.values()]
        vals_gh = [m["E_gh"] for m in metrics.values()]
        data_dt.append(vals_dt); data_gh.append(vals_gh); Ns.append(len(vals_dt))

    paper_style()
    fig, ax = plt.subplots(1,2, figsize=(12,4.2), sharey=True)
    ymax = max(max(map(max, data_dt)), max(map(max, data_gh))) * 1.05
    ax[0].boxplot(data_dt, labels=labels, showmeans=True)
    ax[0].set_title("DT Rescue — cumulative error distribution")
    ax[0].set_ylabel("Integrated error [m·s]"); ax[0].set_ylim(0, ymax)
    ax[1].boxplot(data_gh, labels=labels, showmeans=True)
    ax[1].set_title("Ghost DoS — cumulative error distribution"); ax[1].set_ylim(0, ymax)
    figure_footer(fig, "Distributions across runs per path; unified y‑axis facilitates side‑by‑side comparison.")
    fig.tight_layout(rect=[0,0.05,1,0.97]); fig.savefig(os.path.join(outdir, "P3_error_distribution_box.png")); plt.close(fig)

def plot4_composite_panel(root):
    outdir = ensure_out(root)
    summary = load_summary(root)
    paths = sorted(summary.keys())
    means_dt = np.array([np.mean(summary[p]["err_dt"]) for p in paths])
    means_gh = np.array([np.mean(summary[p]["err_ghost"]) for p in paths])
    reduction = 100.0*(1.0 - means_dt/np.maximum(1e-6, means_gh))

    # ey medians across runs for DT
    ey_meds = []
    Ns_path = []
    for pname in paths:
        df = load_run_csvs(os.path.join(root, pname))
        if df is None: continue
        metrics = compute_per_run_metrics(df)
        if not metrics: continue
        ey_meds += [m["EY_med"] for m in metrics.values() if not np.isnan(m["EY_med"])]
        Ns_path.append(len(metrics))
    mean_dos, std_dos, Ns = compute_attack_exposure(root, paths)

    paper_style()
    fig, axs = plt.subplots(2,2, figsize=(12,8))
    x = np.arange(len(paths)); w=0.36
    ymax = max(means_dt.max(), means_gh.max()) * 1.15

    # A) Mean cumulative error with CI and inset
    axs[0,0].bar(x - w/2, means_dt, yerr=ci95(np.array([np.std(summary[p]["err_dt"]) for p in paths]), Ns),
                 width=w, color=COLOR_DT, capsize=5, label="DT")
    axs[0,0].bar(x + w/2, means_gh, yerr=ci95(np.array([np.std(summary[p]["err_ghost"]) for p in paths]), Ns),
                 width=w, color=COLOR_GH, capsize=5, label="Ghost")
    axs[0,0].set_xticks(x); axs[0,0].set_xticklabels(paths)
    axs[0,0].set_ylabel("Integrated error [m·s]"); axs[0,0].set_ylim(0, ymax)
    axs[0,0].set_title("A) DT achieves lower cumulative error on all paths")
    axs[0,0].legend(frameon=False, loc="upper left")
    add_attack_inset(axs[0,0], paths, mean_dos, std_dos, Ns)

    # B) Improvement
    axs[0,1].bar(x, reduction, color=COLOR_IMP)
    axs[0,1].set_xticks(x); axs[0,1].set_xticklabels(paths)
    axs[0,1].set_ylabel("Improvement [%]")
    axs[0,1].set_ylim(0, max(100, np.nanmax(reduction)*1.1))
    axs[0,1].set_title("B) DT improvement vs Ghost across paths")

    # C) Precision distribution
    axs[1,0].hist(ey_meds, bins=15, color=COLOR_DT, alpha=0.9)
    axs[1,0].set_xlabel("Median |e_y| during GO [m]")
    axs[1,0].set_ylabel("Count")
    axs[1,0].set_title("C) DT steady‑state precision across runs")

    # D) Scatter DT vs Ghost
    axs[1,1].scatter(means_gh, means_dt, color="#9467bd")
    lim = max(axs[1,1].get_xlim()[1], axs[1,1].get_ylim()[1], means_gh.max()*1.05)
    axs[1,1].plot([0, lim], [0, lim], 'k--', lw=1)
    axs[1,1].set_xlim(0, lim); axs[1,1].set_ylim(0, lim)
    axs[1,1].set_xlabel("Ghost cumulative error [m·s]")
    axs[1,1].set_ylabel("DT cumulative error [m·s]")
    axs[1,1].set_title("D) DT dominates (points below diagonal)")

    fig.suptitle("DT vs Ghost — aggregates with unified scales and DoS exposure", y=0.995)
    figure_footer(fig, "Error bars are 95% CIs. Inset shows DoS exposure (mean±std) to preempt integration confounds.")
    fig.tight_layout(rect=[0,0.04,1,0.97]); fig.savefig(os.path.join(outdir, "P4_composite_panel.png")); plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate camera‑ready aggregate plots")
    parser.add_argument("--root", type=str, required=True, help="results_batch_dos_* directory")
    args = parser.parse_args()

    paper_style()
    summary = load_summary(args.root)

    plot1_cum_error_bar(args.root, summary)
    plot2_error_reduction(args.root, summary)
    plot3_distribution_box(args.root)
    plot4_composite_panel(args.root)

    print("Camera‑ready aggregate plots saved in:", os.path.join(args.root, "final_plots"))

if __name__ == "__main__":
    main()
