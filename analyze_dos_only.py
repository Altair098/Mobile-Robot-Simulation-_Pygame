import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def load_summary(root):
    rows=[]
    for pname in sorted(os.listdir(root)):
        pdir=os.path.join(root,pname)
        f=os.path.join(pdir,"summary.csv")
        if os.path.isfile(f):
            df=pd.read_csv(f)
            if "path" not in df.columns:
                df["path"]=pname
            rows.append(df)
    if not rows: raise RuntimeError(f"No summary.csv in {root}")
    return pd.concat(rows, ignore_index=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="exp_batch_dos",
                    help="batch results directory (default: exp_batch_dos)")
    ap.add_argument("--out", type=str, default="exp_plots_dos",
                    help="output plots directory (default: exp_plots_dos)")
    args=ap.parse_args()

    ensure_dir(args.out)
    df=load_summary(args.root)

    # ------------------------------------------------------------------
    # Aggregate numeric summary (with 95% CI)
    # ------------------------------------------------------------------
    agg = df.groupby("path").agg(
        n_runs=("err_real_mean","size"),
        dt_mean=("err_real_mean","mean"),
        dt_ci=("err_real_mean", lambda s: 1.96*s.std(ddof=1)/np.sqrt(max(len(s),1))),
        dos_mean=("err_dos_mean","mean"),
        dos_ci=("err_dos_mean", lambda s: 1.96*s.std(ddof=1)/np.sqrt(max(len(s),1))),
        time_in_dt=("time_in_dt_ratio","mean")
    ).reset_index()
    agg.to_csv(os.path.join(args.out,"table_per_path.csv"), index=False)
    print("Summary per path:\n", agg.to_string(index=False))

    # ------------------------------------------------------------------
    # Aggregate CDF across all paths
    # ------------------------------------------------------------------
    plt.figure(figsize=(6,4))
    for col, lab in [("err_real_mean","DT Rescue"),("err_dos_mean","Ghost DoS")]:
        vals = df[col].dropna().values
        v = np.sort(vals)
        y = np.linspace(0,1,len(v),endpoint=True)
        plt.plot(v,y,label=lab)
    plt.xlabel("Per-run mean |e| [m]"); plt.ylabel("CDF")
    plt.grid(alpha=0.3); plt.legend()
    plt.title("CDF of per-run mean error — All paths")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"cdf_mean_error_all.png"), dpi=180)

    # ------------------------------------------------------------------
    # Bar plot: mean error per path (with 95% CI)
    # ------------------------------------------------------------------
    labels = agg["path"].tolist()
    x = np.arange(len(labels)); w = 0.36
    plt.figure(figsize=(8,4))
    plt.bar(x-w/2, agg["dt_mean"],  width=w, yerr=agg["dt_ci"],  capsize=3, label="DT Rescue")
    plt.bar(x+w/2, agg["dos_mean"], width=w, yerr=agg["dos_ci"], capsize=3, label="Ghost DoS")
    plt.xticks(x, labels, rotation=10); plt.ylabel("Mean tracking error [m]")
    plt.title("DT-Rescue vs Ghost DoS — per path")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"bar_err_per_path.png"), dpi=180)

    # ------------------------------------------------------------------
    # Per-path CDFs
    # ------------------------------------------------------------------
    for pname, g in df.groupby("path"):
        plt.figure(figsize=(6,4))
        for col, lab in [("err_real_mean","DT Rescue"),("err_dos_mean","Ghost DoS")]:
            vals = g[col].dropna().values
            v = np.sort(vals)
            y = np.linspace(0,1,len(v),endpoint=True)
            plt.plot(v,y,label=lab)
        plt.xlabel("Per-run mean |e| [m]"); plt.ylabel("CDF")
        plt.grid(alpha=0.3); plt.legend()
        plt.title(f"CDF of per-run mean error — {pname}")
        plt.tight_layout(); plt.savefig(os.path.join(args.out,f"cdf_mean_error_{pname}.png"), dpi=180)

    # ------------------------------------------------------------------
    # Time-series plots: median DT run vs ghost DoS for each path
    # ------------------------------------------------------------------
    for pname in sorted(os.listdir(args.root)):
        pdir = os.path.join(args.root,pname)
        sfile = os.path.join(pdir, "summary.csv")
        lfile = os.path.join(pdir, "logs.csv")
        if not (os.path.isfile(sfile) and os.path.isfile(lfile)):
            continue
        S = pd.read_csv(sfile)
        if "err_real_mean" not in S: continue
        # choose run with median DT error
        median_idx = S["err_real_mean"].rank(pct=True).sub(0.5).abs().idxmin()
        med_id = S.loc[median_idx, "run_id"]

        L = pd.read_csv(lfile)
        if "run_id" not in L.columns:
            continue
        L = L[L["run_id"] == med_id]
        if L.empty: continue

        t = L["t"].values
        e_real = np.hypot(L["x"].values - L["xd"].values, L["y"].values - L["yd"].values)
        e_dos  = np.hypot(L["x_dos"].values - L["xd"].values, L["y_dos"].values - L["yd"].values)
        atk = L["dos"].values.astype(int)

        fig, ax = plt.subplots(2,1, figsize=(9,6), sharex=True)
        ax[0].plot(t, e_real, label="|e| DT Rescue"); ax[0].set_ylabel("[m]"); ax[0].legend()
        ax[1].plot(t, e_dos, color="0.25", label="|e| Ghost DoS"); ax[1].set_ylabel("[m]"); ax[1].set_xlabel("time [s]"); ax[1].legend()
        for a in ax:
            a.fill_between(t, 0, 1, where=atk>0, color="0.2", alpha=0.08, transform=a.get_xaxis_transform())
        fig.suptitle(f"Median-run time series — {pname}")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"time_series_median_{pname}.png"), dpi=180)

    print(f"[done] plots → {args.out}")

if __name__=="__main__":
    main()
