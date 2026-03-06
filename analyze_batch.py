# analyze_batch.py
"""
Usage (all optional now):
  python analyze_batch.py                # root=exp_batch, out=exp_plots
  python analyze_batch.py --root exp_batch --out exp_plots
"""
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
    if not rows:
        raise RuntimeError(f"No summary.csv files found under '{root}'.")
    return pd.concat(rows, ignore_index=True)

def cdf_plot(vals, title, fn, xlabel, outdir):
    v=np.sort(np.asarray(vals)); y=np.linspace(0,1,len(v),endpoint=True)
    plt.figure(figsize=(6,5)); plt.plot(v,y)
    plt.xlabel(xlabel); plt.ylabel("CDF"); plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, fn), dpi=180); plt.close()

def main():
    ap=argparse.ArgumentParser(add_help=True)
    ap.add_argument("--root", type=str, default="exp_batch",
                    help="folder created by batch_expts_new.py (default: exp_batch)")
    ap.add_argument("--out", type=str, default="exp_plots",
                    help="where to save plots (default: exp_plots)")
    args=ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"[error] root folder not found: {args.root}")

    ensure_dir(args.out)
    df=load_summary(args.root)

    # Per-path boxplots
    for pname,g in df.groupby("path"):
        plt.figure(figsize=(6,5))
        data=[g["err_real_mean"], g["err_dos_mean"], g["err_fdi_mean"]]
        plt.boxplot(data, labels=["DT Rescue","Ghost DoS","Ghost FDI"],
                    showmeans=True, meanprops=dict(marker="^"))
        plt.ylabel("Mean tracking error [m]")
        plt.title(f"Per-run mean error — {pname}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"box_err_{pname}.png"), dpi=180)
        plt.close()

    # CDFs across all paths
    cdf_plot(df["err_real_mean"], "CDF of mean tracking error — DT Rescue (all paths)",
             "cdf_err_real.png", "err real mean", args.out)
    cdf_plot(df["err_dos_mean"],  "CDF of mean tracking error — Ghost DoS (all paths)",
             "cdf_err_dos.png",  "err dos mean",  args.out)
    cdf_plot(df["err_fdi_mean"],  "CDF of mean tracking error — Ghost FDI (all paths)",
             "cdf_err_fdi.png",  "err fdi mean",  args.out)

    # Aggregates per path
    agg=df.groupby("path").agg(
        time_in_dt_ratio=("time_in_dt_ratio","mean"),
        tube_violations=("tube_violations","mean"),
        err_real_median=("err_real_mean","median"),
    ).reset_index()

    plt.figure(figsize=(10,5)); plt.bar(agg["path"], agg["time_in_dt_ratio"])
    plt.ylabel("time in dt ratio"); plt.title("Mean DT usage per path")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"bar_dt_ratio.png"), dpi=180); plt.close()

    plt.figure(figsize=(10,5)); plt.bar(agg["path"], agg["err_real_median"])
    plt.ylabel("err real mean"); plt.title("Median mean error per path — DT Rescue")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"bar_err_real.png"), dpi=180); plt.close()

    plt.figure(figsize=(10,5)); plt.bar(agg["path"], agg["tube_violations"])
    plt.ylabel("tube violations"); plt.title("Mean tube violations per path")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"bar_tube_viol.png"), dpi=180); plt.close()

    print(f"[done] plots → {args.out}")

if __name__=="__main__":
    main()
