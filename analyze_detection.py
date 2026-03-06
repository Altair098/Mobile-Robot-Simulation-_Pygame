# analyze_detection.py
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_runs(batch_dir):
    logs = []
    for f in glob.glob(os.path.join(batch_dir, "*", "seed_*", "log.csv")):
        df = pd.read_csv(f)
        df["path"] = f.split(os.sep)[-3]
        logs.append(df)
    return pd.concat(logs, ignore_index=True) if logs else None

def build_labels(df):
    # Positive when FDI is ON (integrity attack); negatives otherwise.
    return (df["fdi"] == 1).astype(int).values

def roc_curve(scores, labels, num=50):
    ths = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), num)
    tpr, fpr = [], []
    for th in ths:
        pred = (scores > th).astype(int)
        TP = np.sum((pred==1) & (labels==1))
        FP = np.sum((pred==1) & (labels==0))
        FN = np.sum((pred==0) & (labels==1))
        TN = np.sum((pred==0) & (labels==0))
        tpr.append(TP / max(1, TP+FN))
        fpr.append(FP / max(1, FP+TN))
    return np.array(fpr), np.array(tpr)

def detection_delay(df, th):
    delays = []
    for path, g in df.groupby(["path"]):
        g = g.sort_values("t")
        in_attack = (g["fdi"].values==1)
        scores = g["r_ema"].values
        times  = g["t"].values
        i = 0
        while i < len(g):
            if in_attack[i]:
                t0 = times[i]
                while i<len(g) and in_attack[i] and scores[i] <= th: i += 1
                if i<len(g) and in_attack[i] and scores[i] > th:
                    delays.append(times[i]-t0)
                while i<len(g) and in_attack[i]: i += 1
            else:
                i += 1
    return np.array(delays)

def main():
    batch_dir = sys.argv[1] if len(sys.argv)>1 else "results_batch_all/<STAMP>"
    df = load_runs(batch_dir)
    assert df is not None, "no runs found"

    labels = build_labels(df)

    # ROC using raw residual and EMA residual
    for score_name in ["r", "r_ema"]:
        fpr, tpr = roc_curve(df[score_name].values, labels)
        plt.figure(figsize=(5,5)); plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'k--',lw=0.8)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({score_name})")
        plt.tight_layout(); plt.savefig(os.path.join(batch_dir, f"roc_{score_name}.png"), dpi=160)

    # Delay at an operating point (pick ~1% FPR on ROC of EMA)
    scores = df["r_ema"].values
    fpr, tpr = roc_curve(scores, labels)
    # choose threshold closest to 1% FPR
    ths = np.linspace(np.percentile(scores,1), np.percentile(scores,99), len(fpr))
    th_star = ths[np.argmin(np.abs(fpr - 0.01))]
    d = detection_delay(df, th_star)
    plt.figure(figsize=(6,4)); plt.hist(d, bins=30)
    plt.xlabel("Detection delay [s]"); plt.ylabel("# windows")
    plt.title(f"Detection delay @ ~1% FPR (th={th_star:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(batch_dir, "delay_hist.png"), dpi=160)

    # Summary CSV
    pd.DataFrame({"delay_s": d}).to_csv(os.path.join(batch_dir,"detection_delays.csv"), index=False)
    print(f"Saved ROC & delay plots in {batch_dir}")

if __name__ == "__main__":
    main()
