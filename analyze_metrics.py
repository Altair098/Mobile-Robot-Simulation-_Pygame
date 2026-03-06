# analyze_metrics.py
import os, sys, csv, math, json
from glob import glob
import numpy as np
import pandas as pd

"""
Usage:
  python analyze_metrics.py results_compare/<STAMP>
Outputs:
  <root>/metrics.csv
  <root>/metrics_summary.csv
Notes:
  Expects each run dir to contain log.csv with columns:
  t,s,attack_dos,attack_fdi,x,y,xd,yd,x_dos,y_dos,x_fdi,y_fdi,mode,v_out,w_out
"""

# ---------- config ----------
SETTLE_ERR_THRESH = 0.10   # meters
SETTLE_WINDOW_N   = 15     # consecutive samples below thresh to count as “stabilized”

def load_log(path):
    rows = []
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return None
    # Convert to arrays
    def arr(key, cast=float):
        return np.array([cast(r[key]) for r in rows])
    out = {
        't':      arr('t'),
        's':      arr('s'),
        'x':      arr('x'),
        'y':      arr('y'),
        'xd':     arr('xd'),
        'yd':     arr('yd'),
        'x_dos':  arr('x_dos'),
        'y_dos':  arr('y_dos'),
        'x_fdi':  arr('x_fdi'),
        'y_fdi':  arr('y_fdi'),
        'v_out':  arr('v_out'),
        'w_out':  arr('w_out'),
        'attack_dos':  np.array([int(r['attack_dos']) for r in rows], dtype=int),
        'attack_fdi':  np.array([int(r['attack_fdi']) for r in rows], dtype=int),
        'mode':   np.array([r['mode'] for r in rows])
    }
    return out

def error_norm(x, y, xd, yd):
    return np.sqrt((x - xd)**2 + (y - yd)**2)

def fraction_true(a):
    return float(np.mean(a.astype(float)))

def time_to_stabilize(t, e, attack_flag, thresh=SETTLE_ERR_THRESH, win_n=SETTLE_WINDOW_N):
    """
    For each attack episode, find first time after attack onset where |e| stays below 'thresh'
    for win_n consecutive samples. Return mean across episodes (np.nan if none).
    """
    if t.size == 0:
        return np.nan
    # Find attack-on to attack-off segments
    onsets = np.where((attack_flag[1:] == 1) & (attack_flag[:-1] == 0))[0] + 1
    offsets = np.where((attack_flag[1:] == 0) & (attack_flag[:-1] == 1))[0] + 1
    # Handle if attack ends at the last sample
    if attack_flag[-1] == 1:
        offsets = np.append(offsets, len(attack_flag)-1)
    if attack_flag[0] == 1:
        onsets = np.insert(onsets, 0, 0)
    if len(onsets) == 0:
        return np.nan

    settles = []
    for k in range(min(len(onsets), len(offsets))):
        i0, i1 = onsets[k], offsets[k]
        # scan from i0 to i1 for first window of win_n below threshold
        idx = np.arange(i0, i1)
        ok = (e[idx] < thresh).astype(int)
        if ok.size < win_n:
            continue
        # rolling sum
        roll = np.convolve(ok, np.ones(win_n, dtype=int), mode='valid')
        hit = np.where(roll == win_n)[0]
        if hit.size > 0:
            j = idx[0] + hit[0] + win_n - 1
            settles.append(t[j] - t[i0])
    return float(np.mean(settles)) if settles else np.nan

def compute_metrics(run_dir):
    log_path = os.path.join(run_dir, "log.csv")
    if not os.path.isfile(log_path):
        return None
    L = load_log(log_path)
    if L is None:
        return None

    t   = L['t'];  s = L['s']
    x,y = L['x'], L['y']
    xd,yd = L['xd'], L['yd']
    e = error_norm(x,y,xd,yd)

    # Ghost baselines (some runs may store NaN—filter those)
    xdos, ydos = L['x_dos'], L['y_dos']
    xfdi, yfdi = L['x_fdi'], L['y_fdi']
    mask_dos = np.isfinite(xdos) & np.isfinite(ydos)
    mask_fdi = np.isfinite(xfdi) & np.isfinite(yfdi)
    e_dos = error_norm(xdos[mask_dos], ydos[mask_dos], xd[mask_dos], yd[mask_dos]) if mask_dos.any() else np.array([])
    e_fdi = error_norm(xfdi[mask_fdi], yfdi[mask_fdi], xd[mask_fdi], yd[mask_fdi]) if mask_fdi.any() else np.array([])

    # Mode / attack flags
    mode = L['mode']
    dt_frac = float(np.mean(mode == 'dt'))
    dos_flag = L['attack_dos']
    fdi_flag = L['attack_fdi']
    any_attack = ((dos_flag + fdi_flag) > 0).astype(int)

    # Completion (proxy)
    completion = float(np.nanmax(s))  # in [0,1]

    # Command stats
    v_out = L['v_out']; w_out = L['w_out']
    dv = np.diff(v_out)/np.diff(t) if len(t) > 1 else np.array([])
    dw = np.diff(w_out)/np.diff(t) if len(t) > 1 else np.array([])

    # Stabilize after attack (on the REAL robot)
    tts = time_to_stabilize(t, e, any_attack)

    # Aggregate
    metrics = {
        "run": os.path.basename(run_dir.rstrip(os.sep)),
        "path_dir": os.path.basename(os.path.dirname(run_dir.rstrip(os.sep))),

        # Tracking (real)
        "e_mean": float(np.nanmean(e)),
        "e_median": float(np.nanmedian(e)),
        "e_max": float(np.nanmax(e)),
        "completion": completion,
        "dt_frac": dt_frac,
        "tts_after_attack": tts,

        # Safety-ish proxy
        "pct_under_10cm": float(np.mean(e < 0.10)),
        "pct_under_20cm": float(np.mean(e < 0.20)),

        # Control effort / smoothness
        "v_mean": float(np.nanmean(np.abs(v_out))),
        "w_mean": float(np.nanmean(np.abs(w_out))),
        "dv_rms": float(np.sqrt(np.nanmean(dv**2))) if dv.size else np.nan,
        "dw_rms": float(np.sqrt(np.nanmean(dw**2))) if dw.size else np.nan,

        # Ghost baselines
        "e_mean_dos": float(np.nanmean(e_dos)) if e_dos.size else np.nan,
        "e_mean_fdi": float(np.nanmean(e_fdi)) if e_fdi.size else np.nan,
        "e_max_dos": float(np.nanmax(e_dos)) if e_dos.size else np.nan,
        "e_max_fdi": float(np.nanmax(e_fdi)) if e_fdi.size else np.nan,
    }
    return metrics

def main(root):
    run_dirs = sorted([d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)])
    rows = []
    for rd in run_dirs:
        m = compute_metrics(rd)
        if m is not None:
            rows.append(m)

    if not rows:
        print("No runs found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "metrics.csv"), index=False)

    # Summary grouped by (path_dir) if present, otherwise whole root
    group_cols = ["path_dir"] if "path_dir" in df.columns else []
    if group_cols:
        g = df.groupby(group_cols).agg(['mean','std','median'])
    else:
        g = df.agg(['mean','std','median']).to_frame().T
    g.to_csv(os.path.join(root, "metrics_summary.csv"))

    print(f"Wrote {os.path.join(root, 'metrics.csv')}")
    print(f"Wrote {os.path.join(root, 'metrics_summary.csv')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics.py <results_root>")
        sys.exit(1)
    main(sys.argv[1])
