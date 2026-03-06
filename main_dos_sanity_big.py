# batch_dos_ablation.py
# Batch evaluation: DT Rescue vs Ghost DoS under randomized DoS windows.
# Produces per-path logs and 7 MVP validation plots with publication-ready styling.

import os, math, csv, argparse, json, time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.interpolate import splprep, splev

from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from generic_trajectory import GenericTrajectory
from dt_rescue_dos import DTRescueController
from config import (
    gains, gamma, d_min, d_max, initial_depth,
    lambda_v, lambda_omega, dt, total_time
)

# ----------------------------- PATH GENERATORS -----------------------------
def path_circle(N=2400, R=1.2):
    t = np.linspace(0, 2*np.pi, N, endpoint=True)
    return np.c_[R*np.cos(t), R*np.sin(t)]

def path_figure8(N=2800, a=1.0):
    t = np.linspace(0, 2*np.pi, N, endpoint=True)
    x = a*np.sin(t)
    y = a*np.sin(t)*np.cos(t)
    return np.c_[x, y]

def path_s_curve(N=2500):
    pts = np.array([
        [0.50, 1.00],[0.30, 0.95],[0.20, 0.85],[0.20, 0.75],[0.30, 0.65],
        [0.50, 0.60],[0.70, 0.55],[0.80, 0.45],[0.80, 0.35],[0.70, 0.25],
        [0.50, 0.20],[0.30, 0.15],[0.50, 0.00],
    ])
    tck, _ = splprep([pts[:,0], pts[:,1]], s=0, k=3)
    u = np.linspace(0, 1, N)
    x, y = splev(u, tck)
    return np.c_[np.array(x), np.array(y)]

def path_sharpL(Nh=1200, Nv=1200):
    x1 = np.linspace(-2.5, 1.0, Nh); y1 = np.full_like(x1, 1.0)
    y2 = np.linspace(1.0, -2.5, Nv); x2 = np.full_like(y2, 1.0)
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    return np.c_[x, y]

def build_waypoints(name:str) -> np.ndarray:
    n = name.lower()
    if n == "circle":   return path_circle()
    if n == "figure8":  return path_figure8()
    if n == "scurve":   return path_s_curve()
    if n == "sharpl":   return path_sharpL()
    raise ValueError(f"unknown path '{name}'")

# ----------------------------- STYLE HELPERS -------------------------------
def paper_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 180,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.25
    })

def compute_plot_params(waypoints, path_name):
    x_min, x_max = waypoints[:, 0].min(), waypoints[:, 0].max()
    y_min, y_max = waypoints[:, 1].min(), waypoints[:, 1].max()
    xr = x_max - x_min; yr = y_max - y_min
    x_margin = 0.1 * xr if xr > 0.01 else 0.1
    y_margin = 0.1 * yr if yr > 0.01 else 0.1
    aspect = yr / xr if xr > 0.01 else 1.0
    if path_name.lower() == "scurve":
        fig_w, fig_h = 8.0, 14.0
    elif path_name.lower() == "figure8":
        fig_w, fig_h = 12.0, 8.0
    else:
        if aspect > 1.5:
            fig_w, fig_h = 8.0, min(14.0, 8.0*aspect)
        elif aspect < 0.67:
            fig_w, fig_h = min(14.0, 8.0/aspect), 8.0
        else:
            fig_w = fig_h = 9.5
    return fig_w, fig_h, x_margin, y_margin

# ----------------------------- DoS SCHEDULER ------------------------------
def random_dos_schedule(T, Lmin=5.0, Lmax=9.0, Gmin=6.0, Gmax=12.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    windows, t = [], rng.uniform(Gmin, Gmax)
    while t < T:
        L = rng.uniform(Lmin, Lmax)
        a, b = t, min(t+L, T)
        windows.append((a, b))
        t = b + rng.uniform(Gmin, Gmax)
    return windows

def dos_flag(t, windows):
    return any(a <= t <= b for (a,b) in windows)

# ----------------------------- SIMULATION CORE -----------------------------
def simulate_once(path_name, rng, dos_params, outdir_run):
    # Build trajectory
    waypoints = build_waypoints(path_name)
    closed = (path_name in {"circle","figure8"})
    traj = GenericTrajectory.from_waypoints(waypoints, closed=closed, speed=lambda_v)

    # Initial pose at s=0 aligned with tangent
    x0, y0 = float(waypoints[0,0]), float(waypoints[0,1])
    t_hat,_ = traj.tangent_normal(0.0)
    th0 = float(math.atan2(t_hat[1], t_hat[0]))

    # Robots
    real = Robot(); ghost = Robot()
    try:
        real.set_pose(x0, y0, th0); ghost.set_pose(x0, y0, th0)
    except TypeError:
        real.set_pose((x0, y0, th0)); ghost.set_pose((x0, y0, th0))

    # Controller
    ctrl = DTRescueController(real, traj, dt=dt)
    ctrl.s = float(traj.nearest_s(x0, y0))
    ctrl.x_hat = np.array([x0, y0, th0], float)

    d_hat = float(initial_depth)

    # DoS schedule
    Lmin, Lmax, Gmin, Gmax = dos_params
    windows = random_dos_schedule(total_time, Lmin, Lmax, Gmin, Gmax, rng=rng)

    s_phase = ctrl.s
    steps = int(total_time/dt)
    tail_threshold = 0.995

    # Logs for MVP plots
    T, X, Y, XD, YD = [], [], [], [], []
    XDOS, YDOS = [], []
    V_OUT, W_OUT = [], []
    MODE = []
    ATK = []
    # Diagnostics for MVP: ct, safety tube, CBF ratio placeholder
    CT = []          # alignment cosine
    V_DES, V_PROJ = [], []  # desired vs projected v for CBF ratio (if exposed)
    EY = []          # perpendicular error magnitude

    for k in range(steps):
        tcur = k*dt
        attack = dos_flag(tcur, windows)

        xr, yr, thr = real.get_pose()

        # Phase advancement: forward-only with cosine gate and low-pass gain 0.30
        s_near = traj.nearest_s_local(xr, yr, s_phase, max_jump=0.08)
        x_ref, _, _ = traj.sample(s_phase)
        theta_path = float(x_ref[2])
        c_t = math.cos(theta_path)*math.cos(thr) + math.sin(theta_path)*math.sin(thr)
        if c_t < 0.2:
            s_near = min(s_near, s_phase)
        ds = max(0.0, s_near - s_phase)
        s_phase += 0.30*ds
        s_phase = float(np.clip(s_phase, 0.0, 0.999))
        if (not closed) and (s_phase >= tail_threshold):
            s_phase = float(tail_threshold)
        ctrl.s = s_phase

        # Reference and feedforward
        x_d, tvec, _ = traj.sample(s_phase)
        theta_d = float(x_d[2])
        kappa = float(traj.curvature(s_phase))
        v_d_star = min(lambda_v, lambda_omega/(abs(kappa)+1e-6))
        omega_d = v_d_star*kappa

        # Errors and main command
        dx = x_d[0]-xr; dy = x_d[1]-yr
        e_x =  math.cos(thr)*dx + math.sin(thr)*dy
        e_y = -math.sin(thr)*dx + math.cos(thr)*dy
        e_th = (theta_d - thr + math.pi) % (2*math.pi) - math.pi
        v_cmd, w_cmd = compute_control(e_x, e_y, e_th, v_d_star, omega_d, d_hat, gains)
        v_cmd = float(np.clip(v_cmd, -lambda_v, lambda_v))
        w_cmd = float(np.clip(w_cmd, -lambda_omega, lambda_omega))
        u_main = np.array([v_cmd, w_cmd], float)

        # DT rescue arbitration
        u_out = ctrl.step(tcur, attack, np.array([xr,yr,thr], float), u_main)
        real.update(float(u_out[0]), float(u_out[1]), dt)

        # Ghost baseline
        u_dos = np.array([0.0, 0.0], float) if attack else u_main.copy()
        ghost.update(float(u_dos[0]), float(u_dos[1]), dt)

        # Runner-side depth adaptation only in main mode
        if ctrl.mode == 'main':
            d_hat = update_depth_estimate(
                d_hat, e_x, e_y, e_th, v_d_star, gamma, gains['k3'], d_min, d_max, dt
            )

        # Log arrays
        xr2, yr2, _ = real.get_pose()
        xdos, ydos, _ = ghost.get_pose()

        T.append(tcur); X.append(xr2); Y.append(yr2)
        XD.append(x_d[0]); YD.append(x_d[1])
        XDOS.append(xdos); YDOS.append(ydos)
        V_OUT.append(float(u_out[0])); W_OUT.append(float(u_out[1]))
        MODE.append(1 if ctrl.mode=="dt" else 0); ATK.append(int(attack))

        # Diagnostics
        CT.append(c_t)
        # Approximate lateral error magnitude using path normal from tangent
        nx, ny = -tvec[1], tvec[0]
        EY.append(abs(nx*dx + ny*dy))
        # If dt controller exposes CBF ratio, capture; otherwise treat projected=commanded v
        V_DES.append(v_cmd)
        V_PROJ.append(float(u_out[0]) if ctrl.mode=="dt" else v_cmd)

    # Convert to arrays
    T = np.array(T); X=np.array(X); Y=np.array(Y); XD=np.array(XD); YD=np.array(YD)
    XDOS=np.array(XDOS); YDOS=np.array(YDOS)
    V_OUT=np.array(V_OUT); W_OUT=np.array(W_OUT)
    MODE=np.array(MODE); ATK=np.array(ATK)
    CT=np.array(CT); EY=np.array(EY); V_DES=np.array(V_DES); V_PROJ=np.array(V_PROJ)

    # Cumulative errors
    err_dt = np.hypot(X-XD, Y-YD).sum()*dt
    err_ghost = np.hypot(XDOS-XD, YDOS-YD).sum()*dt

    # Save per-run CSV
    os.makedirs(outdir_run, exist_ok=True)
    with open(os.path.join(outdir_run, "log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","x","y","xd","yd","x_dos","y_dos","mode","attack","v_out","w_out","ct","ey","v_des","v_proj"])
        for i in range(len(T)):
            w.writerow([T[i],X[i],Y[i],XD[i],YD[i],XDOS[i],YDOS[i],MODE[i],ATK[i],V_OUT[i],W_OUT[i],CT[i],EY[i],V_DES[i],V_PROJ[i]])

    return {
        "T":T, "X":X, "Y":Y, "XD":XD, "YD":YD, "XDOS":XDOS, "YDOS":YDOS,
        "V_OUT":V_OUT, "W_OUT":W_OUT, "MODE":MODE, "ATK":ATK, "CT":CT,
        "EY":EY, "V_DES":V_DES, "V_PROJ":V_PROJ, "err_dt":err_dt, "err_ghost":err_ghost,
        "waypoints":waypoints
    }

# ----------------------------- PLOTTING (7 MVP) ----------------------------
def plot_mvp(path_name, run_data, outdir_path):
    paper_style()
    T = run_data["T"]; ATK=run_data["ATK"]; MODE=run_data["MODE"]

    # 1) Safety Tube Violation: d_perp vs ±r_eff (if tube radius available; use percentile as proxy)
    EY = run_data["EY"]
    r_eff = max(1e-6, np.percentile(EY, 95))  # proxy bound for visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(T, EY, label="|d_perp|", color="#1f77b4")
    ax.hlines([r_eff, -r_eff], T[0], T[-1], colors=["#d62728","#d62728"], linestyles="--", label="±r_eff")
    ax.fill_between(T, 0, 1, where=ATK>0, color="0.1", alpha=0.10, transform=ax.get_xaxis_transform())
    ax.set_title("Safety Tube Violation — d_perp vs ±r_eff")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Distance [m]")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(outdir_path, "mvp1_safety_tube.png")); plt.close(fig)

    # 2) Control Signal Smoothness: u_main vs u_dt vs u_out (here we show v_out and w_out; main not logged per-step; approximate with V_DES)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(T, run_data["V_DES"], label="v_main (desired)", color="#2ca02c", alpha=0.9)
    ax[0].plot(T, run_data["V_OUT"], label="v_out (blended)", color="#1f77b4")
    ax[1].plot(T, run_data["W_OUT"], label="w_out (blended)", color="#1f77b4")
    for a in ax:
        a.fill_between(T, 0, 1, where=ATK>0, color="0.1", alpha=0.10, transform=a.get_xaxis_transform())
        a.legend(frameon=False); a.grid(alpha=0.25)
    ax[0].set_ylabel("v [m/s]"); ax[1].set_ylabel("ω [rad/s]"); ax[1].set_xlabel("Time [s]")
    ax[0].set_title("Control Signal Smoothness — Washout Handover")
    fig.tight_layout(); fig.savefig(os.path.join(outdir_path, "mvp2_washout_smoothness.png")); plt.close(fig)

    # 3) DoS Attack Recovery Trajectories: Reference vs Ghost vs DT
    fig_w, fig_h, x_margin, y_margin = compute_plot_params(run_data["waypoints"], path_name)
    plt.figure(figsize=(fig_w, fig_h))
    plt.plot(run_data["XD"], run_data["YD"], linestyle=":", color="#2ca02c", lw=2.5, label="Reference")
    # DT colored by mode
    X, Y = run_data["X"], run_data["Y"]
    for i in range(len(X)-1):
        c = "#d62728" if (MODE[i]==1 and MODE[i+1]==1) else "#1f77b4"
        plt.plot(X[i:i+2], Y[i:i+2], color=c, lw=3.0)
    plt.plot(run_data["XDOS"], run_data["YDOS"], color="0.25", lw=2.2, label="Ghost DoS")
    x_min, x_max = run_data["XD"].min(), run_data["XD"].max()
    y_min, y_max = run_data["YD"].min(), run_data["YD"].max()
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(frameon=False)
    plt.title("DoS Attack Recovery Trajectories")
    plt.tight_layout(); plt.savefig(os.path.join(outdir_path, "mvp3_recovery_xy.png")); plt.close()

    # 4) FSM State and Alignment: c_t with TURN/GO shading (mode is a proxy for DT active; TURN periods when MODE==1 and c_t < threshold)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(T, run_data["CT"], label="c_t (alignment cosine)", color="#1f77b4")
    # TURN shading: DT active and c_t below 0.6 (proxy threshold)
    turn = (MODE==1) & (run_data["CT"] < 0.6)
    ax.fill_between(T, 0, 1, where=turn, color="#d62728", alpha=0.15, transform=ax.get_xaxis_transform(), label="TURN")
    go = (MODE==1) & (run_data["CT"] >= 0.6)
    ax.fill_between(T, 0, 1, where=go, color="#2ca02c", alpha=0.10, transform=ax.get_xaxis_transform(), label="GO")
    ax.fill_between(T, 0, 1, where=ATK>0, color="0.1", alpha=0.08, transform=ax.get_xaxis_transform(), label="DoS")
    ax.set_title("FSM Alignment Progression — TURN→GO")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("c_t")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(outdir_path, "mvp4_fsm_alignment.png")); plt.close(fig)

    # 5) CBF Intervention Rate: v_proj/v_des (1.0 ideal)
    ratio = np.divide(run_data["V_PROJ"], np.maximum(1e-6, run_data["V_DES"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(T, ratio, label="v_proj / v_des", color="#1f77b4")
    ax.hlines(1.0, T[0], T[-1], colors="#2ca02c", linestyles="--", label="ideal")
    ax.fill_between(T, 0, 1, where=ATK>0, color="0.1", alpha=0.10, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, max(1.2, ratio.max()*1.05))
    ax.set_title("CBF Intervention Rate")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Ratio")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(outdir_path, "mvp5_cbf_ratio.png")); plt.close(fig)

    # 6) Precision during GO: |e_y|
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(T, EY, color="#1f77b4", label="|e_y| (perp error)")
    ax.fill_between(T, 0, 1, where=ATK>0, color="0.1", alpha=0.10, transform=ax.get_xaxis_transform())
    ax.set_title("Precision during GO — Perpendicular Error")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("|e_y| [m]")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(outdir_path, "mvp6_precision_ey.png")); plt.close(fig)

    # 7) Comparative Error: DT vs Ghost
    e_dt = np.hypot(run_data["X"]-run_data["XD"], run_data["Y"]-run_data["YD"])
    e_ghost = np.hypot(run_data["XDOS"]-run_data["XD"], run_data["YDOS"]-run_data["YD"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(T, e_dt, label="DT Rescue", color="#1f77b4")
    ax.plot(T, e_ghost, label="Ghost DoS", color="0.25")
    ax.fill_between(T, 0, 1, where=ATK>0, color="0.1", alpha=0.10, transform=ax.get_xaxis_transform())
    ax.set_title("Comparative Error During DoS Attack")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("|error| [m]")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(outdir_path, "mvp7_comparative_error.png")); plt.close(fig)

def plot_path_aggregates(path_name, all_runs, outdir_path):
    paper_style()
    # Box plots for cumulative error across runs
    errs_dt = [r["err_dt"] for r in all_runs]
    errs_gh = [r["err_ghost"] for r in all_runs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([errs_dt, errs_gh], labels=["DT Rescue", "Ghost DoS"], showmeans=True)
    ax.set_title(f"Cumulative Error over 10 Runs — {path_name}")
    ax.set_ylabel("Integrated error [m·s]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir_path, "aggregate_cum_error_box.png")); plt.close(fig)

    # Bar chart with mean ± std
    means = [np.mean(errs_dt), np.mean(errs_gh)]
    stds  = [np.std(errs_dt),  np.std(errs_gh)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    ax.bar(x, means, yerr=stds, color=["#1f77b4", "0.25"], alpha=0.9, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(["DT Rescue", "Ghost DoS"])
    ax.set_title(f"Cumulative Error Mean±Std — {path_name}")
    ax.set_ylabel("Integrated error [m·s]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir_path, "aggregate_cum_error_bar.png")); plt.close(fig)

# ----------------------------- BATCH DRIVER --------------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch DoS Ablation: DT vs Ghost")
    ap.add_argument("--paths", type=str, default="circle,figure8,scurve,sharpl")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dos-Lmin", type=float, default=5.0)
    ap.add_argument("--dos-Lmax", type=float, default=9.0)
    ap.add_argument("--dos-Gmin", type=float, default=6.0)
    ap.add_argument("--dos-Gmax", type=float, default=12.0)
    args = ap.parse_args()

    # Output root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out or f"results_batch_dos_{stamp}"
    os.makedirs(out_root, exist_ok=True)

    # Normalize paths
    if args.paths.strip().lower() == "all":
        paths = ["circle","figure8","scurve","sharpl"]
    else:
        paths = [p.strip().lower() for p in args.paths.split(",") if p.strip()]

    rng = np.random.default_rng(args.seed)
    dos_params = (args.dos_Lmin, args.dos_Lmax, args.dos_Gmin, args.dos_Gmax)

    summary = {}
    for pname in paths:
        path_dir = os.path.join(out_root, pname)
        os.makedirs(path_dir, exist_ok=True)
        all_runs = []
        for r in range(args.runs):
            run_dir = os.path.join(path_dir, f"run_{r:02d}")
            data = simulate_once(pname, rng, dos_params, run_dir)
            all_runs.append(data)
            # MVP 7 plots for first run only (representative) to avoid clutter
            if r == 0:
                plot_mvp(pname, data, run_dir)
        # Aggregates
        plot_path_aggregates(pname, all_runs, path_dir)
        summary[pname] = {
            "err_dt": [float(d["err_dt"]) for d in all_runs],
            "err_ghost": [float(d["err_ghost"]) for d in all_runs]
        }

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved batch results to: {out_root}")

if __name__ == "__main__":
    main()
