# run_dos_only.py
import os, math, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib.lines import Line2D

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

def path_figure8(N=2800, a=1.0):  # horizontal ∞
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

# ----------------------------- PLOT SIZE HELPER -----------------------------
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

# ------------------------------- SIMULATION --------------------------------
def run_one(path_name:str, outroot:str, rng,
            Lmin=5.0, Lmax=9.0, Gmin=6.0, Gmax=12.0):
    path_name = path_name.strip().lower()

    waypoints = build_waypoints(path_name)
    closed = (path_name in {"circle","figure8"})
    traj = GenericTrajectory.from_waypoints(waypoints, closed=closed, speed=lambda_v)

    # initial pose at s=0 aligned with tangent
    x0, y0 = float(waypoints[0,0]), float(waypoints[0,1])
    t_hat, _ = traj.tangent_normal(0.0)
    th0 = float(math.atan2(t_hat[1], t_hat[0]))

    real = Robot(); ghost = Robot()
    try:
        real.set_pose(x0, y0, th0); ghost.set_pose(x0, y0, th0)
    except TypeError:
        real.set_pose((x0, y0, th0)); ghost.set_pose((x0, y0, th0))

    ctrl = DTRescueController(real, traj, dt=dt)
    ctrl.s = float(traj.nearest_s(x0, y0))
    ctrl.x_hat = np.array([x0, y0, th0], float)

    d_hat = float(initial_depth)

    windows = random_dos_schedule(total_time, Lmin, Lmax, Gmin, Gmax, rng=rng)

    outdir = os.path.join(outroot, path_name)
    os.makedirs(outdir, exist_ok=True)
    log = []

    s_phase = ctrl.s
    steps = int(total_time/dt)
    tail_threshold = 0.995  # pin open-path end but do NOT zero commands

    for k in range(steps):
        tcur = k*dt
        attack = dos_flag(tcur, windows)

        # --- phase advancement (original policy) ---
        xr, yr, thr = real.get_pose()
        s_near = traj.nearest_s_local(xr, yr, s_phase, max_jump=0.08)

        x_ref, _, _ = traj.sample(s_phase)        # [x, y, theta]
        theta_path = float(x_ref[2])
        c_t = math.cos(theta_path)*math.cos(thr) + math.sin(theta_path)*math.sin(thr)
        if c_t < 0.2:
            s_near = min(s_near, s_phase)

        ds = max(0.0, s_near - s_phase)
        s_phase += 0.30 * ds
        s_phase = float(np.clip(s_phase, 0.0, 0.999))

        if (not closed) and (s_phase >= tail_threshold):
            s_phase = float(tail_threshold)

        ctrl.s = s_phase

        # --- reference generation (heading, curvature, feedforward) ---
        x_d, _, _ = traj.sample(s_phase)          # pose with heading in x_d[2]
        theta_d = float(x_d[2])

        kappa = float(traj.curvature(s_phase))
        v_d_star = min(lambda_v, lambda_omega/(abs(kappa) + 1e-6))
        omega_d = v_d_star * kappa

        # --- main control (body frame errors) ---
        dx = x_d[0] - xr; dy = x_d[1] - yr
        e_x =  math.cos(thr)*dx + math.sin(thr)*dy
        e_y = -math.sin(thr)*dx + math.cos(thr)*dy
        e_th = (theta_d - thr + math.pi) % (2*math.pi) - math.pi

        v_cmd, w_cmd = compute_control(e_x, e_y, e_th, v_d_star, omega_d, d_hat, gains)
        v_cmd = float(np.clip(v_cmd, -lambda_v, lambda_v))
        w_cmd = float(np.clip(w_cmd, -lambda_omega, lambda_omega))
        u_main = np.array([v_cmd, w_cmd], float)

        # --- DT arbitration ---
        u_out = ctrl.step(tcur, attack, np.array([xr, yr, thr], float), u_main)

        # apply to real plant
        real.update(float(u_out[0]), float(u_out[1]), dt)

        # Ghost DoS baseline: actuator drop during attack
        u_dos = np.array([0.0, 0.0], float) if attack else u_main.copy()
        ghost.update(float(u_dos[0]), float(u_dos[1]), dt)

        # --- runner-side depth adaptation: ONLY in MAIN mode ---
        if ctrl.mode == 'main':
            d_hat = update_depth_estimate(
                d_hat, e_x, e_y, e_th, v_d_star,
                gamma, gains['k3'], d_min, d_max, dt
            )

        # log
        xr2, yr2, _ = real.get_pose()
        xdos, ydos, _ = ghost.get_pose()
        log.append({
            "t": tcur, "s": s_phase, "attack_dos": int(attack),
            "x": xr2, "y": yr2, "xd": x_d[0], "yd": x_d[1],
            "x_dos": xdos, "y_dos": ydos,
            "mode": ctrl.mode, "v_out": float(u_out[0]), "w_out": float(u_out[1])
        })

    # ---------- SAVE ----------
    with open(os.path.join(outdir, "log.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        wcsv.writeheader(); wcsv.writerows(log)

    t_arr = np.array([r["t"] for r in log])
    x     = np.array([r["x"] for r in log]);      y    = np.array([r["y"] for r in log])
    xd    = np.array([r["xd"] for r in log]);     yd   = np.array([r["yd"] for r in log])
    xdos  = np.array([r["x_dos"] for r in log]);  ydos = np.array([r["y_dos"] for r in log])
    atk   = np.array([r["attack_dos"] for r in log], int)
    v_out = np.array([r["v_out"] for r in log], float)
    w_out = np.array([r["w_out"] for r in log], float)
    mode  = np.array([1 if r["mode"] == "dt" else 0 for r in log], int)  # 1=DT

    # XY trajectories
    fig_w, fig_h, x_margin, y_margin = compute_plot_params(waypoints, path_name)
    plt.figure(figsize=(fig_w, fig_h), dpi=180)
    plt.plot(xd, yd, linestyle=":", color="#2ca02c", lw=2.5, label="reference")
    for i in range(len(x)-1):
        c = "#d62728" if (mode[i] == 1 and mode[i+1] == 1) else "#1f77b4"
        plt.plot(x[i:i+2], y[i:i+2], color=c, lw=3.0)
    plt.plot(xdos, ydos, color="0.25", lw=2.2, label="ghost no-DT")
    x_min, x_max = xd.min(), xd.max()
    y_min, y_max = yd.min(), yd.max()
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.gca().set_aspect("equal", adjustable="box")
    legend_handles = [
        Line2D([0],[0], color="#2ca02c", lw=2.5, ls=":", label="reference"),
        Line2D([0],[0], color="#1f77b4", lw=3.0, label="Main controller"),
        Line2D([0],[0], color="#d62728", lw=3.0, label="DT takeover"),
        Line2D([0],[0], color="0.25",   lw=2.2, label="ghost no-DT"),
    ]
    plt.legend(handles=legend_handles, frameon=False, fontsize=12)
    plt.title(f"{path_name} — trajectories", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_xy.png"))
    plt.close()

    # Errors with axis-transform shading
    e_real = np.hypot(x - xd, y - yd)
    e_dos  = np.hypot(xdos - xd, ydos - yd)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), dpi=180, sharex=True)
    ax[0].plot(t_arr, e_real, color="#1f77b4", lw=2.0, label="|e| DT-Rescue")
    ax[1].plot(t_arr, e_dos,  color="0.25",   lw=2.0, label="|e| Ghost DoS")
    ax[0].set_ylabel("error [m]", fontsize=12)
    ax[1].set_ylabel("error [m]", fontsize=12)
    ax[1].set_xlabel("time [s]", fontsize=12)
    for a in ax:
        a.legend(loc="upper right", frameon=False)
        a.fill_between(t_arr, 0, 1, where=atk>0, color="0.1", alpha=0.10,
                       transform=a.get_xaxis_transform())
        a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "compare_errors.png"))
    plt.close(fig)

    # Commands + mode with axis-transform shading
    fig2, ax2 = plt.subplots(3, 1, figsize=(12, 8.5), dpi=180, sharex=True)
    ax2[0].plot(t_arr, v_out, lw=1.8, label="v_out"); ax2[0].set_ylabel("v [m/s]"); ax2[0].legend(frameon=False)
    ax2[1].plot(t_arr, w_out, lw=1.8, label="w_out"); ax2[1].set_ylabel("w [rad/s]"); ax2[1].legend(frameon=False)
    ax2[2].step(t_arr, mode, where="post", lw=1.5); ax2[2].set_ylabel("mode (1=DT)"); ax2[2].set_xlabel("time [s]")
    for a in ax2:
        a.fill_between(t_arr, 0, 1, where=atk>0, color="0.1", alpha=0.08,
                       transform=a.get_xaxis_transform())
        a.grid(alpha=0.25)
    fig2.tight_layout(); fig2.savefig(os.path.join(outdir, "compare_cmds_mode.png"))
    plt.close(fig2)

    # Save DoS windows
    with open(os.path.join(outdir, "dos_windows.txt"), "w") as f:
        for (a,b) in windows: f.write(f"{a:.3f},{b:.3f}\n")

    print(f"Saved: {outdir}")

# ---------------------------------- MAIN -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=str, default="circle,figure8,scurve,sharpl",
                    help="comma list or 'all'")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for DoS schedule")
    ap.add_argument("--out", type=str, default="demo2")
    ap.add_argument("--dos-Lmin", type=float, default=5.0)
    ap.add_argument("--dos-Lmax", type=float, default=9.0)
    ap.add_argument("--dos-Gmin", type=float, default=6.0)
    ap.add_argument("--dos-Gmax", type=float, default=12.0)
    args = ap.parse_args()

    if args.paths.lower().strip() == "all":
        paths = ["circle","figure8","scurve","sharpl"]
    else:
        paths = [p.strip().lower() for p in args.paths.split(",") if p.strip()]

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # RNG advances once per path for random DoS windows; runs are reproducible
    # for a given seed and path order.
    for p in paths:
        run_one(p, args.out, rng, args.dos_Lmin, args.dos_Lmax, args.dos_Gmin, args.dos_Gmax)

if __name__ == "__main__":
    main()
