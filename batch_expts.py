# batch_expts.py
"""
Headless batch experiments for DT-rescue vs baselines (ghost DoS / ghost FDI).

Examples
--------
python batch_expts.py --paths_dir paths --runs_per_path 10 --T 120 --seed 0
python batch_expts.py --paths_dir paths --paths figure8,sharpL --runs_per_path 5 --T 90 --dt_batch 0.05 --no_xy

Outputs
-------
<outdir>/<stamp>/<path_name>/seed_<k>/{log.csv, summary.json, xy.png (optional)}
<outdir>/<stamp>/summary.csv
"""
import os, json, math, csv, argparse, random, sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ---- path discovery helpers (works whether helpers are in /paths or project root)
def _import_paths_helpers():
    try:
        from paths.paths_utlis import discover_paths, ensure_circle
    except Exception:
        from paths.paths_utlis import discover_paths, ensure_circle
    return discover_paths, ensure_circle

discover_paths, ensure_circle = _import_paths_helpers()

# ---- project modules
from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from generic_trajectory import GenericTrajectory
from dt_rescue3 import DTRescueController
from attack_model import FDIInjector
import config as CFG

# ---------------- built-in paths ----------------
def builtin_waypoints(name: str) -> np.ndarray:
    if name == "figure8":
        t = np.linspace(0, 2*np.pi, 240)
        x = -0.8 + 0.9*np.sin(t); y = 1.6 + 0.7*np.sin(t)*np.cos(t)
        return np.c_[x, y]
    if name == "sharpL":
        return np.array([
            (-2.8,2.8),(-1.2,2.8),(-0.2,2.6),(0.4,2.2),(0.8,1.6),(1.0,1.0),
            (1.0,0.0),(1.0,-0.6),(0.9,-1.2),(0.6,-1.8),(0.2,-2.2),(-0.4,-2.5)
        ], float)
    if name == "sCurve":
        t = np.linspace(0, 1, 220)
        x = -2.8 + 4.0*t; y = 2.8 - 3.6*np.sin(1.2*np.pi*t)*np.exp(-0.1*t)
        return np.c_[x, y]
    raise ValueError(f"unknown built-in path: {name}")

# --------------- attack schedule ---------------
def make_attack_windows(T, rng, min_len=6.0, max_len=10.0, gap=8.0, p_start=0.5):
    wins, t = [], gap * rng.uniform(0.3, 1.0)
    while t < T - min_len:
        if rng.random() < p_start:
            L = rng.uniform(min_len, max_len)
            wins.append((t, min(T, t+L)))
            t += L
        t += rng.uniform(gap*0.6, gap*1.4)
    return wins

def in_windows(t, wins):
    return any(a <= t <= b for a, b in wins)

# --------------- tube diag (logging only) ---------------
R0, A_KAPPA, B_ALIGN = 0.16, 0.90, 0.30
def tube_state(traj, s, pose, c_t):
    x_d, _, _ = traj.sample(s)
    th = pose[2]; th_ref = float(x_d[2])
    tx, ty = math.cos(th_ref), math.sin(th_ref)
    nx, ny = -ty, tx
    d_perp = (pose[0]-x_d[0])*nx + (pose[1]-x_d[1])*ny
    kappa  = float(traj.curvature(s))
    r_eff  = R0 * (1.0 + A_KAPPA*abs(kappa)) * (1.0 - B_ALIGN*max(0.0, c_t))
    r_eff  = max(0.75*R0, min(r_eff, 2.0*R0))
    return d_perp, r_eff

def ensure_dir(d): os.makedirs(d, exist_ok=True)

# ---------------- one run ----------------
def run_once(waypoints, outdir_run, seed=0, T=120.0, dt=CFG.dt,
             dos_windows=None, fdi_windows=None, save_xy=True):
    rng = random.Random(seed)
    traj = GenericTrajectory.from_waypoints(waypoints, closed=False, speed=CFG.lambda_v)

    # robots
    real, ghost_dos, ghost_fdi = Robot(), Robot(), Robot()

    x0, y0 = float(waypoints[0,0]), float(waypoints[0,1])
    t_hat, _ = traj.tangent_normal(0.0)
    th0 = math.atan2(t_hat[1], t_hat[0])
    for r in (real, ghost_dos, ghost_fdi):
        try: r.set_pose(x0, y0, th0)
        except TypeError: r.set_pose((x0, y0, th0))

    ctl = DTRescueController(real, traj, dt=dt)
    ctl.s = float(traj.nearest_s(x0, y0))
    ctl.x_hat = np.array([x0, y0, th0], float)
    d_hat = float(CFG.initial_depth)

    injector = FDIInjector(kind="drift", amp=0.25, omega=0.6, noise_std=(0.0,0.0,0.0))

    s_phase = ctl.s
    rows = []
    N = int(T/dt)

    for k in range(N):
        t = k*dt
        dos_on = in_windows(t, dos_windows) if dos_windows else False
        fdi_on = in_windows(t, fdi_windows) if fdi_windows else False

        # phase from REAL
        xr, yr, thr = real.get_pose()
        s_near = traj.nearest_s_pruned(xr, yr, thr, s_phase, max_jump=0.08, cone_deg=70.0) \
                 if hasattr(traj, "nearest_s_pruned") \
                 else traj.nearest_s_local(xr, yr, s_phase, max_jump=0.08)
        th_path = traj.sample(s_phase)[0][2]
        c_t = math.cos(th_path)*math.cos(thr) + math.sin(th_path)*math.sin(thr)
        if c_t < 0.2: s_near = min(s_near, s_phase)
        s_phase = float(np.clip(s_phase + 0.30*(s_near - s_phase), 0.0, 0.999))
        ctl.s = s_phase

        # reference
        x_d, _, _ = traj.sample(s_phase)
        theta_d = float(x_d[2])
        kappa   = float(traj.curvature(s_phase))
        v_d_star= min(CFG.lambda_v, CFG.lambda_omega/(abs(kappa)+1e-6))
        omega_d = v_d_star * kappa

        # main control at REAL pose
        dx, dy = x_d[0]-xr, x_d[1]-yr
        e_x =  math.cos(thr)*dx + math.sin(thr)*dy
        e_y = -math.sin(thr)*dx + math.cos(thr)*dy
        e_th= (theta_d - thr + math.pi) % (2*math.pi) - math.pi
        v, w = compute_control(e_x, e_y, e_th, v_d_star, omega_d, d_hat, CFG.gains)
        u_main = np.array([float(np.clip(v, -CFG.lambda_v, CFG.lambda_v)),
                           float(np.clip(w, -CFG.lambda_omega, CFG.lambda_omega))], float)

        # REAL with DT rescue (one flag for either attack)
        u_out = ctl.step(t, (dos_on or fdi_on), np.array([xr, yr, thr], float), u_main)
        real.update(float(u_out[0]), float(u_out[1]), dt)

        # ghosts
        u_dos = np.array([0.0, 0.0], float) if dos_on else u_main.copy()
        ghost_dos.update(float(u_dos[0]), float(u_dos[1]), dt)

        xg, yg, thg = ghost_fdi.get_pose()
        pose_true = np.array([xg, yg, thg], float)
        pose_meas = injector.corrupt(pose_true, t) if fdi_on else pose_true
        xm, ym, thm = pose_meas
        dxm, dym = x_d[0]-xm, x_d[1]-ym
        e_xm =  math.cos(thm)*dxm + math.sin(thm)*dym
        e_ym = -math.sin(thm)*dxm + math.cos(thm)*dym
        e_thm= (theta_d - thm + math.pi) % (2*math.pi) - math.pi
        v_fdi, w_fdi = compute_control(e_xm, e_ym, e_thm, v_d_star, omega_d, d_hat, CFG.gains)
        v_fdi = float(np.clip(v_fdi, -CFG.lambda_v, CFG.lambda_v))
        w_fdi = float(np.clip(w_fdi, -CFG.lambda_omega, CFG.lambda_omega))
        ghost_fdi.update(v_fdi, w_fdi, dt)

        # adapt depth only when main is healthy
        if ctl.mode == 'main':
            d_hat = update_depth_estimate(d_hat, e_x, e_y, e_th, v_d_star,
                                          CFG.gamma, CFG.gains['k3'], CFG.d_min, CFG.d_max, dt)

        # log row
        xr2, yr2, _ = real.get_pose()
        xdos, ydos, _ = ghost_dos.get_pose()
        xfdi, yfdi, _ = ghost_fdi.get_pose()
        c_t_now = math.cos(theta_d)*math.cos(thr) + math.sin(theta_d)*math.sin(thr)
        d_perp, r_eff = tube_state(traj, s_phase, (xr, yr, thr), c_t_now)

        rows.append({
            "t": t, "s": s_phase, "mode": ctl.mode,
            "dos": int(dos_on), "fdi": int(fdi_on),
            "x": xr2, "y": yr2, "xd": x_d[0], "yd": x_d[1],
            "x_dos": xdos, "y_dos": ydos, "x_fdi": xfdi, "y_fdi": yfdi,
            "v_out": float(u_out[0]), "w_out": float(u_out[1]),
            "d_perp": float(d_perp), "r_eff": float(r_eff)
        })

        # light progress print
        if (k % max(1, N//5) == 0) and (k > 0):
            print(f"    step {k}/{N} ({100*k//N}%)", end="\r")

    # write logs
    ensure_dir(outdir_run)
    with open(os.path.join(outdir_run, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # optional XY
    if save_xy:
        x,y,xd,yd = (np.array([r[k] for r in rows]) for k in ("x","y","xd","yd"))
        xdos,ydos,xfdi,yfdi = (np.array([r[k] for r in rows]) for k in ("x_dos","y_dos","x_fdi","y_fdi"))
        plt.figure(figsize=(6,6))
        plt.plot(xd, yd, "g:", label="ref")
        plt.plot(x, y,  "b", lw=1.5, label="real (Main/DT)")
        plt.plot(xdos, ydos, color="0.25", lw=1, label="ghost DoS")
        plt.plot(xfdi, yfdi, color="0.6",  lw=1, label="ghost FDI")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend(); plt.title("XY"); plt.tight_layout()
        plt.savefig(os.path.join(outdir_run, "xy.png"), dpi=180); plt.close()

    # per-run summary (for analyze_metrics.py)
    x,y,xd,yd = (np.array([r[k] for r in rows]) for k in ("x","y","xd","yd"))
    xdos,ydos,xfdi,yfdi = (np.array([r[k] for r in rows]) for k in ("x_dos","y_dos","x_fdi","y_fdi"))
    err_real = np.hypot(x-xd, y-yd).mean()
    err_dos  = np.hypot(xdos-xd, ydos-yd).mean()
    err_fdi  = np.hypot(xfdi-xd, yfdi-yd).mean()
    time_in_dt = np.mean([1.0 if r["mode"]=="dt" else 0.0 for r in rows])
    tube_viol  = int(np.sum([abs(r["d_perp"]) > r["r_eff"] for r in rows]))
    summary = {
        "s_final": float(rows[-1]["s"]),
        "err_real_mean": float(err_real),
        "err_dos_mean":  float(err_dos),
        "err_fdi_mean":  float(err_fdi),
        "time_in_dt_ratio": float(time_in_dt),
        "tube_violations": tube_viol,
        "dos_windows": len(dos_windows or []),
        "fdi_windows": len(fdi_windows or [])
    }
    with open(os.path.join(outdir_run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

# ---------------- batch driver ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_dir", type=str, default="paths",
                    help="folder with circle.npy & freehand_*.npy")
    ap.add_argument("--paths", type=str, default="",
                    help="comma subset to include, e.g. 'figure8,sharpL,circle'")
    ap.add_argument("--runs_per_path", type=int, default=5)
    ap.add_argument("--T", type=float, default=120.0)
    ap.add_argument("--dt_batch", type=float, default=None, help="override CFG.dt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_xy", action="store_true", help="skip per-run XY PNG")
    ap.add_argument("--outdir", type=str, default="results_batch_all")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ensure_dir(args.outdir)

    # discover paths (built-ins + circle + recorded freehand)
    ensure_circle(args.paths_dir)
    discovered = discover_paths(args.paths_dir)  # list[(name, ndarray|None)]
    paths = []
    for name, arr in discovered:
        wp = builtin_waypoints(name) if arr is None else np.asarray(arr, float)
        if wp.shape[0] >= 2: paths.append((name, wp))

    if args.paths:
        keep = set(s.strip() for s in args.paths.split(",") if s.strip())
        paths = [(n,w) for (n,w) in paths if n in keep]

    if not paths:
        print("No paths found."); sys.exit(1)

    dt_run = args.dt_batch if args.dt_batch is not None else CFG.dt
    total_steps = len(paths) * args.runs_per_path * int(args.T/dt_run)
    print(f"[plan] paths={len(paths)} runs/path={args.runs_per_path} steps/run≈{int(args.T/dt_run)} "
          f"total_steps≈{total_steps:,} dt={dt_run} save_xy={not args.no_xy}")

    summary_rows = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.outdir, stamp)
    ensure_dir(root)

    for pname, waypoints in paths:
        for r_id in range(args.runs_per_path):
            seed = rng.randint(0, 10**9-1)
            dos_w = make_attack_windows(args.T, rng)
            fdi_w = make_attack_windows(args.T, rng)
            outdir_run = os.path.join(root, pname, f"seed_{seed}")
            print(f"[run] path={pname:<12} run={r_id+1}/{args.runs_per_path} seed={seed} "
                  f"T={args.T} #DoS={len(dos_w)} #FDI={len(fdi_w)}")
            summary = run_once(
                waypoints, outdir_run, seed=seed, T=args.T, dt=dt_run,
                dos_windows=dos_w, fdi_windows=fdi_w, save_xy=(not args.no_xy)
            )
            summary_rows.append({"path": pname, "seed": seed, **summary})

    # batch summary
    keys = list(summary_rows[0].keys())
    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary_rows)
    print(f"[done] wrote {len(summary_rows)} runs → {os.path.join(root, 'summary.csv')}")

if __name__ == "__main__":
    main()
