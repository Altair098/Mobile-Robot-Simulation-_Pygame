# batch_dos_only.py
# DoS-only batch runner with per-run DoS stats and optional healthy baseline.
# Outputs per path:
#   logs.csv      – all timesteps, all runs
#   summary.csv   – one row per run (adds dos_duty, dos_switches, baseline flag)
#
# Example:
#   python batch_dos_only.py --paths_dir paths --runs_per_path 20 --T 120 \
#       --outdir exp_batch_dos --include_healthy --seed 0

import os, csv, math, argparse, random
import numpy as np

from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from generic_trajectory import GenericTrajectory
from dt_rescue_dos import DTRescueController
import config as CFG

# ---------- canonical paths ----------
def _load_path(paths_dir, name):
    fn = dict(circle="circle.npy", figure8="figure8.npy",
              sCurve="sCurve.npy", sharpL="sharpL.npy")[name]
    p = os.path.join(paths_dir, fn)
    if not os.path.isfile(p): raise FileNotFoundError(p)
    xy = np.load(p).astype(float)
    if xy.ndim != 2 or xy.shape[1] != 2: raise ValueError(f"bad path shape in {p}")
    return name, xy

def discover_paths(paths_dir):
    return [_load_path(paths_dir, n) for n in ("circle","figure8","sCurve","sharpL")]

# ---------- DoS windows ----------
def make_dos_windows(T, rng, min_len=5.0, max_len=9.0, gap=8.0, p_start=0.55):
    wins, t = [], gap * rng.uniform(0.3, 1.0)
    while t < T - min_len:
        if rng.random() < p_start:
            L = rng.uniform(min_len, max_len)
            wins.append((t, min(T, t+L)))
            t += L
        t += rng.uniform(gap*0.6, gap*1.4)
    return wins

def in_windows(t, wins): return any(a<=t<=b for a,b in wins)

def windows_stats(wins, T):
    duty = float(sum(max(0.0, min(T, b) - max(0.0, a)) for a,b in wins)) / max(T,1e-9)
    switches = int(len(wins))
    return duty, switches

# ---------- one rollout ----------
def run_once(run_id, waypoints, T, dt, dos_wins, baseline=False):
    traj = GenericTrajectory.from_waypoints(waypoints, closed=False, speed=CFG.lambda_v)
    real, ghost = Robot(), Robot()

    x0, y0 = float(waypoints[0,0]), float(waypoints[0,1])
    t_hat,_ = traj.tangent_normal(0.0); th0 = math.atan2(t_hat[1], t_hat[0])
    for r in (real, ghost):
        try: r.set_pose(x0,y0,th0)
        except TypeError: r.set_pose((x0,y0,th0))

    ctl = DTRescueController(real, traj, dt=dt)
    ctl.s = float(traj.nearest_s(x0,y0)); ctl.x_hat = np.array([x0,y0,th0], float)
    d_hat = float(CFG.initial_depth)

    rows = []
    s = ctl.s
    N = int(T/dt)
    for k in range(N):
        t = k*dt
        dos_on = in_windows(t, dos_wins)

        xr,yr,thr = real.get_pose()
        # phase
        s_near = traj.nearest_s_local(xr, yr, s, max_jump=0.08)
        th_path = traj.sample(s)[0][2]
        c_t = math.cos(th_path)*math.cos(thr)+math.sin(th_path)*math.sin(thr)
        if c_t < 0.2: s_near = min(s_near, s)
        s += 0.30*(s_near - s); s = float(np.clip(s,0.0,0.999)); ctl.s = s

        # reference
        x_d,_,_ = traj.sample(s); thd = float(x_d[2]); kappa = float(traj.curvature(s))
        v_d_star = min(CFG.lambda_v, CFG.lambda_omega/(abs(kappa)+1e-6)); w_d = v_d_star*kappa

        # errors
        dx,dy = x_d[0]-xr, x_d[1]-yr
        e_x =  math.cos(thr)*dx + math.sin(thr)*dy
        e_y = -math.sin(thr)*dx + math.cos(thr)*dy
        e_th= (thd - thr + math.pi)%(2*math.pi) - math.pi

        v,w = compute_control(e_x,e_y,e_th, v_d_star,w_d, d_hat, CFG.gains)
        u_main = np.array([np.clip(v,-CFG.lambda_v,CFG.lambda_v),
                           np.clip(w,-CFG.lambda_omega,CFG.lambda_omega)], float)

        # real under DT rescue
        u_out = ctl.step(t, dos_on, np.array([xr,yr,thr],float), u_main)
        real.update(float(u_out[0]), float(u_out[1]), dt)

        # ghost DoS baseline
        u_ghost = np.array([0.0,0.0], float) if dos_on else u_main.copy()
        ghost.update(float(u_ghost[0]), float(u_ghost[1]), dt)

        if ctl.mode == 'main':
            d_hat = update_depth_estimate(d_hat, e_x,e_y,e_th, v_d_star,
                                          CFG.gamma, CFG.gains['k3'], CFG.d_min, CFG.d_max, dt)

        xr2,yr2,_ = real.get_pose(); xg,yg,_ = ghost.get_pose()
        rows.append({
            "run_id":run_id, "t":t, "s":s,
            "dos":int(dos_on), "baseline":int(baseline),
            "x":xr2, "y":yr2, "xd":x_d[0], "yd":x_d[1],
            "x_dos":xg, "y_dos":yg, "mode":ctl.mode,
            "v_out":float(u_out[0]), "w_out":float(u_out[1])
        })

    # metrics
    X  = np.array([r["x"] for r in rows]);   Y  = np.array([r["y"] for r in rows])
    XD = np.array([r["xd"] for r in rows]);  YD = np.array([r["yd"] for r in rows])
    XG = np.array([r["x_dos"] for r in rows]); YG = np.array([r["y_dos"] for r in rows])
    e_real = float(np.mean(np.hypot(X-XD, Y-YD)))
    e_dos  = float(np.mean(np.hypot(XG-XD, YG-YD)))
    ratio_dt = float(np.mean([1.0 if r["mode"]=="dt" else 0.0 for r in rows]))
    duty, switches = windows_stats(dos_wins, T)
    return rows, {
        "run_id":run_id,
        "baseline": int(baseline),
        "err_real_mean": e_real,
        "err_dos_mean": e_dos,
        "time_in_dt_ratio": ratio_dt,
        "dos_duty": duty,
        "dos_switches": switches
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_dir", type=str, default="paths")
    ap.add_argument("--outdir",    type=str, default="exp_batch_dos")
    ap.add_argument("--runs_per_path", type=int, default=10)
    ap.add_argument("--T", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include_healthy", action="store_true",
                    help="also log healthy baseline runs (no DoS)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    dt = args.dt if args.dt is not None else CFG.dt

    paths = discover_paths(args.paths_dir)
    print(f"[plan] outdir={args.outdir}  paths={len(paths)}  runs/path={args.runs_per_path}")

    for pname, waypoints in paths:
        pdir = os.path.join(args.outdir, pname); os.makedirs(pdir, exist_ok=True)
        logs_csv    = os.path.join(pdir, "logs.csv")
        summary_csv = os.path.join(pdir, "summary.csv")
        if args.overwrite:
            for f in (logs_csv, summary_csv):
                if os.path.exists(f): os.remove(f)

        log_f  = open(logs_csv, "a", newline="")
        sum_f  = open(summary_csv, "a", newline="")
        log_w = sum_w = None

        try:
            for rix in range(args.runs_per_path):
                run_id = f"{pname}_{args.seed}_{rix:03d}_{rng.randint(0,999999):06d}"

                # healthy baseline (optional)
                if args.include_healthy:
                    rows_h, summ_h = run_once(run_id + "_healthy", waypoints, T=args.T, dt=dt,
                                              dos_wins=[], baseline=True)
                    if log_w is None:
                        log_w = csv.DictWriter(log_f, fieldnames=list(rows_h[0].keys()))
                        if args.overwrite or os.stat(logs_csv).st_size == 0: log_w.writeheader()
                    if sum_w is None:
                        sum_w = csv.DictWriter(sum_f, fieldnames=list(summ_h.keys()))
                        if args.overwrite or os.stat(summary_csv).st_size == 0: sum_w.writeheader()
                    log_w.writerows(rows_h); sum_w.writerow(summ_h)

                # attacked run
                dos_w = make_dos_windows(args.T, rng)
                rows, summ = run_once(run_id, waypoints, T=args.T, dt=dt, dos_wins=dos_wins_from_tuple(dos_w))

                # init writers if needed
                if log_w is None:
                    log_w = csv.DictWriter(log_f, fieldnames=list(rows[0].keys()))
                    if args.overwrite or os.stat(logs_csv).st_size == 0: log_w.writeheader()
                if sum_w is None:
                    sum_w = csv.DictWriter(sum_f, fieldnames=list(summ.keys()))
                    if args.overwrite or os.stat(summary_csv).st_size == 0: sum_w.writeheader()

                log_w.writerows(rows)
                sum_w.writerow(summ)
                print(f"[run] {pname}: {rix+1}/{args.runs_per_path}")
        finally:
            log_f.close(); sum_f.close()

    print(f"[done] all data in: {args.outdir}")

# helper to ensure list of tuples survives argparse-free path
def dos_wins_from_tuple(wins):
    # wins already list[(a,b)], but keep a tiny sanitizer
    return [(float(a), float(b)) for a,b in wins]

if __name__ == "__main__":
    main()
