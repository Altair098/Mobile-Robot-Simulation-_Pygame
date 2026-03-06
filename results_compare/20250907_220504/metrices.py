# metrics.py
import math, json, csv, os
import numpy as np

def _wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

def _project_series_to_traj(xs, ys, thetas, traj, s0=0.0):
    """Progressive nearest projection -> s, e_perp, e_theta."""
    n = len(xs)
    s    = np.zeros(n, float)
    eperp= np.zeros(n, float)
    eth  = np.zeros(n, float)
    s_prev = float(s0)
    for i in range(n):
        x, y = float(xs[i]), float(ys[i])
        if hasattr(traj, "nearest_s_local"):
            si = float(traj.nearest_s_local(x, y, s_prev, max_jump=0.08))
        else:
            si = float(traj.nearest_s(x, y))
        x_d, _, _ = traj.sample(si)
        th_ref = float(x_d[2])
        t_x, t_y = math.cos(th_ref), math.sin(th_ref)
        n_x, n_y = -t_y, t_x
        # signed lateral error
        eperp[i] = (x - x_d[0]) * n_x + (y - x_d[1]) * n_y
        if thetas is not None:
            eth[i] = _wrap(th_ref - float(thetas[i]))
        else:
            eth[i] = np.nan
        s[i] = si
        s_prev = si
    return s, eperp, eth

def _summary(t, s, eperp, eth, r_tube=0.25):
    rms_perp = float(np.sqrt(np.mean(eperp**2)))
    max_perp = float(np.max(np.abs(eperp)))
    rms_eth  = float(np.sqrt(np.nanmean(eth**2))) if np.any(~np.isnan(eth)) else np.nan
    prog     = float(np.nanmax(s))
    comp     = float(np.sum(np.diff(s)>=-1e-6)/max(1,len(s)-1))  # monotonicity ratio
    outside  = float(np.sum(np.abs(eperp)>r_tube) * (t[1]-t[0]) ) if len(t)>1 else 0.0
    return {
        "rms_e_perp": rms_perp,
        "max_e_perp": max_perp,
        "rms_e_theta": rms_eth,
        "final_progress_s": prog,
        "monotone_ratio": comp,
        "time_outside_tube[s]": outside
    }

def compute_all_metrics(traj, log_rows, results_dir, r_tube=0.25):
    # pull logged arrays
    t    = np.array([r["t"] for r in log_rows], float)
    x    = np.array([r["x"] for r in log_rows], float)
    y    = np.array([r["y"] for r in log_rows], float)
    th   = None
    if "theta" in log_rows[0]:
        th = np.array([r["theta"] for r in log_rows], float)

    # Optional ghost tracks if present
    have_dos = "x_dos" in log_rows[0]
    have_fdi = "x_fdi" in log_rows[0]
    x_dos = np.array([r.get("x_dos", np.nan) for r in log_rows], float) if have_dos else None
    y_dos = np.array([r.get("y_dos", np.nan) for r in log_rows], float) if have_dos else None
    x_fdi = np.array([r.get("x_fdi", np.nan) for r in log_rows], float) if have_fdi else None
    y_fdi = np.array([r.get("y_fdi", np.nan) for r in log_rows], float) if have_fdi else None

    # Real robot metrics
    s_real, eperp_real, eth_real = _project_series_to_traj(x, y, th, traj, s0=float(log_rows[0]["s"]))
    m_real = _summary(t, s_real, eperp_real, eth_real, r_tube)

    # Ghost DoS metrics
    m_dos = None
    if have_dos:
        mask = ~np.isnan(x_dos)
        s_dos, eperp_dos, eth_dos = _project_series_to_traj(x_dos[mask], y_dos[mask], None, traj, s0=float(log_rows[0]["s"]))
        t_dos = t[mask]
        m_dos = _summary(t_dos, s_dos, eperp_dos, eth_dos, r_tube)

    # Ghost FDI metrics
    m_fdi = None
    if have_fdi:
        mask = ~np.isnan(x_fdi)
        s_fdi, eperp_fdi, eth_fdi = _project_series_to_traj(x_fdi[mask], y_fdi[mask], None, traj, s0=float(log_rows[0]["s"]))
        t_fdi = t[mask]
        m_fdi = _summary(t_fdi, s_fdi, eperp_fdi, eth_fdi, r_tube)

    # Save JSON + CSV
    metrics = {"real": m_real}
    if m_dos is not None: metrics["ghost_DoS"] = m_dos
    if m_fdi is not None: metrics["ghost_FDI"] = m_fdi

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # flat CSV
    rows = []
    for k, v in metrics.items():
        r = {"track": k}; r.update(v); rows.append(r)
    with open(os.path.join(results_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    return metrics
