# cbf_filters.py
import math
import numpy as np

def _ang(a):  # wrap to [-pi,pi]
    return (a + math.pi) % (2 * math.pi) - math.pi

def _curvature_from_traj(traj, s: float):
    sA = max(0.0, s - 0.01)
    sB = min(1.0, s + 0.01)
    thA = traj.sample(sA)[0][2]
    thB = traj.sample(sB)[0][2]
    dth = _ang(thB - thA)
    ds = (sB - sA) * max(traj.total_length(), 1e-9)
    return 0.0 if ds < 1e-9 else (dth / ds)

def clamp_v_discrete(pose_xyztheta: np.ndarray,
                     v_cmd: float,
                     traj,
                     s_ref: float,
                     r_base: float = 0.15,
                     beta: float = 0.8,
                     lam: float = 2.0):
    """
    Enforce discrete CBF: h_{k+1} - h_k >= -lam * h_k * dt  (dt cancels here via derivation)
    Using h = r^2 - e_perp^2 and e_perp dynamics ~ v * c_n.
    Returns clamped v.
    """
    x, y, th = float(pose_xyztheta[0]), float(pose_xyztheta[1]), float(pose_xyztheta[2])

    # path ref & normal at s_ref
    x_d, _, _ = traj.sample(s_ref)
    th_path = float(x_d[2])
    n_hat = np.array([-math.sin(th_path), math.cos(th_path)])

    # lateral error and heading-normal projection
    e_perp = (x - x_d[0]) * n_hat[0] + (y - x_d[1]) * n_hat[1]
    c_n = n_hat[0] * math.cos(th) + n_hat[1] * math.sin(th)  # ∈[-1,1]

    # curvature-adaptive tube
    kappa = abs(_curvature_from_traj(traj, s_ref))
    r_eff = max(r_base, min(r_base * (1.0 + beta * kappa), 1.5 * r_base))
    h = r_eff * r_eff - e_perp * e_perp

    # if nearly centered or heading parallel to tangent → no clamp needed
    den = -2.0 * e_perp * c_n
    if abs(den) < 1e-8 or h <= 0 and abs(e_perp) < 1e-6:
        return float(v_cmd)

    # inequality: -2 e_perp v c_n >= -lam h  ⇒ solve for v bound
    v_bound = (lam * h) / den
    if den > 0.0:
        # need v >= v_bound
        return float(max(v_cmd, v_bound))
    else:
        # need v <= v_bound
        return float(min(v_cmd, v_bound))
