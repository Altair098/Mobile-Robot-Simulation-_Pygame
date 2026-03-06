"""
Base Paper Replication: Trajectory Tracking for Mobile Robots
FULL REPLICATION w/ SCALED MODEL (Wang et al.)
- Uses scaled errors (x*, y*) and v_d* = v_d / D_TRUE
- Continuous depth adaptation with projection
- Saves each plot to results/ and also displays it
- Stops simulation when phase completes (s >= 0.99)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Controller gains (tuned for stability)
K1 = 0.8   # Forward tracking
K2 = 1.5   # Heading correction
K3 = 8.0   # Lateral error correction

# Depth estimator (identifies D_TRUE when scaled model is used)
GAMMA = 0.3      # Adaptation rate
D_MIN = 1.7      # Minimum depth
D_MAX = 2.1      # Maximum depth
D_INIT = 1.95    # Initial depth estimate
D_TRUE = 2.0     # True (constant) depth used to scale states in sim

# Velocity limits
V_MAX = 0.12     # Max linear velocity
W_MAX = 0.25     # Max angular velocity

# Simulation
DT = 0.05        # Time step (s)
T_TOTAL = 200.0  # Total simulation time (s)

# Outputs
RESULTS_DIR = Path("results")
SHOW_PLOTS = True  # Show plots on screen (in addition to saving)

# ============================================================================
# ROBOT MODEL
# ============================================================================

class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, v, omega, dt):
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_pose(self):
        return self.x, self.y, self.theta

# ============================================================================
# TRAJECTORY CLASS
# ============================================================================

class Trajectory:
    def __init__(self, waypoints, closed=True, speed=V_MAX):
        self.waypoints = waypoints
        self.closed = closed
        self.speed = speed

        diffs = np.diff(waypoints, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
        self.total_length = self.arc_lengths[-1]
        if self.total_length > 0:
            self.s_normalized = self.arc_lengths / self.total_length
        else:
            self.s_normalized = self.arc_lengths

    def sample(self, s):
        s = np.clip(s, 0.0, 0.999)
        idx = np.searchsorted(self.s_normalized, s) - 1
        idx = max(0, min(idx, len(self.waypoints) - 2))
        s_local = (s - self.s_normalized[idx]) / max(1e-9, self.s_normalized[idx+1] - self.s_normalized[idx])
        s_local = np.clip(s_local, 0.0, 1.0)

        p0 = self.waypoints[idx]
        p1 = self.waypoints[idx + 1]
        x = p0[0] + s_local * (p1[0] - p0[0])
        y = p0[1] + s_local * (p1[1] - p0[1])

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        theta = np.arctan2(dy, dx)
        return np.array([x, y, theta])

    def curvature(self, s, ds=0.01):
        s1 = max(0.0, s - ds)
        s2 = min(0.999, s + ds)
        p1 = self.sample(s1)
        p2 = self.sample(s2)
        dtheta = (p2[2] - p1[2] + np.pi) % (2 * np.pi) - np.pi
        arc_len = max(1e-6, (s2 - s1) * self.total_length)
        return dtheta / arc_len

    def nearest_s_local(self, x, y, s_current, search_window=0.15):
        s_min = max(0.0, s_current - search_window)
        s_max = min(0.999, s_current + search_window)
        s_samples = np.linspace(s_min, s_max, 50)

        min_dist = float('inf')
        best_s = s_current
        for s_test in s_samples:
            ref = self.sample(s_test)
            dist = np.hypot(x - ref[0], y - ref[1])
            if dist < min_dist:
                min_dist = dist
                best_s = s_test
        return best_s

# ============================================================================
# TRAJECTORY GENERATORS
# ============================================================================

def generate_circle(N=2400, R=1.2):
    t = np.linspace(0, 2*np.pi, N, endpoint=True)
    return np.c_[R * np.cos(t), R * np.sin(t)]

def generate_figure8(N=2800, a=1.0):
    t = np.linspace(0, 2*np.pi, N, endpoint=True)
    return np.c_[a * np.sin(t), a * np.sin(t) * np.cos(t)]

def generate_s_curve(N=3500):
    pts = np.array([
        [0.5, 1.0], [0.35, 0.97], [0.25, 0.92], [0.2, 0.85], [0.18, 0.77],
        [0.2, 0.7], [0.25, 0.65], [0.35, 0.61], [0.5, 0.58], [0.65, 0.56],
        [0.75, 0.52], [0.8, 0.45], [0.82, 0.37], [0.8, 0.3], [0.75, 0.23],
        [0.65, 0.18], [0.5, 0.15], [0.35, 0.1], [0.5, 0.0]
    ])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=3)
    return np.array(splev(np.linspace(0, 1, N), tck)).T

def generate_l_shape(Nh=1500, Nv=1500):
    x1 = np.linspace(-2.5, 0.95, Nh)
    y1 = np.full_like(x1, 1.0)
    corner_t = np.linspace(0, np.pi/2, 50)
    x_corner = 0.95 + 0.05 * (1 - np.cos(corner_t))
    y_corner = 1.0 - 0.05 * np.sin(corner_t)
    y2 = np.linspace(0.95, -2.5, Nv)
    x2 = np.full_like(y2, 1.0)
    return np.c_[np.concatenate([x1, x_corner, x2]),
                 np.concatenate([y1, y_corner, y2])]

# ============================================================================
# CONTROLLER & UPDATES (SCALED MODEL)
# ============================================================================

def compute_control_scaled(e_x, e_y, e_theta, v_d_star, omega_d, d_hat, k1, k2, k3):
    """
    Wang et al. controller using scaled errors and v_d*:
    v = k1*tanh(e_x) + d_hat * v_d* * cos(e_theta)
    w = omega_d + k2*tanh(e_theta) + v_d* * (sin(e_theta)/e_theta) * (k3*e_y / (1 + e_x^2 + e_y^2))
    """
    v = k1 * np.tanh(e_x) + d_hat * v_d_star * np.cos(e_theta)
    steering = (v_d_star * np.sin(e_theta) / e_theta) * (k3 * e_y / (1 + e_x**2 + e_y**2)) if abs(e_theta) > 1e-6 else 0.0
    omega = omega_d + k2 * np.tanh(e_theta) + steering
    return v, omega

def update_depth_scaled(d_hat, e_x, e_y, e_theta, v_d_star, gamma, k3, d_min, d_max, dt):
    """
    Depth adaptation (continuous) with projection:
    d_dot = gamma * Proj( k3*e_x * v_d* * cos(e_theta) / (1 + e_x^2 + e_y^2) )
    """
    num = k3 * e_x * v_d_star * np.cos(e_theta)
    den = 1 + e_x**2 + e_y**2
    d_dot = gamma * (num / den)

    # Simple projection to keep d_hat in [d_min, d_max]
    if (d_hat <= d_min and d_dot < 0) or (d_hat >= d_max and d_dot > 0):
        d_dot = 0.0

    return np.clip(d_hat + d_dot * dt, d_min, d_max)

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_trajectory(path_name, waypoints, closed=True):
    print(f"\nSimulating: {path_name}...")
    traj = Trajectory(waypoints, closed=closed, speed=V_MAX)
    robot = Robot(waypoints[0, 0], waypoints[0, 1], traj.sample(0.0)[2])
    d_hat = D_INIT

    log = {k: [] for k in ['t', 'x', 'y', 'xd', 'yd', 'ex', 'ey', 'eth', 'v', 'w', 'd_hat']}
    s = 0.0
    steps = int(T_TOTAL / DT)

    for step in range(steps):
        t = step * DT

        # Stop when phase completes
        if s >= 0.99:
            print(f"-> Finished at t={t:.2f}s")
            break

        x, y, theta = robot.get_pose()

        # Phase tracking (nearest point + phase smoothing)
        s_near = traj.nearest_s_local(x, y, s)
        ref_now = traj.sample(s)
        c_t = np.cos(ref_now[2]) * np.cos(theta) + np.sin(ref_now[2]) * np.sin(theta)
        if c_t < 0.3:
            s_near = min(s_near, s)  # avoid backward jumps
        phase_gain = 0.4 if c_t > 0.7 else 0.2
        s += phase_gain * (s_near - s)

        # Advance phase using previous forward speed (if any)
        if step > 0 and len(log['v']) > 0:
            v_prev = max(0.0, log['v'][-1])
            s += v_prev * DT / max(1e-9, traj.total_length)

        # Boundary handling
        if closed:
            s_lookup = s % 1.0
        else:
            s = np.clip(s, 0.0, 0.999)
            s_lookup = s

        # Reference pose
        ref = traj.sample(s_lookup)
        x_ref, y_ref, theta_ref = ref

        # --------------------------
        # SCALED ERRORS (x*, y*)
        # --------------------------
        x_star, y_star = x / D_TRUE, y / D_TRUE
        x_ref_star, y_ref_star = x_ref / D_TRUE, y_ref / D_TRUE

        dx_star = x_ref_star - x_star
        dy_star = y_ref_star - y_star

        # Body-frame errors from scaled coordinates
        e_x =  np.cos(theta) * dx_star + np.sin(theta) * dy_star
        e_y = -np.sin(theta) * dx_star + np.cos(theta) * dy_star
        e_theta = (theta_ref - theta + np.pi) % (2*np.pi) - np.pi

        # Feedforward signals (geometric)
        kappa = traj.curvature(s_lookup)
        v_d = min(V_MAX, W_MAX / (abs(kappa) + 1e-6))
        omega_d = v_d * kappa

        # Scaled desired speed
        v_d_star = v_d / D_TRUE

        # Control
        v, omega = compute_control_scaled(e_x, e_y, e_theta, v_d_star, omega_d, d_hat, K1, K2, K3)

        # Saturate
        v = float(np.clip(v, -V_MAX, V_MAX))
        omega = float(np.clip(omega, -W_MAX, W_MAX))

        # Update robot
        robot.update(v, omega, DT)

        # Continuous depth adaptation (no gating)
        d_hat = update_depth_scaled(d_hat, e_x, e_y, e_theta, v_d_star, GAMMA, K3, D_MIN, D_MAX, DT)

        # Log
        log['t'].append(t);      log['x'].append(x);     log['y'].append(y)
        log['xd'].append(x_ref); log['yd'].append(y_ref)
        log['ex'].append(e_x);   log['ey'].append(e_y);  log['eth'].append(e_theta)
        log['v'].append(v);      log['w'].append(omega); log['d_hat'].append(d_hat)

    return {k: np.array(v) for k, v in log.items()}

# ============================================================================
# PLOTTING (SAVE + SHOW)
# ============================================================================

def save_and_show(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path.resolve()}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

def save_separate_plots(path_name, log, waypoints):
    base = RESULTS_DIR / path_name

    # 1) XY Trajectory
    fig = plt.figure(figsize=(8, 8))
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'g--', lw=3, label='Reference')
    plt.plot(log['x'], log['y'], 'b-', lw=2, label='Robot')
    plt.scatter([log['x'][0]], [log['y'][0]], c='g', s=150, label='Start', zorder=5)
    plt.scatter([log['x'][-1]], [log['y'][-1]], c='r', s=150, marker='X', label='End', zorder=5)
    plt.xlabel('X [m]'); plt.ylabel('Y [m]')
    plt.title(f'{path_name}: Trajectory Tracking')
    plt.legend(); plt.grid(True, alpha=0.4); plt.axis('equal'); plt.tight_layout()
    save_and_show(fig, base.with_name(f"{path_name}_Trajectory.png"))

    # 2) Position Errors (scaled body-frame)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(log['t'], log['ex'], label='$e_x$ (scaled)', lw=2)
    plt.plot(log['t'], log['ey'], label='$e_y$ (scaled)', lw=2)
    plt.xlabel('Time [s]'); plt.ylabel('Error [scaled m]')
    plt.title(f'{path_name}: Position Errors (scaled)')
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    save_and_show(fig, base.with_name(f"{path_name}_Position_Errors.png"))

    # 3) Heading Error
    fig = plt.figure(figsize=(10, 6))
    plt.plot(log['t'], np.rad2deg(log['eth']), 'k-', lw=2)
    plt.xlabel('Time [s]'); plt.ylabel('Error [deg]')
    plt.title(f'{path_name}: Heading Error')
    plt.grid(True, alpha=0.4); plt.tight_layout()
    save_and_show(fig, base.with_name(f"{path_name}_Heading_Error.png"))

    # 4) Velocities
    fig = plt.figure(figsize=(10, 6))
    plt.plot(log['t'], log['v'], 'b-', label='Linear v', lw=2)
    plt.plot(log['t'], log['w'], 'r-', label='Angular ω', lw=2)
    plt.axhline(V_MAX,  color='b', linestyle='--', alpha=0.5, label='V limit')
    plt.axhline(-V_MAX, color='b', linestyle='--', alpha=0.5)
    plt.axhline(W_MAX,  color='r', linestyle='--', alpha=0.5, label='W limit')
    plt.axhline(-W_MAX, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]'); plt.ylabel('Magnitude')
    plt.title(f'{path_name}: Control Inputs')
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    save_and_show(fig, base.with_name(f"{path_name}_Velocities.png"))

    # 5) Depth Identification
    fig = plt.figure(figsize=(10, 6))
    plt.plot(log['t'], log['d_hat'], 'b-', lw=2, label='Estimated Depth')
    plt.axhline(D_TRUE, color='g', linestyle='--', lw=3, label='True Depth')
    plt.fill_between(log['t'], D_MIN, D_MAX, color='gray', alpha=0.15, label='Bounds')
    plt.xlabel('Time [s]'); plt.ylabel('Depth')
    plt.title(f'{path_name}: Depth Identification')
    plt.legend(); plt.grid(True, alpha=0.4); plt.ylim(1.6, 2.2); plt.tight_layout()
    save_and_show(fig, base.with_name(f"{path_name}_Depth.png"))

# ============================================================================
# MAIN
# ============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved under: {RESULTS_DIR.resolve()}")

    trajectories = {
        'Circle':  (generate_circle(), True),
        'Figure-8':(generate_figure8(), True),
        'S-Curve': (generate_s_curve(), False),
        'L-Shape': (generate_l_shape(), False),
    }

    print("Starting simulations...")
    for name, (waypoints, closed) in trajectories.items():
        log = simulate_trajectory(name, waypoints, closed)
        save_separate_plots(name, log, waypoints)
        print(f"Generated and saved plots for: {name}")

    print("\nDone! All files generated.")

if __name__ == "__main__":
    main()
