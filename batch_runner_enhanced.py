"""
Enhanced Batch Runner for DT-Rescue DoS Attack Experiments
===========================================================

Features:
---------
1. Saves individual trajectory plots per run
2. Comprehensive logging (timestep-level + run-level summaries)
3. Organized folder structure: DoS_Attack_Experiments/path_name/run_XXX/
4. Attack window visualization overlays
5. Meta-information files for reproducibility
6. Progress tracking and error handling

Usage:
------
python batch_runner_enhanced.py \
    --paths circle,figure8,scurve,sharpl \
    --runs 30 \
    --T 120 \
    --seed 42 \
    --outdir DoS_Attack_Experiments
"""

import os
import csv
import math
import json
import argparse
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Project imports
from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from generic_trajectory import GenericTrajectory
from dt_rescue_dos import DTRescueController
import config as CFG

# ============================================================================
# PATH GENERATORS
# ============================================================================

def path_circle(N=2400, R=1.2):
    t = np.linspace(0, 2*np.pi, N, endpoint=True)
    return np.c_[R*np.cos(t), R*np.sin(t)]

def path_figure8(N=2800, a=1.0):
    t = np.linspace(0, 2*np.pi, N, endpoint=True)
    x = a*np.sin(t)
    y = a*np.sin(t)*np.cos(t)
    return np.c_[x, y]

def path_s_curve(N=2500):
    from scipy.interpolate import splprep, splev
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

def build_waypoints(name: str) -> np.ndarray:
    """Generate waypoints for named path"""
    n = name.lower().strip()
    if n == "circle":   return path_circle()
    if n == "figure8":  return path_figure8()
    if n == "scurve":   return path_s_curve()
    if n == "sharpl":   return path_sharpL()
    raise ValueError(f"Unknown path '{name}'")

# ============================================================================
# DOS ATTACK SCHEDULER
# ============================================================================

def generate_dos_schedule(T, rng, Lmin=5.0, Lmax=9.0, Gmin=6.0, Gmax=12.0):
    """
    Generate random DoS attack windows
    
    Parameters:
    -----------
    T : float
        Total simulation time
    Lmin, Lmax : float
        Min/max attack duration
    Gmin, Gmax : float
        Min/max gap between attacks
    rng : np.random.Generator
        Random number generator
    
    Returns:
    --------
    windows : list of (start, end) tuples
    """
    windows = []
    t = rng.uniform(Gmin, Gmax)  # Initial delay
    
    while t < T:
        L = rng.uniform(Lmin, Lmax)
        start, end = t, min(t + L, T)
        windows.append((start, end))
        t = end + rng.uniform(Gmin, Gmax)
    
    return windows

def is_dos_active(t, windows):
    """Check if time t is within any attack window"""
    return any(start <= t <= end for (start, end) in windows)

# ============================================================================
# TUBE SAFETY DIAGNOSTICS
# ============================================================================

def compute_tube_state(traj, s, pose, c_t):
    """
    Compute perpendicular distance and effective tube radius
    
    Returns:
    --------
    d_perp : float
        Signed perpendicular distance (+ = left, - = right)
    r_eff : float
        Effective tube radius
    """
    R0, A_KAPPA, B_ALIGN = 0.16, 0.90, 0.30
    
    x_d, _, _ = traj.sample(s)
    th_ref = float(x_d[2])
    
    # Tangent and normal vectors
    t_x, t_y = math.cos(th_ref), math.sin(th_ref)
    n_x, n_y = -t_y, t_x
    
    # Perpendicular distance
    d_perp = (pose[0] - x_d[0]) * n_x + (pose[1] - x_d[1]) * n_y
    
    # Adaptive radius
    kappa = float(traj.curvature(s))
    r_eff = R0 * (1.0 + A_KAPPA * abs(kappa))
    r_eff *= (1.0 - B_ALIGN * max(0.0, c_t))
    r_eff = max(0.75 * R0, min(r_eff, 2.0 * R0))
    
    return d_perp, r_eff

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_trajectory_comparison(log_data, dos_windows, save_path, path_name):
    """
    Create comprehensive trajectory comparison plot
    
    Includes:
    - Reference path
    - DT trajectory (colored by mode)
    - Ghost trajectory
    - Attack window overlays
    - Error metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    
    # Extract data
    t_arr = np.array([r['t'] for r in log_data])
    x = np.array([r['x'] for r in log_data])
    y = np.array([r['y'] for r in log_data])
    xd = np.array([r['xd'] for r in log_data])
    yd = np.array([r['yd'] for r in log_data])
    xdos = np.array([r['x_dos'] for r in log_data])
    ydos = np.array([r['y_dos'] for r in log_data])
    mode = np.array([r['mode'] for r in log_data])
    attack = np.array([r['attack'] for r in log_data], dtype=int)
    
    # --- Panel A: XY Trajectories ---
    ax = axes[0, 0]
    ax.plot(xd, yd, ':', color='#2ca02c', lw=2.5, label='Reference', zorder=1)
    
    # DT trajectory with mode coloring
    for i in range(len(x) - 1):
        color = '#d62728' if mode[i] == 'dt' else '#1f77b4'
        ax.plot(x[i:i+2], y[i:i+2], color=color, lw=2.5, zorder=3)
    
    # Ghost trajectory
    ax.plot(xdos, ydos, color='0.35', lw=2.0, label='Ghost DoS', zorder=2)
    
    # Start/end markers
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start', zorder=4)
    ax.plot(x[-1], y[-1], 'r^', markersize=10, label='End', zorder=4)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]', fontsize=11)
    ax.set_ylabel('y [m]', fontsize=11)
    ax.set_title(f'A) Trajectory Comparison: {path_name}', fontsize=12, weight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # --- Panel B: Tracking Error Time Series ---
    ax = axes[0, 1]
    e_dt = np.hypot(x - xd, y - yd)
    e_ghost = np.hypot(xdos - xd, ydos - yd)
    
    ax.plot(t_arr, e_dt, color='#1f77b4', lw=2.0, label='DT Rescue')
    ax.plot(t_arr, e_ghost, color='0.35', lw=2.0, label='Ghost DoS')
    
    # Attack window shading
    for (t_start, t_end) in dos_windows:
        ax.axvspan(t_start, t_end, color='red', alpha=0.15, zorder=0)
    
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Tracking Error [m]', fontsize=11)
    ax.set_title('B) Error Evolution', fontsize=12, weight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # --- Panel C: Control Commands ---
    ax = axes[1, 0]
    v_out = np.array([r['v_out'] for r in log_data])
    w_out = np.array([r['w_out'] for r in log_data])
    
    ax2 = ax.twinx()
    l1 = ax.plot(t_arr, v_out, color='#ff7f0e', lw=1.8, label='v [m/s]')
    l2 = ax2.plot(t_arr, w_out, color='#9467bd', lw=1.8, label='ω [rad/s]')
    
    # Attack shading
    for (t_start, t_end) in dos_windows:
        ax.axvspan(t_start, t_end, color='red', alpha=0.15, zorder=0)
    
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Linear Velocity [m/s]', fontsize=11, color='#ff7f0e')
    ax2.set_ylabel('Angular Velocity [rad/s]', fontsize=11, color='#9467bd')
    ax.set_title('C) Control Commands', fontsize=12, weight='bold')
    
    # Combined legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # --- Panel D: Mode & Safety ---
    ax = axes[1, 1]
    mode_numeric = np.array([1 if m == 'dt' else 0 for m in mode])
    d_perp = np.array([r['d_perp'] for r in log_data])
    r_eff = np.array([r['r_eff'] for r in log_data])
    safety_ratio = np.abs(d_perp) / r_eff
    
    ax2 = ax.twinx()
    l1 = ax.fill_between(t_arr, 0, mode_numeric, color='#d62728', alpha=0.3, 
                          step='post', label='DT Active')
    l2 = ax2.plot(t_arr, safety_ratio, color='#2ca02c', lw=1.5, 
                  label='|d⊥|/r_eff')
    ax2.axhline(1.0, color='red', linestyle='--', lw=1.0, label='Safety Limit')
    
    # Attack shading
    for (t_start, t_end) in dos_windows:
        ax.axvspan(t_start, t_end, color='red', alpha=0.15, zorder=0)
    
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Mode (1=DT)', fontsize=11)
    ax2.set_ylabel('Safety Ratio', fontsize=11)
    ax.set_title('D) Mode & Safety Constraints', fontsize=12, weight='bold')
    ax.set_ylim(-0.1, 1.2)
    
    # Combined legend
    lns = [l1] + l2
    labs = ['DT Active', '|d⊥|/r_eff', 'Safety Limit']
    ax2.legend(lns, labs, loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Overall title with metrics
    final_error_dt = e_dt[-1]
    final_error_ghost = e_ghost[-1]
    cum_error_dt = np.trapz(e_dt, t_arr)
    cum_error_ghost = np.trapz(e_ghost, t_arr)
    improvement = 100 * (1 - cum_error_dt / max(cum_error_ghost, 1e-6))
    
    fig.suptitle(
        f'DT Rescue Performance | '
        f'Final Error: DT={final_error_dt:.3f}m, Ghost={final_error_ghost:.3f}m | '
        f'Improvement: {improvement:.1f}%',
        fontsize=13, weight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# SINGLE RUN EXECUTION
# ============================================================================

def run_single_experiment(run_id, path_name, waypoints, T, dt, dos_windows, 
                         run_dir, rng_seed):
    """
    Execute a single experimental run
    
    Returns:
    --------
    log_data : list of dicts
        Timestep-level data
    summary : dict
        Run-level summary metrics
    """
    print(f"  → Running: {run_id}")
    
    # Create trajectory
    closed = (path_name.lower() in {'circle', 'figure8'})
    traj = GenericTrajectory.from_waypoints(waypoints, closed=closed, speed=CFG.lambda_v)
    
    # Initialize robots
    real = Robot()
    ghost = Robot()
    
    # Initial pose
    x0, y0 = float(waypoints[0, 0]), float(waypoints[0, 1])
    t_hat, _ = traj.tangent_normal(0.0)
    th0 = float(math.atan2(t_hat[1], t_hat[0]))
    
    try:
        real.set_pose(x0, y0, th0)
        ghost.set_pose(x0, y0, th0)
    except TypeError:
        real.set_pose((x0, y0, th0))
        ghost.set_pose((x0, y0, th0))
    
    # Initialize controller
    ctrl = DTRescueController(real, traj, dt=dt)
    ctrl.s = float(traj.nearest_s(x0, y0))
    ctrl.x_hat = np.array([x0, y0, th0], dtype=float)
    
    # Depth estimate
    d_hat = float(CFG.initial_depth)
    
    # Simulation loop
    log_data = []
    s_phase = ctrl.s
    steps = int(T / dt)
    
    for k in range(steps):
        t = k * dt
        attack = is_dos_active(t, dos_windows)
        
        # --- Phase advancement ---
        xr, yr, thr = real.get_pose()
        s_near = traj.nearest_s_local(xr, yr, s_phase, max_jump=0.08)
        
        x_ref, _, _ = traj.sample(s_phase)
        theta_path = float(x_ref[2])
        c_t = math.cos(theta_path) * math.cos(thr) + \
              math.sin(theta_path) * math.sin(thr)
        
        if c_t < 0.2:
            s_near = min(s_near, s_phase)
        
        ds = max(0.0, s_near - s_phase)
        s_phase += 0.30 * ds
        s_phase = float(np.clip(s_phase, 0.0, 0.999))
        
        ctrl.s = s_phase
        
        # --- Reference generation ---
        x_d, _, _ = traj.sample(s_phase)
        theta_d = float(x_d[2])
        kappa = float(traj.curvature(s_phase))
        v_d_star = min(CFG.lambda_v, CFG.lambda_omega / (abs(kappa) + 1e-6))
        omega_d = v_d_star * kappa
        
        # --- Main controller (tracking errors) ---
        dx = x_d[0] - xr
        dy = x_d[1] - yr
        e_x = math.cos(thr) * dx + math.sin(thr) * dy
        e_y = -math.sin(thr) * dx + math.cos(thr) * dy
        e_th = (theta_d - thr + math.pi) % (2 * math.pi) - math.pi
        
        v_cmd, w_cmd = compute_control(e_x, e_y, e_th, v_d_star, omega_d, 
                                       d_hat, CFG.gains)
        v_cmd = float(np.clip(v_cmd, -CFG.lambda_v, CFG.lambda_v))
        w_cmd = float(np.clip(w_cmd, -CFG.lambda_omega, CFG.lambda_omega))
        u_main = np.array([v_cmd, w_cmd], dtype=float)
        
        # --- DT arbitration ---
        u_out = ctrl.step(t, attack, np.array([xr, yr, thr], dtype=float), u_main)
        
        # Apply to real robot
        real.update(float(u_out[0]), float(u_out[1]), dt)
        
        # Ghost baseline (actuator drop during attack)
        u_ghost = np.array([0.0, 0.0], dtype=float) if attack else u_main.copy()
        ghost.update(float(u_ghost[0]), float(u_ghost[1]), dt)
        
        # --- Depth adaptation (only in MAIN mode) ---
        if ctrl.mode == 'main':
            d_hat = update_depth_estimate(
                d_hat, e_x, e_y, e_th, v_d_star,
                CFG.gamma, CFG.gains['k3'], CFG.d_min, CFG.d_max, dt
            )
        
        # --- Logging ---
        xr2, yr2, thr2 = real.get_pose()
        xdos, ydos, _ = ghost.get_pose()
        d_perp, r_eff = compute_tube_state(traj, s_phase, (xr2, yr2, thr2), c_t)
        
        log_data.append({
            't': float(t),
            's': float(s_phase),
            'attack': int(attack),
            'mode': ctrl.mode,
            'x': float(xr2),
            'y': float(yr2),
            'theta': float(thr2),
            'xd': float(x_d[0]),
            'yd': float(x_d[1]),
            'theta_d': float(theta_d),
            'x_dos': float(xdos),
            'y_dos': float(ydos),
            'v_out': float(u_out[0]),
            'w_out': float(u_out[1]),
            'e_x': float(e_x),
            'e_y': float(e_y),
            'e_th': float(e_th),
            'd_perp': float(d_perp),
            'r_eff': float(r_eff),
            'd_hat': float(d_hat)
        })
    
    # --- Compute summary metrics ---
    t_arr = np.array([r['t'] for r in log_data])
    x = np.array([r['x'] for r in log_data])
    y = np.array([r['y'] for r in log_data])
    xd = np.array([r['xd'] for r in log_data])
    yd = np.array([r['yd'] for r in log_data])
    xdos = np.array([r['x_dos'] for r in log_data])
    ydos = np.array([r['y_dos'] for r in log_data])
    d_perp = np.array([r['d_perp'] for r in log_data])
    r_eff = np.array([r['r_eff'] for r in log_data])
    attack_arr = np.array([r['attack'] for r in log_data])
    
    e_dt = np.hypot(x - xd, y - yd)
    e_ghost = np.hypot(xdos - xd, ydos - yd)
    
    summary = {
        'run_id': run_id,
        'path_name': path_name,
        'rng_seed': rng_seed,
        'total_time': float(T),
        'dt': float(dt),
        'n_dos_windows': len(dos_windows),
        'total_dos_time': float(np.sum(attack_arr) * dt),
        'mean_error_dt': float(np.mean(e_dt)),
        'max_error_dt': float(np.max(e_dt)),
        'final_error_dt': float(e_dt[-1]),
        'cumulative_error_dt': float(np.trapz(e_dt, t_arr)),
        'mean_error_ghost': float(np.mean(e_ghost)),
        'max_error_ghost': float(np.max(e_ghost)),
        'final_error_ghost': float(e_ghost[-1]),
        'cumulative_error_ghost': float(np.trapz(e_ghost, t_arr)),
        'improvement_percent': float(100 * (1 - np.trapz(e_dt, t_arr) / 
                                            max(np.trapz(e_ghost, t_arr), 1e-6))),
        'tube_violations': int(np.sum(np.abs(d_perp) > r_eff)),
        'time_in_dt_mode': float(np.sum([r['mode'] == 'dt' for r in log_data]) * dt),
        'dt_mode_ratio': float(np.mean([r['mode'] == 'dt' for r in log_data]))
    }
    
    # --- Save timestep log ---
    log_csv = os.path.join(run_dir, 'log_timesteps.csv')
    with open(log_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_data[0].keys()))
        writer.writeheader()
        writer.writerows(log_data)
    
    # --- Save DoS windows ---
    windows_file = os.path.join(run_dir, 'dos_windows.txt')
    with open(windows_file, 'w') as f:
        f.write("# DoS Attack Windows (start, end) in seconds\n")
        for (start, end) in dos_windows:
            f.write(f"{start:.3f},{end:.3f}\n")
    
    # --- Save summary ---
    summary_json = os.path.join(run_dir, 'summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # --- Generate trajectory plot ---
    plot_path = os.path.join(run_dir, 'trajectory_comparison.png')
    plot_trajectory_comparison(log_data, dos_windows, plot_path, path_name)
    
    print(f"    ✓ Completed: {run_id} | "
          f"Error: {summary['mean_error_dt']:.4f}m (DT) vs "
          f"{summary['mean_error_ghost']:.4f}m (Ghost) | "
          f"Improvement: {summary['improvement_percent']:.1f}%")
    
    return log_data, summary

# ============================================================================
# BATCH RUNNER
# ============================================================================

def run_batch_experiments(paths, runs_per_path, T, dt, outdir, seed,
                         dos_Lmin, dos_Lmax, dos_Gmin, dos_Gmax):
    """
    Execute batch experiments across multiple paths and runs
    """
    # Create output directory structure
    base_dir = Path(outdir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize RNG
    rng = np.random.default_rng(seed)
    
    # Save experiment configuration
    config = {
        'experiment_name': 'DoS Attack Resilience Study',
        'timestamp': datetime.now().isoformat(),
        'paths': paths,
        'runs_per_path': runs_per_path,
        'total_time': T,
        'dt': dt,
        'seed': seed,
        'dos_parameters': {
            'Lmin': dos_Lmin,
            'Lmax': dos_Lmax,
            'Gmin': dos_Gmin,
            'Gmax': dos_Gmax
        },
        'control_parameters': {
            'lambda_v': CFG.lambda_v,
            'lambda_omega': CFG.lambda_omega,
            'gains': CFG.gains,
            'gamma': CFG.gamma,
            'd_min': CFG.d_min,
            'd_max': CFG.d_max,
            'initial_depth': CFG.initial_depth
        }
    }
    
    config_file = base_dir / 'experiment_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"DoS Attack Resilience Experiments")
    print(f"{'='*70}")
    print(f"Output Directory: {outdir}")
    print(f"Paths: {', '.join(paths)}")
    print(f"Runs per Path: {runs_per_path}")
    print(f"Total Time: {T}s | Timestep: {dt}s")
    print(f"Random Seed: {seed}")
    print(f"{'='*70}\n")
    
    # Master summary for all runs
    all_summaries = []
    
    # Process each path
    for path_idx, path_name in enumerate(paths):
        print(f"\n[{path_idx+1}/{len(paths)}] Processing Path: {path_name.upper()}")
        print("-" * 70)
        
        # Create path directory
        path_dir = base_dir / path_name
        path_dir.mkdir(exist_ok=True)
        
        # Generate waypoints
        waypoints = build_waypoints(path_name)
        
        # Save waypoints
        np.save(path_dir / 'waypoints.npy', waypoints)
        
        # Run experiments for this path
        path_summaries = []
        
        for run_idx in range(runs_per_path):
            # Create run directory
            run_id = f"{path_name}_run{run_idx:03d}"
            run_dir = path_dir / f"run_{run_idx:03d}"
            run_dir.mkdir(exist_ok=True)
            
            # Generate unique DoS schedule for this run
            dos_windows = generate_dos_schedule(
                T, rng, dos_Lmin, dos_Lmax, dos_Gmin, dos_Gmax
            )
            
            # Execute run
            try:
                log_data, summary = run_single_experiment(
                    run_id, path_name, waypoints, T, dt, dos_windows,
                    run_dir, seed + run_idx
                )
                path_summaries.append(summary)
                all_summaries.append(summary)
            except Exception as e:
                print(f"    ✗ ERROR in {run_id}: {e}")
                continue
        
        # Save path-level aggregate summary
        path_summary_file = path_dir / 'path_summary.csv'
        if path_summaries:
            with open(path_summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(path_summaries[0].keys()))
                writer.writeheader()
                writer.writerows(path_summaries)
        
        print(f"\n  ✓ Completed {len(path_summaries)}/{runs_per_path} runs for {path_name}")
    
    # Save master summary
    master_summary_file = base_dir / 'master_summary.csv'
    if all_summaries:
        with open(master_summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_summaries)
    
    # Generate aggregate statistics
    generate_aggregate_report(base_dir, all_summaries, paths)
    
    print(f"\n{'='*70}")
    print(f"✓ Batch Experiments Complete!")
    print(f"{'='*70}")
    print(f"Total Runs: {len(all_summaries)}")
    print(f"Results Directory: {outdir}")
    print(f"{'='*70}\n")

# ============================================================================
# AGGREGATE STATISTICS REPORT
# ============================================================================

def generate_aggregate_report(base_dir, all_summaries, paths):
    """
    Generate aggregate statistics report across all runs
    """
    report_file = base_dir / 'aggregate_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DoS Attack Resilience Experiments - Aggregate Report\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        
        all_improvements = [s['improvement_percent'] for s in all_summaries]
        all_dt_errors = [s['cumulative_error_dt'] for s in all_summaries]
        all_ghost_errors = [s['cumulative_error_ghost'] for s in all_summaries]
        all_violations = [s['tube_violations'] for s in all_summaries]
        
        f.write(f"Total Runs: {len(all_summaries)}\n")
        f.write(f"Mean Improvement: {np.mean(all_improvements):.2f}% ± {np.std(all_improvements):.2f}%\n")
        f.write(f"Min Improvement: {np.min(all_improvements):.2f}%\n")
        f.write(f"Max Improvement: {np.max(all_improvements):.2f}%\n")
        f.write(f"Total Tube Violations: {sum(all_violations)}\n")
        f.write(f"Success Rate (no violations): {100*np.mean([v==0 for v in all_violations]):.1f}%\n\n")
        
        # Per-path statistics
        f.write("PER-PATH STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        for path_name in paths:
            path_summaries = [s for s in all_summaries if s['path_name'] == path_name]
            
            if not path_summaries:
                continue
            
            f.write(f"Path: {path_name.upper()}\n")
            f.write("  " + "-"*76 + "\n")
            
            improvements = [s['improvement_percent'] for s in path_summaries]
            dt_errors = [s['cumulative_error_dt'] for s in path_summaries]
            ghost_errors = [s['cumulative_error_ghost'] for s in path_summaries]
            dos_times = [s['total_dos_time'] for s in path_summaries]
            violations = [s['tube_violations'] for s in path_summaries]
            
            f.write(f"  Runs: {len(path_summaries)}\n")
            f.write(f"  Improvement: {np.mean(improvements):.2f}% ± {np.std(improvements):.2f}%\n")
            f.write(f"  DT Cumulative Error: {np.mean(dt_errors):.4f} ± {np.std(dt_errors):.4f} m·s\n")
            f.write(f"  Ghost Cumulative Error: {np.mean(ghost_errors):.4f} ± {np.std(ghost_errors):.4f} m·s\n")
            f.write(f"  Mean DoS Exposure: {np.mean(dos_times):.2f}s ± {np.std(dos_times):.2f}s\n")
            f.write(f"  Tube Violations: {sum(violations)} total, {np.mean(violations):.2f} per run\n")
            f.write(f"  Success Rate: {100*np.mean([v==0 for v in violations]):.1f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")
    
    print(f"\n  → Aggregate report saved: {report_file}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Batch Runner for DT-Rescue DoS Experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment parameters
    parser.add_argument('--paths', type=str, 
                       default='circle,figure8,scurve,sharpl',
                       help='Comma-separated list of paths to test')
    parser.add_argument('--runs', type=int, default=30,
                       help='Number of runs per path')
    parser.add_argument('--T', type=float, default=120.0,
                       help='Total simulation time [s]')
    parser.add_argument('--dt', type=float, default=None,
                       help='Timestep [s] (uses config.dt if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--outdir', type=str, 
                       default='DoS_Attack_Experiments',
                       help='Output directory name')
    
    # DoS attack parameters
    parser.add_argument('--dos-Lmin', type=float, default=5.0,
                       help='Minimum DoS attack duration [s]')
    parser.add_argument('--dos-Lmax', type=float, default=9.0,
                       help='Maximum DoS attack duration [s]')
    parser.add_argument('--dos-Gmin', type=float, default=6.0,
                       help='Minimum gap between attacks [s]')
    parser.add_argument('--dos-Gmax', type=float, default=12.0,
                       help='Maximum gap between attacks [s]')
    
    args = parser.parse_args()
    
    # Parse paths
    paths = [p.strip().lower() for p in args.paths.split(',') if p.strip()]
    
    # Use config dt if not specified
    dt = args.dt if args.dt is not None else CFG.dt
    
    # Run batch experiments
    run_batch_experiments(
        paths=paths,
        runs_per_path=args.runs,
        T=args.T,
        dt=dt,
        outdir=args.outdir,
        seed=args.seed,
        dos_Lmin=args.dos_Lmin,
        dos_Lmax=args.dos_Lmax,
        dos_Gmin=args.dos_Gmin,
        dos_Gmax=args.dos_Gmax
    )

if __name__ == '__main__':
    main()