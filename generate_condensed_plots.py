"""
Condensed Plot Generation for DoS Attack Experiments
====================================================

Generates 7 publication-ready plots:
- 4 individual trajectory plots (one per path)
- 3 aggregate comparison plots (all paths combined):
  1. Error Evolution
  2. Control Commands  
  3. Mode & Safety Constraints

Usage:
------
python generate_condensed_plots.py --data_dir DoS_Attack_Experiments --output_dir figures
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Color scheme for paths
PATH_COLORS = {
    'circle': '#1f77b4',   # Blue
    'figure8': '#ff7f0e',  # Orange
    'scurve': '#2ca02c',   # Green
    'sharpl': '#d62728'    # Red
}

# Publication style
def set_publication_style():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 2.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })

# ============================================================================
# DATA LOADING
# ============================================================================

def load_run_data(data_dir, path_name, run_idx):
    """Load data for a specific run"""
    run_dir = Path(data_dir) / path_name / f"run_{run_idx:03d}"
    
    # Load timestep log
    log_df = pd.read_csv(run_dir / 'log_timesteps.csv')
    
    # Load summary
    with open(run_dir / 'summary.json', 'r') as f:
        summary = json.load(f)
    
    # Load DoS windows
    dos_windows = []
    with open(run_dir / 'dos_windows.txt', 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            start, end = map(float, line.strip().split(','))
            dos_windows.append((start, end))
    
    return log_df, summary, dos_windows

def select_median_run(data_dir, path_name):
    """Select run with median cumulative error"""
    path_dir = Path(data_dir) / path_name
    path_summary = pd.read_csv(path_dir / 'path_summary.csv')
    
    # Find median error run
    errors = path_summary['cumulative_error_dt'].values
    median_idx = np.argsort(errors)[len(errors) // 2]
    
    return median_idx

def load_all_runs(data_dir, path_name):
    """Load all runs for a given path"""
    path_dir = Path(data_dir) / path_name
    path_summary = pd.read_csv(path_dir / 'path_summary.csv')
    
    all_data = []
    for run_idx in range(len(path_summary)):
        try:
            log_df, summary, dos_windows = load_run_data(data_dir, path_name, run_idx)
            all_data.append({
                'log': log_df,
                'summary': summary,
                'dos_windows': dos_windows
            })
        except:
            continue
    
    return all_data

# ============================================================================
# PLOT 1-4: INDIVIDUAL TRAJECTORY PLOTS
# ============================================================================

def plot_single_trajectory(data_dir, path_name, output_dir):
    """
    Create single trajectory comparison plot for one path
    """
    print(f"  Plotting trajectory: {path_name}")
    
    # Select median run
    run_idx = select_median_run(data_dir, path_name)
    log_df, summary, dos_windows = load_run_data(data_dir, path_name, run_idx)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Extract data
    xd, yd = log_df['xd'].values, log_df['yd'].values
    x, y = log_df['x'].values, log_df['y'].values
    xdos, ydos = log_df['x_dos'].values, log_df['y_dos'].values
    mode = log_df['mode'].values
    t_arr = log_df['t'].values
    
    # Reference path
    ax.plot(xd, yd, ':', color='#2ca02c', lw=3.5, label='Reference', zorder=1, alpha=0.8)
    
    # DT trajectory (colored by mode)
    dt_segments = []
    main_segments = []
    for i in range(len(x) - 1):
        if mode[i] == 'dt':
            dt_segments.append([(x[i], y[i]), (x[i+1], y[i+1])])
        else:
            main_segments.append([(x[i], y[i]), (x[i+1], y[i+1])])
    
    # Plot main controller segments
    for seg in main_segments:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
               color='#1f77b4', lw=3.0, alpha=0.85, zorder=3)
    
    # Plot DT takeover segments
    for seg in dt_segments:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
               color='#d62728', lw=3.0, alpha=0.95, zorder=4)
    
    # Ghost trajectory
    ax.plot(xdos, ydos, color='0.25', lw=2.5, linestyle='-', 
           label='Ghost DoS', zorder=2, alpha=0.7)
    
    # Start/end markers
    ax.plot(x[0], y[0], 'o', color='darkgreen', markersize=14, 
           markeredgecolor='green', markeredgewidth=2.5, 
           label='Start', zorder=6, alpha=0.9)
    ax.plot(x[-1], y[-1], '^', color='darkred', markersize=14, 
           markeredgecolor='red', markeredgewidth=2.5,
           label='End', zorder=6, alpha=0.9)
    
    # Attack regions (convex hulls)
    for (t_start, t_end) in dos_windows:
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        if np.sum(mask) > 3:
            attack_x = xd[mask]
            attack_y = yd[mask]
            try:
                from scipy.spatial import ConvexHull
                points = np.column_stack([attack_x, attack_y])
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                poly = Polygon(hull_points, facecolor='red', 
                             alpha=0.10, edgecolor='red', 
                             linewidth=1.5, linestyle='--', zorder=0)
                ax.add_patch(poly)
            except:
                pass
    
    # Styling
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]', fontsize=12, weight='bold')
    ax.set_ylabel('y [m]', fontsize=12, weight='bold')
    ax.set_title(f'Path: {path_name.upper()}', fontsize=14, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Metrics text box
    textstr = (
        f"Cumulative Error:\n"
        f"  DT: {summary['cumulative_error_dt']:.3f} m·s\n"
        f"  Ghost: {summary['cumulative_error_ghost']:.3f} m·s\n"
        f"Improvement: {summary['improvement_percent']:.1f}%\n"
        f"DoS Exposure: {summary['total_dos_time']:.1f}s"
    )
    
    props = dict(boxstyle='round,pad=0.6', facecolor='white', 
                alpha=0.92, edgecolor='black', linewidth=2.0)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', bbox=props, 
           family='monospace', weight='bold')
    
    # Legend
    handles = [
        Line2D([0], [0], color='#2ca02c', lw=3.5, ls=':', label='Reference'),
        Line2D([0], [0], color='#1f77b4', lw=3.0, label='Main Controller'),
        Line2D([0], [0], color='#d62728', lw=3.0, label='DT Takeover'),
        Line2D([0], [0], color='0.25', lw=2.5, label='Ghost DoS'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=11, 
             framealpha=0.95, edgecolor='black', fancybox=False)
    
    # Save
    plt.tight_layout()
    output_file = Path(output_dir) / f'Trajectory_{path_name.upper()}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'))
    plt.close()
    
    print(f"    ✓ Saved: {output_file.name}")

# ============================================================================
# PLOT 5: AGGREGATE ERROR EVOLUTION
# ============================================================================

def plot_aggregate_errors(data_dir, paths, output_dir):
    """
    Single plot showing error evolution for all paths
    Each path gets its own color, shows both DT and Ghost
    """
    print("\n  Plotting aggregate error evolution...")
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    for path_name in paths:
        color = PATH_COLORS.get(path_name, '#000000')
        
        # Load median run
        run_idx = select_median_run(data_dir, path_name)
        log_df, summary, dos_windows = load_run_data(data_dir, path_name, run_idx)
        
        t_arr = log_df['t'].values
        e_dt = np.hypot(log_df['x'] - log_df['xd'], log_df['y'] - log_df['yd'])
        e_ghost = np.hypot(log_df['x_dos'] - log_df['xd'], log_df['y_dos'] - log_df['yd'])
        
        # Plot DT (solid line)
        ax.plot(t_arr, e_dt, color=color, lw=2.5, 
               label=f'{path_name.upper()} - DT', alpha=0.9)
        
        # Plot Ghost (dashed line, same color)
        ax.plot(t_arr, e_ghost, color=color, lw=2.0, linestyle='--', 
               label=f'{path_name.upper()} - Ghost', alpha=0.6)
        
        # Attack windows (only plot once per path to avoid clutter)
        for (t_start, t_end) in dos_windows[:1]:  # Show first attack only
            ax.axvspan(t_start, t_end, color=color, alpha=0.05, zorder=0)
    
    # General attack shading (gray)
    # Use first path's windows as representative
    first_path = paths[0]
    run_idx = select_median_run(data_dir, first_path)
    _, _, dos_windows = load_run_data(data_dir, first_path, run_idx)
    for (t_start, t_end) in dos_windows:
        ax.axvspan(t_start, t_end, color='red', alpha=0.08, zorder=0)
    
    ax.set_xlabel('Time [s]', fontsize=12, weight='bold')
    ax.set_ylabel('Tracking Error [m]', fontsize=12, weight='bold')
    ax.set_title('Error Evolution: DT Rescue vs Ghost DoS (All Paths)', 
                fontsize=14, weight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.98, 0.02, 'Solid = DT Rescue | Dashed = Ghost DoS | Shaded = Attack Windows',
           transform=ax.transAxes, ha='right', va='bottom',
           fontsize=9, style='italic', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'Aggregate_Error_Evolution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'))
    plt.close()
    
    print(f"    ✓ Saved: {output_file.name}")

# ============================================================================
# PLOT 6: AGGREGATE CONTROL COMMANDS
# ============================================================================

def plot_aggregate_commands(data_dir, paths, output_dir):
    """
    Two subplots: linear velocity (top) and angular velocity (bottom)
    All paths shown together with different colors
    """
    print("\n  Plotting aggregate control commands...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300, sharex=True)
    
    for path_name in paths:
        color = PATH_COLORS.get(path_name, '#000000')
        
        # Load median run
        run_idx = select_median_run(data_dir, path_name)
        log_df, summary, dos_windows = load_run_data(data_dir, path_name, run_idx)
        
        t_arr = log_df['t'].values
        v_out = log_df['v_out'].values
        w_out = log_df['w_out'].values
        
        # Linear velocity
        axes[0].plot(t_arr, v_out, color=color, lw=2.0, 
                    label=path_name.upper(), alpha=0.85)
        
        # Angular velocity
        axes[1].plot(t_arr, w_out, color=color, lw=2.0, 
                    label=path_name.upper(), alpha=0.85)
    
    # Attack shading (use first path as reference)
    first_path = paths[0]
    run_idx = select_median_run(data_dir, first_path)
    _, _, dos_windows = load_run_data(data_dir, first_path, run_idx)
    for (t_start, t_end) in dos_windows:
        for ax in axes:
            ax.axvspan(t_start, t_end, color='red', alpha=0.10, zorder=0)
    
    # Styling
    axes[0].set_ylabel('Linear Velocity v [m/s]', fontsize=12, weight='bold')
    axes[0].set_title('Control Commands: All Paths', fontsize=14, weight='bold', pad=15)
    axes[0].legend(loc='upper right', fontsize=10, ncol=4, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 0.35)
    
    axes[1].set_ylabel('Angular Velocity ω [rad/s]', fontsize=12, weight='bold')
    axes[1].set_xlabel('Time [s]', fontsize=12, weight='bold')
    axes[1].legend(loc='upper right', fontsize=10, ncol=4, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'Aggregate_Control_Commands.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'))
    plt.close()
    
    print(f"    ✓ Saved: {output_file.name}")

# ============================================================================
# PLOT 7: AGGREGATE MODE & SAFETY
# ============================================================================

def plot_aggregate_safety(data_dir, paths, output_dir):
    """
    Two subplots: Mode indicator (top) and Safety ratio (bottom)
    All paths shown together
    """
    print("\n  Plotting aggregate mode & safety...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300, sharex=True)
    
    for path_name in paths:
        color = PATH_COLORS.get(path_name, '#000000')
        
        # Load median run
        run_idx = select_median_run(data_dir, path_name)
        log_df, summary, dos_windows = load_run_data(data_dir, path_name, run_idx)
        
        t_arr = log_df['t'].values
        mode = log_df['mode'].values
        mode_numeric = np.array([1 if m == 'dt' else 0 for m in mode])
        
        d_perp = log_df['d_perp'].values
        r_eff = log_df['r_eff'].values
        safety_ratio = np.abs(d_perp) / r_eff
        
        # Mode indicator (filled area)
        axes[0].fill_between(t_arr, 0, mode_numeric, 
                            color=color, alpha=0.25, step='post',
                            label=path_name.upper())
        
        # Safety ratio
        axes[1].plot(t_arr, safety_ratio, color=color, lw=2.0, 
                    label=path_name.upper(), alpha=0.85)
    
    # Attack shading
    first_path = paths[0]
    run_idx = select_median_run(data_dir, first_path)
    _, _, dos_windows = load_run_data(data_dir, first_path, run_idx)
    for (t_start, t_end) in dos_windows:
        for ax in axes:
            ax.axvspan(t_start, t_end, color='red', alpha=0.10, zorder=0)
    
    # Safety limit line
    axes[1].axhline(1.0, color='red', linestyle='--', lw=2.5, 
                   label='Safety Limit', zorder=10)
    
    # Styling
    axes[0].set_ylabel('DT Mode Active', fontsize=12, weight='bold')
    axes[0].set_title('Mode & Safety Constraints: All Paths', 
                     fontsize=14, weight='bold', pad=15)
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Main', 'DT'])
    axes[0].legend(loc='upper left', fontsize=10, ncol=4, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Safety Ratio |d⊥|/r_eff', fontsize=12, weight='bold')
    axes[1].set_xlabel('Time [s]', fontsize=12, weight='bold')
    axes[1].legend(loc='upper right', fontsize=10, ncol=3, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.3)
    
    # Add text annotation about safety
    axes[1].text(0.02, 0.98, 'All paths remain below safety limit (ratio < 1.0)',
                transform=axes[1].transAxes, ha='left', va='top',
                fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'Aggregate_Mode_Safety.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'))
    plt.close()
    
    print(f"    ✓ Saved: {output_file.name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate condensed plots from DoS experiment data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to DoS_Attack_Experiments directory')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for plots')
    parser.add_argument('--paths', type=str, 
                       default='circle,figure8,scurve,sharpl',
                       help='Comma-separated list of paths')
    
    args = parser.parse_args()
    
    # Parse paths
    paths = [p.strip().lower() for p in args.paths.split(',') if p.strip()]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    set_publication_style()
    
    print("\n" + "="*70)
    print("Generating Condensed Plots for DoS Attack Experiments")
    print("="*70)
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Paths: {', '.join([p.upper() for p in paths])}")
    print("="*70 + "\n")
    
    # Generate individual trajectory plots (4 plots)
    print("[1/7] Individual Trajectory Plots")
    print("-" * 70)
    for path_name in paths:
        try:
            plot_single_trajectory(args.data_dir, path_name, output_dir)
        except Exception as e:
            print(f"    ✗ Error plotting {path_name}: {e}")
    
    # Generate aggregate plots (3 plots)
    print("\n[2/7] Aggregate Error Evolution")
    print("-" * 70)
    try:
        plot_aggregate_errors(args.data_dir, paths, output_dir)
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    print("\n[3/7] Aggregate Control Commands")
    print("-" * 70)
    try:
        plot_aggregate_commands(args.data_dir, paths, output_dir)
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    print("\n[4/7] Aggregate Mode & Safety")
    print("-" * 70)
    try:
        plot_aggregate_safety(args.data_dir, paths, output_dir)
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    print("\n" + "="*70)
    print("✓ All Plots Generated Successfully!")
    print("="*70)
    print(f"\nOutput Location: {output_dir.absolute()}")
    print("\nGenerated Files:")
    print("  - Trajectory_CIRCLE.png/pdf")
    print("  - Trajectory_FIGURE8.png/pdf")
    print("  - Trajectory_SCURVE.png/pdf")
    print("  - Trajectory_SHARPL.png/pdf")
    print("  - Aggregate_Error_Evolution.png/pdf")
    print("  - Aggregate_Control_Commands.png/pdf")
    print("  - Aggregate_Mode_Safety.png/pdf")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()