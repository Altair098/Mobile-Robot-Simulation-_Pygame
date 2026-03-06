# make_paths.py
# Generate and save clean PNGs for: circle, figure-8 (∞), S-curve, and sharp-L

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

OUTDIR = "paths_final"
COLOR = "#1f77b4"
LW = 8
DPI = 300

def _save_path(x, y, fname, figsize=(8, 8), pad=0.06):
    os.makedirs(OUTDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color=COLOR, linewidth=LW, solid_capstyle="round")
    ax.set_aspect("equal", adjustable="box")
    # tight limits with padding
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xr = xmax - xmin if xmax > xmin else 1.0
    yr = ymax - ymin if ymax > ymin else 1.0
    ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
    ax.set_ylim(ymin - pad * yr, ymax + pad * yr)
    ax.axis("off")
    fig.savefig(os.path.join(OUTDIR, fname), dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# 1) Circle
def make_circle():
    R = 1.2
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=True)
    x = R * np.cos(t)
    y = R * np.sin(t)
    _save_path(x, y, "circle.png")

# 2) Figure-8 (∞) — lemniscate of Gerono (horizontal)
def make_figure8():
    a = 1.0
    t = np.linspace(0, 2 * np.pi, 2500, endpoint=True)
    x = a * np.sin(t)
    y = a * np.sin(t) * np.cos(t)
    _save_path(x, y, "figure8.png")

# 3) S-curve — smooth cubic spline through “S” control points
def make_s_curve():
    pts = np.array([
        [0.50, 1.00],  # top
        [0.30, 0.95],
        [0.20, 0.85],
        [0.20, 0.75],
        [0.30, 0.65],
        [0.50, 0.60],  # mid crossing
        [0.70, 0.55],
        [0.80, 0.45],
        [0.80, 0.35],
        [0.70, 0.25],
        [0.50, 0.20],
        [0.30, 0.15],
        [0.50, 0.00]   # bottom
    ])
    tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=3)
    u_new = np.linspace(0, 1, 2000)
    x_s, y_s = splev(u_new, tck)
    _save_path(np.array(x_s), np.array(y_s), "s_curve.png", figsize=(8, 10))

# 4) Sharp-L — exact right-angle polyline (no rounding)
def make_sharpL():
    # Horizontal segment (left→right), then vertical (top→down)
    x1 = np.linspace(-2.5, 1.0, 800)          # y = 1.0
    y1 = np.full_like(x1, 1.0)
    y2 = np.linspace(1.0, -2.5, 800)          # x = 1.0
    x2 = np.full_like(y2, 1.0)
    # Concatenate; include corner once
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    _save_path(x, y, "sharpL.png", figsize=(8, 10))

if __name__ == "__main__":
    make_circle()
    make_figure8()
    make_s_curve()
    make_sharpL()
    print(f"Saved PNGs to: {os.path.abspath(OUTDIR)}")
