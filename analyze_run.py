# analyze_run.py
import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt

def load_log(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    def col(name, cast=float):
        return np.array([cast(r[name]) for r in rows])
    return rows, col

def main(log_csv):
    rows, col = load_log(log_csv)
    t = col("t")
    x, y = col("x"), col("y")
    xr, yr = col("x_ref"), col("y_ref")
    ex, ey, eth = col("ex"), col("ey"), col("eth")
    d_perp, r_eff, ct, kappa = col("d_perp"), col("r_eff"), col("ct"), col("kappa")
    v_out, w_out = col("v_out"), col("w_out")
    v_main, w_main = col("v_main"), col("w_main")
    attack = col("attack", int)

    e_norm = np.sqrt(ex**2 + ey**2)

    # 1) XY path with DT/Main coloring
    fig1, ax1 = plt.subplots(figsize=(6,6))
    ax1.plot(xr, yr, "g:", linewidth=1.5, label="reference")
    # color segments by attack flag (as proxy for DT mode)
    for i in range(len(x)-1):
        c = "r" if attack[i] == 1 else "b"
        ax1.plot(x[i:i+2], y[i:i+2], color=c, linewidth=2)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Trajectory (blue=Main, red=DT)")
    ax1.legend()
    fig1.tight_layout()

    # 2) Errors
    fig2, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    ax[0].plot(t, e_norm); ax[0].set_ylabel(r"$\|e\|$ [m]")
    ax[1].plot(t, d_perp); ax[1].plot(t, r_eff, 'k--', linewidth=0.8); ax[1].plot(t, -r_eff, 'k--', linewidth=0.8)
    ax[1].set_ylabel(r"$d_\perp$ [m]")
    ax[2].plot(t, eth); ax[2].set_ylabel(r"$e_\theta$ [rad]"); ax[2].set_xlabel("time [s]")
    ax[0].set_title("Tracking errors")
    fig2.tight_layout()

    # 3) Commands and curvature
    fig3, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    ax[0].plot(t, v_main, '0.6', label="v_main"); ax[0].plot(t, v_out, 'b', label="v_out")
    ax[0].set_ylabel("v [m/s]"); ax[0].legend(loc="upper right")
    ax[1].plot(t, w_main, '0.6', label="w_main"); ax[1].plot(t, w_out, 'b', label="w_out")
    ax[1].set_ylabel("w [rad/s]"); ax[1].legend(loc="upper right")
    ax[2].plot(t, kappa); ax[2].set_ylabel("κ [1/m]"); ax[2].set_xlabel("time [s]")
    ax[0].set_title("Commands and curvature")
    fig3.tight_layout()

    outdir = os.path.dirname(log_csv)
    fig1.savefig(os.path.join(outdir, "plot_xy.png"), dpi=180)
    fig2.savefig(os.path.join(outdir, "plot_errors.png"), dpi=180)
    fig3.savefig(os.path.join(outdir, "plot_cmds.png"), dpi=180)
    print(f"Saved plots to {outdir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_run.py results/<run_id>/log.csv")
        sys.exit(1)
    main(sys.argv[1])

plt.show()

