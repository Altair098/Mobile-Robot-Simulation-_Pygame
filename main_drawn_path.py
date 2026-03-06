# main_drawn_path.py
import pygame, math, time
import numpy as np
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from generic_trajectory import GenericTrajectory
from dt_rescue3 import DTRescueController
from path_input import FreehandPathRecorder
from math import cos, sin
from datetime import datetime
from config import (
    gains, gamma, d_min, d_max, initial_depth,
    lambda_v, lambda_omega, dt, total_time
)
os.makedirs("results", exist_ok=True)
# ------------------ Pygame Setup ------------------
pygame.init()
screen_width, screen_height = 800, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Draw a Path → DT-Rescue Tracking")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 16)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", run_id)
os.makedirs(results_dir, exist_ok=True)
log_rows = []  # will store one dict per tick



SCALE = 100.0  # pixels per meter
def world_to_screen(x, y):
    return int(screen_width/2 + x*SCALE), int(screen_height/2 - y*SCALE)

def screen_to_world(px, py):
    return (px - screen_width/2)/SCALE, (screen_height/2 - py)/SCALE

def draw_car(surface, x, y, theta, color, scale=12):
    """
    Draws a triangle arrow (car-like) at (x,y) pointing at heading theta.
    (x,y) are in screen coords. scale controls size in pixels.
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Triangle points in local coords
    pts = [
        ( scale, 0),        # nose
        (-scale*0.6, scale*0.5),  # rear left
        (-scale*0.6,-scale*0.5)   # rear right
    ]
    # Rotate + translate
    pts_rot = []
    for px, py in pts:
        sx = x + px*cos_t - py*sin_t
        sy = y + px*sin_t + py*cos_t
        pts_rot.append((int(sx), int(sy)))
    pygame.draw.polygon(surface, color, pts_rot)


# ------------------ 1) Draw path ------------------
rec = FreehandPathRecorder(screen, screen_to_world)
waypoints_world = None
while waypoints_world is None:
    waypoints_world = rec.run()

# ------------------ 2) Build trajectory ------------------
traj = GenericTrajectory.from_waypoints(waypoints_world, closed=False, speed=lambda_v)

# Desired path samples (green)
s_grid = np.linspace(0.0, 1.0, 500)
path_points = [world_to_screen(*traj.sample(s)[0][:2]) for s in s_grid]

# ------------------ 3) Init system ------------------
x0, y0 = waypoints_world[0]
t_hat, _ = traj.tangent_normal(0.0)
theta0 = math.atan2(t_hat[1], t_hat[0])

main_robot = Robot()
dt_robot   = Robot()

def _set_robot_pose_safe(robot, x, y, th):
    if hasattr(robot, "set_pose"):
        try:
            robot.set_pose(x, y, th)
        except TypeError:
            robot.set_pose((x, y, th))
    else:
        if hasattr(robot, "x"):     robot.x = x
        if hasattr(robot, "y"):     robot.y = y
        if hasattr(robot, "theta"): robot.theta = th

_set_robot_pose_safe(main_robot, x0, y0, theta0)
_set_robot_pose_safe(dt_robot,   x0, y0, theta0)

controller = DTRescueController(dt_robot, traj, dt=dt)

# Initialize phase at start point and sync DT
s_phase = float(traj.nearest_s(x0, y0))
controller.s = float(s_phase)
controller.x_hat = np.array([x0, y0, theta0], dtype=float)

d_hat = float(initial_depth)

attack_active = False
attack_start_time = None
attack_duration = 8.0

TRAIL_MAX = 4000
trail_main, trail_dt = [], []   # blue for Main, red for DT

start_time = time.time()
running = True
t = 0.0

# ------------------ 4) Main Loop ------------------
while running and t < total_time:

    t = time.time() - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_a and not attack_active:
                attack_active = True
                attack_start_time = t
                controller.set_depth_from_main(d_hat)

    if attack_start_time is not None:
        attack_active = (attack_start_time <= t <= (attack_start_time + attack_duration))
    else:
        attack_active = False

    # ---- Robot pose (BEFORE control) ----
    x, y, theta = main_robot.get_pose()

    # --- heading-pruned local projection (no cross-lobe jumps, no backward snaps)
    if hasattr(traj, "nearest_s_pruned"):
        s_near = traj.nearest_s_pruned(x, y, theta, s_phase, max_jump=0.08, cone_deg=70.0)
    else:
        s_near = traj.nearest_s_local(x, y, s_phase, max_jump=0.08)

    # --- freeze forward progress until facing roughly forward
    th_path = traj.sample(s_phase)[0][2]
    c_t = math.cos(th_path)*math.cos(theta) + math.sin(th_path)*math.sin(theta)
    if c_t < 0.2:
        s_near = min(s_near, s_phase)   # don’t advance while pointing backward

    # --- smooth and clamp
    s_phase += 0.30*(s_near - s_phase)
    s_phase = float(np.clip(s_phase, 0.0, 0.999))
    controller.s = s_phase

    # ---- Desired state at s (curvature-capped speed) ----
    x_d, _, _ = traj.sample(s_phase)
    theta_d   = float(x_d[2])
    kappa     = float(traj.curvature(s_phase))
    v_d_star  = min(lambda_v, lambda_omega / (abs(kappa) + 1e-6))
    omega_d   = v_d_star * kappa

    t_x, t_y = math.cos(theta_d), math.sin(theta_d)
    n_x, n_y = -t_y, t_x
    c_t = t_x*math.cos(theta) + t_y*math.sin(theta)
    d_perp = (x - x_d[0]) * n_x + (y - x_d[1]) * n_y

    # match tube used in DT controller (keep constants in sync)
    r0, a_kappa, b_align = 0.16, 0.90, 0.30
    r_eff = r0 * (1.0 + a_kappa * abs(kappa)) * (1.0 - b_align * max(0.0, c_t))
    r_eff = max(0.75*r0, min(r_eff, 2.0*r0))

    # diagnostics for logging
    # t_x, t_y = math.cos(theta_d), math.sin(theta_d)
    # n_x, n_y = -t_y, t_x
    # c_t = t_x*math.cos(theta) + t_y*math.sin(theta)          # alignment
    # d_perp = (x - x_d[0]) * n_x + (y - x_d[1]) * n_y         # signed normal error

    # match DT tube used in dt_rescue3.py (r0,a,b must mirror controller)
    

    # ---- Errors in robot frame ----
    dx = x_d[0] - x
    dy = x_d[1] - y
    e_x =  math.cos(theta)*dx + math.sin(theta)*dy
    e_y = -math.sin(theta)*dx + math.cos(theta)*dy
    e_th = (theta_d - theta + math.pi) % (2*math.pi) - math.pi

    # ---- Main controller (healthy) ----
    v, omega = compute_control(e_x, e_y, e_th, v_d_star, omega_d, d_hat, gains)
    v     = max(min(v,     lambda_v), -lambda_v)
    omega = max(min(omega, lambda_omega), -lambda_omega)
    u_main = np.array([v, omega], dtype=float)

    # ---- DT-Rescue decides who drives ----
    u_out = controller.step(t, attack_active, np.array([x, y, theta], dtype=float), u_main)

    mode_flag = ("dt" if attack_active else "main") if not hasattr(controller, "mode") else controller.mode
    


    # ---- Apply to robot ----
    main_robot.update(u_out[0], u_out[1], dt)

    # ---- Pose AFTER update (for drawing/logging) ----
    x_cur, y_cur, _ = main_robot.get_pose()

    # mode_flag = controller.mode if hasattr(controller, "mode") else ("dt" if attack_active else "main")
    log_rows.append({
        "t": float(t),
        "mode": mode_flag,
        "attack": int(attack_active),
        "s": float(s_phase),
        "x": float(x_cur), "y": float(y_cur),
        "x_ref": float(x_d[0]), "y_ref": float(x_d[1]),
        "ex": float(e_x), "ey": float(e_y), "eth": float(e_th),
        "d_perp": float(d_perp), "r_eff": float(r_eff),
        "ct": float(c_t), "kappa": float(kappa),
        "v_out": float(u_out[0]), "w_out": float(u_out[1]),
        "v_main": float(u_main[0]), "w_main": float(u_main[1]),
    })

    if controller.mode == 'main':
        trail_main.append(world_to_screen(x_cur, y_cur))
        if len(trail_main) > TRAIL_MAX:
            trail_main.pop(0)
    else:
        trail_dt.append(world_to_screen(x_cur, y_cur))
        if len(trail_dt) > TRAIL_MAX:
            trail_dt.pop(0)

    # ---- Adaptation ----
    if not attack_active:
        d_hat = update_depth_estimate(
            d_hat, e_x, e_y, e_th, v_d_star, gamma, gains['k3'], d_min, d_max, dt
        )

    # ---- Render ----
    screen.fill((255,255,255))

    # desired path (green)
    for pt in path_points:
        pygame.draw.circle(screen, (0,200,0), pt, 1)

    # trails: Main=blue, DT=red
    for pt in trail_main[::3]:
        pygame.draw.circle(screen, (0,102,204), pt, 2)
    for pt in trail_dt[::3]:
        pygame.draw.circle(screen, (255,0,0), pt, 2)

    # # robot markers
    pygame.draw.circle(screen, (255,0,0), world_to_screen(x_cur, y_cur), 4)                 # main pose
    pygame.draw.circle(screen, (0,102,204),   world_to_screen(*controller.x_hat[:2]), 4)        # DT pose
    


    # HUD
    hud = [
        f"t={t:5.2f}s  mode={controller.mode}  attack={'ON' if attack_active else 'off'}",
        f"s={s_phase:0.3f}  v_out={u_out[0]: .2f}  w_out={u_out[1]: .2f}"
    ]
    for i, txt in enumerate(hud):
        screen.blit(font.render(txt, True, (30,30,30)), (8, 8 + i*18))

    pygame.display.flip()
    clock.tick(int(1/dt))


    



# Save outputs
# ---- SAVE last frame and log once ----
pygame.image.save(screen, os.path.join(results_dir, "final_frame.png"))

with open(os.path.join(results_dir, "log.csv"), "w", newline="") as f:
    fieldnames = list(log_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(log_rows)


# ---- SAVE last frame and log once ----
os.makedirs(results_dir, exist_ok=True)
pygame.image.save(screen, os.path.join(results_dir, "final_frame.png"))

if not log_rows:
    pygame.display.quit()
    pygame.quit()
    raise RuntimeError("No data collected; simulation ended before first tick.")

with open(os.path.join(results_dir, "log.csv"), "w", newline="") as f:
    fieldnames = list(log_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(log_rows)

# Close Pygame before plotting
pygame.display.quit()
pygame.quit()

# ---------- GENERATE PLOTS ----------
t       = np.array([r["t"] for r in log_rows])
x       = np.array([r["x"] for r in log_rows])
y       = np.array([r["y"] for r in log_rows])
xr      = np.array([r["x_ref"] for r in log_rows])
yr      = np.array([r["y_ref"] for r in log_rows])
ex      = np.array([r["ex"] for r in log_rows])
ey      = np.array([r["ey"] for r in log_rows])
eth     = np.array([r["eth"] for r in log_rows])
d_perp  = np.array([r["d_perp"] for r in log_rows])
r_eff   = np.array([r["r_eff"] for r in log_rows])
kappa   = np.array([r["kappa"] for r in log_rows])
v_out   = np.array([r["v_out"] for r in log_rows])
w_out   = np.array([r["w_out"] for r in log_rows])
v_main  = np.array([r["v_main"] for r in log_rows])
w_main  = np.array([r["w_main"] for r in log_rows])
attack  = np.array([r["attack"] for r in log_rows], dtype=int)

e_norm = np.sqrt(ex**2 + ey**2)

# 1) XY
fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.plot(xr, yr, "g:", linewidth=1.5, label="reference")
for i in range(len(x)-1):
    c = "r" if attack[i] == 1 else "b"
    ax1.plot(x[i:i+2], y[i:i+2], color=c, linewidth=2)
ax1.set_aspect("equal", adjustable="box")
ax1.set_title("Trajectory (blue=Main, red=DT)")
ax1.legend()
fig1.tight_layout()
fig1.savefig(os.path.join(results_dir, "plot_xy.png"), dpi=180)
plt.close(fig1)

# 2) Errors
fig2, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
ax[0].plot(t, e_norm); ax[0].set_ylabel(r"$||e||$ [m]")
ax[1].plot(t, d_perp); ax[1].plot(t, r_eff, 'k--', linewidth=0.8); ax[1].plot(t, -r_eff, 'k--', linewidth=0.8)
ax[1].set_ylabel(r"$d_\perp$ [m]")
ax[2].plot(t, eth); ax[2].set_ylabel(r"$e_\theta$ [rad]"); ax[2].set_xlabel("time [s]")
ax[0].set_title("Tracking errors")
fig2.tight_layout()
fig2.savefig(os.path.join(results_dir, "plot_errors.png"), dpi=180)
plt.close(fig2)

# 3) Commands and curvature
fig3, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
ax[0].plot(t, v_main, '0.6', label="v_main"); ax[0].plot(t, v_out, 'b', label="v_out")
ax[0].set_ylabel("v [m/s]"); ax[0].legend(loc="upper right")
ax[1].plot(t, w_main, '0.6', label="w_main"); ax[1].plot(t, w_out, 'b', label="w_out")
ax[1].set_ylabel("w [rad/s]"); ax[1].legend(loc="upper right")
ax[2].plot(t, kappa); ax[2].set_ylabel("κ [1/m]"); ax[2].set_xlabel("time [s]")
ax[0].set_title("Commands and curvature")
fig3.tight_layout()
fig3.savefig(os.path.join(results_dir, "plot_cmds.png"), dpi=180)
plt.close(fig3)

print(f"Saved plots and logs to {results_dir}")


# finally close pygame
#pygame.quit()
if not log_rows:
    pygame.display.quit()
    pygame.quit()
    raise RuntimeError("No data collected; simulation ended before first tick.")

# ---------- METRICS ----------
import json

def pct95(a): 
    return float(np.percentile(a, 95)) if len(a) else float("nan")

total_time = float(t[-1] - t[0]) if len(t) > 1 else 0.0
dt_mask    = (attack == 1)                         # DT active
mn_mask    = (attack == 0)                         # Main active
valid_diff = len(t) > 1

# progress (use s in [0,1])
s_arr = np.array([r["s"] for r in log_rows])
ds    = np.diff(s_arr) / np.maximum(1e-9, np.diff(t)) if valid_diff else np.array([])
ds_dt   = ds[1:][dt_mask[2:]]   if len(ds) > 2 else np.array([])
ds_main = ds[1:][mn_mask[2:]]   if len(ds) > 2 else np.array([])

# tube violations
viol = (np.abs(d_perp) > r_eff)
viol_rate = float(np.mean(viol)) if len(viol) else 0.0
viol_time = viol_rate * total_time

# speed/turn matching during DT
vm_diff_dt = (v_out - v_main)[dt_mask]
wm_diff_dt = (w_out - w_main)[dt_mask]

# smoothness (variation) – smaller is smoother
if valid_diff:
    dv_dt = np.diff(v_out) / np.maximum(1e-9, np.diff(t))
    dw_dt = np.diff(w_out) / np.maximum(1e-9, np.diff(t))
    smooth_v = float(np.mean(np.abs(dv_dt)))
    smooth_w = float(np.mean(np.abs(dw_dt)))
else:
    smooth_v = smooth_w = float("nan")

metrics = {
    "duration_s": total_time,
    "final_progress_s": float(s_arr[-1]) if len(s_arr) else 0.0,
    "time_fraction_dt": float(np.mean(dt_mask)) if len(dt_mask) else 0.0,

    # Tracking error
    "e_norm_rms_m":  float(np.sqrt(np.mean(e_norm**2))) if len(e_norm) else 0.0,
    "e_norm_p95_m":  pct95(e_norm),
    "e_norm_max_m":  float(np.max(e_norm)) if len(e_norm) else 0.0,

    # Lateral & heading
    "d_perp_rms_m":  float(np.sqrt(np.mean(d_perp**2))) if len(d_perp) else 0.0,
    "d_perp_p95_m":  pct95(np.abs(d_perp)),
    "e_theta_rms_rad": float(np.sqrt(np.mean(eth**2))) if len(eth) else 0.0,
    "e_theta_p95_rad": pct95(np.abs(eth)),

    # Tube safety
    "tube_violation_rate": viol_rate,
    "tube_violation_time_s": viol_time,

    # Progress speed
    "progress_rate_dt_mean":  float(np.mean(ds_dt))   if len(ds_dt)   else float("nan"),
    "progress_rate_main_mean":float(np.mean(ds_main)) if len(ds_main) else float("nan"),
    "progress_rate_ratio_dt_over_main": (
        float(np.mean(ds_dt)/max(1e-9, np.mean(ds_main))) 
        if len(ds_dt) and len(ds_main) and np.mean(ds_main)>0 else float("nan")
    ),

    # Command similarity during DT
    "v_dt_minus_main_rms": float(np.sqrt(np.mean(vm_diff_dt**2))) if len(vm_diff_dt) else float("nan"),
    "w_dt_minus_main_rms": float(np.sqrt(np.mean(wm_diff_dt**2))) if len(wm_diff_dt) else float("nan"),

    # Smoothness of applied commands
    "smooth_v_mean_abs_dvdt": smooth_v,
    "smooth_w_mean_abs_dwdt": smooth_w,
}

# Save JSON + a readable TXT
with open(os.path.join(results_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

print("Saved metrics to", os.path.join(results_dir, "metrics.json"))

