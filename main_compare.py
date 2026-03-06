# main_compare.py
import os, time, math, csv
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from generic_trajectory import GenericTrajectory
from dt_rescue3 import DTRescueController   
from path_input import FreehandPathRecorder
from attack_model import FDIInjector
from datetime import datetime
from config import (
    gains, gamma, d_min, d_max, initial_depth,
    lambda_v, lambda_omega, dt, total_time
)

# ---------- setup ----------
pygame.init()
W, H = 800, 800
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("DT vs Baselines: Main + Ghost DoS + Ghost FDI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 16)

SCALE = 100.0
def w2s(x, y): return int(W/2 + x*SCALE), int(H/2 - y*SCALE)
def s2w(px,py): return (px - W/2)/SCALE, (H/2 - py)/SCALE

def draw_dot(surface, x,y, color, r=3):
    pygame.draw.circle(surface, color, (x,y), r)

# ---------- path input ----------
rec = FreehandPathRecorder(screen, s2w)
waypoints = None
while waypoints is None:
    waypoints = rec.run()

traj = GenericTrajectory.from_waypoints(waypoints, closed=False, speed=lambda_v)
s_grid = np.linspace(0,1,500)
path_points = [w2s(*traj.sample(s)[0][:2]) for s in s_grid]

# ---------- robots ----------
real = Robot()
ghost_dos = Robot()
ghost_fdi = Robot()

x0, y0 = waypoints[0]
t_hat,_ = traj.tangent_normal(0.0)
th0 = math.atan2(t_hat[1], t_hat[0])
for r in (real, ghost_dos, ghost_fdi):
    try:
        r.set_pose(x0,y0,th0)
    except TypeError:
        r.set_pose((x0,y0,th0))

controller = DTRescueController(real, traj, dt=dt)
controller.s = float(traj.nearest_s(x0,y0))
controller.x_hat = np.array([x0,y0,th0], float)

d_hat = float(initial_depth)

# attacks
attack_DoS = False
DoS_start, DoS_duration = None, 6.0
injector = FDIInjector(kind="drift", amp=0.25, omega=0.5, noise_std=(0.0,0.0,0.0))
FDI_active = False
FDI_start, FDI_duration = None, 8.0

# ---------- logs ----------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
root_dir = "results_compare"
os.makedirs(root_dir, exist_ok=True)
outdir = os.path.join(root_dir, run_id)
os.makedirs(outdir, exist_ok=True)
log = []

trail_main, trail_dt = [], []
trail_dos, trail_fdi = [], []
TRAIL_MAX = 6000

t0 = time.time()
running = True
s_phase = controller.s

while running and (time.time()-t0) < total_time:
    t = time.time() - t0
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running=False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q:
                running=False
            elif e.key == pygame.K_a and not attack_DoS:
                attack_DoS, DoS_start = True, t  # start DoS window
            elif e.key == pygame.K_f and not FDI_active:
                FDI_active, FDI_start = True, t  # start FDI window

    # windows
    attack_DoS = False if (DoS_start is None) else (DoS_start <= t <= DoS_start+DoS_duration)
    FDI_active = False if (FDI_start is None) else (FDI_start <= t <= FDI_start+FDI_duration)

    # ---------- phase update (shared) ----------
    xr, yr, thr = real.get_pose()
    s_near = traj.nearest_s_local(xr, yr, s_phase, max_jump=0.08)
    # freeze forward progress if pointing backward
    th_path = traj.sample(s_phase)[0][2]
    c_t = math.cos(th_path)*math.cos(thr) + math.sin(th_path)*math.sin(thr)
    if c_t < 0.2:
        s_near = min(s_near, s_phase)
    s_phase += 0.30*(s_near - s_phase)
    s_phase = float(np.clip(s_phase, 0.0, 0.999))
    controller.s = s_phase

    # ---------- reference ----------
    x_d, _, _ = traj.sample(s_phase)
    theta_d = float(x_d[2])
    kappa = float(traj.curvature(s_phase))
    v_d_star = min(lambda_v, lambda_omega/(abs(kappa)+1e-6))
    omega_d = v_d_star * kappa

    # ---------- MAIN errors and u_main ----------
    dx = x_d[0]-xr; dy = x_d[1]-yr
    e_x =  math.cos(thr)*dx + math.sin(thr)*dy
    e_y = -math.sin(thr)*dx + math.cos(thr)*dy
    e_th = (theta_d - thr + math.pi) % (2*math.pi) - math.pi

    v, w = compute_control(e_x, e_y, e_th, v_d_star, omega_d, d_hat, gains)
    v = np.clip(v, -lambda_v, lambda_v)
    w = np.clip(w, -lambda_omega, lambda_omega)
    u_main = np.array([v, w], float)

    # ---------- DT vs Main on REAL ----------
    u_out = controller.step(t, attack_DoS or FDI_active,
                            np.array([xr,yr,thr],float), u_main)

    # apply to real
    real.update(float(u_out[0]), float(u_out[1]), dt)

    # ---------- GHOST DoS ----------
    u_dos = np.array([0.0,0.0], float) if attack_DoS else u_main.copy()
    ghost_dos.update(float(u_dos[0]), float(u_dos[1]), dt)

    # ---------- GHOST FDI ----------
    pose_true_fdi = np.array(ghost_fdi.get_pose(), float)
    pose_meas_fdi = injector.corrupt(pose_true_fdi, t) if FDI_active else pose_true_fdi.copy()
    xm, ym, thm = pose_meas_fdi
    dxm = x_d[0]-xm; dym = x_d[1]-ym
    e_xm =  math.cos(thm)*dxm + math.sin(thm)*dym
    e_ym = -math.sin(thm)*dxm + math.cos(thm)*dym
    e_thm= (theta_d - thm + math.pi) % (2*math.pi) - math.pi
    v_fdi, w_fdi = compute_control(e_xm, e_ym, e_thm, v_d_star, omega_d, d_hat, gains)
    v_fdi = np.clip(v_fdi, -lambda_v, lambda_v)
    w_fdi = np.clip(w_fdi, -lambda_omega, lambda_omega)
    ghost_fdi.update(float(v_fdi), float(w_fdi), dt)

    # ---------- log ----------
    xr2, yr2, _ = real.get_pose()
    xdos, ydos, _ = ghost_dos.get_pose()
    xfdi, yfdi, _ = ghost_fdi.get_pose()
    log.append({
        "t":t, "s":s_phase,
        "attack_dos":int(attack_DoS), "attack_fdi":int(FDI_active),
        "x":xr2, "y":yr2, "xd":x_d[0], "yd":x_d[1],
        "x_dos":xdos, "y_dos":ydos, "x_fdi":xfdi, "y_fdi":yfdi,
        "mode":controller.mode, "v_out":float(u_out[0]), "w_out":float(u_out[1])
    })

    # ---------- trails ----------
    if controller.mode == 'main':
        trail_main.append(w2s(xr2, yr2));  trail_main = trail_main[-TRAIL_MAX:]
    else:
        trail_dt.append(w2s(xr2, yr2));    trail_dt = trail_dt[-TRAIL_MAX:]

    trail_dos.append(w2s(xdos, ydos)); trail_dos = trail_dos[-TRAIL_MAX:]
    trail_fdi.append(w2s(xfdi, yfdi)); trail_fdi = trail_fdi[-TRAIL_MAX:]

    # ---------- adapt depth when healthy ----------
    if controller.mode == 'main':
        d_hat = update_depth_estimate(d_hat, e_x, e_y, e_th,
                                      v_d_star, gamma, gains['k3'],
                                      d_min, d_max, dt)

    # ---------- draw ----------
    screen.fill((255,255,255))
    for pt in path_points: pygame.draw.circle(screen, (0,200,0), pt, 1)
    for pt in trail_dos[::3]:  draw_dot(screen, pt[0], pt[1], (70,70,70), 2)     # dark gray
    for pt in trail_fdi[::3]:  draw_dot(screen, pt[0], pt[1], (160,160,160), 2)  # light gray
    for pt in trail_main[::3]: draw_dot(screen, pt[0], pt[1], (0,102,204), 2)    # blue
    for pt in trail_dt[::3]:   draw_dot(screen, pt[0], pt[1], (255,0,0), 2)      # red

    color_now = (0,102,204) if controller.mode=='main' else (255,0,0)
    xpix, ypix = w2s(xr2, yr2)
    draw_dot(screen, xpix, ypix, color_now, 5)

    hud = f"t={t:5.2f}s   mode={controller.mode}   DoS={'ON' if attack_DoS else 'off'}   FDI={'ON' if FDI_active else 'off'}"
    screen.blit(font.render(hud, True, (30,30,30)), (8,8))
    pygame.display.flip()
    clock.tick(int(1/dt))

# ---------- save ----------
pygame.image.save(screen, os.path.join(outdir, "final_frame.png"))
with open(os.path.join(outdir, "log.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
    w.writeheader()
    w.writerows(log)

# plots
t   = np.array([r["t"] for r in log])
x   = np.array([r["x"] for r in log]);   y   = np.array([r["y"] for r in log])
xd  = np.array([r["xd"] for r in log]);  yd  = np.array([r["yd"] for r in log])
xdos= np.array([r["x_dos"] for r in log]); ydos= np.array([r["y_dos"] for r in log])
xfdi= np.array([r["x_fdi"] for r in log]); yfdi= np.array([r["y_fdi"] for r in log])

plt.figure(figsize=(6,6))
plt.plot(xd, yd, "g:", label="ref")
plt.plot(x, y,  "b",  lw=1.5, label="real (Main/DT)")
plt.plot(xdos, ydos, color="0.25", lw=1, label="ghost DoS")
plt.plot(xfdi, yfdi, color="0.6",  lw=1, label="ghost FDI")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend(); plt.title("Comparison")
plt.tight_layout(); plt.savefig(os.path.join(outdir, "compare_xy.png"), dpi=180)
print("Plots saved in:", outdir)

print(f"Saved to {outdir}")

# --- Extra plots: time-series errors and mode/attacks ---
e_real = np.hypot(x - xd, y - yd)
e_dos  = np.hypot(xdos - xd, ydos - yd)
e_fdi  = np.hypot(xfdi - xd, yfdi - yd)
attack_dos = np.array([r["attack_dos"] for r in log], int)
attack_fdi = np.array([r["attack_fdi"] for r in log], int)
v_out = np.array([r["v_out"] for r in log], float)
w_out = np.array([r["w_out"] for r in log], float)
mode  = np.array([1 if r["mode"]=="dt" else 0 for r in log], int)  # 1=DT, 0=Main

# Errors over time
fig, ax = plt.subplots(3, 1, figsize=(9,7), sharex=True)
ax[0].plot(t, e_real, label="real err"); ax[0].set_ylabel("|e| real [m]")
ax[1].plot(t, e_dos,  color="0.25", label="ghost DoS err"); ax[1].set_ylabel("|e| DoS [m]")
ax[2].plot(t, e_fdi,  color="0.6",  label="ghost FDI err"); ax[2].set_ylabel("|e| FDI [m]"); ax[2].set_xlabel("time [s]")
for a in ax:
    a.legend(loc="upper right")
    # shade attacks
    a.fill_between(t, 0, 1, where=attack_dos>0, color="0.2", alpha=0.08, transform=a.get_xaxis_transform(), label=None)
    a.fill_between(t, 0, 1, where=attack_fdi>0, color="0.6", alpha=0.08, transform=a.get_xaxis_transform(), label=None)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "compare_errors.png"), dpi=180)

# Commands + mode
fig2, ax2 = plt.subplots(3, 1, figsize=(9,7), sharex=True)
ax2[0].plot(t, v_out, label="v_out"); ax2[0].set_ylabel("v [m/s]"); ax2[0].legend()
ax2[1].plot(t, w_out, label="w_out"); ax2[1].set_ylabel("w [rad/s]"); ax2[1].legend()
ax2[2].step(t, mode, where="post"); ax2[2].set_ylabel("mode (1=DT)"); ax2[2].set_xlabel("time [s]")
for a in ax2:
    a.fill_between(t, a.get_ylim()[0], a.get_ylim()[1], where=attack_dos>0, color="0.2", alpha=0.06)
    a.fill_between(t, a.get_ylim()[0], a.get_ylim()[1], where=attack_fdi>0, color="0.6", alpha=0.06)
fig2.tight_layout(); fig2.savefig(os.path.join(outdir, "compare_cmds_mode.png"), dpi=180)

print("Plots saved in:", outdir)

pygame.quit()
