import pygame
import math
import numpy as np
import time
import matplotlib.pyplot as plt








from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from trajectory import TrajectoryGenerator


# use your latest DT controller module here:
from dt_rescue2 import DTRescueController  # <-- pick the newest file you kept


from config import (
    gains, gamma, d_min, d_max, initial_depth,
    lambda_v, lambda_omega, dt, total_time
)


# ------------------ Pygame Setup ------------------
pygame.init()
screen_width, screen_height = 800, 800
screen = pygame.display.set_mode((screen_width, screen_height))


radius = 100
center = (screen_width // 2, screen_height // 2)
circle_points = []
for i in range(0, 360, 2):
    angle = math.radians(i)
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    circle_points.append((int(x), int(y)))


pygame.display.set_caption("Robot with DT-Rescue Control (DoS Simulation)")
clock = pygame.time.Clock()


def world_to_screen(x, y):
    return int(screen_width / 2 + x * 100), int(screen_height / 2 - y * 100)


import numpy as np   # make sure you already have this at the top


def nearest_s_on_polyline(px, py, path_xy, cumlen):
    """
    Project point p=(px,py) onto the polyline path_xy.
    Returns s_near ∈ [0,1] (normalized arc-length).
    """
    p = np.array([px, py], dtype=float)
    N = path_xy.shape[0]


    # Quick coarse guess: closest vertex
    d2 = np.sum((path_xy - p)**2, axis=1)
    i0 = int(np.argmin(d2))


    # Search local window around i0
    i_start = max(0, i0 - 10)
    i_end   = min(N - 1, i0 + 10)


    best_s = 0.0
    best_d2 = float('inf')


    for i in range(i_start, i_end):
        a = path_xy[i]
        b = path_xy[i+1]
        ab = b - a
        L2 = float(np.dot(ab, ab)) + 1e-12
        t = float(np.dot(p - a, ab) / L2)
        t_clamped = max(0.0, min(1.0, t))
        proj = a + t_clamped * ab
        d2_seg = float(np.dot(p - proj, p - proj))
        if d2_seg < best_d2:
            best_d2 = d2_seg
            s_len = cumlen[i] + t_clamped * np.linalg.norm(ab)
            best_s = s_len


    return float(best_s / (cumlen[-1] if cumlen[-1] > 0 else 1.0))




def circle_nearest_s(x, y):
    # For your circular trajectory centered at (0,0)
    ang = math.atan2(y, x)                 # [-pi, pi]
    return (ang % (2*math.pi)) / (2*math.pi)  # [0,1)




WHITE = (255, 255, 255)
BLUE  = (0, 102, 204)
RED   = (255, 0, 0)
BLACK = (0, 0, 0)


# ------------------ Simulation Setup ------------------
main_robot = Robot()
dt_robot   = Robot()  # placeholder; DT uses same kinematics internally
traj = TrajectoryGenerator()
controller = DTRescueController(dt_robot, traj, dt=dt)


# ---- Diagnostics logs ----
log = {
    "t": [], "x": [], "y": [], "th": [],
    "xd": [], "yd": [], "thd": [],
    "e_perp": [], "c_t": [], "h": [], "r_tube": [],
    "s": [], "s_near": [],
    "u_out_v": [], "u_out_w": [],
    "u_main_v": [], "u_main_w": [],
    "u_dt_v": [], "u_dt_w": [],
    "d_main": [], "d_dt": [],
    "mode_dt": [], "attack": []
}




trajectory_log   = []
dt_trajectory_log = []


# ---- depth estimate (PERSISTENT) ----
d_hat = float(initial_depth)              # <-- keep a single evolving estimate


start_time = time.time()
running = True
t = 0.0


attack_active      = False
prev_attack_active = False
attack_start_time  = None
attack_duration    = 10.0


# ---- Build a polyline approximation of the desired path for diagnostics ----
N_PATH_SAMPLES = 800
s_grid = np.linspace(0.0, 1.0, N_PATH_SAMPLES)
path_xy = np.array([traj.sample(s)[0][:2] for s in s_grid])  # (N,2)


# cumulative arc-length
seg = np.diff(path_xy, axis=0)
seg_len = np.linalg.norm(seg, axis=1)
cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])




# ------------------ Main Simulation Loop ------------------
while running and t < total_time:
    screen.fill(WHITE)
    for point in circle_points:
        pygame.draw.circle(screen, (0, 200, 0), point, 1)


    t = time.time() - start_time


    # ---- UI / Events ----
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_a and not attack_active:
                # start attack
                attack_active = True
                attack_start_time = t
                controller.set_depth_from_main(d_hat)   # <<< sync DT with Main depth
                print(f"⚠️ DoS attack started at {t:.2f}s")


    # auto end of attack after duration
    if attack_start_time is not None:
        attack_active = (attack_start_time <= t <= (attack_start_time + attack_duration))
    else:
        attack_active = False


    # ---- Desired trajectory and current pose ----
    x_d_star, y_d_star, theta_d, v_d_star, omega_d = traj.get_desired_state(t)
    x, y, theta = main_robot.get_pose()


    # ---- Tracking error in robot frame ----
    dx = x_d_star - x
    dy = y_d_star - y
    e_x = math.cos(theta) * dx + math.sin(theta) * dy
    e_y = -math.sin(theta) * dx + math.cos(theta) * dy
    e_theta = (theta_d - theta + math.pi) % (2 * math.pi) - math.pi


    # ---- Main controller command (computed for healthy mode) ----
    v, omega = compute_control(e_x, e_y, e_theta, v_d_star, omega_d, d_hat, gains)
    v     = max(min(v,     lambda_v), -lambda_v)
    omega = max(min(omega, lambda_omega), -lambda_omega)
    u_main = np.array([v, omega])


    # ---- DT-Rescue tick (decides who drives + performs handover) ----
    u_out = controller.step(t, attack_active, np.array([x, y, theta]), u_main)


    # ---- Apply the command chosen by DT-Rescue ----
    main_robot.update(u_out[0], u_out[1], dt)




    # --- DIAGNOSTICS: log stability, tracking, and handover quality ---


    # Reference at controller.s (what both controllers are targeting)
    x_d, _, _ = traj.sample(controller.s)
    th_path = x_d[2]


    # Path normal & tangent at the reference
    n_x, n_y = -math.sin(th_path), math.cos(th_path)   # normal
    t_x, t_y =  math.cos(th_path), math.sin(th_path)   # tangent


    # Robot pose error wrt reference
    dxp = x - x_d[0]
    dyp = y - x_d[1]


    # Signed lateral error to path (Frenet)
    e_perp = dxp * n_x + dyp * n_y


    # Tangent alignment c_t ∈ [-1, 1] (1 = perfectly aligned forward)
    c_t = t_x * math.cos(theta) + t_y * math.sin(theta)


    # Curvature estimate around current s (matches controller’s estimate)
    sA = max(0.0, controller.s - 0.01)
    sB = min(1.0, controller.s + 0.01)
    thA = traj.sample(sA)[0][2]
    thB = traj.sample(sB)[0][2]
    dth = (thB - thA + math.pi) % (2 * math.pi) - math.pi
    kappa = abs(dth / max(1e-3, (sB - sA)))


    # Effective tube radius used by DT (curvature-adaptive)
    r_tube_eff = min(
        max(controller.r_tube * (1.0 + 0.8 * kappa), controller.r_tube),
        1.5 * controller.r_tube
    )


    # CBF slack h (positive inside tube, negative = violation)
    h = r_tube_eff**2 - e_perp**2


    # Nearest progress on the desired path (generic polyline projection)
    s_near = nearest_s_on_polyline(x, y, path_xy, cumlen)


    # DT depth (for comparison with main’s d_hat)
    d_dt = controller.get_depth_for_main()


    # ---- Log everything ----
    log["t"].append(t)


    log["x"].append(x);            log["y"].append(y);            log["th"].append(theta)
    log["xd"].append(x_d[0]);      log["yd"].append(x_d[1]);      log["thd"].append(th_path)


    log["e_perp"].append(e_perp);  log["c_t"].append(c_t)
    log["h"].append(h);            log["r_tube"].append(r_tube_eff)


    log["s"].append(controller.s); log["s_near"].append(s_near)


    log["u_out_v"].append(u_out[0]);   log["u_out_w"].append(u_out[1])
    log["u_main_v"].append(u_main[0]); log["u_main_w"].append(u_main[1])
    log["u_dt_v"].append(controller.u_dt[0]); log["u_dt_w"].append(controller.u_dt[1])


    log["d_main"].append(d_hat);   log["d_dt"].append(d_dt)


    log["mode_dt"].append(1 if controller.mode == 'dt' else 0)
    log["attack"].append(1 if attack_active else 0)








    # ---- Depth estimation (single source of truth) ----
    # While attack is active, DT updates its own depth internally and drives the robot.
    # Freeze Main's estimator to avoid desync.
    if not attack_active:
        d_hat = update_depth_estimate(
            d_hat, e_x, e_y, e_theta, v_d_star, gamma, gains['k3'], d_min, d_max, dt
        )


    # Detect end of attack (rising edge False)
    if prev_attack_active and not attack_active:
        # copy DT's depth back to Main so we re-align estimators
        d_hat = controller.get_depth_for_main()


    prev_attack_active = attack_active


    # ---- Logging ----
    trajectory_log.append(main_robot.get_pose())
    if attack_active:
        dt_trajectory_log.append(controller.x_hat.copy())


    # ---- Draw ----
    for px, py, _ in trajectory_log:
        sx, sy = world_to_screen(px, py)
        pygame.draw.circle(screen, BLUE, (sx, sy), 2)


    for px, py, _ in dt_trajectory_log:
        sx, sy = world_to_screen(px, py)
        pygame.draw.circle(screen, RED, (sx, sy), 2)


    rx, ry = world_to_screen(*main_robot.get_pose()[:2])
    pygame.draw.circle(screen, BLACK, (rx, ry), 6)


    if attack_active:
        dxs, dys = world_to_screen(*controller.x_hat[:2])
        pygame.draw.circle(screen, RED, (dxs, dys), 6)


    pygame.display.flip()
    clock.tick(1 / dt)


# ------------------ Exit Gracefully ------------------
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            waiting = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                waiting = False
    pygame.display.flip()


pygame.quit()






# ================== DIAGNOSTIC PLOTS ==================






t_arr   = np.array(log["t"])
e_perp  = np.array(log["e_perp"])
r_tube  = np.array(log["r_tube"])
c_t_arr = np.array(log["c_t"])
s_arr   = np.array(log["s"])
s_near  = np.array(log["s_near"])


# 1) XY Trajectory with desired path and attack intervals
plt.figure(figsize=(6,6))
plt.plot(log["xd"], log["yd"], '--', label='Desired Path', color='orange')
plt.plot(log["x"],  log["y"],  label='Robot Path',  color='tab:blue')
atk_idx = np.where(np.array(log["attack"]) > 0.5)[0]
if atk_idx.size > 0:
    plt.scatter(np.array(log["x"])[atk_idx], np.array(log["y"])[atk_idx],
                s=8, c='red', label='During attack', alpha=0.7)
plt.gca().set_aspect('equal', 'box')
plt.title('Trajectory'); plt.xlabel('x [m]'); plt.ylabel('y [m]')
plt.grid(True); plt.legend()


# 2) Lateral error with tube bounds
plt.figure(figsize=(10,3))
plt.plot(t_arr, e_perp, label='e_perp (lateral error)')
plt.plot(t_arr,  r_tube, 'k--', lw=1, label='+r_tube')
plt.plot(t_arr, -r_tube, 'k--', lw=1, label='-r_tube')
plt.fill_between(t_arr, -r_tube, r_tube, color='gray', alpha=0.12)
plt.title('Lateral Error vs Time'); plt.xlabel('time [s]'); plt.ylabel('meters')
plt.grid(True); plt.legend(loc='upper right')


# 3) Tangent alignment (forward gating)
plt.figure(figsize=(10,3))
plt.plot(t_arr, c_t_arr, label='c_t (tangent alignment)')
plt.axhline(0.0, color='k', lw=1)
plt.axhline(0.3, color='k', lw=1, ls='--', label='gating threshold (~0.3)')
plt.title('Tangent Alignment c_t ∈ [-1,1]')
plt.xlabel('time [s]'); plt.ylabel('c_t')
plt.grid(True); plt.legend(loc='lower right')


# 4) Progress s(t) — phase lock check
plt.figure(figsize=(10,3))
plt.plot(t_arr, s_arr, label='controller.s (phase-locked)')
plt.plot(t_arr, s_near, '--', label='s_near (nearest on path)', alpha=0.75)
plt.title('Progress / Phase Lock'); plt.xlabel('time [s]'); plt.ylabel('s ∈ [0,1]')
plt.grid(True); plt.legend()


# 5) Commands: applied vs Main vs DT
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t_arr, log["u_out_v"],  label='v_out (applied)')
plt.plot(t_arr, log["u_main_v"], label='v_main', alpha=0.7)
plt.plot(t_arr, log["u_dt_v"],   label='v_dt',   alpha=0.7)
plt.title('Linear Velocity'); plt.ylabel('v [m/s]')
plt.grid(True); plt.legend()
plt.subplot(2,1,2)
plt.plot(t_arr, log["u_out_w"],  label='w_out (applied)')
plt.plot(t_arr, log["u_main_w"], label='w_main', alpha=0.7)
plt.plot(t_arr, log["u_dt_w"],   label='w_dt',   alpha=0.7)
plt.title('Angular Velocity'); plt.xlabel('time [s]'); plt.ylabel('ω [rad/s]')
plt.grid(True); plt.legend()


# 6) Depth estimates: Main vs DT
plt.figure(figsize=(10,3))
plt.plot(t_arr, log["d_main"], label='d_hat (Main)')
plt.plot(t_arr, log["d_dt"],   label='d_hat (DT)', ls='--')
plt.title('Depth Estimates'); plt.xlabel('time [s]'); plt.ylabel('depth scale')
plt.grid(True); plt.legend()


# 7) Attack & Mode timeline
plt.figure(figsize=(10,2.8))
plt.plot(t_arr, log["attack"],  drawstyle='steps-post', label='Attack=1')
plt.plot(t_arr, log["mode_dt"], drawstyle='steps-post', label='Mode DT=1')
plt.ylim(-0.1, 1.1)
plt.title('Timeline: Attack & Handover'); plt.xlabel('time [s]')
plt.yticks([0,1], ['off','on'])
plt.grid(True, axis='x'); plt.legend(loc='upper right')


plt.tight_layout()
plt.show()
