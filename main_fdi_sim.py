# main_fdi_sim.py
import pygame, math, time, os, csv
import numpy as np
from datetime import datetime

from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from generic_trajectory import GenericTrajectory
from dt_rescue_fdi import DTRescueController
from path_input import FreehandPathRecorder

from attacker_fdi import FDIInjector
from fdi_detector import ResidualDetector

from config import gains, gamma, d_min, d_max, initial_depth, lambda_v, lambda_omega, dt, total_time

# ---------- setup ----------
pygame.init()
W,H = 800,800
screen = pygame.display.set_mode((W,H))
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("arial", 16)

SCALE = 100.0
def w2s(x,y): return int(W/2 + x*SCALE), int(H/2 - y*SCALE)
def s2w(px,py): return (px - W/2)/SCALE, (H/2 - py)/SCALE

# draw path
rec = FreehandPathRecorder(screen, s2w)
wps = None
while wps is None:
    wps = rec.run()

traj = GenericTrajectory.from_waypoints(wps, closed=False, speed=lambda_v)
s_grid = np.linspace(0,1,600)
path_pts = [w2s(*traj.sample(s)[0][:2]) for s in s_grid]

# robots and controller
main = Robot()
dtbot = Robot()
x0,y0 = wps[0]
t_hat,_ = traj.tangent_normal(0.0)
th0 = math.atan2(t_hat[1], t_hat[0])

def set_pose(r,x,y,th):
    if hasattr(r,'set_pose'):
        try: r.set_pose(x,y,th)
        except TypeError: r.set_pose((x,y,th))
    else:
        r.x, r.y, r.theta = x,y,th

set_pose(main,x0,y0,th0); set_pose(dtbot,x0,y0,th0)

ctrl = DTRescueController(dtbot, traj, dt=dt)
ctrl.s = float(traj.nearest_s(x0,y0))
ctrl.x_hat = np.array([x0,y0,th0], float)

# attacker + detector
att = FDIInjector(mode='bias')
det = ResidualDetector(pos_thresh=0.12, th_thresh=math.radians(8.0), n_consec=6, n_clear=12)

# logs
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join("results_fdi", run_id)
os.makedirs(outdir, exist_ok=True)
log = []

# state
d_hat = float(initial_depth)
trail_main, trail_dt = [], []
TRAIL_MAX = 4000

# ---------- loop ----------
start = time.time()
running = True
s_phase = ctrl.s
while running:
    t = time.time() - start

    # events
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q: running = False
            elif e.key == pygame.K_f: att.set_enabled(not att.enabled)          # toggle attack
            elif e.key == pygame.K_1: att.set_mode('bias')
            elif e.key == pygame.K_2: att.set_mode('drift')
            elif e.key == pygame.K_3: att.set_mode('spike')
            elif e.key == pygame.K_4: att.set_mode('replay')

    # truth pose
    x_true, y_true, th_true = main.get_pose()

    # MAIN sees corrupted pose
    x_main_meas = att.corrupt(np.array([x_true,y_true,th_true], float), dt)

    # phase update from truth pose (safe local projection)
    if hasattr(traj,'nearest_s_pruned'):
        s_near = traj.nearest_s_pruned(x_true, y_true, th_true, s_phase, max_jump=0.08, cone_deg=70.0)
    else:
        s_near = traj.nearest_s_local(x_true, y_true, s_phase, max_jump=0.08)
    th_path = traj.sample(s_phase)[0][2]
    c_t = math.cos(th_path)*math.cos(th_true) + math.sin(th_path)*math.sin(th_true)
    if c_t < 0.2: s_near = min(s_near, s_phase)
    s_phase += 0.30*(s_near - s_phase)
    s_phase = float(np.clip(s_phase, 0.0, 0.999))
    ctrl.s = s_phase

    # reference
    x_d, _, _ = traj.sample(s_phase)
    theta_d   = float(x_d[2])
    kappa     = float(traj.curvature(s_phase))
    v_d_star  = min(lambda_v, lambda_omega/(abs(kappa)+1e-6))
    omega_d   = v_d_star * kappa

    # MAIN computes control using CORRUPTED pose
    dx = x_d[0] - x_main_meas[0]
    dy = x_d[1] - x_main_meas[1]
    thm = x_main_meas[2]
    e_x =  math.cos(thm)*dx + math.sin(thm)*dy
    e_y = -math.sin(thm)*dx + math.cos(thm)*dy
    e_th = (theta_d - thm + math.pi)%(2*math.pi)-math.pi
    v_main, w_main = compute_control(e_x,e_y,e_th, v_d_star, omega_d, d_hat, gains)
    v_main = max(min(v_main, lambda_v), -lambda_v)
    w_main = max(min(w_main, lambda_omega), -lambda_omega)

    # DETECTOR: compare MAIN's measurement vs DT's mirror
    attack_detected, r_pos, r_th = det.update(x_main_meas, ctrl.x_hat)

    # DT gets the decision. Freeze adaptation while under attack.
    ctrl.set_adaptation(not attack_detected)

    # DT decides who drives
    u_out = ctrl.step(t, attack_detected, np.array([x_true,y_true,th_true], float), np.array([v_main,w_main], float))

    # apply to plant
    main.update(u_out[0], u_out[1], dt)

    # logging
    x_cur, y_cur, _ = main.get_pose()
    mode_flag = getattr(ctrl, "mode", "main")
    log.append(dict(
        t=float(t), mode=mode_flag, fdi_on=int(att.enabled), fdi_type=att.mode,
        fdi_detected=int(attack_detected),
        s=float(s_phase),
        x=float(x_cur), y=float(y_cur),
        x_ref=float(x_d[0]), y_ref=float(x_d[1]),
        r_pos=float(r_pos), r_th=float(r_th),
        v_out=float(u_out[0]), w_out=float(u_out[1]),
        v_main=float(v_main), w_main=float(w_main),
    ))

    # trails
    if mode_flag == 'main':
        trail_main.append(w2s(x_cur,y_cur))
        if len(trail_main)>TRAIL_MAX: trail_main.pop(0)
    else:
        trail_dt.append(w2s(x_cur,y_cur))
        if len(trail_dt)>TRAIL_MAX: trail_dt.pop(0)

    # adapt depth only when no attack (same rule as before)
    if not attack_detected:
        d_hat = update_depth_estimate(d_hat, e_x, e_y, e_th, v_d_star, gamma, gains['k3'], d_min, d_max, dt)

    # draw
    screen.fill((255,255,255))
    for p in path_pts: pygame.draw.circle(screen,(0,200,0),p,1)
    for p in trail_main[::3]: pygame.draw.circle(screen,(0,102,204),p,2)
    for p in trail_dt[::3]:   pygame.draw.circle(screen,(255,0,0),p,2)

    # robot markers: main true pose = blue when main, red when dt
    col = (0,102,204) if mode_flag=='main' else (255,0,0)
    pygame.draw.circle(screen, col, w2s(x_cur,y_cur), 4)
    # DT pose marker
    pygame.draw.circle(screen, (150,150,150), w2s(*ctrl.x_hat[:2]), 3)

    hud = [
        f"t={t:5.2f}s  mode={mode_flag}  FDI_on={att.enabled}({att.mode})  detected={attack_detected}",
        f"s={s_phase:0.3f}  r_pos={r_pos:.3f}  r_th={r_th:.2f}rad"
    ]
    for i,txt in enumerate(hud):
        screen.blit(font.render(txt, True, (30,30,30)), (8,8+i*18))
    pygame.display.flip()
    clock.tick(int(1/dt))

# save logs
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir,"log.csv"),"w",newline="") as f:
    import csv
    ks = list(log[0].keys())
    w = csv.DictWriter(f, fieldnames=ks); w.writeheader(); w.writerows(log)
import pygame
pygame.image.save(screen, os.path.join(outdir,"final_frame.png"))
pygame.quit()
print("Saved to", outdir)
