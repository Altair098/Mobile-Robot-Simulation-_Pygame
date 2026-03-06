import numpy as np
import math
from collections import deque

from controller import compute_control
from depth_estimator import update_depth_estimate
import config as CFG
from cbf_filters import clamp_v_discrete

class DTRescueController:
    """
    DT-Rescue v5 (circle-polished):
    - SAME public API & fields as before.
    - Continuous progress lock for circle (no phase slip).
    - Curvature-adaptive CBF + forward-tangent gating.
    - Depth sync helpers for main <-> DT.
    """

    def __init__(self, robot_model, trajectory_gen, dt=0.1):
        self.dt = dt
        self.robot = robot_model
        self.traj = trajectory_gen

        # --- config ---
        self.gains = CFG.gains
        self.gamma = CFG.gamma
        self.d_min = CFG.d_min
        self.d_max = CFG.d_max
        self.d_hat = CFG.initial_depth
        self.v_max = CFG.lambda_v
        self.w_max = CFG.lambda_omega

        # EKF state
        self.x_hat = np.zeros(3)
        self.P = np.eye(3) * 0.01

        # Control memory
        self.last_u_main = np.zeros(2)  # last healthy main cmd
        self.u_dt = np.zeros(2)
        self.u_out = np.zeros(2)

        # Progress
        self.s = 0.0
        self.s_prev_pose = None

        # Washout
        self.in_transition = False
        self.transition_timer = 0.0
        self.transition_duration = 0.1
        self.bt_Tmin = 0.06
        self.bt_Tmax = 0.50   # a bit longer for tough switches
        self.bt_ku   = 0.12

        # Mode
        self.mode = 'main'

        # Logs
        self.ring_buffer = deque(maxlen=50)

        # Safety / gating
        self.r_tube = 0.22         # base tube radius [m]
        self.lambda_cbf = 12.0     # stronger push back
        self.forward_only = True
        # --- NEW: parameters for discrete CBF clamp ---
        self.lambda_cbf = 2.0      # CBF gain λ (tune between 1–3)
        self.beta_kappa = 0.8      # curvature scaling factor for tube radius
        self.k_align_boost = max(1.5, self.gains['k2'])

    # ---------------- Depth sync for main <-> DT ----------------
    def set_depth_from_main(self, d_main: float):
        self.d_hat = float(np.clip(d_main, self.d_min, self.d_max))

    def get_depth_for_main(self) -> float:
        return float(np.clip(self.d_hat, self.d_min, self.d_max))

    # ---------------- Utilities ----------------
    def normalize_angle(self, th):
        return (th + np.pi) % (2 * np.pi) - np.pi

    def clamp(self, u):
        return np.array([
            float(np.clip(u[0], -self.v_max, self.v_max)),
            float(np.clip(u[1], -self.w_max, self.w_max))
        ])

    # EKF integrates the actually applied command
    def ekf_update(self, applied_control):
        v, omega = applied_control
        theta = self.x_hat[2]
        dx = v * math.cos(theta) * self.dt
        dy = v * math.sin(theta) * self.dt
        dtheta = omega * self.dt
        self.x_hat += np.array([dx, dy, dtheta])
        self.x_hat[2] = self.normalize_angle(self.x_hat[2])

    def update_phase_progress(self, pose):
        if self.s_prev_pose is not None:
            dx = pose[0] - self.s_prev_pose[0]
            dy = pose[1] - self.s_prev_pose[1]
            arc = math.hypot(dx, dy)
            self.s += arc / self.traj.total_length()
            self.s = min(1.0, max(0.0, self.s))
        self.s_prev_pose = pose.copy()

    # ---------------- Handover ----------------
    def handover(self, attack_flag):
        if attack_flag and self.mode == 'main':
            self.mode = 'dt'
            self._start_adaptive_washout(to_mode='dt')
            print("⚠️ Handover: MAIN → DT")
        elif not attack_flag and self.mode == 'dt':
            self.mode = 'main'
            self._start_adaptive_washout(to_mode='main')
            print("✅ Handover: DT → MAIN")

    def _start_adaptive_washout(self, to_mode):
        if to_mode == 'dt':
            u_src = self.last_u_main.copy()
            self._shadow_control_raw()
            u_dst = self.u_dt.copy()
        else:
            u_src = self.u_dt.copy()
            u_dst = self.last_u_main.copy()

        du = float(np.linalg.norm(u_src - u_dst))
        T = self.bt_Tmin + self.bt_ku * du
        T = float(np.clip(T, self.bt_Tmin, self.bt_Tmax))
        self.transition_duration = T
        self.in_transition = True
        self.transition_timer = 0.0

    # ---------------- Continuous progress lock (CIRCLE) ----------------
    def _lock_progress_circle(self, xhat):
        """
        Lock s to the nearest angle on the known circle (center at 0,0).
        Smoothly blend to avoid jumps.
        """
        ang = math.atan2(xhat[1], xhat[0])           # [-pi, pi]
        s_near = (ang % (2*math.pi)) / (2*math.pi)   # [0,1)
        alpha = 0.35                                  # blend factor
        self.s = float((1 - alpha) * self.s + alpha * s_near)

    # ---------------- DT shadow control (same law as main) ---------------
    def _shadow_control_raw(self):
        x_d, _, u_ff = self.traj.sample(self.s)
        v_d_star = float(u_ff[0])
        omega_d  = float(u_ff[1])

        dx = x_d[0] - self.x_hat[0]
        dy = x_d[1] - self.x_hat[1]
        th = self.x_hat[2]
        e_x = math.cos(th) * dx + math.sin(th) * dy
        e_y = -math.sin(th) * dx + math.cos(th) * dy
        e_th = self.normalize_angle(x_d[2] - th)

        v, w = compute_control(e_x, e_y, e_th, v_d_star, omega_d, self.d_hat, self.gains)

        # Forward-only takeover: align first if v would be negative
        if self.forward_only and v < 0.0:
            v = 0.0
            w += self.k_align_boost * math.sin(e_th)

        v = max(min(v, self.v_max), -self.v_max)
        w = max(min(w, self.w_max), -self.w_max)
        self.u_dt = np.array([v, w], dtype=float)

        # DT's depth adaptation (main is frozen during attack)
        self.d_hat = update_depth_estimate(
            self.d_hat, e_x, e_y, e_th, v_d_star, self.gamma, self.gains['k3'],
            self.d_min, self.d_max, self.dt
        )

    # ---------------- CBF (curvature-adaptive) + tangent gating ----------
    def _cbf_filter(self, u_cmd, xhat, ref_tuple):
        x_d, _, _ = ref_tuple
        th = xhat[2]
        th_path = x_d[2]

        # curvature estimate for adaptive tube
        sA = max(0.0, self.s - 0.01); sB = min(1.0, self.s + 0.01)
        thA = self.traj.sample(sA)[0][2]; thB = self.traj.sample(sB)[0][2]
        dth = (thB - thA + math.pi) % (2*math.pi) - math.pi
        kappa = abs(dth / max(1e-3, (sB - sA)))
        r_tube_eff = float(np.clip(self.r_tube * (1.0 + 0.8 * kappa),
                                   self.r_tube, 1.5 * self.r_tube))

        # path normal / tangent at ref
        n_x, n_y = -math.sin(th_path), math.cos(th_path)
        t_x, t_y =  math.cos(th_path), math.sin(th_path)

        dx = xhat[0] - x_d[0]; dy = xhat[1] - x_d[1]
        d_perp = dx * n_x + dy * n_y
        h = r_tube_eff ** 2 - d_perp ** 2

        # CBF inequality in v: a*v >= rhs
        if h > 0.02 * (r_tube_eff ** 2):
            v, w = u_cmd[0], u_cmd[1]
        else:
            c_n = n_x * math.cos(th) + n_y * math.sin(th)
            rhs = -self.lambda_cbf * h
            a = -2.0 * d_perp * c_n

            v = float(u_cmd[0])
            if abs(a) >= 1e-6:
                if a > 0 and a * v < rhs:
                    v = rhs / a
                elif a < 0 and a * v < rhs:
                    v = rhs / a
            w = float(u_cmd[1])

        # Forward-tangent gating (prevents wrong-way motion)
        c_t = t_x * math.cos(th) + t_y * math.sin(th)  # alignment ∈ [-1,1]
        if c_t < 0.0:
            v = 0.0
        elif c_t < 0.3:
            v = min(v, max(0.0, self.v_max * c_t))

        v = float(np.clip(v, -self.v_max, self.v_max))
        w = float(np.clip(w, -self.w_max, self.w_max))
        return np.array([v, w], dtype=float)

    # ---------------- Main tick ----------------
    def step(self, t, attack_flag, main_pose, main_control):
        self.handover(attack_flag)

        # Progress using continuous circle lock (eliminates phase slip)
        self.update_phase_progress(main_pose)
        self._lock_progress_circle(self.x_hat)

        # MAIN mode -> mirror state and apply main
        if self.mode == 'main':
            self.x_hat = main_pose.copy()
            self.u_out = main_control
            self.s_prev_pose = main_pose.copy()
            applied = self.u_out
            self.last_u_main = main_control

        # DT mode -> DT sole authority
        else:
            self._shadow_control_raw()
            self.u_out = self.u_dt
            self.u_out = self._cbf_filter(self.u_out, self.x_hat, self.traj.sample(self.s))
            applied = self.u_out

        # Adaptive bumpless washout (short ramp only at switch)
        if self.in_transition:
            alpha = float(np.clip(self.transition_timer / self.transition_duration, 0.0, 1.0))
            if self.mode == 'dt':      # main -> dt
                self.u_out = (1 - alpha) * self.last_u_main + alpha * self.u_out
            else:                      # dt -> main
                self.u_out = (1 - alpha) * self.u_dt + alpha * self.u_out
            self.transition_timer += self.dt
            if self.transition_timer >= self.transition_duration:
                self.in_transition = False
            applied = self.u_out  # EKF uses the actually applied command

        # Final clamp and EKF integrate applied command
        self.u_out = self.clamp(self.u_out)
        self.ekf_update(applied)

        # Log
        self.ring_buffer.append((self.x_hat.copy(), self.s, self.u_dt.copy(), t))
        return self.u_out.copy()

    # ---------------- Debug ----------------
    def get_debug(self):
        return {
            'x_hat': self.x_hat.copy(),
            's': self.s,
            'u_dt': self.u_dt.copy(),
            'u_out': self.u_out.copy(),
            'mode': self.mode,
            'transition': self.in_transition,
            'ring': list(self.ring_buffer),
        }
