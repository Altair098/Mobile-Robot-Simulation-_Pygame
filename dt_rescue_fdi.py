# dt_rescue_fdi.py
import math
import numpy as np
from collections import deque

from controller import compute_control
from depth_estimator import update_depth_estimate
import config as CFG
from cbf_filters import clamp_v_discrete


def _wrap(a):  # angle wrap to [-pi,pi]
    return (a + math.pi) % (2 * math.pi) - math.pi


class DTRescueController:
    """
    Digital-Twin Rescue (generic path, handover-safe)

    Key features
    ------------
    • Forward-only with TURN→GO hysteresis (no reversals).
    • Micro-anchor (local forward reprojection) to keep DT centered on path.
    • ω-first budgeting + curvature preview: allocate ω, cap v via κ and remaining ω margin.
    • Inward-aware CBF: never move further outward near tube boundary; then discrete CBF clamp.
    • Dynamic progress floor when safe & aligned to avoid DT under-speed.
    • Rate regularization for (v, ω) and smooth washout during handovers.
    """

    # ---------------- init ----------------
    def __init__(self, robot_model, trajectory_gen, dt=0.1):
        self.dt   = float(dt)
        self.traj = trajectory_gen
        self.robot = robot_model   # (mirror only)

        # toggleable adaptation (for FDI experiments)
        self.adapt_enabled = True

        # controller/config
        self.gains = CFG.gains
        self.gamma = CFG.gamma
        self.d_min = CFG.d_min
        self.d_max = CFG.d_max
        self.d_hat = float(CFG.initial_depth)
        self.v_max = float(CFG.lambda_v)
        self.w_max = float(CFG.lambda_omega)

        # hysteresis on alignment
        self.tau_turn = 0.40                # enter TURN if c_t < tau_turn
        self.tau_go   = 0.65                # leave TURN when c_t > tau_go
        self.theta_go = math.radians(6.0)

        # velocity synchronization & rate regularization
        self.lambda_sync = 0.35             # attraction to last main command
        self.lambda_rate = 0.10             # attraction to last DT command
        self.v_last = 0.0
        self.w_last = 0.0
        self.w_mem  = 0.0                   # low-pass for ω
        self.beta_w_rate = 0.35             # 0→no smoothing, 1→fast

        # EKF (pose mirror integrated by applied command)
        self.x_hat = np.zeros(3, dtype=float)

        # memory
        self.last_u_main = np.zeros(2, dtype=float)
        self.u_dt  = np.zeros(2, dtype=float)
        self.u_out = np.zeros(2, dtype=float)

        # phase (authoritative in main; DT reads it)
        self.s = 0.0

        # washout
        self.in_transition      = False
        self.transition_timer   = 0.0
        self.bt_Tmin, self.bt_Tmax, self.bt_ku = 0.06, 0.50, 0.12
        self.transition_duration = 0.10

        # mode + FSM
        self.mode = 'main'
        self.just_switched = None           # 'main2dt' on takeover
        self.dt_state = 'GO'                # {'TURN','GO'}

        # safety / gating
        self.forward_only   = True
        self.k_align_boost  = max(2.0, 1.5 * self.gains['k2'])

        # tube/CBF (base radius & softening)
        self.r0        = 0.16
        self.a_kappa   = 0.90
        self.b_align   = 0.30
        self.lambda_cbf = 1.2

        # preview & corner handling / speed smoothing
        self.ds_preview  = 0.10
        self.kappa_corner = 1.0
        self.beta_accel   = 0.45            # a bit more eager than before
        self.v_mem        = 0.0

        # dynamic progress floor (helps prevent DT under-speed)
        self.v_floor_progress_base = 0.06

        # anti-stall
        self.stall_timeout = 0.35
        self.stall_timer   = 0.0

        # micro-anchor (forward-only local reprojection)
        self.ds_micro = 0.03                # look-ahead window in s
        self.N_micro  = 7                   # samples
        self.w_perp   = 1.0                 # cost weights
        self.w_heading = 0.25

        # debug
        self.ring_buffer = deque(maxlen=50)

    def set_adaptation(self, enabled: bool):
        self.adapt_enabled = bool(enabled)

    # ---------------- utilities ----------------
    def normalize_angle(self, a): return _wrap(a)

    def clamp(self, u):
        return np.array([
            float(np.clip(u[0], -self.v_max, self.v_max)),
            float(np.clip(u[1], -self.w_max, self.w_max))
        ], dtype=float)

    def ekf_update(self, applied):
        v, w = float(applied[0]), float(applied[1])
        th = self.x_hat[2]
        self.x_hat[0] += v * math.cos(th) * self.dt
        self.x_hat[1] += v * math.sin(th) * self.dt
        self.x_hat[2]  = _wrap(self.x_hat[2] + w * self.dt)

    def _plus(self, x): return x if x > 0.0 else 0.0

    # ---------------- depth sync ----------------
    def set_depth_from_main(self, d_main): self.d_hat = float(np.clip(d_main, self.d_min, self.d_max))
    def get_depth_for_main(self):          return float(np.clip(self.d_hat, self.d_min, self.d_max))

    # ---------------- handover ----------------
    def handover(self, attack_flag):
        if attack_flag and self.mode == 'main':
            self.mode = 'dt'
            self.just_switched = 'main2dt'
            self.dt_state = 'TURN'
            self.v_mem = 0.0
            self._start_washout(to_mode='dt')
            print("⚠️ Handover: MAIN → DT")
        elif (not attack_flag) and self.mode == 'dt':
            self.mode = 'main'
            self.just_switched = None
            self._start_washout(to_mode='main')
            print("✅ Handover: DT → MAIN")

    def _start_washout(self, to_mode):
        if to_mode == 'dt':
            u_src = self.last_u_main.copy()
            u_dst = self.u_dt.copy()
        else:
            u_src = self.u_dt.copy()
            u_dst = self.last_u_main.copy()
        du = float(np.linalg.norm(u_src - u_dst))
        T = np.clip(self.bt_Tmin + self.bt_ku * du, self.bt_Tmin, self.bt_Tmax)
        self.transition_duration = float(T)
        self.in_transition = True
        self.transition_timer = 0.0

    def _snap_phase_on_takeover(self, main_pose):
        """Monotone, heading-aware snap to local path near current pose."""
        x, y, th = float(main_pose[0]), float(main_pose[1]), float(main_pose[2])
        if hasattr(self.traj, "nearest_s_pruned"):
            self.s = float(self.traj.nearest_s_pruned(x, y, th, self.s, max_jump=0.08, cone_deg=70.0))
        elif hasattr(self.traj, "nearest_s_local"):
            self.s = float(self.traj.nearest_s_local(x, y, self.s, max_jump=0.08))
        elif hasattr(self.traj, "nearest_s"):
            self.s = float(self.traj.nearest_s(x, y))
        # set state based on alignment
        th_ref = self.traj.sample(self.s)[0][2]
        c_t = math.cos(th_ref)*math.cos(th) + math.sin(th_ref)*math.sin(th)
        self.dt_state = 'TURN' if c_t < 0.3 else 'GO'

    # ---------------- helpers ----------------
    def _curvature_preview(self, s, ds):
        sA = max(0.0, s - ds); sB = min(1.0, s + ds)
        thA = self.traj.sample(sA)[0][2]; thB = self.traj.sample(sB)[0][2]
        dth = _wrap(thB - thA)
        return abs(dth / max(1e-3, (sB - sA)))

    def _tube_state(self, pose, s_use, c_t):
        """Compute signed normal distance and effective tube radius."""
        x_d, _, _ = self.traj.sample(s_use)
        th_ref = float(x_d[2])
        t_x, t_y = math.cos(th_ref), math.sin(th_ref)
        n_x, n_y = -t_y, t_x
        d_perp = (pose[0]-x_d[0]) * n_x + (pose[1]-x_d[1]) * n_y

        kappa = float(self.traj.curvature(s_use))
        r_eff = self.r0 * (1.0 + self.a_kappa * abs(kappa)) * (1.0 - self.b_align * max(0.0, c_t))
        r_eff = max(0.75*self.r0, min(r_eff, 2.0*self.r0))
        return d_perp, r_eff, th_ref, kappa

    def _micro_anchor(self, s, pose):
        """Forward-only local reprojection to reduce lateral/heading error."""
        s_min = s
        s_max = min(1.0, s + self.ds_micro)
        cand = np.linspace(s_min, s_max, self.N_micro)
        best_s = s
        best_J = 1e9
        th = float(pose[2])
        for si in cand:
            x_d, _, _ = self.traj.sample(si)
            dx, dy = x_d[0]-pose[0], x_d[1]-pose[1]
            e_x =  math.cos(th)*dx + math.sin(th)*dy
            e_y = -math.sin(th)*dx + math.cos(th)*dy
            e_th = _wrap(x_d[2] - th)
            J = self.w_perp*(e_y**2) + self.w_heading*(e_th**2)
            if J < best_J:
                best_J, best_s = J, float(si)
        return best_s

    def _stall_guard(self, pose, x_d, v_cmd, w_cmd):
        if abs(v_cmd) > 1e-3:
            self.stall_timer = 0.0
            return v_cmd, w_cmd
        self.stall_timer += self.dt
        if self.stall_timer < self.stall_timeout:
            return v_cmd, w_cmd
        # nudge forward + stronger heading correction
        v_cmd = 0.02
        e_th  = _wrap(x_d[2] - pose[2])
        w_cmd = float(np.clip(w_cmd + 1.2*self.k_align_boost*math.sin(e_th), -self.w_max, self.w_max))
        return v_cmd, w_cmd

    # ------ inward-aware CBF shaping then discrete clamp (safe) ------
    def _cbf_inward_project(self, pose, s_use, v_star, c_t):
        """
        Discourage outward motion near tube boundary;
        then apply discrete CBF clamp for hard safety.
        """
        x_d, _, _ = self.traj.sample(s_use)
        th_ref = float(x_d[2])
        th = float(pose[2])
        # small-signal normal velocity approx: d_perp_dot ≈ v * sin(e_th)
        e_th = _wrap(th_ref - th)
        d_perp, r_eff, _, _ = self._tube_state(pose, s_use, c_t)
        dperp_dot = v_star * math.sin(e_th)

        # If heading would push outward & we are near the wall, attenuate v
        if (d_perp * dperp_dot) > 0.0 and abs(d_perp) > 0.6 * r_eff:
            # linear ramp attenuation to zero at wall
            alpha = (abs(d_perp) - 0.6*r_eff) / (0.4*r_eff)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            v_star = (1.0 - alpha) * v_star

        # Hard safety clamp (discrete CBF)
        v_star = clamp_v_discrete(
            pose, v_star, self.traj, s_use,
            r_base=self.r0, beta=self.a_kappa, lam=self.lambda_cbf
        )
        return v_star

    # ---------------- ω-first projection with sync & floors ----------------
    def _shadow_project(self, v_ref, w_ref, c_t, e_th, kappa_max, pose, s_use, x_d):
        """
        Project DT command to feasible safe set with forward-only constraint.
        Corner-aware: allocate ω first, cap v from κ and ω margin.
        Uses micro-anchor s_use.
        """
        # ω synchronization baseline (blend ref, rate memory, main)
        w_main = float(self.last_u_main[1])
        den = (1.0 + self.lambda_rate + self.lambda_sync)
        w_sync = (w_ref + self.lambda_rate*self.w_last + self.lambda_sync*w_main) / den
        w_align = 0.3*self.k_align_boost*math.sin(e_th)
        w_base = float(np.clip(w_sync + w_align, -self.w_max, self.w_max))

        # low-pass ω to avoid jitter
        self.w_mem = (1.0 - self.beta_w_rate)*self.w_mem + self.beta_w_rate*w_base
        w_base = self.w_mem

        # ω budget ⇒ cap v by remaining ω & previewed κ
        w_margin = max(0.0, self.w_max - abs(w_base))
        v_cap = self.v_max if kappa_max < 1e-6 else min(self.v_max, w_margin / (kappa_max + 1e-6))

        # weighted v (ref, rate, sync-to-main)
        v_main = float(self.last_u_main[0])
        v_star = (v_ref + self.lambda_rate*self.v_last + self.lambda_sync*v_main) / den
        v_star = float(np.clip(v_star, 0.0, v_cap))  # forward-only + cap

        # inward-aware shaping + hard CBF clamp
        v_star = self._cbf_inward_project(pose, s_use, v_star, c_t)
        v_star = max(0.0, float(v_star))

        # extra forward gating when poorly aligned
        if c_t < 0.3:
            v_star = min(v_star, max(0.0, self.v_max * (c_t / 0.3)))

        # dynamic progress floor when safely inside tube & well aligned
        d_perp, r_eff, _, _ = self._tube_state(pose, s_use, c_t)
        if abs(d_perp) <= r_eff and c_t > self.tau_go:
            gain = 0.10 * (c_t - self.tau_go) * max(0.0, 1.0 - abs(d_perp)/r_eff)
            v_floor = self.v_floor_progress_base + gain
            v_star = max(v_star, min(v_cap, v_floor))

        # smooth v
        self.v_mem = (1.0 - self.beta_accel)*self.v_mem + self.beta_accel*v_star
        v_star = max(0.0, float(self.v_mem))

        # anti-stall
        v_star, w_base = self._stall_guard(pose, x_d, v_star, w_base)
        v_star = max(0.0, float(v_star))

        # update rate memory
        self.v_last = float(v_star)
        self.w_last = float(w_base)
        return v_star, w_base

    # ---------------- main tick ----------------
    def step(self, t, attack_flag, main_pose, main_control):
        self.handover(attack_flag)

        # ===== MAIN =====
        if self.mode == 'main':
            self.x_hat = main_pose.copy()
            self.u_out = self.clamp(main_control)
            self.last_u_main = self.u_out.copy()
            applied = self.u_out

        # ===== DT =====
        else:
            if self.just_switched == 'main2dt':
                self.x_hat = main_pose.copy()
                self._snap_phase_on_takeover(main_pose)
                self.just_switched = None

            # ---- micro-anchor: small forward reprojection for DT only
            s_use = self._micro_anchor(self.s, main_pose)

            # reference at s_use
            x_d, _, u_ff = self.traj.sample(s_use)
            th_ref = float(x_d[2])

            # errors at REAL pose
            x, y, th = float(main_pose[0]), float(main_pose[1]), float(main_pose[2])
            dx, dy = x_d[0]-x, x_d[1]-y
            e_x  =  math.cos(th)*dx + math.sin(th)*dy
            e_y  = -math.sin(th)*dx + math.cos(th)*dy
            e_th =  _wrap(th_ref - th)

            # raw reference from same law as Main (shadow controller)
            v_ref, w_ref = compute_control(e_x, e_y, e_th,
                                           float(u_ff[0]), float(u_ff[1]),
                                           self.d_hat, self.gains)
            v_ref = float(np.clip(v_ref, -self.v_max, self.v_max))
            w_ref = float(np.clip(w_ref, -self.w_max, self.w_max))

            # ---- ADAPTATION (guarded) ----
            if self.adapt_enabled:
                self.d_hat = update_depth_estimate(
                    self.d_hat, e_x, e_y, e_th, float(u_ff[0]),
                    self.gamma, self.gains['k3'], self.d_min, self.d_max, self.dt
                )

            # alignment & curvature preview
            c_t = math.cos(th_ref)*math.cos(th) + math.sin(th_ref)*math.sin(th)
            kappa_max = self._curvature_preview(s_use, self.ds_preview)

            # FSM: TURN→GO to avoid reverse; ω-only in TURN
            if self.dt_state == 'TURN' or c_t < self.tau_turn:
                v_cmd = 0.0
                w_cmd = float(np.clip(self.k_align_boost*math.sin(e_th), -self.w_max, self.w_max))
                if c_t > self.tau_go and abs(e_th) < self.theta_go:
                    self.dt_state = 'GO'
            else:
                v_cmd, w_cmd = self._shadow_project(
                    v_ref, w_ref, c_t, e_th, kappa_max, main_pose, s_use, x_d
                )

            self.u_dt  = np.array([v_cmd, w_cmd], dtype=float)
            self.u_out = self.clamp(self.u_dt)
            applied    = self.u_out

        # ---- washout around switch
        if self.in_transition:
            alpha = float(np.clip(self.transition_timer/self.transition_duration, 0.0, 1.0))
            if self.mode == 'dt':   # main → dt
                self.u_out = (1 - alpha)*self.last_u_main + alpha*self.u_out
            else:                   # dt → main
                self.u_out = (1 - alpha)*self.u_dt + alpha*self.u_out
            self.transition_timer += self.dt
            if self.transition_timer >= self.transition_duration:
                self.in_transition = False
            applied = self.u_out

        # apply (for EKF mirror)
        self.u_out = self.clamp(self.u_out)
        self.ekf_update(self.u_out)

        # debug ring
        self.ring_buffer.append((self.x_hat.copy(), float(self.s), self.u_dt.copy(), float(t)))
        return self.u_out.copy()

    # ---------------- debug getter ----------------
    def get_debug(self):
        return {
            "x_hat": self.x_hat.copy(),
            "s": float(self.s),
            "u_dt": self.u_dt.copy(),
            "u_out": self.u_out.copy(),
            "mode": self.mode,
            "transition": self.in_transition,
            "ring": list(self.ring_buffer),
        }
