# dt_rescue_dos.py – DT shadow controller for DoS attacks only
# Supports: circle, figure8, sCurve, sharpL paths
import math
import numpy as np
from collections import deque

from controller import compute_control
from depth_estimator import update_depth_estimate
import config as CFG
from cbf_filters import clamp_v_discrete


def _wrap(a):
    """Wrap angle to [-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class DTRescueController:
    """
    Digital-Twin Rescue Controller for DoS Attacks
    
    Supports predefined paths: circle, figure8, sCurve, sharpL
    
    Architecture:
    -------------
    • Main Controller: Normal operation (no attack)
    • DT Takeover: Digital Twin assumes control during DoS attacks
    • Smooth Handover: Bumpless transfer between modes
    
    Key Features:
    -------------
    • Forward-only motion with TURN→GO finite state machine
    • Micro-anchor for local path reprojection
    • ω-first control allocation with curvature preview
    • Control Barrier Function (CBF) for tube safety
    • Inward-aware velocity shaping near boundaries
    • Dynamic progress floor to prevent under-speed
    • Rate regularization for smooth commands
    • Washout blending during mode transitions
    """

    def __init__(self, robot_model, trajectory_gen, dt=0.1):
        """
        Initialize DT Rescue Controller
        
        Parameters:
        -----------
        robot_model : Robot
            Robot dynamics model (for state mirroring)
        trajectory_gen : GenericTrajectory
            Trajectory generator for reference path
        dt : float
            Control timestep in seconds
        """
        self.dt   = float(dt)
        self.traj = trajectory_gen
        self.robot = robot_model

        # Load configuration parameters
        self.gains = CFG.gains
        self.gamma = CFG.gamma
        self.d_min = CFG.d_min
        self.d_max = CFG.d_max
        self.d_hat = float(CFG.initial_depth)
        self.v_max = float(CFG.lambda_v)
        self.w_max = float(CFG.lambda_omega)

        # TURN→GO hysteresis thresholds
        self.tau_turn = 0.40                # Enter TURN if alignment < 0.40
        self.tau_go   = 0.65                # Exit TURN if alignment > 0.65
        self.theta_go = math.radians(6.0)   # Max heading error to exit TURN

        # Command synchronization weights
        self.lambda_sync = 0.35             # Weight for main controller sync
        self.lambda_rate = 0.10             # Weight for rate regularization
        
        # Command memory for rate regularization
        self.v_last = 0.0
        self.w_last = 0.0
        self.w_mem  = 0.0                   # Low-pass filtered ω
        self.beta_w_rate = 0.35             # ω filter coefficient

        # State estimation (Extended Kalman Filter mirror)
        self.x_hat = np.zeros(3, dtype=float)  # [x, y, theta]

        # Control memory
        self.last_u_main = np.zeros(2, dtype=float)  # Last main command
        self.u_dt  = np.zeros(2, dtype=float)         # DT command
        self.u_out = np.zeros(2, dtype=float)         # Output command

        # Path phase variable (0 to 1)
        self.s = 0.0

        # Washout (smooth handover) parameters
        self.in_transition      = False
        self.transition_timer   = 0.0
        self.bt_Tmin = 0.06                 # Min washout duration
        self.bt_Tmax = 0.50                 # Max washout duration
        self.bt_ku   = 0.12                 # Command-gap scaling
        self.transition_duration = 0.10

        # Mode and state machine
        self.mode = 'main'                  # 'main' or 'dt'
        self.just_switched = None           # Tracks recent switch
        self.dt_state = 'GO'                # 'TURN' or 'GO'

        # Control constraints
        self.forward_only   = True
        self.k_align_boost  = max(2.0, 1.5 * self.gains['k2'])

        # Safety tube parameters
        self.r0        = 0.16               # Base tube radius
        self.a_kappa   = 0.90               # Curvature expansion factor
        self.b_align   = 0.30               # Alignment contraction factor
        self.lambda_cbf = 1.2               # CBF aggressiveness

        # Curvature preview and speed control
        self.ds_preview  = 0.10             # Preview distance in s
        self.kappa_corner = 1.0             # Corner detection threshold
        self.beta_accel   = 0.45            # Speed filter coefficient
        self.v_mem        = 0.0             # Low-pass filtered v

        # Dynamic progress floor
        self.v_floor_progress_base = 0.06

        # Anti-stall mechanism
        self.stall_timeout = 0.35
        self.stall_timer   = 0.0

        # Micro-anchor (local path reprojection)
        self.ds_micro = 0.03                # Forward look-ahead in s
        self.N_micro  = 7                   # Number of candidate points
        self.w_perp   = 1.0                 # Lateral error weight
        self.w_heading = 0.25               # Heading error weight

        # Debug ring buffer
        self.ring_buffer = deque(maxlen=50)

    # ============================================================================
    # UTILITIES
    # ============================================================================
    
    def normalize_angle(self, a):
        """Wrap angle to [-π, π]"""
        return _wrap(a)

    def clamp(self, u):
        """Clamp control commands to limits"""
        return np.array([
            float(np.clip(u[0], -self.v_max, self.v_max)),
            float(np.clip(u[1], -self.w_max, self.w_max))
        ], dtype=float)

    def ekf_update(self, applied):
        """Update state estimate using unicycle kinematics"""
        v, w = float(applied[0]), float(applied[1])
        th = self.x_hat[2]
        self.x_hat[0] += v * math.cos(th) * self.dt
        self.x_hat[1] += v * math.sin(th) * self.dt
        self.x_hat[2]  = _wrap(self.x_hat[2] + w * self.dt)

    def _plus(self, x):
        """Positive part function"""
        return x if x > 0.0 else 0.0

    # ============================================================================
    # DEPTH ESTIMATION INTERFACE
    # ============================================================================
    
    def set_depth_from_main(self, d_main):
        """Sync depth estimate from main controller"""
        self.d_hat = float(np.clip(d_main, self.d_min, self.d_max))
    
    def get_depth_for_main(self):
        """Provide depth estimate to main controller"""
        return float(np.clip(self.d_hat, self.d_min, self.d_max))

    # ============================================================================
    # HANDOVER LOGIC (DoS Attack Handling)
    # ============================================================================
    
    def handover(self, attack_flag):
        """
        Handle mode switching based on DoS attack status
        
        Parameters:
        -----------
        attack_flag : bool
            True if DoS attack is active, False if clear
        """
        if attack_flag and self.mode == 'main':
            # DoS attack detected → switch to DT
            self.mode = 'dt'
            self.just_switched = 'main2dt'
            self.dt_state = 'TURN'
            self.v_mem = 0.0
            self._start_washout(to_mode='dt')
            print("⚠️  DoS Attack Detected: MAIN → DT")
            
        elif (not attack_flag) and self.mode == 'dt':
            # DoS attack cleared → switch to main
            self.mode = 'main'
            self.just_switched = None
            self._start_washout(to_mode='main')
            print("✅ DoS Cleared: DT → MAIN")

    def _start_washout(self, to_mode):
        """
        Initialize smooth washout transition between modes
        
        Duration scales with command gap to ensure bumpless transfer
        """
        if to_mode == 'dt':
            u_src = self.last_u_main.copy()
            u_dst = self.u_dt.copy()
        else:
            u_src = self.u_dt.copy()
            u_dst = self.last_u_main.copy()
        
        # Scale duration by command difference
        du = float(np.linalg.norm(u_src - u_dst))
        T = np.clip(self.bt_Tmin + self.bt_ku * du, self.bt_Tmin, self.bt_Tmax)
        self.transition_duration = float(T)
        self.in_transition = True
        self.transition_timer = 0.0

    def _snap_phase_on_takeover(self, main_pose):
        """
        Snap phase variable to nearest point on path at DT takeover
        Uses heading-aware search to avoid backward jumps
        """
        x, y, th = float(main_pose[0]), float(main_pose[1]), float(main_pose[2])
        
        # Try different nearest-point methods (trajectory-dependent)
        if hasattr(self.traj, "nearest_s_pruned"):
            self.s = float(self.traj.nearest_s_pruned(
                x, y, th, self.s, max_jump=0.08, cone_deg=70.0
            ))
        elif hasattr(self.traj, "nearest_s_local"):
            self.s = float(self.traj.nearest_s_local(
                x, y, self.s, max_jump=0.08
            ))
        elif hasattr(self.traj, "nearest_s"):
            self.s = float(self.traj.nearest_s(x, y))
        
        # Set initial FSM state based on alignment
        th_ref = self.traj.sample(self.s)[0][2]
        c_t = math.cos(th_ref)*math.cos(th) + math.sin(th_ref)*math.sin(th)
        self.dt_state = 'TURN' if c_t < 0.3 else 'GO'

    # ============================================================================
    # PATH GEOMETRY HELPERS
    # ============================================================================
    
    def _curvature_preview(self, s, ds):
        """
        Preview path curvature in window [s-ds, s+ds]
        Returns maximum curvature magnitude in window
        """
        sA = max(0.0, s - ds)
        sB = min(1.0, s + ds)
        thA = self.traj.sample(sA)[0][2]
        thB = self.traj.sample(sB)[0][2]
        dth = _wrap(thB - thA)
        return abs(dth / max(1e-3, (sB - sA)))

    def _tube_state(self, pose, s_use, c_t):
        """
        Compute tube geometry at given phase
        
        Returns:
        --------
        d_perp : float
            Signed perpendicular distance to path (+ = left, - = right)
        r_eff : float
            Effective tube radius (adapts to curvature and alignment)
        th_ref : float
            Reference heading angle
        kappa : float
            Path curvature
        """
        x_d, _, _ = self.traj.sample(s_use)
        th_ref = float(x_d[2])
        
        # Path tangent and normal vectors
        t_x, t_y = math.cos(th_ref), math.sin(th_ref)
        n_x, n_y = -t_y, t_x
        
        # Signed perpendicular distance
        d_perp = (pose[0]-x_d[0]) * n_x + (pose[1]-x_d[1]) * n_y

        # Adaptive tube radius
        kappa = float(self.traj.curvature(s_use))
        r_eff = self.r0 * (1.0 + self.a_kappa * abs(kappa))
        r_eff *= (1.0 - self.b_align * max(0.0, c_t))
        r_eff = max(0.75*self.r0, min(r_eff, 2.0*self.r0))
        
        return d_perp, r_eff, th_ref, kappa

    def _micro_anchor(self, s, pose):
        """
        Forward-only local reprojection to reduce tracking error
        Searches small window ahead of current phase
        
        Returns:
        --------
        best_s : float
            Phase with minimal lateral and heading error
        """
        s_min = s
        s_max = min(1.0, s + self.ds_micro)
        cand = np.linspace(s_min, s_max, self.N_micro)
        
        best_s = s
        best_J = 1e9
        th = float(pose[2])
        
        for si in cand:
            x_d, _, _ = self.traj.sample(si)
            dx, dy = x_d[0]-pose[0], x_d[1]-pose[1]
            
            # Errors in robot frame
            e_x =  math.cos(th)*dx + math.sin(th)*dy
            e_y = -math.sin(th)*dx + math.cos(th)*dy
            e_th = _wrap(x_d[2] - th)
            
            # Cost: weighted lateral + heading error
            J = self.w_perp*(e_y**2) + self.w_heading*(e_th**2)
            
            if J < best_J:
                best_J, best_s = J, float(si)
        
        return best_s

    def _stall_guard(self, pose, x_d, v_cmd, w_cmd):
        """
        Anti-stall mechanism: nudge forward if stopped too long
        """
        if abs(v_cmd) > 1e-3:
            self.stall_timer = 0.0
            return v_cmd, w_cmd
        
        self.stall_timer += self.dt
        if self.stall_timer < self.stall_timeout:
            return v_cmd, w_cmd
        
        # Stall detected → apply gentle forward nudge
        v_cmd = 0.02
        e_th  = _wrap(x_d[2] - pose[2])
        w_cmd = float(np.clip(
            w_cmd + 1.2*self.k_align_boost*math.sin(e_th),
            -self.w_max, self.w_max
        ))
        
        return v_cmd, w_cmd

    # ============================================================================
    # SAFETY FILTERS (CBF + INWARD SHAPING)
    # ============================================================================
    
    def _cbf_inward_project(self, pose, s_use, v_star, c_t):
        """
        Two-stage safety filtering:
        1. Inward-aware shaping: attenuate outward motion near boundary
        2. Discrete CBF clamp: hard safety guarantee
        
        Parameters:
        -----------
        pose : array
            Current pose [x, y, theta]
        s_use : float
            Phase for reference
        v_star : float
            Desired forward velocity
        c_t : float
            Alignment cosine
            
        Returns:
        --------
        v_safe : float
            Safe forward velocity
        """
        x_d, _, _ = self.traj.sample(s_use)
        th_ref = float(x_d[2])
        th = float(pose[2])
        
        # Approximate normal velocity: d_perp_dot ≈ v * sin(e_th)
        e_th = _wrap(th_ref - th)
        d_perp, r_eff, _, _ = self._tube_state(pose, s_use, c_t)
        dperp_dot = v_star * math.sin(e_th)

        # Stage 1: Inward-aware attenuation
        # If heading outward and near wall, reduce speed
        if (d_perp * dperp_dot) > 0.0 and abs(d_perp) > 0.6 * r_eff:
            # Linear ramp: full speed at 0.6*r_eff, zero at r_eff
            alpha = (abs(d_perp) - 0.6*r_eff) / (0.4*r_eff)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            v_star = (1.0 - alpha) * v_star

        # Stage 2: Hard CBF safety clamp
        v_star = clamp_v_discrete(
            pose, v_star, self.traj, s_use,
            r_base=self.r0, beta=self.a_kappa, lam=self.lambda_cbf
        )
        
        return v_star

    # ============================================================================
    # ω-FIRST FEASIBLE PROJECTION
    # ============================================================================
    
    def _shadow_project(self, v_ref, w_ref, c_t, e_th, kappa_max, pose, s_use, x_d):
        """
        Project DT command to feasible safe set with forward-only constraint
        
        Strategy:
        ---------
        1. Allocate ω first (for heading correction)
        2. Cap v based on remaining ω margin and previewed curvature
        3. Apply safety filters (inward shaping + CBF)
        4. Enforce dynamic progress floor
        5. Smooth with rate regularization
        
        Parameters:
        -----------
        v_ref, w_ref : float
            Raw reference commands from shadow controller
        c_t : float
            Alignment cosine
        e_th : float
            Heading error
        kappa_max : float
            Maximum curvature in preview window
        pose : array
            Current pose
        s_use : float
            Micro-anchored phase
        x_d : array
            Reference state
            
        Returns:
        --------
        v_cmd, w_cmd : float
            Safe, feasible commands
        """
        # === ω ALLOCATION ===
        # Blend: reference + rate memory + main sync
        w_main = float(self.last_u_main[1])
        den = (1.0 + self.lambda_rate + self.lambda_sync)
        w_sync = (w_ref + self.lambda_rate*self.w_last + self.lambda_sync*w_main) / den
        
        # Add alignment correction
        w_align = 0.3*self.k_align_boost*math.sin(e_th)
        w_base = float(np.clip(w_sync + w_align, -self.w_max, self.w_max))

        # Low-pass filter to reduce jitter
        self.w_mem = (1.0 - self.beta_w_rate)*self.w_mem + self.beta_w_rate*w_base
        w_base = self.w_mem

        # === v CAPPING ===
        # Compute remaining ω budget after allocation
        w_margin = max(0.0, self.w_max - abs(w_base))
        
        # Cap v by curvature constraint: v ≤ ω_margin / κ
        if kappa_max < 1e-6:
            v_cap = self.v_max
        else:
            v_cap = min(self.v_max, w_margin / (kappa_max + 1e-6))

        # === v COMPUTATION ===
        # Weighted blend: ref + rate + sync
        v_main = float(self.last_u_main[0])
        v_star = (v_ref + self.lambda_rate*self.v_last + self.lambda_sync*v_main) / den
        v_star = float(np.clip(v_star, 0.0, v_cap))  # Forward-only + cap

        # === SAFETY FILTERS ===
        v_star = self._cbf_inward_project(pose, s_use, v_star, c_t)
        v_star = max(0.0, float(v_star))

        # Extra gating when poorly aligned
        if c_t < 0.3:
            v_star = min(v_star, max(0.0, self.v_max * (c_t / 0.3)))

        # === DYNAMIC PROGRESS FLOOR ===
        # Prevent DT under-speed when safe and aligned
        d_perp, r_eff, _, _ = self._tube_state(pose, s_use, c_t)
        if abs(d_perp) <= r_eff and c_t > self.tau_go:
            # Scale floor by alignment and distance from boundary
            gain = 0.10 * (c_t - self.tau_go) * max(0.0, 1.0 - abs(d_perp)/r_eff)
            v_floor = self.v_floor_progress_base + gain
            v_star = max(v_star, min(v_cap, v_floor))

        # === SMOOTHING ===
        self.v_mem = (1.0 - self.beta_accel)*self.v_mem + self.beta_accel*v_star
        v_star = max(0.0, float(self.v_mem))

        # === ANTI-STALL ===
        v_star, w_base = self._stall_guard(pose, x_d, v_star, w_base)
        v_star = max(0.0, float(v_star))

        # Update rate memory
        self.v_last = float(v_star)
        self.w_last = float(w_base)
        
        return v_star, w_base

    # ============================================================================
    # MAIN CONTROL STEP
    # ============================================================================
    
    def step(self, t, attack_flag, main_pose, main_control):
        """
        Main control tick - handles both normal and DT modes
        
        Parameters:
        -----------
        t : float
            Current time [s]
        attack_flag : bool
            True if DoS attack is active
        main_pose : np.ndarray
            Current robot pose [x, y, theta]
        main_control : np.ndarray
            Main controller output [v, omega]
        
        Returns:
        --------
        u_out : np.ndarray
            Control command to apply [v, omega]
        """
        # Check for mode transitions
        self.handover(attack_flag)

        # ========================================================================
        # MAIN MODE (No Attack)
        # ========================================================================
        if self.mode == 'main':
            self.x_hat = main_pose.copy()
            self.u_out = self.clamp(main_control)
            self.last_u_main = self.u_out.copy()
            applied = self.u_out

        # ========================================================================
        # DT MODE (During DoS Attack)
        # ========================================================================
        else:
            # Initialize on first DT step after takeover
            if self.just_switched == 'main2dt':
                self.x_hat = main_pose.copy()
                self._snap_phase_on_takeover(main_pose)
                self.just_switched = None

            # --- Micro-anchor: forward reprojection ---
            s_use = self._micro_anchor(self.s, main_pose)

            # --- Reference state at anchored phase ---
            x_d, _, u_ff = self.traj.sample(s_use)
            th_ref = float(x_d[2])

            # --- Tracking errors in robot frame ---
            x, y, th = float(main_pose[0]), float(main_pose[1]), float(main_pose[2])
            dx, dy = x_d[0]-x, x_d[1]-y
            e_x  =  math.cos(th)*dx + math.sin(th)*dy
            e_y  = -math.sin(th)*dx + math.cos(th)*dy
            e_th =  _wrap(th_ref - th)

            # --- Shadow controller (same law as main) ---
            v_ref, w_ref = compute_control(
                e_x, e_y, e_th,
                float(u_ff[0]), float(u_ff[1]),
                self.d_hat, self.gains
            )
            v_ref = float(np.clip(v_ref, -self.v_max, self.v_max))
            w_ref = float(np.clip(w_ref, -self.w_max, self.w_max))

            # --- Depth adaptation during DT ---
            self.d_hat = update_depth_estimate(
                self.d_hat, e_x, e_y, e_th, float(u_ff[0]),
                self.gamma, self.gains['k3'], self.d_min, self.d_max, self.dt
            )

            # --- Geometry queries ---
            c_t = math.cos(th_ref)*math.cos(th) + math.sin(th_ref)*math.sin(th)
            kappa_max = self._curvature_preview(s_use, self.ds_preview)

            # --- FSM: TURN→GO (prevent reversals) ---
            if self.dt_state == 'TURN' or c_t < self.tau_turn:
                # TURN mode: rotate only, no forward motion
                v_cmd = 0.0
                w_cmd = float(np.clip(
                    self.k_align_boost*math.sin(e_th),
                    -self.w_max, self.w_max
                ))
                # Check transition to GO
                if c_t > self.tau_go and abs(e_th) < self.theta_go:
                    self.dt_state = 'GO'
            else:
                # GO mode: full projection with safety
                v_cmd, w_cmd = self._shadow_project(
                    v_ref, w_ref, c_t, e_th, kappa_max,
                    main_pose, s_use, x_d
                )

            self.u_dt  = np.array([v_cmd, w_cmd], dtype=float)
            self.u_out = self.clamp(self.u_dt)
            applied    = self.u_out

        # ========================================================================
        # WASHOUT (Smooth Handover)
        # ========================================================================
        if self.in_transition:
            # Linear blend from source to target command
            alpha = float(np.clip(
                self.transition_timer/self.transition_duration, 0.0, 1.0
            ))
            
            if self.mode == 'dt':
                # Transitioning main → dt
                self.u_out = (1 - alpha)*self.last_u_main + alpha*self.u_out
            else:
                # Transitioning dt → main
                self.u_out = (1 - alpha)*self.u_dt + alpha*self.u_out
            
            self.transition_timer += self.dt
            if self.transition_timer >= self.transition_duration:
                self.in_transition = False
            
            applied = self.u_out

        # ========================================================================
        # FINALIZE
        # ========================================================================
        
        # Ensure limits
        self.u_out = self.clamp(self.u_out)
        
        # Update state estimate
        self.ekf_update(self.u_out)

        # Log to debug ring buffer
        self.ring_buffer.append((
            self.x_hat.copy(),
            float(self.s),
            self.u_dt.copy(),
            float(t)
        ))
        
        return self.u_out.copy()

    # ============================================================================
    # DEBUG INTERFACE
    # ============================================================================
    
    def get_debug(self):
        """
        Return debug information for logging/visualization
        
        Returns:
        --------
        dict with keys:
            x_hat : np.ndarray - estimated pose
            s : float - phase variable
            u_dt : np.ndarray - DT command
            u_out : np.ndarray - actual output
            mode : str - 'main' or 'dt'
            transition : bool - in washout?
            ring : list - recent history
        """
        return {
            "x_hat": self.x_hat.copy(),
            "s": float(self.s),
            "u_dt": self.u_dt.copy(),
            "u_out": self.u_out.copy(),
            "mode": self.mode,
            "transition": self.in_transition,
            "ring": list(self.ring_buffer),
        }