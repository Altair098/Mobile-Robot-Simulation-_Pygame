# import numpy as np
# import math
# from collections import deque


# class DTRescueController:
#     def __init__(self, robot_model, trajectory_gen, dt=0.1):
#         self.dt = dt
#         self.robot = robot_model
#         self.traj = trajectory_gen

#         # EKF states (pose estimate and covariance placeholder)
#         self.x_hat = np.zeros(3)  # [x, y, theta]
#         self.P = np.eye(3) * 0.01

#         # Control memory
#         self.last_u_main = np.zeros(2)
#         self.u_dt = np.zeros(2)
#         self.u_out = np.zeros(2)

#         # Progress tracking
#         self.s = 0.0

#         # Washout transition state
#         self.in_transition = False
#         self.transition_timer = 0.0
#         self.transition_duration = 0.1  # 100ms

#         # Modes: 'main', 'dt'
#         self.mode = 'main'

#         # Buffers for debug/logging
#         self.ring_buffer = deque(maxlen=50)

#     def ekf_update(self, control):
#         v, omega = control
#         theta = self.x_hat[2]

#         # Simple EKF predict step
#         dx = v * math.cos(theta) * self.dt
#         dy = v * math.sin(theta) * self.dt
#         dtheta = omega * self.dt

#         self.x_hat += np.array([dx, dy, dtheta])
#         self.x_hat[2] = self.normalize_angle(self.x_hat[2])

#     def compute_s_from_pose_delta(self, x_prev, x_curr):
#         dx = x_curr[0] - x_prev[0]
#         dy = x_curr[1] - x_prev[1]
#         ds = math.hypot(dx, dy)
#         self.s = min(1.0, self.s + ds / self.traj.total_length())

#     def shadow_control(self):
#         x_d, v_d, u_ff = self.traj.sample(self.s)
#         v_hat = np.array([0.0, 0.0])  # Placeholder

#         kx = np.eye(3) * 0.2
#         kv = np.eye(2) * 0.0

#         state_error = x_d - self.x_hat
#         u_fb = kx @ state_error[:3]  # position feedback

#         self.u_dt = u_ff + u_fb[:2]  # combine FF + FB
#         self.u_dt = self.clamp(self.u_dt)

#     def clamp(self, u):
#         v = np.clip(u[0], -1.0, 1.0)
#         omega = np.clip(u[1], -1.0, 1.0)
#         return np.array([v, omega])

#     def normalize_angle(self, theta):
#         return (theta + np.pi) % (2 * np.pi) - np.pi

#     def handover(self, attack_flag):
#         if attack_flag and self.mode == 'main':
#             self.mode = 'dt'
#             self.in_transition = True
#             self.transition_timer = 0.0
#             print("⚠️ Handover: MAIN → DT")

#         elif not attack_flag and self.mode == 'dt':
#             self.mode = 'main'
#             self.in_transition = True
#             self.transition_timer = 0.0
#             print("✅ Handover: DT → MAIN")

#     def step(self, t, attack_flag, main_pose, main_control):
#         # Log previous pose
#         self.ring_buffer.append((self.x_hat.copy(), self.s, self.u_dt.copy(), t))

#         self.handover(attack_flag)

#         # Shadow: update EKF every tick
#         self.ekf_update(main_control)

#         # Update phase progress (s)
#         self.compute_s_from_pose_delta(self.x_hat, main_pose)

#         # Sample trajectory
#         x_d, v_d, u_ff = self.traj.sample(self.s)

#         if self.mode == 'main':
#             self.x_hat = main_pose.copy()
#             self.u_out = main_control

#         elif self.mode == 'dt':
#             self.shadow_control()
#             self.u_out = self.u_dt

#         # Washout transition blending
#         if self.in_transition:
#             alpha = self.transition_timer / self.transition_duration
#             alpha = np.clip(alpha, 0.0, 1.0)

#             if self.mode == 'dt':
#                 self.u_out = (1 - alpha) * self.last_u_main + alpha * self.u_dt
#             else:
#                 self.u_out = (1 - alpha) * self.u_dt + alpha * self.last_u_main

#             self.transition_timer += self.dt

#             if self.transition_timer >= self.transition_duration:
#                 self.in_transition = False

#         if self.mode == 'main':
#             self.last_u_main = main_control

#         return self.u_out.copy()

#     def get_debug(self):
#         return {
#             'x_hat': self.x_hat.copy(),
#             's': self.s,
#             'u_dt': self.u_dt.copy(),
#             'u_out': self.u_out.copy(),
#             'mode': self.mode,
#             'transition': self.in_transition,
#             'ring': list(self.ring_buffer),
#         }

# import numpy as np
# import math
# from collections import deque


# class DTRescueController:
#     def __init__(self, robot_model, trajectory_gen, dt=0.1):
#         self.dt = dt
#         self.robot = robot_model
#         self.traj = trajectory_gen

#         # EKF states (pose estimate and covariance placeholder)
#         self.x_hat = np.zeros(3)  # [x, y, theta]
#         self.P = np.eye(3) * 0.01

#         # Control memory
#         self.last_u_main = np.zeros(2)
#         self.u_dt = np.zeros(2)
#         self.u_out = np.zeros(2)

#         # Progress tracking
#         self.s = 0.0
#         self.s_prev_pose = None  # For arc-length based tracking

#         # Washout transition state
#         self.in_transition = False
#         self.transition_timer = 0.0
#         self.transition_duration = 0.1  # 100ms

#         # Modes: 'main', 'dt'
#         self.mode = 'main'

#         # Buffers for debug/logging
#         self.ring_buffer = deque(maxlen=50)

#     def ekf_update(self, control):
#         v, omega = control
#         theta = self.x_hat[2]

#         dx = v * math.cos(theta) * self.dt
#         dy = v * math.sin(theta) * self.dt
#         dtheta = omega * self.dt

#         self.x_hat += np.array([dx, dy, dtheta])
#         self.x_hat[2] = self.normalize_angle(self.x_hat[2])

#     def update_phase_progress(self, pose):
#         if self.s_prev_pose is not None:
#             dx = pose[0] - self.s_prev_pose[0]
#             dy = pose[1] - self.s_prev_pose[1]
#             arc = math.hypot(dx, dy)
#             self.s += arc / self.traj.total_length()
#             self.s = min(1.0, max(0.0, self.s))
#         self.s_prev_pose = pose.copy()

#     def shadow_control(self):
#         x_d, v_d, u_ff = self.traj.sample(self.s)

#         kx = np.diag([0.5, 0.5, 0.2])
#         kv = np.diag([0.0, 0.0])

#         state_error = x_d - self.x_hat
#         state_error[2] = self.normalize_angle(state_error[2])
#         u_fb = kx @ state_error[:3]

#         self.u_dt = u_ff + u_fb[:2]
#         self.u_dt = self.clamp(self.u_dt)

#     def clamp(self, u):
#         v = np.clip(u[0], -1.0, 1.0)
#         omega = np.clip(u[1], -1.0, 1.0)
#         return np.array([v, omega])

#     def normalize_angle(self, theta):
#         return (theta + np.pi) % (2 * np.pi) - np.pi

#     def handover(self, attack_flag):
#         if attack_flag and self.mode == 'main':
#             self.mode = 'dt'
#             self.in_transition = True
#             self.transition_timer = 0.0
#             print("⚠️ Handover: MAIN → DT")

#         elif not attack_flag and self.mode == 'dt':
#             self.mode = 'main'
#             self.in_transition = True
#             self.transition_timer = 0.0
#             print("✅ Handover: DT → MAIN")

#     def step(self, t, attack_flag, main_pose, main_control):
#         self.handover(attack_flag)

#         # Shadow: always update EKF with main control
#         self.ekf_update(main_control)

#         # Phase progress update from actual motion
#         self.update_phase_progress(main_pose)

#         # Mirror state from main when healthy
#         if self.mode == 'main':
#             self.x_hat = main_pose.copy()
#             self.u_out = main_control
#             self.s_prev_pose = main_pose.copy()

#         # Compute DT control if active
#         elif self.mode == 'dt':
#             self.shadow_control()
#             self.u_out = self.u_dt

#         # Washout transition
#         if self.in_transition:
#             alpha = self.transition_timer / self.transition_duration
#             alpha = np.clip(alpha, 0.0, 1.0)

#             if self.mode == 'dt':
#                 self.u_out = (1 - alpha) * self.last_u_main + alpha * self.u_dt
#             else:
#                 self.u_out = (1 - alpha) * self.u_dt + alpha * self.last_u_main

#             self.transition_timer += self.dt
#             if self.transition_timer >= self.transition_duration:
#                 self.in_transition = False

#         if self.mode == 'main':
#             self.last_u_main = main_control

#         self.u_out = self.clamp(self.u_out)

#         # Log debug ring
#         self.ring_buffer.append((self.x_hat.copy(), self.s, self.u_dt.copy(), t))

#         return self.u_out.copy()

#     def get_debug(self):
#         return {
#             'x_hat': self.x_hat.copy(),
#             's': self.s,
#             'u_dt': self.u_dt.copy(),
#             'u_out': self.u_out.copy(),
#             'mode': self.mode,
#             'transition': self.in_transition,
#             'ring': list(self.ring_buffer),
#         }

import numpy as np
import math
from collections import deque


class DTRescueController:
    def __init__(self, robot_model, trajectory_gen, dt=0.1):
        self.dt = dt
        self.robot = robot_model
        self.traj = trajectory_gen

        self.x_hat = np.zeros(3)  # [x, y, theta]
        self.P = np.eye(3) * 0.01

        self.last_u_main = np.zeros(2)
        self.u_dt = np.zeros(2)
        self.u_out = np.zeros(2)

        self.s = 0.0
        self.s_prev_pose = None
        self.s_main = 0.0
        self.last_main_pose = np.zeros(3)

        self.in_transition = False
        self.transition_timer = 0.0
        self.transition_duration = 0.1

        self.mode = 'main'
        self.ring_buffer = deque(maxlen=50)

    def ekf_update(self, control):
        v, omega = control
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

    def shadow_control(self):
        x_d, v_d, u_ff = self.traj.sample(self.s)

        kx = np.diag([0.5, 0.5, 0.2])
        kv = np.diag([0.0, 0.0])

        state_error = x_d - self.x_hat
        state_error[2] = self.normalize_angle(state_error[2])
        u_fb = kx @ state_error[:3]

        self.u_dt = u_ff + u_fb[:2]
        self.u_dt = self.clamp(self.u_dt)

    def clamp(self, u):
        v = np.clip(u[0], -1.0, 1.0)
        omega = np.clip(u[1], -1.0, 1.0)
        return np.array([v, omega])

    def normalize_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def handover(self, attack_flag):
        if attack_flag and self.mode == 'main':
            self.mode = 'dt'
            self.in_transition = True
            self.transition_timer = 0.0
            print("⚠️ Handover: MAIN → DT")

            self.x_hat = self.last_main_pose.copy()
            self.s_prev_pose = self.last_main_pose.copy()
            self.s = self.s_main
            self.s_prev_pose = self.last_main_pose.copy()
            print(f'🔁 DT takeover @ s={self.s:.3f}, x={self.x_hat}')

        elif not attack_flag and self.mode == 'dt':
            self.mode = 'main'
            self.in_transition = True
            self.transition_timer = 0.0
            print("✅ Handover: DT → MAIN")

    def step(self, t, attack_flag, main_pose, main_control):
        self.handover(attack_flag)

        self.ekf_update(main_control)

        self.last_main_pose = main_pose.copy()

        if self.mode == 'main':
            self.x_hat = main_pose.copy()
            self.s_main = self.s
            self.update_phase_progress(main_pose)
            self.u_out = main_control
            self.s_prev_pose = main_pose.copy()

        elif self.mode == 'dt':
            self.update_phase_progress(self.x_hat)
            self.shadow_control()
            self.u_out = self.u_dt

        if self.in_transition:
            alpha = self.transition_timer / self.transition_duration
            alpha = np.clip(alpha, 0.0, 1.0)

            if self.mode == 'dt':
                self.u_out = (1 - alpha) * self.last_u_main + alpha * self.u_dt
            else:
                self.u_out = (1 - alpha) * self.u_dt + alpha * self.last_u_main

            self.transition_timer += self.dt
            if self.transition_timer >= self.transition_duration:
                self.in_transition = False

        if self.mode == 'main':
            self.last_u_main = main_control

        self.u_out = self.clamp(self.u_out)
        self.ring_buffer.append((self.x_hat.copy(), self.s, self.u_dt.copy(), t))

        return self.u_out.copy()

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


