# main.py

import math
import matplotlib.pyplot as plt
import numpy as np

from robot import Robot
from trajectory import TrajectoryGenerator
from controller import compute_control
from depth_estimator import update_depth_estimate
from config import gains, gamma, d_min, d_max, initial_depth, lambda_v, lambda_omega, dt, total_time

# --------------------------
# Utility: Compute Tracking Error
# --------------------------
def compute_tracking_error(x, y, theta, x_d_star, y_d_star, theta_d):
    """
    Compute tracking error in the robot's local frame
    """
    dx = x_d_star - x
    dy = y_d_star - y

    e_x = math.cos(theta) * dx + math.sin(theta) * dy
    e_y = -math.sin(theta) * dx + math.cos(theta) * dy
    e_theta = theta_d - theta

    # Normalize angle to [-pi, pi]
    e_theta = (e_theta + math.pi) % (2 * math.pi) - math.pi

    return e_x, e_y, e_theta


#------------------------------
# DoS attack
attack_start_time = 10.0  # seconds
attack_duration = 5.0     # seconds

# --------------------------
# Initialize System
# --------------------------
robot = Robot()
traj = TrajectoryGenerator()
d_hat = initial_depth
true_depth = 2.0  # ground truth depth (for validation)

# --------------------------
# Data Logs
# --------------------------
pose_log = []
desired_pose_log = []
error_log = []
depth_log = []
depth_error_log = []

# --------------------------
# Simulation Loop
# --------------------------
t = 0.0
while t < total_time:
    # 1. Get current robot pose
    x, y, theta = robot.get_pose()

    # 2. Get desired (scaled) trajectory at time t
    x_d_star, y_d_star, theta_d, v_d_star, omega_d = traj.get_desired_state(t)

    # 3. Log desired pose
    desired_pose_log.append((x_d_star, y_d_star))

    # 4. Compute tracking error in robot's frame
    e_x, e_y, e_theta = compute_tracking_error(x, y, theta, x_d_star, y_d_star, theta_d)

    # 5. Compute control commands
    v, omega = compute_control(e_x, e_y, e_theta, v_d_star, omega_d, d_hat, gains)

    # 6. Saturate velocities
    v = max(min(v, lambda_v), -lambda_v)
    omega = max(min(omega, lambda_omega), -lambda_omega)

    # 7. Update robot state
    robot.update(v, omega, dt)

    # 8. Update depth estimate
    d_hat = update_depth_estimate(d_hat, e_x, e_y, e_theta, v_d_star, gamma, gains['k3'], d_min, d_max, dt)

    # 9. Log everything
    pose_log.append((x, y))
    error_log.append((e_x, e_y, e_theta))
    depth_log.append(d_hat)
    depth_error_log.append(abs(true_depth - d_hat))

    # 10. Advance time
    t += dt
    
    omega = 0.0
else:
    v_control = v
    omega_control = omega


# --------------------------
# Plotting
# --------------------------
# 1. Trajectory
xs, ys = zip(*pose_log)
xd, yd = zip(*desired_pose_log)

plt.figure()
plt.plot(xs, ys, label='Robot Path')
plt.plot(xd, yd, '--', label='Desired Path', color='orange')
plt.title('Robot vs Desired Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid()
plt.axis('equal')

# 2. Depth Estimate
plt.figure()
plt.plot(depth_log)
plt.title('Depth Estimate Over Time')
plt.xlabel('Timestep')
plt.ylabel('Estimated Depth')
plt.grid()

# 3. Tracking Errors
errors = np.array(error_log)
plt.figure()
plt.plot(errors[:, 0], label='e_x')
plt.plot(errors[:, 1], label='e_y')
plt.plot(errors[:, 2], label='e_theta')
plt.title('Tracking Errors Over Time')
plt.xlabel('Timestep')
plt.ylabel('Error')
plt.legend()
plt.grid()

# 4. Depth Estimation Error
plt.figure()
plt.plot(depth_error_log)
plt.title('Absolute Depth Estimation Error Over Time')
plt.xlabel('Timestep')
plt.ylabel('|d* - d_hat|')
plt.grid()

plt.show()

