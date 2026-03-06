import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import irsim
import math
from trajectory import TrajectoryGenerator
from controller import compute_control
from depth_estimator import update_depth_estimate
from config import gains, gamma, d_min, d_max, initial_depth, lambda_v, lambda_omega, dt, total_time

# Load the IR-SIM environment (ensure path is correct)
env = irsim.make("ir_sim/circle_world.yaml")
traj = TrajectoryGenerator()
d_hat = initial_depth

# Main simulation loop
for step in range(int(total_time / dt)):
    t = step * dt
    x, y, theta = env.robot.state  # ✅ Correct way to access robot pose

    # Desired trajectory
    x_d, y_d, theta_d, v_d_star, omega_d = traj.get_desired_state(t)

    # Tracking error
    dx = x_d - x
    dy = y_d - y
    e_x = math.cos(theta) * dx + math.sin(theta) * dy
    e_y = -math.sin(theta) * dx + math.cos(theta) * dy
    e_theta = theta_d - theta
    e_theta = (e_theta + math.pi) % (2 * math.pi) - math.pi

    # Control output
    v, omega = compute_control(e_x, e_y, e_theta, v_d_star, omega_d, d_hat, gains)
    v = max(min(v, lambda_v), -lambda_v)
    omega = max(min(omega, lambda_omega), -lambda_omega)

    # Update depth estimate
    d_hat = update_depth_estimate(d_hat, e_x, e_y, e_theta, v_d_star,
                                   gamma, gains['k3'], d_min, d_max, dt)

    # ✅ Apply control to robot
    # env.robot.set_control([v, omega])
    env.robot.control = [v, omega]


    # Advance simulation
    env.step()
    env.render()
