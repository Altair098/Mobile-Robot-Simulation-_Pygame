import pygame
import math
import numpy as np
import time

import torch
from lstm_model import RobotDynamicsLSTM


from robot import Robot
from controller import compute_control
from depth_estimator import update_depth_estimate
from trajectory import TrajectoryGenerator
from robot import normalize_angle

from config import (
    gains, gamma, d_min, d_max, initial_depth,
    lambda_v, lambda_omega, dt, total_time
)

# ------------------ Pygame Setup ------------------
pygame.init()
screen_width, screen_height = 800, 800
screen = pygame.display.set_mode((screen_width, screen_height))

# Precompute desired circular path (drawn once)
radius = 100  # pixels (adjust for visual scaling)
center = (screen_width // 2, screen_height // 2)
circle_points = []

for i in range(0, 360, 2):
    angle = math.radians(i)
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    circle_points.append((int(x), int(y)))



pygame.display.set_caption("Robot with Digital Twin (DoS Simulation)")
clock = pygame.time.Clock()

# Coordinate transformation
def world_to_screen(x, y):
    return int(screen_width / 2 + x * 100), int(screen_height / 2 - y * 100)

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 102, 204)    # main robot trajectory
RED = (255, 0, 0)        # digital twin
BLACK = (0, 0, 0)

# ------------------ Simulation Parameters ------------------
# attack_start_time = 35.0    # seconds
# attack_duration = 15.0       # seconds
true_depth = 2.0

# Initialize robots
main_robot = Robot()
dt_robot = Robot()  # Digital Twin robot
d_hat = initial_depth
traj = TrajectoryGenerator()


# ------------------ Simulation Components ------------------
main_robot = Robot()
dt_robot = Robot()
d_hat = initial_depth
traj = TrajectoryGenerator()

# ------------------ LSTM Model for Digital Twin ------------------
from lstm_model import RobotDynamicsLSTM
import torch

lstm_model = RobotDynamicsLSTM()
lstm_model.reset_hidden()

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# For training
previous_state = None
previous_control = None


# Logging
trajectory_log = []
dt_trajectory_log = []

# ------------------ Simulation Loop ------------------
start_time = time.time()
running = True
t = 0.0

attack_active = False
attack_start_time = None
attack_duration = 10.0  # seconds (you can change this later)


while running and t < total_time:
    screen.fill(WHITE)

    # Draw desired circular path in background
    for point in circle_points:
        pygame.draw.circle(screen, (0, 200, 0), point, 1)  # Green dots

    # Time update
    t = time.time() - start_time

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_a and not attack_active:
                attack_active = True
                attack_start_time = t
                print("🚨 DoS attack started at {:.2f}s".format(t))

    # Determine if attack is active
    if attack_start_time is not None:
        attack_active = attack_start_time <= t <= (attack_start_time + attack_duration)
    else:
        attack_active = False

    # Sync DT with main robot before attack
    if attack_start_time is not None:
        if attack_start_time - dt < t < attack_start_time:
            dt_robot.set_pose(main_robot.get_pose())

            print(f"⏱️ Syncing DT at time: {t:.2f}")


    # Get desired trajectory
    x_d_star, y_d_star, theta_d, v_d_star, omega_d = traj.get_desired_state(t)
    x, y, theta = main_robot.get_pose()

    # Compute tracking error
    dx = x_d_star - x
    dy = y_d_star - y
    e_x = math.cos(theta) * dx + math.sin(theta) * dy
    e_y = -math.sin(theta) * dx + math.cos(theta) * dy
    e_theta = theta_d - theta
    e_theta = (e_theta + math.pi) % (2 * math.pi) - math.pi

    # Compute controller output
    v, omega = compute_control(e_x, e_y, e_theta, v_d_star, omega_d, d_hat, gains)
    v = max(min(v, lambda_v), -lambda_v)
    omega = max(min(omega, lambda_omega), -lambda_omega)

    # DoS attack handling
    if attack_active:
        if t - attack_start_time <= attack_duration:
            v_control, omega_control = 0.0, 0.0  # freeze main robot
            ####dt_robot.update(v, omega, dt)        # DT keeps moving
            # Freeze main robot
            v_control, omega_control = 0.0, 0.0
            # Get last DT state
            x_dt, y_dt, theta_dt = dt_robot.get_pose()
            # Prepare input: [x, y, θ, v, ω]
            lstm_input = torch.tensor(
                [[[x_dt, y_dt, theta_dt, v, omega]]], dtype=torch.float32
            )

            # Predict next state using LSTM
            lstm_model.detach_hidden()
            with torch.no_grad():
                #predicted_state = lstm_model(lstm_input).squeeze().numpy()
                delta = lstm_model(lstm_input).squeeze().numpy()
            x_next = x_dt + delta[0]
            y_next = y_dt + delta[1]
            theta_next = normalize_angle(theta_dt + delta[2])

            # Update DT with predicted state (overwrite pose)
            #dt_robot.set_pose(predicted_state)
            print(f"📡 LSTM predicted Δ: {delta}, new DT pose: {[x_next, y_next, theta_next]}")

            dt_robot.set_pose([x_next, y_next, theta_next])


        else:
            attack_active = False
            main_robot.set_pose(dt_robot.get_pose())
            print("✅ Attack ended. Robot resumed from DT state.")
            v_control, omega_control = v, omega
    else:
        v_control, omega_control = v, omega

    # Update main robot
    main_robot.update(v_control, omega_control, dt)

    # # ------------ ONLINE LSTM TRAINING ------------
    # current_state = main_robot.get_pose()
    # current_control = [v_control, omega_control]

    # if previous_state is not None and previous_control is not None:
    #     input_tensor = torch.tensor(
    #         [[[previous_state[0], previous_state[1], previous_state[2],
    #            previous_control[0], previous_control[1]]]],
    #         dtype=torch.float32
    #     )
    #     target_tensor = torch.tensor(
    #         [[current_state[0], current_state[1], current_state[2]]],
    #         dtype=torch.float32
    #     )
    #     lstm_model.detach_hidden()
    #     output = lstm_model(input_tensor)
    #     loss = loss_fn(output, target_tensor)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     # Optional print:
    #     print(f"LSTM loss: {loss.item():.6f}")

    # previous_state = current_state
    # previous_control = current_control

    # ------------- ONLINE LSTM TRAINING (PREDICT DELTAS) ---------------
    
    current_state = main_robot.get_pose()
    current_control = [v_control, omega_control]

    if previous_state is not None and previous_control is not None:
        # INPUT: x, y, theta, v, omega (previous)
        input_tensor = torch.tensor(
            [[[previous_state[0], previous_state[1], previous_state[2],
                previous_control[0], previous_control[1]]]],
            dtype=torch.float32
        )

        # TARGET = delta x, delta y, delta theta
        target_dx = current_state[0] - previous_state[0]
        target_dy = current_state[1] - previous_state[1]
        target_dtheta = normalize_angle(current_state[2] - previous_state[2])

        target_tensor = torch.tensor([[target_dx, target_dy, target_dtheta]], dtype=torch.float32)

        # Train LSTM
        lstm_model.detach_hidden()
        output = lstm_model(input_tensor)
        loss = loss_fn(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Δx={target_dx:.4f}, Δy={target_dy:.4f}, Δθ={target_dtheta:.4f}, loss={loss.item():.6f}")


        # Optional:
        # print(f"LSTM loss: {loss.item():.6f}")

    # Save for next step
    previous_state = current_state
    previous_control = current_control


    # Update depth estimate
    d_hat = update_depth_estimate(
        d_hat, e_x, e_y, e_theta, v_d_star, gamma, gains['k3'],
        d_min, d_max, dt
    )

    # Log paths
    trajectory_log.append(main_robot.get_pose())
    if attack_active:
        dt_trajectory_log.append(dt_robot.get_pose())

    # Draw trajectories
    for px, py, _ in trajectory_log:
        sx, sy = world_to_screen(px, py)
        pygame.draw.circle(screen, BLUE, (sx, sy), 2)

    for px, py, _ in dt_trajectory_log:
        sx, sy = world_to_screen(px, py)
        pygame.draw.circle(screen, RED, (sx, sy), 2)

    # Draw robots
    rx, ry = world_to_screen(*main_robot.get_pose()[:2])
    pygame.draw.circle(screen, BLACK, (rx, ry), 6)

    if attack_active:
        dx, dy = world_to_screen(*dt_robot.get_pose()[:2])
        pygame.draw.circle(screen, RED, (dx, dy), 6)

    pygame.display.flip()
    clock.tick(1 / dt)


# ------------------ Wait for exit ------------------
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
