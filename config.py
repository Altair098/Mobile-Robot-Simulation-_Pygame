# --- Controller Gains ---
gains = {
    'k1': 1.0,   # Good for smooth forward control
    'k2': 1.0,   # Heading angle correction is stable
    'k3': 10.0   # ⬆️ Stronger lateral correction to suppress ey
}

# --- Depth Estimator Settings ---
gamma = 0.8         # ⬇️ Slightly reduce update speed for stability
d_min = 1.7         # ⬆️ Prevent underestimation that causes overshooting
d_max = 2.1         # ⬇️ Prevent overestimation that slows robot
initial_depth = 1.9 # ✅ Start closer to expected scale (true = 2.0)

# --- Velocity Saturation Limits ---
lambda_v = 0.15
lambda_omega = 0.2

# --- Simulation Settings ---
dt = 0.05
total_time = 200
