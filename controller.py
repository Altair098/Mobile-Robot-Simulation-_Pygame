# controller.py

import math
import numpy as np

# ---- Diagnostics logs ----
log = {
    "t": [],
    "x": [], "y": [], "th": [],
    "xd": [], "yd": [], "thd": [],
    "e_perp": [], "c_t": [], "h": [], "r_tube": [],
    "s": [], "s_near": [],
    "u_out_v": [], "u_out_w": [],
    "u_main_v": [], "u_main_w": [],
    "u_dt_v": [], "u_dt_w": [],
    "d_main": [], "d_dt": [],
    "mode_dt": [],          # 1=DT active, 0=Main
    "attack": [],           # 1=attack active
}


def compute_control(e_x, e_y, e_theta, v_d_star, omega_d, d_hat, gains):
    """
    Compute control inputs v and omega using the controller from the paper.

    Args:
        e_x, e_y, e_theta: tracking errors
        v_d_star: desired linear velocity (scaled)
        omega_d: desired angular velocity
        d_hat: current estimate of depth
        gains: dictionary with keys 'k1', 'k2', 'k3'

    Returns:
        v, omega: control commands
    """
    k1 = gains['k1']
    k2 = gains['k2']
    k3 = gains['k3']

    # Equation (11): Linear velocity with saturation
    v = k1 * math.tanh(e_x) + d_hat * v_d_star * math.cos(e_theta)

    # Equation (12): Angular velocity
    steering_term = 0.0
    if abs(e_theta) > 1e-6:  # avoid division by zero
        steering_term = (v_d_star * math.sin(e_theta) / e_theta) * (k3 * e_y / (1 + e_x**2 + e_y**2))

    omega = omega_d + k2 * math.tanh(e_theta) + steering_term

    return v, omega

    
