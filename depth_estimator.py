# depth_estimator.py
import math


def project(value, d_hat, d_min, d_max):
    """
    Projection operator to keep d_hat within [d_min, d_max]
    """
    if (d_hat <= d_min and value < 0) or (d_hat >= d_max and value > 0):
        return 0.0
    return value

def update_depth_estimate(d_hat, e_x, e_y, e_theta, v_d_star, gamma, k3, d_min, d_max, dt):
    """
    Update the estimated depth (hat{d}) using the adaptive law from the paper.
    
    Args:
        d_hat: current depth estimate
        e_x, e_y, e_theta: tracking errors
        v_d_star: desired linear velocity (scaled)
        gamma: learning rate (Gamma)
        k3: control gain
        d_min, d_max: projection bounds
        dt: time step

    Returns:
        Updated depth estimate (float)
    """
    numerator = k3 * e_x * v_d_star * math.cos(e_theta)
    denominator = 1 + e_x**2 + e_y**2
    d_hat_dot = gamma * project(numerator / denominator, d_hat, d_min, d_max)

    d_hat_new = d_hat + d_hat_dot * dt
    d_hat_new = max(min(d_hat_new, d_max), d_min)  # hard clamp (redundant but safe)
    
    return d_hat_new
