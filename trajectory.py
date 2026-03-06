# trajectory.py

import math
import numpy as np

class TrajectoryGenerator:
    def __init__(self, radius=1.0, speed=0.1):
        """
        Initialize a circular trajectory.
        radius: radius of the circular path (in meters)
        speed: desired forward speed (in m/s)
        """
        self.radius = radius
        self.speed = speed
        self.angular_speed = speed / radius  # omega_d = v_d / r
    def total_length(self):
    # If you have a known radius and circular path
        return 2 * math.pi * self.radius


    def get_desired_state(self, t):
        """
        Return desired pose and velocities at time t
        Pose and velocity are scaled (i.e., x_d*, y_d*, v_d*)
        """
        x_d_star = self.radius * math.cos(self.angular_speed * t)
        y_d_star = self.radius * math.sin(self.angular_speed * t)
        theta_d = self.angular_speed * t + math.pi / 2  # heading tangent to the circle

        v_d_star = self.speed / 1.0  # Simulated scaled velocity (assume d* = 1 for now)
        omega_d = self.angular_speed

        return x_d_star, y_d_star, theta_d, v_d_star, omega_d

    def sample(self, s):
        # Clamp s to [0, 1]
        s = max(0.0, min(1.0, s))

        # Assume circular path of known radius centered at (0, 0)
        angle = 2 * math.pi * s

        # Desired pose (x, y, theta)
        x = self.radius * math.cos(angle)
        y = self.radius * math.sin(angle)
        theta = angle + math.pi / 2  # Face tangent to path

        x_d = np.array([x, y, theta])

        # Desired velocity vector
        #v = self.linear_speed
        v = self.speed
        v_x = -v * math.sin(angle)
        v_y = v * math.cos(angle)
        v_d = np.array([v_x, v_y])

        # Feedforward control (just v, omega for now)
        u_ff = np.array([v, v / self.radius])

        return x_d, v_d, u_ff
