# robot.py

import math


def normalize_angle(theta):
    """
    Normalize angle to [-pi, pi]
    """
    return (theta + math.pi) % (2 * math.pi) - math.pi

class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """
        Initialize robot at position (x, y) with orientation theta (in radians)
        """
        self.x = x
        self.y = y
        self.theta = theta  # Orientation angle

    def get_pose(self):
        """
        Return the current pose as a tuple: (x, y, theta)
        """
        return self.x, self.y, self.theta

    def update(self, v, omega, dt):
        """
        Update the robot's pose based on given linear velocity v,
        angular velocity omega, and time step dt.
        """
        # Unicycle model kinematics
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt

        # Keep theta in range [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

    def set_pose(self, pose):
        self.x, self.y, self.theta = pose

    # def normalize_angle(theta):
    #     return (theta + math.pi) % (2 * math.pi) - math.pi


