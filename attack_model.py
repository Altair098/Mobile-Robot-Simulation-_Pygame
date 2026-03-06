# attack_models.py
import math
import numpy as np

def _wrap(a):  # [-pi, pi]
    return (a + math.pi) % (2*math.pi) - math.pi

class FDIInjector:
    """
    False-data injection on pose measurements:
      z = x_true + b(t) + noise
    You can switch between bias patterns below.
    """
    def __init__(self, kind="drift", amp=0.25, omega=0.35, noise_std=(0.00,0.00,0.00)):
        self.kind = kind
        self.amp = float(amp)      # meters or radians
        self.omega = float(omega)  # rad/s for sinusoidal
        self.noise_std = tuple(noise_std)  # (σx, σy, σθ)

    def corrupt(self, pose_true, t):
        x, y, th = map(float, pose_true)
        bx = by = bth = 0.0

        if self.kind == "drift":
            # linear drift in x,y and small yaw bias
            bx  = 0.15 * t
            by  = -0.05 * t
            bth = 0.02 * t
        elif self.kind == "sinus":
            bx  = self.amp * math.sin(self.omega * t)
            by  = 0.5*self.amp * math.cos(0.8*self.omega * t)
            bth = 0.3 * self.amp * math.sin(0.6*self.omega * t)
        elif self.kind == "step":
            # a step bias after 3 s
            if t > 3.0:
                bx, by, bth = 0.4, -0.2, 0.25

        nx = np.random.randn()*self.noise_std[0]
        ny = np.random.randn()*self.noise_std[1]
        nth= np.random.randn()*self.noise_std[2]

        x_m  = x + bx + nx
        y_m  = y + by + ny
        th_m = _wrap(th + bth + nth)
        return np.array([x_m, y_m, th_m], dtype=float)
