# detector.py
import numpy as np
import math

def angwrap(a): return (a + math.pi) % (2*math.pi) - math.pi

class PoseResidualDetector:
    """
    Compare the *reported* main pose z_t with DT's kinematic mirror x_hat_t.
    r_t = ||[dx, dy, dθ*wθ]||. Raise alarm if a smoothed residual > threshold.
    """
    def __init__(self, w_theta=0.5, alpha=0.2, thresh=0.35, min_separation=0.8):
        self.w_theta = float(w_theta)      # radians → meters weight
        self.alpha   = float(alpha)        # EMA smoothing
        self.thresh  = float(thresh)       # decision threshold (swept in ROC)
        self.min_sep = float(min_separation) # s between alarms (debounce)
        self.r_ema = 0.0
        self.last_alarm_t = -1e9

    def residual(self, x_hat, z):
        dx = z[0] - x_hat[0]
        dy = z[1] - x_hat[1]
        dth = angwrap(z[2] - x_hat[2])
        return math.sqrt(dx*dx + dy*dy + (self.w_theta*dth)**2)

    def step(self, t, x_hat, z):
        r = self.residual(x_hat, z)
        self.r_ema = (1 - self.alpha)*self.r_ema + self.alpha*r
        alarm = (self.r_ema > self.thresh) and (t - self.last_alarm_t > self.min_sep)
        if alarm: self.last_alarm_t = t
        return r, self.r_ema, alarm
