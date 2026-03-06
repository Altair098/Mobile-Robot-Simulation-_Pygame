# fdi_detector.py
import numpy as np
import math
def _wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

class ResidualDetector:
    """
    Compares MAIN-reported pose vs DT mirror x_hat.
    Declares attack if residuals exceed thresholds for N_consec ticks.
    """
    def __init__(self, pos_thresh=0.12, th_thresh=math.radians(8.0), n_consec=6, n_clear=10):
        self.pos_thresh = float(pos_thresh)
        self.th_thresh  = float(th_thresh)
        self.n_consec   = int(n_consec)
        self.n_clear    = int(n_clear)
        self.bad_count  = 0
        self.good_count = 0
        self.attacked   = False

    def update(self, x_main_meas, x_dt_hat):
        dx = float(x_main_meas[0] - x_dt_hat[0])
        dy = float(x_main_meas[1] - x_dt_hat[1])
        dpos = math.hypot(dx, dy)
        dth  = abs(_wrap(float(x_main_meas[2] - x_dt_hat[2])))

        bad = (dpos > self.pos_thresh) or (dth > self.th_thresh)
        if bad:
            self.bad_count += 1
            self.good_count = 0
        else:
            self.good_count += 1
            self.bad_count = 0

        if not self.attacked and self.bad_count >= self.n_consec:
            self.attacked = True
        elif self.attacked and self.good_count >= self.n_clear:
            self.attacked = False

        return self.attacked, dpos, dth
