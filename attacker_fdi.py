# attacker_fdi.py
import numpy as np
import math
from collections import deque

def _wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

class FDIInjector:
    """
    Corrupts the pose seen by MAIN only.
    Modes: 'bias', 'drift', 'spike', 'replay'
    """
    def __init__(self, mode='bias'):
        self.mode = mode
        self.enabled = False
        self.bias_pos = np.array([0.10, -0.08], float)   # 10 cm, -8 cm
        self.bias_th  = math.radians(5.0)
        self.drift_rate = np.array([0.0007, -0.0005], float)  # m/s equivalent over time
        self.drift_rate_th = math.radians(0.18/60.0)          # rad/s equivalent
        self.spike_prob = 0.05
        self.spike_pos_mag = 0.15
        self.spike_th_mag  = math.radians(10.0)
        self.replay_buf = deque(maxlen=240)  # ~24s if dt=0.1
        self.replay_delay_steps = 30         # 3s delay at dt=0.1
        self.t = 0.0

    def set_mode(self, mode: str):
        assert mode in ('bias','drift','spike','replay')
        self.mode = mode

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)

    def corrupt(self, x_true: np.ndarray, dt: float) -> np.ndarray:
        """
        x_true = [x,y,theta]; returns what MAIN 'sees'
        """
        self.t += dt
        x, y, th = float(x_true[0]), float(x_true[1]), float(x_true[2])

        # maintain replay buffer of truth
        self.replay_buf.append(np.array([x,y,th], float))

        if not self.enabled:
            return np.array([x,y,th], float)

        if self.mode == 'bias':
            x_c = x + self.bias_pos[0]
            y_c = y + self.bias_pos[1]
            th_c = _wrap(th + self.bias_th)
            return np.array([x_c,y_c,th_c], float)

        if self.mode == 'drift':
            drift_xy = self.drift_rate * self.t
            drift_th = self.drift_rate_th * self.t
            return np.array([x + drift_xy[0], y + drift_xy[1], _wrap(th + drift_th)], float)

        if self.mode == 'spike':
            dx, dy, dth = 0.0, 0.0, 0.0
            if np.random.rand() < self.spike_prob:
                ang = 2*np.pi*np.random.rand()
                r   = self.spike_pos_mag
                dx  = r*np.cos(ang); dy = r*np.sin(ang)
                dth = (2*np.random.rand()-1)*self.spike_th_mag
            return np.array([x+dx, y+dy, _wrap(th+dth)], float)

        if self.mode == 'replay':
            if len(self.replay_buf) > self.replay_delay_steps:
                return self.replay_buf[-self.replay_delay_steps].copy()
            else:
                return np.array([x,y,th], float)
