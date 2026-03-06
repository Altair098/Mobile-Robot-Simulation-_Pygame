# paths/make_circle.py
import numpy as np, os, math
os.makedirs("paths", exist_ok=True)

R = 1.2          # meters
N = 400          # samples
cx, cy = 0.0, 0.0
th = np.linspace(math.pi/2, math.pi/2 - 2*math.pi, N)  # start at top, counter-clockwise
xy = np.c_[cx + R*np.cos(th), cy + R*np.sin(th)]
np.save("paths/circle.npy", xy)
print("saved paths/circle.npy", xy.shape)
