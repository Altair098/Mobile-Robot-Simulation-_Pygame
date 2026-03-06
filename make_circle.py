import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Define control points for the S shape
# These points create the characteristic S curve
points = np.array([
    [0.5, 1.0],    # Top end
    [0.3, 0.95],   # Top left curve
    [0.2, 0.85],   
    [0.2, 0.75],   
    [0.3, 0.65],   # Upper middle curve
    [0.5, 0.6],    # Middle crossing
    [0.7, 0.55],   
    [0.8, 0.45],   # Lower middle curve
    [0.8, 0.35],   
    [0.7, 0.25],   
    [0.5, 0.2],    
    [0.3, 0.15],   # Bottom curve
    [0.5, 0.0]     # Bottom end
])

# Create smooth spline through the points
tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=3)
u_new = np.linspace(0, 1, 1000)
x_smooth, y_smooth = splev(u_new, tck)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.plot(x_smooth, y_smooth, 'b-', linewidth=8, solid_capstyle='round')
ax.set_aspect('equal')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.axis('off')
ax.set_title('Letter S', fontsize=20, pad=20)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Define control points for the L shape
# These points create a vertical line down and horizontal line right
points = np.array([
    [0.2, 1.0],    # Top of vertical stroke
    [0.2, 0.8],
    [0.2, 0.6],
    [0.2, 0.4],
    [0.2, 0.2],
    [0.2, 0.05],   # Just before corner
    [0.2, 0.02],   # Right at corner
    [0.23, 0.0],   # Sharp turn
    [0.3, 0.0],    # Start of horizontal stroke
    [0.5, 0.0],
    [0.7, 0.0],
    [0.9, 0.0]     # End of horizontal stroke
])

# Create smooth spline through the points with tighter curve
tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=1)
u_new = np.linspace(0, 1, 1000)
x_smooth, y_smooth = splev(u_new, tck)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.plot(x_smooth, y_smooth, 'b-', linewidth=8, solid_capstyle='round')
ax.set_aspect('equal')
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.1, 1.1)
ax.axis('off')
ax.set_title('Letter L', fontsize=20, pad=20)

plt.tight_layout()
plt.show()