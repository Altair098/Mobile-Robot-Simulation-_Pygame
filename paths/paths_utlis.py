# paths_util.py
import os, glob, math
import numpy as np

def ensure_circle(paths_dir="paths", R=1.2, N=400):
    os.makedirs(paths_dir, exist_ok=True)
    circle_path = os.path.join(paths_dir, "circle.npy")
    if not os.path.exists(circle_path):
        th = np.linspace(math.pi/2, math.pi/2 - 2*math.pi, N)
        xy = np.c_[R*np.cos(th), R*np.sin(th)]
        np.save(circle_path, xy)
    return circle_path

def discover_paths(paths_dir="paths"):
    """
    Returns: list of (name, ndarray[N,2]) including:
      - builtin placeholders: 'figure8', 'sCurve', 'sharpL'  (return None for waypoints; your batch should generate them procedurally)
      - 'circle' from paths/circle.npy
      - all freehand*.npy from paths/
    """
    out = []

    # built-ins: let your batch code generate waypoints for these names
    out.append(("figure8", None))
    out.append(("sCurve",  None))
    out.append(("sharpL",  None))

    # circle
    circle_file = ensure_circle(paths_dir)
    out.append(("circle", np.load(circle_file)))

    # all freehand
    for f in sorted(glob.glob(os.path.join(paths_dir, "freehand*.npy"))):
        nm = os.path.splitext(os.path.basename(f))[0]
        try:
            arr = np.load(f)
            if isinstance(arr, np.ndarray) and arr.ndim==2 and arr.shape[1]==2:
                out.append((nm, arr))
        except Exception:
            pass
    return out
