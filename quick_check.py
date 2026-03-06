import csv, math, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results_batch_foo/20250908_094330/figure8/seed_xxx/log.csv"

with open(path, "r") as f:
    rows = list(csv.DictReader(f))

x  = np.array([float(r["x"]) for r in rows])
y  = np.array([float(r["y"]) for r in rows])
xd = np.array([float(r["xd"]) for r in rows])
yd = np.array([float(r["yd"]) for r in rows])

e = np.sqrt((x-xd)**2 + (y-yd)**2)

print("Samples:", len(e))
print("Mean error:", np.mean(e))
print("Max error:", np.max(e))
print("90th percentile:", np.percentile(e, 90))
