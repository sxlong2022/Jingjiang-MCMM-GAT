"""
Convert monthly centerline .npy files to MCMM .xy format.
MCMM requires dimensionless coordinates, normalized by channel width W as the characteristic length.
"""
import argparse
import numpy as np
from pathlib import Path

# Configuration
NPY_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "monthly_centerlines"
OUTPUT_DIR = Path(__file__).parent
INITIAL_MONTH = "201601"  # Initial centerline month
W_REF = 880.0  # Reference channel width (m) for non-dimensionalization
TARGET_DELTAS = 1.0

parser = argparse.ArgumentParser()
parser.add_argument("--initial_month", default=INITIAL_MONTH)
parser.add_argument("--w_ref", type=float, default=W_REF)
parser.add_argument("--target_deltas", type=float, default=TARGET_DELTAS)
parser.add_argument("--n_out", type=int, default=None)
parser.add_argument("--npy_dir", default=None)
parser.add_argument("--output", default="jingjiang_centerline.xy")
args = parser.parse_args()

INITIAL_MONTH = str(args.initial_month)
W_REF = float(args.w_ref)
TARGET_DELTAS = float(args.target_deltas)
if args.npy_dir:
    NPY_DIR = Path(args.npy_dir)

out_path = Path(args.output)
if not out_path.is_absolute():
    out_path = OUTPUT_DIR / out_path

# Load centerline
npy_dirs = [NPY_DIR]
if not args.npy_dir:
    pass

npy_file = None
for d in npy_dirs:
    cand = Path(d) / f"{INITIAL_MONTH}_centerline_geo_1000pts.npy"
    if cand.exists():
        npy_file = cand
        break
if npy_file is None:
    tried = "\n".join([str(Path(d) / f"{INITIAL_MONTH}_centerline_geo_1000pts.npy") for d in npy_dirs])
    raise FileNotFoundError(f"Cannot find centerline npy for initial_month={INITIAL_MONTH}. Tried:\n{tried}")

coords = np.load(npy_file)  # shape: (1000, 2), UTM coordinates (m)

coords = np.asarray(coords, dtype=float)
if coords.ndim != 2 or coords.shape[1] != 2:
    raise ValueError(f"Invalid centerline array shape: {coords.shape}")

if len(coords) < 2:
    raise ValueError("Centerline must contain at least 2 points")

seg_len_m = np.sqrt(np.sum((coords[1:] - coords[:-1]) ** 2, axis=1))
keep_mask = np.concatenate(([True], seg_len_m > 1e-6))
coords = coords[keep_mask]

print(f"Loaded: {npy_file}")
print(f"Original coordinate range:")
print(f"  X: {coords[:,0].min():.1f} ~ {coords[:,0].max():.1f} m")
print(f"  Y: {coords[:,1].min():.1f} ~ {coords[:,1].max():.1f} m")

# Translate to near origin
x = coords[:, 0] - coords[0, 0]
y = coords[:, 1] - coords[0, 1]

# Non-dimensionalize (divide by reference width)
x_nd = x / W_REF
y_nd = y / W_REF

dx = np.diff(x_nd)
dy = np.diff(y_nd)
seg_len = np.sqrt(dx**2 + dy**2)
s = np.concatenate(([0.0], np.cumsum(seg_len)))
total_len = float(s[-1])

if not np.isfinite(total_len) or total_len <= 0:
    raise ValueError(f"Invalid total centerline length: {total_len}")

n_out = int(args.n_out) if args.n_out is not None else (int(round(total_len / TARGET_DELTAS)) + 1)
if n_out < 2:
    n_out = 2
target_s = np.linspace(0.0, total_len, n_out)

x_nd = np.interp(target_s, s, x_nd)
y_nd = np.interp(target_s, s, y_nd)

print(f"\nAfter non-dimensionalization (W_ref={W_REF}m):")
print(f"  x: {x_nd.min():.2f} ~ {x_nd.max():.2f}")
print(f"  y: {y_nd.min():.2f} ~ {y_nd.max():.2f}")
print(f"  Channel length: {total_len:.1f} W")

# Save in .xy format
output_file = out_path
np.savetxt(output_file, np.column_stack([x_nd, y_nd]), fmt='%.6f', delimiter='\t')
print(f"\nSaved: {output_file}")
print(f"Points: {len(x_nd)}")
