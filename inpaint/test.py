#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

def load_points(path):
    """
    Try loading as:
     - NumPy .npy → returns (N,3) array
     - CSV with three columns → returns (N,3) array
    """
    if path.endswith(".npy"):
        return np.load(path)
    else:
        # assume whitespace or comma-delimited with 3 columns
        return np.loadtxt(path, delimiter=None)

def plot_points(pts, color=None, size=20):
    """
    pts: (N,3) array of XYZ points
    color: single color or array of N colors
    size: marker size
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    X,Y,Z = pts[:,0], pts[:,1], pts[:,2]
    ax.scatter(X, Y, Z, c=color, s=size, depthshade=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud")
    # equal aspect so axes are not distorted
    max_range = np.ptp(pts, axis=0).max()
    mid = np.mean(pts, axis=0)
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim],
                       [(mid[0]-max_range/2, mid[0]+max_range/2),
                        (mid[1]-max_range/2, mid[1]+max_range/2),
                        (mid[2]-max_range/2, mid[2]+max_range/2)]):
        axis(m)
    for i, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, str(i), size=10, zorder=1, color='black')
    out_path = "test.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Visualize 3D points")
    p.add_argument("--color", default=None,
                   help="Optional: path to N×3 RGB (0–1) .npy or single color string")
    p.add_argument("--size", type=int, default=20,
                   help="Marker size")
    args = p.parse_args()

    pts = np.array([
        [
          0.0048382160859864395,
          -0.2872272634326261,
          0.47042710812589406
        ],
        [
          -0.08285250591766462,
          -0.35470333261817394,
          0.5042090749026459
        ],
        [
          -0.07182495046420902,
          -0.3833287256983218,
          0.5257232267732652
        ],
        [
          -0.050411376391814866,
          -0.4002346335936205,
          0.5280779463374317
        ],
        [
          -0.009913750584193793,
          -0.4179996716861221,
          0.521823726721215
        ],
        [
          0.04491200724140029,
          -0.38316809153945963,
          0.3778728754858506
        ],
        [
          0.01797586035040593,
          -0.328037388201692,
          0.4309398976171123
        ],
        [
          0.009222308923000532,
          -0.3056495225812515,
          0.4546908916169194
        ],
        [
          -0.027521178684621547,
          -0.3597387308006171,
          0.44133864056810207
        ],
        [
          -0.056066468666795294,
          -0.3596210095906685,
          0.4663111321454506
        ],
        [
          -0.070368449647878,
          -0.35647535748937104,
          0.485708453086105
        ],
        [
          -0.02137903994848972,
          -0.3805058792567551,
          0.4468372963590051
        ],
        [
          -0.050219563298995074,
          -0.3869712779456369,
          0.47796945868212004
        ],
        [
          -0.06173003468974489,
          -0.3850029433305991,
          0.5029217046505069
        ],
        [
          -0.006485710673700091,
          -0.3941820174046296,
          0.45170561334878184
        ],
        [
          -0.030011226486615222,
          -0.40289630279137834,
          0.4815599598379378
        ],
        [
          -0.041153980531306256,
          -0.40292747014579416,
          0.505684660254431
        ],
        [
          0.012339904162240027,
          -0.40474628941944707,
          0.45505763238449404
        ],
        [
          -0.0005353075578182244,
          -0.41558847518688175,
          0.48075597823005617
        ],
        [
          -0.00431658342848968,
          -0.4177683083684337,
          0.5005937908690666
        ],
        [
          -0.002139567012704161,
          -0.36823315325364486,
          0.44685830506564955
        ]], dtype=np.float32
    )
    if args.color:
        try:
            col = load_points(args.color)
        except:
            col = args.color
    else:
        col = None

    plot_points(pts, color=col, size=args.size)
