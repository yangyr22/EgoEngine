#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_translations(json_path):
    """
    Load the list of 3‐vectors t from each “ob_in_cam” 4×4 matrix.
    """
    data    = json.load(open(json_path))
    ts      = []
    for entry in data:
        T = np.array(entry["ob_in_cam"], dtype=float)
        t = T[:3, 3]
        ts.append(t)
    return np.stack(ts, axis=0)  # shape (N,3)

def plot_translations(ts, save_path="translations.png"):
    """
    Scatter all translations in 3D and save to disk.
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    xs, ys, zs = ts[:,0], ts[:,1], ts[:,2]
    ax.scatter(xs, ys, zs, c='k', s=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Object Translations in Camera Frame")

    # equal aspect
    mins = ts.min(axis=0)
    maxs = ts.max(axis=0)
    mid  = (mins + maxs) / 2
    span = (maxs - mins).max() / 2
    ax.set_xlim(mid[0]-span, mid[0]+span)
    ax.set_ylim(mid[1]-span, mid[1]+span)
    ax.set_zlim(mid[2]-span, mid[2]+span)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved translation scatter to {save_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot object translations as points")
    p.add_argument("pose_json",
                   help="Path to your ob_in_cam_all.json")
    p.add_argument("--out", default="translations.png",
                   help="Filename for the output image")
    args = p.parse_args()

    ts = load_translations(args.pose_json)
    plot_translations(ts, save_path=args.out)
