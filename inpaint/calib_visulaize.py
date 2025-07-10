#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

def load_poses(path):
    with open(path, 'r') as f:
        data = json.load(f)
    frames = data.get("frames", [])
    poses = []
    for entry in frames:
        T = entry.get("T_world_camera_rebased", entry.get("T_world_camera"))
        if T is None:
            continue
        T = np.array(T)
        R = T[:3, :3]
        t = T[:3, 3]
        poses.append((R, t))
    return poses

def plot_trajectory_with_axes(poses, axis_length=0.05, sample_rate=50, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [t[0] for _, t in poses]
    ys = [t[1] for _, t in poses]
    zs = [t[2] for _, t in poses]
    ax.plot(xs, ys, zs, linewidth=1, label='Camera Path')
    ax.scatter(xs, ys, zs, s=10, c='k')

    for i, (R, t) in enumerate(poses):
        if i % sample_rate != 0:
            continue
        x, y, z = t
        # Draw camera axes directions
        ax.quiver(x, y, z, R[0,0], R[1,0], R[2,0], length=axis_length, color='r')
        ax.quiver(x, y, z, R[0,1], R[1,1], R[2,1], length=axis_length, color='g')
        ax.quiver(x, y, z, R[0,2], R[1,2], R[2,2], length=axis_length, color='b')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Camera Trajectory (axes every {sample_rate} frames)')
    ax.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def main():
    import argparse
    p = argparse.ArgumentParser(description="Plot camera trajectory with pose axes")
    p.add_argument("json_file", help="Path to your JSON file")
    p.add_argument("-o", "--out", help="Output image file (e.g. traj_pose.png)")
    p.add_argument("--axis_len", type=float, default=0.0005,
                   help="Length of orientation axes")
    p.add_argument("--sample_rate", type=int, default=50,
                   help="Only draw axes every N frames")
    args = p.parse_args()

    poses = load_poses(args.json_file)
    if not poses:
        print("No valid poses found in JSON.")
        return
    plot_trajectory_with_axes(poses,
                              axis_length=args.axis_len,
                              sample_rate=args.sample_rate,
                              save_path=args.out)

if __name__ == "__main__":
    main()

