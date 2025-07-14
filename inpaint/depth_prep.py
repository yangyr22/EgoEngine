import numpy as np
import cv2
import json
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def recover_metric_depth(
    depth_rel,        # H×W float array of relative depth
    pts2d,            # N×2 array of (u,v) pixel coords
    pts3d,            # N×3 array of corresponding cam‐frame points
    mode="affine",     # "scale" (b=0) or "affine" (solve s,b)
    mask_idx = None
):
    """
    Returns depth_metric = s*depth_rel + b, where s,b are fit
    so that depth_metric[u_i,v_i] ≈ pts3d[i,2] for all i.
    
    depth_rel:    H×W numpy float32
    pts2d:        (N,2) float or int pixel coordinates [u,v]
    pts3d:        (N,3) float camera‐frame X,Y,Z
    mode:         "scale" or "affine"
    """
    # 1) sample the relative depth at your 2D points
    #    use bilinear interpolation for subpixel accuracy:
    uvs = np.array(pts2d, dtype=np.float32)
    # 2) get your true depths from the 3D points
    z_true = pts3d[:,2].astype(np.float32)  
    if mask_idx is not None:
        uvs = uvs[mask_idx]
        z_true = z_true[mask_idx]
    sampled = cv2.remap(
        depth_rel.astype(np.float32),
        uvs[:,0], uvs[:,1],
        interpolation=cv2.INTER_LINEAR
    ).ravel()  # shape (N,)
    A = np.stack([sampled, np.ones_like(sampled)], axis=1)  # N×2
    sol, *_ = np.linalg.lstsq(A, z_true, rcond=None)
    s, b = sol[0], sol[1]
    
    return s, b

def main():
    p = argparse.ArgumentParser(
        description="Convert relative‐depth PNGs to metric depth using 2‐point calibration"
    )
    p.add_argument(
        "-b", "--base_dir", required=True,
        help="Base directory containing uv.json, all_pose.json, depth_raw_png/"
    )
    args = p.parse_args()
    base_dir = args.base_dir
    pose_path = os.path.join(base_dir, "all_pose.json")
    depth_dir = os.path.join(base_dir, "depth_raw_png")
    out_dir   = os.path.join(base_dir, "depth_prep_png")

    pose_data = json.load(open(pose_path))
    os.makedirs(out_dir, exist_ok=True)
    depth_files = sorted(os.listdir(depth_dir))
    
    first_mm = cv2.imread(os.path.join(depth_dir, depth_files[0]), cv2.IMREAD_UNCHANGED)
    H, W = first_mm.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(os.path.join(base_dir, "absolute_depth.mp4"), fourcc, 30, (W, H), isColor=True)

    prev_s = None
    prev_b = None
    max_ds = 0.05    # maximum allowed change per frame in scale
    max_db = 0.05
    real_sb = []
    clipped_sb = []
    
    for i, fname in enumerate(tqdm(depth_files, desc="Processing frames", unit="frame")):
        entry_p   = pose_data["frames"][i]
        fi        = int(fname.split(".")[0])    # just for logging
        depth_rel = cv2.imread(depth_dir + f"/{fname}", cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0

        pts2d = np.concatenate([
            np.array(entry_p["right_uv"]),
            np.array(entry_p["left_uv"])
        ])
        pts3d = np.concatenate([np.array(entry_p["right_camera"]), np.array(entry_p["left_camera"])])

        scale, offset = recover_metric_depth(
            depth_rel, pts2d, pts3d, mode="affine", mask_idx = [5, 20, 26, 41]
        )
        real_sb.append([scale, offset])
        if prev_s is not None:
            scale = float(np.clip(scale, prev_s - max_ds, prev_s + max_ds))
            offset = float(np.clip(offset, prev_b - max_db, prev_b + max_db))
        clipped_sb.append([scale,offset])
        prev_s = scale  
        prev_b = offset
        
        depth_metric = scale * depth_rel + offset

        depth_mm = np.clip(depth_metric * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(f"temp_data/result_Trial/depth_prep_png/{fi:06d}.png", depth_mm)

        depth_u8 = np.clip(depth_metric * 200.0, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
        vw.write(frame)

    vw.release()
    # Plotting (s, b) as points in 2D
    real_sb = np.array(real_sb)
    clipped_sb = np.array(clipped_sb)
    plt.figure(figsize=(8, 6))
    plt.scatter(real_sb[:, 0], real_sb[:, 1], label='Real (s, b)', alpha=0.7)
    plt.scatter(clipped_sb[:, 0], clipped_sb[:, 1], label='Clipped (s, b)', alpha=0.7, marker='x')
    plt.xlabel("Scale (s)")
    plt.ylabel("Offset (b)")
    plt.title("(s, b) Scatter Plot: Real vs Clipped")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sb_scatter_plot.png")
        
if __name__ == "__main__":
    main()
