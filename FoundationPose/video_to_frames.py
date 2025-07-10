#!/usr/bin/env python3
import os
import cv2
import numpy as np
from matplotlib import cm
from tqdm import tqdm

# --- CONFIGURE THESE ---
orig_mp4   = "/coc/flash7/yliu3735/workspace/inpaint/temp_data/result_Trial/cropped_video.mp4"
depth_mp4  = "/coc/flash7/yliu3735/workspace/inpaint/temp_data/result_Trial/cropped_video_depth.mp4"
out_dir    = "demo_data/test"
dmin, dmax = 0.1, 700.0        # the actual min/max depth (in meters) you used
cmap_name  = "Spectral_r"
# ------------------------

# prepare folders
color_dir = os.path.join(out_dir, "rgb")
depth_dir = os.path.join(out_dir, "depth")
mask_dir  = os.path.join(out_dir, "mask")
os.makedirs(color_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(mask_dir,  exist_ok=True)

# build LUT for inverting the colormap
cmap = cm.get_cmap(cmap_name, 256)
lut  = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)  # RGB table

def invert_colormap(frame_bgr):
    """Given a BGR color‐mapped depth, return a float32 depth map in meters."""
    rgb = frame_bgr[..., ::-1]  # BGR→RGB
    flat = rgb.reshape(-1,3).astype(int)
    # compute squared color‐distance to every LUT entry
    diffs = flat[:, None, :] - lut[None, :, :]
    idx   = np.argmin((diffs**2).sum(-1), axis=1)
    depth = (idx / 255.0) * (dmax - dmin) + dmin
    return depth.reshape(frame_bgr.shape[:2]).astype(np.float32)

def dump_frames(video_path, out_folder, invert_depth=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count), desc=f"Unpacking {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(out_folder, f"{i:06d}.png")
        if invert_depth:
            # depth_m = invert_colormap(frame)
            # save as 16‐bit PNG scaled by 1000 (mm), or .exr if you prefer floats:
            cv2.imwrite(filename, (frame / 1000).astype(np.uint16))
        else:
            # RGB frames just save directly
            cv2.imwrite(filename, frame)
    cap.release()

if __name__ == "__main__":
    # 1) dump your original RGB
    # dump_frames(orig_mp4, color_dir, invert_depth=False)
    # 2) dump/invert your color‐mapped depth
    dump_frames(depth_mp4, depth_dir, invert_depth=True)

    # 3) create a dummy mask (all‐valid) if you don’t have one
    #    otherwise replace this with your actual mask extraction
    for imgname in os.listdir(color_dir):
        mask_path = os.path.join(mask_dir, imgname)
        # white mask = entire image valid
        h, w = cv2.imread(os.path.join(color_dir, imgname)).shape[:2]
        cv2.imwrite(mask_path, 255 * np.ones((h, w), dtype=np.uint8))

    print("Done! Directory structure:")
    print(f"  {color_dir}/  ← RGB frames")
    print(f"  {depth_dir}/  ← depth frames in mm‐encoded PNG")
    print(f"  {mask_dir}/   ← per‐frame masks (all‐white dummy)")
    print("\nNow point your demo at `--test_scene_dir test_scene_dir` and you’re set.")
