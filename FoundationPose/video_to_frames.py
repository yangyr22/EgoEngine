#!/usr/bin/env python3
import os
import cv2
import numpy as np
from matplotlib import cm
from tqdm import tqdm

# --- CONFIGURE THESE ---
orig_mp4   = "/coc/flash7/yliu3735/workspace/inpaint/temp_data/result_Trial/cropped_video.mp4"
out_dir    = "demo_data/test"
# ------------------------

# prepare folders
color_dir = os.path.join(out_dir, "rgb")
mask_dir  = os.path.join(out_dir, "mask")
os.makedirs(color_dir, exist_ok=True)
os.makedirs(mask_dir,  exist_ok=True)

def dump_frames(video_path, out_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count), desc=f"Unpacking {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(out_folder, f"{i:06d}.png")
        cv2.imwrite(filename, frame)
    cap.release()

if __name__ == "__main__":
    # 1) dump your original RGB
    dump_frames(orig_mp4, color_dir, invert_depth=False)

    print("Done! Directory structure:")
    print(f" {color_dir}/  ← RGB frames")
