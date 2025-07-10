#!/usr/bin/env python3
import sys
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
import torch
from typing import List, Optional
import tempfile


# SAM2 imports
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Hydra for config loading
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

# Utility functions
from utils import load_img_to_array, save_array_to_img, dilate_mask


def predict_masks_with_sam2(
    img: np.ndarray,
    point_coords: List[List[float]],
    point_labels: List[int],
    sam_cfg: str,
    sam_ckpt: str,
    device: str = "cuda"
):
    """
    Run SAM2 inference on a single image with point prompts.
    """
    model = build_sam2(sam_cfg, sam_ckpt)
    predictor = SAM2ImagePredictor(model, device=device)

    np_coords = np.array(point_coords, dtype=np.float32)
    np_labels = np.array(point_labels, dtype=np.int32)

    predictor.set_image(img)
    try:
        masks, scores, logits = predictor.predict(
            point_coords=np_coords,
            point_labels=np_labels,
            box=None,
            multimask_output=True,
            return_logits=True
        )
    except TypeError:
        masks, scores = predictor.predict(
            point_coords=np_coords,
            point_labels=np_labels,
            box=None,
            multimask_output=True
        )
        logits = None

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    masks = (masks > 0).astype(np.uint8)
    return masks, scores, logits


def build_sam2_model(
    sam_cfg: str,
    sam_ckpt: str,
    device: str = "cuda"
):
    """
    Build and return a SAM2 image predictor via Hydra.
    """
    cfg = Path(sam_cfg)
    config_dir = str(cfg.parent)
    config_name = cfg.stem

    GlobalHydra.instance().clear()
    initialize(config_path=config_dir, job_name="sam2_image_inference", version_base=None)
    model = build_sam2(config_name, sam_ckpt)
    GlobalHydra.instance().clear()

    return SAM2ImagePredictor(model, device=device)


def build_sam2_model_video(
    sam_cfg: str,
    sam_ckpt: str,
    device: str = "cuda",
    vos_optimized: bool = False
):
    """
    Build and return a SAM2 video predictor via Hydra.
    """
    cfg = Path(sam_cfg)
    config_dir = str(cfg.parent)
    config_name = cfg.name

    GlobalHydra.instance().clear()
    initialize(config_path=config_dir, job_name="sam2_video_inference", version_base=None)
    predictor = build_sam2_video_predictor(
        config_name,
        sam_ckpt,
        device=device,
        vos_optimized=vos_optimized
    )
    GlobalHydra.instance().clear()

    try:
        predictor = predictor.to(device)
    except Exception:
        pass
    return predictor


def predict_masks_with_sam2_video(
    video_path: str,
    predictor,
    frame_coords_left: List[List[List[float]]],
    frame_coords_right: List[List[List[float]]],
    dilate_kernel_size: Optional[int] = None,
    propagate: bool = True,
    save_seg: bool = False,
    output_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Run SAM2 video inference with per-frame point prompts for two arms.
    Returns a numpy array of shape (T, 2, H, W), with values 0/1:
      [:,0,:,:] = left‐arm masks, [:,1,:,:] = right‐arm masks.
    If save_seg=True, also writes out mask_0.png and mask_1.png under
    output_dir/<video_name>/frame_000000/…
    """
    # 1) extract frames to a temp folder (or reuse output_dir if saving)
    if save_seg and output_dir is not None:
        base_out = Path(output_dir) / Path(video_path).stem
        base_out.mkdir(parents=True, exist_ok=True)
        frames_dir = base_out / "frames"
        frames_dir.mkdir(exist_ok=True)
    else:
        frames_dir = Path(tempfile.mkdtemp())

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()

    T = idx
    # read one to get H,W
    sample = cv2.imread(str(frames_dir / "000000.jpg"), cv2.IMREAD_GRAYSCALE)
    H, W = sample.shape[:2]

    # 2) init SAM2‐video state
    state = predictor.init_state(str(frames_dir))

    # 3) prepare storage
    left_masks  = np.zeros((T, H, W), dtype=np.uint8)
    right_masks = np.zeros((T, H, W), dtype=np.uint8)

    # 4) per‐frame prompts
    for i in range(T):
        # mark CUDA graph step
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except AttributeError:
            pass
        dev = predictor.device if isinstance(predictor.device, str) else predictor.device

        # LEFT arm → object_id=0
        pts_left = frame_coords_left[i] if i < len(frame_coords_left) else []
        if pts_left:
            pts  = torch.tensor(pts_left, dtype=torch.float32, device=dev)
            labs = torch.tensor([1] * len(pts_left), dtype=torch.int32, device=dev)
            _, object_ids, masks = predictor.add_new_points_or_box(
                state, i, 0, points=pts, labels=labs
            )
            m = masks[object_ids.index(0)]  # find mask for object 0
            m = m.squeeze().cpu().numpy() > 0
            m = m.astype(np.uint8)
            if dilate_kernel_size:
                m = dilate_mask(m, dilate_kernel_size)
            left_masks[i] = m
            if save_seg:
                frame_out = frames_dir.parent / f"frame_{i:06d}"
                frame_out.mkdir(parents=True, exist_ok=True)
                save_array_to_img(m * 255, frame_out / "mask_0.png")


        # RIGHT arm → object_id=1
        pts_right = frame_coords_right[i] if i < len(frame_coords_right) else []
        if pts_right:
            pts  = torch.tensor(pts_right, dtype=torch.float32, device=dev)
            labs = torch.tensor([1] * len(pts_right), dtype=torch.int32, device=dev)
            _, object_ids, masks = predictor.add_new_points_or_box(
                state, i, 1, points=pts, labels=labs
            )
            m = masks[object_ids.index(1)]
            m = m.squeeze().cpu().numpy() > 0
            m = m.astype(np.uint8)
            if dilate_kernel_size:
                m = dilate_mask(m, dilate_kernel_size)
            right_masks[i] = m
            if save_seg:
                frame_out = frames_dir.parent / f"frame_{i:06d}"
                frame_out.mkdir(parents=True, exist_ok=True)
                save_array_to_img(m * 255, frame_out / "mask_1.png")


    # 5) optional propagation (fills in missing frames if needed)
    if propagate:
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            m_arr = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
            for mid, obj in enumerate(object_ids):
                m = m_arr[mid].squeeze() > 0
                m = m.astype(np.uint8)
                if dilate_kernel_size:
                    m = dilate_mask(m, dilate_kernel_size)
                if obj == 0:
                    left_masks[frame_idx] = m
                elif obj == 1:
                    right_masks[frame_idx] = m

    # 6) stack and return
    return np.stack([left_masks, right_masks], axis=1)



def _save_masks(
    frame_idx: int,
    object_ids: List[int],
    masks,
    base_out: Path,
    dilate_kernel_size: Optional[int]
):
    """
    Internal helper: save masks for a single frame under frame_{idx:06d}.
    Applies dilation only when mask is 2D and non-empty.
    """
    frame_dir = base_out / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(exist_ok=True)

    # Convert to numpy if necessary
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    for mid, obj in enumerate(object_ids):
        mask = masks[mid]
        # Squeeze any extra dimensions to get a 2D mask
        while mask.ndim > 2:
            mask = mask.squeeze()
        # Binarize and scale to uint8
        mask2d = (mask > 0).astype(np.uint8) * 255
        # Only dilate if configured and mask is 2D
        if dilate_kernel_size and mask2d.ndim == 2 and mask2d.size > 0:
            mask2d = dilate_mask(mask2d, dilate_kernel_size)
        save_array_to_img(mask2d, frame_dir / f"mask_{obj}.png")

def assemble_dual_overlay_video(
    masks_dir: Path,
    frames_dir: Path,
    fps: int,
    output_path: Path,
    alpha: float = 0.7,
    beta: float = 0.3,
):
    """
    Assemble a color video overlaying both mask_0 (red) and mask_1 (green)
    on the original frames in `frames_dir`.
    """
    # 1) 找出所有包含两种 mask 的帧文件夹
    frame_dirs = sorted(
        d for d in masks_dir.iterdir()
        if d.is_dir() and (d / "mask_0.png").exists() and (d / "mask_1.png").exists()
    )
    if not frame_dirs:
        raise RuntimeError(f"No mask_0.png/mask_1.png found under {masks_dir}")

    # 2) 从第一帧中读尺寸
    first_idx   = int(frame_dirs[0].name.split("_")[-1])
    first_frame = cv2.imread(str(frames_dir / f"{first_idx:06d}.jpg"))
    if first_frame is None:
        raise RuntimeError(f"Could not load frame {first_idx:06d}.jpg")
    h, w = first_frame.shape[:2]

    # 3) 建 VideoWriter（彩色视频）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), True)

    # 4) 遍历每一帧：先叠加左臂（红色），再叠加右臂（绿色）
    for d in frame_dirs:
        idx = int(d.name.split("_")[-1])
        orig_path = frames_dir / f"{idx:06d}.jpg"
        frame     = cv2.imread(str(orig_path))
        if frame is None:
            print(f"Warning: missing original frame {orig_path}, skipping")
            continue

        m0 = cv2.imread(str(d / "mask_0.png"), cv2.IMREAD_GRAYSCALE)
        m1 = cv2.imread(str(d / "mask_1.png"), cv2.IMREAD_GRAYSCALE)
        if m0 is None or m1 is None:
            print(f"Warning: missing mask0/mask1 in {d}, skipping")
            continue

        colored0 = np.zeros_like(frame)
        colored0[:, :, 2] = m0

        colored1 = np.zeros_like(frame)
        colored1[:, :, 1] = m1

        over = cv2.addWeighted(frame, alpha, colored0, beta, 0)
        over = cv2.addWeighted(over, alpha, colored1, beta, 0)

        writer.write(over)

    writer.release()
    print(f"Dual-overlay video assembled at {output_path}")


def setup_args(parser):
    parser.add_argument("--input_img", type=str)
    parser.add_argument("--input_video", type=str)
    parser.add_argument("--uv", type=str)
    parser.add_argument("--mask_idx", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--point_coords", type=float, nargs='+')
    parser.add_argument("--point_labels", type=int, nargs='+')
    parser.add_argument("--dilate_kernel_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sam_cfg", type=str, required=True)
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--vos_optimized", action="store_true")
    parser.add_argument("--propagate", action="store_true")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.input_img:
        pass

    elif args.input_video:
        if not args.uv:
            raise ValueError("--uv required for video mode.")

        # 1) Parse UV file into left/right per-frame lists
        frame_coords_left, frame_coords_right = [], []
        with open(args.uv) as f:
            for line in f:
                uL, vL, uR, vR = [
                    float(x) if x.lower() != "nan" else np.nan
                    for x in line.strip().split(',')
                ]
                frame_coords_left.append(
                    [] if np.isnan(uL) or np.isnan(vL) else [[uL, vL]]
                )
                frame_coords_right.append(
                    [] if np.isnan(uR) or np.isnan(vR) else [[uR, vR]]
                )

        # 2) Build the video predictor
        predictor = build_sam2_model_video(
            args.sam_cfg,
            args.sam_ckpt,
            device=device,
            vos_optimized=args.vos_optimized
        )

        # 3) Run per-frame prompts (left=obj0, right=obj1)
        masks_dir = predict_masks_with_sam2_video(
            args.input_video,
            predictor,
            frame_coords_left,
            frame_coords_right,
            output_dir =args.output_dir,
            save_seg=True,
            dilate_kernel_size=args.dilate_kernel_size,
            propagate=args.propagate
        )

        # 4) Overlay and save
        masks_dir = Path(masks_dir)
        frames_dir = masks_dir / "frames"
        out_path = masks_dir / "mask_left_right.mp4"
        assemble_dual_overlay_video(
        masks_dir=masks_dir,
        frames_dir=frames_dir,
        fps=args.fps,
        output_path=out_path
        )
        print(f"Mask video saved to {out_path}")

    else:
        parser.error("Provide --input_img or --input_video.")

'''
python /coc/flash7/yliu3735/workspace/inpaint/sam2_segment_vid.py   --input_video /coc/flash7/yliu3735/workspace/inpaint/cropped_video.mp4   --uv /coc/flash7/yliu3735/workspace/inpaint/uv.txt   --mask_idx 0   --fps 25   --dilate_kernel_size 10   --output_dir ./results   --sam_ckpt weights/sam2.1_hiera_base_plus.pt   --sam_cfg segment_anything2/sam2/configs/sam2.1/sam2.1_hiera_b.yaml   --propagate
'''