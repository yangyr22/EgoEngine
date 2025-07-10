import argparse
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM2 segmentation on only the first frame of a video and save mask/overlay")
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to the input video"
    )
    parser.add_argument(
        "--sam_cfg", type=str, required=True,
        help="Path to the SAM2 config YAML file"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="Path to the SAM2 checkpoint (.pt file)"
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="List of point prompt coordinates: x1 y1 [x2 y2 ...], even number of floats"
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="List of point prompt labels: length must match number of coordinate pairs; integers, non-zero is foreground (1), zero is background (0)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory to save masks and overlays"
    )
    parser.add_argument(
        "--mask_idx", type=int, default=None,
        help="Optional: index of the mask to save (0-based). If out of range or not specified, save all masks."
    )
    parser.add_argument(
        "--no-overlay", action="store_true", dest="no_overlay",
        help="If specified, do not generate overlays, only save mask images"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on: 'cpu' or 'cuda'. If not specified, automatically choose (cuda if available)."
    )
    parser.add_argument(
        "--multimask", action="store_true", default=True,
        help="Allow SAM2 to return multiple masks. Default is True. If you only want a single mask, set --no-multimask."
    )
    parser.add_argument(
        "--no-multimask", action="store_false", dest="multimask",
        help="Do not return multiple masks; only a single mask."
    )
    return parser.parse_args()

def build_sam2(segmentor_cfg_path: str, sam_ckpt_path: str, device: torch.device):
    """
    Build and return the SAM2 segmentation model instance. Depends on sam2_segment.build_sam2_model.
    """
    # Delay import to ensure sam2_segment is in PYTHONPATH
    try:
        from sam2_segment import build_sam2_model
    except ImportError as e:
        print(f"ERROR: Could not import sam2_segment.build_sam2_model: {e}", file=sys.stderr)
        sys.exit(1)
    # build_sam2_model interface should accept cfg path, checkpoint path, device
    segmentor = build_sam2_model(sam_cfg=segmentor_cfg_path, sam_ckpt=sam_ckpt_path, device=device)
    return segmentor

def read_first_frame(video_path: str):
    """
    Use cv2.VideoCapture to read the first frame of the video.
    Returns a BGR ndarray, or exits if failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"ERROR: Cannot read first frame of the video: {video_path}", file=sys.stderr)
        sys.exit(1)
    return frame

def normalize_point_labels(labels: np.ndarray):
    """
    Convert arbitrary integer labels to 0/1 for SAM2: non-zero -> 1, zero -> 0.
    """
    norm = np.where(labels != 0, 1, 0).astype(np.int32)
    return norm

def save_mask(mask: np.ndarray, save_path: Path):
    """
    Save a single mask: mask is bool or 0/1 ndarray; convert to 0/255 uint8 and save as PNG.
    """
    if mask.dtype != np.uint8:
        # mask is True/False or 0/1
        mask_uint8 = (mask.astype(np.uint8) * 255)
    else:
        # mask may already be 0/255 or 0/1
        unique_vals = np.unique(mask)
        if set(unique_vals.tolist()) <= {0, 1}:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask  # assume already 0/255 or correct format
    cv2.imwrite(str(save_path), mask_uint8)

def save_overlay(frame_bgr: np.ndarray, mask: np.ndarray, save_path: Path, color=(0,0,255), alpha=0.5):
    """
    Generate an overlay visualization: overlay the mask area on the original image with a specified color and transparency, and save as PNG.
    - frame_bgr: uint8 BGR image
    - mask: bool or 0/1 ndarray
    - color: BGR tuple for the overlay color
    - alpha: float in [0,1], transparency of overlay (0 = full original, 1 = full color)
    """
    # Convert mask to boolean
    mask_bool = mask.astype(bool)
    # Create a color layer
    color_arr = np.zeros_like(frame_bgr, dtype=np.uint8)
    color_arr[:] = color
    # Blend only in mask region
    blended = frame_bgr.astype(np.float32).copy()
    blended[mask_bool] = (
        alpha * color_arr[mask_bool].astype(np.float32)
        + (1 - alpha) * frame_bgr[mask_bool].astype(np.float32)
    )
    out_img = blended.astype(np.uint8)
    cv2.imwrite(str(save_path), out_img)

def main():
    args = parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}", file=sys.stderr)

    # Check input_video
    if not os.path.isfile(args.input_video):
        print(f"ERROR: Input video not found: {args.input_video}", file=sys.stderr)
        sys.exit(1)

    # Parse point_coords and point_labels
    coords = args.point_coords
    if len(coords) % 2 != 0:
        print("ERROR: point_coords length must be even, corresponding to x,y pairs", file=sys.stderr)
        sys.exit(1)
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    labels = np.array(args.point_labels, dtype=np.int32)
    if labels.shape[0] != pts.shape[0]:
        print(f"ERROR: Number of point_labels ({labels.shape[0]}) does not match number of point_coords pairs ({pts.shape[0]})", file=sys.stderr)
        sys.exit(1)
    labels = normalize_point_labels(labels)

    # Read the first frame
    frame_bgr = read_first_frame(args.input_video)
    # Convert to RGB for SAM2 if needed
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Build SAM2 model
    print("[INFO] Building SAM2 model...", file=sys.stderr)
    segmentor = build_sam2(args.sam_cfg, args.sam_ckpt, device)

    # Run segmentation on the first frame
    print("[INFO] Running SAM2 segmentation on the first frame...", file=sys.stderr)
    try:
        segmentor.set_image(frame_rgb)
    except Exception as e:
        print(f"ERROR: Calling segmentor.set_image failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Call predict
    try:
        masks, scores, logits = segmentor.predict(
            point_coords=pts,
            point_labels=labels,
            box=None,
            mask_input=None,
            multimask_output=args.multimask,
            return_logits=False
        )
        # masks: ndarray, shape (N_masks, H, W), dtype bool or uint8
        # scores: array or list of floats
    except TypeError:
        # If SAM2 API does not support return_logits or other parameters, adjust call
        try:
            masks, scores = segmentor.predict(
                point_coords=pts,
                point_labels=labels,
                box=None,
                mask_input=None,
                multimask_output=args.multimask
            )
        except Exception as e2:
            print(f"ERROR: Calling segmentor.predict failed: {e2}", file=sys.stderr)
            sys.exit(1)

    # Determine number of masks
    if isinstance(masks, np.ndarray):
        num_masks = masks.shape[0]
    else:
        try:
            num_masks = len(masks)
        except Exception:
            num_masks = 1

    print(f"[INFO] Obtained {num_masks} mask(s)", file=sys.stderr)

    # Prepare output directories
    out_dir = Path(args.output_dir)
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    # Decide which masks to save
    idx_list = list(range(num_masks))
    if args.mask_idx is not None:
        if 0 <= args.mask_idx < num_masks:
            idx_list = [args.mask_idx]
        else:
            print(f"WARNING: Specified mask_idx={args.mask_idx} is out of range (total {num_masks}), saving all masks.", file=sys.stderr)

    # Save masks and overlays
    for idx in idx_list:
        mask = masks[idx]
        # Save mask PNG
        mask_path = masks_dir / f"mask_{idx:03d}.png"
        save_mask(mask, mask_path)
        print(f"[INFO] Saved mask {idx} to {mask_path}", file=sys.stderr)
        # Save overlay if requested
        if not args.no_overlay:
            overlay_path = overlays_dir / f"overlay_{idx:03d}.png"
            # Overlay color can be customized; here using red
            save_overlay(frame_bgr, mask, overlay_path, color=(0,0,255), alpha=0.5)
            print(f"[INFO] Saved overlay {idx} to {overlay_path}", file=sys.stderr)

    print("[INFO] Segmentation complete. Results saved.", file=sys.stderr)

if __name__ == "__main__":
    main()
