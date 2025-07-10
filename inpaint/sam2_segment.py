#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
from typing import List

# Ensure SAM2 imports work: SAM2 must be importable in your PYTHONPATH or installed.
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask, show_points

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


def predict_masks_with_sam2(
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        sam_cfg: str,
        sam_ckpt: str,
        device="cuda",
        predictor=None
):
    """
    Run SAM2 inference on a single image with point prompts.
    """
    # If build_sam2 does not move to device internally, uncomment:
    # model.to(device)
    if predictor is None:
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
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if logits is not None and isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()

    masks = (masks > 0).astype(np.uint8)
    return masks, scores, logits


def build_sam2_model(sam_cfg: str, sam_ckpt: str, device="cuda"):
    """
    Build and return a SAM2 predictor, for reuse across multiple inferences.
    Args:
      sam_cfg: Path to SAM2 config YAML, 例如 '/.../configs/sam2.1/sam2.1_hiera_b.yaml'
      sam_ckpt: Path to SAM2 checkpoint .pt 文件
      device: 'cuda' 或 'cpu'
    Returns:
      predictor: SAM2ImagePredictor 实例
    """
    # 1. 分离配置目录和配置名称
    cfg_full_path = Path(sam_cfg)
    config_dir = str(cfg_full_path.parent)   # 例如 '/.../configs/sam2.1'
    config_name = cfg_full_path.name         # 例如 'sam2.1_hiera_b'

    # 2. 清理全局 Hydra 状态（若已有初始化）
    GlobalHydra.instance().clear()

    # 3. 初始化 Hydra，使得 config_path 包含 SAM2 的配置目录
    # version_base=None 可使用默认 Hydra 版本兼容
    initialize(config_path=config_dir, job_name="sam2_inference", version_base=None)

    # 4. 调用 build_sam2，传入配置名称，不带扩展名
    # 注意：build_sam2 的签名应接收 config_name 参数
    model = build_sam2(config_name, sam_ckpt)  
    # 通常 build_sam2 通过内部的 Hydra compose 加载配置

    # 5. 清理 Hydra，如果后续还要再次初始化
    GlobalHydra.instance().clear()

    # 6. 构造 Predictor
    predictor = SAM2ImagePredictor(model, device=device)
    return predictor


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input image",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate(s) of the point prompt, e.g. two values [x y].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0. Provide same length as point_coords/2.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size to apply to each mask. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory path to save results.",
    )
    parser.add_argument(
        "--sam_cfg", type=str, required=True,
        help="Path to the SAM2 config YAML (e.g., segment_anything2/configs/sam2.1/sam2.1_hiera_base_plus.yaml).",
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="Path to the SAM2 checkpoint .pt file (e.g., sam2.1_hiera_base_plus.pt).",
        
    )


if __name__ == "__main__":
    """
    Example usage:
    python sam2_segment.py \
        --input_img /path/to/image.png \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_cfg /.../segment_anything2/configs/sam2.1/sam2.1_hiera_base_plus.yaml \
        --sam_ckpt /.../segment_anything2/checkpoints/sam2.1_hiera_base_plus.pt
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_img_to_array(args.input_img)

    coords = args.point_coords
    if len(coords) % 2 != 0:
        raise ValueError("point_coords must contain pairs of [x y].")
    point_coords: List[List[float]] = []
    for i in range(0, len(coords), 2):
        point_coords.append([coords[i], coords[i+1]])
    if len(args.point_labels) != len(point_coords):
        raise ValueError(
            f"Number of point_labels ({len(args.point_labels)}) != number of point_coords ({len(point_coords)})."
        )
    point_labels = args.point_labels
    
    predictor = build_sam2_model(args.sam_cfg, args.sam_ckpt, device=device)

    masks, scores, _ = predict_masks_with_sam2(
        img=img,
        point_coords=point_coords,
        point_labels=point_labels,
        sam_cfg=args.sam_cfg,
        sam_ckpt=args.sam_ckpt,
        device=device,
        predictor=predictor
    )
    masks_uint8 = [(mask.astype(np.uint8) * 255) for mask in masks]
    if args.dilate_kernel_size is not None:
        masks_uint8 = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks_uint8]

    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, mask in enumerate(masks_uint8):
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{mask_p.name}"

        save_array_to_img(mask, mask_p)

        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), point_coords, point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"Saved {len(masks_uint8)} mask(s) under {out_dir}")
