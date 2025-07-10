import torch
import numpy as np
import cv2
import glob
import torch.nn as nn
from typing import Any, Dict, List
from pathlib import Path
from PIL import Image
import os
import sys
import argparse
import tempfile
import imageio
import imageio.v2 as iio


from sam2_segment_vid import build_sam2_model_video, predict_masks_with_sam2_video

from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from sttn_video_inpaint import build_sttn_model, \
    inpaint_video_with_builded_sttn
from pytracking.lib.test.evaluation.data import Sequence
from utils import dilate_mask, show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to a single input video",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"],
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    # 原来是 --sam_model_type，现在替换为 SAM2 需要的 cfg 参数
    parser.add_argument(
        "--sam_cfg", type=str, required=True,
        help="Path to the SAM2 config YAML.",
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="Path to the SAM2 checkpoint .pt file.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--tracker_ckpt", type=str, required=True,
        help="The path to tracker checkpoint.",
    )
    parser.add_argument(
        "--vi_ckpt", type=str, required=True,
        help="The path to video inpainter checkpoint.",
    )
    parser.add_argument(
        "--mask_idx", type=int, default=2, required=True,
        help="Which mask in the first frame to determine the inpaint region.",
    )
    parser.add_argument(
        "--fps", type=int, default=25, required=True,
        help="FPS of the input and output videos.",
    )
    # 新增 segment-only 选项
    parser.add_argument(
        "--segment-only", action="store_true", default=False,
        help="Only run segmentation on the key frame and save masks/overlays, then exit.",
    )
    parser.add_argument(
        "--uv", type=str, required=True,
        help="Path to your per-frame UV txt file."
    )
    parser.add_argument(
        "--propagate", default=True,
        help="If set, SAM2 will propagate masks across frames."
    )


class RemoveAnythingVideo(nn.Module):
    def __init__(
            self, 
            args,
            tracker_target="ostrack",
            segmentor_target="sam",
            inpainter_target="sttn",
    ):
        super().__init__()
        tracker_build_args = {
            "tracker_param": args.tracker_ckpt
        }
        segmentor_build_args = {
            "sam_cfg": args.sam_cfg,
            "sam_ckpt": args.sam_ckpt,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        inpainter_build_args = {
            "lama": {
                "lama_config": args.lama_config,
                "lama_ckpt": args.lama_ckpt
            },
            "sttn": {
                "model_type": "sttn",
                "ckpt_p": args.vi_ckpt
            }
        }

        self.uv           = args.uv
        self.vos_optimized= False
        self.propagate_vid= args.propagate

        self.tracker = self.build_tracker(
            tracker_target, **tracker_build_args)
        self.segmentor = self.build_segmentor(
            segmentor_target, **segmentor_build_args)
        self.inpainter = self.build_inpainter(
            inpainter_target, **inpainter_build_args[inpainter_target])
        self.tracker_target = tracker_target
        self.segmentor_target = segmentor_target
        self.inpainter_target = inpainter_target

    def build_tracker(self, target, **kwargs):
        assert target == "ostrack", "Only support ostrack tracker now."
        return build_ostrack_model(**kwargs)

    def build_segmentor(self, target="sam", **kwargs):
        assert target == "sam", "Only support sam now."
        return build_sam2_model(**kwargs)

    def build_inpainter(self, target="sttn", **kwargs):
        if target == "lama":
            return build_lama_model(**kwargs)
        elif target == "sttn":
            return build_sttn_model(**kwargs)
        else:
            raise NotImplementedError("Only support lama and sttn")

    def forward_segmentor(self, img, point_coords=None, point_labels=None,
                          box=None, mask_input=None, multimask_output=True,
                          return_logits=False):
        # img: numpy HxWx3 (uint8)
        # point_coords: np.ndarray shape (N,2)
        # point_labels: np.ndarray shape (N,)
        self.segmentor.set_image(img)
        masks, scores, logits = self.segmentor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        # 如果 SAM2 的 API 需要 reset_image，可在此调用，视 build_sam2_model 返回的对象而定
        # self.segmentor.reset_image()
        return masks, scores

    def forward_inpainter(self, frames, masks):
        if self.inpainter_target == "lama":
            for idx in range(len(frames)):
                frames[idx] = inpaint_img_with_builded_lama(
                    self.inpainter, frames[idx], masks[idx], device=self.device)
        elif self.inpainter_target == "sttn":
            frames = [Image.fromarray(frame) for frame in frames]
            masks = [Image.fromarray(np.uint8(mask * 255)) for mask in masks]
            frames = inpaint_video_with_builded_sttn(
                self.inpainter, frames, masks, device=self.device)
        else:
            raise NotImplementedError
        return frames

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_box_from_mask(mask):
        x, y, w, h = cv2.boundingRect(mask)
        return np.array([x, y, w, h])

    def forward(
            self,
            frame_ps: List[str],
            key_frame_idx: int,
            key_frame_point_coords: np.ndarray,
            key_frame_point_labels: np.ndarray,
            key_frame_mask_idx: int = None,
            dilate_kernel_size: int = 10,
    ):

        # ─── begin replacement ────────────────────────────
        print("Segmenting via SAM2-video + UV prompts")

        # 1) parse UV into left/right per-frame lists
        frame_coords_left, frame_coords_right = [], []
        with open(self.uv) as f:
            for line in f:
                uL, vL, uR, vR = [float(x) if x.lower()!="nan" else np.nan
                                  for x in line.strip().split(',')]
                frame_coords_left .append([] if np.isnan(uL) or np.isnan(vL) else [[uL, vL]])
                frame_coords_right.append([] if np.isnan(uR) or np.isnan(vR) else [[uR, vR]])

        # 2) build the SAM2-video predictor
        video_pred = build_sam2_model_video(
            sam_cfg       = args.sam_cfg,
            sam_ckpt      = args.sam_ckpt,
            device        = self.device,
            vos_optimized = self.vos_optimized
        )

        # 3) run full-video segmentation
        masks_dir = predict_masks_with_sam2_video(
            video_path         = frame_ps,           # or args.input_video
            predictor          = video_pred,
            frame_coords_left  = frame_coords_left,
            frame_coords_right = frame_coords_right,
            output_dir         = args.output_dir,
            dilate_kernel_size = dilate_kernel_size,
            propagate          = self.propagate_vid
        )
        masks_dir = Path(masks_dir)

        # 4) load and combine mask_0 + mask_1 per frame
        all_mask, all_frame = [], []
        for i, p in enumerate(frame_ps):
            d = masks_dir / f"frame_{i:06d}"
            m0 = cv2.imread(str(d/"mask_0.png"), cv2.IMREAD_GRAYSCALE) or np.zeros_like(key_mask)
            m1 = cv2.imread(str(d/"mask_1.png"), cv2.IMREAD_GRAYSCALE) or np.zeros_like(key_mask)
            comb = ((m0>0)|(m1>0)).astype(np.uint8)
            if dilate_kernel_size is not None:
                comb = dilate_mask(comb, dilate_kernel_size)
            all_mask .append(comb)
            all_frame.append(iio.imread(p))

        # 5) inpaint exactly as before
        print("Inpainting ...")
        all_frame = self.forward_inpainter(all_frame, all_mask)

        return all_frame, all_mask

def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)


def show_img_with_mask(img, mask):
    if np.max(mask) == 1:
        mask = np.uint8(mask * 255)
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_mask(plt.gca(), mask, random_color=False)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)


def show_img_with_point(img, point_coords, point_labels):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), point_coords, point_labels,
                size=(width * 0.04) ** 2)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)


if __name__ == "__main__":
    """Example usage:
    python remove_anything_video_new.py \
      --input_video /coc/flash7/yliu3735/workspace/inpaint/example/video/paragliding/original_video.mp4 \
      --coords_type key_in \
      --point_coords 652 162 \
      --point_labels 2 \
      --dilate_kernel_size 5 \
      --output_dir ./results \
      --sam_ckpt weights/sam2.1_hiera_base_plus.pt \
      --sam_cfg segment_anything2/sam2/configs/sam2.1/sam2.1_hiera_b.yaml \
      --lama_config lama/configs/prediction/default.yaml \
      --lama_ckpt ./pretrained_models/big-lama \
      --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
      --vi_ckpt ./pretrained_models/sttn.pth \
      --mask_idx 2 \
      --fps 25 \
      [--segment-only]
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import logging
    logger = logging.getLogger('imageio')
    logger.setLevel(logging.ERROR)

    dilate_kernel_size = args.dilate_kernel_size
    key_frame_mask_idx = args.mask_idx
    video_raw_p = args.input_video
    frame_raw_glob = None
    fps = args.fps
    num_frames = 10000
    output_dir = args.output_dir
    output_dir = Path(f"{output_dir}")
    frame_mask_dir = output_dir / f"mask_{dilate_kernel_size}"
    video_mask_p = output_dir / f"mask_{dilate_kernel_size}.mp4"
    video_rm_w_mask_p = output_dir / f"removed_w_mask_{dilate_kernel_size}.mp4"
    video_w_mask_p = output_dir / f"w_mask_{dilate_kernel_size}.mp4"
    video_w_box_p = output_dir / f"w_box_{dilate_kernel_size}.mp4"
    frame_mask_dir.mkdir(exist_ok=True, parents=True)

    # load raw video or raw frames
    if Path(video_raw_p).exists():
        all_frame = iio.mimread(video_raw_p, memtest=False)
        # 尝试获取 fps
        try:
            fps = imageio.v3.immeta(video_raw_p, exclude_applied=False)["fps"]
        except Exception:
            pass

        frame_ps = []
        for i in range(len(all_frame)):
            frame_p = str(mkstemp(suffix=f"{i:0>6}.png"))
            frame_ps.append(frame_p)
            iio.imwrite(frame_ps[i], all_frame[i])
    else:
        assert frame_raw_glob is not None
        frame_ps = sorted(glob.glob(frame_raw_glob))
        all_frame = [iio.imread(frame_p) for frame_p in frame_ps]
        fps = args.fps or 25
        # 保存临时视频（若需要）
        iio.mimwrite(video_raw_p, all_frame, fps=fps)

    frame_ps = frame_ps[:num_frames]
    
    point_labels = np.array(args.point_labels)
    if args.coords_type == "click":
        point_coords = get_clicked_point(frame_ps[0])
    elif args.coords_type == "key_in":
        point_coords = args.point_coords
    point_coords = np.array([point_coords])  # shape (1,2)

    # 构造模型
    model = RemoveAnythingVideo(args)
    model.to(device)

    # 如果只做 segment-only，就在关键帧上运行分割并保存结果，然后退出
    if args.segment_only:
        print("Segment-only 模式: 仅对关键帧做分割并保存结果后退出。")
        # 读取关键帧
        key_frame = iio.imread(frame_ps[0])
        # 运行前向分割
        masks, scores = model.forward_segmentor(
            key_frame, point_coords, point_labels)
        # 保存所有 mask 以及 overlay
        seg_dir = output_dir / "segment_only"
        seg_dir.mkdir(parents=True, exist_ok=True)
        # 如果 key_frame_mask_idx 指定了一个 index，可只保存该 mask；否则保存所有输出 masks
        if key_frame_mask_idx is not None:
            idx = key_frame_mask_idx
            if 0 <= idx < len(masks):
                mask = masks[idx]
                mask_np = np.uint8(mask * 255) if mask.dtype != np.uint8 else mask
                mask_path = seg_dir / f"mask_{idx}.png"
                iio.imwrite(str(mask_path), mask_np)
                # overlay
                overlay = show_img_with_mask(key_frame, mask)
                overlay_path = seg_dir / f"overlay_{idx}.png"
                iio.imwrite(str(overlay_path), overlay)
                print(f"Saved mask and overlay for mask_idx={idx} to {seg_dir}")
            else:
                print(f"Warning: 指定的 mask_idx={key_frame_mask_idx} 超出范围，共 {len(masks)} 个 masks。将保存所有 masks。")
                for i, mask in enumerate(masks):
                    mask_np = np.uint8(mask * 255) if mask.dtype != np.uint8 else mask
                    mask_path = seg_dir / f"mask_{i}.png"
                    iio.imwrite(str(mask_path), mask_np)
                    overlay = show_img_with_mask(key_frame, mask)
                    overlay_path = seg_dir / f"overlay_{i}.png"
                    iio.imwrite(str(overlay_path), overlay)
                print(f"Saved all {len(masks)} masks and overlays to {seg_dir}")
        else:
            # 保存所有 masks
            for i, mask in enumerate(masks):
                mask_np = np.uint8(mask * 255) if mask.dtype != np.uint8 else mask
                mask_path = seg_dir / f"mask_{i}.png"
                iio.imwrite(str(mask_path), mask_np)
                overlay = show_img_with_mask(key_frame, mask)
                overlay_path = seg_dir / f"overlay_{i}.png"
                iio.imwrite(str(overlay_path), overlay)
            print(f"Saved all {len(masks)} masks and overlays to {seg_dir}")
        sys.exit(0)

    # 否则，正常流程：tracking + segment + inpaint
    with torch.no_grad():
        all_frame_rm_w_mask, all_mask, all_box = model(
            frame_ps, 0, point_coords, point_labels, key_frame_mask_idx,
            dilate_kernel_size
        )
    # visual removed results
    iio.mimwrite(video_rm_w_mask_p, all_frame_rm_w_mask, fps=fps)

    # visual mask
    all_mask = [np.uint8(mask * 255) for mask in all_mask]
    for i in range(len(all_mask)):
        mask_p = frame_mask_dir /  f"{i:0>6}.jpg"
        iio.imwrite(str(mask_p), all_mask[i])
    iio.mimwrite(video_mask_p, all_mask, fps=fps)

    # visual video with mask overlay
    tmp = []
    for i in range(len(all_mask)):
        # 这里使用原始帧列表 all_frame 进行 overlay；如果 all_frame 可能被修改，也可用 frame_ps[i] 重新读取
        # 读取原始帧确保尺寸一致：
        frame = iio.imread(frame_ps[i])
        tmp.append(show_img_with_mask(frame, all_mask[i]))
    iio.mimwrite(video_w_mask_p, tmp, fps=fps)
