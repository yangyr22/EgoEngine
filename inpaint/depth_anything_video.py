import os
import sys
import argparse
import cv2
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="对视频逐帧估计深度并输出可视化视频，仅做深度，不做分割。")
    parser.add_argument('--input_video', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出视频路径，例如 depth_video.mp4')
    parser.add_argument('--depth_encoder', type=str, required=True,
                        help='Depth-Anything V2 encoder，如 vits, vitb, vitl 等')
    parser.add_argument('--depth_ckpt', type=str, required=True,
                        help='Depth-Anything V2 checkpoint (.pth) 路径')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备，cuda 或 cpu，若不指定则自动判断')
    parser.add_argument('--resize_factor', type=float, default=1.0,
                        help='可选缩放系数 (<1 加速)，在估计前对帧做 cv2.resize，再在可视化阶段放大回原尺寸；默认 1.0')
    return parser.parse_args()

def load_depth_model(encoder: str, ckpt_path: str, device: torch.device):
    """
    根据 encoder 加载 Depth-Anything V2 模型并载入权重，返回 model。
    需要 depth_anything_v2 代码在 PYTHONPATH 中。
    """
    # 延迟导入，确保 PYTHONPATH 已设置
    from depth_anything_v2.dpt import DepthAnythingV2

    # 根据官方配置选择
    if encoder == 'vits':
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    elif encoder == 'vitb':
        model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
    elif encoder == 'vitl':
        model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")
    # 加载权重
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    将任意 float depth_map 归一化到 [0,1]，返回 float32。
    处理 NaN/inf：无效区域设为0。
    """
    depth = depth_map.astype(np.float32)
    mask_valid = np.isfinite(depth)
    if not np.any(mask_valid):
        return np.zeros_like(depth)
    valid = depth[mask_valid]
    minv, maxv = valid.min(), valid.max()
    if maxv > minv:
        norm = (depth - minv) / (maxv - minv + 1e-8)
    else:
        norm = np.zeros_like(depth)
    norm[~mask_valid] = 0.0
    return norm

def depth_to_color(norm_depth: np.ndarray) -> np.ndarray:
    """
    将归一化深度 [0,1] 转伪彩 BGR 图像
    """
    vis = (norm_depth * 255.0).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return vis_color

def main():
    args = parse_args()

    # 设备选择
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}", file=sys.stderr)

    # 检查输入视频
    if not os.path.isfile(args.input_video):
        print(f"[ERROR] The video does not exits!: {args.input_video}", file=sys.stderr)
        sys.exit(1)

    # 加载模型
    print("[INFO] loading Depth-Anything Model...", file=sys.stderr)
    model = load_depth_model(args.depth_encoder, args.depth_ckpt, device)

    # 打开视频
    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video info: width={orig_w}, height={orig_h}, FPS={fps:.2f}, total frames={total_frames}", file=sys.stderr)

    # 确定处理尺寸
    resize_factor = args.resize_factor if args.resize_factor > 0 else 1.0
    if resize_factor != 1.0:
        proc_w = int(orig_w * resize_factor)
        proc_h = int(orig_h * resize_factor)
        print(f"[INFO] resize factors {resize_factor:.3f}: processing size {proc_w}x{proc_h}", file=sys.stderr)
    else:
        proc_w, proc_h = orig_w, orig_h

    # 输出视频：仅深度可视化，尺寸保持原始大小
    out_w, out_h = orig_w, orig_h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"[ERROR] Can't read VideoWriter: {args.output_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # 可能缩放以加速
        if resize_factor != 1.0:
            frame_small = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame_bgr

        # Depth-Anything 推理
        # 注意：根据模型实现，infer_image 接口可能接受 BGR 或 RGB，若需要 RGB，请做 cvtColor
        img_input = frame_small  # 若需要 RGB: cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            depth_map_small = model.infer_image(img_input)  # numpy float32 array, shape (proc_h, proc_w)
        # 归一化
        norm_small = normalize_depth(depth_map_small)  # float32 [0,1]
        # 放大回原始尺寸
        if resize_factor != 1.0:
            norm_full = cv2.resize(norm_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            norm_full = norm_small
        # 伪彩
        depth_vis = depth_to_color(norm_full)  # BGR, uint8, shape (orig_h, orig_w, 3)

        # 写入输出
        writer.write(depth_vis)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[INFO] Finished! Video save to: {args.output_path}", file=sys.stderr)

if __name__ == '__main__':
    main()