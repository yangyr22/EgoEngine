#!/usr/bin/env python3
"""
脚本功能：读取视频的第一帧，在给定坐标处绘制一个明显的红色标记（圆形 + 可选矩形框），并保存为 PNG 文件，方便确认该点位置。

用法示例：
    python visualize_first_frame_point.py \
        --video_path ./example/video/paragliding/original_video.mp4 \
        --x 420 --y 350 \
        --output_path ./first_frame_vis.png \
        --circle_radius 20 \
        --rect_half_size 30

参数说明：
    --video_path:  输入视频路径
    --x, --y:      要标记的点坐标 (像素)；假设与 remove_anything_video.py 中的 point_coords 保持一致
    --output_path: 输出 PNG 文件路径
    --circle_radius: 圆形标记半径，默认为 20，可根据需要调大或调小
    --circle_thickness: 圆环线宽，默认为 2；如果想填充实心圆，可设为 -1
    --rect_half_size:  如果想在点周围画一个正方形框，该框中心在 (x, y)，边长为 2 * rect_half_size。设为 0 时不绘制矩形。
    --rect_thickness:   矩形线宽，默认为 2
"""

import cv2
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="可视化视频第一帧上的一个点，并保存为 PNG")
    parser.add_argument('--video_path', type=str, 
                        default='/coc/flash7/yliu3735/workspace/inpaint/marked_video.mp4')
    parser.add_argument('--x', type=int, default=1185, # 1100,
                        help='要标记的点的 x 坐标（像素）')
    parser.add_argument('--y', type=int, default=700,# 900,
                        help='要标记的点的 y 坐标（像素）')
    parser.add_argument('--output_path', type=str, default='./first_frame_vis.png',
                        help='输出 PNG 文件路径')
    parser.add_argument('--circle_radius', type=int, default=20,
                        help='绘制红色圆形标记的半径，默认 20 像素')
    parser.add_argument('--circle_thickness', type=int, default=2,
                        help='绘制红色圆环的线宽；如果想填充实心圆，可设为 -1')
    parser.add_argument('--rect_half_size', type=int, default=30,
                        help='在点周围绘制正方形框时，半边长度；设为 0 则不绘制矩形')
    parser.add_argument('--rect_thickness', type=int, default=2,
                        help='正方形框的线宽，默认为 2')
    return parser.parse_args()

def main():
    args = parse_args()
    video_path = args.video_path
    x = args.x
    y = args.y
    out_path = args.output_path
    radius = args.circle_radius
    circ_thick = args.circle_thickness
    rect_half = args.rect_half_size
    rect_thick = args.rect_thickness

    if not os.path.isfile(video_path):
        print(f"错误：找不到视频文件: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频: {video_path}", file=sys.stderr)
        sys.exit(1)

    # 读取第一帧
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("错误：无法读取第一帧", file=sys.stderr)
        sys.exit(1)

    # 检查坐标是否在图像范围内
    h, w = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        print(f"警告：给定坐标 (x={x}, y={y}) 超出图像范围 (width={w}, height={h})", file=sys.stderr)
        # 仍然在图像中心附近画一个点以示意
    # 绘制圆形标记：红色 BGR=(0,0,255)
    # 如果 thickness=-1，则绘制实心圆
    cv2.circle(frame, center=(x, y), radius=radius, color=(0, 0, 255), thickness=circ_thick)

    # 可选：在点周围绘制正方形框，半边长度 rect_half
    if rect_half > 0:
        x0 = max(0, x - rect_half)
        y0 = max(0, y - rect_half)
        x1 = min(w - 1, x + rect_half)
        y1 = min(h - 1, y + rect_half)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=rect_thick)

    # 可选：在图像上绘制文本坐标信息
    text = f"({x}, {y})"
    # 选择文本位置：在点的下方或上方，避免越界
    text_org = (x + 5, y + radius + 20) if y + radius + 20 < h else (x + 5, y - radius - 10)
    cv2.putText(frame, text, text_org, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    # 保存结果
    # 确保父目录存在
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(out_path, frame)
    if not success:
        print(f"错误：无法保存图像到 {out_path}", file=sys.stderr)
        sys.exit(1)
    print(f"已保存带标记的第一帧到: {out_path}")

if __name__ == '__main__':
    main()
