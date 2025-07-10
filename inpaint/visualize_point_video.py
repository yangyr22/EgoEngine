#!/usr/bin/env python3
"""
脚本功能：读取视频每一帧，在 uv.txt 中对应行的两个点位置绘制标记，如果坐标为 NaN 则跳过该点绘制，并保存为新视频。

用法示例：
    python visualize_uv_video.py \
        --video_path ./example/video/paragliding/original_video.mp4 \
        --uv_path ./uv.txt \
        --output_path ./annotated_video.mp4 \
        --circle_radius 10 \
        --circle_thickness 2 \
        --rect_half_size 0 \
        --rect_thickness 2
"""

import cv2
import argparse
import os
import sys
import math  # 用于 isnan 检查

def parse_args():
    parser = argparse.ArgumentParser(description="读取 uv.txt，为视频每帧的两个点绘制标记并生成新视频，NaN 坐标则跳过绘制该点")
    parser.add_argument('--video_path', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--uv_path', type=str, required=True,
                        help='uv 文本路径，每行: right_u,right_v,left_u,left_v')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出视频路径，例如 annotated.mp4')
    parser.add_argument('--circle_radius', type=int, default=10,
                        help='绘制圆形标记的半径，默认 10 像素')
    parser.add_argument('--circle_thickness', type=int, default=2,
                        help='绘制圆环线宽；若设 -1 则实心圆，默认 2')
    parser.add_argument('--rect_half_size', type=int, default=0,
                        help='正方形框半边长度；设为 0 不绘制矩形，默认 0')
    parser.add_argument('--rect_thickness', type=int, default=2,
                        help='正方形框线宽，默认 2')
    parser.add_argument('--right_color', type=str, default='0,0,255',
                        help='右点颜色 BGR，如 "0,0,255"，默认红色')
    parser.add_argument('--left_color', type=str, default='0,255,0',
                        help='左点颜色 BGR，如 "0,255,0"，默认绿色')
    parser.add_argument('--show_label', action='store_true', default=True,
                        help='是否绘制标签 "R"/"L"，默认开启；如不想显示，可加 --no-show_label')
    parser.add_argument('--no-show_label', dest='show_label', action='store_false',
                        help='关闭标签显示')
    parser.add_argument('--font_scale', type=float, default=0.6,
                        help='标签字体缩放，默认 0.6')
    parser.add_argument('--font_thickness', type=int, default=1,
                        help='标签字体线宽，默认 1')
    return parser.parse_args()

def read_uv_file(uv_path):
    """
    读取 uv.txt，返回列表，每项为 (right_u, right_v, left_u, left_v) 浮点数（可能为 NaN）。
    跳过空行和不符合格式的行（会打印警告）。
    """
    uv_list = []
    if not os.path.isfile(uv_path):
        print(f"错误：找不到 uv 文件: {uv_path}", file=sys.stderr)
        sys.exit(1)
    with open(uv_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 4:
                print(f"警告：第 {idx+1} 行格式不正确（期望 4 列），跳过: {line}", file=sys.stderr)
                continue
            try:
                ru = float(parts[0]); rv = float(parts[1])
                lu = float(parts[2]); lv = float(parts[3])
                uv_list.append((ru, rv, lu, lv))
            except ValueError:
                print(f"警告：第 {idx+1} 行数值无法解析，跳过: {line}", file=sys.stderr)
                continue
    if len(uv_list) == 0:
        print("错误：未读取到任何有效的 uv 坐标", file=sys.stderr)
        sys.exit(1)
    return uv_list

def parse_color(s):
    """
    将 "B,G,R" 字符串解析为 (B, G, R) 三元组 int。
    """
    parts = s.split(',')
    if len(parts) != 3:
        print(f"错误：颜色参数格式应为 B,G,R，收到: {s}", file=sys.stderr)
        sys.exit(1)
    try:
        b = int(parts[0]); g = int(parts[1]); r = int(parts[2])
        for v in (b, g, r):
            if not (0 <= v <= 255):
                raise ValueError
        return (b, g, r)
    except:
        print(f"错误：颜色值应为 0-255 整数，收到: {s}", file=sys.stderr)
        sys.exit(1)

def clamp_point(x, y, max_w, max_h, frame_idx, side):
    """
    将点 (x,y) clamp 到图像边界内，如果超出则 clamp 并打印警告。
    side: "R" 或 "L"，用于提示信息。
    """
    x_clamped = max(0, min(x, max_w-1))
    y_clamped = max(0, min(y, max_h-1))
    if (x != x_clamped) or (y != y_clamped):
        print(f"警告：第 {frame_idx} 帧 {side} 点坐标超出边界，已 clamp: 原({x},{y}) -> 新({x_clamped},{y_clamped})", file=sys.stderr)
    return x_clamped, y_clamped

def main():
    args = parse_args()
    video_path = args.video_path
    uv_path = args.uv_path
    out_path = args.output_path

    # 读取 uv 列表
    uv_list = read_uv_file(uv_path)
    total_uv = len(uv_list)

    # 打开视频
    if not os.path.isfile(video_path):
        print(f"错误：找不到视频文件: {video_path}", file=sys.stderr)
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频: {video_path}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: 宽度={width}, 高度={height}, FPS={fps:.2f}, 总帧数={total_frames}", file=sys.stderr)
    print(f"UV 坐标行数: {total_uv}", file=sys.stderr)

    # 创建输出目录
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 选择编码，常见 mp4v 可写 mp4；若要写 avi 可改 'XVID' 等
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"错误：无法打开 VideoWriter，检查编码器或输出路径: {out_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # 颜色解析
    right_color = parse_color(args.right_color)
    left_color = parse_color(args.left_color)

    idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= total_uv:
            print(f"警告：帧索引 {idx} >= uv 行数 {total_uv}，停止处理", file=sys.stderr)
            break

        ru, rv, lu, lv = uv_list[idx]
        ru, rv = rv, ru
        h, w = frame.shape[:2]

        # 右点绘制：先检查是否为 NaN
        if not (math.isnan(ru) or math.isnan(rv)):
            # 四舍五入到整数像素
            x_r = int(round(ru))
            y_r = int(round(rv))
            # clamp 到边界
            x_r, y_r = clamp_point(x_r, y_r, w, h, idx, "R")
            # 绘制右点圆
            cv2.circle(frame, center=(x_r, y_r), radius=args.circle_radius,
                       color=right_color, thickness=args.circle_thickness)
            # 可选矩形框
            if args.rect_half_size > 0:
                x0_r = max(0, x_r - args.rect_half_size)
                y0_r = max(0, y_r - args.rect_half_size)
                x1_r = min(w - 1, x_r + args.rect_half_size)
                y1_r = min(h - 1, y_r + args.rect_half_size)
                cv2.rectangle(frame, (x0_r, y0_r), (x1_r, y1_r),
                              color=right_color, thickness=args.rect_thickness)
            # 可选标签
            if args.show_label:
                text_r = "R"
                text_size = cv2.getTextSize(text_r, font, args.font_scale, args.font_thickness)[0]
                tx = x_r + 5
                ty = y_r - 5 if y_r - 5 > text_size[1] else y_r + text_size[1] + 5
                cv2.putText(frame, text_r, (tx, ty), font, args.font_scale, right_color, args.font_thickness, cv2.LINE_AA)
        else:
            # 对于 NaN，打印一次性提示（每帧一次）
            print(f"第 {idx} 帧右点坐标为 NaN，跳过绘制右点", file=sys.stderr)

        # 左点绘制：先检查是否为 NaN
        if not (math.isnan(lu) or math.isnan(lv)):
            x_l = int(round(lu))
            y_l = int(round(lv))
            x_l, y_l = clamp_point(x_l, y_l, w, h, idx, "L")
            cv2.circle(frame, center=(x_l, y_l), radius=args.circle_radius,
                       color=left_color, thickness=args.circle_thickness)
            if args.rect_half_size > 0:
                x0_l = max(0, x_l - args.rect_half_size)
                y0_l = max(0, y_l - args.rect_half_size)
                x1_l = min(w - 1, x_l + args.rect_half_size)
                y1_l = min(h - 1, y_l + args.rect_half_size)
                cv2.rectangle(frame, (x0_l, y0_l), (x1_l, y1_l),
                              color=left_color, thickness=args.rect_thickness)
            if args.show_label:
                text_l = "L"
                text_size = cv2.getTextSize(text_l, font, args.font_scale, args.font_thickness)[0]
                tx2 = x_l + 5
                ty2 = y_l - 5 if y_l - 5 > text_size[1] else y_l + text_size[1] + 5
                cv2.putText(frame, text_l, (tx2, ty2), font, args.font_scale, left_color, args.font_thickness, cv2.LINE_AA)
        else:
            print(f"第 {idx} 帧左点坐标为 NaN，跳过绘制左点", file=sys.stderr)

        # 将帧写入输出
        writer.write(frame)

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx+1} 帧", file=sys.stderr)
        idx += 1

    cap.release()
    writer.release()
    print(f"处理完成，已保存到: {out_path}", file=sys.stderr)

if __name__ == '__main__':
    main()
