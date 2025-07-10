#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path
import tyro

def convert_all_mp4_in_dir(
    input_dir: str,
):
    """
    从 input_dir 中查找所有 .mp4 文件，并使用 FFmpeg 批量转换为 H.264 + yuv420p 格式。

    输出文件名示例： original.mp4 -> original_converted.mp4
    输出位置：与原文件同目录。
    """
    input_dir_path = Path(input_dir)
    if not input_dir_path.is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        return

    # 找出所有 .mp4 文件
    mp4_files = list(input_dir_path.rglob("*.mp4"))
    if not mp4_files:
        print(f"No .mp4 files found in {input_dir}.")
        return

    for mp4_file in mp4_files:
        # 生成输出文件名：<stem>_converted.mp4
        output_file = mp4_file.with_name(mp4_file.stem + "_converted.mp4")
        print(f"Converting: {mp4_file} -> {output_file}")

        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(mp4_file),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",  # 覆盖输出文件
            str(output_file)
        ]
        subprocess.run(ffmpeg_cmd, check=True)
    
    print("Conversion completed.")


def main(
    input_dir: str
):
    """
    Example usage:
      python3 convert_mp4.py --input-dir /path/to/videos
    """
    convert_all_mp4_in_dir(input_dir)


if __name__ == "__main__":
    tyro.cli(main)
