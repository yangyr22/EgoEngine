#!/usr/bin/env python3

import pickle
import subprocess
from pathlib import Path
from typing import Optional, List, Union, Dict

import cv2
import numpy as np
import tqdm
import tyro

from sapien.utils import Viewer
from sapien.asset import create_dome_envmap
import sapien  # 如果你习惯 "import sapien.core as sapien" 也可以

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


# ---------------------------------------------------------------------
# 第一步：保存 clip 为 mp4 + pickle
# ---------------------------------------------------------------------
def save_clip_and_pickle(
    frames: List[np.ndarray],
    qposes: List[np.ndarray],
    output_dir: Path,
    clip_count: int,
    config_path: str,
    dof: int,
    joint_names: List[str],
) -> (Path, Path):
    """
    先用 OpenCV (mp4v, fps=50) 写到临时文件，再用 ffmpeg 转码为 H.264 + yuv420p，
    生成最终 <clip_name>.mp4，并保存对应 <clip_name>.pickle。
    返回 (video_file, pickle_file) 供后续使用。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_name = f"{clip_count:04d}"
    # 最终输出的 .mp4 与 .pickle
    video_file = output_dir / f"{clip_name}.mp4"
    pickle_file = output_dir / f"{clip_name}.pickle"

    # 存储元信息 + qposes 到 pickle
    meta_data = dict(
        config_path=config_path,
        dof=dof,
        joint_names=joint_names
    )
    with pickle_file.open("wb") as f:
        pickle.dump(dict(data=qposes, meta_data=meta_data), f)

    # 先写到临时文件
    temp_file = video_file.with_name(f"{video_file.stem}_temp.mp4")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_file), fourcc, 50.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    # 用 ffmpeg 转码为 H.264 + yuv420p
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", str(temp_file),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-y",
        str(video_file)
    ]
    print(f"[FFMPEG] Converting temp -> {video_file}")
    subprocess.run(ffmpeg_cmd, check=True)

    # 删除临时文件
    temp_file.unlink()
    return video_file, pickle_file


# ---------------------------------------------------------------------
# 第二步：渲染刚才生成的 pickle（代码 B）
# ---------------------------------------------------------------------
def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: bool = True,
):
    """
    根据 meta_data & data，用 SAPIEN 渲染机器人手部动作，并输出 <...>.mp4 (H.264 + yuv420p)。
    逻辑：
      1) 用 OpenCV (mp4v, fps=100) 写到临时 mp4
      2) 用 ffmpeg 转码到 libx264 + yuv420p
    """

    use_rt = headless
    if not use_rt:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
    else:
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(16)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

    # 创建场景
    scene = sapien.Scene()

    # 地面
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])


    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    cam = scene.add_camera("Cheese!", 600, 600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.5, 0, 0.0], [0, 0, 0, -1]))

    # Viewer
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None

    record_video = output_video_path is not None
    writer = None
    temp_video_file = None
    fps = 30.0  # 你也可以改成 50

    if record_video:
        out_path = Path(output_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video_file = out_path.with_name(f"{out_path.stem}_temp.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video_file), fourcc, fps, (cam.get_width(), cam.get_height()))

    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True

    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    # 给不同手指定初始姿态
    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "inspire" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))

    sapien_joint_names = [j.get_name() for j in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array([retargeting_joint_names.index(n) for n in sapien_joint_names]).astype(int)

    for qpos in tqdm.tqdm(data, desc="Rendering with sapien"):
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        if viewer is not None:
            for _ in range(2):
                viewer.render()

        if record_video and writer is not None:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

    if record_video and writer is not None:
        writer.release()
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(temp_video_file),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            str(output_video_path),
        ]
        print(f"Converting to H.264 (yuv420p): {temp_video_file} -> {output_video_path}")
        subprocess.run(ffmpeg_cmd, check=True)
        temp_video_file.unlink()

    scene = None


def retarget_folder(retargeting: SeqRetargeting, video_path: str, config_path: str, robot_name: str, headless: bool):
    """
    遍历指定文件夹的 .jpg 帧：连续 >=50 帧检测到手 -> 生成 <clip>.mp4 + <clip>.pickle -> 立刻 render_by_sapien -> <clip>_<robot_name>.mp4
    """
    image_paths = sorted(Path(video_path).glob("*.jpg"))
    if not image_paths:
        print("Error: No image files found in the specified folder.")
        return

    # 推断输出到 ../clip/Pxx_xx/
    p = Path(video_path)
    if p.name == "rgb":
        print("Error: video_path should be a subdirectory under 'rgb'")
        return
    output_path = p.parent.parent / "clip" / p.name

    detector = SingleHandDetector(hand_type="Right", selfie=False)
    length = len(image_paths)

    frames_buffer = []
    qpos_buffer = []
    clip_count = 1

    dof = len(retargeting.optimizer.robot.dof_joint_names)
    joint_names = retargeting.optimizer.robot.dof_joint_names

    def finalize_segment():
        nonlocal frames_buffer, qpos_buffer, clip_count
        if len(frames_buffer) >= 50:
            clip_mp4, clip_pkl = save_clip_and_pickle(
                frames_buffer,
                qpos_buffer,
                output_path,
                clip_count,
                config_path,
                dof,
                joint_names
            )

            pickle_data = np.load(clip_pkl, allow_pickle=True)
            meta_data, data = pickle_data["meta_data"], pickle_data["data"]
            rendered_name = f"{clip_mp4.stem}_{robot_name}.mp4"
            rendered_path = str(clip_mp4.parent / rendered_name)

            render_by_sapien(meta_data, data, output_video_path=rendered_path, headless=headless)

            clip_count += 1

        frames_buffer.clear()
        qpos_buffer.clear()

    with tqdm.tqdm(total=length) as pbar:
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                pbar.update(1)
                continue

            rgb = frame[..., ::-1]
            num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(rgb)

            if num_box > 0:
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                qpos = retargeting.retarget(ref_value)
                frames_buffer.append(frame)
                qpos_buffer.append(qpos)
            else:
                if frames_buffer:
                    finalize_segment()

            pbar.update(1)

    if frames_buffer:
        finalize_segment()

    retargeting.verbose()


def main(
    robot_name: RobotName,
    video_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    headless: bool = True,
):
    """
    用法示例：
      python3 retargeting.py \
        --robot-name allegro \
        --video-path /srv/rail-lab/flash5/yzheng494/0206_output/raw_video/video1.mp4 \
        --retargeting-type dexpilot \
        --hand-type right \
        --headless
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retarget_folder(retargeting, video_path, str(config_path), robot_name, headless)


if __name__ == "__main__":
    tyro.cli(main)
