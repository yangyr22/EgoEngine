import os
import subprocess
from pathlib import Path
from typing import Optional, List, Union, Dict

import cv2
import numpy as np
import tqdm
import tyro

import sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


GLOBAL_RENDER_CONTEXT = {
    "scene": None,
    "viewer": None,
    "camera": None,
    "headless": None,
}


def init_sapien_if_needed(headless: bool):
    if GLOBAL_RENDER_CONTEXT["scene"] is not None:
        return

    GLOBAL_RENDER_CONTEXT["headless"] = headless

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

    # 只创建一次场景
    scene = sapien.Scene()

    # 地面
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # 光照
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # 相机
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

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

    GLOBAL_RENDER_CONTEXT["scene"] = scene
    GLOBAL_RENDER_CONTEXT["viewer"] = viewer
    GLOBAL_RENDER_CONTEXT["camera"] = cam


def save_clip_and_pickle(
    frames: List[np.ndarray],
    qposes: List[np.ndarray],
    output_dir: Path,
    clip_count: int,
    config_path: str,
    dof: int,
    joint_names: List[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_name = f"{clip_count:04d}"
    video_file = output_dir / f"{clip_name}.mp4"
    pickle_file = output_dir / f"{clip_name}.pickle"

    meta_data = dict(
        config_path=config_path,
        dof=dof,
        joint_names=joint_names,
    )
    import pickle
    with pickle_file.open("wb") as f:
        pickle.dump(dict(data=qposes, meta_data=meta_data), f)

    temp_file = video_file.with_name(video_file.stem + "_temp.mp4")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_file), fourcc, 50.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", str(temp_file),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-y",
        str(video_file),
    ]
    print(f"[FFMPEG] Converting temp -> {video_file}")
    subprocess.run(ffmpeg_cmd, check=True)

    temp_file.unlink()

    return video_file, pickle_file


def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: bool = False,
):
    """
    使用全局 scene/viewer, 每次加载 robot -> 渲染 -> 删除robot
    """
    # 确保scene只初始化一次
    init_sapien_if_needed(headless)

    scene = GLOBAL_RENDER_CONTEXT["scene"]
    viewer = GLOBAL_RENDER_CONTEXT["viewer"]
    cam = GLOBAL_RENDER_CONTEXT["camera"]

    record_video = output_video_path is not None
    writer = None
    temp_video_file = None
    fps = 50.0

    if record_video:
        out_path = Path(output_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video_file = out_path.with_name(f"{out_path.stem}_temp.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video_file), fourcc, fps, (cam.get_width(), cam.get_height()))

    # =============== 加载配置, robot urdf, scale ===============
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True

    # scale
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

    # 将返回“articulation”对象
    robot_articulation = loader.load(filepath)

    if "ability" in robot_name:
        robot_articulation.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot_articulation.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot_articulation.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot_articulation.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot_articulation.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot_articulation.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot_articulation.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "inspire" in robot_name:
        robot_articulation.set_pose(sapien.Pose([0, 0, -0.15]))

    sapien_joint_names = [j.get_name() for j in robot_articulation.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array([retargeting_joint_names.index(n) for n in sapien_joint_names], dtype=int)

    # --- 开始渲染 ---
    for qpos in tqdm.tqdm(data, desc="Rendering with sapien"):
        robot_articulation.set_qpos(np.array(qpos)[retargeting_to_sapien])

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
        print(f"[FFMPEG] Converting to H.264 (yuv420p): {temp_video_file} -> {output_video_path}")
        subprocess.run(ffmpeg_cmd, check=True)
        temp_video_file.unlink()

    # ---------删除这个articulation-----------
    scene.remove_articulation(robot_articulation)



def retarget_folder(
    retargeting: SeqRetargeting,
    video_path: Path,
    config_path: str,
    headless: bool = True,
):
    """
    当 >= 100 帧连续检测到手，就保存 clip 并立即 render_by_sapien
    """
    image_paths = sorted(video_path.glob("*.jpg"))
    if not image_paths:
        print(f"Error: No image files found in {video_path}")
        return

    if video_path.name == "rgb":
        print("Error: video_path should be a subdirectory under 'rgb'")
        return

    output_path = video_path.parent.parent / "clip" / video_path.name

    # 修正检查特定子字符串的逻辑
    exclude_substrings = [
        "P02_134", "P02_135", "P02_14", "P02_127",
        "P02_114", "P02_11", "P02_05", "P01_16", "P03_08", "P02_13",
        "P03_108", "P03_122", "P03_111", "P03_13", "P04_26", "P05_05", "P03_110",
        "P03_114", "P03_115", "P03_12", "P03_15", "P03_18", "P03_26",
        "P04_19", "P05_02", "P05_05", "P05_06", "P05_07", "P05_09",
        "P06_04", "P06_08",
    ]
    if any(sub in str(video_path) for sub in exclude_substrings):
        print(f"Folder {video_path} contains excluded substrings. Skipping.")
        try:
            output_path.rmdir()
            print(f"Deleted empty output folder: {output_path}")
        except OSError:
            print(f"Output folder {output_path} is not empty or cannot be deleted.")
        return

    if output_path.exists() and any(output_path.iterdir()):
        print(f"Output path {output_path} already exists and is not empty. Skipping.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    detector = SingleHandDetector(hand_type="Right", selfie=False)
    length = len(image_paths)

    frames_buffer = []
    qpos_buffer = []
    clip_count = 1

    dof = len(retargeting.optimizer.robot.dof_joint_names)
    joint_names = retargeting.optimizer.robot.dof_joint_names

    clips_saved = 0  # 记录保存的clip数量

    def finalize_segment():
        nonlocal frames_buffer, qpos_buffer, clip_count, clips_saved
        if len(frames_buffer) >= 100:
            clip_mp4_path, clip_pkl_path = save_clip_and_pickle(
                frames_buffer,
                qpos_buffer,
                output_path,
                clip_count,
                config_path,
                dof,
                joint_names
            )
            # B) 加载 pickle, 调 render_by_sapien
            pickle_data = np.load(clip_pkl_path, allow_pickle=True)
            meta_data, data = pickle_data["meta_data"], pickle_data["data"]

            # 输出 mp4 命名: <clip_name>_allegro.mp4 (也可改成其它)
            rendered_mp4 = clip_mp4_path.with_name(f"{clip_mp4_path.stem}_allegro.mp4")
            render_by_sapien(meta_data, data, str(rendered_mp4), headless=headless)

            clip_count += 1
            clips_saved += 1  # 增加clip计数

        frames_buffer.clear()
        qpos_buffer.clear()

    with tqdm.tqdm(total=length, desc=f"Processing {video_path.name}") as pbar:
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

    if clips_saved == 0:
        try:
            output_path.rmdir()
            print(f"No clips were saved. Deleted output folder: {output_path}")
        except OSError:
            print(f"Output folder {output_path} is not empty or cannot be deleted.")
    else:
        print(f"Total clips saved in {output_path}: {clips_saved}")

    retargeting.verbose()




def process_all(
    base_dir: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    headless: bool = True
):
    base_dir_path = Path(base_dir)
    if not base_dir_path.is_dir():
        print(f"Error: base_dir={base_dir} is not a valid directory.")
        return

    # 先构建 retargeting
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    # 查找 /Pxx/rgb/Pxx_xx/
    valid_folders = []
    for participant_dir in base_dir_path.iterdir():
        if participant_dir.is_dir() and participant_dir.name.startswith("P"):
            rgb_dir = participant_dir / "rgb"
            if rgb_dir.is_dir():
                for seq_dir in rgb_dir.iterdir():
                    if seq_dir.is_dir() and seq_dir.name.startswith(participant_dir.name + "_"):
                        valid_folders.append(seq_dir)

    # ============ 关键：先做 SAPIEN 初始化 (一次) ============
    # 这样后面 render_by_sapien 就会复用同一个 scene/viewer
    init_sapien_if_needed(headless)

    for rgb_sequence_path in sorted(valid_folders):
        print(f"\n=== Processing folder: {rgb_sequence_path} ===")
        retarget_folder(retargeting, rgb_sequence_path, str(config_path), headless)


def main(
    base_dir: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    headless: bool = True
):
    """
    用法示例:
      python3 retarget_epic_kitchens.py \
        --base-dir /coc/flash7/yliu3735/datasets/EpicKitchens/EPIC-KITCHENS \
        --robot-name allegro \
        --retargeting-type dexpilot \
        --hand-type right \
        --headless
    """
    process_all(base_dir, robot_name, retargeting_type, hand_type, headless)


if __name__ == "__main__":
    tyro.cli(main)
