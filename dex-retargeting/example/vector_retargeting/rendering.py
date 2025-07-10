import subprocess
from pathlib import Path
from typing import Optional, List, Union, Dict

import cv2
import numpy as np
import sapien
import tqdm
import tyro
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.retargeting_config import RetargetingConfig


def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):
    """
    在此函数中，我们先使用 OpenCV 将帧写到 temp_output.mp4，
    然后再调用 FFmpeg 转码为 libx264 + yuv420p (即 final mp4) 并删除临时文件。
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

    scene = sapien.Scene()
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
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

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
    temp_video_file = None
    writer = None
    if record_video:
        # 1) 创建临时文件路径，注意用 with_suffix(".mp4") 以保证是 mp4 后缀
        output_path_obj = Path(output_video_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        temp_video_file = output_path_obj.with_name(output_path_obj.stem + "_temp.mp4")

        # 2) 用 OpenCV 写到临时文件, 编码器先用 "mp4v" (或你想用的)
        writer = cv2.VideoWriter(
            str(temp_video_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            100.0,  # fps
            (cam.get_width(), cam.get_height())
        )

    # -------- 加载机器人，与原先逻辑相同 --------
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

    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    # -------- 渲染 & 写帧 --------
    for qpos in tqdm.tqdm(data):
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        if not headless:
            for _ in range(2):
                viewer.render()

        if record_video and writer is not None:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

    # -------- 收尾，释放资源 & 调用 FFmpeg 转码 --------
    if record_video and writer is not None:
        writer.release()

        # 用 ffmpeg 将临时文件转为 H.264 + yuv420p，输出到 output_video_path
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(temp_video_file),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",  # 覆盖输出文件
            str(output_video_path)
        ]
        print(f"Converting to H.264 (yuv420p): {temp_video_file} -> {output_video_path}")
        subprocess.run(ffmpeg_cmd, check=True)

        # 删除临时文件
        temp_video_file.unlink()

    scene = None


def main(
    pickle_path: str,
    headless: bool = False,
    robot_name: str = "allegro",
):
    """
    遍历 pickle_path 目录下所有 .pickle 文件，对每个文件生成 <stem>_<robot_name>.mp4；
    其中 mp4 会在内部先用 OpenCV 写到一个 temp.mp4，再用 FFmpeg 转码为 H.264 (yuv420p)。
    """
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    pkl_folder = Path(pickle_path)
    pkl_files = sorted(pkl_folder.glob("*.pickle"))

    if not pkl_files:
        print(f"Error: No .pickle files found in {pkl_folder}.")
        return

    for pkl_file in pkl_files:
        output_video_name = pkl_file.stem + f"_{robot_name}.mp4"
        output_video_path = str(pkl_file.parent / output_video_name)

        pickle_data = np.load(pkl_file, allow_pickle=True)
        meta_data, data = pickle_data["meta_data"], pickle_data["data"]

        render_by_sapien(meta_data, data, output_video_path, headless)


if __name__ == "__main__":
    tyro.cli(main)
