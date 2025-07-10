import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import sapien
from tqdm import trange
from pytransform3d import rotations

from hand_viewer import HandDatasetSAPIENViewer, SingleHandDatasetSAPIENViewer, SingleHandOfflineRetargetingSAPIENViewer
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RobotName,
    RetargetingType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

class RobotOfflineRetargetingSAPIENViewer(SingleHandOfflineRetargetingSAPIENViewer):

    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        headless: bool = True,
        use_ray_tracing: bool = False,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True

        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )
            # Override: add 6-dof dummy joint to make the root movable
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Process URDF path, with *_glb.urdf if needed
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                new_name = f"{urdf_path.stem}_glb{urdf_path.suffix}"
                urdf_path = urdf_path.with_name(new_name)

            # Load URDF using yourdfpy and write to a temporary file for the SAPIEN loader
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_path.name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)

            # Record the mapping relationship between retargeting joints and SAPIEN joints
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def render_retargeting_data(self, data: Dict, fps=5, y_offset=0.8, out_video: str = None):
        """
        Simplified rendering loop: only visualize the robot(s) retargeted from the provided hand_pose data.
        Hand mesh visualization is removed.
        Args:
            data: Dictionary containing 'hand_pose' (shape: [T, 21, 3]), 'tvec' (shape: [T, 3]),
                and 'rvec' (shape: [T, 3]). Only 'hand_pose' is used for retargeting.
            fps: Rendering frame rate.
            y_offset: Offset along the y-axis to separate multiple robots.
            out_video: If specified (and in headless mode), the path for saving the output video.
        """
        # 1) Initialize camera and table visualization position
        global_y_offset = -y_offset * len(self.robots) / 2

        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.3, global_y_offset, -2.]))
            self.camera.set_local_pose(local_pose)

        # 2) Get hand pose data (human gesture), translation, and wrist rotation
        hand_pose = data["pred_3d_joints"]  # shape: [T, 21, 3]
        translation = data["tvec"]     # shape: [T, 3]
        wrist_rot = data["rvec"]        # shape: [T, 3]
        num_frame = hand_pose.shape[0]

        # Assign each robot an offset pose so that they are separated along the y-axis
        for i, robot in enumerate(self.robots):
            offset_pose = sapien.Pose([0, -y_offset * i, 0])
            robot.set_pose(offset_pose)

        # 3) Skip invalid frames and find the first valid hand pose
        start_frame = 0
        for i in range(num_frame):
            joint = hand_pose[i]  # assume valid if non-zero
            if np.abs(joint).sum() > 1e-5:
                start_frame = i
                break

        # 4) If out_video is specified in headless mode, initialize VideoWriter
        writer = None
        if self.headless and out_video:
            out_path = Path(out_video).absolute()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temp_video_file = out_path.with_name(f"{out_path.stem}_temp.mp4")

            width, height = self.camera.get_width(), self.camera.get_height()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_video_file), fourcc, float(fps), (width, height))


        step_per_frame = int(60 / fps)  # In GUI mode, render multiple times per frame for smooth animation

        # 5) Warm-start retargeting for each robot using the first valid frame
        '''
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(wrist_rot[start_frame])
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        for robot, retargeting, retarget2sapien in zip(self.robots, self.retargetings, self.retarget2sapien):
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )
        '''
        
        joint = hand_pose[start_frame]
        # 6) Main loop: update robot joints and render
        for i in trange(start_frame, num_frame):
            joint = hand_pose[i] + translation[i]

            # Update robot joints
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            # Render the scene
            self.scene.update_render()

            if self.headless:
                if writer is not None:
                    self.camera.take_picture()
                    rgb = self.camera.get_picture("Color")[..., :3]
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        # 7) Finalize: release video writer and convert video if needed
        if self.headless:
            if writer is not None:
                writer.release()
                print(f"[INFO] MP4 saved to: {out_video}")
                # Use the temporary video file created earlier and define out_path from out_video
                # Assume 'video_path' was defined earlier when initializing the writer.
                subprocess.run([
                    "ffmpeg",
                    "-y",
                    "-i", str(temp_video_file),
                    "-c:v", "libx264",
                    "-crf", "23",
                    "-preset", "fast",
                    str(out_path)
                ])
                temp_video_file.unlink()
        else:
            self.viewer.paused = True
            self.viewer.render()

class RobotOnlineRetargetingSAPIENViewer(SingleHandOnlineRetargetingSAPIENViewer):
    """
    A class that extends SingleHandOnlineRetargetingSAPIENViewer to load robots 
    and retarget single-hand 3D keypoints to the robot joints in real-time.
    """

    def __init__(
        self,
        robot_names: List[str],
        hand_type: str,
        use_ray_tracing: bool = False,
    ):
        """
        Initialize the robot-based retargeting viewer.
        
        Args:
            robot_names (List[str]): Names or identifiers of target robots
            hand_type (str): For example, "right" or "left"
            use_ray_tracing (bool): Whether to enable ray tracing
        """
        # Initialize the base class (SingleHandOnlineRetargetingSAPIENViewer)
        super().__init__(use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        # Create a URDF loader for this scene
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True

        # Load and configure each robot
        for robot_name in robot_names:
            # Build retargeting
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            self.retargetings.append(retargeting)

            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)

            # Modify URDF path if "_glb" is not present
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                new_name = f"{urdf_path.stem}_glb{urdf_path.suffix}"
                urdf_path = urdf_path.with_name(new_name)

            # Load URDF using urdfpy, write to a temporary file, and let SAPIEN parse it
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_path.name}"
            robot_urdf.write_xml_file(temp_path)

            # Finally load the articulation in SAPIEN
            robot = loader.load(temp_path)
            self.robots.append(robot)

            # Determine the mapping between retargeting joints and SAPIEN joints
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien_idx = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.retarget2sapien.append(retarget2sapien_idx)

    def render_init(self, y_offset: float = 0.8):
        """
        Initialize camera and robot poses for real-time rendering.
        This places each robot at a different offset along the Y-axis.
        """
        # Set a global camera offset based on the number of robots
        global_y_offset = -y_offset * len(self.robots) / 2
        self.viewer.set_camera_xyz(1.5, global_y_offset, 1)

        # Place each robot along the Y-axis
        for i, robot in enumerate(self.robots):
            offset_pose = sapien.Pose([0, -y_offset * i, 0])
            robot.set_pose(offset_pose)

        # Optionally, add or configure more scene elements here:
        self.scene.add_ground(0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([1, -1, -1], [0.5, 0.5, 0.5], shadow=True)

    def render_retargeting_data(self, data: Dict):
        """
        Render a single frame in real-time: update robot joints from the input hand data, then draw the scene.
        
        Args:
            data (Dict): 
                - "pred_3d_joints": shape [21, 3], 3D keypoints of the hand for the current frame
                - "tvec": shape [3], translation vector
                - "rvec": shape [3], rotation vector (optional, if needed)
        """
        hand_pose = data["pred_3d_joints"]   # shape: [21, 3]
        translation = data["tvec"]           # shape: [3]
        # wrist_rot = data["rvec"]           # shape: [3] (use if needed)

        # Update each robot by retargeting hand keypoints to robot joint angles
        for robot, retargeting, retarget2sapien_idx in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            indices = retargeting.optimizer.target_link_human_indices
            ref_value = (hand_pose[indices, :] + translation).astype(np.float32)
            qpos = retargeting.retarget(ref_value)[retarget2sapien_idx]
            robot.set_qpos(qpos)

        # Step the simulation and render
        # (You could call this.scene.step() if your retargeting also needs real physics stepping)
        self.scene.update_render()
        self.viewer.render()

'''
class RobotDatasetSAPIENViewer(SingleHandDatasetSAPIENViewer):

    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        headless: bool = True,
        use_ray_tracing: bool = False,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True

        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )
            # override: 6-dof dummy joint, make the root movable
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # process URDF path, with *_glb.urdf
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                new_name = f"{urdf_path.stem}_glb{urdf_path.suffix}"
                urdf_path = urdf_path.with_name(new_name)

            # with yourdfpy to load URDF, and write to a temp file, SAPIEN loader
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_path.name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)

            # recording the mapping relationship
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        """
        Simply working on Hands
        """
        super().load_object_hand(data)


    def render_dexycb_data(
        self, data: Dict, fps=5, y_offset=0.8, out_video: str = None
    ):
        """
        Args:
        data: `sampled_data` read from DexYCB, containing the hand pose (`hand_pose`).
            (If `object_pose` is present, it will not be used here.)
        fps: Rendering frame rate.
        y_offset: Offset in the y-direction for different robots to avoid overlap during visualization.
        out_video: If specified and in headless mode, the rendered video will be saved as `out_video`.
        """
        # 1) 初始化相机和桌子的可视化位置
        global_y_offset = -y_offset * len(self.robots) / 2

        if not self.headless:
            # GUI 模式下可直接设置 viewer 的 camera
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            # headless 模式下，可以手动设定 camera pose
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        # 2) 获取手部姿态 (human gesture)
        hand_pose = data["hand_pose"]  # shape: [T, ...]
        num_frame = hand_pose.shape[0]

        # 给每个机器人一个 offset pose，让它们在y方向排开
        for i, robot in enumerate(self.robots):
            offset_pose = sapien.Pose([0, -y_offset * (i + 1), 0])
            robot.set_pose(offset_pose)

        # 3) 跳过无效帧，找到第一个可用的手部姿势
        start_frame = 0
        for i in range(num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        # 4) 如果指定 out_video，且在 headless 模式下，则初始化 VideoWriter
        writer = None
        if self.headless and out_video:
            out_path = Path(out_video).absolute()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # 先写入临时文件，再用 ffmpeg 转换
            temp_video_file = out_path.with_name(f"{out_path.stem}_temp.mp4")

            width, height = self.camera.get_width(), self.camera.get_height()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_video_file), fourcc, float(fps), (width, height))

        # 5) for robot: retargeting warm_start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(hand_pose_start[0, 0:3])
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        # 6) 主循环：逐帧更新手部和机器人关节，并渲染/录制
        step_per_frame = int(60 / fps)  # 在GUI模式下，可以多 render 几次，保证动画平滑
        for i in trange(start_frame, num_frame):
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # (a) 更新手部顶点
            self._update_hand(vertex)

            # (b) 更新机器人关节
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            # (c) 渲染
            self.scene.update_render()

            # headless 下写入视频
            if self.headless:
                if writer is not None:
                    self.camera.take_picture()
                    rgb = self.camera.get_picture("Color")[..., :3]  # shape [H, W, 3], float32 in [0,1]
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    writer.write(rgb[..., ::-1])
            else:
                # 实时GUI渲染
                for _ in range(step_per_frame):
                    self.viewer.render()

        # 7) 收尾：如果有写视频，结束后释放并转码
        if self.headless:
            if writer is not None:
                writer.release()
                print(f"[INFO] MP4 saved to: {out_video}")
                # 用 ffmpeg 转成 h264
                subprocess.run([
                    "ffmpeg",
                    "-y",
                    "-i", str(temp_video_file),
                    "-c:v", "libx264",
                    "-crf", "23",
                    "-preset", "fast",
                    str(out_path)
                ])
                temp_video_file.unlink()
        else:
            self.viewer.paused = True
            self.viewer.render()
'''