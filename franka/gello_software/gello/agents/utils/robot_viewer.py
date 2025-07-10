import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import sapien
from tqdm import trange
from pytransform3d import rotations

from .hand_viewer import SingleHandOfflineRetargetingSAPIENViewer, SingleHandOnlineRetargetingSAPIENViewer
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RobotName,
    RetargetingType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

import os
import numpy as np
from pathlib import Path

class RobotOfflineRetargetingSAPIENViewer(SingleHandOfflineRetargetingSAPIENViewer):
    def __init__(
        self,
        robot_names: List[str],
        hand_type: str,
        headless: bool = True,
        use_ray_tracing: bool = False,
        visualize: bool = True,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing, visualize=visualize)
        
        # 1) Define the joints to preserve for each robot
        self.preserve_robot_joint = {
            RobotName["robotiq"]: [7],
            RobotName["umi"]: [7],
            RobotName["leap"]: [6, 14, 18, 10,  7, 15, 19, 11,  8, 16, 20, 12,  9, 17, 21, 13],
            RobotName["allegro"]: [6, 14, 18, 10,  7, 15, 19, 11,  8, 16, 20, 12,  9, 17, 21, 13],
            RobotName["ability"]: [6, 8, 12, 10, 15, 14],
        }
        
        self.robot_names = robot_names
        self.hand_type = hand_type

        # Initialize storage for each robot's qpos history
        self.qpos_history = {robot_name: [] for robot_name in robot_names}

        # Dictionary to store preserved joint indices for each robot
        self.robot_preserve_idx = {}

        self.robots: List[sapien.Articulation] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []

        # Create a URDF loader
        if self.visualize:
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True

        # Load each robot's URDF and calculate preserved joint indices
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            self.retargetings.append(retargeting)

            if self.visualize:
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

            else:
                self.robots.append(None)
                self.retarget2sapien.append(0)

            # 4) Calculate preserved joint indices directly from retargeting.joint_names
            preserve_list = self.preserve_robot_joint.get(robot_name, [])
            preserve_idx = preserve_list
            self.robot_preserve_idx[robot_name] = preserve_idx
            # preserve_idx = [i for i, n in enumerate(retargeting.joint_names) if n in preserve_list]
            # self.robot_preserve_idx[robot_name] = np.array(preserve_idx, dtype=int) 

    def render_retargeting_data(self, data: Dict, fps=5, y_offset=0.8, out_dir: str = None):
        """
        Retarget hand pose data to each robot, store qpos for each frame, and render if `visualize=True`.
        If `visualize=False`, only qpos is collected without rendering.

        Args:
            data (Dict): Hand pose data, including 'pred_3d_joints', 'tvec', 'rvec'.
            fps (int): Rendering frame rate.
            y_offset (float): Offset along the y-axis to separate multiple robots.
            out_dir (str): Directory to save qpos and video.
        """
        assert out_dir is not None, "out_dir must be specified to save qpos and video."
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        hand_pose = data["pred_3d_joints"]  # shape: [T, 21, 3]
        translation = data["tvec"]          # shape: [T, 3]
        rotation = data["rvec"]             # shape: [T, 3]
        num_frame = hand_pose.shape[0]

        # Visualization setup
        if self.visualize and not self.headless:
            global_y_offset = -y_offset * len(self.robots) / 2
            self.viewer.set_camera_xyz(4, global_y_offset, 1)

        writer = None
        if self.visualize:
            for i, robot in enumerate(self.robots):
                offset_pose = sapien.Pose([4, -y_offset * i, 2.0])
                robot.set_pose(offset_pose)

            step_per_frame = int(60 / fps)
            if self.headless:
                out_video = str(Path(out_dir) / "rendered_video.mp4")
                temp_video_file = Path(out_video).with_name(
                    f"{Path(out_video).stem}_temp.mp4"
                )

                width, height = self.camera.get_width(), self.camera.get_height()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), (width, height))
                # writer = cv2.VideoWriter(str(temp_video_file), fourcc, float(fps), (width, height))
        '''
        start_frame = 0
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(rotation[start_frame])
        for robot, retargeting, retarget2sapien in zip(self.robots, self.retargetings, self.retarget2sapien):
            retargeting.warm_start(
                hand_pose[start_frame] + translation[start_frame],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )
        '''
        # Process each frame
        for i in range(num_frame):
            joint_3d = hand_pose[i] # + translation[i]
            for robot, retargeting, retarget2sapien, robot_name in zip(
                self.robots, self.retargetings, self.retarget2sapien, self.robot_names
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint_3d[indices, :]  # shape: [8, 3]
                qpos = retargeting.retarget(ref_value)

                if robot_name == RobotName.umi or robot_name == RobotName.robotiq:
                    if np.linalg.norm(joint_3d[4] - joint_3d[8]) >= 0.04:
                        qpos[7] = 0.041
                    else:
                        qpos[7] = 0

                if self.visualize:
                    qpos_full = qpos[retarget2sapien]  # [num_joints]

                # Filter qpos using the precomputed preserve_idx                
                preserve_idx = self.robot_preserve_idx[robot_name]
                filtered_qpos = qpos[preserve_idx]

                full_qpos_entry = np.concatenate([translation[i], rotation[i], filtered_qpos])
                self.qpos_history[robot_name].append(full_qpos_entry)

                if self.visualize:
                    # Set robot qpos for rendering
                    robot.set_qpos(qpos_full)
            # Render the scene if visualize=True
            if self.visualize:
                self.scene.update_render()

                if self.headless and writer is not None:
                    self.camera.take_picture()
                    rgb = self.camera.get_picture("Color")[..., :3]
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    writer.write(rgb[..., ::-1])
                else:
                    for _ in range(step_per_frame):
                        self.viewer.render()

        # Save qpos history for each robot to .npy files
        for robot_name, qpos_history in self.qpos_history.items():
            save_path = Path(out_dir) / f"{robot_name}.npy"
            np.save(save_path, np.array(qpos_history))
            # print(np.array(qpos_history).shape)  # should be (T, 6 + D)
            print(f"[INFO] Saved qpos history for {robot_name} to {save_path}")

        
        
        # If headless and visualize=True, convert the video to .mp4 using ffmpeg
        if self.visualize and self.headless and writer is not None:
            writer.release()
            print("[INFO] Converting video to .mp4 with x264 codec...")

'''
class RobotOnlineRetargetingSAPIENViewer(SingleHandOnlineRetargetingSAPIENViewer):
    def __init__(
        self,
        robot_names: List[str],
        hand_type: str,
        use_ray_tracing: bool = False,
        visualize: bool = True,
    ):
        super().__init__(use_ray_tracing=use_ray_tracing, visualize=visualize)

        # 1) Define the joints to preserve for each robot
        # self.preserve_robot_joint = {
        #     RobotName["panda"]: ["panda_finger_joint1"],
        #     RobotName["leap"]: [
        #         '0', '1', '2', '3', '12', '13', '14', '15',
        #         '5', '4', '6', '7', '9', '8', '10', '11'
        #     ],
        #     RobotName["allegro"]: [
        #         'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0',
        #         'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0',
        #         'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0',
        #         'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
        #     ],
        #     RobotName["ability"]: [
        #         'index_q1', 'index_q2', 'middle_q1', 'middle_q2',
        #         'pinky_q1', 'pinky_q2', 'ring_q1', 'ring_q2',
        #         'thumb_q1', 'thumb_q2'
        #     ]
        # }
        self.preserve_robot_joint = {
            RobotName["umi"]: [7],
            RobotName["robotiq"]: [7],
            RobotName["leap"]: [6, 14, 18, 10,  7, 15, 19, 11,  8, 16, 20, 12,  9, 17, 21, 13],
            RobotName["allegro"]: [6, 14, 18, 10,  7, 15, 19, 11,  8, 16, 20, 12,  9, 17, 21, 13],
            RobotName["ability"]: [5, 8, 11, 14, 2, 5], # [5, 8, 11, 14, 3, 15]
        }
        
        self.robot_names = robot_names
        self.hand_type = hand_type

        # Initialize storage for each robot's qpos history
        self.qpos_history = {robot_name: [] for robot_name in robot_names}

        # Dictionary to store preserved joint indices for each robot
        self.robot_preserve_idx = {}

        self.robots: List[sapien.Articulation] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []

        # Create a URDF loader
        if self.visualize:
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True

        # Load each robot's URDF and calculate preserved joint indices
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            self.retargetings.append(retargeting)

            if self.visualize:
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

            else:
                self.robots.append(None)
                self.retarget2sapien.append(0)

            # 4) Calculate preserved joint indices directly from retargeting.joint_names
            preserve_list = self.preserve_robot_joint.get(robot_name, [])
            preserve_idx = preserve_list
            self.robot_preserve_idx[robot_name] = preserve_idx
            # preserve_idx = [i for i, n in enumerate(retargeting.joint_names) if n in preserve_list]
            # self.robot_preserve_idx[robot_name] = np.array(preserve_idx, dtype=int) 

    def render_retargeting_data(self, data: Dict, y_offset=0.8):
        """
        Retarget hand pose data for a single frame and render it in real-time.
        
        Args:
            data (Dict): Single frame hand pose data with:
                'pred_3d_joints' (np.ndarray): Shape (21, 3)
                'tvec' (np.ndarray): Shape (3,)
                'rvec' (np.ndarray): Shape (3,)
            y_offset (float): Offset along the y-axis to separate multiple robots.
        """
        hand_pose = data["pred_3d_joints"]  # shape: (21, 3)
        translation = data["tvec"]           # shape: (3,)
        rotation = data["rvec"]              # shape: (3,)

        # Visualization setup (only once)
        if self.visualize and not hasattr(self, 'viewer_initialized'):
            global_y_offset = -y_offset * len(self.robots) / 2
            self.viewer.set_camera_xyz(4, global_y_offset, 1)
            for i, robot in enumerate(self.robots):
                offset_pose = sapien.Pose([4, -y_offset * i, 2.0])
                robot.set_pose(offset_pose)
            self.viewer_initialized = True

        # Retarget for each robot
        joint_3d = hand_pose # + translation
        wrist_quat = rotations.quaternion_from_compact_axis_angle(rotation)
        filtered_qpos = None
        for robot, retargeting, retarget2sapien, robot_name in zip(
            self.robots, self.retargetings, self.retarget2sapien, self.robot_names
        ):

            indices = retargeting.optimizer.target_link_human_indices
            ref_value = joint_3d[indices, :]  # shape: (8, 3)
            qpos = retargeting.retarget(ref_value)
            # Filter qpos using the precomputed preserve_idx
            preserve_idx = self.robot_preserve_idx[robot_name]
            filtered_qpos = filtered_qpos[preserve_idx]

            # Save qpos to the history for this frame
            full_qpos_entry = np.concatenate([translation, rotation, filtered_qpos])
            self.qpos_history[robot_name].append(full_qpos_entry)
            
            if self.visualize:
                # Set robot qpos for rendering
                qpos_full = qpos[retarget2sapien]
                robot.set_qpos(qpos_full)
            
        # Render the scene
        if self.visualize:
            self.scene.update_render()
            self.viewer.render()
        return filtered_qpos
'''
    
class RobotOnlineRetargetingSAPIENViewer(SingleHandOnlineRetargetingSAPIENViewer):
    def __init__(
        self,
        robot_names: List[str],
        hand_type: str,
        use_ray_tracing: bool = False,
        visualize: bool = True,
    ):
        super().__init__(use_ray_tracing=use_ray_tracing, visualize=visualize)

        self.preserve_robot_joint = {
            RobotName["umi"]: [7],
            RobotName["robotiq"]: [7],
            RobotName["leap"]: [6, 14, 18, 10, 7, 15, 19, 11, 8, 16, 20, 12, 9, 17, 21, 13],
            RobotName["allegro"]: [6, 14, 18, 10, 7, 15, 19, 11,  8, 16, 20, 12, 9, 17, 21, 13],
            RobotName["ability"]: [6, 8, 12, 10, 15, 14],
        }

        self.robot_names = robot_names
        self.hand_type = hand_type

        # Initialize storage for each robot's qpos history
        self.qpos_history = {robot_name: [] for robot_name in robot_names}

        self.robot_preserve_idx = {}

        self.robots: List[sapien.Articulation] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []

        if self.visualize:
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True

        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.vector, hand_type  # Change to vector retargeting
            )
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            self.retargetings.append(retargeting)

            if self.visualize:
                urdf_path = Path(config.urdf_path)
                if "glb" not in urdf_path.stem:
                    new_name = f"{urdf_path.stem}_glb{urdf_path.suffix}"
                    urdf_path = urdf_path.with_name(new_name)

                robot_urdf = urdf.URDF.load(
                    str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
                )
                temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
                temp_path = f"{temp_dir}/{urdf_path.name}"
                robot_urdf.write_xml_file(temp_path)

                robot = loader.load(temp_path)
                self.robots.append(robot)

                # Map retargeting joints to SAPIEN joints
                sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
                retarget2sapien = np.array(
                    [retargeting.joint_names.index(n) for n in sapien_joint_names]
                ).astype(int)
                self.retarget2sapien.append(retarget2sapien)

            else:
                self.robots.append(None)
                self.retarget2sapien.append(0)

            preserve_list = self.preserve_robot_joint.get(robot_name, [])
            preserve_idx = preserve_list
            self.robot_preserve_idx[robot_name] = preserve_idx

    def render_retargeting_data(self, data: Dict, y_offset=0.8):
        """
        Retarget hand pose data for a single frame using VECTOR-based retargeting.
        
        Args:
            data (Dict): Single frame hand pose data with:
                'pred_3d_joints' (np.ndarray): Shape (21, 3)
                'tvec' (np.ndarray): Shape (3,)
                'rvec' (np.ndarray): Shape (3,)
            y_offset (float): Offset along the y-axis to separate multiple robots.
        """
        hand_pose = data["pred_3d_joints"]  # shape: (21, 3)
        translation = data["tvec"]           # shape: (3,)
        rotation = data["rvec"]              # shape: (3,)

        # Visualization setup (only once)
        if self.visualize and not hasattr(self, 'viewer_initialized'):
            global_y_offset = -y_offset * len(self.robots) / 2
            self.viewer.set_camera_xyz(4, global_y_offset, 1)
            for i, robot in enumerate(self.robots):
                offset_pose = sapien.Pose([4, -y_offset * i, 2.0])
                robot.set_pose(offset_pose)
            self.viewer_initialized = True

        joint_3d = hand_pose  # Extract hand pose
        wrist_quat = rotations.quaternion_from_compact_axis_angle(rotation)
        filtered_qpos = None

        for robot, retargeting, retarget2sapien, robot_name in zip(
            self.robots, self.retargetings, self.retarget2sapien, self.robot_names
        ):
            indices = retargeting.optimizer.target_link_human_indices

            # **VECTOR Retargeting Calculation**
            origin_indices = indices[0, :]  # Reference keypoints
            task_indices = indices[1, :]  # Keypoints being tracked
            ref_value = joint_3d[task_indices, :] - joint_3d[origin_indices, :]

            qpos = retargeting.retarget(ref_value)
            
            if robot_name == RobotName.umi or robot_name == RobotName.robotiq:
                if np.linalg.norm(joint_3d[4] - joint_3d[8]) >= 0.03:
                    qpos[7] = 0.041
                else:
                    qpos[7] = 0

            # Filter qpos using the precomputed preserve_idx
            preserve_idx = self.robot_preserve_idx[robot_name]
            filtered_qpos = qpos[preserve_idx]

            # Save qpos to the history for this frame
            full_qpos_entry = np.concatenate([translation, rotation, filtered_qpos])
            self.qpos_history[robot_name].append(full_qpos_entry)

            if self.visualize:
                qpos_full = qpos[retarget2sapien]
                robot.set_qpos(qpos_full)

        if self.visualize:
            self.scene.update_render()
            self.viewer.render()
        return filtered_qpos
