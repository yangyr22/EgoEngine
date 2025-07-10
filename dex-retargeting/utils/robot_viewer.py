import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import sapien
from tqdm import trange
from pytransform3d import rotations

from hand_viewer import SingleHandOfflineRetargetingSAPIENViewer, SingleHandOnlineRetargetingSAPIENViewer
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

from scipy.spatial.transform import Rotation as R

class Trans:
    def __init__(self):
        """
        Initialize the transformation class with predefined rotation (R_opt) and translation (t_opt)
        from the camera frame to the robot frame.
        """
        self.R_opt = np.array([
            [ 0.901696 ,   0.39898718, -0.16659397],
            [ 0.38820481, -0.57743289 , 0.71823972],
            [ 0.1903716 , -0.71230646, -0.67555767]
        ])
        self.t_opt = np.array([0.84769182, -0.47822804,  0.47235281])
        # self.t_opt = np.array([0.74769182, -0.30822804,  0.57235281])

        # Convert R_opt to a Rotation object for Euler angle transformations
        self.R_robot = R.from_matrix(self.R_opt)

    def transform_tvec(self, tvec):
        """
        Transform a 3D position (XYZ) from the camera frame to the robot frame.

        Args:
            tvec (list): 3D position in the camera frame [x, y, z].

        Returns:
            list: Transformed 3D position in the robot frame [x', y', z'].
        """
        # tvec = np.array(tvec)  # Convert to NumPy array for matrix operations
        transformed_tvec = self.R_opt @ tvec + self.t_opt
        return transformed_tvec # .tolist()  # Convert back to list

    def transform_rvec(self, rvec):
        """
        Transform a 3D orientation (Euler angles) from the camera frame to the robot frame.

        Args:
            rvec (list): Euler angles in the camera frame [roll, pitch, yaw] (xyz order, radians).

        Returns:
            list: Transformed Euler angles in the robot frame [roll', pitch', yaw'].
        """
        # rvec = np.array(rvec)  # Convert to NumPy array
        R_camera = R.from_euler('xyz', rvec, degrees=False)

        # Combine rotations
        R_combined = self.R_robot * R_camera

        # Convert back to Euler angles and return as a list
        return R_combined.as_euler('xyz', degrees=False) # .tolist()


class RobotOfflineRetargetingSAPIENViewer(SingleHandOfflineRetargetingSAPIENViewer):
    def __init__(
        self,
        robot_names: List[str],
        hand_type: str,
        headless: bool = True,
        use_ray_tracing: bool = False,
        visualize: bool = True,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing, visualize=False) # visualize)

        self.trans = Trans()
        
        # 1) Define the joints to preserve for each robot
        self.preserve_robot_joint = {
            RobotName["robotiq"]: [7],
            RobotName["umi"]: [7],
            RobotName["leap"]: [6, 7, 8, 9,  14, 15, 16, 17,  18, 19, 20, 21,  10, 11, 12, 13],
            RobotName["allegro"]: [6, 7, 8, 9,  14, 15, 16, 17,  18, 19, 20, 21,  10, 11, 12, 13],
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
    
    @staticmethod
    def estimate_hand_rvec(hand_pose: np.ndarray) -> np.ndarray:
        """
        估计手部的旋转（以 Euler angles 表示，单位：弧度）。
        方法：利用手腕、食指 MCP 和中指 MCP 构造局部坐标系。
        
        Args:
            hand_pose: np.ndarray, shape [T, 21, 3]，手部关键点坐标（例如 MediaPipe 格式），
                    其中 keypoint 0 为手腕keypoint 5 为食指 MCP，keypoint 9 为中指 MCP。
        
        Returns:
            np.ndarray, shape [T, 3]，对应的 Euler angles（roll, pitch, yaw），单位为弧度。
        """
        T = hand_pose.shape[0]
        rvecs = []
        
        for i in range(T):
            wrist = hand_pose[i, 0]
            index_mcp = hand_pose[i, 5]
            middle_mcp = hand_pose[i, 9]
            
            # z轴：从手腕指向中指 MCP
            z_axis = middle_mcp - wrist
            norm = np.linalg.norm(z_axis)
            if norm < 1e-6:
                z_axis = np.array([0, 0, 1])
            else:
                z_axis /= norm
            
            # x轴：从手腕指向食指 MCP，去除 z 分量
            x_temp = index_mcp - wrist
            proj = np.dot(x_temp, z_axis) * z_axis
            x_axis = x_temp - proj
            norm = np.linalg.norm(x_axis)
            if norm < 1e-6:
                x_axis = np.array([1, 0, 0])
            else:
                x_axis /= norm
            
            # y轴：叉乘保证右手系
            y_axis = np.cross(z_axis, x_axis)
            
            R_mat = np.column_stack((x_axis, y_axis, z_axis))
            euler_angles = R.from_matrix(R_mat).as_euler('xyz', degrees=False)
            rvecs.append(euler_angles)
        
        return np.array(rvecs)


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
        rotation = self.estimate_hand_rvec(hand_pose)             # shape: [T, 3]
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

        hand_pose_start = hand_pose[0]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(np.array([2.3256, 0.1674, 0.4166]))
        for robot, retargeting, retarget2sapien in zip(self.robots, self.retargetings, self.retarget2sapien):
            retargeting.warm_start(
                translation[0],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        # Process each frame
        for i in range(num_frame):
            joint_3d = hand_pose[i] + translation[i]
            for robot, retargeting, retarget2sapien, robot_name in zip(
                self.robots, self.retargetings, self.retarget2sapien, self.robot_names
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint_3d[indices, :]  # shape: [8, 3]
                qpos = retargeting.retarget(ref_value)
                
                if robot_name == RobotName.umi or robot_name == RobotName.robotiq:
                    dist = np.linalg.norm(joint_3d[4] - joint_3d[8])
                    if dist <= 0.04:
                        qpos[7] = 0.0
                    elif dist >= 0.06:
                        qpos[7] = 0.041
                    else:
                        qpos[7] = (dist - 0.04) / (0.06 - 0.04) * 0.041
                
                if self.visualize:
                    qpos_full = qpos[retarget2sapien]  # [num_joints]

                # Filter qpos using the precomputed preserve_idx                
                preserve_idx = self.robot_preserve_idx[robot_name]

                filtered_qpos = qpos[preserve_idx]

                # print(qpos[:3]-translation[i])
                retargeted_translation = self.trans.transform_tvec(qpos[:3])
                retargeted_rotation = self.trans.transform_rvec(rotation[i])

                full_qpos_entry = np.concatenate([retargeted_translation, retargeted_rotation, filtered_qpos])
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
            RobotName["robotiq"]: [7],
            RobotName["umi"]: [7],
            RobotName["leap"]: [6, 7, 8, 9,  14, 15, 16, 17,  18, 19, 20, 21,  10, 11, 12, 13],
            RobotName["allegro"]: [6, 7, 8, 9,  14, 15, 16, 17,  18, 19, 20, 21,  10, 11, 12, 13],
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
        joint_3d = hand_pose + translation
        wrist_quat = rotations.quaternion_from_compact_axis_angle(rotation)
        for robot, retargeting, retarget2sapien, robot_name in zip(
            self.robots, self.retargetings, self.retarget2sapien, self.robot_names
        ):

            indices = retargeting.optimizer.target_link_human_indices
            ref_value = joint_3d[indices, :]  # shape: (8, 3)
            qpos = retargeting.retarget(ref_value)
            if robot_name == RobotName.umi or robot_name == RobotName.robotiq:
                dist = np.linalg.norm(joint_3d[4] - joint_3d[8])
                if dist <= 0.04:
                    qpos[7] = 0.0
                elif dist >= 0.06:
                    qpos[7] = 0.041
                else:
                    qpos[7] = (dist - 0.04) / (0.06 - 0.04) * 0.041
            # Filter qpos using the precomputed preserve_idx
            preserve_idx = self.robot_preserve_idx[robot_name]
            filtered_qpos = qpos[preserve_idx]
            
            # Save qpos to the history for this frame
            full_qpos_entry = np.concatenate([translation, rotation, filtered_qpos])
            self.qpos_history[robot_name].append(full_qpos_entry)
            
            if self.visualize:
                # Set robot qpos for rendering
                qpos_full = qpos[retarget2sapien]
                robot.set_qpos(qpos_full)
            else:
                return filtered_qpos

        # Render the scene
        if self.visualize:
            self.scene.update_render()
            self.viewer.render()