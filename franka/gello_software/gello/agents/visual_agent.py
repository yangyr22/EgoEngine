"""
Oculus Quest controller tracking developed at GTRI
"""

from pathlib import Path
from typing import Optional, Tuple, List
import sys
import os
import collections

from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
sys.path.append(os.path.abspath("./gello/agents"))

from fm_vis.hand_detector.hand_reconstruction import OnlineHandCapture
from utils.robot_viewer import RobotOnlineRetargetingSAPIENViewer
from utils.camera_reader import BGR_Reader
from scipy.spatial.transform import Rotation as R

import time
from enum import Enum
import ctypes
import threading

import numpy as np
import quaternion
import cv2

from deoxys.utils import YamlConfig, transform_utils

class Trans():
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
        tvec = np.array(tvec)  # Convert to NumPy array for matrix operations
        transformed_tvec = self.R_opt @ tvec + self.t_opt
        return transformed_tvec.tolist()  # Convert back to list

    def transform_rvec(self, rvec):
        """
        Transform a 3D orientation (Euler angles) from the camera frame to the robot frame.

        Args:
            rvec (list): Euler angles in the camera frame [roll, pitch, yaw] (xyz order, radians).

        Returns:
            list: Transformed Euler angles in the robot frame [roll', pitch', yaw'].
        """
        rvec = np.array(rvec)  # Convert to NumPy array
        R_camera = R.from_euler('xyz', rvec, degrees=False)

        # Combine rotations
        R_combined = self.R_robot * R_camera

        # Convert back to Euler angles and return as a list
        return R_combined.as_euler('xyz', degrees=False).tolist()

class Visual_Teleop_Agent():
    def __init__(self, robots: Optional[Tuple[RobotName]], robot_interface=None, debug=False, fps=30, depth=True):
        self.robot_interface = robot_interface
        self.debug = debug

        self.reader = BGR_Reader(width=640, height=480, fps=fps, visualize=True, depth=depth) # For teleoperation
        self.hand_capture = OnlineHandCapture(hand_type='right', visualize=False, depth_enabled=depth)
        
        robot_list = list(robots) if robots else []
        self.robot_type = robots[0]
        self.viewer = RobotOnlineRetargetingSAPIENViewer(
            robot_names=robot_list,
            hand_type=HandType.right,
            use_ray_tracing=False,
            visualize=False,
        )

        self.Trans = Trans()

        self.hand_offset = np.quaternion(1, 0, 0, 0)

        # Maintain a deque for storing the last 10 states
        self.state_history = collections.deque(maxlen=10)

        self.prev_controller_state = None
        self.controller_state_lock = threading.Lock()

        if self.robot_interface is not None:
            self.set_robot_transform()
        
        self.reader.start()

        self.threshold = 20
        self.tot = -1 # consequtive number of detected frames
        self.prev_hand_quat = None

    def initialize_pose(self, output_frame):
        current_pose = self.robot_interface.last_eef_pose
        self.initial_ee_pos = current_pose[:3, 3]

        translation = output_frame["tvec"]
        self.controller_offset = translation - self.initial_ee_pos  # (3)

        rvec = output_frame["rvec"]  # 3D rotation vector from hand detector
        Rot, _ = cv2.Rodrigues(np.array(rvec))  # Convert to 3x3 rotation matrix

        # Use SciPy's Rotation class for conversion
        rot_obj = R.from_matrix(Rot)
        hand_quat_list = rot_obj.as_quat()  # Returns [x, y, z, w]

        # Convert to [w, x, y, z] order for np.quaternion
        hand_quat = np.quaternion(hand_quat_list[3],
                                hand_quat_list[0],
                                hand_quat_list[1],
                                hand_quat_list[2])

        # --- Define the robot's initial rotation ---
        current_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity matrix
        rot_init_obj = R.from_matrix(current_rot)
        current_quat_list = rot_init_obj.as_quat()  # [x, y, z, w]

        # Convert to [w, x, y, z] order for np.quaternion
        robot_init_quat = np.quaternion(current_quat_list[3],
                                        current_quat_list[0],
                                        current_quat_list[1],
                                        current_quat_list[2])
        
        self.hand_offset = hand_quat.inverse() * robot_init_quat
        self.prev_hand_quat = None
        print(f'robot_init_quat: {robot_init_quat}, hand_quat: {hand_quat}')
        print(f'[initialize_pose] Hand offset computed: {self.hand_offset}')

    def get_controller_state(self):
        with self.controller_state_lock:
            frame_bgr, depth_map = self.reader.read()
            output_frame = self.hand_capture.hand_reconstruction_frame(frame_bgr, depth_map)
            
            if output_frame is not None:
                output_frame["tvec"] = self.Trans.transform_tvec(output_frame["tvec"])
                output_frame["rvec"] = self.Trans.transform_rvec(output_frame["rvec"])
                self.reader.visualizing(output_frame["image_points"])
                self.tot += 1
                if self.tot == self.threshold:
                    self.initialize_pose(output_frame)
                elif self.tot < self.threshold:
                    return self.prev_controller_state

                gripper_act = self.viewer.render_retargeting_data(output_frame) # (1) for panda (robotiq), (1) for umi, (16) for allegro, (6) for ability

                rotation = output_frame["rvec"]  # (3)
                Rot, _ = cv2.Rodrigues(np.array(rotation))  # Convert to (3x3) Rotation matrix

                r = R.from_matrix(Rot)  
                hand_quat_list = r.as_quat()  # Returns [x, y, z, w]

                # Convert to `np.quaternion` format [w, x, y, z]
                hand_quat = np.quaternion(hand_quat_list[3],  # w
                                        hand_quat_list[0],  # x
                                        hand_quat_list[1],  # y
                                        hand_quat_list[2])  # z

                # Store the previous hand quaternion
                self.prev_hand_quat = hand_quat

                # Correct quaternion multiplication
                final_quat = hand_quat * self.hand_offset  # Both should be `np.quaternion`

                # CONSTRAINT: SCALINING FACTOR
                identity_quat = np.quaternion(1, 0, 0, 0)  # Identity quaternion
                t1, t2 = 0.0, 1.0  # Define SLERP start and end times
                t_w = np.array([0.8])  # Define interpolation factor (can be a single value or array)
                t_x = np.array([1.0])
                t_y = np.array([0.5])
                t_z = np.array([0.8])

                target_ori = [quaternion.slerp(identity_quat, final_quat, t1, t2, t_w)[0].w,
                            quaternion.slerp(identity_quat, final_quat, t1, t2, t_x)[0].x * -1,
                            quaternion.slerp(identity_quat, final_quat, t1, t2, t_y)[0].y * -1,
                            quaternion.slerp(identity_quat, final_quat, t1, t2, t_z)[0].z * -1]


                # Create target_pose
                translation = output_frame["tvec"] - self.controller_offset  # (3)

                # CONSTRAINT: REMOVE GLITCH
                # This part is to make sure that the eef does not go far to some dangerous zone.
                current_pose = self.robot_interface.last_eef_pose
                eef_pos = current_pose[:3, 3]
                if np.linalg.norm(np.array(eef_pos)-np.array(translation)) > 0.2:
                    target_pose = list(current_pose) + target_ori

                # target_ori = [1,0,0,0]
                target_pose = list(translation) + target_ori

                current_pose = self.robot_interface.last_eef_pose
                current_pos = current_pose[:3, 3]
                if np.linalg.norm(np.array(current_pos) - np.array(translation)) > 0.1:
                    controller_state = self.state_history[-1]

                if self.robot_type == RobotName.umi or self.robot_type == RobotName.robotiq:
                    gripper_act = min(max(gripper_act[0], 0.001), 0.041)
                    gripper_act = [(1 - gripper_act / 0.041) * 2 -1]

                # Define the controller state
                controller_state = {
                    'target_pose': target_pose,
                    'gripper_act': gripper_act
                }

                # Store the new state while preserving the latest original state
                self.state_history.append(controller_state)

                # Compute the average of the last 10 states
                avg_controller_state = self.compute_average_state()

                self.prev_controller_state = avg_controller_state
                return avg_controller_state

            else:  # output_frame is None
                # NO HAND DETECTED THIS FRAME
                self.tot = -1
                time.sleep(0.001)
                return self.prev_controller_state

    def compute_average_state(self):
        """ Compute the average of the last 10 stored states. """
        if not self.state_history:
            return self.prev_controller_state  # Return last known state if history is empty

        # Extract all stored states
        target_poses = np.array([state["target_pose"] for state in self.state_history])
        gripper_acts = np.array([state["gripper_act"] for state in self.state_history])

        # Compute the average
        avg_target_pose = np.mean(target_poses, axis=0).tolist()
        avg_gripper_act = np.mean(gripper_acts, axis=0).tolist()

        # Preserve the newest state while returning the average
        avg_controller_state = {
            "target_pose": avg_target_pose,
            "gripper_act": avg_gripper_act
        }
        return avg_controller_state

    def reset_internal_state(self):
        self.initialize_pose = True
        self.prev_controller_state = None
        self.set_robot_transform()
        self.state_history.clear()  # Reset state history

    def set_robot_transform(self):
        current_pose = self.robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)

        self.ee_init_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
        self.ee_init_rot = np.quaternion(current_quat[3], current_quat[0], current_quat[1], current_quat[2])

        if self.prev_controller_state is None:
            target_pose = current_pos.tolist() + current_quat.tolist()
            if self.robot_type == RobotName.robotiq or self.robot_type == RobotName.umi:
                self.prev_controller_state = dict(
                    target_pose=target_pose,
                    target_pos=current_pos.tolist(),
                    target_ori=current_quat.tolist(),
                    dpos=[0, 0, 0],
                    gripper_act=[-1]
                )
            elif self.robot_type == RobotName.ability:
                self.prev_controller_state = dict(
                    target_pose=target_pose,
                    target_pos=current_pos.tolist(),
                    target_ori=current_quat.tolist(),
                    dpos=[0, 0, 0],
                    gripper_act=[0,0,0,0,0,0]
                )
            elif self.robot_type == RobotName.allegro:
                pass
            else:
                raise Exception("no such robot type")
