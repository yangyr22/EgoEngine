import os
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from mediapipe.python.solution_base import SolutionBase
from torchvision.transforms import transforms, Normalize
import sys
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fm_vis"))

sys.path.append(base_dir)


from hand_detector.mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from .hand_mode_detector import SingleHandDetector,HandMocap
import os, sys, shutil
import os.path as osp
from fm_vis.hand_detector.mocap_utils import general_utils as gnu
from fm_vis.hand_detector.mocap_utils import demo_utils as demo_utils
import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time
# from renderer.screen_free_visualizer import Visualizer
from renderer.visualizer import Visualizer
from utils.offline_dataset import visualize_3d_traj_video, visualize_mano_3d_traj_video, visualize_2d_traj_video
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

class HandCapture:
    def __init__(self, 
            hand_type, 
            input_dir, 
            fx = 322.470, 
            fy = 322.470, 
            cx = 321.048, 
            cy = 177.549, 
            dist_coeffs = np.zeros(5), 
            visualize = False,
            save_visual_result = None
        ):
        self.hand_type = hand_type
        self.input_dir = input_dir
        self.fx = fx  # Focal length in x
        self.fy = fy # Focal length in y
        self.cx = cx  # Principal point x
        self.cy = cy  # Principal point y
        self.dist_coeffs = dist_coeffs  # Assuming no distortion; replace if distortion coefficients are known
        self.output_dict = {
            'rvec': np.zeros((0, 3)),  # (0, 3)
            'tvec': np.zeros((0, 3)),  # (0, 3)
            'pred_3d_joints': np.zeros((0, 21, 3))  # (0, 21, 3)
        }
        self.visualize = visualize
        self.save_result = save_visual_result
        self.discard_nums = 35
    
    def PnP(self, object_points, image_points):
        camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, camera_matrix, self.dist_coeffs
        )
        return rvec, tvec
    def rotation_matrix_to_quaternion(self, R):
        """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
        trace = np.trace(R)
        
        if trace > 0:
            w = np.sqrt(1.0 + trace) / 2
            x = (R[2, 1] - R[1, 2]) / (4 * w)
            y = (R[0, 2] - R[2, 0]) / (4 * w)
            z = (R[1, 0] - R[0, 1]) / (4 * w)
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                x = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
                w = (R[2, 1] - R[1, 2]) / (4 * x)
                y = (R[0, 1] + R[1, 0]) / (4 * x)
                z = (R[0, 2] + R[2, 0]) / (4 * x)
            elif R[1, 1] > R[2, 2]:
                y = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) / 2
                w = (R[0, 2] - R[2, 0]) / (4 * y)
                x = (R[0, 1] + R[1, 0]) / (4 * y)
                z = (R[1, 2] + R[2, 1]) / (4 * y)
            else:
                z = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) / 2
                w = (R[1, 0] - R[0, 1]) / (4 * z)
                x = (R[0, 2] + R[2, 0]) / (4 * z)
                y = (R[1, 2] + R[2, 1]) / (4 * z)

        return np.array([w, x, y, z])

    def interpolate_joints(self, data):
        """ Interpolates missing values in a (T, N) numpy array. """
        data = np.array(data)
        valid = ~np.isnan(data).any(axis=1)  # Valid frames (not all NaN)
        indices = np.arange(len(data))

        if valid.any():
            for i in range(data.shape[1]):  # Interpolate each column separately
                data[:, i] = np.interp(indices, indices[valid], data[valid, i])
        return data
    
    def interpolate_tvec(self, data):
            """Interpolates missing values in a (T, N) numpy array."""
            data = np.array(data)
            valid = ~np.isnan(data).any(axis=1)  # Valid frames (not all NaN)
            indices = np.arange(len(data))

            if valid.any():
                for i in range(data.shape[1]):  # Interpolate each column separately
                    nan_indices = np.isnan(data[:, i])

                    # If there are missing frames at the start, use quadratic estimation
                    first_valid_indices = indices[valid][:3]  # Take first 3 valid points
                    first_valid_values = data[valid, i][:3]

                    if len(first_valid_indices) >= 3 and nan_indices[0]:
                        poly_coeffs = np.polyfit(first_valid_indices, first_valid_values, 2)
                        poly_func = np.poly1d(poly_coeffs)
                        missing_start = np.where(nan_indices & (indices < first_valid_indices[0]))[0]
                        data[missing_start, i] = poly_func(missing_start)

                    # If there are missing frames at the end, use extrapolation
                    last_valid_indices = indices[valid][-3:]  # Take last 3 valid points
                    last_valid_values = data[valid, i][-3:]

                    if len(last_valid_indices) >= 3 and nan_indices[-1]:
                        poly_coeffs = np.polyfit(last_valid_indices, last_valid_values, 1)  # Linear extrapolation
                        poly_func = np.poly1d(poly_coeffs)
                        missing_end = np.where(nan_indices & (indices > last_valid_indices[-1]))[0]
                        data[missing_end, i] = poly_func(missing_end)

                    # Linear interpolation for missing values in the middle
                    middle_valid = valid & ~np.isnan(data[:, i])  # Ensure middle valid values
                    data[:, i] = np.interp(indices, indices[middle_valid], data[middle_valid, i])

            return data

    def interpolate_slerp(self, rvec):
        """ Uses SLERP for smooth rotation interpolation. """
        rvec = np.array(rvec)
        valid = ~np.isnan(rvec).any(axis=1)

        if valid.sum() < 2:
            return rvec  # Not enough data to interpolate

        # Convert rvec to quaternions
        rotations = R.from_rotvec(rvec[valid])
        quaternions = rotations.as_quat()

        # SLERP interpolation
        indices = np.arange(len(rvec))
        interp_func = R.from_quat(interp1d(indices[valid], quaternions, kind="linear", axis=0, fill_value="extrapolate")(indices))
        return interp_func.as_rotvec()

    def interpolate_hand_data(self, hand_data):
        """ Interpolates missing frames in rvec, tvec, and pred_3d_joints. """
        hand_data["rvec"] = self.interpolate_slerp(hand_data["rvec"])
        hand_data["tvec"] = self.interpolate_tvec(hand_data["tvec"])

        # Interpolating each joint in (T,21,3)
        pred_3d_joints = np.array(hand_data["pred_3d_joints"])
        # pred_3d_joints = self.remove_trailing_nans(pred_3d_joints) 
        for joint in range(pred_3d_joints.shape[1]):  # 21 joints
            pred_3d_joints[:, joint, :] = self.interpolate_joints(pred_3d_joints[:, joint, :])

        hand_data["pred_3d_joints"] = pred_3d_joints
        return hand_data
    
    def compute_relative_distances(self, pred_3d_joints):
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
        """Computes the relative distances between connected joint points in each frame."""
        pred_3d_joints = np.array(pred_3d_joints)  # Ensure it's a NumPy array
        T = pred_3d_joints.shape[0]  # Number of frames
        distances = np.zeros((T, len(connections)))  # Store distances per frame
        
        for t in range(T):
            if np.isnan(pred_3d_joints[t]).all():  # Skip frames where all values are NaN
                continue
            
            for idx, (joint1, joint2) in enumerate(connections):
                diff = pred_3d_joints[t, joint1, :] - pred_3d_joints[t, joint2, :]
                distances[t, idx] = np.linalg.norm(diff)  # Compute Euclidean distance
        
        return distances

    def hand_reconstruction(self):
        video_path = self.input_dir
        if not os.path.exists(video_path):
            print(f"Error: The video file '{video_path}' does not exist.")
            sys.exit(1)
        
        hand_detector_dir = Path(__file__).parent
        default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
        default_checkpoint_body_smpl = './extra_data/smpl'
        
        # Initialize the detector with the specified hand type
        detector = SingleHandDetector(hand_type=self.hand_type.capitalize())  # Capitalize to match expected input ('Left' or 'Right')
        
        hand_mocap = HandMocap(str(hand_detector_dir / default_checkpoint_hand),
                            str(hand_detector_dir / default_checkpoint_body_smpl))

        vid = cv2.VideoCapture(video_path)
        frame_idx = 0
        # Variables to store the last valid frame's results
        last_rvec, last_tvec, last_pred_3d_joints = None, None, None
        detect = False
        no_res_count = 0

        if self.save_result is not None:
            os.makedirs(self.save_result, exist_ok=True)
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
            # out = cv2.VideoWriter(os.path.join(self.save_result, "hand_construction.mp4"), fourcc, 30, (1280, 480))
            skeleton_dir_ori = os.path.join(self.save_result, "hand_skeleton_before.mp4")
            motion_dir_ori = os.path.join(self.save_result, "hand_motion_3d_before.mp4")
            motion_2d_dir_ori = os.path.join(self.save_result, "hand_motion_2d_before.mp4")

            skeleton_dir = os.path.join(self.save_result, "hand_skeleton.mp4")
            motion_dir = os.path.join(self.save_result, "hand_motion_3d.mp4")
            motion_2d_dir = os.path.join(self.save_result, "hand_motion_2d.mp4")
        while True:
            ret, image_bgr = vid.read()
            if not ret:
                break  # Exit if video ends
            
            _, bbox = detector.detect_hand_bbox(image_bgr)
            hand_bbox_list = [{"left_hand": None, "right_hand": None}]

            if bbox is not None:
                detect = True
                if self.hand_type == 'left':
                    hand_bbox_list[0]["left_hand"] = bbox[0]
                elif self.hand_type == 'right':
                    hand_bbox_list[0]["right_hand"] = bbox[0]
                
                original_output = hand_mocap.regress_original(image_bgr, hand_bbox_list, add_margin=False)
                
                pred_output = original_output[0]

                if self.hand_type == "right":
                    rvec, tvec = self.PnP(pred_output["right_hand"]['pred_joints_original'], pred_output["right_hand"]["pred_joints_img"][:, :2])
                    object_points = pred_output["right_hand"]['pred_joints_original']
                else:
                    rvec, tvec = self.PnP(pred_output["left_hand"]['pred_joints_original'], pred_output["left_hand"]["pred_joints_img"][:, :2])
                    object_points = pred_output["left_hand"]['pred_joints_original']

                # Store the latest valid results
                last_rvec, last_tvec, last_pred_3d_joints = rvec, tvec, object_points
                no_res_count = 0

            else:
                print(f"Warning: No hand detected in frame {frame_idx}")
                no_res_count += 1
                detect = False
                # if no_res_count >= self.discard_nums:
                #     print(f"Warning: No hand detected for consequent 30 frames, discard this video")
                #     return None

                rvec, tvec = np.full((1, 3), np.nan), np.full((1, 3), np.nan)
                object_points = np.full((1, 21, 3), np.nan)

            # Update the dictionary
            new_rvec = rvec.reshape(1, 3)  # (1, 3)
            new_tvec = tvec.reshape(1, 3)  # (1, 3)
            new_pred_3d_joints = object_points.reshape(1, 21, 3)  # (1, 21, 3)

            self.output_dict['rvec'] = np.vstack((self.output_dict['rvec'], new_rvec))
            self.output_dict['tvec'] = np.vstack((self.output_dict['tvec'], new_tvec))
            self.output_dict['pred_3d_joints'] = np.concatenate((self.output_dict['pred_3d_joints'], new_pred_3d_joints), axis=0)

            if self.visualize:
                
                pred_mesh_list = demo_utils.extract_mesh_from_output(original_output)
                visualizer = Visualizer(rendererType="opengl")
                
                res_img = visualizer.visualize(
                    image_bgr, 
                    pred_mesh_list=pred_mesh_list, 
                    hand_bbox_list=hand_bbox_list)

                res_img = res_img.astype(np.uint8)
                ImShow(res_img)

            # if self.save_result is not None:
            #     pred_mesh_list = demo_utils.extract_mesh_from_output(original_output)
            #     visualizer = Visualizer(rendererType="opengl")
                
            #     res_img = visualizer.visualize(
            #         image_bgr, 
            #         pred_mesh_list=pred_mesh_list, 
            #         hand_bbox_list=hand_bbox_list)
            #     res_img = res_img.astype(np.uint8)

            #     out.write(res_img) 

            frame_idx += 1

        # visualize_mano_3d_traj_video(self.output_dict, skeleton_dir_ori, fps=30)
        # visualize_3d_traj_video(self.output_dict, motion_dir_ori, fps=30)
        # visualize_2d_traj_video(self.output_dict, motion_2d_dir_ori, fps=30)
        # print("before", self.output_dict['rvec'].shape)

        self.output_dict = self.interpolate_hand_data(self.output_dict)


        # pred_3d = self.output_dict["pred_3d_joints"]
        # dist = self.compute_relative_distances(pred_3d)
        # print(dist)

        # print("after", self.output_dict['rvec'].shape)
        # print(frame_idx)

        if self.save_result is not None:
            # out.release()
            visualize_mano_3d_traj_video(self.output_dict, skeleton_dir, fps=15)
            visualize_3d_traj_video(self.output_dict, motion_dir, fps=15)
            visualize_2d_traj_video(self.output_dict, motion_2d_dir, fps=15)


class HandCaptureOnly4Preprocessing:
    def __init__(
        self, 
        hand_type, 
        input_dir, 
        fx=322.470, 
        fy=322.470, 
        cx=321.048, 
        cy=177.549, 
        dist_coeffs=np.zeros(5), 
        visualize=False,
        save_visual_result=None
    ):
        self.hand_type = hand_type
        self.input_dir = input_dir
        self.fx = fx  # Focal length in x
        self.fy = fy  # Focal length in y
        self.cx = cx  # Principal point x
        self.cy = cy  # Principal point y
        self.dist_coeffs = dist_coeffs  # Assuming no distortion; replace if known
        self.output_dict = {
            'rvec': np.zeros((0, 3)),          # shape=(0, 3)
            'tvec': np.zeros((0, 3)),          # shape=(0, 3)
            'pred_3d_joints': np.zeros((0,21,3)) # shape=(0,21,3)
        }
        self.visualize = visualize
        default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
        default_checkpoint_body_smpl = './extra_data/smpl'

        self.detector = SingleHandDetector(hand_type=self.hand_type.capitalize())
        self.save_result = save_visual_result
        hand_detector_dir = Path(__file__).parent
        self.hand_mocap = HandMocap(
            str(hand_detector_dir / default_checkpoint_hand),
            str(hand_detector_dir / default_checkpoint_body_smpl)
        )

    def PnP(self, object_points, image_points):
        camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, camera_matrix, self.dist_coeffs
        )
        return rvec, tvec
    
    def interpolate_joints(self, data):
        """ Interpolates missing values in a (T, N) numpy array. """
        data = np.array(data)
        valid = ~np.isnan(data).any(axis=1)  # Valid frames (not all NaN)
        indices = np.arange(len(data))

        if valid.any():
            for i in range(data.shape[1]):  # Interpolate each column separately
                data[:, i] = np.interp(indices, indices[valid], data[valid, i])
        return data
    
    def interpolate_tvec(self, data):
            """Interpolates missing values in a (T, N) numpy array."""
            data = np.array(data)
            valid = ~np.isnan(data).any(axis=1)  # Valid frames (not all NaN)
            indices = np.arange(len(data))

            if valid.any():
                for i in range(data.shape[1]):  # Interpolate each column separately
                    nan_indices = np.isnan(data[:, i])

                    # If there are missing frames at the start, use quadratic estimation
                    first_valid_indices = indices[valid][:3]  # Take first 3 valid points
                    first_valid_values = data[valid, i][:3]

                    if len(first_valid_indices) >= 3 and nan_indices[0]:
                        poly_coeffs = np.polyfit(first_valid_indices, first_valid_values, 2)
                        poly_func = np.poly1d(poly_coeffs)
                        missing_start = np.where(nan_indices & (indices < first_valid_indices[0]))[0]
                        data[missing_start, i] = poly_func(missing_start)

                    # If there are missing frames at the end, use extrapolation
                    last_valid_indices = indices[valid][-3:]  # Take last 3 valid points
                    last_valid_values = data[valid, i][-3:]

                    if len(last_valid_indices) >= 3 and nan_indices[-1]:
                        poly_coeffs = np.polyfit(last_valid_indices, last_valid_values, 1)  # Linear extrapolation
                        poly_func = np.poly1d(poly_coeffs)
                        missing_end = np.where(nan_indices & (indices > last_valid_indices[-1]))[0]
                        data[missing_end, i] = poly_func(missing_end)

                    # Linear interpolation for missing values in the middle
                    middle_valid = valid & ~np.isnan(data[:, i])  # Ensure middle valid values
                    data[:, i] = np.interp(indices, indices[middle_valid], data[middle_valid, i])

            return data

    def interpolate_slerp(self, rvec):
        """ Uses SLERP for smooth rotation interpolation. """
        rvec = np.array(rvec)
        valid = ~np.isnan(rvec).any(axis=1)

        if valid.sum() < 2:
            return rvec  # Not enough data to interpolate

        # Convert rvec to quaternions
        rotations = R.from_rotvec(rvec[valid])
        quaternions = rotations.as_quat()

        # SLERP interpolation
        indices = np.arange(len(rvec))
        interp_func = R.from_quat(interp1d(indices[valid], quaternions, kind="linear", axis=0, fill_value="extrapolate")(indices))
        return interp_func.as_rotvec()

    def interpolate_hand_data(self, hand_data):
        """ Interpolates missing frames in rvec, tvec, and pred_3d_joints. """
        hand_data["rvec"] = self.interpolate_slerp(hand_data["rvec"])
        hand_data["tvec"] = self.interpolate_tvec(hand_data["tvec"])

        # Interpolating each joint in (T,21,3)
        pred_3d_joints = np.array(hand_data["pred_3d_joints"])
        # pred_3d_joints = self.remove_trailing_nans(pred_3d_joints) 
        for joint in range(pred_3d_joints.shape[1]):  # 21 joints
            pred_3d_joints[:, joint, :] = self.interpolate_joints(pred_3d_joints[:, joint, :])

        hand_data["pred_3d_joints"] = pred_3d_joints
        return hand_data

    def hand_reconstruction(self):
        video_path = self.input_dir
        if not os.path.exists(video_path):
            print(f"Error: The video file '{video_path}' does not exist.")
            sys.exit(1)

        # Load all frames
        vid = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            frames.append(frame)
        vid.release()

        if len(frames) == 0:
            print("No frames found in video.")
            return

        num_frames = len(frames)

        # Initialize output_dict with NaNs
        self.output_dict = {
            'rvec': np.full((num_frames, 3), np.nan),
            'tvec': np.full((num_frames, 3), np.nan),
            'pred_3d_joints': np.full((num_frames, 21, 3), np.nan)
        }

        # Helper function to run detection
        def detect_and_estimate(image_bgr):
            _, bbox = self.detector.detect_hand_bbox(image_bgr)
            if bbox is None:
                return None, None, None

            hand_bbox_list = [{"left_hand": None, "right_hand": None}]
            hand_bbox_list[0][f"{self.hand_type}_hand"] = bbox[0]
            original_output = self.hand_mocap.regress_original(image_bgr, hand_bbox_list, add_margin=False)
            pred_output = original_output[0]

            if self.hand_type == "right":
                joints_3d = pred_output["right_hand"]['pred_joints_original']
                joints_2d = pred_output["right_hand"]["pred_joints_img"][:, :2]
            else:
                joints_3d = pred_output["left_hand"]['pred_joints_original']
                joints_2d = pred_output["left_hand"]["pred_joints_img"][:, :2]

            rvec, tvec = self.PnP(joints_3d, joints_2d)
            return rvec, tvec, joints_3d

        # Step 1: Reverse pass (fill initially detected frames)
        for idx_rev in reversed(range(num_frames)):
            image_bgr = frames[idx_rev]
            result = detect_and_estimate(image_bgr)
            if result[0] is not None:
                rvec, tvec, joints_3d = result
                self.output_dict['rvec'][idx_rev] = rvec.reshape(3)
                self.output_dict['tvec'][idx_rev] = tvec.reshape(3)
                self.output_dict['pred_3d_joints'][idx_rev] = joints_3d

        # Step 2: Forward pass (overwrite if better results appear)
        for idx_fwd, image_bgr in enumerate(frames):
            result = detect_and_estimate(image_bgr)
            if result[0] is not None:
                rvec, tvec, joints_3d = result
                self.output_dict['rvec'][idx_fwd] = rvec.reshape(3)
                self.output_dict['tvec'][idx_fwd] = tvec.reshape(3)
                self.output_dict['pred_3d_joints'][idx_fwd] = joints_3d

        # After two passes, check for frames without any result
        invalid_frames = np.isnan(self.output_dict['rvec']).any(axis=1)
        if invalid_frames.any():
            first_invalid_frame = np.where(invalid_frames)[0]
            print(f"Error: No valid hand detection at frame {invalid_frames} in both passes.")
            if len(invalid_frames) >= 30:
                print("Error: Too many consecutive invalid frames. Exiting.")
                return None

        # Interpolate remaining missing values (if needed)
        self.output_dict = self.interpolate_hand_data(self.output_dict)

        # Optionally visualize/save the results
        if self.save_result is not None:
            os.makedirs(self.save_result, exist_ok=True)
            skeleton_dir = os.path.join(self.save_result, "hand_skeleton.mp4")
            motion_dir = os.path.join(self.save_result, "hand_motion_3d.mp4")
            motion_2d_dir = os.path.join(self.save_result, "hand_motion_2d.mp4")

            visualize_mano_3d_traj_video(self.output_dict, skeleton_dir, fps=30)
            visualize_3d_traj_video(self.output_dict, motion_dir, fps=30)
            visualize_2d_traj_video(self.output_dict, motion_2d_dir, fps=30)

        print(f"Done. Total frames processed: {num_frames}")

    
class OnlineHandCapture:
    def __init__(
        self,
        hand_type,
        fx = 322.470, 
        fy = 322.470, 
        cx = 321.048, 
        cy = 240, 
        dist_coeffs=None,
        visualize=False,
        max_no_detection_count=0,
        depth_enabled=True
    ):
        """
        A class to capture and estimate hand pose on a per-frame basis.
        
        Args:
            hand_type (str): 'left' or 'right' hand to detect.
            fx (float): Focal length along the x-axis.
            fy (float): Focal length along the y-axis.
            cx (float): Principal point (x).
            cy (float): Principal point (y).
            dist_coeffs (ndarray): Camera distortion coefficients.
            visualize (bool): If True, visualize via OpenGL or any chosen renderer.
            max_no_detection_count (int): Maximum consecutive frames without detection 
                                          before stopping or skipping.
        """
        self.hand_type = hand_type
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.visualize = visualize
        self.max_no_detection_count = max_no_detection_count
        
        # Prepare camera matrix once
        self.camera_matrix = np.array([
            [self.fx, 0,      self.cx],
            [0,      self.fy, self.cy],
            [0,      0,       1     ],
        ], dtype=np.float32)
        
        # Prepare hand detector & MoCap
        hand_detector_dir = Path(__file__).parent
        default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
        default_checkpoint_body_smpl = "./extra_data/smpl"

        # Initialize the hand detector (bbox)
        # e.g., SingleHandDetector expects 'Left' or 'Right' capitalized
        from .hand_mode_detector import SingleHandDetector, HandMocap
        self.detector = SingleHandDetector(hand_type=self.hand_type.capitalize())

        # Initialize the regression model
        self.hand_mocap = HandMocap(
            str(hand_detector_dir / default_checkpoint_hand),
            str(hand_detector_dir / default_checkpoint_body_smpl),
        )

        # We store the "last valid" detection results here
        self.last_rvec = None
        self.last_tvec = None
        self.last_pred_joints_3d = None  # shape (21, 3)

        # Count consecutive frames without valid detection
        self.no_res_count = 0
        self.detected_once = False

        # Renderer setup if visualize == True
        if self.visualize:
            self.Visualizer = Visualizer
            self.ImShow = ImShow
        else:
            self.Visualizer = None
            self.ImShow = None
        
        self.depth_enabled = depth_enabled

    def solve_PnP(self, object_points, image_points):
        """
        Solve PnP to find rvec and tvec from matched 3D-2D correspondences.
        """
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        # If needed, check 'success' or 'inliers' for quality of fit
        return rvec, tvec

    def compute_tvec_from_depth(self, object_points, depth_map, bbox):
        """
        Computes tvec using depth data (3D-3D mapping).
        
        Args:
            object_points (np.ndarray): (21,3) 3D predicted joints from the model.
            depth_map (np.ndarray): Depth image from the camera.
            bbox (tuple): Bounding box (x, y, w, h).

        Returns:
            np.ndarray: Estimated tvec (3,).
        """
        x, y, w, h = bbox
        bbox_center_x = int(x + w / 2)
        bbox_center_y = int(y + h / 2)

        # Fetch depth value from depth map
        depth_value = depth_map[bbox_center_y, bbox_center_x]

        if depth_value <= 0:  # Invalid depth fallback
            return np.zeros(3)  # Default fallback

        # Convert to meters
        depth_value = depth_value / 1000.0  # Assuming depth in millimeters

        # Convert image space to real-world coordinates using intrinsic parameters
        tvec_x = (bbox_center_x - self.cx) * depth_value / self.fx
        tvec_y = (bbox_center_y - self.cy) * depth_value / self.fy
        tvec_z = depth_value

        return np.array([tvec_x, tvec_y, tvec_z])

    def hand_reconstruction_frame(self, frame, depth_map=None):
        """
        Process a single BGR frame to detect hand and estimate 3D joints, rvec, and tvec.
        
        Returns:
            dict or None:
                {
                  'rvec':  (3,) or None
                  'tvec':  (3,) or None
                  'pred_3d_joints': (21, 3) or None
                  'bbox':  (4,) bounding box if detected or None
                  'used_previous_result': bool
                }
            Returns None if no detection available and no previous result to fall back on.
        """
        # Detect hand bounding box
        _, bbox = self.detector.detect_hand_bbox(frame)
        hand_bbox_list = [{"left_hand": None, "right_hand": None}]

        if bbox is not None:
            self.detected_once = True
            self.no_res_count = 0

            # Assign bbox to correct hand
            hand_bbox_list[0][f"{self.hand_type}_hand"] = bbox[0]

            # Regress 3D joints
            mocap_output = self.hand_mocap.regress_original(frame, hand_bbox_list, add_margin=False)
            pred_output = mocap_output[0]

            # Extract hand keypoints
            hand_key = f"{self.hand_type}_hand"
            object_points = pred_output[hand_key]['pred_joints_original']

            # Use 3D-3D mapping if depth is enabled
            if self.depth_enabled and depth_map is not None:
                tvec = self.compute_tvec_from_depth(object_points, depth_map, bbox[0])
                rvec = pred_output[hand_key]['pred_hand_rotation']
            else:
                image_points = pred_output[hand_key]["pred_joints_img"][:, :2]
                rvec, tvec = self.solve_pnp(object_points, image_points)

            # Cache results for fallback
            self.last_rvec, self.last_tvec = rvec.copy(), tvec.copy()
            self.last_pred_joints_3d = object_points.copy()
            used_previous_result = False
        else:
            # If no detection, use last known good results
            if self.detected_once and self.last_rvec is not None and self.last_tvec is not None:
                self.no_res_count += 1
                if self.no_res_count > self.max_no_detection_count:
                    return None  # Too many missed frames, return nothing

                # Fallback to previous valid detection
                rvec, tvec = self.last_rvec, self.last_tvec
                object_points = self.last_pred_joints_3d
                used_previous_result = True
            else:
                return None  # No previous result available

        # Visualization
        if self.visualize:
            pred_mesh_list = self.demo_utils.extract_mesh_from_output(mocap_output) if bbox else []
            res_img = self.visualizer.visualize(frame, pred_mesh_list=pred_mesh_list, hand_bbox_list=hand_bbox_list)
            self.ImShow(res_img.astype(np.uint8))

        # Return results
        return {
            'rvec': rvec.reshape(-1), 
            'tvec': tvec.reshape(-1), 
            'pred_3d_joints': object_points,
            'bbox': bbox[0] if bbox is not None else None,  
            'used_previous_result': used_previous_result
        }
    
def main():
    ### this is an example of calling the class
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hand Reconstruction Test")
    parser.add_argument("--hand_type", type=str, required=True, choices=["left", "right"], help="Type of hand to process (left or right)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input video file")
    args = parser.parse_args()

    # Initialize HandCapture
    hand_capture = HandCapture(hand_type=args.hand_type, input_dir=args.input_dir, visualize=True)

    # Perform hand reconstruction
    hand_capture.hand_reconstruction()

    # Print results
    print(hand_capture.output_dict) # this is the result for the pipeline
    print("rvec:", hand_capture.output_dict['rvec'].shape)
    print("tvec:", hand_capture.output_dict['tvec'].shape)
    print("pred_3d_joints:", hand_capture.output_dict['pred_3d_joints'].shape)

if __name__ == "__main__":
    main()                



    



