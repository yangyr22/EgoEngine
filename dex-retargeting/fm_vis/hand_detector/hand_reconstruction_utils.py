import os
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from mediapipe.python.solution_base import SolutionBase
from torchvision.transforms import transforms
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

class HandCaptureMesh:
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
            out = cv2.VideoWriter(os.path.join(self.save_result, "hand_construction.mp4"), fourcc, 15, (1280, 480))
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
                            # Update the dictionary
                new_rvec = rvec.reshape(1, 3)  # (1, 3)
                new_tvec = tvec.reshape(1, 3)  # (1, 3)
                new_pred_3d_joints = object_points.reshape(1, 21, 3)  # (1, 21, 3)

                self.output_dict['rvec'] = np.vstack((self.output_dict['rvec'], new_rvec))
                self.output_dict['tvec'] = np.vstack((self.output_dict['tvec'], new_tvec))
                self.output_dict['pred_3d_joints'] = np.concatenate((self.output_dict['pred_3d_joints'], new_pred_3d_joints), axis=0)

            else:
                print(f"Warning: No hand detected in frame {frame_idx}")
                no_res_count += 1
                # if no_res_count >= self.discard_nums:
                #     print(f"Warning: No hand detected for consequent 30 frames, discard this video")
                #     return None
                if detect:
                    rvec, tvec = last_rvec, last_tvec
                    object_points = last_pred_3d_joints
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

            if self.save_result is not None and last_rvec is not None:
                pred_mesh_list = demo_utils.extract_mesh_from_output(original_output)
                visualizer = Visualizer(rendererType="opengl")
                
                res_img = visualizer.visualize(
                    image_bgr, 
                    pred_mesh_list=pred_mesh_list, 
                    hand_bbox_list=hand_bbox_list)
                res_img = res_img.astype(np.uint8)

                out.write(res_img) 

            frame_idx += 1

        # visualize_mano_3d_traj_video(self.output_dict, skeleton_dir_ori, fps=30)
        # visualize_3d_traj_video(self.output_dict, motion_dir_ori, fps=30)
        # visualize_2d_traj_video(self.output_dict, motion_2d_dir_ori, fps=30)
        # print("before", self.output_dict['rvec'].shape)

        # pred_3d = self.output_dict["pred_3d_joints"]
        # dist = self.compute_relative_distances(pred_3d)
        # print(dist)

        # print("after", self.output_dict['rvec'].shape)
        # print(frame_idx)

        if self.save_result is not None:
            out.release()
            visualize_mano_3d_traj_video(self.output_dict, skeleton_dir, fps=15)
            visualize_3d_traj_video(self.output_dict, motion_dir, fps=15)
            visualize_2d_traj_video(self.output_dict, motion_2d_dir, fps=15)

class HandCaptureRotation:
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
    def normalize(self, v):
        """ self.normalize a vector to unit length """
        return v / np.linalg.norm(v)

    def align_hand_coordinate(self, joints):
        """
        Align hand joints to a fixed coordinate system.
        
        Args:
        - joints (numpy.ndarray): (21, 3) array of 3D joint positions.
        
        Returns:
        - aligned_joints (numpy.ndarray): (21, 3) array of transformed hand joints.
        - R (numpy.ndarray): (3, 3) Rotation matrix applied.
        """

        # Step 1: Define Key Points
        wrist = joints[0]  # Wrist at (0,0,0) (Assumed given)
        middle_finger_base = joints[9]  # Base of the middle finger
        thumb_base = joints[1]  # Base of the thumb

        # Step 2: Define Axes
        y_axis = self.normalize(middle_finger_base - wrist)  # Y-axis along middle finger
        x_axis = self.normalize(thumb_base - wrist)  # X-axis along thumb direction
        z_axis = np.cross(x_axis, y_axis)  # Z-axis (perpendicular to X and Y)
        z_axis = self.normalize(z_axis)  # self.normalize Z

        x_axis = np.cross(y_axis, z_axis)  # Recompute X to ensure orthogonality
        x_axis = self.normalize(x_axis)

        # Step 3: Construct Rotation Matrix (Columns are new basis vectors)
        R = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3 Rotation matrix

        # Step 4: Apply Rotation (Inverse to remove existing rotation)
        aligned_joints = (R.T @ joints.T).T  # Rotate all joints

        return aligned_joints, R
    

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
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
            # out = cv2.VideoWriter(os.path.join(self.save_result, "hand_construction.mp4"), fourcc, 15, (1280, 480))
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
                    object_points = pred_output["right_hand"]['pred_joints_original']
                    object_points, R = self.align_hand_coordinate(object_points)
                    rvec, tvec = self.PnP(object_points, pred_output["right_hand"]["pred_joints_img"][:, :2])
                    
                    
                else:
                    rvec, tvec = self.PnP(pred_output["left_hand"]['pred_joints_original'], pred_output["left_hand"]["pred_joints_img"][:, :2])
                    object_points = pred_output["left_hand"]['pred_joints_original']

                # Store the latest valid results
                last_rvec, last_tvec, last_pred_3d_joints = rvec, tvec, object_points
                no_res_count = 0
                            # Update the dictionary
                new_rvec = rvec.reshape(1, 3)  # (1, 3)
                new_tvec = tvec.reshape(1, 3)  # (1, 3)
                new_pred_3d_joints = object_points.reshape(1, 21, 3)  # (1, 21, 3)
                print("frame:", frame_idx)
                print(new_rvec)
                print(pred_output["right_hand"]['pred_hand_rotation'])
                print("")

                self.output_dict['rvec'] = np.vstack((self.output_dict['rvec'], new_rvec))
                self.output_dict['tvec'] = np.vstack((self.output_dict['tvec'], new_tvec))
                self.output_dict['pred_3d_joints'] = np.concatenate((self.output_dict['pred_3d_joints'], new_pred_3d_joints), axis=0)

            else:
                print(f"Warning: No hand detected in frame {frame_idx}")
                no_res_count += 1
                # if no_res_count >= self.discard_nums:
                #     print(f"Warning: No hand detected for consequent 30 frames, discard this video")
                #     return None
                if detect:
                    rvec, tvec = last_rvec, last_tvec
                    object_points = last_pred_3d_joints
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



            frame_idx += 1

        # visualize_mano_3d_traj_video(self.output_dict, skeleton_dir_ori, fps=30)
        # visualize_3d_traj_video(self.output_dict, motion_dir_ori, fps=30)
        # visualize_2d_traj_video(self.output_dict, motion_2d_dir_ori, fps=30)
        # print("before", self.output_dict['rvec'].shape)

        # pred_3d = self.output_dict["pred_3d_joints"]
        # dist = self.compute_relative_distances(pred_3d)
        # print(dist)

        # print("after", self.output_dict['rvec'].shape)
        # print(frame_idx)

        if self.save_result is not None:
            # out.release()
            visualize_mano_3d_traj_video(self.output_dict, skeleton_dir, fps=30)
            visualize_3d_traj_video(self.output_dict, motion_dir, fps=30)
            visualize_2d_traj_video(self.output_dict, motion_2d_dir, fps=30)  

class HandCaptureRotationOnline:
    def __init__(
        self,
        hand_type,
        fx = 322.470, 
        fy = 322.470, 
        cx = 321.048, 
        cy = 177.549, 
        dist_coeffs=None,
        visualize=False,
        max_no_detection_count=10,
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

    def solve_PnP(self, object_points, image_points):
        """
        Solve PnP to find rvec and tvec from matched 3D-2D correspondences.
        """
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        # If needed, check 'success' or 'inliers' for quality of fit
        return rvec, tvec
    
    def normalize(self, v):
        """ self.normalize a vector to unit length """
        return v / np.linalg.norm(v)

    def align_hand_coordinate(self, joints):
        """
        Align hand joints to a fixed coordinate system.
        
        Args:
        - joints (numpy.ndarray): (21, 3) array of 3D joint positions.
        
        Returns:
        - aligned_joints (numpy.ndarray): (21, 3) array of transformed hand joints.
        - R (numpy.ndarray): (3, 3) Rotation matrix applied.
        """

        # Step 1: Define Key Points
        wrist = joints[0]  # Wrist at (0,0,0) (Assumed given)
        middle_finger_base = joints[9]  # Base of the middle finger
        thumb_base = joints[1]  # Base of the thumb

        # Step 2: Define Axes
        y_axis = self.normalize(middle_finger_base - wrist)  # Y-axis along middle finger
        x_axis = self.normalize(thumb_base - wrist)  # X-axis along thumb direction
        z_axis = np.cross(x_axis, y_axis)  # Z-axis (perpendicular to X and Y)
        z_axis = self.normalize(z_axis)  # self.normalize Z

        x_axis = np.cross(y_axis, z_axis)  # Recompute X to ensure orthogonality
        x_axis = self.normalize(x_axis)

        # Step 3: Construct Rotation Matrix (Columns are new basis vectors)
        R = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3 Rotation matrix

        # Step 4: Apply Rotation (Inverse to remove existing rotation)
        aligned_joints = (R.T @ joints.T).T  # Rotate all joints

        return aligned_joints, R

    def hand_reconstruction_frame(self, frame):
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
        # Detect bounding box
        _, bbox = self.detector.detect_hand_bbox(frame)
        hand_bbox_list = [{"left_hand": None, "right_hand": None}]

        # Decide if the bounding box is valid
        if bbox is not None:
            self.detected_once = True
            self.no_res_count = 0  # reset
            if self.hand_type.lower() == 'left':
                hand_bbox_list[0]["left_hand"] = bbox[0]
            else:
                hand_bbox_list[0]["right_hand"] = bbox[0]

            # Regress 3D original
            original_output = self.hand_mocap.regress_original(frame, hand_bbox_list, add_margin=False)
            pred_output = original_output[0]

            # For left or right hand
            if self.hand_type.lower() == "right":
                object_points = pred_output["right_hand"]['pred_joints_original']            # (21,3)
                image_points = pred_output["right_hand"]["pred_joints_img"][:, :2]          # (21,2)
            else:
                object_points = pred_output["left_hand"]['pred_joints_original']
                image_points = pred_output["left_hand"]["pred_joints_img"][:, :2]

            object_points, _ = self.align_hand_coordinate(object_points)
            # Solve PnP to get rvec & tvec
            rvec, tvec = self.solve_PnP(object_points, image_points)
            print(rvec)


            # Cache results
            self.last_rvec = rvec.copy()
            self.last_tvec = tvec.copy()
            self.last_pred_joints_3d = object_points.copy()

            used_previous_result = False
        else:
            # If no bounding box, fallback to the last known result if it exists
            if self.detected_once and self.last_rvec is not None and self.last_tvec is not None:
                self.no_res_count += 1
                if self.no_res_count > self.max_no_detection_count:
                    # If we exceed maximum no-detection frames, return None or break
                    print("[Warning] Max no-detection count exceeded. Returning None.")
                    return None

                # Use last known
                rvec = self.last_rvec
                tvec = self.last_tvec
                object_points = self.last_pred_joints_3d
                used_previous_result = True
                print("[Warning] No hand detected, using previous result.")
            else:
                # If we have never detected anything, or no last known results, return None
                print("[Warning] No detection and no previous result. Skipping.")
                return None

        # Visualization if requested
        if self.visualize:
            from fm_vis.hand_detector.mocap_utils import demo_utils
            pred_mesh_list = demo_utils.extract_mesh_from_output(original_output) if bbox is not None else []
            # Render results
            visualizer = self.Visualizer(rendererType="opengl")
            res_img = visualizer.visualize(frame, pred_mesh_list=pred_mesh_list, hand_bbox_list=hand_bbox_list)
            # Show the result in a 2D window
            res_img = res_img.astype(np.uint8)
            self.ImShow(res_img)

        # Return the dictionary for this single frame
        result = {
            'rvec': rvec.reshape(-1),                # shape (3,)
            'tvec': tvec.reshape(-1),                # shape (3,)
            'pred_3d_joints': object_points,         # shape (21, 3)
            'bbox': bbox[0] if bbox is not None else None,  # shape (4,) or None
            'used_previous_result': used_previous_result
        }
        return result        


def main():
    ### this is an example of calling the class
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hand Reconstruction Test")
    parser.add_argument("--hand_type", type=str, required=True, choices=["left", "right"], help="Type of hand to process (left or right)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output video file")
    args = parser.parse_args()

    # Initialize HandCapture
    hand_capture = HandCaptureRotation(hand_type=args.hand_type, input_dir=args.input_dir, save_visual_result=args.output_dir)

    # Perform hand reconstruction
    hand_capture.hand_reconstruction()

    # Print results
    print(hand_capture.output_dict) # this is the result for the pipeline
    print("rvec:", hand_capture.output_dict['rvec'].shape)
    print("tvec:", hand_capture.output_dict['tvec'].shape)
    print("pred_3d_joints:", hand_capture.output_dict['pred_3d_joints'].shape)

if __name__ == "__main__":
    main()     