import projectaria_tools.core.mps as mps
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import cv2
import os
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sophus import SE3
from tqdm import tqdm
from pathlib import Path
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process hand tracking data and generate videos.')
    
    # Input paths
    parser.add_argument('--vrs_path', type=str, required=True,
                       help='Path to the input VRS file')
    
    # Output paths
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path')
    
    # Video parameters
    parser.add_argument('--width', type=int, default=None,
                       help='Output video width (default: use original)')
    
    parser.add_argument('--all_pose', action='store_true',
                       help='Process all landmark')
    
    return parser.parse_args()


def find_max(hand_tracking_results):
    max_length = 0
    current_length = 0
    max_start_idx = 0
    max_end_idx = 0
    current_start_idx = None

    # Iterate through all frames
    for i, result in enumerate(hand_tracking_results):
        # Check if both hands are present in current frame
        if result.left_hand is not None and result.right_hand is not None:
            if current_start_idx is None:
                current_start_idx = i  # Start new segment
            current_length += 1
        else:
            # Segment ended, check if it's the longest
            if current_length > max_length:
                max_length = current_length
                max_start_idx = current_start_idx
                max_end_idx = i - 1
            current_start_idx = None
            current_length = 0

    if current_length > max_length:
        max_length = current_length
        max_start_idx = current_start_idx
        max_end_idx = len(hand_tracking_results) - 1

    if max_length > 6:
        print(f"Start frame: {max_start_idx + 3}")
        print(f"End frame: {max_end_idx - 3}")
        return max_start_idx + 3, max_end_idx - 3
    else:
        print("No segments found where both hands were tracked simultaneously")
        return -1, -1

def main():
    args = parse_arguments()
    vrs_path = args.vrs_path
    hand_tracking_results_path =  os.path.join(
        os.path.dirname(vrs_path), 
        f"mps_{Path(vrs_path).stem}_vrs",
        "hand_tracking",
        "hand_tracking_results.csv"
    )
    trajectory_path =  os.path.join(
        os.path.dirname(vrs_path), 
        f"mps_{Path(vrs_path).stem}_vrs",
        "slam",
        "closed_loop_trajectory.csv"
    )
    if args.output_path is None:
        args.output_path = os.path.join(
            os.path.dirname(vrs_path), 
            f"result_{Path(vrs_path).stem}"
        )
    os.makedirs(args.output_path, exist_ok=True)
    hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
        hand_tracking_results_path
    )
    start, end = find_max(hand_tracking_results)

    provider = data_provider.create_vrs_data_provider(vrs_path)
    rgb_id = provider.get_stream_id_from_label("camera-rgb")
    num_frames = provider.get_num_data(rgb_id)
    device_calib = provider.get_device_calibration()
    rgb_calib = device_calib.get_camera_calib("camera-rgb")
    width = args.width
    focal_length = int(rgb_calib.get_focal_lengths()[0] * args.width / int(rgb_calib.get_image_size()[0]))
    dst_calib_rgb = calibration.get_linear_camera_calibration(
                width,
                width,
                focal_length,
                "pinhole",
                rgb_calib.get_transform_device_camera(),
            )
    T_device_camera = dst_calib_rgb.get_transform_device_camera()

    if args.all_pose:
        if start != -1 and end != -1:
            cropped_video = args.output_path + "cropped_video.mp4"
            all_pose_file = args.output_path + "all_pose.json"
            calib_file = args.output_path + "calib.json"
            trajectory = mps.read_closed_loop_trajectory(trajectory_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            video_writer1 = cv2.VideoWriter(cropped_video, fourcc, fps, (width, width))
            pose_data = {"frames": []}
            calib_data = {
                "dst_calib_rgb": str(dst_calib_rgb),
                "frames": []
            } 
            try:
                for frame_idx in tqdm(range(start, end - 1)):
                    device_pose = trajectory[frame_idx - 1]
                    T_world_device = device_pose.transform_world_device
                    record_rgb = provider.get_image_data_by_index(rgb_id, frame_idx)
                    rgb_img = record_rgb[0].to_numpy_array()
                    
                    rotated_img = np.rot90(rgb_img, k=-1)  # 逆时针旋转90度
                    colored_img = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR)
                    rectified_image = calibration.distort_by_calibration(colored_img, dst_calib_rgb, rgb_calib)
                    video_writer1.write(rectified_image)
                    
                    hand_tracking_result = hand_tracking_results[frame_idx - 1]
                    right_landmarks = np.array(hand_tracking_result.right_hand.landmark_positions_device)
                    right_landmarks_camera = (T_device_camera.inverse() @ right_landmarks.T).T
                    left_landmarks = np.array(hand_tracking_result.left_hand.landmark_positions_device)
                    left_landmarks_camera = (T_device_camera.inverse() @ left_landmarks.T).T

                    if hand_tracking_result.left_hand.wrist_and_palm_normal_device:
                        left_wrist_normal_device = hand_tracking_result.left_hand.wrist_and_palm_normal_device.wrist_normal_device
                        left_palm_normal_device = hand_tracking_result.left_hand.wrist_and_palm_normal_device.palm_normal_device
                        left_wrist_normal_camera = (T_device_camera.inverse() @ left_wrist_normal_device).flatten()
                        left_palm_normal_camera = (T_device_camera.inverse() @ left_palm_normal_device).flatten()

                    if hand_tracking_result.right_hand.wrist_and_palm_normal_device:
                        right_wrist_normal_device = hand_tracking_result.right_hand.wrist_and_palm_normal_device.wrist_normal_device
                        right_palm_normal_device = hand_tracking_result.right_hand.wrist_and_palm_normal_device.palm_normal_device
                        right_wrist_normal_camera = (T_device_camera.inverse() @ right_wrist_normal_device).flatten()
                        right_palm_normal_camera = (T_device_camera.inverse() @ right_palm_normal_device).flatten() 

                    frame_pose = {
                        "frame_idx" : frame_idx,
                        "right_device": right_landmarks.tolist(),
                        "right_camera": right_landmarks_camera.tolist(),
                        "left_device": left_landmarks.tolist(),
                        "left_camera": left_landmarks_camera.tolist(),
                        "normal_device": {
                            "right_wrist": right_wrist_normal_device.tolist(),
                            "right_palm": right_palm_normal_device.tolist(),
                            "left_wrist": left_wrist_normal_device.tolist(),
                            "left_palm": left_palm_normal_device.tolist()
                        },
                        "normal_camera": {  
                            "right_wrist": right_wrist_normal_camera.tolist(),
                            "right_palm": right_palm_normal_camera.tolist(),
                            "left_wrist": left_wrist_normal_camera.tolist(),
                            "left_palm": left_palm_normal_camera.tolist()
                        }
                    }
                    T_world_camera = T_world_device @ T_device_camera
                    frame_calib = {"frame_idx" : frame_idx, "T_world_camera" : T_world_camera.to_matrix().tolist()}
                    pose_data["frames"].append(frame_pose)
                    calib_data["frames"].append(frame_calib)
                with open(all_pose_file, 'w') as f:
                    json.dump(pose_data, f, indent=2)
                
                with open(calib_file, 'w') as f:
                    json.dump(calib_data, f, indent=2)
            finally:
                video_writer1.release()
            print(f"Saved marked video to {cropped_video}")
        else:
            print("No valid hand tracking segment found")
    else:
        if start != -1 and end != -1:
            cropped_video = args.output_path + "cropped_video.mp4"
            marked_video = args.output_path + "marked_video.mp4"
            uv_file = args.output_path + "uv.json"
            uv_data = {"frames": []}
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            video_writer1 = cv2.VideoWriter(cropped_video, fourcc, fps, (width, width)) 
            video_writer2 = cv2.VideoWriter(marked_video, fourcc, fps, (width, width)) 
            try:
                for frame_idx in tqdm(range(start, end - 1)):
                    # 获取当前帧
                    record_rgb = provider.get_image_data_by_index(rgb_id, frame_idx)
                    rgb_img = record_rgb[0].to_numpy_array()
                    
                    rotated_img = np.rot90(rgb_img, k=-1)  # 逆时针旋转90度
                    colored_img = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR)
                    rectified_image = calibration.distort_by_calibration(colored_img, dst_calib_rgb, rgb_calib)
                    video_writer1.write(rectified_image)
                    
                    marked_frame = cv2.rotate(rectified_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    right_landmark = hand_tracking_results[frame_idx - 1].right_hand.landmark_positions_device[0]  # First landmark of right hand
                    uv_right = dst_calib_rgb.project(T_device_camera.inverse() @ right_landmark)

                    left_landmark = hand_tracking_results[frame_idx - 1].left_hand.landmark_positions_device[5]  # First landmark of right hand
                    uv_left = dst_calib_rgb.project(T_device_camera.inverse() @ left_landmark)
                    if uv_left is not None:
                        cv2.circle(marked_frame, tuple(uv_left.astype(int)), 10, (0, 0, 255), -1) 
                    if uv_right is not None:
                        cv2.circle(marked_frame, tuple(uv_right.astype(int)), 10, (0, 0, 255), -1) 
                    
                    marked_frame = cv2.rotate(marked_frame, cv2.ROTATE_90_CLOCKWISE)
                    video_writer2.write(marked_frame)
                    frame_uv = {"frame_idx" : frame_idx,
                                "right_u" : width - uv_right[1] if uv_right is not None else 'nan',
                                "right_v" : uv_right[0] if uv_right is not None else 'nan',
                                "left_u" : width - uv_left[1] if uv_left is not None else 'nan',
                                "left_v" : uv_left[0] if uv_left is not None else 'nan'}
                    uv_data["frames"].append(frame_uv)
                with open(uv_file, 'w') as f:
                    json.dump(uv_data, f, indent=2)
            finally:
                video_writer1.release()
                video_writer2.release()
            print(f"Saved marked video to {cropped_video}")
        else:
            print("No valid hand tracking segment found")
        

if __name__ == "__main__":
    main()
    
"""
python hand_tracking_prep.py --vrs_path /coc/flash7/yliu3735/workspace/inpaint/temp_data/Trial.vrs --width 640 --output_path /coc/flash7/yliu3735/workspace/inpaint/temp_data/result_Trial/
"""