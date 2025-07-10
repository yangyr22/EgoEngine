from matplotlib import pyplot as plt
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


from hand_detector.handmocap.hand_modules.h3dw_model import H3DWModel
from hand_detector.handmocap.hand_modules.test_options import TestOptions
from hand_detector.mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from hand_mode_detector import SingleHandDetector,HandMocap
import os, sys, shutil
import os.path as osp

import json

import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time

# from renderer.screen_free_visualizer import Visualizer

from renderer.visualizer import Visualizer
def save_pred_output(pred_output, frame_idx, output_dir):
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy_arrays(d):
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
            elif isinstance(value, dict):
                convert_numpy_arrays(value)  # Recurse for nested dictionaries
        return d

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert pred_output dictionary to a serializable format
    serializable_pred_output = convert_numpy_arrays(pred_output)

    # Write to a JSON file
    json_file_path = os.path.join(output_dir, f"{frame_idx:05d}.json")
    with open(json_file_path, 'w') as f:
        json.dump(serializable_pred_output, f, indent=4)

def frames2video(frame_folder, output_video_path):
    fps = 30  # Adjust as needed

    # Get all frames and sort them numerically
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")], key=lambda x: int(os.path.splitext(x)[0]))

    # Read the first frame to get width and height
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, layers = first_frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        out.write(frame)

    out.release()

def main():
    parser = argparse.ArgumentParser(description='Hand tracking with input video.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('--hand_type', type=str, choices=['left', 'right'], default='left', help='Specify which hand to track (left or right).')
    args = parser.parse_args()
    
    
    video_path = args.video_path
    hand_type = args.hand_type  # Get the hand type from the argument
    
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        sys.exit(1)
    
    hand_detector_dir = Path(__file__).parent
    default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
    default_checkpoint_body_smpl = './extra_data/smpl'
    
    # Initialize the detector with the specified hand type
    detector = SingleHandDetector(hand_type=hand_type.capitalize())  # Capitalize to match expected input ('Left' or 'Right')
    
    hand_mocap = HandMocap(str(hand_detector_dir / default_checkpoint_hand),
                           str(hand_detector_dir / default_checkpoint_body_smpl))

    vid = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, image_bgr = vid.read()
        if not ret:
            break  # Exit if video ends
        
        _, bbox = detector.detect_hand_bbox(image_bgr)
        hand_bbox_list = [{"left_hand": None, "right_hand": None}]
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        parent_dir = os.path.dirname(video_path)

        # create directory that save the image and json file for the video 
        img_dir = os.path.join(parent_dir, video_name, "image_folder")
        js_dir = os.path.join(parent_dir, video_name, "json_folder")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(js_dir, exist_ok=True)

        if bbox is not None:
            if hand_type == 'left':
                hand_bbox_list[0]["left_hand"] = bbox[0]
            elif hand_type == 'right':
                hand_bbox_list[0]["right_hand"] = bbox[0]
                
            pred_output = hand_mocap.regress_original(image_bgr, hand_bbox_list, add_margin=False)
            
            pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output)
            visualizer = Visualizer(rendererType="opengl")
            
            res_img = visualizer.visualize(
                image_bgr, 
                pred_mesh_list=pred_mesh_list, 
                hand_bbox_list=hand_bbox_list)

            res_img = res_img.astype(np.uint8)
            ImShow(res_img)

            cv2.imwrite(os.path.join(img_dir, f"{frame_idx}.jpg"), res_img)
            save_pred_output(pred_output[0], frame_idx, js_dir)
        
        frame_idx += 1

if __name__ == '__main__':
    main()

