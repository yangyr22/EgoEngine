from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro

from offline_dataset import OfflineDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
import sys
import os
# Add the Retargeting folder to sys.path
script_dir = os.path.dirname(os.path.realpath(__file__))
retargeting_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(retargeting_dir)
from fm_vis.hand_detector.hand_reconstruction import HandCapture, HandCaptureOnly4Preprocessing
from fm_vis.hand_detector.hand_reconstruction_utils import HandCaptureRotation
from robot_viewer import RobotOfflineRetargetingSAPIENViewer
import json

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


import json

def offlineretargeting(
    robots: Optional[Tuple[RobotName]],
    data_root: Path,
    fps: int,
    out_dir: Optional[str] = None,
    robotonly: bool = True,
):

    # hand_capture = HandCaptureOnly4Preprocessing(hand_type="right", input_dir=data_root, save_visual_result=out_dir)
    # hand_capture = HandCapture(hand_type="right", input_dir=data_root, save_visual_result=out_dir)
    hand_capture = HandCaptureRotation(hand_type="right", input_dir=data_root, save_visual_result=out_dir)


    hand_capture.hand_reconstruction()
    # if hand_capture == None:
    #     return None
    sample_data = hand_capture.output_dict


    robot_list = list(robots) if robots else []
    viewer = RobotOfflineRetargetingSAPIENViewer(
        robot_names=robot_list,
        hand_type=HandType.right,
        headless=True,
        use_ray_tracing=False,
        visualize=True,
    )

    viewer.render_retargeting_data(sample_data, fps, out_dir=out_dir)


def offlineretargeting4json(
    robots: Optional[Tuple[RobotName]],
    data_root: Path,
    fps: int,
    out_dir: Optional[str] = None,
    robotonly: bool = True,
):
    with open(data_root, "r") as file:
        data = json.load(file) 
    rvec = np.array(data["rvec"])  # Shape (T, 3)
    tvec = np.array(data["tvec"])  # Shape (T, 3)
    pred_3d_joints = np.array(data["pred_3d_joints"])  # Shape (T, 21, 3)
    sample_data = {'rvec': rvec,          # shape=(0, 3)
                'tvec': tvec,          # shape=(0, 3)
                'pred_3d_joints': pred_3d_joints # shape=(0,21,3)
                }

    robot_list = list(robots) if robots else []
    viewer = RobotOfflineRetargetingSAPIENViewer(
        robot_names=robot_list,
        hand_type=HandType.right,
        headless=True,
        use_ray_tracing=False,
        visualize=True,
    )

    viewer.render_retargeting_data(sample_data, fps, out_dir=out_dir)


def main(
    dexycb_dir: str,
    robots: Optional[List[RobotName]] = None,
    fps: int = 30,
    out_dir: Optional[str] = None,
    robotonly: bool = True,
):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset,

    Process all videos in the input directory and store outputs in corresponding subdirectories under output_dir.
    Each output is saved in a folder named after the video file (without extension) inside output_dir.

    """
    data_root = dexycb_dir
    robot_dir = Path(__file__).absolute().parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    input_path = Path(dexycb_dir)
    output_path = Path(out_dir)
    
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory {input_path} does not exist or is not a directory.")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    for video_file in input_path.glob("*.json"):  # Adjust pattern if needed to match video formats
        video_name = video_file.stem
        video_output_dir = output_path / video_name

        if os.path.isdir(video_output_dir):
            print(f"Skipping {video_file} -> Output: {video_output_dir} as it already exists")
            continue
        # video_output_dir.mkdir(parents=True, exist_ok=True)
        print(video_name)
        
        print(f"Processing {video_file} -> Output: {video_output_dir}")

        offlineretargeting4json(
            robots=robots,
            data_root=video_file,
            fps=fps,
            out_dir=str(video_output_dir),
            robotonly=True,
        )

    

if __name__ == "__main__":
    tyro.cli(main)

# python offline_retargeting.py   --dexycb-dir  /directory/to/videos   --out-dir /directory/to/output   --robots ability robotiq umi ability --fps 30