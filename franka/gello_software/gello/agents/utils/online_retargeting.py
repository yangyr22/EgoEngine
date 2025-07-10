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
from fm_vis.hand_detector.hand_reconstruction import OnlineHandCapture
from robot_viewer import RobotOnlineRetargetingSAPIENViewer
from camera_reader import BGR_Reader

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def onlineretargeting(
    robots: Optional[Tuple[RobotName]],
    fps: int,
    out_dir: Optional[str] = None,
    robotonly: bool = True,
):
    reader = BGR_Reader(width=640, height=480, fps=fps, visualize=False)
    hand_capture = OnlineHandCapture(hand_type='right', visualize=False)
    robot_list = list(robots) if robots else []
    viewer = RobotOnlineRetargetingSAPIENViewer(
        robot_names=robot_list,
        hand_type=HandType.right,
        use_ray_tracing=False,
        visualize=True,
    )

    reader.start()

    try:
        while True:
            frame_bgr,_ = reader.read()
            output_frame = hand_capture.hand_reconstruction_frame(frame_bgr)
            if output_frame != None:
                action = viewer.render_retargeting_data(output_frame) # (7)
                # print(action)

    finally:
        reader.end()

def main(
    robots: Optional[List[RobotName]] = None,
    fps: int = 30,
    out_dir: Optional[str] = None,
    robotonly: bool = True,
):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset,
    in headless mode. If out_video is provided, the result will be saved to an MP4 file.
    If --robotonly is True, we skip object loading and only show the hand + robot.
    """
    robot_dir = "/mnt/data2/dexmimic/workspace/dex-retargeting/assets/robots/hands/"
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    onlineretargeting(
        robots=robots,
        fps=fps,
        out_dir=out_dir,
        robotonly=True,
    )

if __name__ == "__main__":
    tyro.cli(main)

# python online_retargeting.py   --out-dir /coc/flash7/yliu3735/workspace/dex-retargeting/output   --robots ability panda ability --fps 30