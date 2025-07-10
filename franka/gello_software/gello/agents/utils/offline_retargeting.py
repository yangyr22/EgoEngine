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
from fm_vis.hand_detector.hand_reconstruction import HandCapture
from robot_viewer import RobotOfflineRetargetingSAPIENViewer

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def offlineretargeting(
    robots: Optional[Tuple[RobotName]],
    data_root: Path,
    fps: int,
    out_dir: Optional[str] = None,
    robotonly: bool = True,
):
    
    # hand_capture = HandCapture(hand_type="right", input_dir="/coc/flash7/yliu3735/datasets/EpicKitchens/EPIC-KITCHENS/P01/clip/P01_01/0004.mp4")
    # hand_capture.hand_reconstruction()
    # sample_data = hand_capture.output_dict
    sample_data = OfflineDataset("/mnt/data2/dexmimic/datasets/test/1.json")

    robot_list = list(robots) if robots else []
    viewer = RobotOfflineRetargetingSAPIENViewer(
        robot_names=robot_list,
        hand_type=HandType.right,
        headless=False,
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
    in headless mode. If out_video is provided, the result will be saved to an MP4 file.
    If --robotonly is True, we skip object loading and only show the hand + robot.
    """
    data_root = Path(dexycb_dir).absolute()
    robot_dir = Path(__file__).absolute().parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    if not data_root.exists():
        raise ValueError(f"Path to video dir: {data_root} does not exist.")
    else:
        print(f"Using video dir: {data_root}")

    offlineretargeting(
        robots=robots,
        data_root=data_root,
        fps=fps,
        out_dir=out_dir,
        robotonly=True,
    )

if __name__ == "__main__":
    tyro.cli(main)

# python offline_retargeting.py   --dexycb-dir /mnt/data2/dexmimic/datasets/test/1.json   --out-dir /coc/flash7/yliu3735/workspace/dex-retargeting/output   --robots ability panda ability --fps 30