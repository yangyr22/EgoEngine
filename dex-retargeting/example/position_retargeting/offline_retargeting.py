from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro

from dataset import DexYCBVideoDataset
from test_dataset import customeddataset
from customed_dataset import CustomVideoDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig

from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer
from robot_viewer import RobotOfflineRetargetingSAPIENViewer

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def viz_hand_object(
    robots: Optional[Tuple[RobotName]],
    data_root: Path,
    fps: int,
    out_video: Optional[str] = None,
    robotonly: bool = True,
):

    # dataset = DexYCBVideoDataset(data_root, hand_type="right") 
    sample_data = customeddataset(data_root)

    robot_list = list(robots) if robots else []
    viewer = RobotOfflineRetargetingSAPIENViewer(
        robot_names=robot_list,
        hand_type=HandType.right,
        headless=True,
        use_ray_tracing=False,
    )

    viewer.render_retargeting_data(sample_data, fps, out_video=out_video)


def main(
    dexycb_dir: str,
    robots: Optional[List[RobotName]] = None,
    fps: int = 10,
    out_video: Optional[str] = None,
    robotonly: bool = True,
):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset,
    in headless mode. If out_video is provided, the result will be saved to an MP4 file.
    If --robotonly is True, we skip object loading and only show the hand + robot.
    """
    data_root = Path(dexycb_dir).absolute()
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    if not data_root.exists():
        raise ValueError(f"Path to video dir: {data_root} does not exist.")
    else:
        print(f"Using video dir: {data_root}")

    viz_hand_object(
        robots=robots,
        data_root=data_root,
        fps=fps,
        out_video=out_video,
        robotonly=True,
    )


if __name__ == "__main__":
    tyro.cli(main)

