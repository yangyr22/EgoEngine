import time
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import quaternion
import pickle
from scipy.spatial.transform import Rotation as R

import torch
import tyro

import cv2

import sys
sys.path.insert(0, "/mnt/data2/dexmimic/workspace/franka/gello_software")

# --- gello and robot imports ---
from gello.rl2_env import RobotEnv
from gello.robots.panda_deoxys_simple import PandaRobot
from gello.cameras.zed_camera import ZedCamera
from gello.cameras.realsense_camera import RealSenseCamera
# from gello.cameras.kinect_camera import KinectCamera

# --- robomimic imports for loading policy ---
import robomimic.utils.torch_utils as TorchUtils
import egomimic.utils.file_utils as FileUtils
from robomimic.algo import RolloutPolicy

@dataclass
class Args:
    """Arguments for rollout."""
    checkpoint_path: str  # path to Robomimic checkpoint .pth
    norm_path: str        # path to normalization stats file .pkl
    horizon: int = 200    # number of steps in the rollout
    control_rate_hz: int = 30
    camera_config: Optional[str] = None  # path to or dict for camera config (if you prefer to load from file)

def create_cameras() -> dict:
    """
    Example camera creation, matching your data_collection setup.
    You can adjust serial numbers and camera types here.
    """
    cam_config_dict = {"agentview": {"sn" : "213722070937", "type": "RealSense"},
                       "wrist": {"sn": 18482824, "type": "Zed"}}

    cam_dict = {}
    for cam_name, conf in cam_config_dict.items():
        cam_type = conf["type"]
        if cam_type == "RealSense":
            cam_dict[cam_name] = RealSenseCamera(device_id=conf["sn"])
        elif cam_type == "Zed":
            cam_dict[cam_name] = ZedCamera(cam_name, conf)
        else:
            raise ValueError(f"Unknown camera type {cam_type}")
    return cam_dict



def rollout(policy: RolloutPolicy, env: RobotEnv, horizon: int, unnorm_stats: str):
    """
    Simple rollout function to run policy on the real (or simulated) robot for `horizon` steps.
    """
    policy.start_episode()
    env.robot().reset()

    for step_idx in range(horizon):
        obs = env.get_obs()
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.strides and any(s < 0 for s in v.strides):
                obs[k] = v.copy()
        action = policy(ob=obs, unnorm_stats=unnorm_stats)
        print(action)
        
        rvec = action[3:6]
        # rvec[0] -= 3.14159265                 
        Rot, _ = cv2.Rodrigues(np.array(rvec)) 
        r = R.from_matrix(Rot)  
        hand_quat_list = r.as_quat()  # Returns [x, y, z, w]

        # Convert to `np.quaternion` format [w, x, y, z]
        hand_quat = np.array([hand_quat_list[3],  # w
                                hand_quat_list[0],  # x
                                hand_quat_list[1],  # y
                                hand_quat_list[2]])  # z

        action_new = np.concatenate([action[0:3], hand_quat, action[6:]]).tolist()
        # print(action_new)
        # exit(0)

        action_new[0] = min(max(0.30, action_new[0]),0.75)
        action_new[1] = min(max(-0.2, action_new[1]),0.30)
        action_new[2] = min(max(0.15, action_new[2]),0.4) # in case collision with the desk.

        _, _ = env.step(action_new)
        # time.sleep(1.0 / env.control_rate_hz)


def main(checkpoint_path: str, norm_path: str, horizon: int = 200, control_rate_hz: int = 30):
    """Main function to run the rollout."""
    robot_client = PandaRobot("OSC_POSE", gripper_type="umi")
    cam_dict = create_cameras()

    env = RobotEnv(
        robot_client,
        camera_dict=cam_dict,
        save_depth_obs=False,
    )

    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(checkpoint_path)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict, device=device, verbose=True
    )

    with open(norm_path, "rb") as f:
        unnorm_stats = pickle.load(f)

    print("Starting rollout...")
    rollout(policy, env, horizon, unnorm_stats)
    print("Rollout finished.")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args.checkpoint_path, args.norm_path, args.horizon, args.control_rate_hz)
