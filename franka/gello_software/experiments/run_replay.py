import time
import argparse
import h5py
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


# --- gello and robot imports ---
from gello.rl2_env import RobotEnv
from gello.robots.panda_deoxys_simple import PandaRobot
from gello.cameras.zed_camera import ZedCamera
from gello.cameras.realsense_camera import RealSenseCamera

def convert_action(action_abs):
    """
    Convert a 7-DOF action_absolute to an 8-DOF action.
    The input vector is expected as:
        [position (3), rotation vector (3), gripper]
    The output will be:
        [position (3), quaternion (4), gripper]
    """
    pos = action_abs[:3]
    rvec = action_abs[3:6]
    gripper = action_abs[6]
    rvec[0] -= 3.14159265 
    print(action_abs)
    
    # Convert rotation vector to rotation matrix and then to quaternion.
    Rot, _ = cv2.Rodrigues(np.array(rvec))
    r = R.from_matrix(Rot)
    quat = r.as_quat()
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    
    # Concatenate to form an 8-DOF action.
    action_new = np.concatenate([pos, quat, [gripper]])
    
    # Optional: clip the position values to safe limits.
    action_new[0] = np.clip(action_new[0], 0.30, 0.75)
    action_new[1] = np.clip(action_new[1], -0.2, 0.30)
    action_new[2] = np.clip(action_new[2], 0.15, 0.4)
    
    return action_new

def create_cameras() -> dict:
    """
    Creates the camera dictionary based on the configuration.
    Adjust the serial numbers and camera types as needed.
    """
    cam_config_dict = {
        "agentview": {"sn": "213722070937", "type": "RealSense"}
        # "wrist": {"sn": 18482824, "type": "Zed"}
    }
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

def replay_data(hdf5_path: str, control_rate_hz: int = 30):
    """
    Replay collected data from the given HDF5 file.
    Iterates over all chunks in the dataset sequentially,
    converts each action_absolute (7-DOF) to an 8-DOF action, and
    sends it to the robot environment.
    """
    # Initialize the robot client and create the environment.
    robot_client = PandaRobot("OSC_POSE", gripper_type="umi")
    cam_dict = create_cameras()
    env = RobotEnv(robot_client, camera_dict=cam_dict, save_depth_obs=False)
    
    # Open the HDF5 file containing the collected data.
    with h5py.File(hdf5_path, "r") as f:
        # Assuming the data is stored under the group "data/demo_0"
        demo_group = f["data/demo_0"]
        # Sort chunk keys (e.g., chunk_0, chunk_1, ...) to replay in order.
        chunk_keys = sorted(list(demo_group.keys()))
        
        for chunk_key in chunk_keys:
            print(f"Replaying chunk: {chunk_key}")
            chunk_group = demo_group[chunk_key]
            # Load the action_absolute dataset (shape: (N, 7)).
            actions_abs = chunk_group["action_absolute"][:]
            # Iterate over each action and replay.
            for idx, action_abs in enumerate(actions_abs):
                action_new = convert_action(action_abs)
                print(f"Step {idx}: Action {action_new}")
                # env.step(action_new.tolist())
                # Maintain the desired control rate.
                time.sleep(1.0 / control_rate_hz)
    print("Replay finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay collected data from an HDF5 file.")
    parser.add_argument("--hdf5_path", type=str, help="Path to the collected data HDF5 file.")
    parser.add_argument("--control_rate_hz", type=int, default=30, help="Control rate (Hz) for replaying actions.")
    args = parser.parse_args()
    replay_data(args.hdf5_path, args.control_rate_hz)
