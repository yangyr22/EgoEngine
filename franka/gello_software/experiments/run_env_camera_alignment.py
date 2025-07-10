import datetime
import glob
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math
from copy import deepcopy

import numpy as np
import h5py
import json
import tyro

from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.rl2_env import RobotEnv
from gello.robots.panda_deoxys_simple import PandaRobot
from gello.zmq_core.robot_node import ZMQClientRobot

import cv2
import pyrealsense2 as rs
from dt_apriltags import Detector
from scipy.spatial.transform import Rotation

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor
    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "quest_rl2"
    robot_port: int = 6001
    right_camera_port: int = 5002
    left_camera_port: int = 5003
    left_camera_sn: str = "213722070937"
    right_camera_sn: str = "151222077158"
    shoulderview_right_sn: int = 21497414
    shoulderview_left_sn: int = 20036094
    wrist_camera_sn: int = 18482824
    camera_type: str = "both" # can be 'Zed' or 'zmq' or 'RealSense' or 'both'
    save_depth_obs: bool = False
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    robot_controller: str = "osc_pose" # "joint_impedance"
    gello_port: Optional[str] = None
    save_pkl: bool = False
    save_hdf5: bool = True
    data_dir: str = "/mnt/data2/dexmimic/datasets/controller_teleop_data"
    task: str = None
    bimanual: bool = False
    verbose: bool = False


def draw_pose_axes(overlay, camera_params, tag_size, pose, center):
    """Helper to visualize the tag’s axes on the color image."""
    fx, fy, cx, cy = camera_params
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]
    dcoeffs = np.zeros(5)

    opoints = np.float32([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]]).reshape(-1, 3) * tag_size

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
    ipoints = np.round(ipoints).astype(int)

    center = np.round(center).astype(int)
    center = tuple(center.ravel())

    cv2.line(overlay, center, tuple(ipoints[0].ravel()), (0, 0, 255), 2)
    cv2.line(overlay, center, tuple(ipoints[1].ravel()), (0, 255, 0), 2)
    cv2.line(overlay, center, tuple(ipoints[2].ravel()), (255, 0, 0), 2)


def main(args):
    if not args.task:
        args.task = str(datetime.date.today())

    data_save_dir = os.path.join(str(Path(args.data_dir).expanduser()), args.task)
    if args.save_hdf5:
        os.makedirs(data_save_dir, exist_ok=True)

    ################################################################
    # Initialize RealSense pipeline
    ################################################################
    realsense_serial = args.left_camera_sn
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(realsense_serial)
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    intr.fx = 322.470
    intr.fy = 322.470
    intr.ppx = 321.048
    intr.ppy = 177.549
    camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)

    # Apriltag Detector
    at_detector = Detector(
        families='tagStandard41h12',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    ################################################################
    # This will store the calibration data
    ################################################################
    # Each entry is: 
    # {
    #   "robot_act_abs": [x, y, z, qx, qy, qz, qw],
    #   "tag_position" : [x, y, z, qx, qy, qz, qw]
    # }
    calibration_data = []

    ################################################################
    # Setup environment and teleoperation
    ################################################################
    robot_client = PandaRobot("OSC_POSE", gripper_type="umi")
    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=None,
        save_depth_obs=args.save_depth_obs
    )

    task_description = input("Enter description of the task: ")
    ENV_ARGS = {
        "env_name": "EnvRealPandaDeoxys",
        "type": 4,
        'lang': task_description,
        "env_kwargs": {
            "camera_dict": {},
            'general_cfg_file': None,
            'control_freq': 20,
            'controller_type': 'OSC_POSE',
            'controller_cfg_file': None,
            'controller_cfg_dict': None,
            'use_depth_obs': False, 'state_freq': 100.0, 'control_timeout': 1.0,
            'has_gripper': True, 'use_visualizer': False,
            'gripper_type': 'robotiq',
            'reset_joint_positions': env.robot().reset_joint_positions
        }
    }

    if args.bimanual:
        raise Exception("Cannot do bimanual teleop currently")

    # Create teleop agent
    if args.agent == "gello":
        if args.gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            if len(usb_ports) == 0:
                raise ValueError("No Gello device found. Please specify the port or plug in Gello")
            gello_port = usb_ports[0]
        else:
            gello_port = args.gello_port
        agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
    elif args.agent == "quest_rl2":
        from gello.agents.quest_rl2_agent import Quest_Agent
        agent = Quest_Agent(robot_interface=robot_client.robot_interface)
    elif args.agent == "spacemouse":
        from gello.agents.spacemouse_agent import SpacemouseAgent
        agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
    else:
        raise ValueError("Invalid agent name")

    # Reset robot
    print("Resetting robot to a default start position...")
    robot_client.reset()

    # Setup data saving (if needed)
    if args.save_pkl or args.save_hdf5:
        from gello.data_utils.demo_collection_screen import Screen
        screen = Screen()

    if args.save_hdf5:
        from gello.data_utils.data_collector import DataCollector
        data_collector = DataCollector(data_save_dir)

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    curr_time = time.time()
    state = "idle"
    freqs = []

    try:
        obs = env.get_obs()

        while True:
            # -----------------------------------------------------------
            # 1) Grab RealSense frame, detect AprilTags
            # -----------------------------------------------------------
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            tags = at_detector.detect(
                gray_image,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=0.06 * 5 / 9  # Adjust to your actual tag size in meters
            )
            # Optional: visualize tags
            for det in tags:
                pose_mat = np.hstack([det.pose_R, det.pose_t])
                draw_pose_axes(color_image, camera_params, 0.05, pose_mat, det.center)

            # -----------------------------------------------------------
            # 2) Get agent’s control state, step the environment
            # -----------------------------------------------------------
            controller_state = agent.get_controller_state()
            if controller_state is None:
                cv2.imshow("RealSense", color_image)
                cv2.waitKey(1)
                continue

            # action = [dx, dy, dz, dqx, dqy, dqz, dqw, gripper?]
            action = controller_state['target_pose'] + controller_state['gripper_act']

            # Teleop state machine
            if controller_state['engaged'] and state == "idle":
                state = "start"
            elif state == "start":
                state = "recording"
                screen.update_state(state)
                data_collector.start_episode(ENV_ARGS)
            elif state == "recording" and controller_state['save_demo']:
                state = "stop"
                screen.update_state(state)
            elif state == "recording" and controller_state['delete_demo']:
                state = "delete"
                screen.update_state(state)

            # Step environment, record data
            if args.save_hdf5:
                if state == "idle":
                    screen.update_state(state)
                    time.sleep(0.001)

                elif state == "start":
                    obs = env.get_obs()
                    data = {}
                    # step -> returns obs, (act_delta, act_abs)
                    obs, (act_delta, act_abs) = env.step(action)
                    data['obs'] = deepcopy(obs)
                    data['act_delta'] = act_delta
                    data['act_abs'] = act_abs
                    data['control_enabled'] = controller_state['engaged']
                    data_collector.collect(data)

                elif state == "recording":
                    data = {}
                    obs, (act_delta, act_abs) = env.step(action)
                    data['obs'] = deepcopy(obs)
                    data['act_delta'] = act_delta
                    data['act_abs'] = act_abs
                    data['control_enabled'] = controller_state['engaged']
                    data_collector.collect(data)
                        
                    # 3) If we have tags, record calibration info (robot act_abs + tag pose)
                    if len(tags) > 0:
                        for tag in tags:
                            tag_quat = Rotation.from_matrix(tag.pose_R).as_quat()  # [qx, qy, qz, qw]
                            tag_trans = tag.pose_t.flatten()                       # [x, y, z]
                            # act_abs is also length 7: [x, y, z, qx, qy, qz, qw]
                            calibration_data.append({
                                "robot_act_abs": list(act_abs),
                                "tag_position": list(np.concatenate([tag_trans, tag_quat]))
                            })

                elif state == "stop":
                    data_collector.end_episode(success=True)
                    env.robot().reset()
                    agent.reset_internal_state()
                    state = "idle"

                    if len(freqs) > 0:
                        min_f = min(freqs)
                        max_f = max(freqs)
                        avg_f = sum(freqs) / len(freqs)
                        print(f"Episode average frequency: {avg_f} . Max: {max_f}. Min: {min_f}")
                        freqs = []

                elif state == "delete":
                    data_collector.end_episode(success=False)
                    env.robot().reset()
                    agent.reset_internal_state()
                    state = "idle"

                else:
                    # Just skip
                    pass

            else:
                # Not saving: just step
                obs, (act_delta, act_abs) = env.step(action)
                # Still could record calibration if you want, e.g.:
                if len(tags) > 0:
                    for tag in tags:
                        tag_quat = Rotation.from_matrix(tag.pose_R).as_quat()
                        tag_trans = tag.pose_t.flatten()
                        calibration_data.append({
                            "robot_act_abs": list(act_abs),
                            "tag_position": list(np.concatenate([tag_trans, tag_quat]))
                        })

            # -----------------------------------------------------------
            # 4) Show frames, handle exit
            # -----------------------------------------------------------
            cv2.imshow("RealSense", color_image)
            if cv2.waitKey(1) == ord('q'):
                break

            freq = 1.0 / (time.time() - curr_time)
            freqs.append(freq)
            curr_time = time.time()

    except KeyboardInterrupt:
        print("Caught Ctrl+C, saving partial data and shutting down.")

    finally:
        # Stop streaming
        pipeline.stop()

        # Save calibration data to JSON
        calibration_json_path = os.path.join(data_save_dir, "calibration.json")
        with open(calibration_json_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Saved calibration data to: {calibration_json_path}")

        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
