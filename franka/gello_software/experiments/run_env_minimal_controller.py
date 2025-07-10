"""
Run a minimal environment locally and with the panda robot, using either gello or VR
"""



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

import sys
gello_path = "/mnt/data2/dexmimic/workspace/franka/gello_software"
if gello_path not in sys.path:
    sys.path.insert(0, gello_path)

from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.rl2_env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.robots.panda_deoxys_simple import PandaRobot
from gello.zmq_core.robot_node import ZMQClientRobot
# from gello.cameras.zed_camera import ZedCamera
# from gello.cameras.realsense_camera import RealSenseCamera
#TODO install pyzed and use it
# import pyzed.sl as sl


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
    # shoulderview_left_sn: int = 14620168 #Zed mini shoulderview left
    # shoulderview_left_sn: int = 20036094
    wrist_camera_sn: int = 20036094
    camera_type: str = "both" # can be 'Zed' or 'zmq' or 'RealSense' or 'both'
    save_depth_obs: bool = False
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    robot_controller: str = "osc_pose" # "joint_impedance" ## make sure this matches with the spun-up robot node

    gello_port: Optional[str] = None
    save_pkl: bool = False # use_save_interface: bool = False
    save_hdf5: bool = True
    # data_dir: str = "/media/aloha/Data/robomimic-v2/Demos/Retriever"  # provide save dir here
    data_dir: str = "/mnt/data2/dexmimic/datasets/controller_teleop_data/pick_place"  # provide save dir here
    task: str = None
    # Dont change below
    bimanual: bool = False
    verbose: bool = False



def main(args):
    # TODO fix to not need keyboard interface
    if not args.task:
        args.task = str(datetime.date.today())

    data_save_dir = os.path.join(str(Path(args.data_dir).expanduser()), args.task)
    if args.save_hdf5:
        os.makedirs(data_save_dir, exist_ok=True)


    ### TODO Starting the Zed cameras
    # cameras = sl.Camera.get_device_list()

    # TODO: correct the cam dict
    cam_dict = {}
    cam_config_dict = {"agentview": {"sn" : "213722070937", "type": "RealSense"},
                       "wrist": {"sn": 18482824, "type": "Zed"}}

    # TODO make the setting of the camera dict cleaner

    if cam_config_dict is not None:
        for cam in cam_config_dict:
            # cam_config_dict is {"camera_name" : {"sn": int or str, type: "Zed", "RealSense" or "Kinect" +
            #                                                   camera-specific configs}
            cam_config = cam_config_dict[cam]
            if cam_config["type"] == "Zed":
                from gello.cameras.zed_camera import ZedCamera
                cam_dict[cam] = ZedCamera(cam, cam_config)
            elif cam_config["type"] == "RealSense":
                from  gello.cameras.realsense_camera import RealSenseCamera
                cam_dict[cam] = RealSenseCamera(device_id=cam_config['sn'])
            elif cam_config["type"] == "Kinect":
                from gello.cameras.kinect_camera import KinectCamera
                cam_dict[cam] = KinectCamera(cam, cam_config)

    # for cam in cameras:
    #     if cam.serial_number == args.shoulderview_left_sn:
    #         cam_dict["shoulderview_left"] = ZedCamera(cam)
    #     elif cam.serial_number == args.shoulderview_right_sn:
    #         cam_dict["shoulderview_right"] = ZedCamera(cam)
    #     elif cam.serial_number == args.wrist_camera_sn:
    #         cam_dict["wrist"] = ZedCamera(cam)

    robot_client = PandaRobot("OSC_POSE", gripper_type="robotiq")
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=cam_dict, save_depth_obs=args.save_depth_obs)

    task_description = input("Enter description of the task: ")


    # create metadata for saving #TODO maybe move this somewhere else for better setting of the metadata
    ENV_ARGS = {
        "env_name": "EnvRealPandaDeoxys",
        "type": 4,
        'lang': task_description,
        "env_kwargs": {
            # "camera_dict" : {'shoulderview_left': {'size' : (128, 128), 'sn' : args.shoulderview_left_sn, 'type': "Zed"},
            #                  'shoulderview_right': {'size': (128, 128), 'sn': args.shoulderview_right_sn, 'type': "Zed"},
            #                  'wrist' : {'size': (128, 128), 'sn': args.wrist_camera_sn, 'type': "Zed"},
                             # 'left': {'size': (128, 128), 'sn': args.left_camera_sn, 'type': "RealSense"},
                             # 'right': {'size': (128, 128), 'sn': args.right_camera_sn, 'type': "RealSense"},
                             # },
            "camera_dict" : cam_config_dict,
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
    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    [0, -90, 90, -90, -90, 0, 0]
                )  # Change this to your own reset joints
            else:
                reset_joints = args.start_joints
            agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
            curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest_rl2":
            from gello.agents.quest_rl2_agent import Quest_Agent

            agent = Quest_Agent(robot_interface=robot_client.robot_interface)
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    robot_client.reset()

    if args.save_pkl or args.save_hdf5:  # args.use_save_interface:
        from gello.data_utils.demo_collection_screen import Screen
        screen = Screen()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    curr_time = time.time()

    # TODO: make this a proper state machine and use the screen to display state machine
    # state can be [idle, start, recording, stop, delete]
    state = "idle"

    # Create a data collector
    if args.save_hdf5:
        from gello.data_utils.data_collector import DataCollector
        data_collector = DataCollector(data_save_dir)

    try:
        obs = env.get_obs()
        freqs = []

        while True:
            controller_state = agent.get_controller_state()
            if controller_state is None:
                continue
            action = controller_state['target_pose'] + controller_state['gripper_act']
            # print(action)

            # TODO: maybe use a state machine library for control loop
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

            if args.save_hdf5:

                if state == "idle":  # busy wait
                    screen.update_state(state)
                    time.sleep(0.001)

                # START RECORDING
                elif state == "start":

                    obs = env.get_obs()
                    controller_state = agent.get_controller_state()
                    action = controller_state['target_pose'] + controller_state['gripper_act']

                    data = {}
                    data['obs'] = deepcopy(obs)
                    obs, (act_delta, act_abs) = env.step(action)
                    data['act_delta'] = act_delta
                    data['act_abs'] = act_abs
                    data['control_enabled'] = controller_state['engaged']
                    data_collector.collect(data)


                # RECORD TRAJECTORY
                elif state == "recording":
                    data = {}
                    data['obs'] = deepcopy(obs)
                    obs, (act_delta, act_abs) = env.step(action)
                    data['act_delta'] = act_delta
                    data['act_abs'] = act_abs
                    data['control_enabled'] = controller_state['engaged']
                    data_collector.collect(data)

                # STOP RECORDING
                elif state == "stop":

                    # Resetting robot
                    data_collector.end_episode(success=True)
                    env.robot().reset()
                    agent.reset_internal_state()


                    # save_path = None
                    # transition_count = 0  # start new demo
                    state = "idle"

                    # print freq for episode
                    min_f = min(freqs)
                    max_f = max(freqs)
                    avg_f = sum(freqs) / len(freqs)

                    print(f"Episode average frequency: {avg_f} . Max: {max_f}. Min: {min_f}")
                    freqs = []


                # DELETE DEMO
                elif state == "delete":

                    data_collector.end_episode(success=False)
                    env.robot().reset()
                    agent.reset_internal_state()

                    state = "idle"

                else:
                    raise ValueError(f"Invalid state {state}")

            else:
                # Just step action without saving
                print(f'action: {action}')
                obs = env.step(action)
            if state != "idle":
                freq = 1 / (time.time() - curr_time)
                freqs.append(freq)
                curr_time = time.time()


    except KeyboardInterrupt as e:
        print("Caught Ctrl+C,  save loop")
        env.close()



if __name__ == "__main__":
    main(tyro.cli(Args))
