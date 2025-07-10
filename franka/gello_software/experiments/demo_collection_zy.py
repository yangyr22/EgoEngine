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
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import h5py
import json
import tyro

from deoxys.utils import YamlConfig, transform_utils

from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.rl2_robot_env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.robots.panda_deoxys_simple import PandaRobot
from gello.zmq_core.robot_node import ZMQClientRobot
# from gello.cameras.zed_camera import ZedCamera
# from gello.cameras.realsense_camera import RealSenseCamera
# import pyzed.sl as sl
from robomimic.dev.multi_processing.envs.rl2_robot_env_process import RL2RobotEnv
from robomimic.utils.time_utils import precise_sleep


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
    shoulderview_left_sn: int = 20036094
    wrist_camera_sn: int = 18482824
    camera_type: str = "both" # can be 'Zed' or 'zmq' or 'RealSense' or 'both'
    save_depth_obs: bool = False
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    env_frequency: int = 20 # frequency of the stacked obs and demo collection frequency

    start_joints: Optional[Tuple[float, ...]] = None
    robot_controller: str = "osc_pose" # "joint_impedance" ## make sure this matches with the spun-up robot node

    gello_port: Optional[str] = None
    save_pkl: bool = False # use_save_interface: bool = False
    save_hdf5: bool = True
    # data_dir: str = "/media/aloha/Data/robomimic-v2/Demos/Retriever"  # provide save dir here
    # data_dir: str = "/home/mbronars/nadun/teleop_testing"  # provide save dir here
    data_dir: str = "/home/mbronars/zhenyang/demos"
    task: str = None
    # Dont change below
    bimanual: bool = False
    verbose: bool = False

def get_act_delta(action, state):
    """
    action: pos + axis_angle + gripper = (7,)
    state: current observation
    """
    current_pos = state['eef_pos'][-1]
    current_quat = state['eef_quat'][-1]

    # Assume action is always an absolute pose
    target_pos = action[0:3]
    target_axis_angle = action[3:6]
    target_quat = transform_utils.axisangle2quat(target_axis_angle)
    gripper_act = action[-1]

    quat_diff = transform_utils.quat_distance(target_quat, current_quat)
    axis_angle_diff = transform_utils.quat2axisangle(quat_diff).tolist()
    action_pos = ((np.array(target_pos) - np.array(current_pos)) * 10).tolist()

    delta_action = action_pos + axis_angle_diff + [gripper_act]
    return delta_action

def main(args):
    if not args.task:
        args.task = str(datetime.date.today())

    data_save_dir = os.path.join(str(Path(args.data_dir).expanduser()), args.task)
    if args.save_hdf5:
        os.makedirs(data_save_dir, exist_ok=True)

    # Setup environment with shared memory manager
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # TODO: add to config Configure observation key mapping
    obs_key_map = {
        "robot_timestamp": "robot_timestamp",
        "joint_positions": "joint_positions",
        "joint_velocities": "joint_velocities",
        "joint_positions_desired": "joint_positions_desired",
        "joint_velocities_desired": "joint_velocities_desired",
        "ee_twist_desired": "ee_twist_desired",
        "ee_pose_desired": "ee_pose_desired",
        "eef_pos": "eef_pos",
        "eef_quat": "eef_quat",
        "eef_axis_angle": "eef_axis_angle",
        "eef_pose": "eef_pose",
        # gripper
        "gripper_state": "gripper_state",
        "gripper_position": "gripper_position",
        "gripper_velocity": "gripper_velocity",
        "gripper_force": "gripper_force",
        "gripper_timestamp": "gripper_timestamp"
    } 

    # TODO: add this to config system later
    camera_config_dict = {"agentview": {"sn": "001039114912",
                                        "type": "Kinect",
                                        "resize": True,
                                        "resize_resolution": (512, 512)},
                          "wrist": {"sn": 14620168,
                                    "fps": 60.0,
                                    "resize": True,
                                    "resize_resolution": (512, 512),
                                    "camera_pos": "right"}  # TODO: change, only for wiping board task after 0118
                                    # "camera_pos": "left"}
                          }  # NOTE: need to change accordingly

    env = RL2RobotEnv(
        # env setup
        shm_manager=shm_manager,
        frequency=args.env_frequency,
        n_obs_steps=1, # for demo collection, only need one
        obs_key_map=obs_key_map,
        # camera setup
        camera_name="agentview",
        camera_config_dict=camera_config_dict,
        save_depth_obs=args.save_depth_obs,
        max_obs_buffer_size=30,
        # robot control setup
        controller_type="OSC_POSE",
        control_rate_robot=60, # TODO: make sure this higher than env rate
        robot_latency=0.0,
        verbose=args.verbose
    )

    # Start the environment and going to start position
    env.start(wait=True)
    time.sleep(2.0)
    print("Going to start  position")
    obs = env.reset()
    time.sleep(1.5)

    task_description = input("Enter description of the task: ")

    # create metadata for saving #TODO maybe move this somewhere else for better setting of the metadata
    ENV_ARGS = {
        "env_name": "RL2RobotEnv",
        "type": 4,
        'lang': task_description,
        "env_kwargs": {
            "camera_config_dict" : camera_config_dict,
            'general_cfg_file': None,
            'control_freq': 20,
            'controller_type': 'OSC_POSE',
            'controller_cfg_file': None,
            'controller_cfg_dict': None,
            'use_depth_obs': False, 'state_freq': 100.0, 'control_timeout': 1.0,
            'has_gripper': True, 'use_visualizer': False,
            'gripper_type': 'robotiq',
            'reset_joint_positions': env._robot.reset_position
        }
    }

    # select agent (teleop controller)
    if args.bimanual:
        raise Exception("Cannot do bimanual teleop currently")
    else:
        if args.agent == "quest_rl2":
            from gello.agents.quest_rl2_agent import QuestAgent
            agent = QuestAgent()
            agent.reset_internal_state(obs['eef_pos'][-1], obs['eef_quat'][-1])
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

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
        dt = 1 / args.env_frequency
        obs = env.get_observation()
        freqs = []

        while True:
            t_start = time.time()
            controller_state = agent.get_controller_state(obs['eef_pos'][-1], obs['eef_quat'][-1])
            if controller_state is None:
                continue

            action = controller_state['target_pos'] + controller_state['target_axis_angle'] + controller_state['gripper_act']
            action = np.array(action)

            # get process control state
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
                    time.sleep(0.01)
                # START RECORDING
                elif state == "start" or state == "recording":
                    obs = env.get_observation()
                    env.exec_actions(action, timestamps=time.time()+dt, compensate_latency=True)
                    data = {}
                    data['obs'] = deepcopy(obs)
                    data['act_delta'] = get_act_delta(action, obs)
                    data['act_abs'] = action
                    data['control_enabled'] = controller_state['engaged']
                    data_collector.collect(data)

                # STOP RECORDING
                elif state == "stop":
                    # Resetting robot
                    data_collector.end_episode(success=True)
                    obs = env.reset()
                    # time.sleep(3.5) # parallel env, wait for reset to finish
                    # obs = env.get_observation()
                    agent.reset_internal_state(obs['eef_pos'][-1], obs['eef_quat'][-1])

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
                    obs = env.reset()
                    # time.sleep(3.5)
                    # obs = env.get_observation()
                    print(f"reset internal state to {obs['eef_pos'][-1]}")
                    agent.reset_internal_state(obs['eef_pos'][-1], obs['eef_quat'][-1])
                    state = "idle"
                else:
                    raise ValueError(f"Invalid state {state}")

            else:
                # Just step action without saving
                env.step(action)

            if state != "idle":
                freq = 1 / (time.time() - t_start)
                freqs.append(freq)

            # regulate frequency in parallel env
            precise_sleep(t_start+dt-time.time())

    except KeyboardInterrupt as e:
        print("Caught Ctrl+C,  save loop")
        env.close()

if __name__ == "__main__":
    main(tyro.cli(Args))
