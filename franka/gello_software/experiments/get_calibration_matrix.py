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

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.rl2_env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.robots.panda_deoxys_simple import PandaRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.cameras.zed_camera import ZedCamera
from gello.cameras.realsense_camera import RealSenseCamera
import pyzed.sl as sl
import quaternion
import itertools

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)

def generate_transformation_matrices():
    base_permutations = list(itertools.permutations([1, 0, 0])) + \
                        list(itertools.permutations([0, 1, 0])) + \
                        list(itertools.permutations([0, 0, 1]))

    # Filter to only take permutations with exactly 3 elements
    base_permutations = [list(p) for p in set(itertools.permutations(base_permutations, 3))]

    matrices = []
    for perm in base_permutations:
        for signs in itertools.product([1, -1], repeat=3):
            signed_perm = [np.array(row) * sign for row, sign in zip(perm, signs)]
            matrix = np.array(signed_perm)
            if (np.abs(matrix).sum(axis=0) == 1).all() and (np.abs(matrix).sum(axis=1) == 1).all():
                matrices.append(matrix)

    return matrices


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
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    robot_controller: str = "osc_pose" # "joint_impedance" ## make sure this matches with the spun-up robot node

    gello_port: Optional[str] = None
    mock: bool = False
    save_pkl: bool = False # use_save_interface: bool = False
    save_hdf5: bool = True
    # data_dir: str = "/home/aloha/nadun/Demos/experiment_1" # provide save dir here
    data_dir: str = "/media/aloha/Data/robomimic-v2/Demos/Experiment-Collector/put_screwdriver_in_drawer"  # provide save dir here
    task: str = None
    # Dont change below
    bimanual: bool = False
    verbose: bool = False


def _flush_buffer_to_disk(ep_data_grp, obs_buffer, act_delta_buffer, act_abs_buffer, transition_count, flush_freq):
    chunk_count = int((transition_count - 1) / flush_freq)
    # flushing observations
    for k in obs_buffer[0]:
        obs_chunk_to_flush = np.stack([obs_buffer[i][k] for i in range(len(obs_buffer))], 0)
        ep_data_grp.create_dataset(f"chunk_{chunk_count}/obs/{k}", data=obs_chunk_to_flush)
    # flushing actions
    act_chunk_to_flush = np.stack(act_delta_buffer)
    ep_data_grp.create_dataset(f"chunk_{chunk_count}/action", data=act_chunk_to_flush)
    act_chunk_to_flush = np.stack(act_abs_buffer)
    ep_data_grp.create_dataset(f"chunk_{chunk_count}/action_absolute", data=act_chunk_to_flush)
    # print(f"\nFlushed {len(act_chunk_to_flush)} transitions to disk.")
##################################################################


def main(args):
    # TODO fix to not need keyboard interface
    if not args.task:
        args.task = str(datetime.date.today())

    # data_save_dir = os.path.join(str(Path(args.data_dir).expanduser()), args.agent, "demos", args.task)
    data_save_dir = os.path.join(str(Path(args.data_dir).expanduser()), args.task)
    if args.save_hdf5:
        os.makedirs(data_save_dir, exist_ok=True)

    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        # camera_clients = {}
        cam_dict = {}
    else:
        ### Starting the Zed cameras
        cameras = sl.Camera.get_device_list()
        cam_list = []
        cam_dict = {}

        # TODO make the setting of the camera dict cleaner
        # for cam in cameras:
        #     if cam.serial_number == args.shoulderview_left_sn:
        #         cam_dict["shoulderview_left"] = ZedCamera(cam)
        #     elif cam.serial_number == args.shoulderview_right_sn:
        #         cam_dict["shoulderview_right"] = ZedCamera(cam)
        #     elif cam.serial_number == args.wrist_camera_sn:
        #         cam_dict["wrist"] = ZedCamera(cam)
        #
        # ### Starting Realsense cameras
        #
        # cam_dict["left"] = RealSenseCamera(device_id=args.left_camera_sn)
        # cam_dict["right"] = RealSenseCamera(device_id=args.right_camera_sn)


        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),

        }
        # robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        robot_client = PandaRobot("OSC_POSE", gripper_type="robotiq")
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=cam_dict, save_depth_obs=args.save_depth_obs)

    # task_description = input("Enter description of the task: ")
    task_description = "Test"

    # create metadata for saving #TODO maybe move this somewhere else for better setting of the metadata
    ENV_ARGS = {
        "env_name": "EnvRealPandaDeoxys",
        "type": 4,
        'lang': task_description,
        "env_kwargs": {
            # 'camera_names_to_sizes':
            #     {'shoulderview_left': (128, 128),
            #      'shoulderview_right': (128, 128),
            #      "left": (128, 128),
            #      "right": (128, 128)},
            # 'camera_type': args.camera_type,
            # 'camera_names_to_sn': {"shoulderview_left": args.shoulderview_left_sn,
            #                        "shoulderview_right": args.shoulderview_right_sn,
            #                        "left": args.left_camera_sn,
            #                        "right": args.right_camera_sn},

            "camera_dict" : {'shoulderview_left': {'size' : (128, 128), 'sn' : args.shoulderview_left_sn, 'type': "Zed"},
                             'shoulderview_right': {'size': (128, 128), 'sn': args.shoulderview_right_sn, 'type': "Zed"},
                             'wrist' : {'size': (128, 128), 'sn': args.wrist_camera_sn, 'type': "Zed"},
                             'left': {'size': (128, 128), 'sn': args.left_camera_sn, 'type': "RealSense"},
                             'right': {'size': (128, 128), 'sn': args.right_camera_sn, 'type': "RealSense"},
                             },
            # "camera_dict" : {},
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
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    robot_client.reset()
    obs = env.get_obs()

    if args.save_pkl or args.save_hdf5:  # args.use_save_interface:


        from gello.data_utils.demo_collection_screen import Screen

        screen = Screen()

        if args.save_hdf5:
            transition_count = 0
            flush_freq = 100
            obs_buffer = []
            # act_buffer = []
            act_delta_buffer = []
            act_abs_buffer = []

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    start_time = time.time()
    curr_time = time.time()

    # state can be [idle, start, recording, stop, delete]
    state = "idle"


    # Create a data collector
    if args.save_hdf5:
        from gello.data_utils.data_collector import DataCollector
        data_collector = DataCollector(data_save_dir)

    try:
        obs = env.get_obs()
        freqs = []

        matrices = generate_transformation_matrices()
        matrix_counter = 0
        matrix = matrices[matrix_counter]
        agent.controller_offset = quaternion.from_rotation_matrix(matrix)
        print(f"This is the matrix: {matrices[matrix_counter]}")

        while True:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "


            controller_state = agent.get_controller_state()
            if controller_state is None:
                continue
            action = controller_state['target_pose'] + controller_state['gripper_act']


            if controller_state['engaged'] and state == "idle":
                state = "start"
                agent.controller_offset = quaternion.from_rotation_matrix(matrix)
                print(f"This is the matrix: {matrix}")
            elif state == "start":
                state = "recording"
                screen.update_state(state)
                data_collector.start_episode(ENV_ARGS)
            elif state == "recording" and controller_state['save_demo']:
                state = "stop"
                screen.update_state(state)
            elif state == "recording" and controller_state['delete_demo']:
                state = "delete"
                matrix_counter += 1
                matrix = matrices[matrix_counter]

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
                    data_collector.collect(data)


                # RECORD TRAJECTORY
                elif state == "recording":
                    data = {}
                    data['obs'] = deepcopy(obs)
                    obs, (act_delta, act_abs) = env.step(action)
                    data['act_delta'] = act_delta
                    data['act_abs'] = act_abs
                    data_collector.collect(data)

                # STOP RECORDING
                elif state == "stop":

                    # Resetting robot
                    data_collector.end_episode(success=True)
                    env.robot().reset()
                    agent.reset_internal_state()

                    state = "idle"

                    # print freq for episode
                    min_f = min(freqs)
                    max_f = max(freqs)
                    avg_f = sum(freqs) / len(freqs)

                    print(f"Episode average frequency: {avg_f} . Max: {max_f}. Min: {min_f}")
                    freqs = []


                # DELETE DEMO
                elif state == "delete":

                    # obs_buffer = []
                    # act_delta_buffer = []
                    # act_abs_buffer = []
                    data_collector.end_episode(success=False)
                    env.robot().reset()
                    agent.reset_internal_state()
                    # print(f"Deleting demo at {save_path}")
                    # if os.path.exists(save_path):
                    #     os.remove(save_path)
                    #
                    # save_path = None
                    # transition_count = 0
                    state = "idle"

                else:
                    raise ValueError(f"Invalid state {state}")

            else:
                # Just step action without saving
                obs = env.step(action)
            if state != "idle":
                freq = 1 / (time.time() - curr_time)
                freqs.append(freq)
                # print(freq, "Hz")
                curr_time = time.time()


    except KeyboardInterrupt as e:
        print("Caught Ctrl+C,  save loop")
        env.close()
        # agent.set_torque_mode(False)



if __name__ == "__main__":
    main(tyro.cli(Args))
