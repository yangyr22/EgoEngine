# python experiments/run_env.py --agent=gello --save_hdf5 --agentview_camera_port 5002TODO --wrist_camera_port 5003TODO --shoulderview_right_camera_port 5005

import datetime
import glob
import time
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import h5py
import os
import json

import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    agentview_camera_port: int = 5002
    wrist_camera_port: int = 5003
    base_camera_port: int = 5002
    shoulderview_left_camera_port: int = 5004
    shoulderview_right_camera_port: int = 5005
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    robot_controller: str = "osc_pose" # "joint_impedance" ## make sure this matches with the spun-up robot node

    gello_port: Optional[str] = None
    mock: bool = False
    save_pkl: bool = False # use_save_interface: bool = False
    save_hdf5: bool = False
    # data_dir: str = "/home/mbronars/Documents/legibility"
    # data_dir: str = "/media/robot/0e230ee7-e486-472e-8972-20b4e9e4cf0f/bc_data/legibility/gello/demos/make_dinner_2/" # provide save dir here, keep it as bc_data/rmv2, but pass in args.task for new tasks
    data_dir: str = "/media/robot/0e230ee7-e486-472e-8972-20b4e9e4cf0f/bc_data/rmv2/gello/" # provide save dir here
    task: str = None
    bimanual: bool = False
    verbose: bool = False


def main(args):
    if not args.task:
        args.task = str(datetime.date.today())
    
    data_save_dir = os.path.join(str(Path(args.data_dir).expanduser()), args.agent, "demos", args.task)
    if args.save_hdf5: 
        os.makedirs(data_save_dir, exist_ok=True)

    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            "agentview": ZMQClientCamera(port=args.agentview_camera_port, host=args.hostname),
            "eye_in_hand": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
            "shoulderview_left": ZMQClientCamera(port=args.shoulderview_left_camera_port, host=args.hostname),
            "shoulderview_right": ZMQClientCamera(port=args.shoulderview_right_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
    # TODO add support for env that controls two robot arms using 14-dof action

    task_description = input("Enter description of the task: ")

    #create metadata for saving #TODO maybe move this somewhere else for better setting of the metadata
    ENV_ARGS = {
        "env_name": "EnvRealPandaDeoxys",
        "type": 4,
        'lang': task_description,
        "env_kwargs": {
            'camera_names_to_sizes':
                {'agentview': (128, 128), 'eye_in_hand': (128, 128), "shoulderview_left" : (128, 128), "shoulderview_right" : (128, 128)},
            'general_cfg_file': None,
            'control_freq': 20,
            'controller_type': 'OSC_POSE',
            'controller_cfg_file': None,
            'controller_cfg_dict': None,
            'use_depth_obs': False, 'state_freq': 100.0, 'control_timeout': 1.0,
            'has_gripper': True, 'use_visualizer': False,
            'gripper_type':'robotiq'
        }
    }

    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8J0SM3-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISNDP-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        # reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        # reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints_left = np.deg2rad([0, 0, 0, -90, 0, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, 0, 0, -90, 0, 90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
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
                # reset_joints = np.deg2rad(
                #     [0, -90, 90, -90, -90, 0, 0]
                # )  # Change this to your own reset joints
                reset_joints = np.deg2rad(
                    [0, 0, 0, -90, 0, 90, 0, 0]
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
                print("Resetting robot joints to:", reset_joints)
                env.step(reset_joints)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
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
    start_pos = agent.act(env.get_obs()) # returns joint state of gello
    obs = env.get_obs() # returns joint state of the target robot (or sim)
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    # Take gello to current robot position
    # set gello to torque mode
    [ 0.01420327, -0.05111616, -0.01780655, -1.96158285, -0.02636951,
        1.93180636, -0.79413254,  0.        ]
    
    agent.set_torque_mode(True)
    target_jointpos_for_gello = env.get_obs()["joint_positions"]
    max_delta = 0.05
    for _ in range(50):
        command_joints = target_jointpos_for_gello
        current_joints = agent.act(None)
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta: # should move
            delta = delta / max_joint_delta * max_delta
        agent.step(current_joints + delta)
        time.sleep(0.001)

    
    print(f"Action position:{env.get_obs()}")

    
    # turn of gello torque mode
    input("Gello set to robot position. Hold it in position and press the Enter key to continue: ") 
    # agent.set_torque_mode(False)

    


    # Take robot to current gello position
    max_delta = 0.05
    for _ in range(10):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    # obs = env.get_obs()
    # joints = obs["joint_positions"]
    # action = agent.act(obs)
    # import pdb; pdb.set_trace()
    # if (action - joints > 0.5).any():
    #     print("Action is too big")

    #     # print which joints are too big
    #     joint_index = np.where(action - joints > 0.8)
    #     for j in joint_index:
    #         print(
    #             f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
    #         )
    #     exit()

    if args.save_pkl or args.save_hdf5: # args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset
        # Pressing "s" starts recording, "q" stops recording

        kb_interface = KBReset()

        if args.save_hdf5:
            transition_count = 0
            flush_freq = 50
            obs_buffer = []
            # act_buffer = []
            act_delta_buffer = []
            act_abs_buffer = []

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    ############################ HELPERS ############################
    def _convert_action_for_saving(action, controller_type, env):
        if controller_type == "joint_impedance":
            return action
        elif controller_type == "osc_pose":
            eef_pose = env._robot._get_eef_pose_from_jointpos_as_tuple(action)
            return list(eef_pose[0]) + list(eef_pose[2]) + [action[-1]]
        else:
            raise NotImplementedError
        
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

    save_path = None
    start_time = time.time()
    curr_time = time.time()
    try:
        obs = env.get_obs()
        while True:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            # print_color(
            #     message,
            #     color="white",
            #     attrs=("bold",),
            #     end="",
            #     flush=True,
            # )
            action = agent.act(obs)
            dt = datetime.datetime.now()
            # if args.save_pkl: # args.use_save_interface # TODO remove, probably doesn't work anymore
            #     state = kb_interface.update()
            #     #TODO add correct save path below
            #     if state == "start":
            #         dt_time = datetime.datetime.now()
            #         save_path = (
            #             Path(args.data_dir).expanduser()
            #             / args.agent
            #             / dt_time.strftime("%m%d_%H%M%S")
            #         )
            #         save_path.mkdir(parents=True, exist_ok=True)
            #         print(f"Saving to {save_path}")
            #     elif state == "save":
            #         assert save_path is not None, "something went wrong"
            #         save_frame(save_path, dt, obs, action)
            #     elif state == "normal":
            #         save_path = None
            #     else:
            #         raise ValueError(f"Invalid state {state}")
            if args.save_hdf5:
                state = kb_interface.update()
                # START
                if state == "start":
                    agent.set_torque_mode(False)
                    dt_time = datetime.datetime.now()
                    save_path = os.path.join(data_save_dir,
                        dt_time.strftime("%m%d_%H%M%S")+".hdf5"
                    )
                    # save_path.mkdir(parents=True, exist_ok=True)
                    f = h5py.File(save_path, "w")
                    f_grp = f.create_group("data")
                    # f_grp.attrs["timestamp"] = timestamp
                    ep_data_grp = f_grp.create_group("demo_0")
                    ep_data_grp.attrs["env_args"] = json.dumps(ENV_ARGS)

                    print(f"Saving to {save_path}")
                    obs = env.get_obs()
                    action = agent.act(obs)
                    obs_buffer = [obs]
                    obs, (act_delta, act_abs) = env.step(action)
                    act_delta_buffer.append(act_delta) ##
                    act_abs_buffer.append(act_abs)
                    # act_buffer = [_convert_action_for_saving(action, args.robot_controller, env)] #### TODO
                # RECORD TRAJECTORY
                elif state == "save":
                    print("gp0:", env._robot.get_observations()["gripper_position"])
                    transition_count += 1
                    assert save_path is not None, "something went wrong"
                    # save_frame(save_path, dt, obs, action)
                    if transition_count % flush_freq == 0:
                        # import pdb; pdb.set_trace()
                        _flush_buffer_to_disk(ep_data_grp, obs_buffer, act_delta_buffer, act_abs_buffer, transition_count, flush_freq)
                        # resetting buffers
                        obs_buffer = []
                        # act_buffer = []
                        act_delta_buffer = []
                        act_abs_buffer = []
                    else:
                        # obs_buffer.append(obs)
                        # act_buffer.append(_convert_action_for_saving(action, args.robot_controller, env))
                        obs_buffer.append(obs)
                        obs, (act_delta, act_abs) = env.step(action)
                        act_delta_buffer.append(act_delta) ##
                        act_abs_buffer.append(act_abs)
                # SABOTAGE MODE
                elif state == "sabotage":
                    # Follow teleop but do not store to buffer.
                    obs, _ = env.step(action)
                # STOP RECORDING
                elif state == "normal":
                    if len(obs_buffer) > 0:
                        _flush_buffer_to_disk(ep_data_grp, obs_buffer, act_delta_buffer, act_abs_buffer, transition_count, flush_freq)
                        # resetting buffers
                        obs_buffer = []
                        # act_buffer = []
                        act_delta_buffer = []
                        act_abs_buffer = []

                        # Resetting robot
                        curr_joints = env.get_obs()["joint_positions"]
                        if reset_joints.shape == curr_joints.shape:
                            max_delta = (np.abs(curr_joints - reset_joints)).max()
                            steps = min(int(max_delta / 0.01), 100)

                            for jnt in np.linspace(curr_joints, reset_joints, steps):
                                # gripper_position = np.deg2rad(5.0) # set gripper to fully open immediately
                                jnt[-1] = 0
        
                                env.step(jnt)
                                time.sleep(0.001)
                            print("Resetting robot joints to:", reset_joints)
                            env.step(reset_joints)
                        
                        ####### Resetting gello #######
                        # set gello to torque mode
                        agent.set_torque_mode(True)
                        target_jointpos_for_gello = env.get_obs()["joint_positions"]
                        max_delta = 0.05
                        for _ in range(25):
                            command_joints = target_jointpos_for_gello
                            current_joints = agent.act(None)
                            delta = command_joints - current_joints
                            max_joint_delta = np.abs(delta).max()
                            if max_joint_delta > max_delta: # should move
                                delta = delta / max_joint_delta * max_delta
                            agent.step(current_joints + delta)
                            time.sleep(0.001)
                        print("Resetting gello")
                        # turn of gello torque mode
                        # input("Gello set to robot position. Hold it in position and press the Enter key to continue: ") 
                        # agent.set_torque_mode(False)
                        ################################

                    save_path = None
                    transition_count = 0 # start new demo
                    
                else:
                    raise ValueError(f"Invalid state {state}")
                # obs, (act_delta, act_abs) = env.step(action)
                # # if len(act_buffer) == 0:
                # act_delta_buffer.append(act_delta) ##
                # act_abs_buffer.append(act_abs)

            print(1/(time.time() - curr_time), "Hz")
            curr_time = time.time()
    except KeyboardInterrupt as e:
        print("Caught Ctrl+C, releasing gello torque and breaking save loop")
        agent.set_torque_mode(False)


if __name__ == "__main__":
    main(tyro.cli(Args))
