import time
from typing import Dict
import os
import sys
import pathlib
from collections import namedtuple

import numpy as np
# import pybullet as p


from gello.robots.robot import Robot
# from gello.pymodbus_robotiq.robotiq_2f_gripper import Robotiq2FingerGripper

import deoxys
from deoxys.utils import YamlConfig, transform_utils
from deoxys import config_root
from deoxys.experimental.motion_utils import joint_interpolation_traj
from deoxys.franka_interface import FrankaInterface

# FILE_PATH = pathlib.Path(__file__).parent.absolute()

MAX_OPEN = 0.09


class PandaRobot(Robot):
    """A class representing a Franka Panda robot that can use different end effectors."""

    def __init__(self, controller_type, gripper_type):
        config_root = '/mnt/data2/dexmimic/workspace/franka/deoxys_control/deoxys/config'
        print(f"Using config at : {os.path.join(config_root)} charmander.yml")
        self.robot_interface = FrankaInterface(
            general_cfg_file=os.path.join(config_root, "charmander.yml"),
            control_freq=20,
            state_freq=100,
            control_timeout=1.0,
            has_gripper=True,
            use_visualizer=False,
        )

        self.gripper_type = gripper_type

        # Create the appropriate gripper interface
        if self.gripper_type == "umi":
            self.gripper_interface = None
        elif self.gripper_type == "robotiq":
            from gello.pymodbus_robotiq.robotiq_2f_gripper import Robotiq2FingerGripper
            self.gripper_interface = Robotiq2FingerGripper()
        elif self.gripper_type == "ability":
            from gello.ability.ability_2f_gripper import AbilityHandController
            self.gripper_interface = AbilityHandController(
                comport="/dev/ttyUSB1",
                baud_rate=460800,
                stuff_data=False,
                hand_address=0x50,
                reply_mode=0x10
            )
        else:
            raise Exception("No such hand type.")

        self.controller_type = controller_type
        if controller_type == "OSC_POSE":
            self.controller_cfg_file = os.path.join(config_root, "osc-pose-controller-absolute.yml")
        elif controller_type == "JOINT_IMPEDANCE":
            self.controller_cfg_file = os.path.join(config_root, "joint-impedance-controller.yml")
        else:
            raise NotImplementedError
        self.controller_cfg = YamlConfig(self.controller_cfg_file).as_easydict()
        self.delta_control = self.controller_cfg["is_delta"]

        # A separate controller config for joint-position-based resets
        self.reset_controller_type = "JOINT_POSITION"
        self.reset_controller_cfg = YamlConfig(os.path.join(config_root, "joint-position-controller.yml")).as_easydict()

        # Set default "golden" reset joint positions for each end-effector
        self.reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

        self.reset()

    def reset(self):
        """Move the robot to the default 'reset' joint position. Optionally reset the hand, too."""
        if self.gripper_type == "robotiq" and self.gripper_interface is not None:
            self.gripper_interface.grasp(-1)  # e.g. fully open on reset
        elif self.gripper_type == "ability" and self.gripper_interface is not None:
            # Reset the ability hand by opening it
            self.gripper_interface.open_hand()

        self.robot_interface.reset()
        time.sleep(1.0)

        if self.gripper_type == "robotiq" or self.gripper_type == "umi":
            action = self.reset_joint_positions + [-1.0]
        elif self.gripper_type == "ability":
            action = self.reset_joint_positions + [0.,0.,0.,0.,0.,0.]
        elif self.gripper_type == "allegro":
            pass
            
        while True:
            if len(self.robot_interface._state_buffer) > 0:
                current_q = np.array(self.robot_interface._state_buffer[-1].q)
                if np.max(np.abs(current_q[:7] - np.array(self.reset_joint_positions[:7]))) < 5e-3:
                    break
            if self.gripper_type == "umi" or self.gripper_type == "robotiq":
                self.robot_interface.control(
                    controller_type=self.reset_controller_type,
                    action=action,
                    controller_cfg=self.reset_controller_cfg,
                )
            else:
                self.robot_interface.control(
                    controller_type=self.reset_controller_type,
                    action=action[:7]+[-1],
                    controller_cfg=self.reset_controller_cfg,
                )

        time.sleep(1.0)
        print("RESET DONE")

    def num_dofs(self) -> int:
        """Number of active joints + possibly extra for the hand, if you integrate them in one action space."""
        if self.gripper_type in ["umi", "robotiq"]:
            return 8
        elif self.gripper_type == "ability":
            return 13
        elif self.gripper_type == "allegro":
            return 23

    def get_joint_state(self) -> np.ndarray:
        """Return a dictionary with 'joint_positions' and 'joint_velocities' for the robot arm joints."""
        joint_positions = self.robot_interface._state_buffer[-1].q
        joint_velocities = self.robot_interface._state_buffer[-1].dq
        return {
            "joint_positions": np.array(joint_positions),
            "joint_velocities": np.array(joint_velocities)
        }

    def get_gripper_state(self):
        """Return gripper state if available."""
        if self.gripper_type == "umi":
            if len(self.robot_interface._gripper_state_buffer) == 0:
                return None
            return np.array(self.robot_interface._gripper_state_buffer[-1].width)
        elif self.gripper_type == "robotiq" and self.gripper_interface is not None:
            return np.array(self.gripper_interface.get_gripper_act())
        elif self.gripper_type == "ability" and self.gripper_interface is not None:
            return np.array(self.gripper_interface.get_gripper_act())
        else:
            raise Exception("no such gripper mode")

    def step(self, action: np.ndarray) -> tuple:
        """
        Send a command to the robot (and hand) for a single step.
        Args:
            action (np.ndarray): Robot action. For OSC_POSE, it is [x,y,z,qx,qy,qz,qw, gripper_cmd].
        Returns:
            tuple: Relevant command outputs.
        """
        if self.controller_type == "JOINT_IMPEDANCE":
            if action[-1] < 0.01:
                action = np.array(list(action[:-1]) + [-1.])  # e.g., open command
            else:
                action = np.array(list(action[:-1]) + [0.])   # e.g., close command

            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg
            )

            if self.gripper_type == "robotiq" and self.gripper_interface is not None:
                robotiq_grasp_act = action[-1]
                self.gripper_interface.grasp(robotiq_grasp_act)
            elif self.gripper_type == "ability" and self.gripper_interface is not None:
                # Map the gripper command to ability hand actions.
                # For instance, if the command is negative, open the hand; otherwise, close it.
                self.gripper_interface.grasp(robotiq_grasp_act)

            return (action, )

        elif self.controller_type == "OSC_POSE":
            # print(action)
            current_pose = self.robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
            current_quat = transform_utils.mat2quat(current_rot)

            target_pos  = action[0:3]
            target_quat = np.array(action[3:7])
            gripper_act = action[7:]

            # quat_diff = transform_utils.quant_distance(target_quat, current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(
                transform_utils.quat_distance(target_quat, current_quat)
            ).tolist()
            action_pos = (((np.array(target_pos) - np.array(current_pos))) * 10.0).tolist()
            delta_action = action_pos + axis_angle_diff + gripper_act

            if self.delta_control:
                action = delta_action
            else:
                # If we use absolute control. we just need to change the rotation action from quat to axis angle
                target_axis_angle = transform_utils.quat2axisangle(target_quat).tolist()
                gripper_act = action[7:]
                action = target_pos + target_axis_angle + gripper_act

            if self.gripper_type == "umi":
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg
                )

            if self.gripper_type == "robotiq":
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg
                )
                robotiq_grasp_act = gripper_act[0] # [-1, 1]
                self.gripper_interface.grasp(robotiq_grasp_act)

            elif self.gripper_type == "ability":
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg
                )
                self.gripper_interface.grasp(gripper_act)

            return (delta_action, action)

        return ()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Optionally implement a direct joint-state command if needed."""
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of all relevant robot states."""
        joint_state = self.get_joint_state()
        current_pose = self.robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]

        eef_pos = np.squeeze(np.array(current_pos))
        eef_quat = np.squeeze(np.array(transform_utils.mat2quat(current_rot))[:, None])
        eef_angle = transform_utils.quat2axisangle(eef_quat)
        gripper_pos = self.get_gripper_state()

        return {
            "joint_positions": joint_state["joint_positions"],
            "joint_velocities": joint_state["joint_velocities"],
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "eef_axis_angle": eef_angle,
            "eef_pose": np.array(eef_pos.tolist() + eef_quat.tolist()),
            "gripper_position": gripper_pos,
        }



def main():
    robot = PandaRobot("OSC_POSE", gripper_type="robotiq")
    while True:
        current_joints = robot.get_joint_state()
        # robot.reset()
        # robot.gripper_interface.grasp(1)
        # print(f"Current joints are : {current_joints}")
        obs = robot.get_observations()
        print(obs)
        # time.sleep(3)
        # robot.gripper_interface.grasp(-1)
        #
        # time.sleep(1)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
