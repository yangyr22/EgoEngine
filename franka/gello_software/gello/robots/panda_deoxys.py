import time
from typing import Dict
import os
import sys
import pathlib
from collections import namedtuple

import numpy as np
import pybullet as p
path_to_pybullet_planning = os.path.expanduser("~/vaibhav/git/pybullet-planning")
sys.path.insert(0, path_to_pybullet_planning)
from pybullet_tools.utils import get_movable_joints, set_joint_positions, link_from_name

from gello.robots.robot import Robot
from gello.pymodbus_robotiq.robotiq_2f_gripper import Robotiq2FingerGripper

import deoxys
from deoxys.utils import YamlConfig, transform_utils
from deoxys import config_root
from deoxys.experimental.motion_utils import joint_interpolation_traj

FILE_PATH = pathlib.Path(__file__).parent.absolute()

MAX_OPEN = 0.09

def mat2quat(rmat, precise=False):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat: 3x3 rotation matrix
        precise: If isprecise is True, the input matrix is assumed to be a precise
             rotation matrix and a faster algorithm is used.

    Returns:
        vec4 float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    if precise:
        # This code uses a modification of the algorithm described in:
        # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        # which is itself based on the method described here:
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        # Altered to work with the column vector convention instead of row vectors
        m = M.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]
        q = np.array(q)
        q *= 0.5 / np.sqrt(t)
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


class PandaRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
            self, 
            controller_type,
            gripper_type="franka",
            robot_cfg_file="charmander.yml"
            # robot_ip: str = "100.97.47.74",
        ):
        # from polymetis import GripperInterface, RobotInterface
        from deoxys.franka_interface import FrankaInterface

        # self.robot = RobotInterface(
        #     ip_address=robot_ip,
        # )
        # self.gripper = GripperInterface(
        #     ip_address="localhost",
        # )

        self.robot_interface = FrankaInterface(
            general_cfg_file=os.path.join(config_root, robot_cfg_file), # TODO check
            control_freq=20,
            state_freq=100,
            control_timeout=1.0,
            has_gripper=True,
            use_visualizer=False,
        )
        self.gripper_type = gripper_type

        if self.gripper_type == "franka":
            self.gripper_interface = None
        elif self.gripper_type == "robotiq":
            self.gripper_interface = Robotiq2FingerGripper()


        self.controller_type = controller_type
        if controller_type == "OSC_POSE":
            self.controller_cfg_file = os.path.join(config_root, "osc-pose-controller.yml") # TODO check
        elif controller_type == "JOINT_IMPEDANCE":
            self.controller_cfg_file = os.path.join(config_root, "joint-impedance-controller.yml") # TODO check
        else:
            raise NotImplementedError
        self.controller_cfg = YamlConfig(self.controller_cfg_file).as_easydict()
        
        # Using joint position control to reset robot
        self.reset_controller_type = "JOINT_POSITION"
        self.reset_controller_cfg = YamlConfig(os.path.join(config_root, "joint-position-controller.yml")).as_easydict()

        # Golden resetting joints
        self.reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

        # For FK
        disp_option = p.DIRECT # p.GUI
        client = p.connect(disp_option)
        self.bullet_panda = p.loadURDF(os.path.expanduser("/home/mbronars/vaibhav/git/deoxys_control/deoxys/deoxys/franka_interface/robot_models/panda/panda_marker.urdf"), useFixedBase=True) # TODO
        self.movable_joints = get_movable_joints(self.bullet_panda)[:7]
        self.tool_link = link_from_name(self.bullet_panda, "panda_grasptarget")

        self.reset() # self.robot.go_home()

        # self.robot.start_joint_impedance()
        # self.gripper.goto(width=MAX_OPEN, speed=255, force=255)
        # time.sleep(1)

    def _get_eef_pose_from_jointpos(self, joint_positions):
        """
        Does FK to compute world pose of end-effector.
        """
        set_joint_positions(self.bullet_panda, self.movable_joints, joint_positions[:7])
        tool_state = p.getLinkState(self.bullet_panda, self.tool_link, computeForwardKinematics=True)
        LinkState = namedtuple("LinkState", ["pos", "quat_xyzw", "axis_angle"])
        eef_pose = LinkState(tool_state[0], tool_state[1], tuple(transform_utils.quat2axisangle(np.array(tool_state[1]))))
        return eef_pose

    def _get_eef_pose_from_jointpos_as_tuple(self, joint_positions):
        """
        Does FK to compute world pose of end-effector.
        """
        eef_pose = self._get_eef_pose_from_jointpos(joint_positions)
        return (eef_pose.pos, eef_pose.quat_xyzw, eef_pose.axis_angle)
    
    def reset(self):
        self.robot_interface.reset()
        time.sleep(1.0)
        print("restarting the robot interface")

        action = self.reset_joint_positions + [-1.0]

        assert self.reset_controller_type == "JOINT_POSITION", self.reset_controller_type

        # while True:
        #     if len(self.robot_interface._state_buffer) > 0:
        #         if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(self.reset_joint_positions))) < 1e-3:
        #             break
        #     self.robot_interface.control(
        #         controller_type=self.reset_controller_type,
        #         action=action,
        #         controller_cfg=self.reset_controller_cfg,
        #     )

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                # logger.info(f"Current Robot joint: {np.round(self.robot_interface.last_q, 3)}")
                # logger.info(f"Desired Robot joint: {np.round(self.robot_interface.last_q_d, 3)}")

                if (
                    np.max(
                        np.abs(
                            np.array(self.robot_interface._state_buffer[-1].q)
                            - np.array(self.reset_joint_positions)
                        )
                    )
                    < 1e-3
                ):
                    break
            self.robot_interface.control(
                controller_type=self.reset_controller_type,
                action=action,
                controller_cfg=self.reset_controller_cfg,
            )
        
        # curr_joints = np.array(self.robot_interface._state_buffer[-1].q)
        # max_delta = (np.abs(curr_joints - self.reset_joint_positions)).max()
        # steps = min(int(max_delta / 0.01), 100)
        # for jnt in np.linspace(curr_joints, self.reset_joint_positions, steps):
        #     self.command_joint_state(np.append(jnt, -1.))
        #     time.sleep(0.001)

        # We added this sleep here to give the C++ controller time to reset from joint control mode to no control mode
        # to prevent some issues.
        time.sleep(1.0)
        print("RESET DONE")

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return 8

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # robot_joints = self.robot.get_joint_positions()
        robot_joints = self.robot_interface._state_buffer[-1].q
        # print("robot_joints:", robot_joints)

        # # gripper_pos = self.gripper.get_state()
        # gripper_pos = self.robot_interface._gripper_state_buffer[-1]
        # print("gripper_pos:", gripper_pos)
        # pos = np.append(robot_joints, gripper_pos.width / MAX_OPEN)

        if self.gripper_type == "franka":
            gripper_width = self.robot_interface._gripper_state_buffer[-1].width
        elif self.gripper_type == "robotiq":
            gripper_width = self.gripper_interface.get_gripper_act()


        jointpos = np.append(robot_joints, gripper_width)
        ##XXXTODO for gello 0 is open and 1 is closed

        return jointpos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        # import torch

        # self.robot.update_desired_joint_positions(torch.tensor(joint_state[:-1]))
        # self.gripper.goto(width=(MAX_OPEN * (1 - joint_state[-1])), speed=1, force=1)

        if self.controller_type == "JOINT_IMPEDANCE":
            # assert self.controller_type == "JOINT_IMPEDANCE", self.controller_type

            last_q = np.array(self.robot_interface.last_q)
            joint_traj = joint_interpolation_traj(start_q=last_q, end_q=joint_state[:-1])

            print("command joint state:", joint_state)
            # while True:
            #     if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(joint_state[:-1]))) < 8e-3:
            #         break
            #     self.robot_interface.control(
            #         controller_type=self.reset_controller_type,
            #         action=list(joint_state),
            #         controller_cfg=self.reset_controller_cfg,
            #     )
            # for joint in joint_traj:
            #     action = joint.tolist() + [joint_state[-1]]
            #     self.robot_interface.control(
            #         controller_type=self.controller_type,
            #         action=action,
            #         controller_cfg=self.controller_cfg
            #     )
            if joint_state[-1] < 0.01: # ~0
                action = np.array(list(joint_state[:-1])+[-1.])
            else:
                action = np.array(list(joint_state[:-1])+[0.])
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg
            )
            return (joint_state,)
        elif self.controller_type == "OSC_POSE":
            current_pose = self._get_eef_pose_from_jointpos(self.get_joint_state()[:7]) # TODO read from robot state

            #TODO check below
            # import pdb; pdb.set_trace()
            last_robot_state = self.robot_interface._state_buffer[-1]
            ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
            ee_pos = ee_pose[:3, 3]
            ee_quat = mat2quat(ee_pose[:3, :3])
            target_pose = self._get_eef_pose_from_jointpos(joint_state)

            # import pdb; pdb.set_trace()
            # quat_diff = transform_utils.quat_distance(target_pose.quat_xyzw, current_pose.quat_xyzw)
            quat_diff = transform_utils.quat_distance(target_pose.quat_xyzw, ee_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff).tolist()
            # action_pos = ((np.array(target_pose.pos) - np.array(current_pose.pos)) * 10).tolist()
            action_pos = ((np.array(target_pose.pos) - np.array(ee_pos)) * 10).tolist()
            action = action_pos + axis_angle_diff

            # adding gripper_act to action (for franka gripper)
            if joint_state[-1] < 0.01: # ~0 for open
                gripper_act = -1.
                action = np.array(action+[gripper_act])
            else: # close
                gripper_act = 0.
                action = np.array(action+[gripper_act])
            
            # Call interface to command robot
            if self.gripper_type == "franka":
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg
                )
            elif self.gripper_type == "robotiq":
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg
                )
                robotiq_grasp_act = 2*gripper_act + 1 # [-1,1]
                self.gripper_interface.grasp(robotiq_grasp_act)
            else:
                raise NotImplementedError

            # Change gripper_act to [-1,1] for storing to data
            gripper_act = 2*gripper_act + 1. # -1 for open, 1 for close

            print(f"Command absolute action: {list(target_pose.pos)}")
            last_robot_state = self.robot_interface._state_buffer[-1]
            ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
            ee_pos = ee_pose[:3, 3]

            print(f"Actual ee pos: {ee_pos}")

            return (action[:-1].tolist()+[gripper_act], list(target_pose.pos)+list(target_pose.axis_angle)+[gripper_act]) # action_delta, action_abs

        print("DONE COMMAND")

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        # pos_quat = np.zeros(7) ## unused?
        eef_pos_quat = self._get_eef_pose_from_jointpos(joints)
        eef_pos = np.array(eef_pos_quat.pos)
        eef_quat = np.array(eef_pos_quat.quat_xyzw)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints, # unused?
            # "ee_pos_quat": pos_quat,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "gripper_position": gripper_pos,
        }


def main():
    robot = PandaRobot()
    current_joints = robot.get_joint_state()
    # move a small delta 0.1 rad
    move_joints = current_joints + 0.05
    # make last joint (gripper) closed
    move_joints[-1] = 0.5
    time.sleep(1)
    m = 0.09
    robot.gripper.goto(1 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.05 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.1 * m, speed=255, force=255)
    time.sleep(1)


if __name__ == "__main__":
    main()
