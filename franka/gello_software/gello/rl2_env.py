import time
from typing import Any, Dict, Optional
from copy import deepcopy

import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot

# TODO Cleanup this class
class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        save_depth_obs = False
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._save_depth_obs = save_depth_obs

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(action) == (
            self._robot.num_dofs()
        ), f"input:{len(action)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(action)
        action = self._robot.step(action)
        self._rate.sleep()
        return self.get_obs(), action

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            if camera.type == "RealSense":
                data = camera.read()
                observations[f"{name}_rgb"] = data["rgb"]
                observations[f"{name}_image"] = data["rgb"]
                if self._save_depth_obs:
                    observations[f"{name}_depth"] = data["depth"]

            elif camera.type == "Zed":
                data = camera.read()
                observations[f"{name}_rgb_left"] = data["left_image"]
                # Arbitrarily pick left stereo cam image as the "image" obs for the camera
                observations[f"{name}_image"] = data["left_image"]
                observations[f"{name}_rgb_right"] = data["right_image"]
                if self._save_depth_obs:
                    observations[f"{name}_depth_left"] = data["left_depth"]
                    observations[f"{name}_depth_right"] = data["right_depth"]

            elif camera.type == "Kinect":
                data = camera.read()
                observations[f"{name}_image"] = data['rgb']
                if self._save_depth_obs:
                    observations[f"{name}_depth"] = data["depth"]
            else:
                raise Exception("Incorrect camera type set. Check camera!")

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        # assert "joint_velocities" in robot_obs
        # assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["eef_pos"] = robot_obs["eef_pos"]
        observations["eef_quat"] = robot_obs["eef_quat"]
        observations["eef_axis_angle"] = robot_obs["eef_axis_angle"]
        observations["eef_pose"] = robot_obs["eef_pose"]
        # observations["joint_velocities"] = robot_obs["joint_velocities"]
        # observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]

        # TODO Hardcoded asserts, change to based on camera dict
        # assert "shoulderview_left_image" in observations, "LEFT ZED CAMERA NOT STREAMING, RECONNECT!"
        # assert "shoulderview_right_image" in observations, "RIGHT ZED CAMERA NOT STREAMING, RECONNECT!"
        # assert "wrist_image" in observations, "WRIST CAMERA NOT STREAMING, RECONNECT!"
        # assert "left_image" in observations, "LEFT REALSENSE CAMERA NOT STREAMING, RECONNECT!"
        # assert "right_image" in observations, "RIGHT REALSENSE CAMERA NOT STREAMING, RECONNECT!"

        # TODO setting shoulderview_right as the agentview for the Campose experiments, remove later

        # observations['selected_agentview_image'] = deepcopy(observations['shoulderview_right_image'])
        return observations
    
    def close(self):
        # TODO add other cleanup things as needed
        for _, camera in self._camera_dict.items():
            camera.close()


def main() -> None:
    pass


if __name__ == "__main__":
    main()
