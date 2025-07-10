import time
import numpy as np
from gello.robots.panda_deoxys_simple import PandaRobot
from deoxys.utils import transform_utils

def is_at_origin(position, tolerance=1e-3):
    """Check if the given position is close to the origin within a tolerance."""
    return np.allclose(position, [0, 0, 0], atol=tolerance)

def main():
    # Initialize the Franka Panda Robot
    robot_client = PandaRobot("OSC_POSE", gripper_type="franka")
    
    # Reset the robot to its home position
    robot_client.reset()
    time.sleep(2)  # Allow time for movement
    
    # Get the current end-effector position
    ee_pose_matrix = robot_client.robot_interface.last_eef_pose
    ee_position = ee_pose_matrix[:3, 3]  # Extract XYZ coordinates

    # Check if the robot is at the origin
    if is_at_origin(ee_position):
        print("✅ The robot's end-effector is at the origin (0,0,0).")
    else:
        print(f"⚠️ The robot's end-effector is NOT at the origin. Current position: {ee_position}")

if __name__ == "__main__":
    main()
