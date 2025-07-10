import nexusformat.nexus as nx
import h5py
import time

import numpy as np
from gello.robots.panda_deoxys_simple import PandaRobot
from deoxys.utils import transform_utils
import cv2

demo_fn = "/home/aloha/nadun/Demos/experiment_1/merged/move_to_skynet/clear_counter_demo.hdf5"
demo_file = nx.nxload(demo_fn)
render = True

if render:
    cv2.namedWindow("camera_views")

print(demo_file.tree)

demo_file = h5py.File(demo_fn)
data = demo_file['data']
actions = demo_file['data/demo_0/absolute_actions'][:]
env_kwargs = demo_file['data'].attrs['env_args']

### Start robot
robot = PandaRobot('OSC_POSE', gripper_type='robotiq')
for ep in data:
    demo = data[ep]
    actions = demo['absolute_actions'][:]
    if render:
        shoulderview_left_image = demo['obs/shoulderview_left_image'][:]
        shoulderview_right_image = demo['obs/shoulderview_right_image'][:]
    input("Press enter to play next demo")
    for i in range(actions.shape[0]):
        act = actions[i]
        act_pos = act[0:3].tolist()
        act_axis_angles = act[3:6]
        act_quat = transform_utils.axisangle2quat(act_axis_angles).tolist()
        act_gripper = act[-1].tolist()
        act = act_pos + act_quat + [act_gripper]
        robot.step(act)
        if render:
            l = shoulderview_left_image[i]
            r = shoulderview_right_image[i]
            img = np.concatenate([l, r], axis=1)
            cv2.waitKey(1)
            cv2.imshow("camera_views", img)
        time.sleep(0.05)
    robot.reset()

