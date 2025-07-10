import nexusformat.nexus as nx
import numpy as np
import h5py
from collections import defaultdict
import json

demo_fn = "/home/mbronars/nadun/teleop_testing/merged/test_source.hdf5"

demo = nx.nxload(demo_fn)
print(demo.tree)

demo_file = h5py.File(demo_fn)
demo = demo_file['data/demo_0']

gripper_pos = demo["obs/gripper_position"][:]
# absolute_actions = demo['absolute_actions'][:]
# delta_actions = demo['delta_actions'][:]
#
# eef_pos = demo['obs/eef_pos'][:]
# eef_quat = demo['obs/eef_quat'][:]
# eef_angles = demo['obs/eef_axis_angle'][:]
# env_args = json.loads(demo_file['data'].attrs['env_args'])


print()

