import h5py
import nexusformat.nexus as nx
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os

demo_fn = "/media/robot/0e230ee7-e486-472e-8972-20b4e9e4cf0f/bc_data/rmv2/gello/merged_demos/put_cube_in_bowl/put_cube_in_bowl_demo.hdf5"
video_folder = "/media/robot/0e230ee7-e486-472e-8972-20b4e9e4cf0f/bc_data/rmv2/gello/demo_videos"

output_folder_name = "put_cube_in_bowl"

video_folder = os.path.join(video_folder,output_folder_name)
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

demo_file = h5py.File(demo_fn)


demos = demo_file['data']

wrist_video_list = []
front_video_list = []
shoulderview_left_video_list = []
shoulderview_right_video_list = []

for j, demo in enumerate(demos):
    print(f"Processing demo: {j}")
    wrist_camera_images = demos[demo]['obs/eye_in_hand_image']
    front_camera_images = demos[demo]['obs/agentview_image']
    shoulderview_left_images = demos[demo]['obs/shoulderview_left_image']
    shoulderview_right_images = demos[demo]['obs/shoulderview_right_image']
    for i in range(3):
        

        wrist_video_list.append(cv2.cvtColor(wrist_camera_images[i], cv2.COLOR_BGR2RGB))
        front_video_list.append(cv2.cvtColor(front_camera_images[i], cv2.COLOR_BGR2RGB))
        shoulderview_left_video_list.append(cv2.cvtColor(shoulderview_left_images[i], cv2.COLOR_BGR2RGB))
        shoulderview_right_video_list.append(cv2.cvtColor(shoulderview_right_images[i], cv2.COLOR_BGR2RGB))

wrist_video_fn = os.path.join(video_folder, "wrist_view_all_demos_initial_frames.mp4")
front_video_fn = os.path.join(video_folder, "frontview_all_demos_initial_frames.mp4")
shoulderview_left_video_fn = os.path.join(video_folder, "shoulderview_left_all_demos_initial_frames.mp4")
shoulderview_right_video_fn = os.path.join(video_folder, "shoulderview_right_all_demos_initial_frames.mp4")

width = wrist_video_list[1].shape[0]
height = wrist_video_list[1].shape[1]
fps = 20

wrist_video_writer = cv2.VideoWriter(wrist_video_fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
front_video_writer = cv2.VideoWriter(front_video_fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
shoulderview_left_video_writer = cv2.VideoWriter(shoulderview_left_video_fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
shoulderview_right_video_writer = cv2.VideoWriter(shoulderview_right_video_fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))

for i in range(len(wrist_video_list)):
    wrist_video_writer.write(wrist_video_list[i])
    front_video_writer.write(front_video_list[i])
    shoulderview_left_video_writer.write(shoulderview_left_video_list[i])
    shoulderview_right_video_writer.write(shoulderview_right_video_list[i])

front_video_writer.release()
wrist_video_writer.release()
shoulderview_left_video_writer.release()
shoulderview_right_video_writer.release()