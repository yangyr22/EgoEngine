import h5py
import cv2
import nexusformat.nexus as nx
import numpy as np
import os
import argparse

def create_video_from_demo(args):

    demo_fn = args.demo_file
    task = args.task
    video_dir = args.video_dir

    os.makedirs(video_dir, exist_ok=True)

    demo_name = demo_fn.split('/')[-1].split('.')[0]

    demo_file = nx.nxload(demo_fn)
    # print(demo_file.tree)

    demo = demo_file['data/demo_0']

    wrist_image_list = []
    agentview_image_list = []

    for key in demo.keys():
        wrist_image = demo[f"{key}/obs/eye_in_hand_image"][:]
        agentview_image = demo[f"{key}/obs/agentview_image"][:]
        
        wrist_image_list.append(wrist_image)
        agentview_image_list.append(agentview_image)
        

    wrist_image_list = np.concatenate(wrist_image_list, axis=0)
    agentview_image_list = np.concatenate(agentview_image_list, axis=0)

    wrist_video_fn = os.path.join(video_dir, f"{task}_{demo_name}_wrist.mp4")
    agentview_video_fn = os.path.join(video_dir, f"{task}_{demo_name}_agentview.mp4")

    fps = 20

    wrist_width = wrist_image_list.shape[1]
    wrist_height = wrist_image_list.shape[2]
    wrist_video_writer = cv2.VideoWriter(wrist_video_fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, (wrist_height, wrist_width))

    agentview_width = agentview_image_list.shape[1]
    agentview_height = agentview_image_list.shape[2]
    agentview_video_writer = cv2.VideoWriter(agentview_video_fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, (agentview_height, agentview_width))

    for i in range(wrist_image_list.shape[0]):
        wrist_video_writer.write(cv2.cvtColor(wrist_image_list[i], cv2.COLOR_RGB2BGR))
        agentview_video_writer.write(cv2.cvtColor(agentview_image_list[i], cv2.COLOR_RGB2BGR))

    wrist_video_writer.release()
    agentview_video_writer.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", type = str, required=True, help="path to demo file to create videos for")
    parser.add_argument("--video_dir", default= "/home/mbronars/bc_data/rmv2/demo_videos",  type=str, help="where to save the videos")
    parser.add_argument("--task", default="lift", type=str, help="what task the demo is of")

    args = parser.parse_args()

    create_video_from_demo(args)
