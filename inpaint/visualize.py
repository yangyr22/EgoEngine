import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from tqdm import tqdm

def draw_coordinate_frame(ax, pose, length=0.1):
    origin = pose[:3, 3]
    origin = origin[[1, 0, 2]]
    x_axis = pose[:3, 1] * length + origin
    y_axis = pose[:3, 0] * length + origin
    z_axis = pose[:3, 2] * length + origin

    ax.quiver(*origin, *(x_axis - origin), color='r')
    ax.quiver(*origin, *(y_axis - origin), color='g')
    ax.quiver(*origin, *(z_axis - origin), color='b')

def draw_vector(ax, position, normal, length=0.1, color='k'):
    position = np.array(position)
    normal = np.array(normal)

    # 90-degree rotation matrix about Z-axis (counterclockwise)
    Rz = np.array([
        [0,  1, 0],
        [-1, 0, 0],
        [0,  0, 1]
    ])

    rotated_position = Rz @ position
    rotated_normal   = Rz @ normal

    end = rotated_position + rotated_normal * length
    ax.quiver(*rotated_position, *(end - rotated_position), color=color)

def render_scene(pose_list, vector_list, out_path, frame_size=(640, 640)):
    os.makedirs(out_path, exist_ok=True)
    video_path = os.path.join(out_path, "visualize.mp4")
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, frame_size)

    for i, (pose, vector) in enumerate(tqdm(zip(pose_list, vector_list), total=len(pose_list))):
        fig = plt.figure(figsize=(frame_size[0]/100, frame_size[1]/100), dpi=100)
        FigureCanvas(fig) 
        ax = fig.add_subplot(111, projection='3d')

        # Set limits
        ax.set_xlim([-1, 0])
        ax.set_ylim([-1, 0])
        ax.set_zlim([0, 1])

        draw_coordinate_frame(ax, pose)
        draw_vector(ax, vector["right_camera"][5], vector["normal_camera"]["right_wrist"], color = 'k')
        draw_vector(ax, vector["left_camera"][5], vector["normal_camera"]["left_wrist"], color = 'r')

        ax.view_init(elev=20, azim=45)
        ax.set_axis_on()         # makes sure axis is shown
        ax.grid(True)            # optional: show grid
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.tick_params(labelsize=8)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, [1, 2, 3]]
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        plt.close(fig)

    vw.release()

# === Main ===
pose_json_path = "temp_data/result_Trial/ob_in_cam.json"
vector_json_path = "temp_data/result_Trial/all_pose.json"
out_dir = "./"

with open(pose_json_path) as f:
    pose_data = json.load(f)

with open(vector_json_path) as f:
    vector_data = json.load(f)

pose_list = [np.array(p["ob_in_cam"]).reshape(4, 4) for p in pose_data]
vector_list = vector_data["frames"]

render_scene(pose_list, vector_list, out_dir)
