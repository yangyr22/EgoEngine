# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]
# Modified by Yuzhe Qin to use the sequential information inside the dataset

"""Customed dataset."""

from pathlib import Path
import torch
import numpy as np
from mano_layer import MANOLayer
import json
import yaml
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


'''
Calculation:
vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]  # transformation
vertex = np.ascontiguousarray(vertex)
joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
joint = np.ascontiguousarray(joint)
'''

_SUBJECTS = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
]

_MANO_JOINTS = [
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]

_MANO_JOINT_CONNECT = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]

_SERIALS = [
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4

def kalman_filter_1d(measurements, Q, R):
    """
    Apply a simplest Kalman filter (first-order constant model) on a 1D measurement sequence.
    measurements: shape (T,) 1D array
    Q: process noise variance
    R: measurement noise variance
    Returns a smoothed array of the same length.
    """
    T = len(measurements)
    x = np.zeros(T, dtype=np.float32)
    P = np.zeros(T, dtype=np.float32)
    
    x[0] = measurements[0]  
    P[0] = 1.0
    
    for t in range(1, T):
        x_pred = x[t - 1]
        P_pred = P[t - 1] + Q

        K_t = P_pred / (P_pred + R)
        x[t] = x_pred + K_t * (measurements[t] - x_pred)
        P[t] = (1.0 - K_t) * P_pred
    
    return x


def kalman_filter_nd(array_in, Q=1e-5, R=1e-3):
    T, D = array_in.shape
    array_out = np.zeros_like(array_in)
    
    for d in range(D):
        measurements_1d = array_in[:, d]
        smoothed_1d = kalman_filter_1d(measurements_1d, Q, R)
        array_out[:, d] = smoothed_1d
    return array_out

def _kalman_filter_smoothing(array_in, Q=1e-5, R=1e-3):
    """
    Main function: apply a Kalman filter on (T, 1, 51) hand data.
    The final output shape remains (T, 1, 51).
    """
    T = array_in.shape[0]
    D = np.prod(array_in.shape[1:])
    array_flat = array_in.reshape(T, D)
    
    array_smoothed_flat = kalman_filter_nd(array_flat, Q, R)
    
    return array_smoothed_flat.reshape(array_in.shape)

class CustomVideoDataset(Dataset):
    def __init__(self, data_dir, hand_type="right"):
        """
        Args:
            data_dir (str or Path): Path to the directory containing video subdirectories.
            hand_type (str): The hand key to look for in the JSON files.
        """
        self.data_dir = Path(data_dir)
        self.hand_type = hand_type
        # Each subdirectory is assumed to be one video.
        self.video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # Set window size
        self.window_size = 1

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        print("processing: ", video_dir)
        json_files = sorted(video_dir.glob("*.json"))
        
        hand_pose_list = []
        hand_shape_list = []
        hand_joints_list = []
        
        tot = 0
        for json_file in json_files:
            tot += 1
            with open(json_file, "r") as f:
                data = json.load(f)

            # Obtain hand data (assuming "right_hand" or "left_hand" in the example)
            hand_data = data[self.hand_type + "_hand"]
            
            hand_pose = np.array(hand_data["pred_hand_global_pose"])
            hand_shape = np.array(hand_data["pred_hand_betas"])

            hand_joints = np.array(hand_data["pred_joints_smpl"])
            # If the gesture or shape is 1D (51,) / (10,), expand to (1, 51) / (1, 10)
            if hand_pose.ndim == 1:
                hand_pose = hand_pose[None, :]
            if hand_shape.ndim == 1:
                hand_shape = hand_shape[None, :]

            hand_pose_list.append(hand_pose)
            hand_shape_list.append(hand_shape)
            hand_joints_list.append(hand_joints)

        # hand_pose_stack: shape (T, 1, 51) or similar
        hand_pose_stack = np.stack(hand_pose_list, axis=0)
        hand_pose_stack_reordered = np.concatenate(
            (hand_pose_stack[..., 3:], hand_pose_stack[..., :3]), axis=-1)
        hand_joints_stack = np.stack(hand_joints_list, axis=0)
        hand_joints_stack = hand_joints_stack - hand_joints_stack[:,0:1,:]
        
        # Only take the hand_shape from the first frame (assuming the hand shape remains the same for the entire video)
        hand_shape_single = hand_shape_list[0][0, :]
        
        # Apply a sliding average pooling to hand_pose_stack
        hand_pose_stack = self._sliding_average_no_padding(hand_pose_stack_reordered, self.window_size)

        return {
            "hand_pose": hand_pose_stack,      # still T frames of output
            "hand_shape": hand_shape_single,    # shape (10,)
            "hand_joints": hand_joints_stack    # (T,21,3)
        }

    def _sliding_average_no_padding(self, array_in, window_size):
        """
        Perform a sliding average on the input array_in, without padding but maintaining length.
        For example, when window_size=5, for the i-th frame, take frames [i-2 : i+2] for averaging.
        Any part exceeding the boundary is directly truncated.
        """
        T = array_in.shape[0]
        half_window = window_size // 2
        out_list = []

        for i in range(T):
            # Calculate the left and right boundaries of the window
            left = max(0, i - half_window)
            right = min(T, i + half_window + 1)
            # Take the average of this window over axis=0 (the time dimension)
            local_avg = array_in[left:right].mean(axis=0, keepdims=True)
            out_list.append(local_avg)
        
        # Concatenate all frames back to the original length after sliding average
        return np.concatenate(out_list, axis=0)


class DexYCBOnlyHandVideoDataset:
    def __init__(self, data_dir, hand_type="right"):
        self._data_dir = Path(data_dir)
        self._calib_dir = self._data_dir / "calibration"
        
        # Camera and MANO parameters
        self._intrinsics, self._extrinsics = self._load_camera_parameters()
        self._mano_side = hand_type
        self._mano_parameters = self._load_mano()
        
        # Capture data
        self._subject_dirs = [sub for sub in self._data_dir.iterdir() if sub.stem in _SUBJECTS]
        self._capture_meta = {}
        self._capture_pose = {}
        self._captures = []
        
        for subject_dir in self._subject_dirs:
            for capture_dir in subject_dir.iterdir():
                meta_file = capture_dir / "meta.yml"
                with meta_file.open(mode="r") as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)

                if hand_type not in meta["mano_sides"]:
                    continue

                pose = np.load((capture_dir / "pose.npz").resolve().__str__())
                self._capture_meta[capture_dir.stem] = meta
                self._capture_pose[capture_dir.stem] = pose
                self._captures.append(capture_dir.stem)

    def __len__(self):
        return len(self._captures)

    def __getitem__(self, item):
        if item >= self.__len__():
            raise ValueError(f"Index {item} out of range")

        capture_name = self._captures[item]
        meta = self._capture_meta[capture_name]
        pose = self._capture_pose[capture_name]
        hand_pose = pose["pose_m"]

        # Load extrinsic and MANO parameters
        extrinsic_name = meta["extrinsics"]
        extrinsic_mat = np.array(self._extrinsics[extrinsic_name]["extrinsics"]["apriltag"]).reshape([3, 4])
        extrinsic_mat = np.concatenate([extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0)
        mano_name = meta["mano_calib"][0]
        mano_parameters = self._mano_parameters[mano_name]

        ycb_data = dict(
            hand_pose=hand_pose,
            extrinsics=extrinsic_mat,
            hand_shape=mano_parameters,
            capture_name=capture_name,
        )
        return ycb_data

    def _load_camera_parameters(self):
        extrinsics = {}
        intrinsics = {}
        for cali_dir in self._calib_dir.iterdir():
            if not cali_dir.stem.startswith("extrinsics"):
                continue
            extrinsic_file = cali_dir / "extrinsics.yml"
            name = cali_dir.stem[len("extrinsics_") :]
            with extrinsic_file.open(mode="r") as f:
                extrinsic = yaml.load(f, Loader=yaml.FullLoader)
            extrinsics[name] = extrinsic
        return intrinsics, extrinsics

    def _load_mano(self):
        mano_parameters = {}
        for cali_dir in self._calib_dir.iterdir():
            if not cali_dir.stem.startswith("mano"):
                continue
            mano_file = cali_dir / "mano.yml"
            with mano_file.open(mode="r") as f:
                shape_parameters = yaml.load(f, Loader=yaml.FullLoader)
            mano_name = "_".join(cali_dir.stem.split("_")[1:])
            mano_parameters[mano_name] = np.array(shape_parameters["betas"])
        return mano_parameters


def main(dexycb_dir: str):
    from collections import Counter

    dataset = DexYCBVideoDataset(dexycb_dir)
    print(len(dataset))

    ycb_names = []
    for i, data in enumerate(dataset):
        ycb_ids = data["ycb_ids"][0]
        ycb_names.append(YCB_CLASSES[ycb_ids])

    counter = Counter(ycb_names)
    print(counter)

    sample = dataset[0]
    print(sample.keys())

def test(dexycb_dir: str):
    from collections import Counter

    customed_dir = "/srv/rail-lab/flash5/yzheng494/0206_output/json"
    dataset = CustomVideoDataset(customed_dir)
    dataset1 = DexYCBOnlyHandVideoDataset(dexycb_dir)

    sample = dataset[1]
    # visualize_3d_traj_video(sample, output_path="trajectory_0.mp4", fps=30)
    # sample1 = dataset1[1]
    # print(sample["hand_shape"])
    # print(sample1["hand_shape"])


    # visualize_3d_traj_video(sample1, output_path="trajectory_1.mp4", fps=)
    # visualize_mano_3d_traj_video(sample1, output_path="pose_1.mp4", fps=30)
    visualize_mano_3d_traj_video(sample, output_path="pose_0.mp4", fps=30)


    # dataset.visualize_3d_traj(0, "test.png")
    # sample1 = dataset1[2]
    # print(sample["hand_shape"].shape)
    # print(sample1["hand_shape"].shape)


def visualize_3d_traj(data, output_path=""):
    """
    Visualize the 3D hand position trajectory of the video at index idx.
    The 0th frame is silver, and subsequent frames gradually shift toward red, each frame shown as a small sphere.
    
    Args:
        idx (int): Video index
        output_path (str): If specified, save the visualization to this path; otherwise call plt.show()
    """

    hand_pose = data["hand_pose"]  # (T, 1, 51)
    
    # Extract the 3D position of each frame: assume hand_pose[t, 0, :3] is (x, y, z)
    positions = hand_pose[:, 0, :3]  # (T, 3)
    T = positions.shape[0]

    # Define a color gradient from silver to red
    silver_rgb = np.array([192/255.0, 192/255.0, 192/255.0])  # silver
    red_rgb = np.array([1.0, 0.0, 0.0])                       # red
    # Generate a color list from silver to red for T frames
    colors = [
        silver_rgb + (red_rgb - silver_rgb) * (i / (T - 1 if T > 1 else 1))
        for i in range(T)
    ]
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw T points in sequence
    for i, (x, y, z) in enumerate(positions):
        ax.scatter(x, y, z, color=colors[i], s=5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Optionally set axis range or viewpoint
    # ax.set_xlim(...)
    # ax.set_ylim(...)
    # ax.set_zlim(...)
    # ax.view_init(elev=..., azim=...)

    plt.title(f"3D Trajectory for video")

    if output_path:
        plt.savefig(output_path)
        print(f"3D Trajectory save to: {output_path}")
    else:
        plt.show()

def visualize_3d_traj_video(data, output_path="", fps=5):
    """
    Create an animation of the 3D hand position trajectory for the video at index idx in time sequence.
    - The initial frame is silver, subsequent frames gradually shift toward red;
    - At the t-th frame, show all points from frame 0 to t.
    
    Args:
        idx (int): Video index
        output_path (str): If specified, save to this path (mp4/gif, etc.). Otherwise call plt.show() to play the animation.
        fps (int): Video frame rate (only effective when saving to video).
    """
    hand_pose = data["hand_pose"]  # (T, 1, 51)

    # Extract 3D position of each frame: assume hand_pose[t, 0, :3] is (x, y, z)
    positions = hand_pose[:, 0, 48:]  # (T, 3)
    T = positions.shape[0]

    # Color from silver to red
    silver_rgb = np.array([192/255.0, 192/255.0, 192/255.0])  # silver
    red_rgb = np.array([1.0, 0.0, 0.0])                       # red
    color_list = [
        silver_rgb + (red_rgb - silver_rgb) * (i / (T - 1 if T > 1 else 1))
        for i in range(T)
    ]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # -- To keep the coordinate axis fixed during the entire animation, set the range based on all data first --
    x_all, y_all, z_all = positions[:, 0], positions[:, 1], positions[:, 2]
    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    z_min, z_max = np.min(z_all), np.max(z_all)

    # Compute spans
    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_max - z_min

    # Expansion ratio (1.2x): in other words, 10% more on each side of the original span
    expand_ratio = 1.2

    # Expand each dimension
    x_mid = 0.5 * (x_min + x_max)
    x_half_new = 0.5 * span_x * expand_ratio
    x_min_new = x_mid - x_half_new
    x_max_new = x_mid + x_half_new

    y_mid = 0.5 * (y_min + y_max)
    y_half_new = 0.5 * span_y * expand_ratio
    y_min_new = y_mid - y_half_new
    y_max_new = y_mid + y_half_new

    z_mid = 0.5 * (z_min + z_max)
    z_half_new = 0.5 * span_z * expand_ratio
    z_min_new = z_mid - z_half_new
    z_max_new = z_mid + z_half_new

    # Set coordinate ranges
    ax.set_xlim(x_min_new, x_max_new)
    ax.set_ylim(y_min_new, y_max_new)
    ax.set_zlim(z_min_new, z_max_new)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"3D Trajectory for video")

    # -- Here we use a scatter and update how many points are visible each frame --
    scatter = ax.scatter([], [], [], s=5)  # initialize empty scatter

    # Convert the color list to a numpy array
    color_array = np.array(color_list)  # shape (T, 3)

    def init():
        """
        Animation initialization function, called once at the start.
        """
        scatter._offsets3d = ([], [], [])
        scatter.set_color([])
        return (scatter,)

    def update(frame):
        """
        Update each frame:
        - Let the scatter include all points from [0..frame]
        """
        xs = x_all[:frame + 1]
        ys = y_all[:frame + 1]
        zs = z_all[:frame + 1]

        cs = color_array[:frame + 1]

        scatter._offsets3d = (xs, ys, zs)
        scatter.set_color(cs)

        return (scatter,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(T),
        init_func=init,
        blit=False,
        interval=200
    )

    if output_path:
        # If output path is specified, save the animation as video
        # Requires ffmpeg or ImageMagick, etc. installed
        # e.g. output_path="traj.mp4" or "traj.gif"
        ani.save(output_path, fps=fps, writer='ffmpeg')  # or writer='imagemagick'
        print(f"Animation saved to: {output_path}")
    else:
        plt.show()

def visualize_2d_traj_video(data, output_path="", fps=5):
    """
    Create an animation of the 2D hand position trajectory for the video at index idx (ignoring the Z axis).
    - The initial frame is silver, subsequent frames gradually shift toward red;
    - At the t-th frame, show all points from frame 0 to t.
    
    Args:
        idx (int): Video index
        output_path (str): If specified, save to this path (mp4/gif, etc.). Otherwise use plt.show() to play the animation.
        fps (int): Video frame rate (only effective when saving to video).
    """
    hand_pose = data["hand_pose"]  # (T, 1, 51)

    # hand_pose[t, 0, :3] = (x, y, z). Here we only take x, y for 2D
    positions = hand_pose[:, 0, 48:51]
    T = positions.shape[0]

    # Color from silver to red
    silver_rgb = np.array([192/255.0, 192/255.0, 192/255.0])  # silver
    red_rgb = np.array([1.0, 0.0, 0.0])                       # red
    color_list = [
        silver_rgb + (red_rgb - silver_rgb) * (i / (T - 1 if T > 1 else 1))
        for i in range(T)
    ]
    color_array = np.array(color_list)  # shape (T, 3)

    # Extract all x, y
    x_all, y_all = positions[:, 0], positions[:, 1]

    # Compute coordinate axis range (expanded by 1.2x to remain fixed during animation)
    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)

    span_x = x_max - x_min
    span_y = y_max - y_min
    expand_ratio = 1.2

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    x_half_new = 0.5 * span_x * expand_ratio
    y_half_new = 0.5 * span_y * expand_ratio

    x_min_new = x_mid - x_half_new
    x_max_new = x_mid + x_half_new
    y_min_new = y_mid - y_half_new
    y_max_new = y_mid + y_half_new

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_min_new, x_max_new)
    ax.set_ylim(y_min_new, y_max_new)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title(f"2D Trajectory for video")

    scatter = ax.scatter([], [], s=5)

    def init():
        # Use an empty array of shape (0, 2) to avoid indexing issues
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_color([])
        return (scatter,)

    def update(frame):
        """
        Update each frame:
        - Let the scatter include points [0..frame]
        """
        xs = x_all[:frame + 1]
        ys = y_all[:frame + 1]

        cs = color_array[:frame + 1]

        coords = np.column_stack((xs, ys))
        scatter.set_offsets(coords)
        scatter.set_color(cs)
        return (scatter,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(T),
        init_func=init,
        blit=False,
        interval=200
    )

    if output_path:
        # If an output path is specified, save the animation as MP4/GIF
        # Requires ffmpeg or ImageMagick, etc.
        ani.save(output_path, fps=fps, writer='ffmpeg')
        print(f"2D Animation saved to: {output_path}")
    else:
        plt.show()


def visualize_mano_3d_traj_video(data, output_path="", fps=5):
    """
    Visualize a sequence of 51D MANO (right-hand) parameters: each frame we draw
    the 21-joint skeleton in 3D (including global translation) and compile into an animation.

    Args:
        data (dict):
            e.g. data["hand_pose"] has shape (T, 1, 51),
                 where the first 48 dims are MANO pose (rotations),
                 and the last 3 dims are global translation.
        output_path (str):
            If set, save as MP4/GIF; otherwise use plt.show() for interactive display.
        fps (int):
            Frame rate if saving to video.
        rotation (bool):
            If True, apply the first 3 dims of global rotation; if False, ignore them.
    """

    hand_pose = data["hand_pose"]  # shape (T, 1, 51)
    hand_shape = data["hand_shape"]  # shape (10,)
    mymano_layer = MANOLayer("right", hand_shape.astype(np.float32))
    T = hand_pose.shape[0]
    all_coords = []

    for t in range(T):
        pose_48 = torch.tensor(hand_pose[t, 0, :48], dtype=torch.float32).unsqueeze(0)
        trans = torch.tensor(hand_pose[t, 0, 48:], dtype=torch.float32).unsqueeze(0)
        
        _, j = mymano_layer(pose_48, trans)  # j: (1, 21, 3) - 21 joints
        j_numpy = j[0].cpu().numpy()
        all_coords.append(j_numpy)

    all_coords = np.stack(all_coords, axis=0)  # shape (T, 21, 3)

    hand_joints = data["hand_joints"]
    all_coords = hand_joints
  
    # Set up joint connections for MANO
    _MANO_JOINT_CONNECT = [
        [0, 1],  [1, 2],   [2, 3],   [3, 4],  # Thumb
        [0, 5],  [5, 6],   [6, 7],   [7, 8],  # Index
        [0, 9],  [9, 10],  [10, 11], [11, 12], # Middle
        [0, 13], [13, 14], [14, 15], [15, 16], # Ring
        [0, 17], [17, 18], [18, 19], [19, 20], # Little
    ]

    # Get axis limits
    x_all, y_all, z_all = all_coords[..., 0].ravel(), all_coords[..., 1].ravel(), all_coords[..., 2].ravel()
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    z_min, z_max = z_all.min(), z_all.max()

    expand_ratio = 1.2
    x_half, y_half, z_half = (x_max - x_min) * 0.5 * expand_ratio, (y_max - y_min) * 0.5 * expand_ratio, (z_max - z_min) * 0.5 * expand_ratio
    x_mid, y_mid, z_mid = (x_max + x_min) * 0.5, (y_max + y_min) * 0.5, (z_max + z_min) * 0.5

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_mid - x_half, x_mid + x_half)
    ax.set_ylim(y_mid - y_half, y_mid + y_half)
    ax.set_zlim(z_mid - z_half, z_mid + z_half)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"MANO Right Hand 3D Skeleton (51D)")

    # Scatter points (joints) in blue
    scat = ax.scatter([], [], [], s=15, color='blue')
    lines = [ax.plot([], [], [], c='blue', lw=2)[0] for _ in _MANO_JOINT_CONNECT]

    def init():
        scat._offsets3d = ([], [], [])
        for l in lines:
            l.set_data([], [])
            l.set_3d_properties([])
        return [scat] + lines

    def update(frame):
        coords_21 = all_coords[frame]  # shape (21,3)
        xs, ys, zs = coords_21[:, 0], coords_21[:, 1], coords_21[:, 2]
        scat._offsets3d = (xs, ys, zs)

        # Update bone connections
        for conn, line_obj in zip(_MANO_JOINT_CONNECT, lines):
            i0, i1 = conn
            line_obj.set_data(coords_21[[i0, i1], 0], coords_21[[i0, i1], 1])
            line_obj.set_3d_properties(coords_21[[i0, i1], 2])

        return [scat] + lines

    ani = animation.FuncAnimation(fig, update, frames=T, blit=False, interval=200)

    if output_path:
        ani.save(output_path, fps=fps, writer='ffmpeg')
        print(f"Animation saved to: {output_path}")
    else:
        plt.show()

    

if __name__ == "__main__":
    import tyro

    tyro.cli(test)

    # python customed_dataset.py --dexycb-dir /coc/flash7/yliu3735/datasets/DexYC