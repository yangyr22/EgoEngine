import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from dex_retargeting.constants import RobotName

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


def OfflineDataset(json_path=""):
    # read json file
    with open(json_path, "r") as file:
        data = json.load(file)
    
    tvec_list = [np.array(item["tvec"]) for item in data]
    pred_3d_joints_list = [np.array(item["pred_3d_joints"]) for item in data]
    rvec_list = [np.array(item["rvec"]) for item in data]

    tvec_array = np.stack(tvec_list, axis=0)  # (T, 3)
    pred_3d_joints_array = np.stack(pred_3d_joints_list, axis=0)  # (T, 21, 3)
    rvec_array = np.stack(rvec_list, axis=0)  # (T, 3)
    
    # return a dict
    return {
        "tvec": tvec_array[:,:,0], # (T, 3)
        "pred_3d_joints": pred_3d_joints_array, # (T, 21, 3)
        "rvec": rvec_array[:,:,0] # (T, 3)
    }

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
    positions = data["tvec"]  # (T, 3)


    # Extract 3D position of each frame: assume hand_pose[t, 0, :3] is (x, y, z)
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


    hand_joints = data["pred_3d_joints"]  # (T, 21, 3)
    transition = data["tvec"]  # (T, 3)
    all_coords = hand_joints + transition[:, None, :]  # (T, 21, 3)

    T = all_coords.shape[0]
  
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

def test():
    # example usage
    sample = customeddataset("/coc/flash7/yliu3735/datasets/test/1.json")
    # print(sample["tvec"].shape, sample["pred_3d_joints"].shape, sample["rvec"].shape)
    # exit()
    visualize_3d_traj_video(sample, output_path="traj.mp4", fps=30)
    visualize_mano_3d_traj_video(sample, output_path="mano.mp4", fps=30)


if __name__ == "__main__":
    test()