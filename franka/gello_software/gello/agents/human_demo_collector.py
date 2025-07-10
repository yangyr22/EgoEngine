import os
import cv2
import json
import pyrealsense2 as rs
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fm_vis.hand_detector.hand_reconstruction import OnlineHandCapture

_MANO_JOINT_CONNECT = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]
def plot_hand(ax, joints_3d):
    """ Plot 3D hand skeleton using given joint coordinates. """
    ax.clear()
    ax.set_xlim([-0.4, 0.3])
    ax.set_ylim([-0.4, 0.3])
    ax.set_zlim([0.3, 1.0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Hand Skeleton")
    if joints_3d is not None:
        joints_3d = np.array(joints_3d)  # Shape: (21, 3)
        xs, ys, zs = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
        zs = -zs+1.3
        ax.scatter(xs, ys, zs, c='r', marker='o', s=0.1)
        # Draw connection lines
        for connection in _MANO_JOINT_CONNECT:
            p1, p2 = connection
            ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], 'b', linewidth=0.1)
    plt.draw()
    plt.pause(0.001)  # Small pause to refresh the plot

def main(output_path):
    # Create output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)
    # Initialize RealSense device
    pipeline = rs.pipeline()
    config = rs.config()
    hand_capture = OnlineHandCapture(hand_type='right', visualize=False)
    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    # Start camera stream
    pipeline.start(config)
    print("RealSense camera started. Press 'q' to quit, 's' to start recording, 'e' to save video & JSON, 'f' to discard.")
    # Recording state variables
    recording = False
    failed = False
    idx = 172
    max_demos = 200
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frame_size = (640, 360)

    # Initialize Matplotlib 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Store JSON data
    json_data = {"pred_3d_joints": [], "tvec": [], "rvec": []}
    try:
        while True:
            if idx >= max_demos:
                print("All 200 demos have been recorded. Press 'q' to quit.")
            # Get frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            output_frame = hand_capture.hand_reconstruction_frame(color_image, warning=False)
            # Handle recording process
            if recording and idx < max_demos:
                if output_frame is None:
                    print("Failed, please record again.")
                    failed = True
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    partial_file = os.path.join(output_path, f"{idx}.mp4")
                    if os.path.isfile(partial_file):
                        os.remove(partial_file)
                    json_data = {"pred_3d_joints": [], "tvec": [], "rvec": []}  # Reset JSON data
                else:
                    print(output_frame["tvec"])
                    writer.write(color_image)
                    # Store data for JSON
                    json_data["pred_3d_joints"].append(output_frame["pred_3d_joints"].tolist())
                    json_data["tvec"].append(output_frame["tvec"].tolist())
                    json_data["rvec"].append(output_frame["rvec"].tolist())
            # 3D hand plotting
            if output_frame and "pred_3d_joints" in output_frame:
                joints_3d = output_frame["pred_3d_joints"] + output_frame["tvec"]
                plot_hand(ax, joints_3d)
            # Display camera feed
            cv2.imshow("RealSense Feed", color_image)
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not recording and idx < max_demos:
                recording = True
                failed = False
                print(f"Start recording demo {idx}")
                video_path = os.path.join(output_path, f"{idx}.mp4")
                writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
                json_data = {"pred_3d_joints": [], "tvec": [], "rvec": []}  # Reset JSON data
            elif key == ord('e') and recording:
                recording = False
                if writer:
                    writer.release()
                    writer = None
                if not failed:
                    # Save JSON file
                    json_path = os.path.join(output_path, f"{idx}.json")
                    with open(json_path, "w") as json_file:
                        json.dump(json_data, json_file, indent=4)
                    print(f"Saved video at {os.path.join(output_path, f'{idx}.mp4')}")
                    print(f"Saved JSON at {json_path}")
                    idx += 1  # Move to next recording
            elif key == ord('f') and recording:
                recording = False
                print(f"Recording for demo {idx} discarded, re-recording required.")
                if writer:
                    writer.release()
                    writer = None
                # Remove video file if exists
                video_path = os.path.join(output_path, f"{idx}.mp4")
                if os.path.isfile(video_path):
                    os.remove(video_path)
                # Remove JSON file if exists
                json_path = os.path.join(output_path, f"{idx}.json")
                if os.path.isfile(json_path):
                    os.remove(json_path)
                json_data = {"pred_3d_joints": [], "tvec": [], "rvec": []}  # Reset JSON data
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        plt.close()

# Run the script
if __name__ == "__main__":
    output_dir = "/mnt/data2/dexmimic/datasets/human_raw_video/flip"
    main(output_dir)