import os
import json
import numpy as np
import cv2
fx = 388.720  # Focal length in x
fy = 388.720 # Focal length in y
cx = 325.012  # Principal point x
cy = 237.630  # Principal point y

# Camera matrix
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
# Define camera parameters


dist_coeffs = np.zeros(5)  # Assuming no distortion; replace if distortion coefficients are known

# Directory containing the .json files
json_dir = "/home/yzheng494/Documents/test/data11/videos/0206_output/raw_video/video4/json_folder"
output_dir = "/home/yzheng494/Documents/test/data11/videos/0206_output/json/json4"

# Iterate through each .json file
pose_results = []
for file_name in sorted(os.listdir(json_dir)):
    if file_name.endswith(".json"):
        file_path = os.path.join(json_dir, file_name)

        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract 3D points (object coordinates) and 2D points (image plane)
        if "right_hand" in data:
            right_hand = data["right_hand"]

            if "pred_joints_original" in right_hand and "pred_joints_img" in right_hand:
                object_points = np.array(right_hand["pred_joints_original"], dtype=np.float32)  # Nx3
                image_points = np.array(right_hand["pred_joints_img"], dtype=np.float32)[:,:2] # Nx2


                # Ensure valid data
                if object_points.shape[0] >= 4 and image_points.shape[0] >= 4:
                    # SolvePnP
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        object_points, image_points, camera_matrix, dist_coeffs
                    )

                    if success:
                        # Store results for this frame
                        pose_results = ({
                            "rvec": rvec.tolist(),
                            "tvec": tvec.tolist(),
                            "pred_3d_joints": object_points.tolist(),
                            "pred_2d_img":image_points.tolist()

                        })
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, f"{file_name}")
                        with open(output_file, "w") as f:
                            json.dump(pose_results, f, indent = 4)
                    else:
                        print(f"Pose estimation failed for {file_name}")
                else:
                    print(f"Insufficient points in {file_name}")
            else:
                print(f"Missing key data in {file_name}")
        else:
            print(f"No 'right_hand' data in {file_name}")

# # Output the pose results
# for result in pose_results:
#     print(f"Frame: {result['file_name']}")
#     print(f"Rotation Vector: {result['rvec']}")
#     print(f"Translation Vector: {result['tvec']}")

