#!/usr/bin/env python3
"""
AprilTag-Based Camera Pose Estimation via Unified PnP
– Reads a video, detects AprilTags, and fuses all tag corners
  into a single solvePnP call per frame.
– Known 3D tag centers in the table (world) frame are supplied.
– Draws tag axes and the camera frame in the video.
– Saves per-frame annotated images and an output video.
License: Apache 2.0
"""
import os
import sys
import argparse
import cv2
import json
import csv
import numpy as np
from dt_apriltags import Detector

def draw_pose_axes(img, K, R, t, origin, length=0.1):
    """
    Draw 3 axes at 'origin' (in pixel coords) using rotation R, translation t.
    X=red, Y=green, Z=blue.
    """
    # Project the three axes (in world units) into image
    axes3D = np.float32([
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
    ]).reshape(-1,3)
    # Convert world->cam projection:
    rvec, _ = cv2.Rodrigues(R)
    imgpts, _ = cv2.projectPoints(
        axes3D, rvec, t, K, distCoeffs=None
    )
    imgpts = imgpts.reshape(-1,2).astype(int)
    o = tuple(origin)
    cv2.line(img, o, tuple(imgpts[0]), (0,0,255), 2)
    cv2.line(img, o, tuple(imgpts[1]), (0,255,0), 2)
    cv2.line(img, o, tuple(imgpts[2]), (255,0,0), 2)

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified PnP camera pose from AprilTags")
    p.add_argument("--input_video", "-i", required=True,
                   help="Path to the input video file.")
    p.add_argument("--output_dir", "-o", required=True,
                   help="Directory where annotated frames and video will be saved.")
    p.add_argument("--fx",  type=float, default=277.0,
                   help="Camera focal length x.")
    p.add_argument("--cx",  type=float, default=319.5,
                   help="Camera principal point x.")
    p.add_argument("--tag_size", type=float, default=0.06,
                   help="AprilTag size in meters.")
    p.add_argument("--families", type=str, default="tagStandard41h12",
                   help="AprilTag families to detect.")
    return p.parse_args()

def main():
    args = parse_args()

    # === 1) Prepare output ===
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    video_out = os.path.join(args.output_dir, "annotated_video.mp4")

    # === 2) Open input video ===
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error opening video {args.input_video}", file=sys.stderr)
        sys.exit(1)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # === 3) Setup video writer ===
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (width,height), True)

    # === 4) AprilTag detector ===
    detector = Detector(
        families=args.families, nthreads=1,
        quad_decimate=1.0, quad_sigma=0.0,
        refine_edges=1, decode_sharpening=0.25, debug=0
    )

    # === 5) Camera intrinsics matrix ===
    K = np.array([[args.fx,    0, args.cx],
                  [   0,    args.fx, args.cx],
                  [   0,       0,      1 ]], dtype=float)

    # === 6) Define your known tag centers in table/world frame ===
    #    Replace these with your actual measurements (in meters).
    #    Keys are the AprilTag IDs.
    world_centers = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([0.48, 0.34, 0.0]),
        2: np.array([0.0, 0.34, 0.0]),
        3: np.array([0.48, 0.0, 0.0]),
    }

    idx = 0
    best_idx = -1
    best_cam_pose = None
    all_cam_poses = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(
            gray,
            estimate_tag_pose=True,  # we’ll do PnP ourselves
            camera_params=(args.fx, args.fx, args.cx, args.cx),
            tag_size=args.tag_size
        )
        # Accumulate 3D-2D correspondences
        obj_pts = []
        img_pts = []
        half = args.tag_size / 2.0

        for tag in tags:
            tid = tag.tag_id
            # skip IDs outside [0,3]
            if not (0 <= tid < len(world_centers)):
                continue

            wc = world_centers[tid]
            # the 4 corners of the tag in its local frame
            corner_offsets = np.array([
                [ half, -half, 0],
                [ half,  half, 0],
                [-half,  half, 0],
                [-half, -half, 0],
            ], dtype=float)

            for offset, img_pt in zip(corner_offsets, tag.corners):
                obj_pts.append(wc + offset)  # world‐frame 3D point
                img_pts.append(img_pt)       # image‐frame 2D point

        # Only solve PnP if we have at least four correspondences
        if len(obj_pts) >= 4:
            obj_pts = np.asarray(obj_pts, dtype=np.float32)
            img_pts = np.asarray(img_pts, dtype=np.float32)

            # choose solver: IPPE_SQUARE if exactly one tag (4 pts), else ITERATIVE
            
            pnp_flag = cv2.SOLVEPNP_ITERATIVE

            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, None, flags=pnp_flag
            )
            if success:
                R_w2c, _ = cv2.Rodrigues(rvec)
                t_w2c = tvec.reshape(3,1)
                R_c2w = R_w2c.T
                t_c2w = -R_w2c.T @ t_w2c.ravel()
                cam_pose = np.eye(4)
                cam_pose[:3, :3] = R_c2w
                cam_pose[:3, 3] = t_c2w
                all_cam_poses.append(cam_pose)

                score = sum(t.decision_margin for t in tags)
                if score > 10.0 and best_idx < 0:
                    best_idx = idx
                    best_cam_pose = cam_pose
                    print(f"{obj_pts} -> {img_pts}\n -> pose={best_cam_pose},")  
            else:
                all_cam_poses.append(None)  

        # # Also draw each individual tag’s axes for reference
        for tag in tags:
            # get per-tag pose for visualization
            R_tag2c = tag.pose_R
            t_tag2c = tag.pose_t.reshape(3,1)
            center_px = tuple(tag.center.astype(int))
            z_cam = R_tag2c[:, 1]        
            o_cam = t_tag2c.ravel()   
            cv2.putText(
                 frame,
                 str(tag.tag_id),
                 (center_px[0] + 5, center_px[1] - 5),      # offset a bit
                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale=0.6,
                 color=(0, 255, 255),                       # yellow
                 thickness=1,
                 lineType=cv2.LINE_AA
            )
            # if np.dot(z_cam, o_cam) <= 0:
            #     flip_X = np.array([
            #         [ 1,  0,  0],
            #         [ 0, -1,  0],
            #         [ 0,  0,  -1],
            #     ], dtype=float)
            #     R_tag2c = R_tag2c @ flip_X  
            draw_pose_axes(frame, K, R_tag2c, t_tag2c, center_px, length=args.tag_size)

        # Save frame and write video
        fname = os.path.join(frames_dir, f"{idx:06d}.png")
        cv2.imwrite(fname, frame)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"Processed {idx} frames. Output video: {video_out}")
    in_path  = os.path.join(args.output_dir, "calib.json")
    out_path = os.path.join(args.output_dir, "new_calib.json")
    data     = json.load(open(in_path, "r"))

    frames = data["frames"]
    first_frame = frames[0]["frame_idx"]
    for entry in frames:
        if entry["frame_idx"] == best_idx + first_frame:
            T_orig = np.array(entry["T_world_camera"])         # [x,y,z]
            break
    delta = best_cam_pose @ np.linalg.inv(T_orig)
        
    for entry in data["frames"]:
        T = np.array(entry["T_world_camera"])
        T_new = delta @ T
        entry["T_world_camera_rebased"] = T_new.tolist()

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
