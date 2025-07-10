## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
from dt_apriltags import Detector
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation

def draw_pose_axes(overlay, camera_params, tag_size, pose, center):

    fx, fy, cx, cy = camera_params
    
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3,:3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    opoints = np.float32([[1,0,0],
                             [0,-1,0],
                             [0,0,-1]]).reshape(-1,3) * tag_size

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
    ipoints = np.round(ipoints).astype(int)

    center = np.round(center).astype(int)
    center = tuple(center.ravel())

    cv2.line(overlay, center, tuple(ipoints[0].ravel()), (0,0,255), 2)
    cv2.line(overlay, center, tuple(ipoints[1].ravel()), (0,255,0), 2)
    cv2.line(overlay, center, tuple(ipoints[2].ravel()), (255,0,0), 2)

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    print(f"Device: {dev.get_info(rs.camera_info.name)}, Serial Number: {dev.get_info(rs.camera_info.serial_number)}")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()


serial_number = "213722070937"  # Change this to the actual serial number
config.enable_device(serial_number)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
index = 0
# Obtain camera intrinsic params
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        # print(depth_image)
        color_image = np.asanyarray(color_frame.get_data())
        raw_color_img = color_image.copy()

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        at_detector = Detector(families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        

        intr.fx = 386.964
        intr.fy = 386.964

        cam_parms = [intr.ppx, intr.ppy, intr.fx, intr.fy]
        print(cam_parms)

        tags = at_detector.detect(gray_image, 
                                  estimate_tag_pose=True, 
                                  camera_params=cam_parms, 
                                  tag_size=0.06 * 5 / 9)
        
        if len(tags) > 0:
            for i in range(len(tags)):
                tag = tags[i]
                print("rotation")
                rot = Rotation.from_matrix(tag.pose_R)
                print(rot.as_quat())
                print("translation")
                print(tag.pose_t)
                draw_pose_axes(color_image,
                            cam_parms,
                            0.05,
                            np.concatenate([tag.pose_R, tag.pose_t], axis = 1),
                            tag.center)

        raw_directory = os.path.join(os.getcwd(),'color') + '/'
        tags_directory = os.path.join(os.getcwd(),'detected_tags') + '/'
        depth_directory = os.path.join(os.getcwd(), 'depth') + '/'
        depth_color_map_directory = os.path.join(os.getcwd(), 'depth_colormap') + '/'

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image.copy(), alpha=0.5), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Save images
        save_depth = cv2.imwrite(depth_directory + str(index) + '.png', depth_image)
        # np.save(depth_directory + str(index) + '.png', depth_image)
        save_color = cv2.imwrite(tags_directory + str(index) + '.png', color_image)
        save_color = cv2.imwrite(raw_directory + str(index) + '.png', raw_color_img)
        save_depth_color_map = cv2.imwrite(depth_color_map_directory + str(index) + '.png', depth_colormap)
        f = open(os.getcwd() + '/cam_params.txt','w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        f.write('{}'.format(cam_parms))
        f.close()
        # if not save_color or not save_depth_color_map:
        #     print("failed")
        #     break

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imshow('raw depth', depth_image)
        index += 1
        print(index)
        if cv2.waitKey(1) == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()
