import time

import cv2
import numpy as np

from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector

from gello.cameras.zed_camera import ZedCamera
import pyzed.sl as sl


april_detector = AprilTagDetector()

cameras = sl.Camera.get_device_list()
cam_list = []
for cam in cameras:
    cam_list.append(ZedCamera(cam))

camera = cam_list[0]

intrinsics = camera.all_intrinsics
left_intrinsics = intrinsics.left_cam
left_intrinsics = {
    "fx":left_intrinsics.fx,
    "fy":left_intrinsics.fy,
    "cx":left_intrinsics.cx,
    "cy":left_intrinsics.cy
}

# while True:
#     capture = k4a_interface.get_last_obs()
#     if capture is not None:
#         break
#     time.sleep(0.05)
#
# image = capture["color"].astype(np.uint8)
# intrinsics = k4a_interface.get_depth_intrinsics()

# Get AprilTag Detection

while True:
    data = camera.read()

    image = data['left_image']
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    april_detector.detect(image, intrinsics=left_intrinsics, tag_size=0.04255)

    image = april_detector.vis_tag(image)

    cv2.imshow("", image)
    cv2.waitKey(1)

for detection in april_detector.results:
    print(detection)
    print("=================================")