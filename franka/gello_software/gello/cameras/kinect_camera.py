"""
Camera driver for an Azure Kinect camera built on pyk4a.
"""
from typing import List, Optional, Tuple
import numpy as np
import cv2
import time

from gello.cameras.camera import CameraDriver
import pyk4a
from pyk4a import Config, connected_device_count
from pyk4a.config import FPS,ImageFormat, DepthMode, ColorResolution


# example config = {'color_resolution': pyk4a.ColorResolution.RES_720P,
#                       'fps': 30, 'resize': True, 'resize_resolution': (128, 128)}
class KinectCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"KinectCamera(name= {self.cam_name}, serial_number={self.serial_number})"


    def __init__(self, cam_name, config):

        # Setting some class vars
        # TODO: do we need a cam name?
        self.cam_name = cam_name
        self.serial_number = config['sn']

        # Setting the config for the camera
        pyk4a_config = Config()
        pyk4a_config.camera_fps = config.get('fps', FPS.FPS_30)
        pyk4a_config.color_resolution = config.get('color_resolution', ColorResolution.RES_720P)
        pyk4a_config.color_format = config.get('color_format', ImageFormat.COLOR_BGRA32)
        pyk4a_config.depth_mode = config.get('depth_mode', DepthMode.NFOV_UNBINNED)
        pyk4a_config.synchronized_images_only = config.get("synchronize_depth", True)

        # Start the camera
        cnt = connected_device_count()
        if not cnt:
            print(f"No devices available. Cannot start Kinect with serial number: {self.serial_number}")
            exit()

        camera_id = None
        for device_id in range(cnt):
            device = pyk4a.PyK4A(device_id=device_id)
            if device.opened:
                continue
            else:
                device.open()
                if device.serial == self.serial_number: # we found our camera_id
                    camera_id = device_id
                device.close()
        if camera_id is None:
            raise  Exception(f"Could not find Kinect with serial number: {self.serial_number}")


        self._camera = pyk4a.PyK4A(device_id=camera_id, config=pyk4a_config)
        print(f"Starting Kinect camera with sn: {self.serial_number}")
        self._camera.start()

        # Do we resize the images from the camera?
        self.resize = config.get('resize', False)
        if self.resize:
            self.resize_resolution = config.get('resize_resolution', (128, 128))



    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> dict:

        if img_size is not None:
            self.resize_resolution = img_size
            self.resize = True

        # First, grab a frame
        capture = self._camera.get_capture()

        if not np.any(capture.color):
            print("Failed to grab frame from camera")
            return dict()

        # Next get the individual color/depth images
        color_image = capture.color[:, :, :3]
        # TODO: kinect returns images in BGR, we convert them to RGB for Robomimic training, should this be a parameter?
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if self.resize:
            color_image = cv2.resize(color_image, self.resize_resolution, interpolation=cv2.INTER_AREA)

        # Process depth
        if np.any(capture.depth):
            depth = capture.depth
            if self.resize:
                depth = cv2.resize(depth, self.resize_resolution, cv2.INTER_NEAREST)
        else:
            depth = None

        data = {}
        data['rgb'] = color_image
        data['depth'] = depth
        return data

    @property
    def type(self):
        return "Kinect"

def debug_read(camera):
    cv2.namedWindow("kinect_rgb")
    cv2.namedWindow("kinect_depth")

    while True:
        time.sleep(0.05)

        data = camera.read()

        rgb = data['rgb']
        depth = data['depth']
        depth = np.concatenate([depth, depth, depth], axis=-1)
        cv2.waitKey(1)
        cv2.imshow("kinect_rgb", rgb)
        cv2.imshow("kinect_depth", depth)



if __name__ == "__main__":
    from pyk4a import PyK4A, connected_device_count

    cam_name = "agentview"
    serial_number = "001039114912"
    config = {"sn": serial_number}
    camera = KinectCamera(cam_name=cam_name, config=config)
    debug_read(camera)


