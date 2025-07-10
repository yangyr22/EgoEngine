# Camera driver for ZED camera. Based off of https://github.com/droid-dataset/droid/blob/ca54513dc51d3305aa00f72bb1533aa7a7fba59f/droid/camera_utils/camera_readers/zed_camera.py


from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import COLOR_BGRA2BGR, COLOR_BGRA2RGB
from copy import deepcopy
import time
import pyzed.sl as sl


from gello.cameras.camera import CameraDriver

# config = {'zed_resolution': sl.RESOLUTION.VGA, 'fps': 30, 'resize': True, 'resize_resolution': (128, 128)}
class ZedCamera(CameraDriver):

    def __repr__(self) -> str:
        return f"ZedCamera(name= {self.cam_name}, serial_number={self.serial_number})"

    def __init__(self, cam_name, config):
        """

        Args:
            cam_name: what to call this camera view
            config: dictionary containing camera config
        """

        # Setting some class vars

        # TODO: do we need a cam name?
        self.cam_name = cam_name
        self.serial_number = config['sn']
        cameras = sl.Camera.get_device_list()

        camera_properties = None
        for cam in cameras:
            if cam.serial_number == self.serial_number:
                camera_properties = cam
        if camera_properties is None:
            raise Exception(f"Could not find Zed camera with serial number: {self.serial_number}")

        self.serial_number = str(camera_properties.serial_number)
        self.resize = config.get('resize', False)
        if self.resize:
            self.resize_resolution = config.get('resize_resolution', (128, 128))

        # self.resolution = config.get('zed_resolution', sl.RESOLUTION.VGA)
        self.resolution = config.get('zed_resolution', sl.RESOLUTION.HD720)

        # Start the camera

        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(self.serial_number))
        init_params.camera_resolution = self.resolution
        init_params.camera_fps = config.get('fps', 30)
        init_params.enable_right_side_measure = True

        self._camera = sl.Camera()

        ret = self._camera.open(init_params)
        if ret != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : " + repr(ret) + ". Exit program.")
            exit()
        # Open Camera #
        print("Opening Zed: ", self.serial_number)

        # Set some vars to get the camera frames

        self._runtime_params = sl.RuntimeParameters()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._left_depth = sl.Mat()
        self._right_depth = sl.Mat()

        self.camera_information = self._camera.get_camera_information()
        self.all_intrinsics = self.camera_information.camera_configuration.calibration_parameters

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> dict:

        if img_size is not None:
            self.resize_resolution = img_size

        # First, grab a frame
        ret = self._camera.grab(self._runtime_params)
        if ret != sl.ERROR_CODE.SUCCESS:
            print("Failed to grab frame from camera")
            return dict()

        # Next get the individual images
        data = {}
        self._camera.retrieve_image(self._left_img, sl.VIEW.LEFT, resolution=self.resolution)
        self._camera.retrieve_image(self._right_img, sl.VIEW.RIGHT, resolution=self.resolution)
        self._camera.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH, resolution=self.resolution)
        self._camera.retrieve_measure(self._right_depth, sl.MEASURE.DEPTH_RIGHT, resolution=self.resolution)

        # TODO: Zed returns images in BGRA, we convert them to RGB for Robomimic training, should this be a parameter?
        data['left_image'] = cv2.cvtColor(deepcopy(self._left_img.get_data()), COLOR_BGRA2RGB)
        data['right_image'] = cv2.cvtColor(deepcopy(self._right_img.get_data()), COLOR_BGRA2RGB)
        data['left_depth'] = deepcopy(self._left_depth.get_data())[:, :, None]
        data['right_depth'] = deepcopy(self._right_depth.get_data())[:, :, None]

        if self.resize:
            data['left_image'] = cv2.resize(data['left_image'], self.resize_resolution)
            data['right_image'] = cv2.resize(data['right_image'], self.resize_resolution)
            data['left_depth'] = cv2.resize(data['left_depth'], self.resize_resolution)
            data['right_depth'] = cv2.resize(data['right_depth'], self.resize_resolution)

        return data

    @property
    def type(self):
        return "Zed"

    def close(self):
        self._camera.close()

def debug_read(camera):
    cv2.namedWindow("ZED Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ZED Depth", cv2.WINDOW_NORMAL)

    while True:
        time.sleep(0.05)

        data = camera.read()
        if not data:
            print("Failed to read data from the camera.")
            continue  # Skip this frame and retry

        img_left = data['left_image']
        img_right = data['right_image']
        depth_left = data['left_depth']
        depth_right = data['right_depth']

        # Ensure depth images are in 3 channels (OpenCV expects 3 channels for display)
        depth_left_3ch = np.repeat(depth_left, 3, axis=-1)
        depth_right_3ch = np.repeat(depth_right, 3, axis=-1)

        # Concatenate for side-by-side display
        img_concat = np.concatenate((img_left, img_right), axis=1)
        depth_concat = np.concatenate((depth_left_3ch, depth_right_3ch), axis=1)

        # Display images
        cv2.imshow("ZED Image", img_concat)
        cv2.imshow("ZED Depth", depth_concat)

        # Exit on key press (e.g., 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.close()
    cv2.destroyAllWindows()


def debug_read_all(cam_list):
    cv2.namedWindow("zed_image")
    cv2.namedWindow("zed_depth")

    while True:
        time.sleep(0.05)
        img_list = []
        depth_list = []
        for cam in cam_list:
            data = cam.read()
            img_list.append(data['left_image'])
            depth_list.append(data['left_depth'])

        img_concat = np.concatenate(img_list, axis=1)
        depth_concat = np.concatenate(depth_list, axis=1)
        depth_concat = np.concatenate((depth_concat, depth_concat, depth_concat), axis=-1)
        cv2.waitKey(1)
        cv2.imshow("zed_image", img_concat)
        cv2.imshow("zed_depth", depth_concat)

if __name__ == "__main__":
    cameras = sl.Camera.get_device_list()
    if not cameras:
        print("No ZED cameras found.")
        exit(1)

    cam_list = []
    for cam in cameras:
        cam_list.append(ZedCamera(cam.serial_number, {'sn': cam.serial_number}))

    debug_read(cam_list[0])  # Display the first ZED camera feed

