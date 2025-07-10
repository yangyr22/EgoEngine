from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tyro

from gello.zmq_core.camera_node import ZMQClientCamera


@dataclass
class Args:
    ports: Tuple[int, ...] = (5002, 5003, 5004, 5005)
    hostname: str = "127.0.0.1"
    # hostname: str = "128.32.175.167"


def main(args):
    cameras = []
    import cv2

    images_display_names = []
    for port in args.ports:
        cameras.append(ZMQClientCamera(port=port, host=args.hostname))
        images_display_names.append(f"image_{port}")
        cv2.namedWindow(images_display_names[-1], cv2.WINDOW_NORMAL)

    i =0
    while True:
        
        for display_name, camera in zip(images_display_names, cameras):
            image, depth = camera.read()
            stacked_depth = np.dstack([depth, depth, depth]).astype(np.uint8)
            image_depth = cv2.hconcat([image[:, :, ::-1], stacked_depth])
            cv2.imshow("img",image)
            key = cv2.waitKey(1) & 0xFF  # Mask to extract last 8 bits for ASCII
            if key == ord('s'):  # If 's' key is pressed, save the frame
                cv2.imwrite(f'mnt/data2/akutumbaka3/saved_frame_{i}.jpg', image)
                print(f"Frame {i} saved successfully!")
                i += 1  # Increment counter
            elif key == 27:  # If 'Esc' key is pressed, exit the loop
                break


if __name__ == "__main__":
    main(tyro.cli(Args))
