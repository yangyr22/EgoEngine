import pyrealsense2 as rs
import numpy as np
import cv2

import pyrealsense2 as rs
import numpy as np
import cv2

_MANO_JOINT_CONNECT = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]
class BGR_Reader:
    def __init__(self, width=640, height=480, fps=30, visualize=False, depth=True):
        """
        Initialize the RGB-D reader. For teleoperation.
        
        Args:
            width (int): The desired width of the color and depth streams.
            height (int): The desired height of the color and depth streams.
            fps (int): The desired frame rate of the streams.
            visualize (bool): If True, display frames in a window.
            depth (bool): If True, enable depth streaming.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.visualize = visualize
        self.depth_enabled = depth

        self.pipeline = None
        self.align = None
        self._is_running = False

        self.combined_image = None
        self.state = "i" # initialization state
        self.last_state = self.state

    def start(self):
        """
        Configure and start the RealSense pipeline for color and optional depth streaming.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Enable the color stream
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # Enable the depth stream if selected
        if self.depth_enabled:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        config.enable_device('151222077158') 
        self.pipeline.start(config)
        self._is_running = True

        if self.depth_enabled:
            # Align depth to color frame
            self.align = rs.align(rs.stream.color)

        if self.visualize:
            print("[INFO] Visualization is ON. Press 'q' to exit.")

    def read(self):
        """
        Returns the latest RGB and Depth frames as NumPy arrays. 
        
        Returns:
            tuple: (color_image, depth_image)
                   - color_image: RGB frame as a NumPy array.
                   - depth_image: Depth frame as a NumPy array (if depth mode is enabled).
        """
        if not self._is_running or self.pipeline is None:
            return None, None

        # Wait for a new frame
        frames = self.pipeline.wait_for_frames()

        if self.depth_enabled:
            # Align depth to color frame
            frames = self.align.process(frames)

        # Get color frame
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None, None, "i"

        color_image = np.asanyarray(color_frame.get_data())

        # Get depth frame (if enabled)
        depth_image = None
        if self.depth_enabled:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

        if self.visualize:
            self.combined_image = color_image.copy()
        
        current_state = self.state
        if self.state == "s" and self.last_state == "i":
            self.last_state = "s"
        elif self.state in {"e", "f"} and self.last_state == "s":
            self.state = "i"
            self.last_state = self.state

        return color_image, depth_image, current_state
    
    def visualizing(self, joints):
        """
        Visualize the MANO hand skeleton on the RGB frame.

        Args:
            joints (np.ndarray): 2D joint positions with shape (21, 2).
        """
        if not self.visualize or self.combined_image is None:
            return

        # Convert joint positions to integer pixels
        joints = np.round(joints).astype(int)

        # Draw connections
        for joint_pair in _MANO_JOINT_CONNECT:
            pt1, pt2 = joint_pair
            if pt1 < len(joints) and pt2 < len(joints):
                cv2.line(self.combined_image, tuple(joints[pt1]), tuple(joints[pt2]), (0, 255, 0), 2)

        # Draw joint points
        for i, joint in enumerate(joints):
            cv2.circle(self.combined_image, tuple(joint), 4, (0, 0, 255), -1)

        # Display frame
        cv2.imshow("RGB-D Frame with MANO Joints", self.combined_image)

        # If 'q' is pressed, exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.end()
        elif key in {ord('s'), ord('e'), ord('f')}:
            self.update_state(chr(key))
    
    def update_state(self, key):
        """
        Update the state based on user input.

        Args:
            key (str): Key input to update the state ("s", "e", or "f").
        """
        if key == "s" and self.last_state == "i":
            self.state = "s"
        elif key in {"e", "f"} and self.last_state == "s":
            self.state = key

    def end(self):
        """
        Stop the pipeline and release resources.
        """
        if self.pipeline and self._is_running:
            self.pipeline.stop()
        self._is_running = False
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped and resources released.")

# Example usage
if __name__ == "__main__":
    reader = BGR_Reader(visualize=True, depth=True)
    reader.start()

    while True:
        color_image, depth_image = reader.read()
        if color_image is None:
            break

