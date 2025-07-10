import pyrealsense2 as rs
import numpy as np
import cv2

import pyrealsense2 as rs
import numpy as np
import cv2

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
            return None, None

        color_image = np.asanyarray(color_frame.get_data())

        # Get depth frame (if enabled)
        depth_image = None
        if self.depth_enabled:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

        # Show frame if visualization is on
        if self.visualize:
            combined_image = color_image.copy()

            cv2.imshow("RGB-D Frame", combined_image)

            # If 'q' is pressed, end immediately
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.end()
                return None, None

        return color_image, depth_image

    def end(self):
        """
        Stop the pipeline and release resources.
        """
        if self.pipeline and self._is_running:
            self.pipeline.stop()
        self._is_running = False
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped and resources released.")

class VideoReader:
    def __init__(self, width=640, height=480, fps=30, visualize=False):
        """
        Initialize the Video reader. For robot manipulation.
        
        Args:
            width (int): The desired width of the color stream.
            height (int): The desired height of the color stream.
            fps (int): The desired frame rate of the color stream.
            visualize (bool): If True, display frames in a window.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.visualize = visualize
        
        self.pipeline = None
        self._is_running = False

    def start(self):
        """
        Configure and start the RealSense pipeline for color streaming only.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        self._is_running = True
        if self.visualize:
            print("[INFO] Visualization is ON. Press 'q' to exit.")

    def read(self):
        """
        Returns the latest RGB frame as a NumPy array. If visualize=True, also shows it on screen.
        
        Returns:
            np.ndarray: The RGB frame, or None if no frame was retrieved or if 'q' was pressed.
        """
        if not self._is_running or self.pipeline is None:
            return None

        # Wait for a new frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        
        # Convert frame to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Show frame if visualization is on
        if self.visualize:
            cv2.imshow("RGB Frame", color_image)
            # If 'q' is pressed, end immediately
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.end()
                return None

        return color_image

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
