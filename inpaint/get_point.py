import numpy as np
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sophus import SE3

# Fixed camera intrinsics
width = height = 1280
focal_length = 710
principal_point = (width//2, height//2)  # (640, 640)

# Create intrinsic matrix K
K = np.array([
    [focal_length, 0, principal_point[0]],
    [0, focal_length, principal_point[1]],
    [0, 0, 1]
])

def project_to_2d(point_3d_camera):
    """Project 3D point in camera frame to 2D pixels"""
    if point_3d_camera[2] <= 0:  # Behind camera
        return None
    x = point_3d_camera[0] / point_3d_camera[2]  # Perspective divide
    y = point_3d_camera[1] / point_3d_camera[2]
    
    u = K[0,0] * x + K[0,2]  # u = fx*x + cx
    v = K[1,1] * y + K[1,2]  # v = fy*y + cy
    
    
    # Clip to image bounds
    u = np.clip(u, 0, width-1)
    v = np.clip(v, 0, height-1)
    v = height - v - 1
    
    return np.array([u, v], dtype=int)

def transform_and_project(point_3d_device, T_device_camera):
    """Transform device→camera frame then project to 2D"""
    point_homo = np.append(point_3d_device, 1.0)
    transform_matrix = T_device_camera.to_matrix()
    transformed = point_homo @ transform_matrix
    point_3d_camera = transformed[:3] / transformed[3]
    return project_to_2d(point_3d_camera)

# Example usage:
if __name__ == "__main__":
    # Initialize data provider
    provider = data_provider.create_vrs_data_provider(
        "/coc/flash7/yliu3735/workspace/inpaint/temp_data/8e6e12f1-416a-471c-bead-14e671435f71.vrs"
    )
    
    # Get calibration
    device_calib = provider.get_device_calibration()
    rgb_calib = device_calib.get_camera_calib("camera-rgb")
    
    T_device_camera = rgb_calib.get_transform_device_camera()
    
    # Test point in device frame (X,Y,Z in meters)
    test_point_device = np.array([0.142779, -0.29469, 0.277628])  
    
    # Project to 2D
    uv = transform_and_project(test_point_device, T_device_camera)
    if uv is not None:
        print(f"Projected pixel coordinates: {uv}")
    else:
        print("Point is behind camera or invalid")