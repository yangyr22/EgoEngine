from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import trange
import numpy as np
import sapien
import torch
from pytransform3d import transformations as pt
from sapien import internal_renderer as R
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer


def compute_smooth_shading_normal_np(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with numpy
    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)  # (n, 3) normal without normalization to 1

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal

class SingleHandOfflineRetargetingSAPIENViewer:
    def __init__(self, headless=False, use_ray_tracing=False, visualize=True):
        # Setup
        self.visualize = visualize

        if self.visualize:
            if not use_ray_tracing:
                sapien.render.set_viewer_shader_dir("default")
                sapien.render.set_camera_shader_dir("default")
            else:
                sapien.render.set_viewer_shader_dir("rt")
                sapien.render.set_camera_shader_dir("rt")
                sapien.render.set_ray_tracing_samples_per_pixel(64)
                sapien.render.set_ray_tracing_path_depth(8)
                sapien.render.set_ray_tracing_denoiser("oidn")

            # Scene
            scene = sapien.Scene()
            scene.set_timestep(1 / 240)

            # Lighting configuration
            scene.set_environment_map(
                create_dome_envmap(sky_color=[0.5, 0.5, 0.5], ground_color=[0.5, 0.5, 0.5])
            )
            scene.add_directional_light(
                np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True
            )
            scene.add_directional_light(
                [0, 0, -1], [1.8, 1.6, 1.6], shadow=False
            )
            scene.set_ambient_light(np.array([0.3, 0.3, 0.3]))

            # Add a new ground at Z = 5
            visual_material = sapien.render.RenderMaterial()
            visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
            visual_material.set_roughness(0.7)
            visual_material.set_metallic(1)
            visual_material.set_specular(0.04)
            scene.add_ground(5, render_material=visual_material)

            # Viewer / Camera
            self.headless = headless
            if not headless:
                viewer = Viewer()
                viewer.set_scene(scene)
                viewer.set_camera_xyz(1.5, 0, 1)
                viewer.set_camera_rpy(0, -0.8, 3.14)
                viewer.control_window.toggle_origin_frame(False)
                self.viewer = viewer
            else:
                self.camera = scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)
                self.camera.set_local_pose(
                    # sapien.Pose([4, 0, 1], [0.7071, 0., 0.7071, 0.])
                    sapien.Pose([4, 0, 1], [0., 0.7071, 0., 0.7071])
                )

            self.scene = scene

            # Member for camera pose (used for retargeting)
            self.camera_pose: Optional[sapien.Pose] = None

            sapien.render.set_log_level("error")
        
        else:
            pass

class SingleHandOnlineRetargetingSAPIENViewer:
    def __init__(self, use_ray_tracing=False, visualize=True):
        """
        Online SAPIEN Viewer for real-time single-hand retargeting.
        
        Args:
            use_ray_tracing (bool): If True, enable ray tracing for high-quality rendering.
            visualize (bool): If True, show the visualization using SAPIEN Viewer.
        """
        self.visualize = visualize
        self.scene = None
        self.viewer = None

        if self.visualize:
            # Configure rendering mode
            if use_ray_tracing:
                sapien.render.set_viewer_shader_dir("rt")
                sapien.render.set_camera_shader_dir("rt")
                sapien.render.set_ray_tracing_samples_per_pixel(64)
                sapien.render.set_ray_tracing_path_depth(8)
                sapien.render.set_ray_tracing_denoiser("oidn")
            else:
                sapien.render.set_viewer_shader_dir("default")
                sapien.render.set_camera_shader_dir("default")

            # Create SAPIEN scene
            self.scene = sapien.Scene()
            self.scene.set_timestep(1 / 240)

            # Lighting configuration
            self.scene.set_environment_map(
                create_dome_envmap(sky_color=[0.5, 0.5, 0.5], ground_color=[0.5, 0.5, 0.5])
            )
            self.scene.add_directional_light(np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True)
            self.scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
            self.scene.set_ambient_light(np.array([0.3, 0.3, 0.3]))

            # Add a ground plane at Z = 0
            visual_material = sapien.render.RenderMaterial()
            visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
            visual_material.set_roughness(0.7)
            visual_material.set_metallic(1)
            visual_material.set_specular(0.04)
            self.scene.add_ground(0, render_material=visual_material)

            # Initialize the viewer
            self.viewer = Viewer()
            print("5")
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(1.5, 0, 1)
            self.viewer.set_camera_rpy(0, -0.8, 3.14)
            self.viewer.control_window.toggle_origin_frame(False)
            print("6")
            # Optional: Set log level to suppress warnings
            sapien.render.set_log_level("error")

    def update_scene(self):
        """
        Update and render the scene in real time.
        This function should be called in a loop to refresh the viewer.
        """
        if self.visualize:
            self.scene.update_render()
            self.viewer.render()