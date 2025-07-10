import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import sapien
from hand_viewer import HandDatasetSAPIENViewer
from pytransform3d import rotations
from tqdm import trange
from sapien import internal_renderer as R
import subprocess
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting


class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        headless: bool = True,
        use_ray_tracing: bool = False,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        # load URDF
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True

        # for each robot_name load retargeting
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )
            # override: 6-dof dummy joint, make the root movable
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # construct robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                new_name = f"{urdf_path.stem}_glb{urdf_path.suffix}"
                urdf_path = urdf_path.with_name(new_name)

            # use yourdfpy to load, then use SAPIEN loader to read
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_path.name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)

            # record joint name mapping
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        """load hand+objects, then for each robot load the same objects"""
        super().load_object_hand(data)

        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # for n, each have the same YCB objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(
        self, data: Dict, fps=5, y_offset=0.8, out_video: str = None
    ):
        """visualize DexYCB datasets: if headless and out_video not None, write into Mp4

        Args:
            data: DexYCB里读到的sampled_data, 包含手pose和物体pose等信息
            fps: 渲染帧率
            y_offset: 不同机器人在y方向的偏移, 用于排开显示
            out_video: 若指定且在headless模式下, 则将画面写成out_video
        """
        # 1) 一些固定的相机位置/桌子位置，仅影响可视化
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            # under headless, no interactive viewer, but we could adjust the Camera Pose ourselves
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        # 2) read human gesture and object pose
        hand_pose = data["hand_pose"]      # shape: [T, ...]
        object_pose = data["object_pose"]  # shape: [T, n_objects, ...]
        num_frame = hand_pose.shape[0]
        num_ycb_objects = len(data["ycb_ids"])

        # lots of copies (human hand + n robots)
        num_copy = len(self.robots) + 1
        # set Pose offset for each copy
        pose_offsets = []
        for i in range(num_copy):
            offset_pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(offset_pose)
            if i >= 1:
                # 第i个copy对应self.robots[i-1]
                self.robots[i - 1].set_pose(offset_pose)

        # 3) skip "human hands infeasible" initial frame
        start_frame = 0
        for i in range(num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        # 4) if headless + out_video => initialize VideoWriter
        writer = None
        if self.headless and out_video:
            out_path = Path(out_video).absolute()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temp_video_file = out_path.with_name(f"{out_path.stem}_temp.mp4")

            width, height = self.camera.get_width(), self.camera.get_height()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_video_file), fourcc, float(fps), (width, height))

        # 5) robot retargeting warm_start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(hand_pose_start[0, 0:3])
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        # 6) rendering main loop
        step_per_frame = int(60 / fps)  # 在非headless模式下，可以让render多走几次保证流畅
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # (a) renew YCB pose
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]  # [7] => (x, y, z, qx, qy, qz, qw)
                pose = self.camera_pose * sapien.Pose(
                    pos_quat[4:],  # wxyz
                    np.concatenate([pos_quat[3:4], pos_quat[:3]]),
                )
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(
                        pose_offsets[copy_ind] * pose
                    )

            # (b) update human hand
            self._update_hand(vertex)

            # (c) update robots
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            # (d) render
            self.scene.update_render()

            if self.headless:
                if writer is not None:
                    self.camera.take_picture()
                    rgb = self.camera.get_picture("Color")[..., :3]  # shape [H, W, 3], float32 in [0,1]
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    writer.write(rgb[..., ::-1])
            else:
                # 实时 GUI 渲染
                for _ in range(step_per_frame):
                    self.viewer.render()

        # 7) end
        if self.headless:
            if writer is not None:
                writer.release()
                print(f"[INFO] MP4 saved to: {out_video}")
                subprocess.run(["ffmpeg", "-y", "-i", str(temp_video_file), "-c:v", "libx264", "-crf", "23", "-preset", "fast", str(out_path)])
                temp_video_file.unlink()
        else:
            self.viewer.paused = True
            self.viewer.render()