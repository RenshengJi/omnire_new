import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

from datasets.tools.multiprocess_utils import track_parallel_progress
from utils.geometry import get_corners, project_camera_points_to_image
from utils.visualization import color_mapper, dump_3d_bbox_on_image
from .nuplan_utils import get_egopose3d_for_lidarpc_token_from_db, get_tracked_objects_for_lidarpc_token_from_db

NUPLAN_LABELS = [
    'vehicle', 'pedestrian', 'bicycle'
]
NUPLAN_NONRIGID_DYNAMIC_CLASSES = [
    'pedestrian', 'bicycle'
]
NUPLAN_RIGID_DYNAMIC_CLASSES = [
    'vehicle'
]
NUPLAN_DYNAMIC_CLASSES = NUPLAN_NONRIGID_DYNAMIC_CLASSES + NUPLAN_RIGID_DYNAMIC_CLASSES

class NuPlanProcessor(object):
    """Process NUPLAN Dataset
    
    NuPlan Datasets provides 8 cameras and Merged Lidar data.
    Cameras works in 10Hz and Lidar works in 20Hz. Thus we process at 10Hz.
    The duration of each scene is around 8 mins, resulting in ~5000 frames.
    We only process the first max_frame_limit(default=300) frames.

    Args:
        load_dir (str): Directory to load data.
        save_dir (str): Directory to save data in processed format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
        process_keys (list, optional): List of keys to process. Default: ["images", "lidar", "calib", "dynamic_masks", "objects"]
        process_log_list (list, optional): List of scene indices to process. Default: None
    """

    def __init__(
        self,
        load_dir='data/nuplan/raw',
        save_dir='data/nuplan/processed',
        prefix='mini',
        # We skip the first start_frame_idx frames to avoid ego static frames
        start_frame_idx=1000,
        # We only process the max_frame_limit frames
        max_frame_limit=300,
        process_keys=[
            "images",
            "pose",
            "calib",
            "lidar",
            "dynamic_masks",
            "objects"
        ],
        process_id_list=None,
        workers=64,
        undistort=False,
    ):
        self.HW = (1080, 1920)
        print("Raw Image Resolution: ", self.HW)
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)
        self.start_frame_idx = start_frame_idx
        print("We will skip the first {} frames".format(self.start_frame_idx))
        self.max_frame_limit = max_frame_limit
        print("We will process the first {} frames each scene".format(self.max_frame_limit))
        self.undistort = undistort
        if self.undistort:
            print("Undistortion is ENABLED: images will be undistorted and black borders will be cropped.")
        # the lidar data is collected at 20Hz, we need to downsample to 10Hz to match the camera data
        self.lidar_idxs = range(self.start_frame_idx, self.start_frame_idx + self.max_frame_limit * 2, 2)
        # Keep the tail boundary out of processing. Some logs have incomplete
        # camera matches near the final lidar frames.
        self.tail_lidar_margin = 20
        self.camera_match_window_us = 100000
        
        # NUPLAN Provides 8 cameras
        self.cam_list = [    # {frame_idx}_{cam_id}.jpg
            "CAM_F0",        # "xxx_0.jpg"
            "CAM_L0",        # "xxx_1.jpg"
            "CAM_R0",        # "xxx_2.jpg"
            "CAM_L1",        # "xxx_3.jpg"
            "CAM_R1",        # "xxx_4.jpg"
            "CAM_L2",        # "xxx_5.jpg"
            "CAM_R2",        # "xxx_6.jpg"
            "CAM_B0"         # "xxx_7.jpg"
        ]
        
        self.sensor_blobs_dir = os.path.join(load_dir, 'nuplan-v1.1', 'sensor_blobs')
        self.split_dir = os.path.join(load_dir, 'nuplan-v1.1', 'splits', prefix)
        self.nuplandb_wrapper = NuPlanDBWrapper(
            data_root=os.path.join(load_dir, 'nuplan-v1.1'),
            map_root=os.path.join(load_dir, 'maps'),
            db_files=self.split_dir,
            map_version='nuplan-maps-v1.0',
        )
        
        self.process_log_list = process_id_list if process_id_list else self.nuplandb_wrapper.log_names
        
        self.save_dir = os.path.join(save_dir, prefix)
        self.workers = int(workers)
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_log_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_log_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, scene_log_name):
        """Convert action for single file."""
        # get log db
        log_db = self.nuplandb_wrapper.get_log_db(scene_log_name)

        # Clamp lidar_idxs to the actual number of lidar frames in this scene
        total_lidar_frames = len(log_db.lidar_pc)
        effective_total_lidar_frames = max(self.start_frame_idx, total_lidar_frames - self.tail_lidar_margin)
        lidar_idxs_full = list(self.lidar_idxs)
        lidar_idxs_full = [idx for idx in lidar_idxs_full if idx < effective_total_lidar_frames]
        if len(lidar_idxs_full) < len(list(self.lidar_idxs)):
            actual_limit = len(lidar_idxs_full)
            print(f"[{scene_log_name}] max_frame_limit exceeds scene length, clamped to {actual_limit} frames")

        # since lidar and images are captured at different frequency
        # we find the best start frame that lidar and images matches the best
        # lidar_idx:[0]   1   [2]   3   [4]   5   [6]
        # timestamp: 0   0.05 0.1  0.15 0.2  0.25 0.3
        # lidar_pc:  |    |    |    |    |    |    |
        # Images:    |         |         |         |
        # NOTE: the best match should be the frame with the closest timestamp to the lidar_pc (e.g. [0] [2] [4] [6])
        # calulate time shift of original start frame
        lidar_pc = log_db.lidar_pc[self.start_frame_idx]
        images = self.get_closest_images(log_db, lidar_pc)
        images_timestamps = [image.timestamp for image in images]
        lidar_timestamp = lidar_pc.timestamp
        no_shift_time_diff = [abs(lidar_timestamp - timestamp) for timestamp in images_timestamps]
        # calulate time shift of original start frame + 1
        lidar_pc = log_db.lidar_pc[self.start_frame_idx + 1]
        images = self.get_closest_images(log_db, lidar_pc)
        images_timestamps = [image.timestamp for image in images]
        lidar_timestamp = lidar_pc.timestamp
        shift_time_diff = [abs(lidar_timestamp - timestamp) for timestamp in images_timestamps]
        
        if sum(no_shift_time_diff) > sum(shift_time_diff):
            lidar_idxs = [idx + 1 for idx in lidar_idxs_full if idx + 1 < total_lidar_frames]
        else:
            lidar_idxs = lidar_idxs_full
        
        if "images" in self.process_keys:
            self.save_image(log_db, lidar_idxs)
            print(f"Processed images for {scene_log_name}")
        if "calib" in self.process_keys:
            self.save_calib(log_db, lidar_idxs)
            print(f"Processed calib for {scene_log_name}")
        if "lidar" in self.process_keys:
            self.save_lidar(log_db, lidar_idxs)
            print(f"Processed lidar for {scene_log_name}")
        if "pose" in self.process_keys:
            self.save_pose(log_db, lidar_idxs)
            self.save_timestamps(log_db, lidar_idxs)
            print(f"Processed pose for {scene_log_name}")
        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_mask(log_db, lidar_idxs, valid_classes=NUPLAN_DYNAMIC_CLASSES, dir_name='all')
            self.save_dynamic_mask(log_db, lidar_idxs, valid_classes=NUPLAN_NONRIGID_DYNAMIC_CLASSES, dir_name='human')
            self.save_dynamic_mask(log_db, lidar_idxs, valid_classes=NUPLAN_RIGID_DYNAMIC_CLASSES, dir_name='vehicle')
            print(f"Processed dynamic masks for {scene_log_name}")
        
        # process annotated objects
        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(log_db, lidar_idxs)
            print(f"Processed instances info for {scene_log_name}")
            
            # Save instances info and frame instances
            object_info_dir = f"{self.save_dir}/{scene_log_name}/instances"
            with open(f"{object_info_dir}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{object_info_dir}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)
                
            if "objects_vis" in self.process_keys:
                self.visualize_dynamic_objects(
                    log_db, lidar_idxs,
                    output_path=f"{object_info_dir}/debug_vis",
                    instances_info=instances_info,
                    frame_instances=frame_instances
                )
                print(f"Processed objects visualization for {scene_log_name}")

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_log_list)

    def save_image(self, log_db: NuPlanDB, lidar_idxs: List[int]):
        """Parse and save the images in jpg format.

        If self.undistort is True, images are undistorted and black borders are
        cropped before saving. Otherwise, the raw JPEG is copied as-is.
        """
        lidar_pcs = log_db.lidar_pc

        # Pre-compute undistortion parameters once per camera (they are static)
        undistort_params = {}  # channel -> (map1, map2, new_K, roi)
        if self.undistort:
            extrinsics, intrinsics, distortions = self.get_cameras_calib(log_db)
            for channel in self.cam_list:
                K = intrinsics[channel]
                dist = distortions[channel]
                map1, map2, new_K, roi = self.compute_undistort_params(K, dist, self.HW)
                undistort_params[channel] = (map1, map2, new_K, roi)

        for frame_idx, lidar_idx in enumerate(lidar_idxs):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]

            images = self.get_closest_images(log_db, lidar_pc)

            image_cnt = 0
            for cam_id, image in enumerate(images):
                raw_image_path = os.path.join(self.sensor_blobs_dir, image.filename_jpg)
                image_save_path = f"{self.save_dir}/{log_db.log_name}/images/{str(frame_idx).zfill(3)}_{cam_id}.jpg"

                if self.undistort:
                    channel = self.cam_list[cam_id]
                    map1, map2, new_K, roi = undistort_params[channel]
                    img = cv2.imread(raw_image_path)  # BGR
                    img_cropped = self.undistort_and_crop(img, map1, map2, roi)
                    cv2.imwrite(image_save_path, img_cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    os.system(f"cp {raw_image_path} {image_save_path}")
                image_cnt += 1

            assert image_cnt == len(self.cam_list), \
                f"Image number, camera number mismatch: {image_cnt} != {len(self.cam_list)}"

    def get_closest_images(self, log_db: NuPlanDB, lidar_pc):
        """Return one closest image per camera within the configured time window."""
        images = get_images_from_lidar_tokens(
            log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
            tokens=[lidar_pc.token],
            channels=self.cam_list,
            lookahead_window_us=self.camera_match_window_us,
            lookback_window_us=self.camera_match_window_us,
        )

        closest_images = {}
        for image in images:
            channel = image.channel
            time_diff = abs(image.timestamp - lidar_pc.timestamp)
            if channel not in closest_images or time_diff < closest_images[channel][0]:
                closest_images[channel] = (time_diff, image)

        missing_channels = [channel for channel in self.cam_list if channel not in closest_images]
        assert len(missing_channels) == 0, (
            f"Missing camera images within +/-{self.camera_match_window_us / 1000:.0f}ms "
            f"for {log_db.log_name} lidar token {lidar_pc.token}: {missing_channels}"
        )

        return [closest_images[channel][1] for channel in self.cam_list]
                
    def get_cameras_calib(self, log_db: NuPlanDB):
        """Get the camera calibration."""
        cameras = log_db.camera
        extrinsics, intrinsics, distortions = {}, {}, {}
        for cam in cameras:
            channel = cam.channel

            extrinsic = Quaternion(cam.rotation).transformation_matrix
            extrinsic[:3, 3] = np.array(cam.translation)
            extrinsics[channel] = extrinsic

            intrinsic = np.array(cam.intrinsic)
            intrinsics[channel] = intrinsic

            distortions[channel] = np.array(cam.distortion)

        return extrinsics, intrinsics, distortions

    def compute_undistort_params(
        self, K: np.ndarray, dist: np.ndarray, hw: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        """Compute undistortion maps and new intrinsic matrix without black borders.

        Uses cv2.getOptimalNewCameraMatrix with alpha=0 to compute the largest
        all-valid (no black pixel) region, then returns the remap maps and the
        crop ROI so that the final image has no black borders.

        Args:
            K: 3x3 camera intrinsic matrix.
            dist: Distortion coefficients [k1, k2, p1, p2, k3].
            hw: (height, width) of the original image.

        Returns:
            map1, map2: cv2 remap maps (for cv2.remap).
            new_K: 3x3 new intrinsic matrix (after undistortion and crop).
            roi: (x, y, w, h) crop region in the undistorted image (no black border).
        """
        h, w = hw
        # alpha=0 → no black pixels in the output (crops as needed)
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)
        return map1, map2, new_K, roi

    def undistort_and_crop(
        self,
        img_bgr: np.ndarray,
        map1: np.ndarray,
        map2: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Apply remap and crop black borders from an image.

        Args:
            img_bgr: Input image (H, W, 3) in BGR or RGB — both are fine, remap is channel-agnostic.
            map1, map2: Remap maps from compute_undistort_params.
            roi: (x, y, w, h) crop region.

        Returns:
            Cropped undistorted image as numpy array.
        """
        undistorted = cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
        x, y, w, h = roi
        cropped = undistorted[y: y + h, x: x + w]
        return cropped

    def save_calib(self, log_db: NuPlanDB, lidar_idxs: List[int]):
        """Parse and save the calibration data.

        When self.undistort is True, the saved intrinsics reflect the undistorted
        (and cropped) image: fx/fy/cx/cy are updated from the optimal new camera
        matrix, and all distortion coefficients are set to zero.
        """
        extrinsics, intrinsics, distortions = self.get_cameras_calib(log_db)
        for channel in self.cam_list:
            cam_id = self.cam_list.index(channel)

            intrinsic = intrinsics[channel]

            if self.undistort:
                # Compute optimal new camera matrix (alpha=0 → no black pixels)
                dist = distortions[channel]
                h, w = self.HW
                new_K, roi = cv2.getOptimalNewCameraMatrix(intrinsic, dist, (w, h), alpha=0)
                # Adjust cx/cy for the crop offset so that the principal point is
                # correct relative to the cropped image origin.
                x_off, y_off, _, _ = roi
                fx = new_K[0, 0]
                fy = new_K[1, 1]
                cx = new_K[0, 2] - x_off
                cy = new_K[1, 2] - y_off
                # Undistorted image has no distortion
                k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
                k1, k2, p1, p2, k3 = distortions[channel]

            Ks = np.array([fx, fy, cx, cy, k1, k2, p1, p2, k3])

            np.savetxt(
                f"{self.save_dir}/{log_db.log_name}/extrinsics/"
                + f"{str(cam_id)}.txt",
                extrinsics[channel]
            )
            np.savetxt(
                f"{self.save_dir}/{log_db.log_name}/intrinsics/"
                + f"{str(cam_id)}.txt",
                Ks
            )

    def save_lidar(self, log_db: NuPlanDB, lidar_idxs: List[int]):
        """Parse and save the lidar data in psd format."""
        # NOTE: lidar points is already in ego pose frame
        lidar_pcs = log_db.lidar_pc
        for frame_idx, lidar_idx in enumerate(lidar_idxs):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]
            
            lidar_data: LidarPointCloud = lidar_pc.load(log_db)
            # version 1: a numpy array with 5 cols (x, y, z, intensity, ring).
            # version 2: a numpy array with 6 cols (x, y, z, intensity, ring, lidar_id).
            lidar_save_path = f"{self.save_dir}/{log_db.log_name}/lidar/{str(frame_idx).zfill(3)}.bin"
            lidar_data.points.T.astype(np.float32).tofile(lidar_save_path)
    
    def save_pose(self, log_db: NuPlanDB, lidar_idxs: List[int]):
        """Parse and save the pose data."""
        lidar_pcs = log_db.lidar_pc
        for frame_idx, lidar_idx in enumerate(lidar_idxs):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]
            
            ego_pose = get_egopose3d_for_lidarpc_token_from_db(
                log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
                token=lidar_pc.token
            )
            
            np.savetxt(
                f"{self.save_dir}/{log_db.log_name}/ego_pose/"
                + f"{str(frame_idx).zfill(3)}.txt",
                ego_pose
            )

    def save_timestamps(self, log_db: NuPlanDB, lidar_idxs: List[int]):
        """Save per-frame timestamp metadata for robust downstream clip matching."""
        lidar_pcs = log_db.lidar_pc
        frames = []
        for frame_idx, lidar_idx in enumerate(lidar_idxs):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]
            frames.append(
                {
                    "frame_idx": int(frame_idx),
                    "lidar_idx": int(lidar_idx),
                    "lidar_pc_token": str(lidar_pc.token),
                    "timestamp": int(lidar_pc.timestamp),
                    "timestamp_us": int(lidar_pc.timestamp),
                }
            )

        payload = {
            "log_name": log_db.log_name,
            "frames": frames,
        }
        with open(f"{self.save_dir}/{log_db.log_name}/timestamps.json", "w") as fp:
            json.dump(payload, fp, indent=2)

    def save_dynamic_mask(
        self, log_db: NuPlanDB, lidar_idxs: List[int],
        valid_classes: List[str], dir_name: str
    ):
        """Parse and save the segmentation data."""        
        extrinsics, intrinsics, _ = self.get_cameras_calib(log_db)
        lidar_pcs = log_db.lidar_pc
        for frame_idx, lidar_idx in enumerate(lidar_idxs):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]
            
            ego_pose = get_egopose3d_for_lidarpc_token_from_db(
                log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
                token=lidar_pc.token
            )
            
            objects_generator = get_tracked_objects_for_lidarpc_token_from_db(
                log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
                token=lidar_pc.token
            )
            objects = [obj for obj in objects_generator if obj.category in valid_classes]
            
            for channel in self.cam_list:
                cam_id = self.cam_list.index(channel)
                dynamic_mask = np.zeros(self.HW, dtype=np.float32)
                
                # compute dynamic mask according to the instances' bbox projection
                for obj in objects:
                    obj_to_world = obj.pose
                    l, w, h = obj.box_size
                    corners = get_corners(l, w, h)
                    corners_world = obj_to_world[:3, :3] @ corners + obj_to_world[:3, 3:4]
                    
                    world_to_ego = np.linalg.inv(ego_pose)
                    corners_ego = world_to_ego[:3, :3] @ corners_world + world_to_ego[:3, 3:4]
                    
                    ego_to_cam = np.linalg.inv(extrinsics[channel])
                    corners_cam = ego_to_cam[:3, :3] @ corners_ego + ego_to_cam[:3, 3:4]
                    
                    corners_2d, _ = project_camera_points_to_image(corners_cam.T, intrinsics[channel])

                    # Check if the object is in front of the camera and all corners are in the image
                    # NOTE: we use strict visibility check here, requiring all corners to be visible
                    in_front = np.all(corners_cam[2, :] > 0.1)
                    in_image = np.all(corners_2d[0, :] >= 0) & np.all(corners_2d[0, :] < self.HW[1]) & \
                            np.all(corners_2d[1, :] >= 0) & np.all(corners_2d[1, :] < self.HW[0])
                    if not (in_front and in_image):
                        continue

                    # Fill the mask
                    u, v = corners_2d[0, :].astype(np.int32), corners_2d[1, :].astype(np.int32)
                    u = np.clip(u, 0, self.HW[1] - 1) 
                    v = np.clip(v, 0, self.HW[0] - 1)

                    if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                        continue

                    xy = (u.min(), v.min())
                    width = u.max() - u.min()
                    height = v.max() - v.min()

                    dynamic_mask[
                        int(xy[1]): int(xy[1] + height),
                        int(xy[0]): int(xy[0] + width)
                    ] = np.maximum(
                        dynamic_mask[
                            int(xy[1]): int(xy[1] + height),
                            int(xy[0]): int(xy[0] + width)
                        ],
                        1
                    )
                    
                dynamic_mask = np.clip((dynamic_mask > 0.) * 255, 0, 255).astype(np.uint8)
                dynamic_mask = Image.fromarray(dynamic_mask, "L")
                dynamic_mask_path = os.path.join(
                    self.save_dir, log_db.log_name, "dynamic_masks", dir_name,
                    f"{str(frame_idx).zfill(3)}_{cam_id}.png"
                )
                dynamic_mask.save(dynamic_mask_path)

    def save_objects(self, log_db: NuPlanDB, lidar_idxs: List[int]):
        """Parse and save the objects annotation data."""
        instances_info, frame_instances = {}, {}
        
        lidar_pcs = log_db.lidar_pc
        for frame_idx, lidar_idx in enumerate(lidar_idxs):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]
            
            objects_generator = get_tracked_objects_for_lidarpc_token_from_db(
                log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
                token=lidar_pc.token
            )
            objects = [obj for obj in objects_generator if obj.category in NUPLAN_DYNAMIC_CLASSES]
            
            frame_instances[frame_idx] = []
            for obj in objects:
                obj_id = obj.track_token
                if obj_id not in instances_info:
                    instances_info[obj_id] = {
                        "id": obj_id,
                        "class_name": obj.category,
                        "frame_annotations": {
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        }
                    }
                
                obj_to_world = obj.pose
                l, w, h = obj.box_size
                
                instances_info[obj_id]['frame_annotations']['frame_idx'].append(frame_idx)
                instances_info[obj_id]['frame_annotations']['obj_to_world'].append(obj_to_world.tolist())
                instances_info[obj_id]['frame_annotations']['box_size'].append([l, w, h])
                
                frame_instances[frame_idx].append(obj_id)
        
        # Correct ID mapping
        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        # Update keys in instances_info
        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        # Update keys in frame_instances
        new_frame_instances = {}
        for k, v in frame_instances.items():
            new_frame_instances[k] = [id_map[i] for i in v]

        return new_instances_info, new_frame_instances

    def visualize_dynamic_objects(
        self, log_db: NuPlanDB,
        lidar_idxs: List[int], output_path: str, 
        instances_info: Dict, frame_instances: Dict
    ):
        """Visualize the dynamic objects' box with different colors on the image."""
        extrinsics, intrinsics, _ = self.get_cameras_calib(log_db)
        lidar_pcs = log_db.lidar_pc
        
        for frame_idx, lidar_idx in enumerate(lidar_idxs):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]
            
            ego_pose = get_egopose3d_for_lidarpc_token_from_db(
                log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
                token=lidar_pc.token
            )
            
            images = self.get_closest_images(log_db, lidar_pc)
            
            for cam_id, image in enumerate(images):
                raw_image_path = os.path.join(self.sensor_blobs_dir, image.filename_jpg)
                canvas = np.array(Image.open(raw_image_path))
                
                channel = self.cam_list[cam_id]
                objects = frame_instances[frame_idx]
                if len(objects) == 0:
                    img_plotted = canvas
                else:
                    lstProj2d = []
                    color_list = []
                    
                    for obj_id in objects:
                        idx_in_obj = instances_info[obj_id]['frame_annotations']['frame_idx'].index(frame_idx)
                        obj_to_world = np.array(instances_info[obj_id]['frame_annotations']['obj_to_world'][idx_in_obj])
                        l, w, h = instances_info[obj_id]['frame_annotations']['box_size'][idx_in_obj]
                        
                        corners = get_corners(l, w, h)
                        corners_world = obj_to_world[:3, :3] @ corners + obj_to_world[:3, 3:4]
                        
                        world_to_ego = np.linalg.inv(ego_pose)
                        corners_ego = world_to_ego[:3, :3] @ corners_world + world_to_ego[:3, 3:4]
                        
                        ego_to_cam = np.linalg.inv(extrinsics[channel])
                        corners_cam = ego_to_cam[:3, :3] @ corners_ego + ego_to_cam[:3, 3:4]
                        
                        corners_2d, _ = project_camera_points_to_image(corners_cam.T, intrinsics[channel])
                        
                        # Check if the object is in front of the camera and at least one corner is in the image
                        in_front = np.all(corners_cam[2, :] > 0.1)
                        intersected = np.any(
                            (corners_2d[0, :] >= 0) & (corners_2d[0, :] < self.HW[1]) & \
                            (corners_2d[1, :] >= 0) & (corners_2d[1, :] < self.HW[0])
                        )
                        if in_front and intersected:
                            lstProj2d.append(corners_2d.T.tolist())
                            color_list.append(color_mapper(str(obj_id)))
                    
                    img_plotted = dump_3d_bbox_on_image(coords=np.array(lstProj2d), img=canvas, color=color_list)
                
                img_path = os.path.join(output_path, f"{str(frame_idx).zfill(3)}_{cam_id}.jpg")
                Image.fromarray(img_plotted).save(img_path)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_log_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_log_list
        for scene_log_name in id_list:
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_log_name}/images", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{scene_log_name}/sky_masks", exist_ok=True)
            if "pose" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_log_name}/ego_pose", exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_log_name}/extrinsics", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{scene_log_name}/intrinsics", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_log_name}/lidar", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_log_name}/dynamic_masks/all", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{scene_log_name}/dynamic_masks/human", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{scene_log_name}/dynamic_masks/vehicle", exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_log_name}/instances", exist_ok=True)
                if "objects_vis" in self.process_keys:
                    os.makedirs(f"{self.save_dir}/{scene_log_name}/instances/debug_vis", exist_ok=True)
