import json
import pickle

import cv2
import numpy as np

from .path import PathResolver

def read_config(path:PathResolver):
    with open(path.data_config, newline="", encoding="utf-8") as f:
        line = f.readline()
    category = line.split(",")[0]
    use_shape_prior = line.split(",")[1]
    return category, use_shape_prior

class ImageReader:
    def __init__(self, path_resolver:PathResolver):
        self.path = path_resolver

    def load_color_frame(self, camera_idx:int, frame_idx:int):
        raw_img = cv2.imread(self.path.get_color_frame_path(camera_idx,frame_idx))
        return cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(mask_path:str):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask > 0
        return mask
    
    def read_mask_frame(self, camera_idx:int, obj_idx:int, frame_idx:int):
        # Convert the white mask into binary mask
        mask_path = self.path.get_mask_frame_path(camera_idx, obj_idx, frame_idx)
        return self._read_mask(mask_path)

    def read_first_frame_mask_of_all_objects(self, camera_idx:int):
        mask = None
        for mask_path in self.path.list_first_frame_mask_of_all_objects(camera_idx):
            current_mask = self._read_mask(mask_path)
            if mask is None:
                mask = current_mask
            else:
                mask = np.logical_or(mask, current_mask)
        return mask

    def read_object_bbox_and_mask(self, camera_idx:int, frame_idx:int):
        mask_path = self.path.get_object_mask_frame_path(camera_idx,frame_idx)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_binary = np.argwhere(mask_img > 0.8 * 255)
        bbox = (
            np.min(mask_binary[:, 1]),
            np.min(mask_binary[:, 0]),
            np.max(mask_binary[:, 1]),
            np.max(mask_binary[:, 0])
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size_margin = size * 1.2
        bbox_margin = (
            int(center[0] - size_margin / 2),
            int(center[1] - size_margin / 2),
            int(center[0] + size_margin / 2),
            int(center[1] + size_margin / 2)
        )

        # Make sure the bounding box is within the image
        bbox_margin = (
            max(0, bbox_margin[0]),
            max(0, bbox_margin[1]),
            min(mask_img.shape[1], bbox_margin[2]),
            min(mask_img.shape[0], bbox_margin[3]),
        )

        return bbox_margin, mask_img

    def read_object_cropped_image(self, camera_idx:int, frame_idx:int):
        # Get mask bounding box, larger than the original bounding box
        bbox, mask_img = self.read_object_bbox_and_mask(camera_idx,frame_idx)

        img = self.load_color_frame(camera_idx, frame_idx)
        mask_bool = mask_img > 0
        img[~mask_bool] = 0
        crop_img = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        return crop_img, bbox

def trans_points(mat, points):
    points=np.asarray(points)
    trans_points = np.dot(
        mat,
        np.hstack((
            points,
            np.ones((points.shape[0],1))
        )).T
    ).T
    
    return trans_points[:, :3]

class CameraInfo:
    def __init__(self, path_resolver:PathResolver):
        # Load the metadata
        with open(path_resolver.get_camera_metadata_path(), "r") as f:
            data = json.load(f)
        self.intrinsics = np.array(data["intrinsics"])
        self.WH = data["WH"]

        # Load the c2w for the camera
        with open(path_resolver.get_camera_calibrate_pkl_path(), "rb") as f:
            self.c2ws = np.asarray(pickle.load(f))
            self.w2cs = [np.linalg.inv(c2w) for c2w in self.c2ws]

    def calc_fov_horizontal(self, raw_img, camera_idx=0):
        fov = 2 * np.arctan(raw_img.shape[1] / (2 * self.intrinsics[camera_idx][0, 0]))
        return fov
    
    def convert_to_camera_coord(self, points_world_coord, camera_idx):
        return trans_points(self.w2cs[camera_idx],points_world_coord)

