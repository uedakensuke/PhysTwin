import cv2
import numpy as np

from .path import PathResolver

class DataReader:
    def __init__(self, path_resolver:PathResolver):
        self.path = path_resolver

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
   