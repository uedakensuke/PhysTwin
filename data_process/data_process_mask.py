# Process the mask data to filter out the outliers and generate the processed masks

import os
import glob
import json
import pickle
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
from tqdm import tqdm
import cv2

from .utils.path import PathResolver
from .utils.data import DataReader


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


class MaskProcessor:
    def __init__(self, raw_path:str, base_path:str, case_name:str, *, controller_name="hand"):
        self.path = PathResolver(raw_path,base_path,case_name, controller_name=controller_name)
        self.data = DataReader(self.path)

        # Load the mask metadata
        self.mask_info = {
            i:self._get_mask_info(i,controller_name)
            for i in range(self.path.find_num_cam())
        }

    def _get_mask_info(self, camera_idx:int, controller_name:str):
        with open(self.path.get_mask_info_path(camera_idx), "r") as f:
            data = json.load(f)
        mask_info = {}
        for key, value in data.items():
            if value != controller_name:
                if "object" in mask_info[camera_idx]:
                    # TODO: Handle the case when there are multiple objects
                    import pdb
                    pdb.set_trace()
                mask_info["object"] = int(key)
            if value == controller_name:
                if "controller" in mask_info[camera_idx]:
                    mask_info["controller"].append(int(key))
                else:
                    mask_info["controller"] = [int(key)]
        return mask_info

    def process(self):

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        object_pcd = None
        controller_pcd = None
        processed_masks = {}

        for i in tqdm(range(self.path.find_num_frame())):
            processed_mask, temp_object_pcd, temp_controller_pcd = self._process_pcd_mask(i)
            processed_masks[i] = processed_mask
            if i == 0:
                object_pcd = temp_object_pcd
                controller_pcd = temp_controller_pcd
                vis.add_geometry(object_pcd)
                vis.add_geometry(controller_pcd)
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
            else:
                object_pcd.points = o3d.utility.Vector3dVector(temp_object_pcd.points)
                object_pcd.colors = o3d.utility.Vector3dVector(temp_object_pcd.colors)
                controller_pcd.points = o3d.utility.Vector3dVector(
                    temp_controller_pcd.points
                )
                controller_pcd.colors = o3d.utility.Vector3dVector(
                    temp_controller_pcd.colors
                )
                vis.update_geometry(object_pcd)
                vis.update_geometry(controller_pcd)
                vis.poll_events()
                vis.update_renderer()

        # Save the processed masks considering both depth filter, semantic filter and outlier filter
        with open(self.path.processed_masks, "wb") as f:
            pickle.dump(processed_masks, f)

    def _create_rough_masked_pcd_from_all_camera(self, pcd_data:dict, frame_idx:int):
        # pcd_data["masks"] : array of shape[IMG_HEIGHT,IMG_WIDTH]. 値は深度有効範囲（0.2m～1.5m）内に点があるかのフラグ
        object_pcd = o3d.geometry.PointCloud()
        controller_pcd = o3d.geometry.PointCloud()
        for i in range(self.path.find_num_cam()):
            # Load the object mask
            object_mask = self.data.read_mask_frame(i, self.mask_info[i]["object"], frame_idx)
            pcd, _ = self._create_masked_pcd(pcd_data,object_mask,i)
            object_pcd += pcd

            # Load the controller mask
            controllers_mask = np.zeros_like(pcd_data["masks"][i])
            for controller_idx in self.mask_info[i]["controller"]:
                controller_mask = self.data.read_mask_frame(i, controller_idx, frame_idx)
                controllers_mask = np.logical_or(controllers_mask, controller_mask)
            pcd, _ = self._create_masked_pcd(pcd_data,controllers_mask,i)
            controller_pcd += pcd
        return object_pcd, controller_pcd

    @staticmethod
    def _get_processed_mask(mask, target_points, target_outlier_points):
        indices = np.nonzero(mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the object_points in the filtered points
        object_indices = []
        for j, point in enumerate(target_points):
            if tuple(point) in target_outlier_points:
                object_indices.append(j)
        original_indices = [indices_list[j] for j in object_indices]
        # Update the object mask
        for idx in original_indices:
            mask[idx[0], idx[1]] = 0
        return mask

    def _create_masked_pcd(self, pcd_data:dict, mask, camera_idx:int, outlier_points=None):
        valid_mask = np.logical_and(pcd_data["masks"][camera_idx], mask)

        if outlier_points is not None:
            points = pcd_data["points"][camera_idx][valid_mask]
            processed_mask = self._get_processed_mask(valid_mask, points, outlier_points)
        else:
            processed_mask=valid_mask

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data["points"][camera_idx][processed_mask])
        pcd.colors = o3d.utility.Vector3dVector(pcd_data["colors"][camera_idx][processed_mask])
        return pcd,processed_mask

    def _create_fine_masked_pcd_from_all_camera(self, pcd_data:dict, frame_idx:int, object_outlier_points, controller_outlier_points):
        # pcd_data["masks"] : array of shape[IMG_HEIGHT,IMG_WIDTH]. 値は深度有効範囲（0.2m～1.5m）内に点があるかのフラグ
        num_cam = self.path.find_num_cam()

        processed_mask= {}
        object_pcd = o3d.geometry.PointCloud()
        controller_pcd = o3d.geometry.PointCloud()
        for i in range(num_cam):
            processed_mask[i] = {}
            # Load the object mask
            object_mask = self.data.read_mask_frame(i, self.mask_info[i]["object"], frame_idx)
            pcd, object_processed_mask = self._create_masked_pcd(
                pcd_data,
                object_mask,
                i,
                object_outlier_points
            )
            object_pcd += pcd
            processed_mask[i]["object"] = object_processed_mask

            # Load the controller mask
            controllers_mask = np.zeros_like(pcd_data["masks"][i])
            for controller_idx in self.mask_info[i]["controller"]:
                controllers_mask = np.logical_or(
                    controllers_mask,
                    self.data.read_mask_frame(i, controller_idx, frame_idx)
                )
            pcd, controller_processed_mask = self._create_masked_pcd(
                pcd_data,
                controllers_mask,
                i,
                controller_outlier_points
            )
            controller_pcd += pcd
            processed_mask[i]["controller"] = controller_processed_mask

        return processed_mask, object_pcd, controller_pcd
    
    @staticmethod
    def _get_outlier(pcd):
        cl, ind = pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        outlier_points = np.asarray(
            pcd.select_by_index(ind, invert=True).points
        )
        return outlier_points

    def _process_pcd_mask(self, frame_idx:int):
        # Load the pcd data
        pcd_data = np.load(self.path.get_pcd_data_path(frame_idx))
        object_pcd, controller_pcd = self._create_rough_masked_pcd_from_all_camera(
            pcd_data, frame_idx)
        object_outlier_points = self._get_outlier(object_pcd)
        controller_outlier_points  =  self._get_outlier(controller_pcd)
        processed_mask, object_pcd, controller_pcd = self._create_fine_masked_pcd_from_all_camera(
            pcd_data, frame_idx, object_outlier_points, controller_outlier_points)
        # controller_pcd.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([object_pcd, controller_pcd])
        return processed_mask, object_pcd, controller_pcd


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--controller_name", type=str, default="hand")
    args = parser.parse_args()

    mp = MaskProcessor(args.raw_path, args.base_path, args.case_name, controller_name=args.controller_name)
    mp.process()
