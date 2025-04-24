import os
import json
import pickle
from argparse import ArgumentParser
import shutil

import numpy as np
import open3d as o3d

from data_process.utils.path import PathResolver
from data_process.utils.data_reader import CameraInfo, read_config

CONTROLLER_NAME = "hand"

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, default=None)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--gaussian_data_path", type=str, required=True) #e.g. data/gaussian_data
    args = parser.parse_args()

    if args.raw_path is None:
        args.raw_path = args.base_path

    output_path = args.gaussian_data_path
    existDir(output_path)
    case_name = args.case_name

    path_resolver = PathResolver(args.raw_path, args.base_path, args.case_name)
    category, use_shape_prior = read_config(path_resolver)

    # Create the directory for the case
    existDir(f"{output_path}/{case_name}")
    for i in range(path_resolver.find_num_cam()):
        # Copy the original RGB image
        shutil.copy(
            f"{path_resolver.get_color_dir()}/{i}/0.png",
            f"{output_path}/{case_name}/{i}.png"
        )
        # Copy the original mask image
        # Get the mask path for the image
        with open(path_resolver.get_mask_info_path(i), "r") as f:
            data = json.load(f)
        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        shutil.copy(
            path_resolver.get_mask_frame_path(i,obj_idx,0),
            f"{output_path}/{case_name}/mask_{i}.png"
        )
        # Copy the original depth image
        shutil.copy(
            path_resolver.get_depth_frame_path(i,0),
            f"{output_path}/{case_name}/{i}_depth.npy"
        )

        # ToDo: export below
        # # Prepare the high-resolution image
        # os.system(
        #     f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/{i}/0.png --output_path {output_path}/{case_name}/{i}_high.png --category {category}"
        # )
        # # Prepare the segmentation mask of the high-resolution image
        # os.system(
        #     f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}_high.png --TEXT_PROMPT {category} --output_path {output_path}/{case_name}/mask_{i}_high.png"
        # )
        # # Prepare the human mask for the low-resolution image and high-resolution image
        # os.system(
        #     f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}.png --TEXT_PROMPT 'human' --output_path {output_path}/{case_name}/mask_human_{i}.png"
        # )
        # os.system(
        #     f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}_high.png --TEXT_PROMPT 'human' --output_path {output_path}/{case_name}/mask_human_{i}_high.png"
        # )

    # Prepare the intrinsic and extrinsic parameters
    camera_info = CameraInfo(path_resolver)
    data = {}
    data["c2ws"] = camera_info.c2ws
    data["intrinsics"] = camera_info.intrinsics
    with open(f"{output_path}/{case_name}/camera_meta.pkl", "wb") as f:
        pickle.dump(data, f)

    # Prepare the shape initialization data
    # If with shape prior, then copy the shape prior data
    if use_shape_prior.lower() == "true":
        shutil.copy(
            path_resolver.final_mesh_glb,
            f"{output_path}/{case_name}/shape_prior.glb"
        )

    # Save the original pcd data into the world coordinate system
    obs_points = []
    obs_colors = []
    data = np.load(path_resolver.get_pcd_data_path(0))
    with open(path_resolver.processed_masks_pkl, "rb") as f:
        processed_masks = pickle.load(f)
    for i in range(3):
        points = data["points"][i]
        colors = data["colors"][i]
        mask = processed_masks[0][i]["object"]
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])

    obs_points = np.vstack(obs_points)
    obs_colors = np.vstack(obs_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obs_points)
    pcd.colors = o3d.utility.Vector3dVector(obs_colors)
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([pcd, coordinate])
    o3d.io.write_point_cloud(f"{output_path}/{case_name}/observation.ply", pcd)
