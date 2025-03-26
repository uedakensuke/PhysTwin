from pathlib import Path
import csv
import json
import pickle
import numpy as np
import open3d as o3d
import shutil
import subprocess

base_path = Path("./data/different_types")
output_path = Path("./data/gaussian_data")
CONTROLLER_NAME = "hand"


def ensure_dir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


ensure_dir(output_path)

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        case_base_path = base_path / case_name
        if not case_base_path.exists():
            continue

        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        # Create the directory for the case
        case_output_path = output_path / case_name
        ensure_dir(case_output_path)

        for i in range(3):
            # Copy the original RGB image
            shutil.copy2(
                case_base_path / "color" / str(i) / "0.png",
                case_output_path / f"{i}.png"
            )

            # Copy the original mask image
            # Get the mask path for the image
            with open(case_base_path / "mask" / f"mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_idx = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)
            mask_path = case_base_path / "mask" / str(i) / str(obj_idx) / "0.png"
            shutil.copy2(mask_path, case_output_path / f"mask_{i}.png")

            # Prepare the high-resolution image
            subprocess.run([
                "python", "./data_process/image_upscale.py",
                "--img_path", str(case_base_path / "color" / str(i) / "0.png"),
                "--output_path", str(case_output_path / f"{i}_high.png"),
                "--category", category
            ])

            # Prepare the segmentation mask of the high-resolution image
            subprocess.run([
                "python", "./data_process/segment_util_image.py",
                "--img_path", str(case_output_path / f"{i}_high.png"),
                "--TEXT_PROMPT", category,
                "--output_path", str(case_output_path / f"mask_{i}_high.png")
            ])

            # Copy the original depth image
            shutil.copy2(
                case_base_path / "depth" / str(i) / "0.npy",
                case_output_path / f"{i}_depth.npy"
            )

            # Prepare the human mask for the low-resolution image and high-resolution image
            subprocess.run([
                "python", "./data_process/segment_util_image.py",
                "--img_path", str(case_output_path / f"{i}.png"),
                "--TEXT_PROMPT", "human",
                "--output_path", str(case_output_path / f"mask_human_{i}.png")
            ])
            subprocess.run([
                "python", "./data_process/segment_util_image.py",
                "--img_path", str(case_output_path / f"{i}_high.png"),
                "--TEXT_PROMPT", "human",
                "--output_path", str(case_output_path / f"mask_human_{i}_high.png")
            ])

        # Prepare the intrinsic and extrinsic parameters
        with open(case_base_path / "calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        with open(case_base_path / "metadata.json", "r") as f:
            intrinsics = json.load(f)["intrinsics"]
        data = {}
        data["c2ws"] = c2ws
        data["intrinsics"] = intrinsics
        with open(case_output_path / "camera_meta.pkl", "wb") as f:
            pickle.dump(data, f)

        # Prepare the shape initialization data
        # If with shape prior, then copy the shape prior data
        if shape_prior.lower() == "true":
            shutil.copy2(
                case_base_path / "shape" / "matching" / "final_mesh.glb",
                case_output_path / "shape_prior.glb"
            )

        # Save the original pcd data into the world coordinate system
        obs_points = []
        obs_colors = []
        pcd_path = case_base_path / "pcd" / "0.npz"
        processed_mask_path = case_base_path / "mask" / "processed_masks.pkl"
        data = np.load(pcd_path)
        with open(processed_mask_path, "rb") as f:
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
        o3d.io.write_point_cloud(str(case_output_path / "observation.ply"), pcd)
