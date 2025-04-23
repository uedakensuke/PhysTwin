import json
import os
import pickle
from argparse import ArgumentParser

import open3d as o3d
import numpy as np
import trimesh
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree

from .utils.align_util import (
    render_multi_images,
    render_image,
    calc_intrinsics,
    as_mesh,
    project_2d_to_3d,
    plot_mesh_with_points,
    plot_image_with_points,
    select_point,
)
from .utils.match_pairs import image_pair_matching
from .utils.path import PathResolver
from .utils.data import ImageReader, CameraInfo, trans_points

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def pose_selection_render_superglue(
    raw_img, fov, mesh_path, mesh, crop_img, output_dir
):
    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)

    # Render multimle images and feature matching
    # Calculate intrinsics
    colors, depths, camera_poses = render_multi_images(
        mesh_path,
        raw_img.shape[1],
        raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device="cuda",
    )
    # Use superglue to match the features
    best_idx, match_result = image_pair_matching(
        [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors],
        cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY),
        output_dir,
        viz_best=True
    )
    print("matched point number", np.sum(match_result["matches"] > -1))

    best_color = colors[best_idx]
    best_depth = depths[best_idx]
    best_pose = camera_poses[best_idx].cpu().numpy()
    return best_color, best_depth, best_pose, match_result


def registration_pnp(mesh_matching_points, raw_matching_points, intrinsic):
    # Solve the PNP and verify the reprojection error
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )
    assert success, "solvePnP failed"
    projected_points, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        tvec,
        intrinsic,
        np.zeros(4, dtype=np.float32),
    )
    error = np.linalg.norm(
        np.float32(raw_matching_points) - projected_points.reshape(-1, 2), axis=1
    ).mean()
    print(f"Reprojection Error: {error}")
    if error > 50:
        print(f"solvePnP failed.$$$$$$$$$$$$$$$$$$$$$$$$$$")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    mesh2raw_camera = np.eye(4, dtype=np.float32)
    mesh2raw_camera[:3, :3] = rotation_matrix
    mesh2raw_camera[:3, 3] = tvec.squeeze()

    return mesh2raw_camera


def registration_scale(mesh_matching_points_cam, matching_points_cam):
    # After PNP, optimize the scale in the camera coordinate
    def objective(scale, mesh_points, pcd_points):
        transformed_points = scale * mesh_points
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))
        return loss

    initial_scale = 1
    result = minimize(
        objective,
        initial_scale,
        args=(mesh_matching_points_cam, matching_points_cam),
        method="L-BFGS-B",
    )
    optimal_scale = result.x[0]
    print("Rescale:", optimal_scale)
    return optimal_scale


def deform_ARAP(initial_mesh_world, mesh_matching_points_world, matching_points):
    # Do the ARAP deformation based on the matching keypoints
    mesh_vertices = np.asarray(initial_mesh_world.vertices)
    kdtree = KDTree(mesh_vertices)
    _, mesh_points_indices = kdtree.query(mesh_matching_points_world)
    mesh_points_indices = np.asarray(mesh_points_indices, dtype=np.int32)
    deform_mesh = initial_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(mesh_points_indices),
        o3d.utility.Vector3dVector(matching_points),
        max_iter=1,
    )
    return deform_mesh, mesh_points_indices


def get_matching_ray_registration(
    mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
):
    # Get the matching indices and targets based on the viewpoint
    obs_points_cam = trans_points(w2c,obs_points_world)
    vertices_cam = trans_points(w2c,mesh_world.vertices)

    obs_kd = KDTree(obs_points_cam)

    new_indices = []
    new_targets = []
    # trimesh used to do the ray-casting test
    mesh.vertices = np.asarray(vertices_cam)[trimesh_indices]
    for index, vertex in enumerate(vertices_cam):
        ray_origins = np.array([[0, 0, 0]])
        ray_direction = vertex
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        ray_directions = np.array([ray_direction])
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
        )

        ignore_flag = False

        if len(locations) > 0:
            first_intersection = locations[0]
            vertex_distance = np.linalg.norm(vertex)
            intersection_distance = np.linalg.norm(first_intersection)
            if intersection_distance < vertex_distance - 1e-4:
                # If the intersection point is not the vertex, it means the vertex is not visible from the camera viewpoint
                ignore_flag = True

        if ignore_flag:
            continue
        else:
            # Select the closest point to the ray of the observation points as the matching point
            indices = obs_kd.query_ball_point(vertex, 0.02)
            line_distances = line_point_distance(vertex, obs_points_cam[indices])
            # Get the closest point
            if len(line_distances) > 0:
                closest_index = np.argmin(line_distances)
                target = np.dot(
                    c2w, np.hstack((obs_points_cam[indices][closest_index], 1))
                )
                new_indices.append(index)
                new_targets.append(target[:3])

    new_indices = np.asarray(new_indices)
    new_targets = np.asarray(new_targets)

    return new_indices, new_targets


def deform_ARAP_ray_registration(
    deform_kp_mesh_world,
    obs_points_world,
    mesh,
    trimesh_indices,
    c2ws,
    w2cs,
    mesh_points_indices,
    matching_points,
):
    final_indices = []
    final_targets = []
    for index, target in zip(mesh_points_indices, matching_points):
        if index not in final_indices:
            final_indices.append(index)
            final_targets.append(target)

    for c2w, w2c in zip(c2ws, w2cs):
        new_indices, new_targets = get_matching_ray_registration(
            deform_kp_mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
        )
        for index, target in zip(new_indices, new_targets):
            if index not in final_indices:
                final_indices.append(index)
                final_targets.append(target)

    # Also need to adjust the positions to make sure they are above the table
    indices = np.where(np.asarray(deform_kp_mesh_world.vertices)[:, 2] > 0)[0]
    for index in indices:
        if index not in final_indices:
            final_indices.append(index)
            target = np.asarray(deform_kp_mesh_world.vertices)[index].copy()
            target[2] = 0
            final_targets.append(target)
        else:
            target = final_targets[final_indices.index(index)]
            if target[2] > 0:
                target[2] = 0
                final_targets[final_indices.index(index)] = target

    final_mesh_world = deform_kp_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(final_indices),
        o3d.utility.Vector3dVector(final_targets),
        max_iter=1,
    )
    return final_mesh_world


def line_point_distance(p, points):
    # Compute the distance between points and the line between p and [0, 0, 0]
    p = p / np.linalg.norm(p)
    points_to_origin = points
    cross_product = np.linalg.norm(np.cross(points_to_origin, p), axis=1)
    return cross_product / np.linalg.norm(p)

class AlignProcessor:
    def __init__(self, raw_path:str, base_path:str, case_name:str, *, controller_name="hand", show_window=False, export_visualize_files = True):
        self.path = PathResolver(raw_path,base_path,case_name, controller_name=controller_name)
        self.show_window = show_window
        self.export_visualize_files = export_visualize_files

        self.data = ImageReader(self.path)

    def _load_obs(self):
        obs_points = []
        obs_colors = []
        data = np.load(self.path.get_pcd_data_path(0))
        with open(self.path.processed_masks_pkl, "rb") as f:
            processed_masks = pickle.load(f)
        for i in range(3):
            points = data["points"][i]
            colors = data["colors"][i]
            mask = processed_masks[0][i]["object"]
            obs_points.append(points[mask])
            obs_colors.append(colors[mask])
            if i == 0:
                first_points = points
                first_mask = mask
        return np.vstack(obs_points), np.vstack(obs_colors), first_points, first_mask

    def process(self):
        output_dir = self.path.base_shape_matching_dir
        existDir(output_dir)

        cam_idx = 0
        camera_info = CameraInfo(self.path)

        # Load the shape prior
        mesh = trimesh.load_mesh(self.path.reconstruct_3d_model_glb, force="mesh")
        mesh = as_mesh(mesh)

        # Load and process the image to get a cropped version for easy superglue
        raw_img = self.data.load_color_frame(cam_idx, 0)

        # Calculate camera parameters
        fov = camera_info.calc_fov_horizontal(raw_img, cam_idx)

        if not os.path.exists(self.path.best_match_pkl):
            camera_intrinsics = calc_intrinsics(raw_img.shape[1],raw_img.shape[0],fov)

            # 2D feature Matching to get the best pose of the object
            # Get the masked cropped image used for superglue
            crop_img, bbox = self.data.read_object_cropped_image(cam_idx,0)

            # Render the object and match the features
            best_color, best_depth, best_pose, match_result = (
                pose_selection_render_superglue(
                    raw_img,
                    fov,
                    self.path.reconstruct_3d_model_glb,
                    mesh,
                    crop_img,
                    output_dir=self.path.base_shape_matching_dir,
                )
            )
            with open(self.path.best_match_pkl, "wb") as f:
                pickle.dump(
                    [
                        best_color,
                        best_depth,
                        best_pose,
                        match_result,
                        camera_intrinsics,
                        bbox,
                    ],
                    f,
                )
        else:
            with open(self.path.best_match_pkl, "rb") as f:
                best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
                    pickle.load(f)
                )

        # Process to get the matching points on the mesh and on the image
        # Get the projected 3D matching points on the mesh
        valid_matches = match_result["matches"] > -1
        render_matching_points = match_result["keypoints0"][valid_matches]
        mesh_matching_points, valid_mask = project_2d_to_3d(
            render_matching_points, best_depth, camera_intrinsics, best_pose
        )
        render_matching_points = render_matching_points[valid_mask]
        # Get the matching points on the raw image
        raw_matching_points_box = match_result["keypoints1"][
            match_result["matches"][valid_matches]
        ]
        raw_matching_points_box = raw_matching_points_box[valid_mask]
        raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])

        if self.export_visualize_files:
            # Do visualization for the matching
            plot_mesh_with_points(
                mesh,
                mesh_matching_points,
                self.path.mesh_matching_img,
            )
            plot_image_with_points(
                best_depth,
                render_matching_points,
                self.path.render_matching_img,
            )
            plot_image_with_points(
                raw_img,
                raw_matching_points,
                self.path.raw_matching_img,
            )

        # Do PnP optimization to optimize the rotation between the 3D mesh keypoints and the 2D image keypoints
        mesh2raw_camera = registration_pnp(
            mesh_matching_points,
            raw_matching_points,
            camera_info.intrinsics[cam_idx]
        )

        if self.export_visualize_files:
            pnp_camera_pose = np.eye(4, dtype=np.float32)
            pnp_camera_pose[:3, :3] = np.linalg.inv(mesh2raw_camera[:3, :3])
            pnp_camera_pose[3, :3] = mesh2raw_camera[:3, 3]
            pnp_camera_pose[:, :2] = -pnp_camera_pose[:, :2]
            color, depth = render_image(
                self.path.reconstruct_3d_model_glb,
                pnp_camera_pose,
                raw_img.shape[1],
                raw_img.shape[0],
                fov,
                "cuda"
            )
            vis_mask = depth > 0
            color[0][~vis_mask] = raw_img[~vis_mask]
            plt.imsave(self.path.pnp_results_img, color[0])

        # Transform the mesh into the real world coordinate
        mesh_matching_points_cam = trans_points(mesh2raw_camera,mesh_matching_points)

        # Load the pcd in world coordinate of raw image matching points
        obs_points, obs_colors, first_points, first_mask = self._load_obs()

        # Find the cloest points for the raw_matching_points
        new_match, matching_points = select_point(
            first_points, raw_matching_points, first_mask
        )
        matching_points_cam = camera_info.convert_to_camera_coord(matching_points,cam_idx)

        if self.export_visualize_files:
            # Draw the raw_matching_points and new matching points on the masked
            vis_img = raw_img.copy()
            vis_img[~first_mask] = 0
            plot_image_with_points(
                vis_img,
                raw_matching_points,
                self.path.raw_matching_valid_img,
                new_match,
            )

        # Use the matching points in the camera coordinate to optimize the scame between the mesh and the observation
        optimal_scale = registration_scale(mesh_matching_points_cam, matching_points_cam)

        # Compute the rigid transformation from the original mesh to the final world coordinate
        scale_matrix = np.eye(4) * optimal_scale
        scale_matrix[3, 3] = 1
        mesh2world = np.dot(camera_info.c2ws[cam_idx], np.dot(scale_matrix, mesh2raw_camera))

        mesh_matching_points_world = trans_points(mesh2world,mesh_matching_points)

        # Do the ARAP based on the matching keypoints
        # Convert the mesh to open3d to use the ARAP function
        initial_mesh_world = o3d.geometry.TriangleMesh()
        initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        # Need to remove the duplicated vertices to enable open3d, however, the duplicated points are important in trimesh for texture
        initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
        # Get the index from original vertices to the mesh vertices, mapping between trimesh and open3d
        kdtree = KDTree(initial_mesh_world.vertices)
        _, trimesh_indices = kdtree.query(np.asarray(mesh.vertices))
        trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
        initial_mesh_world.transform(mesh2world)

        # ARAP based on the keypoints
        deform_kp_mesh_world, mesh_points_indices = deform_ARAP(
            initial_mesh_world, mesh_matching_points_world, matching_points
        )

        # Do the ARAP based on both the ray-casting matching and the keypoints
        # Identify the vertex which blocks or blocked by the observation, then match them with the observation points on the ray
        final_mesh_world = deform_ARAP_ray_registration(
            deform_kp_mesh_world,
            obs_points,
            mesh,
            trimesh_indices,
            camera_info.c2ws,
            camera_info.w2cs,
            mesh_points_indices,
            matching_points,
        )

        if self.show_window or self.export_visualize_files:
            final_mesh_world.compute_vertex_normals()

            # Visualize the partial observation and the mesh
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obs_points)
            pcd.colors = o3d.utility.Vector3dVector(obs_colors)

            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # Render the final stuffs as a turntable video
            if self.show_window:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
            dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            height, width, _ = dummy_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(
                self.path.final_matching_video, fourcc, 30, (width, height)
            )
            # final_mesh_world.compute_vertex_normals()
            # final_mesh_world.translate([0, 0, 0.2])
            # mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(final_mesh_world)
            # o3d.visualization.draw_geometries([pcd, final_mesh_world], window_name="Matching")
            if self.show_window:
                vis.add_geometry(pcd)
                vis.add_geometry(final_mesh_world)
                # vis.add_geometry(coordinate)
                view_control = vis.get_view_control()

            for j in range(360):
                if self.show_window:
                    view_control.rotate(10, 0)
                    vis.poll_events()
                    vis.update_renderer()
                frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                frame = (frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            if self.show_window:
                vis.destroy_window()

        mesh.vertices = np.asarray(final_mesh_world.vertices)[trimesh_indices]
        mesh.export(self.path.final_mesh_glb)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--show_window", action='store_true' )
    parser.add_argument("--controller_name", type=str, required=True)
    args = parser.parse_args()

    ap = AlignProcessor(args.raw_path, args.base_path, args.case_name, show_window=args.show_window)
    ap.process()
