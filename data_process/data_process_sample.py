# Optionally do the shape completion for the object points (including both suface and interior points)
# Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
import cv2

from utils.align_util import as_mesh
from .utils.path import PathResolver


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


class SampleProcessor:
    def __init__(
            self,
            raw_path:str,
            base_path:str,
            case_name:str,
            *,
            show_window=False,
            use_shape_prior = False, # Used to judge if using the shape prior
            num_surface_points = 1024,
            volume_sample_size = 0.005            
        ):
        self.path = PathResolver(raw_path,base_path,case_name)
        self.show_window = show_window
        self.use_shape_prior = use_shape_prior
        self.num_surface_points = num_surface_points
        self.volume_sample_size = volume_sample_size

    def output_exists(self):
        if not os.path.exists(self.path.final_data_pkl):
            return False
        return True

    def process(self):
        if self.output_exists():
            print("SKIP: output already exists")
            return False

        with open(self.path.tarck_process_data_pkl, "rb") as f:
            track_data = pickle.load(f)

        final_track_data = self._process_unique_points(track_data)

        with open(self.path.final_data_pkl, "wb") as f:
            pickle.dump(final_track_data, f)

        self._visualize_track(final_track_data)

    def _visualize_track(self, track_data):
        object_points = track_data["object_points"]
        object_visibilities = track_data["object_visibilities"]
        controller_points = track_data["controller_points"]

        frame_num = object_points.shape[0]

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=self.show_window)
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            self.path.final_data_video , fourcc, 30, (width, height)
        )

        controller_meshes = []
        prev_center = []

        y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
        y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
        rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

        for i in range(frame_num):
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_visibilities[i])[0], :]
            )
            # object_pcd.colors = o3d.utility.Vector3dVector(
            #     object_colors[i, np.where(object_motions_valid[i])[0], :]
            # )
            object_pcd.colors = o3d.utility.Vector3dVector(
                rainbow_colors[np.where(object_visibilities[i])[0]]
            )

            if i == 0:
                render_object_pcd = object_pcd
                vis.add_geometry(render_object_pcd)
                # Use sphere mesh for each controller point
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    origin_color = [1, 0, 0]
                    controller_meshes.append(
                        getSphereMesh(origin, color=origin_color, radius=0.01)
                    )
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
            else:
                render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
                render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
                vis.update_geometry(render_object_pcd)
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    controller_meshes[j].translate(origin - prev_center[j])
                    vis.update_geometry(controller_meshes[j])
                    prev_center[j] = origin
                vis.poll_events()
                vis.update_renderer()

            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

    def _sample_object_points_based_on_shape_prior(self, object_points, surface_points, interior_points, min_bound, grid_flag, index):
        final_surface_points = []
        for i in range(surface_points.shape[0]):
            grid_index = tuple(
                np.floor((surface_points[i] - min_bound) / self.volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_surface_points.append(surface_points[i])
        final_interior_points = []
        for i in range(interior_points.shape[0]):
            grid_index = tuple(
                np.floor((interior_points[i] - min_bound) / self.volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_interior_points.append(interior_points[i])
        all_points = np.concatenate(
            [final_surface_points, final_interior_points, object_points[0][index]],
            axis=0,
        )
        return final_surface_points, final_interior_points, all_points

    def _make_index(self, object_points, min_bound):
        index = []
        grid_flag = {}
        for i in range(object_points.shape[1]):
            grid_index = tuple(
                np.floor((object_points[0, i] - min_bound) / self.volume_sample_size).astype(int)
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                index.append(i)
        return index, grid_flag
    
    def _render_pcd_video(self, all_points):
        # Render the final pcd with interior filling as a turntable video
        all_pcd = o3d.geometry.PointCloud()
        all_pcd.points = o3d.utility.Vector3dVector(all_points)
 
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            self.path.final_pcd_video, fourcc, 30, (width, height)
        )

        vis.add_geometry(all_pcd)
        view_control = vis.get_view_control()
        for j in range(360):
            view_control.rotate(10, 0)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        vis.destroy_window()

    def _process_unique_points(self, track_data):
        # Get the unique index in the object points
        unique_idx = np.unique(
            track_data["object_points"][0],
            axis=0,
            return_index=True
        )[1]
        object_points = track_data["object_points"][:, unique_idx, :]
        object_colors = track_data["object_colors"][:, unique_idx, :]
        object_visibilities = track_data["object_visibilities"][:, unique_idx]
        object_motions_valid = track_data["object_motions_valid"][:, unique_idx]

        # Make sure all points are above the ground
        object_points[object_points[..., 2] > 0, 2] = 0

        if self.use_shape_prior:
            trimesh_mesh = trimesh.load(self.path.final_mesh_glb, force="mesh")
            trimesh_mesh = as_mesh(trimesh_mesh)
            # Sample the surface points
            surface_points, _ = trimesh.sample.sample_surface(
                trimesh_mesh, self.num_surface_points
            )
            # Sample the interior points
            interior_points = trimesh.sample.volume_mesh(trimesh_mesh, 10000)

            all_points = np.concatenate(
                [surface_points, interior_points, object_points[0]], axis=0
            )
        else:
            all_points = object_points[0]

        # Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points
        min_bound = np.min(all_points, axis=0)
        index, grid_flag = self._make_index(object_points, min_bound)

        if self.use_shape_prior:
            final_surface_points, final_interior_points, all_points = self._sample_object_points_based_on_shape_prior(
                object_points, surface_points, interior_points, min_bound, grid_flag, index
            )
        else:
            all_points = object_points[0][index]

        self._render_pcd_video(all_points)

        final_track_data = {
            "object_points" : object_points[:, index, :],
            "object_colors" : object_colors[:, index, :],
            "object_visibilities" : object_visibilities[:, index],
            "object_motions_valid" : object_motions_valid[:, index],
            "surface_points" : np.array(final_surface_points) if self.use_shape_prior else np.zeros((0, 3)),
            "interior_points" : np.array(final_interior_points) if self.use_shape_prior else np.zeros((0, 3)),
        }
        return final_track_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--show_window", action='store_true' )
    parser.add_argument("--shape_prior", action="store_true", default=False)
    parser.add_argument("--num_surface_points", type=int, default=1024)
    parser.add_argument("--volume_sample_size", type=float, default=0.005)
    args = parser.parse_args()

    sp = SampleProcessor(
        args.raw_path,
        args.base_path,
        args.case_name,
        show_window=args.show_window,
        use_shape_prior = args.shape_prior, # Used to judge if using the shape prior
        num_surface_points = args.num_surface_points,
        volume_sample_size = args.volume_sample_size
    )
    sp.process()

    
