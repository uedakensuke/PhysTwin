# Merge the RGB-D data from multiple cameras into a single point cloud in world coordinate
# Do some depth filtering to make the point cloud more clean

import json
import pickle
import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

from .utils.path import PathResolver

DEPTH_MIN=0.2
DEPTH_MAX=1.5

# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/camera/cameraHelper.py
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/model/pcdHelper.py
def getPcdFromDepth(
    depth,
    intrinsic,
):
    # depth[IMG_Y,IMG_X] = 画像のIMG_X,IMG_Yの位置のdepth(m単位 ※合わせてintrinsicはm単位とすること)
    # points[IMG_Y,IMG_X,0] = 画像座標（IMG_X,IMG_Y）の点のカメラX座標
    # points[IMG_Y,IMG_X,1] = 画像座標（IMG_X,IMG_Y）の点のカメラY座標
    # points[IMG_Y,IMG_X,2] = 画像座標（IMG_X,IMG_Y）の点のカメラZ座標

    # Depth in meters
    height, width = np.shape(depth)

    # Reshape the depth array to invert the depth values
    depth = -depth

    # Create a grid of (x, y) coordinates
    x_coords = np.arange(width)
    y_coords = np.arange(height)

    # Create a meshgrid for x and y coordinates
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate points using vectorized operations
    old_points = np.stack([(width - X) * depth, Y * depth, depth], axis=-1)

    # Flatten the old_points array and calculate the new points using matrix multiplication
    points = np.dot(np.linalg.inv(intrinsic), old_points.reshape(-1, 3).T).T.reshape(
        old_points.shape
    )

    points[:, :, 1] *= -1
    points[:, :, 2] *= -1

    return points


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class PcdEstimateProcessor:
    def __init__(self, raw_path:str, base_path:str, case_name:str, num_cam = 3):
        self.path = PathResolver(raw_path,base_path,case_name)
        self.num_cam = num_cam

        with open(self.path.raw_camera_meta, "r") as f:
            data = json.load(f)
        self.intrinsics = np.array(data["intrinsics"])
        self.frame_num = data["frame_num"]
        self.c2ws = pickle.load(open(self.path.raw_camera_calibrate, "rb"))

        assert num_cam == len(self.intrinsics)

    def _get_pcd_from_data(self, frame_idx):
        # 複数のカメラから得られる点群を統合して返します
        total_points = []
        total_colors = []
        total_masks = []
        for i in range(self.num_cam):
            color = cv2.imread(self.path.get_color_frame_path(i,frame_idx))
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            color = color.astype(np.float32) / 255.0
            depth = np.load(self.path.get_depth_frame_path(i,frame_idx)) / 1000.0

            points = getPcdFromDepth(
                depth,
                intrinsic=self.intrinsics[i],
            )
            masks = np.logical_and(points[:, :, 2] > DEPTH_MIN, points[:, :, 2] < DEPTH_MAX)
            points_flat = points.reshape(-1, 3)
            # Transform points to world coordinates using homogeneous transformation
            homogeneous_points = np.hstack(
                (points_flat, np.ones((points_flat.shape[0], 1)))
            )
            points_world = np.dot(self.c2ws[i], homogeneous_points.T).T[:, :3]
            points_final = points_world.reshape(points.shape)
            total_points.append(points_final)
            total_colors.append(color)
            total_masks.append(masks)
        # pcd = o3d.geometry.PointCloud()
        # visualize_points = []
        # visualize_colors = []
        # for i in range(num_cam):
        #     visualize_points.append(
        #         total_points[i][total_masks[i]].reshape(-1, 3)
        #     )
        #     visualize_colors.append(
        #         total_colors[i][total_masks[i]].reshape(-1, 3)
        #     )
        # visualize_points = np.concatenate(visualize_points)
        # visualize_colors = np.concatenate(visualize_colors)
        # coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        # mask = np.logical_and(visualize_points[:, 2] > -0.15, visualize_points[:, 0] > -0.05)
        # mask = np.logical_and(mask, visualize_points[:, 0] < 0.4)
        # mask = np.logical_and(mask, visualize_points[:, 1] < 0.5)
        # mask = np.logical_and(mask, visualize_points[:, 1] > -0.2)
        # mask = np.logical_and(mask, visualize_points[:, 2] < 0.2)
        # visualize_points = visualize_points[mask]
        # visualize_colors = visualize_colors[mask]
            
        # pcd.points = o3d.utility.Vector3dVector(np.concatenate(visualize_points).reshape(-1, 3))
        # pcd.colors = o3d.utility.Vector3dVector(np.concatenate(visualize_colors).reshape(-1, 3))
        # o3d.visualization.draw_geometries([pcd])
        total_points = np.asarray(total_points)
        total_colors = np.asarray(total_colors)
        total_masks = np.asarray(total_masks)
        return total_points, total_colors, total_masks


    def get_cameras(self):
        cameras = []
        # Visualize the cameras
        for i in range(len(self.intrinsics)):
            camera = getCamera(
                self.c2ws[i],
                self.intrinsics[i, 0, 0],
                self.intrinsics[i, 1, 1],
                self.intrinsics[i, 0, 2],
                self.intrinsics[i, 1, 2],
                z_flip=True,
                scale=0.2,
            )
            cameras += camera
        return cameras

    def process(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for camera in self.get_cameras():
            vis.add_geometry(camera)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(coordinate)

        exist_dir(self.path.base_pcd_dir)
        for i in tqdm(range(self.frame_num)):
            points, colors, masks = self._get_pcd_from_data(i)

            if i == 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    points.reshape(-1, 3)[masks.reshape(-1)]
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    colors.reshape(-1, 3)[masks.reshape(-1)]
                )
                vis.add_geometry(pcd)
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
            else:
                pcd.points = o3d.utility.Vector3dVector(
                    points.reshape(-1, 3)[masks.reshape(-1)]
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    colors.reshape(-1, 3)[masks.reshape(-1)]
                )
                vis.update_geometry(pcd)

                vis.poll_events()
                vis.update_renderer()

            np.savez(
                self.path.get_pcd_data_path(i),
                points=points,
                colors=colors,
                masks=masks,
            )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    pep = PcdEstimateProcessor(args.raw_path, args.base_path, args.case_name)
    pep.process()
