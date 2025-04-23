import glob
import json
import os

class PathResolver:
    def __init__(self, raw_path:str, base_path:str, case_name:str, *, controller_name="hand"):
        self.raw_path = raw_path
        self.base_path = base_path
        self.case_name = case_name
        self.controller_name = controller_name

        #### raw_path配下
        self.raw_color_dir=f"{self.raw_path}/{self.case_name}/color"
        self.raw_depth_dir=f"{self.raw_path}/{self.case_name}/depth"
        self.camera_metadata=f"{self.raw_path}/{self.case_name}/metadata.json"
        self.camera_calibrate_pkl=f"{self.raw_path}/{self.case_name}/calibrate.pkl"

        #### base_path配下
        self.tarck_process_data_pkl = f"{self.base_path}/{self.case_name}/track_process_data.pkl"

        #### base_path配下(mask)
        self.base_mask_dir = f"{self.base_path}/{self.case_name}/mask"
        self.processed_masks_pkl = f"{self.base_mask_dir}/processed_masks.pkl"

        #### base_path配下(shape)
        self.base_shape_dir = f"{self.base_path}/{self.case_name}/shape"
        self.upscale_image = f"{self.base_shape_dir}/high_resolution.png"
        self.masked_upscale_image = f"{self.base_shape_dir}/masked_image.png"
        self.reconstruct_3d_model_glb = f"{self.base_shape_dir}/object.glb"
        self.reconstruct_3d_model_ply = f"{self.base_shape_dir}/object.ply"
        self.reconstruct_3d_model_video = f"{self.base_shape_dir}/visualization.mp4"

        #### base_path配下(shape/matching)
        self.base_shape_matching_dir = f"{self.base_path}/{self.case_name}/shape/matching"
        self.best_match_pkl = f"{self.base_shape_matching_dir}/best_match.pkl"
        self.mesh_matching_img = f"{self.base_shape_matching_dir}/mesh_matching.png"
        self.render_matching_img = f"{self.base_shape_matching_dir}/render_matching.png"
        self.raw_matching_img = f"{self.base_shape_matching_dir}/raw_matching.png"
        self.pnp_results_img = f"{self.base_shape_matching_dir}/pnp_results.png"
        self.raw_matching_valid_img = f"{self.base_shape_matching_dir}/raw_matching_valid.png"
        self.final_matching_video = f"{self.base_shape_matching_dir}/final_matching.mp4"
        self.final_mesh_glb = f"{self.base_shape_matching_dir}/final_mesh.glb"

        #### base_path配下(cotracker)
        self.base_cotracker_dir = f"{self.base_path}/{self.case_name}/cotracker"

        #### base_path配下(pcd)
        self.base_pcd_dir = f"{self.base_path}/{self.case_name}/pcd"

    def find_num_cam(self):
        num_cam = len(glob.glob(f"{self.raw_color_dir}/*.mp4"))
        assert len(glob.glob(f"{self.raw_color_dir}/*/")) == num_cam
        assert len(glob.glob(f"{self.raw_depth_dir}/*/")) == num_cam
        assert len(glob.glob(f"{self.base_mask_dir}/mask_info_*.json")) in [num_cam,0]
        assert len(glob.glob(f"{self.base_mask_dir}/*/")) in [num_cam,0]
        assert len(glob.glob(f"{self.base_cotracker_dir}/*.npz")) in [num_cam,0]
        return num_cam

    def find_num_frame(self):
        num_frame = len(glob.glob(f"{self.raw_color_dir}/0/*.png"))
        assert len(glob.glob(f"{self.raw_depth_dir}/0/*.npy")) == num_frame
        assert len(glob.glob(f"{self.base_mask_dir}/0/0/*.png")) == num_frame
        assert len(glob.glob(f"{self.base_pcd_dir}/*.npz")) == num_frame
        return num_frame

    def find_object_idx(self, camera_idx:int):
        # Get the mask index of the object
        with open(self.get_mask_info_path(camera_idx), "r") as f:
            data = json.load(f)
        obj_idx = None
        for key, value in data.items():
            if value != self.controller_name:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        return obj_idx

    def get_color_frame_path(self, camera_idx:int, frame_idx:int):
        return f"{self.raw_color_dir}/{camera_idx}/{frame_idx}.png"

    def get_color_video_path(self, camera_idx:int):
        return f"{self.raw_color_dir}/{camera_idx}.mp4"

    def get_depth_frame_path(self, camera_idx:int, frame_idx:int):
        return f"{self.raw_depth_dir}/{camera_idx}/{frame_idx}.npy"

    def get_mask_info_path(self, camera_idx:int):
        return f"{self.base_mask_dir}/mask_info_{camera_idx}.json"

    def get_mask_frame_path(self, camera_idx:int, obj_idx:int, frame_idx:int):
        return f"{self.base_mask_dir}/{camera_idx}/{obj_idx}/{frame_idx}.png"

    def get_object_mask_frame_path(self, camera_idx:int, frame_idx:int):
        # 画像中にobjectは１つであることを仮定します
        obj_idx = self.find_object_idx(camera_idx)
        return f"{self.base_mask_dir}/{camera_idx}/{obj_idx}/{frame_idx}.png"

    def list_first_frame_mask_of_all_objects(self, camera_idx:int):
        return glob.glob(f"{self.base_mask_dir}/{camera_idx}/*/0.png")

    def get_temp_video_frame_dir(self, camera_idx:int):
        return f"{self.base_path}/{self.case_name}/tmp_data_{camera_idx}"

    def get_tracking_data_path(self, camera_idx:int):
        return f"{self.base_cotracker_dir}/{camera_idx}.npz"

    def get_pcd_data_path(self, frame_idx:int):
        return f"{self.base_pcd_dir}/{frame_idx}.npz"
    