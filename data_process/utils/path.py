import glob
import json


class PathResolver:
    def __init__(self, raw_path:str, base_path:str, case_name:str, *, controller_name="hand"):
        self.raw_path = raw_path
        self.base_path = base_path
        self.case_name = case_name
        self.controller_name = controller_name

        self.raw_color_dir=f"{self.raw_path}/{self.case_name}/color"
        self.raw_depth_dir=f"{self.raw_path}/{self.case_name}/depth"
        self.raw_camera_meta=f"{self.raw_path}/{self.case_name}/metadata.json"
        self.raw_camera_calibrate=f"{self.raw_path}/{self.case_name}/calibrate.pkl"

        self.base_mask_dir=f"{self.base_path}/{self.case_name}/mask"

        self.base_shape_dir = f"{self.base_path}/{self.case_name}/shape"
        self.upscale_image_path = f"{self.base_shape_dir}/high_resolution.png"
        self.masked_upscale_image_path = f"{self.base_shape_dir}/masked_image.png"
        self.reconstruct_3d_model_glb = f"{self.base_shape_dir}/object.glb"
        self.reconstruct_3d_model_ply = f"{self.base_shape_dir}/object.ply"
        self.reconstruct_3d_model_video = f"{self.base_shape_dir}/visualization.mp4"

        self.base_cotracker_dir = f"{self.base_path}/{self.case_name}/cotracker"

        self.base_pcd_dir = f"{self.base_path}/{self.case_name}/pcd"

    def assert_num_cam(self, num_cam:int):
        assert len(glob.glob(f"{self.raw_color_dir}/*.mp4")) == num_cam
        assert len(glob.glob(f"{self.raw_color_dir}/*/")) == num_cam
        assert len(glob.glob(f"{self.raw_depth_dir}/*/")) == num_cam

    def get_color_frame_path(self, camera_idx:int, frame_idx:int):
        return f"{self.raw_color_dir}/{camera_idx}/{frame_idx}.png"

    def get_color_video_path(self, camera_idx:int):
        return f"{self.raw_color_dir}/{camera_idx}.mp4"

    def get_depth_frame_path(self, camera_idx:int, frame_idx:int):
        return f"{self.raw_depth_dir}/{camera_idx}/{frame_idx}.npy"

    def get_mask_info_path(self, camera_idx:int):
        return f"{self.base_mask_dir}/mask_info_{camera_idx}.json"

    def get_mask_frame_dir(self, camera_idx:int, obj_idx:int):
        return f"{self.base_mask_dir}/{camera_idx}/{obj_idx}"

    def get_object_mask_frame_path(self, camera_idx:int, frame_idx:int):
        # 画像中にobjectは１つであることを仮定します

        # Get the mask path for the image
        with open(self.get_mask_info_path(0), "r") as f:
            data = json.load(f)
        obj_idx = None
        for key, value in data.items():
            if value != self.controller_name:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        return f"{self.base_mask_dir}/{camera_idx}/{obj_idx}/{frame_idx}.png"

    def get_temp_video_frame_dir(self, camera_idx:int):
        return f"{self.base_path}/{self.case_name}/tmp_data_{camera_idx}"

    def list_first_frame_mask_of_all_objects(self, camera_idx:int):
        return glob.glob(f"{self.base_mask_dir}/{camera_idx}/*/0.png")

    def get_tracking_data_path(self, camera_idx:int):
        return f"{self.base_cotracker_dir}/{camera_idx}.npz"

    def get_pcd_data_path(self, frame_idx:int):
        return f"{self.base_pcd_dir}/{frame_idx}.npz"
