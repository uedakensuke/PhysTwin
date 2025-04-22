import glob

class PathResolver:
    def __init__(self, raw_path:str, base_path:str, case_name:str, num_cam = 3):
        self.raw_path = raw_path
        self.base_path = base_path
        self.case_name = case_name
        self.num_cam = num_cam

        self.raw_color_dir=f"{self.raw_path}/{self.case_name}/color"
        self.raw_depth_dir=f"{self.raw_path}/{self.case_name}/depth"

        assert len(glob.glob(f"{self.raw_depth_dir}/*")) == num_cam

        self.base_mask_dir=f"{self.base_path}/{self.case_name}/mask"
        self.base_cotracker_dir = f"{self.base_path}/{self.case_name}/cotracker"

    def get_color_video_path(self, camera_idx:int):
        return f"{self.raw_color_dir}/{camera_idx}.mp4"

    def get_tracking_data_path(self, camera_idx:int):
        return f"{self.base_cotracker_dir}/{camera_idx}.npz"

    def get_mask_info_path(self, camera_idx:int):
        return f"{self.base_mask_dir}/mask_info_{camera_idx}.json"

    def get_mask_frame_dir(self, camera_idx:int, obj_id):
        return f"{self.base_mask_dir}/{camera_idx}/{obj_id}"
    
    def get_temp_video_frame_dir(self, camera_idx:int):
        return f"{self.base_path}/{self.case_name}/tmp_data_{camera_idx}"

    def list_first_frame_mask_of_all_objects(self, camera_id:int):
        return glob.glob(f"{self.base_mask_dir}/{camera_id}/*/0.png")        