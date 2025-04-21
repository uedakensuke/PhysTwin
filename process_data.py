import os
from argparse import ArgumentParser
import time
import logging
import json
import glob

from data_process.segment_util_video import SegmentProcessor

CONTROLLER_NAME = "hand"


def setup_logger(log_file):
    logger = logging.getLogger("GlobalLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Timer:
    def __init__(self, logger, task_name):
        self.task_name = task_name
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {self.case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )

class DataProcessor:
    def __init__(self, raw_path:str, base_path:str, case_name:str):
        self.raw_path = raw_path
        self.base_path = base_path
        self.case_name = case_name
        self.logger = setup_logger(f"{base_path}/{case_name}/timer.log")
        self.category, self.use_shape_prior = _read_config(base_path, case_name)

    def process(self):
        self._process_seg()
        # if self.use_shape_prior:
        #     self._process_shape_prior()
        # self._process_track()
        # self._process_3d()
        # if self.use_shape_prior:
        #     self._process_align()
        # self._process_final()

    def _process_seg(self, camera_num = 3):
        text_prompt = f"{self.category}.{CONTROLLER_NAME}"
        assert len(glob.glob(f"{self.raw_path}/{self.case_name}/depth/*")) == camera_num
        
        # Get the masks of the controller and the object using GroundedSAM2
        with Timer("Video Segmentation"):
            for camera_idx in range(camera_num):
                print(f"Processing {self.case_name} camera {camera_idx}")
                sp = SegmentProcessor(self.raw_path,self.base_path,self.case_name,camera_idx,text_prompt)
                sp.process()

    def _process_shape_prior(self):
        # Get the mask path for the image
        with open(f"{self.base_path}/{self.case_name}/mask/mask_info_{0}.json", "r") as f:
            data = json.load(f)
        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        mask_path = f"{self.base_path}/{self.case_name}/mask/0/{obj_idx}/0.png"

        existDir(f"{self.base_path}/{self.case_name}/shape")
        # Get the high-resolution of the image to prepare for the trellis generation
        with Timer("Image Upscale"):
            if not os.path.isfile(f"{self.base_path}/{self.case_name}/shape/high_resolution.png"):
                os.system(
                    f"python ./data_process/image_upscale.py --img_path {self.base_path}/{self.case_name}/color/0/0.png --mask_path {mask_path} --output_path {self.base_path}/{self.case_name}/shape/high_resolution.png --category {category}"
                )

        # Get the masked image of the object
        with Timer("Image Segmentation"):
            os.system(
                f"python ./data_process/segment_util_image.py --img_path {self.base_path}/{self.case_name}/shape/high_resolution.png --TEXT_PROMPT {self.category} --output_path {self.base_path}/{self.case_name}/shape/masked_image.png"
            )

        with Timer("Shape Prior Generation"):
            os.system(
                f"python ./data_process/shape_prior.py --img_path {self.base_path}/{self.case_name}/shape/masked_image.png --output_dir {self.base_path}/{self.case_name}/shape"
            )

    def _process_track(self):
        # Get the dense tracking of the object using Co-tracker
        with Timer("Dense Tracking"):
            os.system(
                f"python ./data_process/dense_track.py --base_path {self.base_path} --case_name {self.case_name}"
            )

    def _process_3d(self):
        # Get the pcd in the world coordinate from the raw observations
        with Timer("Lift to 3D"):
            os.system(
                f"python ./data_process/data_process_pcd.py --base_path {self.base_path} --case_name {self.case_name}"
            )

        # Further process and filter the noise of object and controller masks
        with Timer("Mask Post-Processing"):
            os.system(
                f"python ./data_process/data_process_mask.py --base_path {self.base_path} --case_name {self.case_name} --controller_name {CONTROLLER_NAME}"
            )

        # Process the data tracking
        with Timer("Data Tracking"):
            os.system(
                f"python ./data_process/data_process_track.py --base_path {self.base_path} --case_name {self.case_name}"
            )

    def _process_align(self):
        # Align the shape prior with partial observation
        with Timer("Alignment"):
            os.system(
                f"python ./data_process/align.py --base_path {self.base_path} --case_name {self.case_name} --controller_name {CONTROLLER_NAME}"
            )

    def _process_final(self):
        # Get the final PCD used for the inverse physics with/without the shape prior
        with Timer("Final Data Generation"):
            if self.use_shape_prior:
                os.system(
                    f"python ./data_process/data_process_sample.py --base_path {self.base_path} --case_name {self.case_name} --shape_prior"
                )
            else:
                os.system(
                    f"python ./data_process/data_process_sample.py --base_path {self.base_path} --case_name {self.case_name}"
                )

        # Save the train test split
        frame_len = len(glob.glob(f"{self.base_path}/{self.case_name}/pcd/*.npz"))
        split = {}
        split["frame_len"] = frame_len
        split["train"] = [0, int(frame_len * 0.7)]
        split["test"] = [int(frame_len * 0.7), frame_len]
        with open(f"{self.base_path}/{self.case_name}/split.json", "w") as f:
            json.dump(split, f)

def _read_config(raw_path:str, case_name:str):
    with open(f"{raw_path}/{case_name}/data_config.csv", newline="", encoding="utf-8") as f:
        line = f.readline()
    category = line[0]
    use_shape_prior = line[1]
    return category, use_shape_prior

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    dp = DataProcessor(args.raw_path, args.base_path, args.case_name)
    dp.process()
