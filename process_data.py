import os
from argparse import ArgumentParser
import time
import logging
import json
import glob

from data_process.segment_util_video import VideoSegmentProcessor
from data_process.image_upscale import ImageUpscaleProcessor
from data_process.segment_util_image import ImageSegmentProcessor
from data_process.shape_prior import ShapePriorProcessor
from data_process.dense_track import VideoTrackProcessor
from data_process.data_process_pcd import PcdEstimateProcessor
from data_process.data_process_mask import PcdMaskProcessor
from data_process.data_process_track import PcdTrackProcessor
from data_process.align import AlignProcessor
from data_process.data_process_sample import SampleProcessor


CONTROLLER_NAME = "hand"

def setup_logger(log_file):
    logger = logging.getLogger("GlobalLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file),exist_ok=True)
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
    def __init__(self, logger, task_name, case_name):
        self.task_name = task_name
        self.case_name = case_name
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"[start   ] {self.task_name} for {self.case_name}"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.logger.info(
            f"[complete] {self.task_name} for {self.case_name}: {elapsed_time:.2f} sec"
        )

class DataProcessor:
    def __init__(self, raw_path:str, base_path:str, case_name:str, show_window=False):
        self.raw_path = raw_path
        self.base_path = base_path
        self.case_name = case_name
        self.show_window = show_window
        
        self.logger = setup_logger(f"{base_path}/{case_name}/timer.log")
        self.category, self.use_shape_prior = _read_config(raw_path, case_name)

    def process(self):
        self._process_seg()
        self._process_track()
        self._process_3d()
        if self.use_shape_prior:
            self._process_shape_prior() #実行には_process_segの実行が必要
            self._process_align() #実行には_process_3dの実行が必要
        self._process_final()

    def _process_seg(self, camera_num = 3):
        assert len(glob.glob(f"{self.raw_path}/{self.case_name}/depth/*")) == camera_num
        
        # Get the masks of the controller and the object using GroundedSAM2
        with Timer(self.logger,"Video Segmentation",self.case_name):
            for camera_idx in range(camera_num):
                print(f"Processing {self.case_name} camera {camera_idx}")
                vsp = VideoSegmentProcessor(
                    self.raw_path,
                    self.base_path,
                    self.case_name)
                vsp.process(camera_idx, f"{self.category}.{CONTROLLER_NAME}")

    def _process_shape_prior(self):
        # Get the high-resolution of the image to prepare for the trellis generation
        with Timer(self.logger,"Image Upscale",self.case_name):
            iup = ImageUpscaleProcessor(
                self.raw_path,
                self.base_path,
                self.case_name,
                controller_name=CONTROLLER_NAME)
            iup.process(0, self.category) # for camera 0

        # Get the masked image of the object
        with Timer(self.logger,"Image Segmentation",self.case_name):
            isp = ImageSegmentProcessor(
                self.raw_path,
                self.base_path,
                self.case_name)
            isp.process(self.category)

        with Timer(self.logger,"Shape Prior Generation",self.case_name):
            spp = ShapePriorProcessor(
                self.raw_path,
                self.base_path,
                self.case_name)
            spp.process()

    def _process_track(self):
        # Get the dense tracking of the object using Co-tracker
        with Timer(self.logger,"Dense Tracking",self.case_name):
            vtp = VideoTrackProcessor(
                self.raw_path,
                self.base_path,
                self.case_name)
            vtp.process()

    def _process_3d(self):
        # Get the pcd in the world coordinate from the raw observations
        with Timer(self.logger,"Lift to 3D",self.case_name):
            pep = PcdEstimateProcessor(
                self.raw_path,
                self.base_path,
                self.case_name,
                show_window=self.show_window)
            pep.process()

        # Further process and filter the noise of object and controller masks
        with Timer(self.logger,"Mask Post-Processing",self.case_name):
            pmp = PcdMaskProcessor(
                self.raw_path,
                self.base_path,
                self.case_name,
                controller_name=CONTROLLER_NAME,
                show_window=self.show_window)
            pmp.process()

        # Process the data tracking
        with Timer(self.logger,"Data Tracking",self.case_name):
            ptp = PcdTrackProcessor(
                self.raw_path,
                self.base_path,
                self.case_name,
                show_window=self.show_window)
            ptp.process()

    def _process_align(self):
        # Align the shape prior with partial observation
        with Timer(self.logger,"Alignment",self.case_name):
            ap = AlignProcessor(
                self.raw_path,
                self.base_path,
                self.case_name,
                controller_name=CONTROLLER_NAME,
                show_window=self.show_window)
            ap.process()

    def _process_final(self):
        # Get the final PCD used for the inverse physics with/without the shape prior
        with Timer(self.logger,"Final Data Generation",self.case_name):
            sp = SampleProcessor(
                self.raw_path,
                self.base_path,
                self.case_name,
                show_window=self.show_window,
                use_shape_prior=self.use_shape_prior
                )
            sp.process()

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
    category = line.split(",")[0]
    use_shape_prior = line.split(",")[1]
    return category, use_shape_prior

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--show_window", action='store_true' )
    args = parser.parse_args()

    dp = DataProcessor(args.raw_path, args.base_path, args.case_name, show_window=args.show_window)
    dp.process()
