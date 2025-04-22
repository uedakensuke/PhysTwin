# Use co-tracker to track the ibject and controller in the video (pick 5000 pixels in the masked area)
import os
from argparse import ArgumentParser

import torch
import imageio.v3 as iio
import cv2
import numpy as np

from utils.visualizer import Visualizer
from utils.path import PathResolver

def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class TrackProcessor:
    def __init__(self, raw_path:str, base_path:str, case_name:str, num_cam = 3):
        self.path = PathResolver(raw_path,base_path,case_name,num_cam)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process(self):
        exist_dir(self.path.base_cotracker_dir)

        for i in range(self.num_cam):
            print(f"Processing {i}th camera")
            # Load the video
            frames = iio.imread(self.path.get_color_video_path(i), plugin="FFMPEG")
            video = (
                torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.device)
            )  # B T C H W
            # Load the first-frame mask to get all query points from all masks
            mask = None
            for mask_path in self.path.list_first_frame_mask_of_all_objects(i):
                current_mask = read_mask(mask_path)
                if mask is None:
                    mask = current_mask
                else:
                    mask = np.logical_or(mask, current_mask)

            # Draw the mask
            query_pixels = np.argwhere(mask)
            # Revert x and y
            query_pixels = query_pixels[:, ::-1]
            query_pixels = np.concatenate(
                [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
            )
            query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(self.device)
            # Randomly select 5000 query points
            query_pixels = query_pixels[torch.randperm(query_pixels.shape[0])[:5000]]

            # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
            # pred_tracks, pred_visibility = cotracker(video, queries=query_pixels[None], backward_tracking=True)
            # pred_tracks, pred_visibility = cotracker(video, grid_query_frame=0)

            # # Run Online CoTracker:
            cotracker = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_online"
            ).to(self.device)
            cotracker(video_chunk=video, is_first_step=True, queries=query_pixels[None])

            # Process the video
            for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
                pred_tracks, pred_visibility = cotracker(
                    video_chunk=video[:, ind : ind + cotracker.step * 2]
                )  # B T N 2,  B T N 1
            vis = Visualizer(
                save_dir=self.path.base_cotracker_dir, pad_value=0, linewidth=3
            )
            vis.visualize(video, pred_tracks, pred_visibility, filename=f"{i}")
            # Save the tracking data into npz
            track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]
            visibility_to_save = pred_visibility[0].cpu().numpy()
            np.savez(
                self.path.get_tracking_data_path(i),
                tracks=track_to_save,
                visibility=visibility_to_save,
            )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    tp = TrackProcessor(args.base_path, args.case_name)
    tp.process()
