import os
import json

from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from argparse import ArgumentParser
import cv2
import numpy as np

from .utils.path import PathResolver

class UpscaleProcessor:
    def __init__(self, raw_path:str, base_path:str , case_name:str, *, model_id = "stabilityai/stable-diffusion-x4-upscaler"):
        self.path = PathResolver(raw_path, base_path, case_name)

        # load model and scheduler
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")

    def process(self, camera_idx:int, category:str):
        output_path = self.path.upscale_image_path

        if os.path.isfile(output_path):
            return False # already exists

        low_res_img = Image.open(self.path.get_color_frame_path(camera_idx,0)).convert("RGB")
        mask = cv2.imread(self.path.get_object_mask_frame_path(camera_idx,0), cv2.IMREAD_GRAYSCALE)
        mask_binary = np.argwhere(mask > 0.8 * 255)
        bbox = (
            np.min(mask_binary[:, 1]),
            np.min(mask_binary[:, 0]),
            np.max(mask_binary[:, 1]),
            np.max(mask_binary[:, 0])
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size_margin = int(size * 1.2)
        bbox_margin = (
            center[0] - size_margin // 2,
            center[1] - size_margin // 2,
            center[0] + size_margin // 2,
            center[1] + size_margin // 2
        )

        upscaled_image = self.pipeline(
            prompt=f"Hand manipulates a {category}.",
            image=low_res_img.crop(bbox_margin)  # type: ignore
        ).images[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        upscaled_image.save(output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--camera_idx", type=int, required=True)
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    up=UpscaleProcessor(args.raw_path, args.base_path, args.case_name)
    up.process(args.camera_idx, args.category)
