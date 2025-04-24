import os

from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from argparse import ArgumentParser

from .utils.path import PathResolver
from .utils.data_reader import ImageReader

class ImageUpscaleProcessor:
    def __init__(self, raw_path:str, base_path:str , case_name:str, *, controller_name="hand", model_id = "stabilityai/stable-diffusion-x4-upscaler"):
        self.path = PathResolver(raw_path, base_path, case_name, controller_name=controller_name)
        self.data = ImageReader(self.path)

        # load model and scheduler
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")

    def process(self, camera_idx:int, category:str):
        output_path = self.path.upscale_image

        if os.path.isfile(output_path):
            return False # already exists

        low_res_img = Image.open(self.path.get_color_frame_path(camera_idx,0)).convert("RGB")
        bbox_margin, mask_img = self.data.read_object_bbox_and_mask(camera_idx,0)

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
    parser.add_argument("--controller_name", type=str, default="hand")
    parser.add_argument("--camera_idx", type=int, required=True)
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    iup=ImageUpscaleProcessor(args.raw_path, args.base_path, args.case_name, controller_name=args.controller_name)
    iup.process(args.camera_idx, args.category)
