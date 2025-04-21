from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from argparse import ArgumentParser
import cv2
import numpy as np

class Upscaler:
    def __init__(self, category:str, *, model_id = "stabilityai/stable-diffusion-x4-upscaler"):
        self.category = category
        # load model and scheduler
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")

    def process(self, img_path:str, mask_path:str, output_path:str):
        # let's download an  image
        low_res_img = Image.open(img_path).convert("RGB")
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            bbox = np.argwhere(mask > 0.8 * 255)
            bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
            center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            size = int(size * 1.2)
            bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
            low_res_img = low_res_img.crop(bbox)  # type: ignore

        upscaled_image = self.pipeline(
            prompt=f"Hand manipulates a {self.category}.",
            image=low_res_img
        ).images[0]
        upscaled_image.save(output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    us=Upscaler(args.category)
    us.process(args.img_path, args.mask_path, args.output_path)
