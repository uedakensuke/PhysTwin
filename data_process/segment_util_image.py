import os
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict

from .utils.path import PathResolver

DIR = os.path.dirname(__file__)

"""
Hyper parameters
"""
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

class ImageSegmentProcessor:
    def __init__(self, raw_path:str, base_path:str , case_name:str):
        self.path = PathResolver(raw_path, base_path, case_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # build SAM2 image predictor
        sam2_checkpoint = f"{DIR}/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=f"{DIR}/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path=f"{DIR}/groundedSAM_checkpoints/groundingdino_swint_ogc.pth",
            device=device,
        )

        # FIXME: figure how does this influence the G-DINO model
        # comment out. it causes error
        # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def process(self, text_prompt:str):
        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot

        image_source, image = load_image(self.path.upscale_image)

        self.sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()

        print(f"Detected {len(masks)} objects")

        raw_img = cv2.imread(self.path.upscale_image)
        mask_img = (masks[0] * 255).astype(np.uint8)

        ref_img = np.zeros((h, w, 4), dtype=np.uint8)
        mask_bool = mask_img > 0
        ref_img[mask_bool, :3] = raw_img[mask_bool]
        ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255
        cv2.imwrite(self.path.masked_upscale_image, ref_img)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)
    args = parser.parse_args()

    isp = ImageSegmentProcessor(args.raw_path, args.base_path, args.case_name, args.case_name)
    isp.process(args.text_prompt)
