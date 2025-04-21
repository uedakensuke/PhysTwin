import os
import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict


DIR = os.path.dirname(__file__)

"""
Hyperparam for Ground and Tracking
"""
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class SegmentProcessor:

    def __init__(self, raw_path:str, base_path:str, case_name:str):
        self.raw_path = raw_path
        self.base_path = base_path
        self.case_name = case_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ann_frame_idx = 0  # the frame index we interact with

        # build grounding dino model from local path
        self.grounding_model = load_model(
            model_config_path= f"{DIR}/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path=f"{DIR}/groundedSAM_checkpoints/groundingdino_swint_ogc.pth",
            device=self.device,
        )

        # init sam image predictor and video predictor model
        sam2_checkpoint = f"{DIR}//groundedSAM_checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

    def process(self, camera_idx:int, text_prompt:str):
        source_video_frame_dir = f"{self.base_path}/{self.case_name}/tmp_data/{self.case_name}/{camera_idx}"
        output_path=f"{self.base_path}/{self.case_name}/mask"

        """
        Step 1: Environment settings
        """
        frame_names =self._extruct_frames(
            f"{self.raw_path}/{self.case_name}/color/{camera_idx}.mp4",
            source_video_frame_dir
        )
        img_path = os.path.join(source_video_frame_dir, frame_names[self.ann_frame_idx])

        """
        Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
        """
        # prompt grounding dino to get the box coordinates on specific frame
        object_ids, input_boxes, image_source = self._predict_boxes(
            img_path,
            text_prompt,
            f"{output_path}/mask_info_{camera_idx}.json"
        )

        # self._predict_masks(image_source, input_boxes)

        """
        Step 3: Register each object's positive points to video predictor with seperate add_new_points call
        """
        inference_state = self._prepare_video_predictor(source_video_frame_dir, object_ids, input_boxes)

        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        Step 5: Visualize the segment results across the video and save them
        """
        self._segment_video(
            inference_state,
            f"{output_path}/{camera_idx}",
        )

        os.system(f"rm -rf {source_video_frame_dir}")

    def _extruct_frames(self, video_path:str, source_video_frame_dir:str):
        existDir(source_video_frame_dir)

        video_info = sv.VideoInfo.from_video_path(video_path)  # get video info
        print(video_info)
        frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)

        # saving video to frames
        source_frames = Path(source_video_frame_dir)
        source_frames.mkdir(parents=True, exist_ok=True)

        with sv.ImageSink(
            target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)

        # scan all the JPEG frame names in this directory
        frame_names = [
            p
            for p in os.listdir(source_video_frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        return frame_names

    def _predict_boxes(self, img_path:str, text_prompt:str, mask_info_path:str):
        image_source, image = load_image(img_path)

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
        confidences = confidences.numpy().tolist()

        print(input_boxes)
        print(labels)

        # Save the id_to_objects into json
        existDir(os.path.dirname(mask_info_path))
        with open(mask_info_path, "w") as f:
            json.dump({i: obj for i, obj in enumerate(labels)}, f)

        return labels, input_boxes, image_source

    def _predict_masks(self, image_source, input_boxes:str):
        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(image_source)

        # FIXME: figure how does this influence the G-DINO model
        # commnet out by UEAD. this cause error
        # torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

    def _prepare_video_predictor(self, source_video_frame_dir, object_ids, input_boxes):
        # init video predictor state
        inference_state = self.video_predictor.init_state(
            video_path=source_video_frame_dir
        )

        assert PROMPT_TYPE_FOR_VIDEO in [
            "point",
            "box",
            "mask",
        ], "SAM 2 video predictor only support point/box/mask prompt"

        if PROMPT_TYPE_FOR_VIDEO == "box":
            for object_id, (label, box) in enumerate(zip(object_ids, input_boxes)):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=self.ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        else:
            raise NotImplementedError(
                "SAM 2 video predictor only support point/box/mask prompts"
            )
        return inference_state

    def _segment_video(self, inference_state, output_path):
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        for frame_idx, masks in video_segments.items():
            for obj_id, mask in masks.items():
                mask_frame_dir = f"{output_path}/{obj_id}"
                existDir(mask_frame_dir)
                # mask is 1 * H * W
                Image.fromarray((mask[0] * 255).astype(np.uint8)).save(
                    f"{mask_frame_dir}/{frame_idx}.png"
                )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--TEXT_PROMPT", type=str, required=True)
    parser.add_argument("--camera_idx", type=int, required=True)
    args = parser.parse_args()

    sp = SegmentProcessor(args.raw_path, args.base_path, args.case_name, args.camera_idx, args.TEXT_PROMPT)
    sp.process()