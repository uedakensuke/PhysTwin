import json
from argparse import ArgumentParser

import numpy as np
import cv2

from data_process.utils.path import PathResolver

height, width = 480, 848
FPS = 30
alpha = 0.7

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, default=None)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--human_mask_path", type=str, required=True)
    parser.add_argument("--inference_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    if args.raw_path is None:
        args.raw_path = args.base_path

    human_mask_path = args.human_mask_path
    inference_path = args.inference_path
    eval_path = args.eval_path
    case_name = args.case_name

    # Set the intrinsic and extrinsic parameters for visualization
    path_resolver = PathResolver(args.raw_path, args.base_path, args.case_name)

    dynamic_scene_dir=f"{inference_path}/{case_name}/dynamic" #gaussian_output_dynamicから変更
    object_mask_dir = f"{eval_path}/{case_name}/render_eval_data"

    with open(path_resolver.get_split_json_path(), "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]

    # Need to prepare the video
    for i in range(3):
        # Process each camera
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(
            f"{dynamic_scene_dir}/{i}_integrate.mp4",
            fourcc,
            FPS,
            (width, height),
        )

        for frame_idx in range(frame_len):
            render_path = f"{dynamic_scene_dir}/{i}/{frame_idx:05d}.png"
            origin_image_path = path_resolver.get_color_frame_path(i,frame_idx)
            human_mask_image_path = (
                f"{human_mask_path}/{case_name}/mask/{i}/0/{frame_idx}.png"
            )
            object_image_path = (
                f"{object_mask_dir}/mask/{i}/{frame_idx}.png"
            )

            render_img = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
            origin_img = cv2.imread(origin_image_path)
            human_mask = cv2.imread(human_mask_image_path)
            human_mask = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY)
            human_mask = human_mask > 0
            object_mask = cv2.imread(object_image_path)
            object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
            object_mask = object_mask > 0

            final_image = origin_img.copy()
            render_mask = np.logical_and(
                (render_img != 0).any(axis=2), render_img[:, :, 3] > 100
            )
            render_img[~render_mask, 3] = 0

            final_image[:, :, :] = alpha * final_image + (1 - alpha) * np.array(
                [255, 255, 255], dtype=np.uint8
            )

            test_alpha = render_img[:, :, 3] / 255
            final_image[:, :, :] = render_img[:, :, :3] * test_alpha[
                :, :, None
            ] + final_image * (1 - test_alpha[:, :, None])

            final_image[human_mask] = alpha * origin_img[human_mask] + (
                1 - alpha
            ) * np.array([255, 255, 255], dtype=np.uint8)

            video_writer.write(final_image)

        video_writer.release()
