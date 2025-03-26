from pathlib import Path
import glob
import json
import numpy as np
import cv2

base_path = Path("./data/different_types")
prediction_dir = Path("./gaussian_output_dynamic")
human_mask_path = Path("./data/different_types_human_mask")
object_mask_path = Path("./data/render_eval_data")

height, width = 480, 848
FPS = 30
alpha = 0.7

dir_paths = list(base_path.glob("*"))
for dir_path in dir_paths:
    case_name = dir_path.name
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with (dir_path / "split.json").open("r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]

    # Need to prepare the video
    for i in range(3):
        # Process each camera
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(
            str(prediction_dir / case_name / f"{i}_integrate.mp4"),
            fourcc,
            FPS,
            (width, height),
        )

        for frame_idx in range(frame_len):
            render_path = prediction_dir / case_name / str(i) / f"{frame_idx:05d}.png"
            origin_image_path = dir_path / "color" / str(i) / f"{frame_idx}.png"
            human_mask_image_path = human_mask_path / case_name / "mask" / str(i) / "0" / f"{frame_idx}.png"
            object_image_path = object_mask_path / case_name / "mask" / str(i) / f"{frame_idx}.png"

            print(render_path)
            render_img = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
            origin_img = cv2.imread(str(origin_image_path))
            human_mask = cv2.imread(str(human_mask_image_path))
            human_mask = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY)
            human_mask = human_mask > 0
            object_mask = cv2.imread(str(object_image_path))
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
