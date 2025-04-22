import os
from PIL import Image
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import json
from argparse import ArgumentParser
from tqdm import tqdm
import torch
# import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import numpy as np


def img2tensor(img):
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0

def eval(dynamic_scene_dir:str, render_path_dir:str, human_mask_dir:str):
    # Load frame split info
    with open(f"{render_path_dir}/split.json", 'r') as f:
        info = json.load(f)
    frame_len = info['frame_len']
    train_f_idx_range = list(range(info["train"][0] + 1, info["train"][1]))   # +1 if ignoring the first frame
    test_f_idx_range = list(range(info["test"][0], info["test"][1]))

    print("train indices range from", train_f_idx_range[0], "to", train_f_idx_range[-1])
    print("test indices range from", test_f_idx_range[0], "to", test_f_idx_range[-1])

    psnrs_train, ssims_train, lpipss_train, ious_train = [], [], [], []
    psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

    # for view_idx in range(3):
    for view_idx in range(1):   # only consider the first view

        for frame_idx in train_f_idx_range:
            gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
            gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
            gt_mask = gt_mask.astype(np.float32) / 255.

            render = np.array(Image.open(os.path.join(dynamic_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
            render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

            human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))
            inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

            gt = gt.astype(np.float32) * gt_mask[..., None]
            bg_mask = gt_mask == 0
            gt[bg_mask] = [255, 255, 255]
            render = render[:, :, :3].astype(np.float32)

            gt = gt * inv_human_mask[..., None]
            render = render * inv_human_mask[..., None]
            render_mask = render_mask * inv_human_mask

            gt_tensor = img2tensor(gt)
            render_tensor = img2tensor(render)

            psnrs_train.append(psnr(render_tensor, gt_tensor).item())
            ssims_train.append(ssim(render_tensor, gt_tensor).item())
            lpipss_train.append(lpips(render_tensor, gt_tensor).item())
            ious_train.append(compute_iou(gt_mask > 0, render_mask > 0))

        for frame_idx in test_f_idx_range:
                
            gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
            gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
            gt_mask = gt_mask.astype(np.float32) / 255.

            render = np.array(Image.open(os.path.join(dynamic_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
            render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

            human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))
            inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

            gt = gt.astype(np.float32) * gt_mask[..., None]
            bg_mask = gt_mask == 0
            gt[bg_mask] = [255, 255, 255]
            render = render[:, :, :3].astype(np.float32)

            gt = gt * inv_human_mask[..., None]
            render = render * inv_human_mask[..., None]
            render_mask = render_mask * inv_human_mask

            gt_tensor = img2tensor(gt)
            render_tensor = img2tensor(render)

            psnrs_test.append(psnr(render_tensor, gt_tensor).item())
            ssims_test.append(ssim(render_tensor, gt_tensor).item())
            lpipss_test.append(lpips(render_tensor, gt_tensor).item())
            ious_test.append(compute_iou(gt_mask > 0, render_mask > 0))

    print(f'\t PSNR (train): {np.mean(psnrs_train):.4f}')
    print(f'\t SSIM (train): {np.mean(ssims_train):.4f}')
    print(f'\t LPIPS (train): {np.mean(lpipss_train):.4f}')
    print(f'\t IoU (train): {np.mean(ious_train):.4f}')

    print(f'\t PSNR (test): {np.mean(psnrs_test):.4f}')
    print(f'\t SSIM (test): {np.mean(ssims_test):.4f}')
    print(f'\t LPIPS (test): {np.mean(lpipss_test):.4f}')
    print(f'\t IoU (test): {np.mean(ious_test):.4f}')

    return {
        'psnr_train': np.mean(psnrs_train),
        'ssim_train': np.mean(ssims_train),
        'lpips_train': np.mean(lpipss_train),
        'iou_train': np.mean(ious_train),
        'psnr_test': np.mean(psnrs_test),
        'ssim_test': np.mean(ssims_test),
        'lpips_test': np.mean(lpipss_test),
        'iou_test': np.mean(ious_test)
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--human_mask_path", type=str, required=True)
    parser.add_argument("--inference_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    human_mask_path = args.human_mask_path
    inference_path = args.inference_path
    eval_path = args.eval_path
    case_name = args.case_name

    result = eval(
        f'{inference_path}/{case_name}/dynamic', # gaussian_output_dynamicから変更
        f'{eval_path}/{case_name}/render_eval_data',
        f'{human_mask_path}/{case_name}'
    )
    result["case_name"] = case_name

    output_dir = f"{eval_path}/{case_name}/results"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/output_dynamic.txt", 'w') as log_file:
        json.dump(result, log_file, indent=2)
