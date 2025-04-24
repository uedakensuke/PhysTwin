from argparse import ArgumentParser
import glob
import os
import pickle
import random

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
import numpy as np
import torch

from data_process.utils.path import PathResolver
from data_process.utils.data_reader import CameraInfo

DIR = os.path.dirname(__file__)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    cfg.load_from_yaml(f"{DIR}/configs/real.yaml")

    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, default=None)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--physics_sparse_path", type=str, required=True)
    parser.add_argument("--physics_dense_path", type=str, required=True)
    parser.add_argument("--gaussian_path", type=str, required=True)
    parser.add_argument("--inference_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    args = parser.parse_args()

    if args.raw_path is None:
        args.raw_path = args.base_path

    base_path = args.base_path
    physics_sparse_path = args.physics_sparse_path
    physics_dense_path = args.physics_dense_path
    inference_path = args.inference_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml(f"{DIR}/configs/cloth.yaml")
    else:
        cfg.load_from_yaml(f"{DIR}/configs/real.yaml")

    out_dir = f"{inference_path}/{case_name}/force"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"{physics_sparse_path}/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    path_resolver = PathResolver(args.raw_path, args.base_path, args.case_name)
    camera_info = CameraInfo(path_resolver)
    cfg.c2ws = camera_info.c2ws
    cfg.w2cs = camera_info.w2cs
    cfg.intrinsics = camera_info.intrinsics
    cfg.WH = camera_info.WH
    cfg.overlay_path = path_resolver.get_color_dir()

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=out_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=path_resolver.final_data_pkl,
        base_dir=out_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"{physics_dense_path}/{case_name}/train/best_*.pth")[0]
    trainer.visualize_force(
        best_model_path, gaussians_path, args.n_ctrl_parts
    )