import random
from argparse import ArgumentParser
import os
import pickle
import json

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
import numpy as np
import torch

from .data_process.utils.path import PathResolver
from .data_process.utils.data_reader import CameraInfo

DIR = os.path.dirname(__file__)

def get_train_frame(base_path,case_name):
    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]
    return train_frame

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
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, default=None)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--physics_sparse_path", type=str, required=True)
    parser.add_argument("--physics_dense_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    if args.raw_path is None:
        args.raw_path = args.base_path

    base_path = args.base_path
    physics_sparse_path = args.physics_sparse_path
    physics_dense_path = args.physics_dense_path
    case_name = args.case_name
    train_frame = get_train_frame(base_path, case_name)

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml(f"{DIR}/configs/cloth.yaml")
    else:
        cfg.load_from_yaml(f"{DIR}/configs/real.yaml")

    print(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"{physics_dense_path}/{case_name}"

    # Read the first-satage optimized parameters
    optimal_path = f"{physics_sparse_path}/{case_name}/optimal_params.pkl"
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

    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=path_resolver.final_data_pkl,
        base_dir=base_dir,
        train_frame=train_frame,
    )
    trainer.train()
