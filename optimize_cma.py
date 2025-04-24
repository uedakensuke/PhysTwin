# The first stage to optimize the sparse parameters using CMA-ES
import os
import random
import sys
import json
from argparse import ArgumentParser

from qqtt import OptimizerCMA
from qqtt.utils import logger, cfg
from qqtt.utils.logger import StreamToLogger, logging
import numpy as np
import torch

from data_process.utils.path import PathResolver
from data_process.utils.data_reader import CameraInfo

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

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, default=None)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--physics_sparse_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--max_iter", type=int, default=20)
    args = parser.parse_args()

    if args.raw_path is None:
        args.raw_path = args.base_path

    base_path = args.base_path
    physics_sparse_path = args.physics_sparse_path
    case_name = args.case_name
    train_frame = get_train_frame(base_path, case_name)
    max_iter = args.max_iter

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml(f"{DIR}/configs/cloth.yaml")
    else:
        cfg.load_from_yaml(f"{DIR}/configs/real.yaml")

    base_dir = f"{physics_sparse_path}/{case_name}"

    # Set the intrinsic and extrinsic parameters for visualization
    path_resolver = PathResolver(args.raw_path, args.base_path, args.case_name)
    camera_info = CameraInfo(path_resolver)
    cfg.c2ws = camera_info.c2ws
    cfg.w2cs = camera_info.w2cs
    cfg.intrinsics = camera_info.intrinsics
    cfg.WH = camera_info.WH
    cfg.overlay_path = path_resolver.get_color_dir()


    logger.set_log_file(path=base_dir, name="optimize_cma_log")
    optimizer = OptimizerCMA(
        data_path=path_resolver.final_data_pkl,
        base_dir=base_dir,
        train_frame=train_frame,
    )
    optimizer.optimize(max_iter=max_iter)
