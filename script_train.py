import glob
import os
import json

DIR = os.path.dirname(__file__)
WORKSPACE_DIR = f"{DIR}/../mount/ws"

base_path = f"{WORKSPACE_DIR}/data/different_types"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    os.system(
        f"python train_warp.py --base_path {base_path} --case_name {case_name}"
    )
