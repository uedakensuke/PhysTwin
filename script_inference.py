import glob
import os
import json

WORKSPACE_DIR = "../mount/ws"

base_path = f"{WORKSPACE_DIR}/data/different_types"
dir_names = glob.glob(f"{WORKSPACE_DIR}/experiments/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    os.system(
        f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
    )
