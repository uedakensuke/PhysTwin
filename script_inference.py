import glob
import os
import json

DATA_DIR = "../data"

base_path = f"{DATA_DIR}/data/different_types"
dir_names = glob.glob(f"{DATA_DIR}/experiments/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    os.system(
        f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
    )
