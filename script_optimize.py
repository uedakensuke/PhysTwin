import glob
import os
import json

WORKSPACE_DIR = "../mount/ws"

base_path = f"{WORKSPACE_DIR}/data/different_types"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    
    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    os.system(
        f"python optimize_cma.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}"
    )