import os
from argparse import ArgumentParser
import json

CONTROLLER_NAME = "hand"

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    eval_path = args.eval_path
    case_name = args.case_name

    output_dir = f"{eval_path}/{case_name}/render_eval_data"

    if not os.path.exists(f"{base_path}/{case_name}"):
        raise Exception(f"{case_name} not found")

    # Create the directory for the case
    existDir(f"{output_dir}/mask")
    for i in range(3):
        # Copy the original RGB image
        os.system(
            f"cp -r {base_path}/{case_name}/color {output_dir}/"
        )
        # Copy only the object mask image
        # Get the mask path for the image
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        existDir(f"{output_dir}/mask/{i}")
        os.system(f"cp -r {base_path}/{case_name}/mask/{i}/{obj_idx}/* {output_dir}/mask/{i}/")

    # Copy the split.json
    os.system(f"cp {base_path}/{case_name}/split.json {output_dir}/")