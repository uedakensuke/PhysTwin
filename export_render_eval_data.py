import csv
import json
import shutil
from pathlib import Path

base_path = Path("./data/different_types")
output_path = Path("./data/render_eval_data") 
CONTROLLER_NAME = "hand"


def ensure_dir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


ensure_dir(output_path)

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]
        
        case_path = base_path / case_name
        if not case_path.exists():
            continue
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        # Create the directory for the case
        case_output_path = output_path / case_name
        ensure_dir(case_output_path)
        ensure_dir(case_output_path / "mask")

        for i in range(3):
            # Copy the original RGB image
            shutil.copytree(
                case_path / "color",
                case_output_path / "color",
                dirs_exist_ok=True
            )

            # Copy only the object mask image
            # Get the mask path for the image
            mask_info_path = case_path / "mask" / f"mask_info_{i}.json"
            with mask_info_path.open("r") as f:
                data = json.load(f)

            obj_idx = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)

            mask_output_path = case_output_path / "mask" / str(i)
            ensure_dir(mask_output_path)
            
            mask_source = case_path / "mask" / str(i) / str(obj_idx)
            shutil.copytree(mask_source, mask_output_path, dirs_exist_ok=True)

        # Copy the split.json
        shutil.copy2(case_path / "split.json", case_output_path / "split.json")