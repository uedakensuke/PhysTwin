from pathlib import Path
import subprocess
import json

base_path = Path("./data/different_types")
for dir_name in base_path.glob("*"):
    case_name = dir_name.name

    # Read the train test split
    with open(base_path / case_name / "split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    subprocess.run(
        f"python train_warp.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}",
    )
