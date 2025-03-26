from pathlib import Path
import subprocess

base_path = Path("./data/different_types")
output_path = Path("./data/different_types_human_mask")

def ensure_dir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)

dir_names = list(base_path.glob("*"))
for dir_path in dir_names:
    case_name = dir_path.name
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    case_output_path = output_path / case_name
    ensure_dir(case_output_path)
    # Process to get the whole human mask for the video

    TEXT_PROMPT = "human"
    camera_num = 3
    assert len(list((base_path / case_name / "depth").glob("*"))) == camera_num

    for camera_idx in range(camera_num):
        print(f"Processing {case_name} camera {camera_idx}")
        subprocess.run([
            "python", "./data_process/segment_util_video.py",
            "--output_path", str(case_output_path),
            "--base_path", str(base_path),
            "--case_name", case_name,
            "--TEXT_PROMPT", TEXT_PROMPT,
            "--camera_idx", str(camera_idx)
        ])
        tmp_path = base_path / case_name / "tmp_data"
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)