import pickle
import csv
import json
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

base_path = Path("./data/different_types")
prediction_path = Path("experiments") 
output_file = Path("results/final_track.csv")


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return np.mean(track_errors)


if not output_file.parent.exists():
    output_file.parent.mkdir()

file = open(output_file, mode="w", newline="", encoding="utf-8")
writer = csv.writer(file)
writer.writerow(
    [
        "Case Name",
        "Train Track Error",
        "Test Track Error",
    ]
)

dir_names = list(base_path.glob("*"))
for dir_path in dir_names:
    case_name = dir_path.name
    # if case_name != "single_lift_dinosor":
    #     continue
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with open(dir_path / "split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]
    train_frame = split["train"][1]
    test_frame = split["test"][1]

    with open(prediction_path / case_name / "inference.pkl", "rb") as f:
        vertices = pickle.load(f)

    with open(dir_path / "gt_track_3d.pkl", "rb") as f:
        gt_track_3d = pickle.load(f)

    # Locate the index of corresponding point index in the vertices, if nan, then ignore the points
    mask = ~np.isnan(gt_track_3d[0]).any(axis=1)

    kdtree = KDTree(vertices[0])
    dis, idx = kdtree.query(gt_track_3d[0][mask])

    train_track_error = evaluate_prediction(
        1, train_frame, vertices, gt_track_3d, idx, mask
    )
    test_track_error = evaluate_prediction(
        train_frame, test_frame, vertices, gt_track_3d, idx, mask
    )
    writer.writerow([case_name, train_track_error, test_track_error])
file.close()
