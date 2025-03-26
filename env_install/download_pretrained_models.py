from pathlib import Path
import urllib.request

def download_file(url, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name
    dest_path = dest_dir / filename
    if not dest_path.exists():
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
    else:
        print(f"File already exists: {dest_path}")

# GroundedSAM2 Checkpoints
grounded_sam_urls = [
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
]

grounded_sam_dir = Path("data_process/groundedSAM_checkpoints")

# SuperGlue Checkpoints
superglue_urls = [
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_indoor.pth",
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_outdoor.pth",
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superpoint_v1.pth"
]

superglue_dir = Path("data_process/models/weights")

# Download files
for url in grounded_sam_urls:
    download_file(url, grounded_sam_dir)

for url in superglue_urls:
    download_file(url, superglue_dir)

print("All required checkpoints have been downloaded successfully.")
