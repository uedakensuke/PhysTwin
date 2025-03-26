import subprocess

# Run evaluation scripts
subprocess.run(["python", "evaluate_chamfer.py"])
subprocess.run(["python", "evaluate_track.py"]) 
subprocess.run(["python", "gaussian_splatting/evaluate_render.py"])