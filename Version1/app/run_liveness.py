from pathlib import Path
import subprocess
import sys

project_folder = Path(__file__).resolve().parents[1]
liveness_folder = project_folder / "external" / "face-antispoof-onnx"

command = [
    sys.executable,
    str(liveness_folder / "demo.py"),
    "--detector_model",
    str(liveness_folder / "models" / "detector.onnx"),
    "--liveness_model",
    str(liveness_folder / "models" / "best" / "98.20" / "best_model.onnx"),
]

subprocess.run(command, cwd=liveness_folder, check=True)
