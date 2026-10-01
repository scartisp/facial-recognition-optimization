from pathlib import Path

import cv2
from insightface.app import FaceAnalysis

# Finds the Version1 folder automatically
project_folder = Path(__file__).resolve().parents[1]

# Your local photo must be stored here:
# Version1/private/reference.jpg
photo_path = project_folder / "private" / "reference.jpg"

image = cv2.imread(str(photo_path))
if image is None:
    raise FileNotFoundError(
        f"Could not open {photo_path}. "
        "Create the private folder and add a photo named reference.jpg."
    )

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

faces = app.get(image)

print(f"Faces found: {len(faces)}")

for number, face in enumerate(faces, start=1):
    print(f"Face {number}: confidence={face.det_score:.3f}")
    print(f"Embedding length: {len(face.embedding)}")

result = app.draw_on(image, faces)

benchmark_folder = project_folder / "benchmarks"
benchmark_folder.mkdir(exist_ok=True)

output_path = benchmark_folder / "private_photo_result.jpg"
cv2.imwrite(str(output_path), result)

print(f"Saved result to: {output_path}")
