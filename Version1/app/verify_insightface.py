import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image

app = FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

image = get_image("t1")
faces = app.get(image)

print(f"Faces found: {len(faces)}")

for number, face in enumerate(faces, start=1):
    print(f"Face {number}: confidence={face.det_score:.3f}")
    print(f"Embedding length: {len(face.embedding)}")

result = app.draw_on(image, faces)
cv2.imwrite("insightface_result.jpg", result)
print("Saved insightface_result.jpg")
