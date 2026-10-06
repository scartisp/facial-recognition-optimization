"""Combined face authentication pipeline.

frame -> SCRFD detect (one shared detector) -> liveness model ----\
                                            \-> recognizer -> AND -> accept / reject

Usage (from the Version1 folder, venv active):
    python app/pipeline.py enroll                          # uses private/reference.jpg
    python app/pipeline.py enroll --image private/me.jpg
    python app/pipeline.py verify --image private/test.jpg
    python app/pipeline.py verify --camera                 # press q to quit

All model paths and thresholds come from a config file (default: configs/pipeline_fp32.json),
so an INT8 run only needs a different config, not different code.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.model_zoo.scrfd import SCRFD, _static_scrfd_model
from insightface.utils import face_align

# Finds the repo root and the Version1 folder automatically
version1_folder = Path(__file__).resolve().parents[1]
repo_folder = version1_folder.parent
liveness_repo = version1_folder / "external" / "face-antispoof-onnx"

default_config = repo_folder / "configs" / "pipeline_fp32.json"
lock_path = repo_folder / "configs" / "models.lock.json"

# Load facenox's own crop/preprocess functions straight from their file, so the
# liveness input is prepared exactly the way the model was trained to expect.
_spec = importlib.util.spec_from_file_location(
    "facenox_preprocess", liveness_repo / "src" / "inference" / "preprocess.py"
)
facenox_preprocess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(facenox_preprocess)


def resolve_path(path_text):
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else repo_folder / path


def sha256_of(path):
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest().upper()


def check_pins(config):
    """Stop if any model file differs from the one recorded in models.lock.json."""
    lock = json.loads(lock_path.read_text())

    for role, model in config["models"].items():
        path = resolve_path(model["path"])
        if not path.exists():
            raise FileNotFoundError(f"{role} model not found: {path}")
        lock_key = model.get("lock_key")
        if lock_key is None:
            continue
        expected = lock["files"][lock_key].upper()
        if sha256_of(path) != expected:
            raise RuntimeError(
                f"{role} model {path} does not match the hash pinned in {lock_path.name} "
                f"({lock_key}). The baseline would not be reproducible."
            )

    expected_commit = lock["liveness_repo"]["commit"].strip()
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(liveness_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if actual_commit != expected_commit:
            print(f"WARNING: liveness repo is at {actual_commit}, pinned commit is {expected_commit}")
    except (OSError, subprocess.CalledProcessError):
        print("WARNING: could not read the liveness repo commit (is git installed?)")


def session_options():
    options = ort.SessionOptions()
    # Errors only: the recognizer declares batch size 1 and ORT warns on every batched run
    options.log_severity_level = 3
    # The three models run one after another. With spinning on (ORT's default), each idle
    # session's thread pool keeps busy-waiting and steals CPU from whichever model is running.
    # Threading only: outputs are unchanged.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def new_session(path_or_bytes):
    source = path_or_bytes if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
    return ort.InferenceSession(source, sess_options=session_options(), providers=["CPUExecutionProvider"])


def scrfd_session_factory(model_file, input_size, reference_session):
    """SCRFD builds one static-shape session per input size; give those our session options too."""
    return new_session(_static_scrfd_model(model_file, input_size, reference_session.get_inputs()[0].name))


class Pipeline:
    def __init__(self, config):
        self.config = config
        models = config["models"]

        # Detector: fixed FP32 in every configuration, so it never explains a difference
        detector_path = resolve_path(models["detector"]["path"])
        self.detector = SCRFD(
            model_file=str(detector_path),
            session=new_session(detector_path),
            resolution_session_factory=scrfd_session_factory,
        )
        self.detector.prepare(
            ctx_id=-1,
            input_size=tuple(config["detector"]["input_size"]),
            det_thresh=config["detector"]["det_thresh"],
        )

        self.recognizer = new_session(resolve_path(models["recognizer"]["path"]))
        self.recognizer_input = self.recognizer.get_inputs()[0].name

        self.liveness = new_session(resolve_path(models["liveness"]["path"]))
        self.liveness_input = self.liveness.get_inputs()[0].name

        real_prob = config["liveness"]["real_prob_threshold"]
        self.liveness_logit_threshold = math.log(real_prob / (1 - real_prob))

    def detect(self, frame_bgr, metric="max"):
        """Return (bbox, keypoints) of one face, or None.

        metric="max" picks the largest face; "default" favours the face nearest the centre.
        """
        boxes, keypoints = self.detector.detect(frame_bgr, max_num=1, metric=metric)
        if boxes.shape[0] == 0:
            return None
        return boxes[0, :4], keypoints[0]

    # The face-crop steps below do not depend on precision. Evaluation and calibration
    # cache their uint8 outputs, so FP32 and INT8 models always see identical inputs.

    def recognizer_crop(self, frame_bgr, keypoints):
        """5-point alignment to a 112x112 BGR uint8 face."""
        size = self.config["recognizer"]["input_size"]
        return face_align.norm_crop(frame_bgr, landmark=keypoints, image_size=size)

    def recognizer_blob(self, aligned_faces):
        """uint8 BGR faces (one or a list) -> normalized NCHW float32 batch."""
        settings = self.config["recognizer"]
        size = settings["input_size"]
        mean = settings["input_mean"]
        if not isinstance(aligned_faces, list):
            aligned_faces = [aligned_faces]
        return cv2.dnn.blobFromImages(
            aligned_faces, 1.0 / settings["input_std"], (size, size),
            (mean, mean, mean), swapRB=settings["swap_rb"],
        )

    def embed_blob(self, blob):
        """Return unit-length embeddings, one row per face."""
        embeddings = self.recognizer.run(None, {self.recognizer_input: blob})[0]
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def embed(self, frame_bgr, keypoints):
        """Align to 112x112 and return a unit-length 512-d embedding."""
        return self.embed_blob(self.recognizer_blob(self.recognizer_crop(frame_bgr, keypoints)))[0]

    def liveness_crop(self, frame_bgr, bbox):
        """facenox crop + letterbox, returned as CHW uint8 (exactly 255 x the model input)."""
        settings = self.config["liveness"]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = (int(v) for v in bbox)
        face = facenox_preprocess.crop(frame_rgb, (x1, y1, x2, y2), settings["bbox_expansion_factor"])
        chw = facenox_preprocess.preprocess(face, settings["input_size"])
        return np.rint(chw * 255.0).astype(np.uint8)

    @staticmethod
    def liveness_blob(crops_u8):
        """CHW uint8 crops (one or a stacked batch) -> NCHW float32 in [0, 1], as facenox does."""
        crops_u8 = np.asarray(crops_u8)
        if crops_u8.ndim == 3:
            crops_u8 = crops_u8[np.newaxis]
        return crops_u8.astype(np.float32) / 255.0

    def liveness_scores_blob(self, blob):
        """Return real_logit - spoof_logit per face (higher means more likely a live face)."""
        logits = self.liveness.run(None, {self.liveness_input: blob})[0]
        return logits[:, 0] - logits[:, 1]

    def liveness_score(self, frame_bgr, bbox):
        return float(self.liveness_scores_blob(self.liveness_blob(self.liveness_crop(frame_bgr, bbox)))[0])

    def authenticate(self, frame_bgr, enrolled_embedding):
        timings = {}

        start = time.perf_counter()
        face = self.detect(frame_bgr)
        timings["detect_ms"] = (time.perf_counter() - start) * 1000
        if face is None:
            return {"accept": False, "reason": "no face", "timings": timings}
        bbox, keypoints = face

        start = time.perf_counter()
        live_score = self.liveness_score(frame_bgr, bbox)
        timings["liveness_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        similarity = float(np.dot(self.embed(frame_bgr, keypoints), enrolled_embedding))
        timings["recognize_ms"] = (time.perf_counter() - start) * 1000

        is_live = live_score >= self.liveness_logit_threshold
        is_match = similarity >= self.config["verification"]["cosine_threshold"]

        if is_live and is_match:
            reason = "live and matched"
        elif not is_live and not is_match:
            reason = "spoof and no match"
        elif not is_live:
            reason = "spoof"
        else:
            reason = "no match"

        return {
            "accept": bool(is_live and is_match),
            "reason": reason,
            "liveness_score": live_score,
            "similarity": similarity,
            "bbox": bbox,
            "timings": timings,
        }


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not open {path}")
    return image


def enrollment_path(config):
    # One enrollment per precision: enroll and verify must use the same recognizer
    return version1_folder / "private" / f"enrollment_{config['precision']}.npy"


def enroll(pipeline, image_path):
    image = load_image(image_path)
    face = pipeline.detect(image)
    if face is None:
        raise RuntimeError(f"No face found in {image_path}")
    embedding = pipeline.embed(image, face[1])

    path = enrollment_path(pipeline.config)
    path.parent.mkdir(exist_ok=True)
    np.save(path, embedding)
    print(f"Enrolled from {image_path} -> {path}")


def print_result(result):
    decision = "ACCEPT" if result["accept"] else "REJECT"
    print(f"{decision} ({result['reason']})")
    if "similarity" in result:
        print(f"  liveness score (logit diff): {result['liveness_score']:.3f}")
        print(f"  cosine similarity:           {result['similarity']:.3f}")
    # Single-run timings for sanity only; real latency numbers come from the benchmark scripts
    print("  " + ", ".join(f"{name}={ms:.1f}" for name, ms in result["timings"].items()))


def verify_camera(pipeline, enrolled, camera_index):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            result = pipeline.authenticate(frame, enrolled)

            color = (0, 200, 0) if result["accept"] else (0, 0, 220)
            label = ("ACCEPT" if result["accept"] else "REJECT") + f": {result['reason']}"
            if "bbox" in result:
                x1, y1, x2, y2 = (int(v) for v in result["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label += f"  live={result['liveness_score']:.2f} sim={result['similarity']:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Face authentication (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Liveness + face verification pipeline")
    parser.add_argument("mode", choices=["enroll", "verify"])
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--image", help="image file (enroll defaults to private/reference.jpg)")
    parser.add_argument("--camera", action="store_true", help="verify from the webcam")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    check_pins(config)
    pipeline = Pipeline(config)

    if args.mode == "enroll":
        enroll(pipeline, args.image or version1_folder / "private" / "reference.jpg")
        return

    path = enrollment_path(config)
    if not path.exists():
        raise FileNotFoundError(f"No enrollment at {path}. Run: python app/pipeline.py enroll")
    enrolled = np.load(path)

    if args.camera:
        verify_camera(pipeline, enrolled, config["camera"]["index"])
    elif args.image:
        print_result(pipeline.authenticate(load_image(args.image), enrolled))
    else:
        parser.error("verify needs --image PATH or --camera")


if __name__ == "__main__":
    main()
