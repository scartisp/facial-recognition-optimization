"""Static INT8 PTQ of the SCRFD-500MF detector, INT8 except depthwise (as int8nodw for the other models).

Run from the Version1 folder, venv active, after quantize/quantize.py:
    python quantize/quantize_detector.py              # detector input 320 (the most efficient Pi setting)
    python quantize/quantize_detector.py --size 640

The detector input size is fixed in the graph first (SCRFD normally takes any size; a static graph is
what pipeline.py builds per size anyway), so each INT8 file serves one input size. Settings match the
other int8nodw models: QDQ, S8S8, MinMax, per-channel, Conv with group > 1 left in FP32. Calibration
uses the same 128 frames as the recognizer and liveness models (the seed-0 manifest: first 4 s of
the p01/p02 team videos, never test frames), as full frames letterboxed exactly as SCRFD does.

Writes models/int8/det_500m_det<size>_int8_per_channel_nodw_s<seed>.onnx (git-ignored), adds its
hash to configs/models.lock.json, and records results/raw/quantization_manifest_detector_det<size>_s<seed>.json.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import quantize as q  # noqa: E402
from prepare_data import sample_video  # noqa: E402

pl = q.pl
from insightface.model_zoo.scrfd import _static_scrfd_model  # noqa: E402

INPUT_MEAN, INPUT_STD = 127.5, 128.0  # SCRFD's own preprocessing (insightface scrfd.py)


def letterbox_blob(frame_bgr, size):
    """SCRFD._detect_candidates + _prepare_input_blob for a square input of `size`."""
    h, w = frame_bgr.shape[:2]
    if h / w > 1.0:
        new_h, new_w = size, int(size / (h / w))
    else:
        new_w, new_h = size, int(size * (h / w))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(frame_bgr, (new_w, new_h))
    return cv2.dnn.blobFromImage(canvas, 1.0 / INPUT_STD, (size, size), (INPUT_MEAN,) * 3, swapRB=True)


def calibration_frames(samples):
    """Full frames for manifest samples like 'p01_live_bright@0.89s'."""
    pseudonyms = json.loads((pl.version1_folder / "private" / "pseudonyms.json").read_text())
    name_of = {}
    for name, code in pseudonyms.items():
        name_of.setdefault(code, name)
    wanted = {}
    for s in samples:
        source, t = s.rsplit("@", 1)
        wanted.setdefault(source, set()).add(round(float(t.rstrip("s")), 2))
    frames = {}
    for source, times in wanted.items():
        code, rest = source.split("_", 1)
        path = next(p for p in (pl.version1_folder / "private").iterdir()
                    if p.stem == f"{name_of[code]}_{rest}" and p.suffix.lower() in {".mp4", ".webm"})
        for t, frame in sample_video(path, lambda t: round(t, 2) in times):
            frames[f"{source}@{round(t, 2)}s"] = frame
    missing = [s for s in samples if s not in frames]
    if missing:
        raise SystemExit(f"{len(missing)} calibration frames not found, e.g. {missing[0]}")
    return [frames[s] for s in samples]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = json.loads(pl.default_config.read_text())
    pl.check_pins(base)
    det_path = pl.resolve_path(base["models"]["detector"]["path"])
    manifest = json.loads((q.results_raw / f"quantization_manifest_s{args.seed}.json").read_text())
    samples = manifest["samples"]
    blobs = np.ascontiguousarray(np.concatenate([letterbox_blob(f, args.size) for f in calibration_frames(samples)]))
    print(f"{len(blobs)} calibration frames at {args.size}x{args.size}")

    out_path = q.int8_folder / f"det_500m_det{args.size}_int8_per_channel_nodw_s{args.seed}.onnx"
    with tempfile.TemporaryDirectory() as temp:
        input_name = onnxruntime.InferenceSession(str(det_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name
        static = Path(temp) / "det_static.onnx"
        static.write_bytes(_static_scrfd_model(det_path, (args.size, args.size), input_name))
        excluded = q.quantize_model(static, blobs, out_path, per_channel=True, exclude_depthwise=True)

    lock_key = f"int8/{out_path.name}"
    lock = json.loads(pl.lock_path.read_text())
    lock["files"][lock_key] = q.sha256_of(out_path)
    pl.lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    record = {k: manifest[k] for k in ("seed", "calibration_size", "calibration_pool_size", "calibration_source",
                                       "samples", "quant_format", "activation_type", "weight_type",
                                       "calibrate_method", "reduce_range", "min_opset")}
    record.update({"model": "detector", "input_size": args.size, "variant": "per_channel_nodw",
                   "onnxruntime_version": onnxruntime.__version__, "excluded_nodes": excluded,
                   "output": {"path": out_path.relative_to(pl.repo_folder).as_posix(), "lock_key": lock_key,
                              "sha256": lock["files"][lock_key]}})
    (q.results_raw / f"quantization_manifest_detector_det{args.size}_s{args.seed}.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print(f"Wrote {out_path.name} ({out_path.stat().st_size / 1e6:.2f} MB), {len(excluded)} depthwise convs left FP32")


if __name__ == "__main__":
    main()
