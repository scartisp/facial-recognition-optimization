"""How closely the INT8 detector reproduces the FP32 detector, on the team videos' test frames.

    python eval/detector_fidelity.py            # detector input 320

Frames: every team video (session 1 after the 4 s calibration window, all of session 2) at 10 frames/s,
the same frames as eval/sequential_eval.py. Both detectors run on each frame; the INT8 one is
quantize/quantize_detector.py's INT8-except-depthwise file. Reports faces found by each, box IoU and
landmark error (mean of the 5 keypoint distances divided by the FP32 inter-ocular distance).
Writes results/processed/detector_fidelity_det<size>.csv (summary) and results/raw/... (per frame).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sequential_eval as se  # noqa: E402
from prepare_data import CALIB_SECONDS, EverySeconds, sample_video  # noqa: E402

pl = se.pl


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()
    base = json.loads(pl.default_config.read_text())
    base["detector"]["input_size"] = [args.size, args.size]
    pl.check_pins(base)
    int8 = json.loads(json.dumps(base))
    int8["models"]["detector"] = dict(se.INT8_DETECTOR[args.size])
    pl.check_pins(int8)
    dets = {"fp32": pl.Pipeline(base), "int8nodw": pl.Pipeline(int8)}

    rows = []
    for path, meta in tqdm(se.team_videos(), desc="videos"):
        for t, frame in sample_video(path, EverySeconds(1.0 / args.fps)):
            if meta["dataset"] == "own" and t < CALIB_SECONDS:
                continue  # calibration window: the INT8 detector was calibrated on some of these frames
            f = dets["fp32"].detect(frame)
            q = dets["int8nodw"].detect(frame)
            row = {"dataset": meta["dataset"], "source": meta["source"], "kind": meta["kind"], "time_s": round(t, 3),
                   "fp32_face": int(f is not None), "int8_face": int(q is not None)}
            if f is not None and q is not None:
                (fb, fk), (qb, qk) = f, q
                iod = float(np.linalg.norm(fk[0] - fk[1]))
                row["iou"] = round(iou(fb, qb), 5)
                row["nme"] = round(float(np.linalg.norm(fk - qk, axis=1).mean() / iod), 5)
                row["face_width_px"] = int(fb[2] - fb[0])
            rows.append(row)

    out_raw = pl.version1_folder / "results" / "raw" / f"detector_fidelity_det{args.size}.csv"
    with open(out_raw, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "source", "kind", "time_s", "fp32_face", "int8_face",
                                           "iou", "nme", "face_width_px"])
        w.writeheader()
        w.writerows(rows)
    summary = []
    for dataset in ["own", "session2", "all"]:
        sel = [r for r in rows if dataset == "all" or r["dataset"] == dataset]
        both = [r for r in sel if "iou" in r]
        ious = np.array([r["iou"] for r in both])
        nmes = np.array([r["nme"] for r in both])
        summary.append({
            "dataset": dataset, "detector_size": args.size, "frames": len(sel),
            "fp32_faces": sum(r["fp32_face"] for r in sel), "int8_faces": sum(r["int8_face"] for r in sel),
            "only_fp32": sum(r["fp32_face"] and not r["int8_face"] for r in sel),
            "only_int8": sum(r["int8_face"] and not r["fp32_face"] for r in sel),
            "iou_mean": round(float(ious.mean()), 4), "iou_p05": round(float(np.percentile(ious, 5)), 4),
            "iou_min": round(float(ious.min()), 4),
            "nme_mean": round(float(nmes.mean()), 4), "nme_p95": round(float(np.percentile(nmes, 95)), 4),
            "nme_max": round(float(nmes.max()), 4),
        })
    out = pl.version1_folder / "results" / "processed" / f"detector_fidelity_det{args.size}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
