"""Latency, memory and model-size benchmark for the 9 ablation configs.

Run from the Version1 folder, venv active:
    python bench/bench.py --label pc          # all 9 cells, each in its own process
    python bench/bench.py --label pi --runs 500 --warmup 50

Each cell runs the full pipeline (detect -> liveness -> recognize) on a fixed set of frames
taken from a team video (any private/*_live_bright.* file), cycling through them. Warm-up runs
are excluded. Each cell gets a fresh process so memory (RSS) is not shared between cells.

Writes:
    results/raw/bench/<label>/<cell>.csv     one row per timed run
    results/raw/bench/<label>/<cell>.json    summary + host/software info
    results/processed/bench_<label>.csv      p50/p95/std per stage, RSS, model sizes
"""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pipeline as pl  # noqa: E402

version1_folder = pl.version1_folder
ablation_folder = pl.repo_folder / "configs" / "ablation"
bench_raw = version1_folder / "results" / "raw" / "bench"
results_processed = version1_folder / "results" / "processed"

STAGES = ["detect_ms", "liveness_prep_ms", "liveness_infer_ms", "recog_prep_ms", "recog_infer_ms", "total_ms"]
FRAME_COUNT = 20


def load_frames():
    """FRAME_COUNT evenly spaced frames (with a detectable face) from one bright live team video."""
    videos = sorted(p for p in (version1_folder / "private").glob("*_live_bright.*"))
    if not videos:
        raise FileNotFoundError("Need a private/*_live_bright.* video to benchmark on")
    capture = cv2.VideoCapture(str(videos[0]))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    step = max(1, len(frames) // FRAME_COUNT)
    return frames[::step][:FRAME_COUNT]


def cpu_temperature():
    """Pi CPU temperature in C, or None elsewhere."""
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def host_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        info["device_model"] = model_file.read_text().strip("\x00\n")
    return info


def summarize(values):
    values = np.asarray(values)
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def run_cell(config_path, label, runs, warmup):
    config = json.loads(Path(config_path).read_text())
    pl.check_pins(config)
    pipeline = pl.Pipeline(config)
    process = psutil.Process(os.getpid())

    frames = load_frames()
    faces = [pipeline.detect(f) for f in frames]
    frames = [f for f, face in zip(frames, faces) if face is not None]
    if not frames:
        raise RuntimeError("No face found in the benchmark frames")

    temp_start = cpu_temperature()
    rows = []
    for run in range(warmup + runs):
        frame = frames[run % len(frames)]
        t0 = time.perf_counter()
        bbox, keypoints = pipeline.detect(frame)
        t1 = time.perf_counter()
        live_blob = pipeline.liveness_blob(pipeline.liveness_crop(frame, bbox))
        t2 = time.perf_counter()
        pipeline.liveness_scores_blob(live_blob)
        t3 = time.perf_counter()
        rec_blob = pipeline.recognizer_blob(pipeline.recognizer_crop(frame, keypoints))
        t4 = time.perf_counter()
        pipeline.embed_blob(rec_blob)
        t5 = time.perf_counter()
        if run < warmup:
            continue
        rows.append({
            "run": run - warmup,
            "detect_ms": (t1 - t0) * 1000, "liveness_prep_ms": (t2 - t1) * 1000,
            "liveness_infer_ms": (t3 - t2) * 1000, "recog_prep_ms": (t4 - t3) * 1000,
            "recog_infer_ms": (t5 - t4) * 1000, "total_ms": (t5 - t0) * 1000,
            "rss_mb": process.memory_info().rss / 2**20,
        })
    temp_end = cpu_temperature()

    cell = config["precision"]
    out_folder = bench_raw / label
    out_folder.mkdir(parents=True, exist_ok=True)
    with open(out_folder / f"{cell}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    sizes = {role: os.path.getsize(pl.resolve_path(m["path"])) / 2**20 for role, m in config["models"].items()}
    summary = {
        "cell": cell, "label": label, "runs": runs, "warmup_excluded": warmup, "frames": len(frames),
        "intra_op_threads": "onnxruntime default",
        "stages": {s: summarize([r[s] for r in rows]) for s in STAGES},
        "rss_mb": {"median": statistics.median(r["rss_mb"] for r in rows), "max": max(r["rss_mb"] for r in rows)},
        "model_size_mb": sizes,
        "cpu_temp_c": {"start": temp_start, "end": temp_end},
        "host": host_info(),
    }
    (out_folder / f"{cell}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{cell}: total p50 {summary['stages']['total_ms']['p50']:.1f} ms, "
          f"p95 {summary['stages']['total_ms']['p95']:.1f} ms, RSS {summary['rss_mb']['max']:.0f} MB")


def run_isolated(label, runs, warmup):
    """Model-only timing (session.run on a fixed preprocessed input) for each of the 6 model files,
    with 1 thread and with ONNX Runtime's default thread count. Separates the quantization effect
    from preprocessing, detection and thread scheduling noise."""
    configs = {v: json.loads((ablation_folder / f"rec-{v}_live-{v}.json").read_text())
               for v in ("fp32", "int8pt", "int8pc")}
    frame = load_frames()[0]
    prep = pl.Pipeline(configs["fp32"])
    bbox, keypoints = prep.detect(frame)
    inputs = {"recognizer": prep.recognizer_blob(prep.recognizer_crop(frame, keypoints)),
              "liveness": prep.liveness_blob(prep.liveness_crop(frame, bbox))}

    rows = []
    for variant, config in configs.items():
        pl.check_pins(config)
        for role in ("recognizer", "liveness"):
            path = pl.resolve_path(config["models"][role]["path"])
            for threads in (1, 0):
                options = ort.SessionOptions()
                options.log_severity_level = 3
                options.intra_op_num_threads = threads
                session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                feed = {session.get_inputs()[0].name: inputs[role]}
                times = []
                for run in range(warmup + runs):
                    start = time.perf_counter()
                    session.run(None, feed)
                    if run >= warmup:
                        times.append((time.perf_counter() - start) * 1000)
                s = summarize(times)
                rows.append({"model": role, "variant": variant, "threads": threads or "default",
                             "runs": runs, **{k: round(v, 4) for k, v in s.items()}})
                print(f"{role:10s} {variant:6s} threads={threads or 'default':7} p50 {s['p50']:.2f} ms  p95 {s['p95']:.2f} ms")

    out_folder = bench_raw / label
    out_folder.mkdir(parents=True, exist_ok=True)
    with open(results_processed / f"bench_isolated_{label}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_folder / "isolated_host.json").write_text(json.dumps(host_info(), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the 9 ablation configs")
    parser.add_argument("--isolated", action="store_true", help="model-only timing instead of the 9 cells")
    parser.add_argument("--label", required=True, help="machine label, e.g. pc or pi")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--config", help="benchmark a single config (used internally)")
    args = parser.parse_args()

    if args.config:
        run_cell(args.config, args.label, args.runs, args.warmup)
        return
    if args.isolated:
        results_processed.mkdir(parents=True, exist_ok=True)
        run_isolated(args.label, args.runs, args.warmup)
        return

    for config_path in sorted(ablation_folder.glob("*.json")):
        subprocess.run([sys.executable, __file__, "--label", args.label, "--runs", str(args.runs),
                        "--warmup", str(args.warmup), "--config", str(config_path)], check=True)

    rows = []
    for path in sorted((bench_raw / args.label).glob("*.json")):
        s = json.loads(path.read_text())
        row = {"cell": s["cell"], "runs": s["runs"]}
        for stage in STAGES:
            for stat in ("p50", "p95", "std"):
                row[f"{stage}_{stat}"] = round(s["stages"][stage][stat], 3)
        row["rss_mb_max"] = round(s["rss_mb"]["max"], 1)
        row["recognizer_mb"] = round(s["model_size_mb"]["recognizer"], 3)
        row["liveness_mb"] = round(s["model_size_mb"]["liveness"], 3)
        rows.append(row)
    results_processed.mkdir(parents=True, exist_ok=True)
    with open(results_processed / f"bench_{args.label}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {results_processed / f'bench_{args.label}.csv'}")


if __name__ == "__main__":
    main()
