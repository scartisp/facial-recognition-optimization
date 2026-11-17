"""Runtime optimization ablation across precision, detector size, detection interval, and concurrency.

Sweeps combinations of:
    - precision:     [fp32, int8pt, int8pc, int8nodw] (via --precision)
    - detector_size: [640, 480, 320]
    - detect_every:  [1, 3, 5]       (reuse bbox/keypoints between detections)
    - exec_mode:     [serial, parallel_auto, parallel_half]

Each cell runs in a fresh subprocess so memory (RSS) and ONNX Runtime thread pools
are isolated. Measures both stage latencies (p50/p95) and score drift / decision flips
against the cell's own (640, detect_every=1, serial) baseline on sequential frames from
private/*_live_bright.*.

Writes:
    results/raw/bench/<raw_subdir>/<cell>.json
    results/processed/<out_prefix>.csv
    results/processed/<out_prefix>.md
"""
import argparse
from collections import deque
import copy
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pipeline as pl  # noqa: E402

version1_folder = pl.version1_folder
ablation_configs = pl.repo_folder / "configs" / "ablation"
bench_raw_root = version1_folder / "results" / "raw" / "bench"
processed_folder = version1_folder / "results" / "processed"

COSINE_THRESHOLD = 0.2421  # LFW FAR=0.1% operating point from AGENTS.md
PRECISIONS = ["fp32", "int8pt", "int8pc", "int8nodw"]
DETECTOR_SIZES = [640, 480, 320]
DETECT_INTERVALS = [1, 3, 5]
EXEC_MODES = ["serial", "parallel_auto", "parallel_half"]
SMOOTH_WINDOWS = [1, 3, 5, 10]


def output_paths(label, precision):
    """Return (raw_folder, csv_path, md_path) for the requested label and precision."""
    if label == "local" and precision == "fp32":
        raw_dir = bench_raw_root / "local_ablation"
        stem = "ablation_local"
    elif precision == "fp32":
        raw_dir = bench_raw_root / f"{label}_ablation"
        stem = f"ablation_{label}"
    else:
        raw_dir = bench_raw_root / f"{label}_ablation_{precision}"
        stem = f"ablation_{label}_{precision}"
    return raw_dir, processed_folder / f"{stem}.csv", processed_folder / f"{stem}.md"


def load_sequential_clip(max_frames):
    """Load up to max_frames consecutive frames from the first private/*_live_bright.* video."""
    videos = sorted(p for p in (version1_folder / "private").glob("*_live_bright.*"))
    if not videos:
        raise FileNotFoundError("Need a private/*_live_bright.* video to run ablation")
    capture = cv2.VideoCapture(str(videos[0]))
    frames = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {videos[0]}")
    return frames, videos[0].name


def load_enrolled(pipeline, first_frame):
    """Load enrollment_<precision>.npy if present, otherwise enroll from first_frame."""
    path = pl.enrollment_path(pipeline.config)
    if path.exists():
        return np.load(path)
    face = pipeline.detect(first_frame)
    if face is None:
        raise RuntimeError("No face detected in first frame for fallback enrollment")
    return pipeline.embed(first_frame, face[1])


def base_config_for(precision):
    """Load the base pipeline config matching a precision variant."""
    if precision == "fp32":
        return json.loads(pl.default_config.read_text())
    cfg_path = ablation_configs / f"rec-{precision}_live-{precision}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing config {cfg_path}; run quantize/quantize.py first"
        )
    return json.loads(cfg_path.read_text())


def build_config(precision, detector_size, exec_mode, half_threads):
    """Return a pipeline config dict for the requested precision, detector_size, and execution mode."""
    config = copy.deepcopy(base_config_for(precision))
    config["detector"]["input_size"] = [detector_size, detector_size]
    config["verification"]["cosine_threshold"] = COSINE_THRESHOLD
    if exec_mode == "serial":
        config["parallel"] = False
        config["intra_op_num_threads"] = 0
    elif exec_mode == "parallel_auto":
        config["parallel"] = True
        config["intra_op_num_threads"] = 0
    elif exec_mode == "parallel_half":
        config["parallel"] = True
        config["intra_op_num_threads"] = half_threads
    else:
        raise ValueError(f"Unknown exec_mode: {exec_mode}")
    return config


def summarize_metric(values):
    """Compute p50, p95, mean, and std for a list of numeric values."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def cell_name(detector_size, detect_every, exec_mode):
    """Format a deterministic cell identifier for filenames and tables."""
    return f"det{detector_size}_every{detect_every}_{exec_mode}"


def run_cell(label, precision, detector_size, detect_every, exec_mode, max_frames, warmup):
    """Run one ablation cell in isolation and save its JSON summary and per-frame scores."""
    physical_cores = psutil.cpu_count(logical=False) or 2
    half_threads = max(1, physical_cores // 2)
    config = build_config(precision, detector_size, exec_mode, half_threads)
    pl.check_pins(config)

    frames, video_name = load_sequential_clip(max_frames)
    pipeline = pl.Pipeline(config)
    process = psutil.Process(os.getpid())

    try:
        enrolled = load_enrolled(pipeline, frames[0])

        for i in range(warmup):
            pipeline.authenticate(frames[i % len(frames)], enrolled)

        detect_ms, post_ms, total_ms, cpu_ms, rss_mb = [], [], [], [], []
        per_frame = []
        cached_face = None

        for idx, frame in enumerate(frames):
            use_cache = cached_face if (cached_face is not None and idx % detect_every != 0) else None
            c0 = time.process_time()
            t0 = time.perf_counter()
            result = pipeline.authenticate(frame, enrolled, cached_face=use_cache)
            t1 = time.perf_counter()
            c1 = time.process_time()

            if "bbox" in result:
                cached_face = (result["bbox"], result["keypoints"])
            else:
                cached_face = None

            timings = result["timings"]
            detect_ms.append(timings.get("detect_ms", 0.0))
            post_ms.append(timings.get("post_detect_ms", 0.0))
            total_ms.append((t1 - t0) * 1000.0)
            cpu_ms.append((c1 - c0) * 1000.0)
            rss_mb.append(process.memory_info().rss / 2**20)
            per_frame.append({
                "frame": idx,
                "accept": result["accept"],
                "liveness_score": result.get("liveness_score"),
                "similarity": result.get("similarity"),
            })
    finally:
        pipeline.close()

    name = cell_name(detector_size, detect_every, exec_mode)
    summary = {
        "cell": name,
        "label": label,
        "precision": precision,
        "video": video_name,
        "frames": len(frames),
        "warmup": warmup,
        "detector_size": detector_size,
        "detect_every": detect_every,
        "exec_mode": exec_mode,
        "intra_op_num_threads": config["intra_op_num_threads"],
        "detect_ms": summarize_metric(detect_ms),
        "post_detect_ms": summarize_metric(post_ms),
        "total_ms": summarize_metric(total_ms),
        "cpu_ms_mean": float(statistics.fmean(cpu_ms)),
        "rss_mb_max": float(max(rss_mb)),
        "per_frame": per_frame,
    }

    raw_folder, _, _ = output_paths(label, precision)
    raw_folder.mkdir(parents=True, exist_ok=True)
    (raw_folder / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    tot = summary["total_ms"]
    print(f"[{label}/{precision}] {name:32s}  total p50={tot['p50']:6.2f} ms  "
          f"mean={tot['mean']:6.2f} ms  fps={1000.0 / tot['mean']:5.1f}  RSS={summary['rss_mb_max']:.0f} MB")


def compute_smoothing_stats(scores, threshold):
    """Evaluate rolling-window liveness score stability for SMOOTH_WINDOWS."""
    valid = [s for s in scores if s is not None]
    rows = []
    for window in SMOOTH_WINDOWS:
        buf = deque(maxlen=window)
        smoothed = []
        for s in valid:
            buf.append(s)
            smoothed.append(sum(buf) / len(buf))
        arr = np.asarray(smoothed, dtype=np.float64)
        rows.append({
            "window": window,
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "spoof_flips": int((arr < threshold).sum()),
            "total_frames": len(arr),
        })
    return rows


def summarize_all(label, precision):
    """Aggregate all cell JSONs for (label, precision) and write CSV + Markdown report."""
    raw_folder, csv_path, md_path = output_paths(label, precision)
    base_path = raw_folder / f"{cell_name(640, 1, 'serial')}.json"
    if not base_path.exists():
        first_json = sorted(raw_folder.glob("det*.json"))
        if not first_json:
            raise FileNotFoundError(f"No cell results found in {raw_folder}")
        base_path = first_json[0]
    base = json.loads(base_path.read_text())
    base_frames = base["per_frame"]

    rows = []
    smoothing_targets = {}
    for size in DETECTOR_SIZES:
        for every in DETECT_INTERVALS:
            for mode in EXEC_MODES:
                path = raw_folder / f"{cell_name(size, every, mode)}.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                frames = data["per_frame"]
                matched = [
                    (b, f) for b, f in zip(base_frames, frames)
                    if b["similarity"] is not None and f["similarity"] is not None
                ]
                sim_diffs = [abs(b["similarity"] - f["similarity"]) for b, f in matched]
                live_diffs = [abs(b["liveness_score"] - f["liveness_score"]) for b, f in matched]
                flips = sum(1 for b, f in zip(base_frames, frames) if b["accept"] != f["accept"])
                total_p50 = data["total_ms"]["p50"]

                rows.append({
                    "cell": data["cell"],
                    "precision": precision,
                    "detector_size": size,
                    "detect_every": every,
                    "exec_mode": mode,
                    "detect_mean_ms": round(data["detect_ms"]["mean"], 2),
                    "detect_p50_ms": round(data["detect_ms"]["p50"], 2),
                    "post_detect_p50_ms": round(data["post_detect_ms"]["p50"], 2),
                    "total_mean_ms": round(data["total_ms"]["mean"], 2),
                    "total_p50_ms": round(total_p50, 2),
                    "total_p95_ms": round(data["total_ms"]["p95"], 2),
                    "total_std_ms": round(data["total_ms"]["std"], 2),
                    "fps_mean": round(1000.0 / data["total_ms"]["mean"], 1),
                    "fps_p50": round(1000.0 / total_p50, 1),
                    "speedup_mean": round(base["total_ms"]["mean"] / data["total_ms"]["mean"], 2),
                    "faces_found": len(matched),
                    "total_frames": len(frames),
                    "sim_mean_diff": round(float(np.mean(sim_diffs)), 5) if sim_diffs else None,
                    "sim_max_diff": round(float(np.max(sim_diffs)), 5) if sim_diffs else None,
                    "live_mean_diff": round(float(np.mean(live_diffs)), 4) if live_diffs else None,
                    "decision_flips": flips,
                    "cpu_ms_mean": round(data["cpu_ms_mean"], 2),
                    "rss_mb_max": round(data["rss_mb_max"], 1),
                })

                if data["cell"] in ("det640_every1_serial", "det320_every3_serial"):
                    smoothing_targets[data["cell"]] = compute_smoothing_stats(
                        [f["liveness_score"] for f in frames], threshold=0.0,
                    )

    processed_folder.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        f"# Runtime Optimization Ablation ({label}, {precision}, {base['video']}, {base['frames']} frames)",
        "",
        f"Baseline: `{base['cell']}` (total mean = {base['total_ms']['mean']:.2f} ms, "
        f"p50 = {base['total_ms']['p50']:.2f} ms, RSS = {base['rss_mb_max']:.0f} MB, "
        f"cosine threshold = {COSINE_THRESHOLD}, liveness threshold = 0.0).",
        "",
        "| Cell | Det (px) | Every | Mode | Det mean | Post p50 | Total mean | Total p50 | Total p95 | FPS (mean) | Speedup | RSS (MB) | Sim mean diff | Live mean diff | Flips |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['cell']} | {r['detector_size']} | {r['detect_every']} | {r['exec_mode']} | "
            f"{r['detect_mean_ms']:.2f} | {r['post_detect_p50_ms']:.2f} | {r['total_mean_ms']:.2f} | "
            f"{r['total_p50_ms']:.2f} | {r['total_p95_ms']:.2f} | {r['fps_mean']:.1f} | {r['speedup_mean']:.2f}x | "
            f"{r['rss_mb_max']:.0f} | {r['sim_mean_diff']:.5f} | {r['live_mean_diff']:.4f} | {r['decision_flips']}/{r['total_frames']} |"
        )

    if smoothing_targets:
        md_lines.extend(["", "## Liveness Temporal Smoothing (jitter across sequential frames)", ""])
        for name, srows in smoothing_targets.items():
            md_lines.extend([
                f"### {name}",
                "",
                "| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |",
                "|---:|---:|---:|---:|---:|---:|",
            ])
            for sr in srows:
                md_lines.append(
                    f"| {sr['window']} | {sr['mean']:.4f} | {sr['std']:.4f} | "
                    f"{sr['min']:.4f} | {sr['max']:.4f} | {sr['spoof_flips']}/{sr['total_frames']} |"
                )
            md_lines.append("")

    md_path.write_text("\n".join(md_lines))
    print(f"\nWrote {csv_path}\nWrote {md_path}")


def main():
    """Orchestrate the runtime ablation grid across isolated subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="local", help="machine label (e.g. local, pi)")
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32",
                        help="model precision variant for recognizer & liveness")
    parser.add_argument("--frames", type=int, default=300, help="consecutive frames per cell")
    parser.add_argument("--warmup", type=int, default=30, help="warm-up iterations before timing")
    parser.add_argument("--det-sizes", nargs="+", type=int, default=DETECTOR_SIZES)
    parser.add_argument("--every", nargs="+", type=int, default=DETECT_INTERVALS)
    parser.add_argument("--modes", nargs="+", choices=EXEC_MODES, default=EXEC_MODES)
    parser.add_argument("--cell", nargs=3, metavar=("SIZE", "EVERY", "MODE"),
                        help="run a single cell (used internally)")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.cell:
        run_cell(args.label, args.precision, int(args.cell[0]), int(args.cell[1]),
                 args.cell[2], args.frames, args.warmup)
        return
    if args.summarize_only:
        summarize_all(args.label, args.precision)
        return

    for size in args.det_sizes:
        for every in args.every:
            for mode in args.modes:
                subprocess.run(
                    [sys.executable, __file__, "--label", args.label, "--precision", args.precision,
                     "--frames", str(args.frames), "--warmup", str(args.warmup),
                     "--cell", str(size), str(every), mode],
                    check=True,
                )
    summarize_all(args.label, args.precision)


if __name__ == "__main__":
    main()
