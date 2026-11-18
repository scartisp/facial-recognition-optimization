"""Latency, memory and model-size benchmark for the 9 ablation configs.

Run from the Version1 folder, venv active:
    python bench/bench.py --label pc          # all 9 cells, each in its own process
    python bench/bench.py --label pi --runs 500 --warmup 50

Each cell runs the full pipeline (detect -> liveness -> recognize) on a fixed set of frames
taken from one team video (private/mabel_live_bright.*, pinned), cycling through them. Warm-up runs
are excluded. Each cell gets a fresh process so memory (RSS) is not shared between cells.

Writes:
    results/raw/bench/<label>/<cell>.csv     one row per timed run
    results/raw/bench/<label>/<cell>.json    summary + host/software info
    results/processed/bench_<label>.csv      p50/p95/std per stage, RSS, model sizes

Power mode (driven from the PC by bench/power_run.py, which also logs the USB power meter):
    python bench/bench.py --label pi --power --seconds 60
An idle baseline, then each config in its own process: load, warm-up, then a sustained run
for --seconds, followed by an idle cool-down. Every phase boundary is printed as a
`PHASE {json}` line with a Unix timestamp, and each config ends with a `RESULT {json}` line.
Nothing is written on this machine; the PC records the stream.
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
BENCH_VIDEO = "mabel_live_bright"  # every earlier PC and Pi benchmark used this video


def load_frames():
    """FRAME_COUNT evenly spaced frames (with a detectable face) from one bright live team video."""
    # Pinned, so adding team videos never silently changes the benchmark input
    videos = sorted(p for p in (version1_folder / "private").glob(f"{BENCH_VIDEO}.*"))
    if not videos:
        raise FileNotFoundError(f"Need private/{BENCH_VIDEO}.<ext> to benchmark on")
    # Decode only the frames we keep. Decoding all of them (331 x 6 MB at 1080p) and then
    # dropping most left ~2 GB of freed memory that glibc could not return to the OS in some
    # configs, which is what made FP32 look like it needed 2,000 MB on the Pi.
    capture = cv2.VideoCapture(str(videos[0]))
    total = 0
    while capture.grab():
        total += 1
    capture.release()
    step = max(1, total // FRAME_COUNT)
    wanted = set(range(0, total, step)[:FRAME_COUNT])  # same frames as frames[::step][:FRAME_COUNT]
    capture = cv2.VideoCapture(str(videos[0]))
    frames = []
    for index in range(total):
        if not capture.grab():
            break
        if index in wanted:
            ok, frame = capture.retrieve()
            if ok:
                frames.append(frame)
    capture.release()
    return frames


def read_sysfs_number(path, scale):
    try:
        return int(Path(path).read_text()) / scale
    except (OSError, ValueError):
        return None


def cpu_temperature():
    """CPU temperature in C (Linux/Pi), or None elsewhere."""
    return read_sysfs_number("/sys/class/thermal/thermal_zone0/temp", 1000.0)


def cpu_frequency_mhz():
    """Current CPU0 clock in MHz (Linux/Pi), or None elsewhere. A drop mid-run means throttling."""
    return read_sysfs_number("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000.0)


def pi_throttled():
    """Raw `vcgencmd get_throttled` flags (Pi only): 0x0 means no under-voltage or thermal throttling."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
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


# INT8-except-depthwise detector (quantize/quantize_detector.py), one static file per input size
INT8_DETECTOR = {320: {"path": "Version1/models/int8/det_500m_det320_int8_per_channel_nodw_s0.onnx",
                       "lock_key": "int8/det_500m_det320_int8_per_channel_nodw_s0.onnx"}}


def load_cell(config_path, det_size=None, det_model="fp32"):
    """Pipeline for one config plus the benchmark frames it finds a face in.
    det_size overrides the detector's square input size (default: the config's, 640);
    det_model="int8nodw" swaps in the INT8 detector for that size."""
    config = json.loads(Path(config_path).read_text())
    if det_size:
        config["detector"]["input_size"] = [det_size, det_size]
    if det_model != "fp32":
        config["models"]["detector"] = dict(INT8_DETECTOR[config["detector"]["input_size"][0]])
    pl.check_pins(config)
    pipeline = pl.Pipeline(config)
    frames = load_frames()
    faces = [pipeline.detect(f) for f in frames]
    frames = [f for f, face in zip(frames, faces) if face is not None]
    if not frames:
        raise RuntimeError("No face found in the benchmark frames")
    return config, pipeline, frames


def time_frame(pipeline, frame, face=None):
    """One full pipeline pass (detect -> liveness -> recognize), timed per stage. With `face`
    (bbox, keypoints) the detector is skipped and that face reused, as in detect-every-K."""
    c0 = time.process_time()
    t0 = time.perf_counter()
    bbox, keypoints = pipeline.detect(frame) if face is None else face
    t1 = time.perf_counter()
    live_blob = pipeline.liveness_blob(pipeline.liveness_crop(frame, bbox))
    t2 = time.perf_counter()
    pipeline.liveness_scores_blob(live_blob)
    t3 = time.perf_counter()
    rec_blob = pipeline.recognizer_blob(pipeline.recognizer_crop(frame, keypoints))
    t4 = time.perf_counter()
    pipeline.embed_blob(rec_blob)
    t5 = time.perf_counter()
    c5 = time.process_time()
    return {
        "detect_ms": (t1 - t0) * 1000, "liveness_prep_ms": (t2 - t1) * 1000,
        "liveness_infer_ms": (t3 - t2) * 1000, "recog_prep_ms": (t4 - t3) * 1000,
        "recog_infer_ms": (t5 - t4) * 1000, "total_ms": (t5 - t0) * 1000,
        # CPU time of all this process's threads: an energy proxy, NOT a power measurement
        "cpu_ms": (c5 - c0) * 1000,
    }, (bbox, keypoints)


def run_cell(config_path, label, runs, warmup):
    config, pipeline, frames = load_cell(config_path)
    process = psutil.Process(os.getpid())

    temp_start, throttled_start = cpu_temperature(), pi_throttled()
    rows = []
    for run in range(warmup + runs):
        times, _ = time_frame(pipeline, frames[run % len(frames)])
        if run < warmup:
            continue
        rows.append({
            "run": run - warmup, **times,
            "rss_mb": process.memory_info().rss / 2**20,
            "cpu_temp_c": cpu_temperature(),
            "cpu_freq_mhz": cpu_frequency_mhz(),
        })
    temp_end, throttled_end = cpu_temperature(), pi_throttled()

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
        "intra_op_threads": "onnxruntime default, spinning off",
        "stages": {s: summarize([r[s] for r in rows]) for s in STAGES},
        "rss_mb": {"median": statistics.median(r["rss_mb"] for r in rows), "max": max(r["rss_mb"] for r in rows)},
        "model_size_mb": sizes,
        "cpu_ms_per_frame": {"mean": statistics.fmean(r["cpu_ms"] for r in rows),
                             "note": "process CPU time, energy proxy only; not power"},
        "cpu_temp_c": {"start": temp_start, "end": temp_end},
        "cpu_freq_mhz": ({"min": min(r["cpu_freq_mhz"] for r in rows), "max": max(r["cpu_freq_mhz"] for r in rows)}
                         if rows[0]["cpu_freq_mhz"] is not None else None),
        "throttled": {"start": throttled_start, "end": throttled_end},
        "host": host_info(),
    }
    (out_folder / f"{cell}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{cell}: total p50 {summary['stages']['total_ms']['p50']:.1f} ms, "
          f"p95 {summary['stages']['total_ms']['p95']:.1f} ms, RSS {summary['rss_mb']['max']:.0f} MB")


def mark(cell, phase, event, **extra):
    """Print one phase boundary for bench/power_run.py, timestamped with this machine's clock."""
    print("PHASE " + json.dumps({"cell": cell, "phase": phase, "event": event, "t": time.time(), **extra}),
          flush=True)


def power_cell_name(config_path, det_size=None, detect_every=1, det_model="fp32"):
    """Cell name for power runs; detector settings are appended only when used."""
    name = json.loads(Path(config_path).read_text())["precision"]
    if det_size or detect_every != 1 or det_model != "fp32":
        name += f"@det{det_size or 640}_k{detect_every}"
    if det_model != "fp32":
        name += f"_dq-{det_model}"
    return name


def run_power_cell(config_path, seconds, warmup, det_size=None, detect_every=1, det_model="fp32"):
    """Sustained run of one config for the power meter: load, warm-up, then `seconds` of
    back-to-back full-pipeline passes. Prints PHASE markers and a final RESULT line.
    det_size: detector input size override. detect_every=K: run the detector on every K-th
    frame and reuse that face in between (Mabel's --detect-every). The benchmark frames are
    ~0.55 s apart, so a reused face can be slightly off on the next frame; the work per frame
    (and so time and energy) is the same as on sequential video."""
    cell = power_cell_name(config_path, det_size, detect_every, det_model)
    process = psutil.Process(os.getpid())

    mark(cell, "load", "start")
    config, pipeline, frames = load_cell(config_path, det_size, det_model)
    mark(cell, "load", "end", rss_mb=process.memory_info().rss / 2**20)

    face = None

    def step(i):
        nonlocal face
        times, found = time_frame(pipeline, frames[i % len(frames)], None if i % detect_every == 0 else face)
        face = found
        return times

    mark(cell, "warmup", "start")
    for run in range(warmup):
        step(run)
    mark(cell, "warmup", "end")

    temp_start, throttled_start = cpu_temperature(), pi_throttled()
    totals, cpu_ms, rss, freqs = [], [], [], []
    mark(cell, "sustained", "start", cpu_temp_c=temp_start)
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < seconds:
        times = step(len(totals))
        totals.append(times["total_ms"])
        cpu_ms.append(times["cpu_ms"])
        if len(totals) % 10 == 1:  # sampled, so the bookkeeping stays out of the timed path
            rss.append(process.memory_info().rss / 2**20)
            freqs.append(cpu_frequency_mhz())
    elapsed = time.perf_counter() - t_start
    temp_end = cpu_temperature()
    mark(cell, "sustained", "end", frames=len(totals), cpu_temp_c=temp_end)
    throttled_end = pi_throttled()

    result = {
        "cell": cell, "precision": config["precision"], "det_size": config["detector"]["input_size"][0],
        "detect_every": detect_every, "det_model": det_model,
        "seconds": elapsed, "frames": len(totals), "fps": len(totals) / elapsed,
        "warmup_excluded": warmup, "total_ms": summarize(totals),
        "cpu_ms_per_frame": statistics.fmean(cpu_ms),
        "rss_mb": {"median": statistics.median(rss), "max": max(rss)},
        "cpu_freq_mhz": {"min": min(freqs), "max": max(freqs)} if freqs[0] is not None else None,
        "cpu_temp_c": {"start": temp_start, "end": temp_end},
        "throttled": {"start": throttled_start, "end": throttled_end},
        "host": host_info(),
    }
    print("RESULT " + json.dumps(result), flush=True)


def idle(cell, seconds, cool_to=None, max_seconds=None):
    """Idle phase: at least `seconds`, then (if cool_to is set) until the CPU is at or below
    cool_to C, capped at max_seconds. The PC uses idle phases as the idle power baseline."""
    mark(cell, "idle", "start", cpu_temp_c=cpu_temperature())
    start = time.time()
    time.sleep(seconds)
    while (cool_to is not None and (cpu_temperature() or 0) > cool_to
           and time.time() - start < max_seconds):
        time.sleep(2)
    mark(cell, "idle", "end", cpu_temp_c=cpu_temperature())


def run_power(label, args):
    configs = sorted(ablation_folder.glob("*.json"))
    if args.cells:
        configs = [c for c in configs if c.stem in args.cells.split(",")]
        if not configs:
            raise SystemExit(f"No ablation config matches --cells {args.cells}")
    every = [int(k) for k in str(args.detect_every).split(",")]
    det_models = str(args.det_model).split(",")
    cells = [(c, k, m) for c in configs for m in det_models for k in every]
    if args.reverse:
        cells = cells[::-1]
    print(f"POWER label={label} cells={len(cells)} seconds={args.seconds} repeats={args.repeats} "
          f"reverse={args.reverse} det_size={args.det_size or 'config'} detect_every={every} det_models={det_models}", flush=True)
    idle("baseline", args.idle)
    for repeat in range(args.repeats):
        # Alternate the order on repeats so thermal drift does not always hit the same cells
        for config_path, k, m in (cells if repeat % 2 == 0 else cells[::-1]):
            command = [sys.executable, __file__, "--label", label, "--power-cell", str(config_path),
                       "--seconds", str(args.seconds), "--warmup", str(args.warmup), "--detect-every", str(k),
                       "--det-model", m]
            if args.det_size:
                command += ["--det-size", str(args.det_size)]
            subprocess.run(command, check=True)
            idle(power_cell_name(config_path, args.det_size, k, m), args.gap, args.cool_to, args.max_cool)
    print("POWER done", flush=True)


def run_isolated(label, runs, warmup):
    """Model-only timing (session.run on a fixed preprocessed input) for each of the 6 model files,
    with 1 thread and with ONNX Runtime's default thread count. Separates the quantization effect
    from preprocessing, detection and thread scheduling noise."""
    configs = {v: json.loads((ablation_folder / f"rec-{v}_live-{v}.json").read_text())
               for v in ("fp32", "int8pt", "int8pc", "int8nodw")}
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
                options = pl.session_options()
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


def summarize_cells(label):
    """Collect the per-cell JSON summaries into results/processed/bench_<label>.csv."""
    rows = []
    for path in sorted((bench_raw / label).glob("rec-*.json")):
        s = json.loads(path.read_text())
        row = {"cell": s["cell"], "runs": s["runs"]}
        for stage in STAGES:
            for stat in ("p50", "p95", "std"):
                row[f"{stage}_{stat}"] = round(s["stages"][stage][stat], 3)
        row["rss_mb_max"] = round(s["rss_mb"]["max"], 1)
        row["cpu_ms_per_frame"] = round(s["cpu_ms_per_frame"]["mean"], 2) if "cpu_ms_per_frame" in s else ""
        row["cpu_temp_end_c"] = s["cpu_temp_c"]["end"]
        row["throttled_end"] = (s.get("throttled") or {}).get("end")
        row["recognizer_mb"] = round(s["model_size_mb"]["recognizer"], 3)
        row["liveness_mb"] = round(s["model_size_mb"]["liveness"], 3)
        rows.append(row)
    results_processed.mkdir(parents=True, exist_ok=True)
    with open(results_processed / f"bench_{label}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {results_processed / f'bench_{label}.csv'}")



def main():
    parser = argparse.ArgumentParser(description="Benchmark the 9 ablation configs")
    parser.add_argument("--isolated", action="store_true", help="model-only timing instead of the 9 cells")
    parser.add_argument("--label", required=True, help="machine label, e.g. pc or pi")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--config", help="benchmark a single config (used internally)")
    parser.add_argument("--summarize-only", action="store_true",
                        help="rebuild bench_<label>.csv from existing per-cell results without rerunning")
    power = parser.add_argument_group("power mode (normally started by bench/power_run.py on the PC)")
    power.add_argument("--power", action="store_true", help="sustained runs with PHASE markers")
    power.add_argument("--power-cell", help="one config's sustained run (used internally)")
    power.add_argument("--seconds", type=float, default=60, help="sustained run length per config")
    power.add_argument("--idle", type=float, default=60, help="idle baseline before the first config, s")
    power.add_argument("--gap", type=float, default=30, help="minimum idle between configs, s")
    power.add_argument("--cool-to", type=float, default=50, help="after the gap, idle until CPU <= this C")
    power.add_argument("--max-cool", type=float, default=300, help="cap on each idle phase, s")
    power.add_argument("--repeats", type=int, default=1, help="passes over the configs (order alternates)")
    power.add_argument("--reverse", action="store_true", help="start with the last config (order check)")
    power.add_argument("--cells", help="comma-separated subset, e.g. rec-fp32_live-fp32,rec-int8pt_live-int8pt")
    power.add_argument("--det-size", type=int, help="detector input size override, e.g. 480 or 320")
    power.add_argument("--detect-every", default="1",
                       help="run the detector every K frames; comma list in --power, e.g. 1,3,5")
    power.add_argument("--det-model", default="fp32",
                       help="detector precision: fp32 or int8nodw (needs --det-size 320); comma list in --power")
    args = parser.parse_args()

    if args.power_cell:
        run_power_cell(args.power_cell, args.seconds, args.warmup, args.det_size, int(args.detect_every),
                       args.det_model)
        return
    if args.power:
        run_power(args.label, args)
        return
    if args.config:
        run_cell(args.config, args.label, args.runs, args.warmup)
        return
    if args.summarize_only:
        summarize_cells(args.label)
        return
    if args.isolated:
        results_processed.mkdir(parents=True, exist_ok=True)
        run_isolated(args.label, args.runs, args.warmup)
        return

    for config_path in sorted(ablation_folder.glob("*.json")):
        subprocess.run([sys.executable, __file__, "--label", args.label, "--runs", str(args.runs),
                        "--warmup", str(args.warmup), "--config", str(config_path)], check=True)

    summarize_cells(args.label)


if __name__ == "__main__":
    main()
