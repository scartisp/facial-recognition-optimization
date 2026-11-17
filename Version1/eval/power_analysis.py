"""Power per config from one bench/power_run.py run: meter samples joined with phase markers.

    python eval/power_analysis.py results/raw/power/pi/<run_id>

For each sustained run (trimmed by --trim s at both ends) it reports mean power with spread,
the idle baseline, power above idle, and energy per authentication (frame):
    energy_mJ        = mean W / fps * 1000           whole-board energy per frame
    active_energy_mJ = (mean W - idle W) / fps * 1000  the part attributable to the workload
The idle baseline is the initial idle phase minus its first --settle s (the ondemand governor
holds 1.8 GHz for ~10 s after the SSH start-up). Idle on the Pi is not flat: with Wi-Fi power
saving on it steps between ~1.6 and ~1.9 W independent of CPU load, so the min-max of 10 s
means within the baseline is reported as its uncertainty. The same baseline is subtracted from
every config, so differences between configs do not depend on it. The last 10 s of the idle
just before each config are reported too, as a drift check.

Phase times come from the Pi's clock (t_pi) when it agrees with the PC's receive time (t_pc)
to within 0.5 s, else from t_pc. The meter is stamped by the PC clock.

A single run writes only into its own folder: <run>/power_summary.csv and <run>/power_trace.png
(whole run with phases shaded, to check alignment). Reported numbers come from --combine, which
averages runs (e.g. forward + reverse order) into results/processed/power_<label>[_<name>].csv:
    python eval/power_analysis.py --combine <run1> <run2>                  # 9-config grid
    python eval/power_analysis.py --combine <run1> <run2> --name nodw      # a separate experiment
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

version1_folder = Path(__file__).resolve().parents[1]
processed = version1_folder / "results" / "processed"
SPS = 100
PHASE_COLOR = {"load": "#c9c7c1", "warmup": "#eb6834", "sustained": "#2a78d6", "idle": None}


def load_power(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data["t_unix"], data["power_W"], data["voltage_V"]


def load_phases(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    offsets = np.array([float(r["t_pi"]) - float(r["t_pc"]) for r in rows])
    offset = float(np.median(offsets))
    use_pi = abs(offset) < 0.5
    key = "t_pi" if use_pi else "t_pc"
    # Pair each start with the next end of the same (cell, phase)
    intervals, open_ = [], {}
    for r in rows:
        k = (r["cell"], r["phase"])
        if r["event"] == "start":
            open_[k] = (float(r[key]), json.loads(r["extra"] or "{}"))
        elif k in open_:
            t0, extra0 = open_.pop(k)
            intervals.append({"cell": r["cell"], "phase": r["phase"], "t0": t0, "t1": float(r[key]),
                              "start": extra0, "end": json.loads(r["extra"] or "{}")})
    clock = {"median_t_pi_minus_t_pc_s": round(offset, 4),
             "max_abs_s": round(float(np.abs(offsets).max()), 4), "used": key}
    return intervals, clock


def window(t, w, v, t0, t1):
    sel = (t >= t0) & (t < t1)
    return w[sel], v[sel], (t1 - t0) * SPS


COMBINE_COLUMNS = ["fps", "latency_p50_ms", "latency_p95_ms", "power_mean_W", "idle_W", "active_W",
                   "energy_mJ_per_frame", "active_energy_mJ_per_frame", "cpu_temp_end_c", "rss_mb_max",
                   "cpu_ms_per_frame"]


def combine(runs, name=None):
    """Average several analysed runs (e.g. forward and reverse order) per config, with the
    min-max across runs, into results/processed/power_<label>.csv."""
    per_run = [{r["cell"]: r for r in csv.DictReader(open(Path(run) / "power_summary.csv"))} for run in runs]
    label = json.loads((Path(runs[0]) / "run.json").read_text()).get("label", Path(runs[0]).parent.name)
    rows = []
    for cell in per_run[0]:
        row = {"cell": cell, "runs": len(per_run), "run_ids": " ".join(Path(r).name for r in runs),
               "throttled_any": any(p[cell]["throttled_end"] not in ("", "0x0") for p in per_run)}
        for col in COMBINE_COLUMNS:
            values = [float(p[cell][col]) for p in per_run]
            row[col] = round(float(np.mean(values)), 4)
            row[f"{col}_min"], row[f"{col}_max"] = round(min(values), 4), round(max(values), 4)
        rows.append(row)
    processed.mkdir(parents=True, exist_ok=True)
    out = processed / (f"power_{label}_{name}.csv" if name else f"power_{label}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'cell':28s} {'fps':>6s} {'mean W':>14s} {'mJ/frame':>18s}   (mean [min-max] over {len(runs)} runs)")
    for r in rows:
        print(f"{r['cell']:28s} {r['fps']:6.2f} {r['power_mean_W']:5.2f} [{r['power_mean_W_min']:.2f}-"
              f"{r['power_mean_W_max']:.2f}] {r['energy_mJ_per_frame']:7.1f} [{r['energy_mJ_per_frame_min']:.0f}-"
              f"{r['energy_mJ_per_frame_max']:.0f}]")
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+", help="results/raw/power/<label>/<run_id> (several with --combine)")
    ap.add_argument("--combine", action="store_true",
                    help="average already-analysed runs into results/processed/power_<label>.csv")
    ap.add_argument("--name", help="with --combine: write power_<label>_<name>.csv instead")
    ap.add_argument("--trim", type=float, default=2.0, help="seconds dropped at each end of a sustained run")
    ap.add_argument("--settle", type=float, default=15.0, help="seconds dropped at the start of the baseline idle")
    args = ap.parse_args()
    if args.combine:
        combine(args.run, args.name)
        return
    run = Path(args.run[0])
    run_info = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    label = run_info.get("label", run.parent.name)

    t, w, v = load_power(run / "power.csv")
    intervals, clock = load_phases(run / "phases.csv")
    results = [json.loads(line) for line in (run / "cells.jsonl").read_text().splitlines() if line.strip()]

    base = next(i for i in intervals if i["cell"] == "baseline" and i["phase"] == "idle")
    base_w, _, _ = window(t, w, v, base["t0"] + args.settle, base["t1"])
    if base_w.size < SPS * 10:
        raise SystemExit("Baseline idle has under 10 s of meter samples; rerun with a longer --idle.")
    idle_w = float(base_w.mean())
    blocks = base_w[: base_w.size // (10 * SPS) * 10 * SPS].reshape(-1, 10 * SPS).mean(axis=1)
    idle_lo, idle_hi = float(blocks.min()), float(blocks.max())

    rows, result_iter = [], iter(results)
    previous_idle = base
    for interval in intervals:
        if interval["phase"] == "idle":
            previous_idle = interval
            continue
        if interval["phase"] != "sustained":
            continue
        result = next(result_iter)
        assert result["cell"] == interval["cell"], (result["cell"], interval["cell"])
        sw, sv, expected = window(t, w, v, interval["t0"] + args.trim, interval["t1"] - args.trim)
        pre_w, _, _ = window(t, w, v, previous_idle["t1"] - 10, previous_idle["t1"])
        mean_w = float(sw.mean())
        fps = result["fps"]
        rows.append({
            "cell": result["cell"],
            "seconds": round(result["seconds"], 2), "frames": result["frames"], "fps": round(fps, 3),
            "latency_p50_ms": round(result["total_ms"]["p50"], 2),
            "latency_p95_ms": round(result["total_ms"]["p95"], 2),
            "power_mean_W": round(mean_w, 4), "power_std_W": round(float(sw.std(ddof=1)), 4),
            "power_p5_W": round(float(np.percentile(sw, 5)), 4),
            "power_p95_W": round(float(np.percentile(sw, 95)), 4),
            "idle_W": round(idle_w, 4), "idle_10s_min_W": round(idle_lo, 4),
            "idle_10s_max_W": round(idle_hi, 4), "idle_before_W": round(float(pre_w.mean()), 4),
            "active_W": round(mean_w - idle_w, 4),
            "energy_mJ_per_frame": round(mean_w / fps * 1000, 2),
            "active_energy_mJ_per_frame": round((mean_w - idle_w) / fps * 1000, 2),
            "voltage_min_V": round(float(sv.min()), 3),
            "sample_coverage": round(sw.size / expected, 3),
            "cpu_temp_start_c": result["cpu_temp_c"]["start"], "cpu_temp_end_c": result["cpu_temp_c"]["end"],
            "cpu_freq_min_mhz": (result.get("cpu_freq_mhz") or {}).get("min"),
            "throttled_end": result["throttled"]["end"],
            "rss_mb_max": round(result["rss_mb"]["max"], 1),
            "cpu_ms_per_frame": round(result["cpu_ms_per_frame"], 2),
        })

    summary = run / "power_summary.csv"
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (run / "clock.json").write_text(json.dumps(clock, indent=2) + "\n")

    print(f"Run {run.name}: idle baseline {idle_w:.3f} W (10 s means {idle_lo:.3f}-{idle_hi:.3f}); phase clock {clock}")
    print(f"{'cell':28s} {'fps':>6s} {'mean W':>7s} {'±std':>6s} {'active W':>8s} {'mJ/frame':>8s} "
          f"{'active mJ':>9s} {'temp end':>8s} {'thr':>4s} {'cov':>5s}")
    for r in rows:
        print(f"{r['cell']:28s} {r['fps']:6.2f} {r['power_mean_W']:7.3f} {r['power_std_W']:6.3f} "
              f"{r['active_W']:8.3f} {r['energy_mJ_per_frame']:8.1f} {r['active_energy_mJ_per_frame']:9.1f} "
              f"{r['cpu_temp_end_c'] or 0:8.1f} {r['throttled_end'] or '':>4s} {r['sample_coverage']:5.2f}")
    warnings = [r["cell"] for r in rows if r["sample_coverage"] < 0.95]
    if warnings:
        print(f"WARNING: meter samples missing in {warnings}; treat those rows as invalid.")
    if any(r["throttled_end"] not in (None, "0x0") for r in rows):
        print("WARNING: the Pi reported throttling or under-voltage (throttled != 0x0).")

    # Whole-run trace with the phases shaded: a visual check that markers line up with the load
    fig, ax = plt.subplots(figsize=(12, 3.6))
    t_zero = t[0]
    kernel = np.ones(SPS // 2) / (SPS // 2)  # 0.5 s moving average, for readability only
    ax.plot(t - t_zero, w, color="#d8d6d0", lw=0.4, label="100 Hz samples")
    ax.plot(t[: len(t) - len(kernel) + 1] - t_zero + 0.25, np.convolve(w, kernel, "valid"),
            color="#0b0b0b", lw=0.8, label="0.5 s mean")
    for i in intervals:
        color = PHASE_COLOR.get(i["phase"])
        if color:
            ax.axvspan(i["t0"] - t_zero, i["t1"] - t_zero, color=color, alpha=0.18, lw=0)
    ax.axhline(idle_w, color="#1baf7a", lw=0.8, ls="--", label=f"idle {idle_w:.2f} W")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in ("#c9c7c1", "#eb6834", "#2a78d6")]
    ax.legend(ax.get_legend_handles_labels()[0] + handles,
              ax.get_legend_handles_labels()[1] + ["load", "warm-up", "sustained"],
              loc="upper right", fontsize=7, ncol=6, frameon=False)
    ax.set_xlabel("seconds since logging started")
    ax.set_ylabel("board power (W)")
    ax.set_title(f"Pi power during power run {run.name}", fontsize=10, loc="left")
    ax.set_xlim(0, t[-1] - t_zero)
    fig.tight_layout()
    fig.savefig(run / "power_trace.png", dpi=150)
    print(f"Wrote {summary} and {run / 'power_trace.png'}")


if __name__ == "__main__":
    main()
