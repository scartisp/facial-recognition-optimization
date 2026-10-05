"""Figures for the quantization ablation (reads results/processed and results/raw).

    python eval/plots.py            # accuracy figures
    python eval/plots.py --bench pc # also the latency figure for that benchmark label

Writes PNGs to results/figures/.
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
raw = version1_folder / "results" / "raw"
figures = version1_folder / "results" / "figures"

VARIANTS = ["fp32", "int8pt", "int8pc"]
LABEL = {"fp32": "FP32", "int8pt": "INT8 per-tensor", "int8pc": "INT8 per-channel"}
# Reference palette, categorical slots 1-3 (validated all-pairs for three series)
COLOR = {"fp32": "#2a78d6", "int8pt": "#eb6834", "int8pc": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False,
})


def read_csv(path):
    with open(path) as file:
        return list(csv.DictReader(file))


def save(fig, name):
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figures / name}")


def plot_roc():
    rows = read_csv(processed / "roc_lfw.csv")
    verification = {r["recognizer"]: r for r in read_csv(processed / "verification_lfw.csv")}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for v in VARIANTS:
        pts = [(float(r["far"]), float(r["tar"])) for r in rows if r["recognizer"] == v and float(r["far"]) > 0]
        far, tar = zip(*pts)
        eer = 100 * float(verification[v]["eer"])
        ax.plot(far, tar, color=COLOR[v], linewidth=2, label=f"{LABEL[v]} (EER {eer:.2f}%)")
    ax.axvline(0.001, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.00105, 0.952, "FAR = 0.1%", color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 1)
    ax.set_ylim(0.95, 1.001)
    ax.set_xlabel("False accept rate (log scale)")
    ax.set_ylabel("True accept rate")
    ax.set_title("Face verification on LFW: FP32 vs INT8 recognizer", loc="left", fontsize=11)
    ax.legend(loc="lower right")
    save(fig, "roc_lfw.png")


def plot_liveness():
    rows = [r for r in read_csv(processed / "liveness.csv") if r["group"] == "all"]
    datasets = [("own (test split)", "Team videos"), ("msu", "MSU-MFSD subset"), ("kaggle", "Kaggle sample")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    width = 0.26
    for ax, (key, title) in zip(axes, datasets):
        for j, v in enumerate(VARIANTS):
            r = next(r for r in rows if r["dataset"] == key and r["liveness"] == v)
            values = [100 * float(r["apcer"]), 100 * float(r["bpcer"])]
            x = np.arange(2) + (j - 1) * (width + 0.02)
            bars = ax.bar(x, values, width, color=COLOR[v], label=LABEL[v], edgecolor=SURFACE, linewidth=2)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}", ha="center",
                        va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(range(2), ["Attacks accepted\n(APCER)", "Live faces rejected\n(BPCER)"])
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Error rate (%)")
    axes[0].set_ylim(0, 75)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Liveness errors at the default threshold: INT8 accepts more attacks", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "liveness_errors.png")


def plot_score_shift():
    rows = read_csv(raw / "liveness_scores.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    attacks = [r for r in rows if r["kind"] != "live"]
    lives = [r for r in rows if r["kind"] == "live"]
    for group, marker, name in ((lives, "o", "live faces"), (attacks, "^", "attacks")):
        fp32 = np.array([float(r["score_fp32"]) for r in group])
        int8 = np.array([float(r["score_int8pc"]) for r in group])
        ax.scatter(fp32, int8, s=14, marker=marker, alpha=0.55,
                   color=COLOR["fp32"] if name == "live faces" else COLOR["int8pt"], label=name, linewidths=0)
    lim = [-15, 20]
    ax.plot(lim, lim, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.axhline(0, color=GRID, linewidth=1.5)
    ax.axvline(0, color=GRID, linewidth=1.5)
    ax.text(12.5, 14.5, "y = x", color=INK_2, fontsize=9)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("FP32 liveness score (logit real - spoof)")
    ax.set_ylabel("INT8 per-channel liveness score")
    ax.set_title("INT8 shifts liveness scores toward \"real\" (all three test sets)", loc="left", fontsize=11)
    ax.legend(loc="upper left")
    save(fig, "liveness_score_shift.png")


def plot_latency(label):
    """Model-only latency from bench.py --isolated (single thread), the cleanest quantization comparison."""
    rows = [r for r in read_csv(processed / f"bench_isolated_{label}.csv") if r["threads"] == "1"]
    host = json.loads((raw / "bench" / label / "isolated_host.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (role, title) in zip(axes, (("recognizer", "Recognizer (MobileFaceNet)"),
                                        ("liveness", "Liveness (MiniFASNetV2-SE)"))):
        for j, v in enumerate(VARIANTS):
            s = next(r for r in rows if r["model"] == role and r["variant"] == v)
            p50, p95 = float(s["p50"]), float(s["p95"])
            ax.bar(j, p50, 0.6, color=COLOR[v], edgecolor=SURFACE, linewidth=2)
            ax.plot([j, j], [p50, p95], color=INK_2, linewidth=1.5)
            ax.plot([j - 0.08, j + 0.08], [p95, p95], color=INK_2, linewidth=1.5)
            ax.text(j + 0.33, p50, f"p50 {p50:.1f}\np95 {p95:.1f}", va="center", fontsize=8, color=INK_2)
        ax.set_xticks(range(3), [LABEL[v] for v in VARIANTS])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.set_ylabel("ms per inference")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Model-only latency on '{label}' ({host['machine']}, 1 thread, {rows[0]['runs']} runs, "
                 "warm-up excluded; bar = p50, whisker = p95)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, f"latency_{label}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", help="benchmark label to plot, e.g. pc or pi")
    args = parser.parse_args()
    plot_roc()
    plot_liveness()
    plot_score_shift()
    if args.bench:
        plot_latency(args.bench)


if __name__ == "__main__":
    main()
