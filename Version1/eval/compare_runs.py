"""Check that two machines produce the same model outputs on the same cached faces.

    python eval/compare_runs.py --a pc --b pi

Reads the raw scores written by run_eval.py for each label (label "pc" = results/raw,
any other label = results/raw/<label>) and reports, per model variant, the largest score
difference and how many accept/reject decisions differ. ONNX Runtime's INT8 kernels differ
between x86 and ARM, so INT8 outputs may not match exactly; FP32 should match closely.

Writes results/processed/compare_<a>_vs_<b>.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

version1_folder = Path(__file__).resolve().parents[1]
raw_root = version1_folder / "results" / "raw"
processed_root = version1_folder / "results" / "processed"

VARIANTS = ["fp32", "int8pt", "int8pc"]
LIVENESS_THRESHOLD = 0.0  # logit(0.5), the facenox default used by run_eval


def raw_folder(label):
    return raw_root if label == "pc" else raw_root / label


def processed_folder(label):
    return processed_root if label == "pc" else processed_root / label


def read_rows(path):
    with open(path) as file:
        return list(csv.DictReader(file))


def compare(name, rows_a, rows_b, key, column, threshold):
    """Match rows by key and compare one score column."""
    b_by_key = {key(r): r for r in rows_b}
    pairs = [(float(r[column]), float(b_by_key[key(r)][column])) for r in rows_a if key(r) in b_by_key]
    if not pairs:
        raise RuntimeError(f"No matching rows for {name}")
    a, b = np.array(pairs).T
    return {"output": name, "matched": len(pairs),
            "max_abs_diff": float(np.abs(a - b).max()), "mean_abs_diff": float(np.abs(a - b).mean()),
            "decision_differences": int(((a >= threshold) != (b >= threshold)).sum()), "threshold": threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", default="pc")
    parser.add_argument("--b", default="pi")
    args = parser.parse_args()

    threshold_rows = read_rows(processed_folder(args.a) / "verification_lfw.csv")
    cosine_threshold = float(threshold_rows[0]["fp32_threshold"])

    lfw_a = read_rows(raw_folder(args.a) / "lfw_pair_scores.csv")
    lfw_b = read_rows(raw_folder(args.b) / "lfw_pair_scores.csv")
    live_a = read_rows(raw_folder(args.a) / "liveness_scores.csv")
    live_b = read_rows(raw_folder(args.b) / "liveness_scores.csv")

    results = []
    for v in VARIANTS:
        results.append({"variant": v, **compare("lfw cosine similarity", lfw_a, lfw_b, lambda r: r["pair"],
                                                f"sim_{v}", cosine_threshold)})
        results.append({"variant": v, **compare("liveness score", live_a, live_b,
                                                lambda r: (r["dataset"], r["source"], r["time_s"], r["index"]),
                                                f"score_{v}",
                                                LIVENESS_THRESHOLD)})

    processed_root.mkdir(parents=True, exist_ok=True)
    out = processed_root / f"compare_{args.a}_vs_{args.b}.csv"
    with open(out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    for r in results:
        print(f"{r['variant']:7s} {r['output']:22s} matched {r['matched']:5d}  max diff {r['max_abs_diff']:.2e}  "
              f"decisions that differ {r['decision_differences']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
