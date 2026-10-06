#!/usr/bin/env bash
# Everything we ran on the PC, run once on the Pi, in order.
# Needs (copied from the PC): ~/.insightface/models/buffalo_sc, models/int8/, private/cache/,
# private/mabel_live_bright.webm. Run from the Version1 folder with the venv active:
#     bash run_pi.sh
set -e

echo "== 1/4 model-only speed =="
python bench/bench.py --label pi --isolated

echo "== 2/4 full-pipeline speed, 9 configs =="
python bench/bench.py --label pi

echo "== 3/4 accuracy =="
python eval/run_eval.py --label pi

echo "== 4/4 do Pi outputs match the PC? =="
python eval/compare_runs.py --a pc --b pi

echo "Done. Results: results/processed/pi/summary.md, results/processed/bench_pi.csv,"
echo "results/processed/bench_isolated_pi.csv, results/processed/compare_pc_vs_pi.csv"
