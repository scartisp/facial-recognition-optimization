#!/usr/bin/env bash
# Everything we ran on the PC, run once on the Pi, in order.
# Needs (copied from the PC): ~/.insightface/models/buffalo_sc, models/int8/, private/cache/,
# private/mabel_live_bright.webm. Run from the Version1 folder with the venv active:
#     bash run_pi.sh
set -e

echo "== 1/3 model-only speed =="
python bench/bench.py --label pi --isolated

echo "== 2/3 full-pipeline speed, 9 configs =="
python bench/bench.py --label pi

echo "== 3/3 accuracy =="
python eval/run_eval.py --label pi

# The PC-vs-Pi comparison (eval/compare_runs.py) runs on the PC afterwards, once both machines
# have evaluated the same cache.
echo "Done. Results: results/processed/pi/summary.md, results/processed/bench_pi.csv,"
echo "results/processed/bench_isolated_pi.csv, results/raw/pi/, results/raw/bench/pi/"
