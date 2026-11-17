# Local Runtime Optimization Ablation (mabel_live_bright.mp4, 300 sequential frames)

Baseline: `det640_every1_serial` (total mean = 18.73 ms, p50 = 17.93 ms, cosine threshold = 0.2421, liveness threshold = 0.0).

| Cell | Det (px) | Every | Mode | Det mean | Post p50 | Total mean | Total p50 | Total p95 | FPS (mean) | Speedup | Sim mean diff | Live mean diff | Flips |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| det640_every1_serial | 640 | 1 | serial | 7.52 | 10.51 | 18.73 | 17.93 | 22.59 | 53.4 | 1.00x | 0.00000 | 0.0000 | 0/300 |
| det640_every1_parallel_auto | 640 | 1 | parallel_auto | 8.07 | 7.49 | 16.26 | 15.25 | 21.55 | 61.5 | 1.15x | 0.00000 | 0.0000 | 0/300 |
| det640_every1_parallel_half | 640 | 1 | parallel_half | 7.57 | 7.28 | 15.16 | 14.82 | 17.44 | 65.9 | 1.24x | 0.00000 | 0.0000 | 0/300 |
| det640_every3_serial | 640 | 3 | serial | 2.44 | 10.35 | 13.36 | 11.04 | 20.20 | 74.8 | 1.40x | 0.00818 | 0.7722 | 24/300 |
| det640_every3_parallel_auto | 640 | 3 | parallel_auto | 2.48 | 6.82 | 9.81 | 7.30 | 16.65 | 102.0 | 1.91x | 0.00818 | 0.7722 | 24/300 |
| det640_every3_parallel_half | 640 | 3 | parallel_half | 2.58 | 7.31 | 10.29 | 8.37 | 17.09 | 97.2 | 1.82x | 0.00818 | 0.7722 | 24/300 |
| det640_every5_serial | 640 | 5 | serial | 1.47 | 10.45 | 12.41 | 10.58 | 18.45 | 80.6 | 1.51x | 0.01566 | 1.0298 | 36/300 |
| det640_every5_parallel_auto | 640 | 5 | parallel_auto | 1.55 | 6.85 | 8.97 | 7.05 | 15.54 | 111.5 | 2.09x | 0.01566 | 1.0298 | 36/300 |
| det640_every5_parallel_half | 640 | 5 | parallel_half | 1.49 | 7.15 | 8.91 | 7.26 | 16.00 | 112.2 | 2.10x | 0.01566 | 1.0298 | 36/300 |
| det480_every1_serial | 480 | 1 | serial | 4.60 | 10.61 | 15.68 | 15.16 | 17.67 | 63.8 | 1.20x | 0.01063 | 1.0381 | 36/300 |
| det480_every1_parallel_auto | 480 | 1 | parallel_auto | 4.93 | 7.31 | 12.82 | 12.13 | 16.87 | 78.0 | 1.46x | 0.01063 | 1.0381 | 36/300 |
| det480_every1_parallel_half | 480 | 1 | parallel_half | 4.95 | 7.56 | 12.87 | 12.46 | 15.46 | 77.7 | 1.46x | 0.01063 | 1.0381 | 36/300 |
| det480_every3_serial | 480 | 3 | serial | 1.58 | 10.70 | 12.73 | 11.44 | 17.32 | 78.6 | 1.47x | 0.01207 | 1.2019 | 39/300 |
| det480_every3_parallel_auto | 480 | 3 | parallel_auto | 1.59 | 6.99 | 9.10 | 7.68 | 13.96 | 109.9 | 2.06x | 0.01207 | 1.2019 | 39/300 |
| det480_every3_parallel_half | 480 | 3 | parallel_half | 1.56 | 7.09 | 8.93 | 7.64 | 13.26 | 112.0 | 2.10x | 0.01207 | 1.2019 | 39/300 |
| det480_every5_serial | 480 | 5 | serial | 0.95 | 10.55 | 12.09 | 11.13 | 17.02 | 82.7 | 1.55x | 0.01834 | 1.3202 | 36/300 |
| det480_every5_parallel_auto | 480 | 5 | parallel_auto | 0.93 | 6.80 | 8.22 | 6.98 | 12.33 | 121.7 | 2.28x | 0.01834 | 1.3202 | 36/300 |
| det480_every5_parallel_half | 480 | 5 | parallel_half | 0.96 | 7.27 | 8.55 | 7.49 | 13.19 | 116.9 | 2.19x | 0.01834 | 1.3202 | 36/300 |
| det320_every1_serial | 320 | 1 | serial | 2.60 | 10.47 | 13.45 | 13.09 | 15.32 | 74.3 | 1.39x | 0.01503 | 1.2695 | 34/300 |
| det320_every1_parallel_auto | 320 | 1 | parallel_auto | 2.68 | 6.91 | 10.06 | 9.70 | 11.91 | 99.4 | 1.86x | 0.01503 | 1.2695 | 34/300 |
| det320_every1_parallel_half | 320 | 1 | parallel_half | 2.80 | 7.64 | 10.66 | 10.42 | 12.87 | 93.8 | 1.76x | 0.01503 | 1.2695 | 34/300 |
| det320_every3_serial | 320 | 3 | serial | 0.85 | 10.13 | 11.51 | 10.87 | 14.88 | 86.9 | 1.63x | 0.01645 | 1.3530 | 43/300 |
| det320_every3_parallel_auto | 320 | 3 | parallel_auto | 0.91 | 6.94 | 8.31 | 8.07 | 11.77 | 120.3 | 2.25x | 0.01645 | 1.3530 | 43/300 |
| det320_every3_parallel_half | 320 | 3 | parallel_half | 0.90 | 6.96 | 8.14 | 7.51 | 10.98 | 122.8 | 2.30x | 0.01645 | 1.3530 | 43/300 |
| det320_every5_serial | 320 | 5 | serial | 0.52 | 10.37 | 11.47 | 10.68 | 14.54 | 87.2 | 1.63x | 0.02184 | 1.3137 | 37/300 |
| det320_every5_parallel_auto | 320 | 5 | parallel_auto | 0.53 | 6.69 | 7.66 | 6.84 | 10.24 | 130.6 | 2.45x | 0.02184 | 1.3137 | 37/300 |
| det320_every5_parallel_half | 320 | 5 | parallel_half | 0.54 | 6.98 | 7.89 | 7.24 | 10.94 | 126.8 | 2.37x | 0.02184 | 1.3137 | 37/300 |

## Liveness Temporal Smoothing (jitter across sequential frames)

### det640_every1_serial

| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.9959 | 4.3134 | -6.4530 | 12.6441 | 133/300 |
| 3 | 2.0245 | 4.2357 | -6.1816 | 12.4339 | 134/300 |
| 5 | 2.0558 | 4.1958 | -5.5018 | 12.2670 | 135/300 |
| 10 | 2.1382 | 4.0908 | -4.5048 | 11.9082 | 139/300 |

### det320_every3_serial

| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.1798 | 4.4764 | -5.4696 | 12.2442 | 130/300 |
| 3 | 2.2025 | 4.3959 | -4.8646 | 11.8945 | 139/300 |
| 5 | 2.2284 | 4.3575 | -4.2273 | 11.8063 | 141/300 |
| 10 | 2.2938 | 4.2679 | -3.5497 | 11.5546 | 138/300 |
