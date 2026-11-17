# Local Runtime Optimization Ablation (mabel_live_bright.mp4, 100 sequential frames)

Baseline: `det640_every1_serial` (total mean = 235.00 ms, p50 = 232.77 ms, cosine threshold = 0.2421, liveness threshold = 0.0).

| Cell | Det (px) | Every | Mode | Det mean | Post p50 | Total mean | Total p50 | Total p95 | FPS (mean) | Speedup | Sim mean diff | Live mean diff | Flips |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| det640_every1_serial | 640 | 1 | serial | 143.97 | 88.07 | 235.00 | 232.77 | 250.72 | 4.3 | 1.00x | 0.00000 | 0.0000 | 0/100 |
| det640_every1_parallel_auto | 640 | 1 | parallel_auto | 148.87 | 89.66 | 240.02 | 237.68 | 260.11 | 4.2 | 0.98x | 0.00000 | 0.0000 | 0/100 |
| det640_every1_parallel_half | 640 | 1 | parallel_half | 147.23 | 92.81 | 240.78 | 238.52 | 257.30 | 4.2 | 0.98x | 0.00000 | 0.0000 | 0/100 |
| det640_every3_serial | 640 | 3 | serial | 48.85 | 86.66 | 140.12 | 98.41 | 243.54 | 7.1 | 1.68x | 0.00461 | 0.6302 | 0/100 |
| det640_every3_parallel_auto | 640 | 3 | parallel_auto | 52.58 | 91.20 | 145.31 | 97.36 | 253.54 | 6.9 | 1.62x | 0.00461 | 0.6302 | 0/100 |
| det640_every3_parallel_half | 640 | 3 | parallel_half | 51.34 | 94.07 | 146.31 | 96.21 | 248.36 | 6.8 | 1.61x | 0.00461 | 0.6302 | 0/100 |
| det640_every5_serial | 640 | 5 | serial | 29.60 | 88.17 | 121.71 | 93.33 | 243.21 | 8.2 | 1.93x | 0.00753 | 0.8102 | 0/100 |
| det640_every5_parallel_auto | 640 | 5 | parallel_auto | 30.16 | 89.86 | 122.11 | 92.94 | 246.78 | 8.2 | 1.92x | 0.00753 | 0.8102 | 0/100 |
| det640_every5_parallel_half | 640 | 5 | parallel_half | 30.15 | 92.57 | 123.35 | 93.00 | 245.50 | 8.1 | 1.91x | 0.00753 | 0.8102 | 0/100 |
| det480_every1_serial | 480 | 1 | serial | 84.30 | 89.48 | 177.44 | 173.83 | 191.03 | 5.6 | 1.32x | 0.00656 | 0.9471 | 0/100 |
| det480_every1_parallel_auto | 480 | 1 | parallel_auto | 86.18 | 89.77 | 177.95 | 176.80 | 192.57 | 5.6 | 1.32x | 0.00656 | 0.9471 | 0/100 |
| det480_every1_parallel_half | 480 | 1 | parallel_half | 84.31 | 94.31 | 179.65 | 178.52 | 187.61 | 5.6 | 1.31x | 0.00656 | 0.9471 | 0/100 |
| det480_every3_serial | 480 | 3 | serial | 28.97 | 88.12 | 121.15 | 100.02 | 187.60 | 8.3 | 1.94x | 0.00757 | 1.0420 | 0/100 |
| det480_every3_parallel_auto | 480 | 3 | parallel_auto | 29.76 | 91.08 | 122.52 | 97.59 | 189.63 | 8.2 | 1.92x | 0.00757 | 1.0420 | 0/100 |
| det480_every3_parallel_half | 480 | 3 | parallel_half | 28.84 | 93.13 | 122.74 | 94.74 | 183.40 | 8.1 | 1.91x | 0.00757 | 1.0420 | 0/100 |
| det480_every5_serial | 480 | 5 | serial | 17.30 | 89.65 | 110.74 | 92.59 | 184.91 | 9.0 | 2.12x | 0.01062 | 1.0910 | 0/100 |
| det480_every5_parallel_auto | 480 | 5 | parallel_auto | 17.34 | 91.62 | 110.38 | 94.48 | 186.06 | 9.1 | 2.13x | 0.01062 | 1.0910 | 0/100 |
| det480_every5_parallel_half | 480 | 5 | parallel_half | 17.18 | 93.59 | 111.44 | 94.40 | 181.39 | 9.0 | 2.11x | 0.01062 | 1.0910 | 0/100 |
| det320_every1_serial | 320 | 1 | serial | 39.69 | 92.06 | 134.99 | 131.49 | 151.52 | 7.4 | 1.74x | 0.01629 | 1.1926 | 0/100 |
| det320_every1_parallel_auto | 320 | 1 | parallel_auto | 43.76 | 88.75 | 135.25 | 134.49 | 151.60 | 7.4 | 1.74x | 0.01629 | 1.1926 | 0/100 |
| det320_every1_parallel_half | 320 | 1 | parallel_half | 39.93 | 93.88 | 134.29 | 133.11 | 145.90 | 7.4 | 1.75x | 0.01629 | 1.1926 | 0/100 |
| det320_every3_serial | 320 | 3 | serial | 13.31 | 89.42 | 107.10 | 101.25 | 142.90 | 9.3 | 2.19x | 0.01779 | 1.2980 | 0/100 |
| det320_every3_parallel_auto | 320 | 3 | parallel_auto | 16.09 | 89.35 | 108.05 | 97.71 | 147.46 | 9.3 | 2.17x | 0.01779 | 1.2980 | 0/100 |
| det320_every3_parallel_half | 320 | 3 | parallel_half | 13.65 | 93.09 | 107.38 | 94.60 | 137.04 | 9.3 | 2.19x | 0.01779 | 1.2980 | 0/100 |
| det320_every5_serial | 320 | 5 | serial | 7.90 | 88.79 | 100.41 | 92.67 | 139.34 | 10.0 | 2.34x | 0.02129 | 1.2500 | 0/100 |
| det320_every5_parallel_auto | 320 | 5 | parallel_auto | 9.08 | 92.04 | 102.47 | 95.39 | 149.51 | 9.8 | 2.29x | 0.02129 | 1.2500 | 0/100 |
| det320_every5_parallel_half | 320 | 5 | parallel_half | 8.77 | 93.89 | 103.71 | 95.16 | 146.31 | 9.6 | 2.27x | 0.02129 | 1.2500 | 0/100 |

## Liveness Temporal Smoothing (jitter across sequential frames)

### det640_every1_serial

| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.7409 | 2.8911 | 0.2045 | 12.8345 | 0/100 |
| 3 | 5.6805 | 2.7148 | 1.3855 | 12.2950 | 0/100 |
| 5 | 5.6243 | 2.6175 | 1.5521 | 11.8115 | 0/100 |
| 10 | 5.4679 | 2.3743 | 2.1673 | 11.4449 | 0/100 |

### det320_every3_serial

| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.7825 | 3.1733 | 1.0178 | 11.7755 | 0/100 |
| 3 | 5.7063 | 3.0640 | 1.4016 | 11.3753 | 0/100 |
| 5 | 5.6354 | 2.9906 | 1.5596 | 11.3395 | 0/100 |
| 10 | 5.4585 | 2.8313 | 2.2849 | 11.0746 | 0/100 |
