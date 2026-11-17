# Runtime Optimization Ablation (pi, int8nodw, mabel_live_bright.mp4, 100 frames)

Baseline: `det640_every1_serial` (total mean = 232.24 ms, p50 = 229.81 ms, RSS = 850 MB, cosine threshold = 0.2421, liveness threshold = 0.0).

| Cell | Det (px) | Every | Mode | Det mean | Post p50 | Total mean | Total p50 | Total p95 | FPS (mean) | Speedup | RSS (MB) | Sim mean diff | Live mean diff | Flips |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| det640_every1_serial | 640 | 1 | serial | 147.12 | 81.06 | 232.24 | 229.81 | 253.44 | 4.3 | 1.00x | 850 | 0.00000 | 0.0000 | 0/100 |
| det640_every3_serial | 640 | 3 | serial | 49.58 | 80.20 | 134.23 | 90.66 | 238.89 | 7.4 | 1.73x | 850 | 0.00529 | 0.6332 | 3/100 |
| det640_every5_serial | 640 | 5 | serial | 28.32 | 80.25 | 112.47 | 84.30 | 233.07 | 8.9 | 2.06x | 850 | 0.00837 | 0.8856 | 1/100 |
| det480_every1_serial | 480 | 1 | serial | 82.20 | 80.64 | 166.68 | 162.67 | 177.94 | 6.0 | 1.39x | 836 | 0.00755 | 1.0570 | 2/100 |
| det480_every3_serial | 480 | 3 | serial | 27.56 | 81.10 | 112.44 | 91.18 | 176.08 | 8.9 | 2.07x | 836 | 0.00825 | 1.1644 | 5/100 |
| det480_every5_serial | 480 | 5 | serial | 16.46 | 80.47 | 100.98 | 84.53 | 173.60 | 9.9 | 2.30x | 836 | 0.00982 | 1.2308 | 2/100 |
| det320_every1_serial | 320 | 1 | serial | 38.04 | 81.63 | 123.16 | 119.79 | 135.43 | 8.1 | 1.89x | 825 | 0.00687 | 1.1327 | 5/100 |
| det320_every3_serial | 320 | 3 | serial | 13.02 | 80.76 | 97.67 | 90.49 | 132.78 | 10.2 | 2.38x | 825 | 0.00835 | 1.0702 | 3/100 |
| det320_every5_serial | 320 | 5 | serial | 7.57 | 80.85 | 92.22 | 85.09 | 130.05 | 10.8 | 2.52x | 825 | 0.01043 | 1.2030 | 3/100 |

## Liveness Temporal Smoothing (jitter across sequential frames)

### det640_every1_serial

| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.7469 | 3.7294 | -2.1481 | 12.4495 | 3/100 |
| 3 | 4.6849 | 3.5777 | -0.0651 | 12.3193 | 2/100 |
| 5 | 4.6159 | 3.4761 | 0.0098 | 12.0003 | 0/100 |
| 10 | 4.4215 | 3.2292 | 0.5712 | 11.5170 | 0/100 |

### det320_every3_serial

| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.0721 | 3.7148 | -0.4882 | 11.7172 | 2/100 |
| 3 | 4.9993 | 3.6124 | 0.0325 | 11.6846 | 0/100 |
| 5 | 4.9286 | 3.5450 | 0.1367 | 11.5414 | 0/100 |
| 10 | 4.7387 | 3.3811 | 0.9276 | 11.1411 | 0/100 |
