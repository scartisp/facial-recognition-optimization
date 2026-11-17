# Runtime Optimization Ablation (pi, int8pc, mabel_live_bright.mp4, 100 frames)

Baseline: `det640_every1_serial` (total mean = 255.47 ms, p50 = 251.15 ms, RSS = 840 MB, cosine threshold = 0.2421, liveness threshold = 0.0).

| Cell | Det (px) | Every | Mode | Det mean | Post p50 | Total mean | Total p50 | Total p95 | FPS (mean) | Speedup | RSS (MB) | Sim mean diff | Live mean diff | Flips |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| det640_every1_serial | 640 | 1 | serial | 145.02 | 105.88 | 255.47 | 251.15 | 274.23 | 3.9 | 1.00x | 840 | 0.00000 | 0.0000 | 0/100 |
| det640_every3_serial | 640 | 3 | serial | 48.74 | 106.30 | 159.01 | 117.07 | 264.94 | 6.3 | 1.61x | 840 | 0.00610 | 0.8519 | 4/100 |
| det640_every5_serial | 640 | 5 | serial | 29.35 | 105.00 | 138.96 | 110.71 | 264.58 | 7.2 | 1.84x | 840 | 0.00889 | 0.8915 | 2/100 |
| det480_every1_serial | 480 | 1 | serial | 83.22 | 107.06 | 193.57 | 189.30 | 203.69 | 5.2 | 1.32x | 828 | 0.01059 | 1.0892 | 5/100 |
| det480_every3_serial | 480 | 3 | serial | 28.41 | 105.58 | 137.84 | 115.45 | 202.83 | 7.3 | 1.85x | 828 | 0.01280 | 1.2010 | 4/100 |
| det480_every5_serial | 480 | 5 | serial | 16.72 | 105.86 | 126.22 | 108.43 | 195.68 | 7.9 | 2.02x | 828 | 0.01593 | 1.1283 | 4/100 |
| det320_every1_serial | 320 | 1 | serial | 38.17 | 106.18 | 147.94 | 144.52 | 159.77 | 6.8 | 1.73x | 821 | 0.01084 | 1.2069 | 5/100 |
| det320_every3_serial | 320 | 3 | serial | 12.95 | 105.82 | 122.42 | 115.18 | 158.31 | 8.2 | 2.09x | 821 | 0.01070 | 1.2967 | 5/100 |
| det320_every5_serial | 320 | 5 | serial | 7.75 | 105.38 | 117.07 | 108.06 | 151.75 | 8.5 | 2.18x | 821 | 0.01371 | 1.3289 | 4/100 |

## Liveness Temporal Smoothing (jitter across sequential frames)

### det640_every1_serial

| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.0989 | 3.8962 | -1.8552 | 12.4495 | 4/100 |
| 3 | 5.0334 | 3.7225 | 0.2929 | 12.4495 | 0/100 |
| 5 | 4.9648 | 3.6117 | 0.2148 | 12.4007 | 0/100 |
| 10 | 4.7728 | 3.3625 | 0.8593 | 11.8978 | 0/100 |

### det320_every3_serial

| Window | Mean logit | Std (jitter) | Min | Max | Spoof flips (<0.0) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.0989 | 3.8256 | -0.5859 | 12.4495 | 3/100 |
| 3 | 5.0110 | 3.7133 | -0.1627 | 11.7660 | 1/100 |
| 5 | 4.9317 | 3.6426 | 0.0586 | 11.2485 | 0/100 |
| 10 | 4.7384 | 3.4828 | 0.6542 | 10.9946 | 0/100 |
