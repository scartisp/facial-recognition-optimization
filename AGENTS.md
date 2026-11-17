# AGENTS.md

## Project

**Optimizing Facial Recognition Models to Improve Power and Speed Efficiency**

A two-model facial authentication pipeline on a Raspberry Pi. We take two existing pretrained FP32 models, quantize both to INT8, and measure what changes in latency, power, memory, and accuracy. We are not training new models.

## Goal

Answer one question with measurements:

> Can we quantize both models from FP32 to INT8 for power/speed efficiency while maintaining accuracy?

"Accuracy" here is security-relevant: FAR, FRR, EER. A quantization that saves power but lets more impostors or spoofs through is a failure, not a trade-off.

Sub-questions:
1. How much do latency (p50/p95) and power draw change on the Pi CPU?
2. How do FAR/FRR/EER change, per model and for the combined pipeline?
3. Is any accuracy loss uniform, or concentrated in a condition (low light) or subgroup (where labeled data allows)?
4. (Stretch) Does a tuned runtime (ONNX Runtime + NEON, TFLite + XNNPACK) add speedup beyond quantization alone?

## Models

| Role | Repo | Architecture | Source precision |
|---|---|---|---|
| Identity verification (does the face match the enrolled user?) | `deepinsight/insightface`, `buffalo_sc` pack | MobileFaceNet (`w600k_mbf`, ArcFace loss, WebFace600K) + SCRFD-500MF detector | FP32 (ONNX) |
| Liveness (real face vs. spoof, e.g. printed photo / screen replay) | `facenox/face-antispoof-onnx` | MiniFASNetV2-SE (single model, 128x128 input, trained on CelebA-Spoof) | FP32 (ONNX, `best_model.onnx`) |

- The liveness repo also ships its own INT8 model (`best_model_quantized.onnx`). Do not use it as our INT8 result. We quantize from `best_model.onnx` ourselves; their INT8 file is at most a reference point, and their reported 98.20% accuracy is a claim, not our measurement.
- Pin the exact liveness repo commit and the `buffalo_sc` model files; record both.
- Known FP32 baseline limitation: per InsightFace's model zoo, MobileFaceNet (`buffalo_s`/`buffalo_sc`) scores far below ResNet-50 (`buffalo_l`) on MR-ALL (71.87 vs 91.25), with the largest gap on the East Asian subset (51.03 vs 74.96). This is a property of the baseline, not of quantization; keep it separate when reporting RQ3.

## Hardware

Raspberry Pi 4 Model B Rev 1.5, 8 GB RAM (Cortex-A72 at 1.8 GHz, no int8 dot-product instructions), Raspberry Pi OS 64-bit (Debian 13 / Trixie, kernel 6.18), Python 3.11.17 via uv (system Python is 3.13), ONNX Runtime 1.30.0. Headless over SSH (user `simion`, key login from the PC), hostname `simiPi`; it still boots the desktop and uses Wi-Fi, which adds idle noise. No webcam. Power meter: FNIRSI FNB58 inline on the Pi's USB-C supply, read from the PC over micro-USB at 100 samples/s (`bench/power_log.py`). Pi and PC clocks are both NTP-synced (measured agreement ~0.04 s). It reaches ~79 °C under load (soft throttle at 80 °C); check `throttled` in the bench JSON. PC results (x86 Windows) are labelled `pc`, Pi results `pi`.

Running on the Pi: `bash run_pi.sh` (or `nohup ... &` so an SSH drop does not kill it). It needs, copied from the PC: `private/cache/`, `private/lfw_pairs.json`, `models/int8/`, `~/.insightface/models/buffalo_sc/`, and `private/mabel_live_bright.webm` (the benchmark video, pinned in `bench.py`). Compare afterwards on the PC with `eval/compare_runs.py --a pc --b pi`.

## Pipeline

```
webcam frame -> face detect + align -> liveness model ----\
                                    \-> verification model -> AND -> accept / reject
```

Accept only if the face is live **and** matches the enrolled embedding (cosine similarity above threshold).

- **Detector:** SCRFD-500MF (`det_500m.onnx` from `buffalo_sc`), shared by both models, always FP32 in every configuration. It picks the largest face (LFW: the face nearest the centre).
- **Liveness input:** facenox's own `crop` + `preprocess` (loaded from the pinned repo): RGB, square crop expanded 1.5x, letterboxed to 128x128, /255, NCHW. Score = real logit - spoof logit; real if >= logit(0.5) = 0 (facenox default).
- **Recognizer input:** InsightFace 5-point `norm_crop` to 112x112, BGR->RGB, (x - 127.5) / 127.5, NCHW. These values are pinned in the config, not auto-detected: InsightFace infers normalization from the first graph nodes, which quantization changes.
- **Verification threshold:** cosine >= 0.2421, the FP32 threshold at FAR = 0.1% on LFW, kept fixed for every INT8 configuration (as a deployed system would). `configs/pipeline_fp32.json` still holds the old placeholder 0.4 for the live demo.
- **Code:** `Version1/app/pipeline.py`. Evaluation, calibration and benchmarks all reuse its `Pipeline` methods, so preprocessing cannot drift between FP32 and INT8.

## Quantization ablation

4 x 4 grid: recognizer {FP32, INT8 per-tensor, INT8 per-channel, INT8 per-channel except depthwise} x liveness {same four}, one config per cell in `configs/ablation/rec-<v>_live-<v>.json` (`fp32`, `int8pt`, `int8pc`, `int8nodw`). `int8nodw` leaves every Conv with group > 1 in FP32 (16 in the recognizer, 17 in liveness); all other convs are INT8.

- ONNX Runtime 1.30 static PTQ: QDQ, S8S8 (signed INT8 weights and activations), MinMax calibration, 128 calibration frames, seed 0. `w600k_mbf` is converted from opset 11 to 13 first (needed for per-channel; FP32 outputs verified bit-identical).
- INT8 files go to `Version1/models/int8/` (git-ignored); their SHA-256 hashes are added to `configs/models.lock.json`; the calibration manifest is `results/raw/quantization_manifest_s<seed>.json`.
- `quantize.py` reuses the manifest's exact calibration samples (the calib pool grew to 214 frames when p03 was added; the seed-0 sample is 128 frames from p01/p02 out of the original 143). Rebuilding `per_channel` this way reproduced the pinned hashes bit for bit. `--variants` builds only the named variants; `--fresh-sample` draws again.

## Data

All local and git-ignored. Results files hold scores only; team members appear as `p01` (Mabel), `p02` (Simion), `p03` (Alphanso; confirmed by cross-session embeddings), mapping in `private/pseudonyms.json`. The team's own folder names number people differently (Simion = p1): map by name, never by those numbers.

| Dataset | Location | Used for |
|---|---|---|
| Team videos `<name>_<live\|print\|screen>_<bright\|dim>.<mp4\|webm>` | `Version1/private/` | Calibration (first 4 s of every video only), liveness test and combined pipeline (rest), enrollment (first 4 s of `live_bright`) |
| Session 2 team videos (2026-11-14, same room, 1080p webcam) `<name>_<live\|print>_<bright\|dim>_<glasses\|noglasses>.mp4`, print = paper photo held up | `Version1/private/session2/` | Test only (never calibration): liveness by lighting and glasses, combined pipeline against the session-1 enrollments (a later day) |
| LFW deep-funneled, Kaggle `jessicali9530/lfw-dataset` | `Version1/data/lfw/archive.zip` | Verification (6,000 pairs) |
| MSU-MFSD first-frame subset, `github.com/sunny3/MSU-MFSD` (unofficial copy, instructor-approved for initial findings; commit pinned) | `Version1/data/msu-mfsd/` | Liveness test only |
| Kaggle `tapakah68/anti-spoofing` free 45-file sample | `Version1/data/anti-spoofing-kaggle/archive.zip` | Liveness test, combined pipeline (enroll from selfie) |

Never calibrate on test frames. Do not evaluate liveness on CelebA-Spoof (its training set).

## Reasoning

- **Two models, not one.** Verification models have no defense against a photo. Liveness is a separate model class, so a photo attack is only a meaningful quantization question if the liveness model is in the pipeline.
- **INT8 on both.** We start from both FP32 models and do the quantization ourselves, even though the liveness repo also distributes a pre-quantized INT8 file.
- **Raspberry Pi CPU, no accelerator.** An Edge TPU would force op-by-op compiler compatibility (PReLU, Transpose, partial graph fragmentation) onto a project that is about quantization, not compiler debugging. Coral is out of scope.
- **Speed is a hypothesis, not a given.** Both models are small. Cortex-A53/A72-class cores lack int8 dot-product instructions, so INT8 speedup on CPU may be modest. Power and memory may show a clearer effect. Measure; do not assume.
- **Security metrics over aggregate accuracy.** FAR/FRR/EER and TAR @ FAR=0.1%, reported per model and combined, so error compounding across the two stages stays visible.
- **Negative results are valid.** "No meaningful efficiency gain" is a reportable finding. Do not reframe it as a win.

## Metrics

| Metric | Unit | How |
|---|---|---|
| Latency | ms, p50 and p95 | end-to-end (detect, align, embed/classify), 500+ runs, warm-up excluded; plus model-only timing (`bench.py --isolated`, 1 thread and default threads) |
| Power | W, mJ/frame | FNB58 inline meter; `bench/power_run.py` (PC) runs a 60 s sustained run per config on the Pi between idle phases; mean W, idle-subtracted active W, energy per frame (whole-board and active) |
| Memory | MB | RSS during sustained run, not just at load |
| Model size | MB | on-disk, per precision |
| Accuracy | % | FAR/FMR, FRR/FNMR, EER, TAR @ FAR=0.1% |
| Liveness accuracy | % | APCER (attacks accepted as live), BPCER (live rejected), EER, AUC, decision flips vs FP32; by attack type and lighting |

Rates are reported with counts and 95% Wilson intervals; frames from one video are correlated, so say the intervals are optimistic.

End-to-end latency is currently unreliable: the detector, liveness and recognizer sessions each spin up a default ORT thread pool and contend for the CPU (std ~50 ms on the PC). Fix the thread settings before trusting it; until then report model-only timing.

Plots: ROC/DET (FP32 vs INT8), Pareto (accuracy vs latency), subgroup degradation (lighting; demographic where data permits).

## Rules for agents

- **Keep the FP32 baseline untouched and reproducible.** Every INT8 result is compared against it.
- **Change one thing at a time** (precision, runtime, calibration set). No bundled changes.
- **Preprocessing must match exactly between FP32 and INT8**: input size, channel layout (NCHW vs NHWC), normalization. Mismatches here look like quantization damage.
- **Calibration data must resemble deployment** (camera, lighting). Fix and record the calibration seed and set size.
- **Try PTQ first.** If accuracy drops unacceptably, escalate to per-channel, then QAT, and log each step.
- **Never report a single latency number.** Report p50, p95, and spread. Exclude warm-up.
- **Script every benchmark.** No manual runs. Commit raw logs (CSV/JSON) next to processed results.
- **Pin versions** (Python, PyTorch, ONNX Runtime, TFLite) and record the exact Pi model, OS, webcam, and power meter.
- **No face images in the repo.** Public benchmark datasets are referenced by version; any team-captured images stay local, consented, and undistributed.
- **Don't invent numbers.** If a result isn't measured, say so.

## Known risks

- PTQ accuracy drop on the liveness model (relies on fine texture/frequency cues). **Observed:** INT8 shifts liveness scores toward "real" by about +0.75 to +1.5 logits on every test set, so more attacks pass at the fixed threshold (MSU-MFSD APCER 47% -> 60-63%). Per-channel does not fix it; AUC barely moves, so threshold or calibration changes may recover it (untested).
- **Observed: INT8 is slower than FP32 on the x86 PC** (model-only, 1 thread: recognizer 13.8 -> 39.4 ms, liveness 5.1 -> 8.6-8.8 ms). Cause: ONNX Runtime's INT8 depthwise convolutions are slow on x86; PReLU is not quantized, so it stays FP32 with quantize/dequantize around it. Format (QDQ vs QOperator) and U8/S8 types made no difference. An unscripted test of INT8-except-depthwise gave recognizer 12.2 ms vs 14.2 ms FP32 (liveness 5.8 vs 4.9 ms); accuracy not measured. ARM kernels differ: the Pi result decides.
- Conversion pitfalls: PyTorch/ONNX to TFLite (unsupported ops, layout changes, silent output-range bugs).
- Models already fit the Pi comfortably, so efficiency gains may be small. Fallback: frame efficiency around sustained continuous inference.
- Subgroup-labeled data may be unavailable. Fallback: narrow the fairness claim to lighting.

## Write-up notes

Findings worth stating in the final report, with the evidence behind them. Add to this section; do not bury these in Status.

### Pi memory: the 2,000 MB with FP32 was a measurement artifact (resolved 2026-11-16)

- **What we saw:** on the Pi, process RSS was ~2,000 MB in every config with the FP32 recognizer and ~350 MB with an INT8 recognizer. The initial-findings deck listed it as an unexplained "memory anomaly". It was never caused by the model.
- **Cause:** `bench.load_frames()` decoded all 331 frames of the 1080p benchmark video (~6.2 MB each, ~2 GB in total) and kept only 20. The INT8-recognizer runs happened to return that freed memory to the OS; in the FP32-recognizer runs glibc could not, because freed heap memory is only released from the top of the heap and a later allocation sat above it.
- **Evidence (Pi, one fresh process per measurement):** each model alone uses ~70-85 MB of process RSS whatever its precision (FP32 recognizer 84 MB, INT8 77-82 MB). Through the benchmark setup, RSS jumps 152 -> 2,004 MB inside `load_frames()`, before any recognizer inference. Calling glibc `malloc_trim(0)` right after it drops 2,004 -> 277 MB; loading the frames before creating the models gives 347 MB.
- **Fix:** `load_frames()` decodes only the 20 frames it keeps (checked pixel-identical to the old selection, so latency and power results are unaffected). All 16 configs now run at 360-378 MB (FP32 378, INT8 360-373).
- **What to say about memory:** INT8 shrinks the model files ~4x (recognizer 12.99 -> 3.4-3.5 MB, liveness 1.82 -> 0.6-0.8 MB, ~11 MB saved in total), but these models are small next to the process (~120 MB Python and libraries, ~160-200 MB frames and detection buffers, ~25-30 MB for all three models), so RSS drops only ~2-5%. Memory is not a constraint on an 8 GB Pi; the saving would matter only on a very RAM-limited device or with much larger models.
- **Stale numbers:** RSS columns measured before the fix (power runs 20261115_202340, 20261115_212636, 20261116_192635, 20261116_194915) overstate FP32-recognizer configs by ~1.6 GB; their latency, power and energy are unaffected. `bench_pi.csv` was re-measured with the fix on 2026-11-16 (all 16 configs 357-376 MB).

### Why ONNX Runtime XNNPACK EP was not used (RQ4, 2026-11-16)

- **Not in prebuilt Linux wheels:** `onnxruntime 1.30.0` from PyPI lists `XnnpackExecutionProvider` in `get_all_providers()`, but `get_available_providers()` only includes `['AzureExecutionProvider', 'CPUExecutionProvider']`. Using XNNPACK on Linux `aarch64` requires compiling ONNX Runtime from source (`--use_xnnpack`).
- **`PRelu` graph fragmentation:** Both models are `NCHW` and rely heavily on `PRelu` (`w600k_mbf`: 49 `Conv`, 34 `PRelu`; `best_model.onnx`: 55 `Conv`, 33 `PRelu`). XNNPACK EP requires `NHWC` and does not support `PRelu`, so ORT would fall back to `CPUExecutionProvider` after nearly every `Conv` layer and insert `NCHW <-> NHWC` `Transpose` copies (~65 pairs per forward pass).
- **Hardware ceiling:** On the Pi 4's Cortex-A72 (no ARMv8.2 `SDOT`/`UDOT` INT8 dot-product instructions), `int8nodw` already achieves 45.6 ms (recognizer) and 21.8 ms (liveness) on `CPUExecutionProvider`, and detector tuning (`det320_every3`) already cuts amortized detector time to ~13 ms.

### Why `int8nodw` beats FP32 while full INT8 is slower on Cortex-A72 (2026-11-16)

- **Memory-bandwidth win on `1x1` and `Gemm` layers (98.7% of recognizer weights):** In `w600k_mbf` (3,386,944 params total), `1x1` pointwise convs hold 1,736,704 params (51.3%) and the final `Gemm` holds 1,605,632 params (47.4%)—~13 MB in FP32, which exceeds the Pi 4's 1 MB shared L2 cache. Quantizing these to INT8 cuts weight traffic from LPDDR4 RAM by ~4x (to ~3.4 MB) and runs on packed GEMM kernels (`MLAS`), speeding up the recognizer from 52.5 -> 45.6 ms (4 threads; 124.5 -> 105.2 ms on 1 thread).
- **Depthwise convs have no cache benefit and high kernel overhead:** The 16 depthwise `3x3` convs (`group > 1`) in `w600k_mbf` hold only 33,408 params (1.0%, ~130 KB FP32), which already fit in cache. Because Cortex-A72 lacks `SDOT`/`UDOT`, ORT's `ConvInteger` depthwise kernel spends more cycles unpacking INT8/zero-points and widening to INT16/INT32 for 9 MACs/pixel than native FP32 NEON (`VFMA`).
- **Eliminating 96 `Quantize`/`Dequantize` transitions around `PRelu`:** ORT runs `PRelu` in FP32. In full INT8 (`int8pc`), every inverted residual block (`Conv_pw -> PRelu -> Conv_dw -> PRelu -> Conv_pw`) converts `INT8 -> FP32 -> INT8 -> FP32 -> INT8` (200 `DequantizeLinear` + 98 `QuantizeLinear` nodes in `w600k_mbf`). Leaving depthwise convs in FP32 (`int8nodw`) fuses `PRelu -> Conv_dw -> PRelu` into a single FP32 span, cutting Q/DQ nodes to 136 `DequantizeLinear` + 66 `QuantizeLinear` (removing 96 full-tensor Q/DQ passes).
- **Why `MiniFASNetV2-SE` (liveness) gains less:** Total params are small (462,464; 75.5% `1x1`, 14.2% `MatMul`, 9.5% depthwise, mostly in the final 8x8 `/conv_6_dw` `group=512` layer), so FP32 weights (~1.8 MB) already mostly fit near cache; it also has channel-pruned irregular widths (`13, 26, 52, 77, 103, 231, 308`) that misalign with 16-lane NEON vectors, plus 33 `PRelu` and 3 `SE` (`Sigmoid`+`Mul`) blocks.

### Operator-level ceiling of `int8nodw` & remaining pipeline-level gains (2026-11-16)

- **ORT profiler breakdown (PC, 1 thread, `int8nodw`):** All INT8 matrix-multiply ops combined (`QLinearConv` + `QGemm` + `QLinearMatMul`) now take only **15.2%** of `w600k_mbf` (`10.42 ms/run`) and **15.4%** of `MiniFASNetV2-SE` (`5.17 ms/run`, slower than FP32 `4.10 ms`). The remaining ~80-85% of inference time is spent in FP32 depthwise `Conv` (38.1% rec / 31.8% live), FP32 `PRelu` (19.5% / 17.5%), and `QuantizeLinear`/`DequantizeLinear`/`Transpose` overhead around `PRelu` (22.1% / 31.1%). Without retraining to replace `PRelu`, static PTQ is at its architectural ceiling.
- **Remaining unimplemented pipeline-level wins (`Version1/app/pipeline.py`):**
  1. **Crop before `BGR -> RGB` in `liveness_crop()`:** Currently converts the full 1080p frame (`~6.2 MB`) to RGB before cropping the face bbox, and converts `float32 -> uint8 -> float32`. Cropping first is bit-identical and avoids full-frame color conversion.
  2. **`AND`-gate short-circuiting & recognition subsampling:** Running liveness first and skipping the recognizer (`~45.6 ms` on Pi) when liveness fails saves ~46 ms on spoof/rejected frames with 0 decision changes; caching the recognition embedding across `detect_every=3` frames while keeping liveness per-frame would drop average Pi latency from ~98 ms toward ~62 ms (~16 FPS).
  3. **Asymmetric precision (`rec-int8nodw_live-fp32`):** Keeps the tiny 1.8 MB liveness model in FP32 (faster than `int8nodw` and zero quantization logit drift) while quantizing the 13 MB recognizer.

## Status

Done: FP32 pipeline, INT8 PTQ of both models, 3 x 3 accuracy evaluation and latency/memory/CPU-time benchmark on both PC and Pi, PC-vs-Pi output comparison, figures. Team videos now include p03 (test only; calibration used p01/p02). Findings: `Version1/private/initial test.md` (local) and the Claude Docs write-up (older, PC-only).

Pi findings: INT8 is slower on the Pi too (model-only, 4 threads: recognizer 54.1 -> 71.3 ms, liveness 20.7 -> 24.4 ms; full pipeline 225 -> 249-258 ms, detector ~138 ms of that). Pi accuracy matches the PC (FP32 outputs within 1e-5; INT8 changes 0 LFW and 2/968 liveness decisions). Pi memory: the ~2,000 MB was a benchmark artifact, now fixed; see Write-up notes.

Session 2 (PC only so far): FP32 liveness rejects 49% of live frames (63% dim) but passes only 8% of paper photos; INT8 shifts live scores up by roughly +2 to +7 logits there, much more than the +0.75 to +1.5 on the older sets, so BPCER falls (49% -> 15-22%) and photo APCER rises (8% -> 16-20%). Recognizer: 0 impostor accepts in 938 cross-person attempts in every config; glasses barely move genuine cosine. p03's paper photo in bright light already passes FP32 liveness 50% of the time.

INT8 except depthwise (`int8nodw`, 2026-11-16): closest to FP32 of all INT8 variants. Recognizer fidelity (mean cos to FP32) 0.99 LFW, 0.98 team (full INT8 0.95-0.98); LFW EER 0.50% vs 0.44%. Liveness score shift mostly gone: MSU APCER 42.9% (FP32 46.7%, full INT8 60-63%), Kaggle 14.4% (16.2%), session 2 10.4% APCER / 30.9% BPCER (FP32 8.0% / 49.0%); decision flips vs FP32 about halved. Pi model-only (4 threads): recognizer 45.6 ms vs FP32 52.5 (-13%; full INT8 68), liveness 21.9 vs 20.9 (+4%). Pi full pipeline and power (`results/processed/power_pi_nodw.csv`, forward + reverse pass, 5 configs in the same runs): p50 221.5 ms vs FP32 222.8 (no measurable change; detector dominates); 4.68 W vs 4.75; 1042 mJ/frame [1033-1051] vs FP32 1065 [1062-1067], about -2% and the only INT8 config below FP32; full INT8 pc 1134 (+7%). Small gain: it is within about 2x the pass-to-pass spread.

Pi latency benchmark re-run (2026-11-16, fixed frame loader, 16 configs, 500 runs each, `results/processed/bench_pi.csv`): p50 FP32 234.0 ms, rec-int8nodw 224-232 ms, full-INT8 recognizer 250-257 ms; recognizer stage 54.5 ms FP32 vs 46.6 nodw vs 69-72 full INT8. The Pi ran at 71-81 °C (no cool-down between configs in this benchmark; `throttled` 0x0, 1.8 GHz throughout), so the detector took ~147 ms vs ~138 in the cooled power runs; compare configs within one run.

Pending: liveness threshold/calibration fix for the score shift, p03 glare test, calibration seeds 1 and 2.

Power (Pi, two passes, forward 20261115_202340 and reverse 20261115_212636, 60 s per config, no throttling, 1.8 GHz throughout; `results/processed/power_pi.csv` = mean with min-max): FP32 4.78 W at 4.42 fps = 1081 mJ/frame whole-board. INT8 recognizer configs draw less (4.47-4.53 W) but run slower (4.0 fps), so energy per frame is 2-4% higher (1107-1123 mJ; per-pass range 1104-1131 vs 1077-1088 for FP32 recognizer, no overlap). INT8 liveness alone: 4.69-4.71 W, 1082-1083 mJ (no change). Lower watts come from fewer busy cores (CPU-s per s 3.0 -> 2.8), not efficiency. Idle baseline 1.94-1.98 W (10 s means 1.74-2.21), so idle-subtracted energy cannot separate the configs. The FNB58's PC data link drops every so often (vanishes from USB; the Pi keeps power): replug the micro-USB and rerun; power_log.py retries 5 times first.

Layout:

```
AGENTS.md / CLAUDE.md          project rules (CLAUDE.md just imports AGENTS.md)
configs/
  models.lock.json             pinned liveness commit, MSU-MFSD commit, SHA-256 of every model (FP32 and INT8)
  pipeline_fp32.json           live demo config
  ablation/*.json              the 9 ablation configs (generated by quantize.py)
Version1/
  app/pipeline.py              shared pipeline (enroll / verify, image or webcam)
  eval/prepare_data.py         detect once, cache crops -> private/cache/
  quantize/quantize.py         INT8 models + ablation configs + lock hashes
  eval/run_eval.py             accuracy -> results/processed/summary.md
  bench/bench.py               latency/RSS/size (--label pc|pi, --isolated); --power = sustained runs with PHASE markers
  bench/power_log.py           FNB58 meter -> CSV (PC)
  bench/power_run.py           PC: meter + Pi bench --power over SSH -> results/raw/power/<label>/<run>/
  eval/power_analysis.py       power per config (per run folder); --combine [--name] -> results/processed/power_<label>[_<name>].csv
  eval/plots.py                figures -> results/figures/
  results/{raw,processed,figures}/   committed (scores and plots only)
  models/int8/, data/, private/, external/   git-ignored
```

Run order and data placement: `Version1/README.md`. Python 3.11 venv in `Version1/.venv`; exact versions in `Version1/requirements.lock.txt`.
