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

### Detector settings save far more energy than quantization (Pi, 2026-11-17)

- **Setup:** 27 configs = precision {fp32, int8pc, int8nodw}, matched on both models x detector input {640, 480, 320} x detect-every K {1, 3, 5} (the detector runs every K-th frame and its face box is reused in between; Mabel's `--detect-every`). Parallel mode left out. 60 s sustained per config, one chunk per detector size, forward and reverse pass (forward 20261116_224745, 20261116_232933, 20261117_000408; reverse 20261117_003852, 20261117_011217, 20261117_014936). No throttling, no meter gaps. `results/processed/power_pi_detector.csv` = mean with min-max over the two passes.
- **Whole-board mJ/frame, mean of the two passes (K = 1 / 3 / 5):**
  - det640: fp32 1106 / 656 / 567; int8nodw 1069 / 603 / 513; int8pc 1129 / 676 / 591
  - det480: fp32 801 / 544 / 495; int8nodw 754 / 496 / 450; int8pc 846 / 592 / 542
  - det320: fp32 588 / 475 / 456; int8nodw 540 / 424 / 402; int8pc 624 / 511 / 483
- **What it shows:** the detector is the dominant cost. 640 -> 320 alone saves ~47% (fp32 1106 -> 588 mJ); 640/K1 -> 320/K5 saves ~59% for fp32. int8nodw is the lowest-energy precision at every setting (3-12% below fp32), and its lead grows as detector work shrinks, because the recognizer becomes a larger share of each frame. Full INT8 per-channel draws the fewest watts (4.3-4.5 W vs 4.8-4.9 W fp32) but runs slower, so it costs 2-10% more energy per frame than fp32 everywhere. Pass-to-pass spread is mostly under 2%, ~6% at det640/K1; only at det640/K1 do the fp32 and int8nodw ranges overlap.
- **Speed:** fps fp32 4.3 (640/K1) -> 10.6 (320/K5); int8nodw 4.4 -> 11.5; int8pc 4.0 -> 8.9. With K > 1, p50 is a frame without detection (fp32 ~85 ms, int8nodw ~79, int8pc ~105) and p95 a frame with detection (about the K = 1 latency; fp32 640 ~228, 480 ~166, 320 ~123 ms). RSS 335-375 MB, slightly lower at smaller detector inputs.
- **Accuracy at smaller detector inputs (PC, `results/processed/det480/`, `det320/`, 640 FP32 threshold kept fixed):** as many or more faces found; LFW EER the same or better; session-2 photo APCER rises a little at 320 (fp32 8.0 -> 9.3%, int8nodw 10.4 -> 12.9%, int8pc 20.1 -> 25.8%). Detect-every K was not evaluated for accuracy: liveness and recognition see a reused (stale) box for K-1 frames, which matters for security; report it as untested. The benchmark frames are ~0.55 s apart, so timing is unaffected but crop quality there would be a worst case.
- **Takeaway for the report:** quantization alone moves energy per frame by a few percent (and in the wrong direction for full INT8); reducing detector work moves it by half or more. Best measured: int8nodw + det320 + K5 at 402 mJ/frame and 11.5 fps, vs the fp32 baseline (640, K1) at 1106 mJ/frame and 4.3 fps (-64%).

### Detect-every-K costs little; calibration-tuned liveness thresholds do not fix INT8 on a new day (PC, 2026-11-17)

- **Setup:** `eval/sequential_eval.py` plays all 36 team videos in frame order at 10 processed frames/s (about the Pi's rate at det320/K5), detector every K-th processed frame with the box reused in between (as `verify_camera` does), no face = reject. Detector 640/320 x K 1/3/5 x fp32/int8pc/int8nodw. Session-1 first 4 s = calibration window (enrollment and threshold tuning only), rest = test; session 2 = test only. Liveness thresholds chosen per precision and setting on the calibration window, rules fixed before looking at test results (primary `apcer_parity`: lowest threshold whose calibration APCER is no higher than FP32's at 0; also `bpcer_parity`, `calib_eer`). FP32 keeps 0 under both parity rules (it is their reference; a first version let tie-breaking move it, corrected before reporting). Results: `results/processed/sequential/` (summary.md, liveness.csv, combined.csv, thresholds.csv). Consecutive frames are highly correlated: treat the Wilson intervals as very optimistic.
- **Detect-every-5 vs every frame (det320, default threshold, session 2):** APCER fp32 9.3 -> 10.1%, int8nodw 13.9 -> 15.0%, int8pc 24.8 -> 26.2%; BPCER unchanged or slightly lower. Combined FRR rises for INT8 (int8nodw 30.6 -> 33.8%, int8pc 15.4 -> 20.8%; fp32 49.7 -> 49.8%), from stale keypoints lowering cosine. Impostor accepts: 0/4622 in every config except one (int8pc det320 K1, 1/4622). So K = 5 costs about 1 point of APCER and a few points of FRR: small next to its ~20-25% energy saving at det320.
- **Threshold tuning (primary rule) does not restore FP32-level security on session 2.** det320/K5: int8nodw threshold +0.19 -> APCER 15.0 -> 14.4% (FP32 10.1%), BPCER 30.3 -> 32.4%; int8pc +1.30 -> APCER 26.2 -> 17.7% (BPCER 15.3 -> 26.9%). On session-1 test frames (same session and attack type as the calibration window) it nearly does: int8nodw 7.5% vs FP32 6.8%, int8pc 7.0%. Reason: the INT8 score shift depends on conditions. On the calibration window it is small (int8nodw needs only +0.2 logits), but on session 2 (later day, paper photos never seen in calibration) it is +2 to +7 logits. A threshold from one session/attack type does not carry over.
- **Not the primary rule, so not a headline:** int8pc with `bpcer_parity` (threshold about +3) matches FP32's session-2 APCER (10.0% vs 10.1%) with lower BPCER (41.6% vs 47.5%), but on session-1 test its BPCER is worse (32.9% vs 26.5%). Same conclusion: the right INT8 threshold changes with conditions.
- **Context:** FP32 itself rejects ~50% of live session-2 frames at the default threshold; the liveness model is poorly matched to our camera regardless of precision.
- **Leave-one-person-out on session 2 (2026-11-17, `analyse_lopo`, `results/processed/sequential/lopo*.{csv,md}`):** tune each precision's threshold on two people's session-2 frames, test on the third, pool the three folds (same rules; FP32 stays at 0). Session 2 is then no longer purely test data, so label these results as such. With tuning data from the same conditions, INT8 thresholds come out much higher (int8nodw +1.0 to +2.6, int8pc +2.8 to +4.6 across folds) and INT8 reaches FP32's operating point: det320/K5 APCER int8nodw 8.0%, int8pc 7.9% vs FP32 10.1%; BPCER 48.5% / 45.1% vs 47.5%; FRR 50.7% / 48.2% vs 49.8%; 0/4622 impostor accepts. det640/K1: int8nodw 5.8% / 50.4%, int8pc 5.7% / 47.7% vs FP32 7.5% / 49.5%. Descriptively (on test, not a deployable threshold), at FP32's APCER INT8's BPCER is the same or lower (det320/K5: int8nodw 47.2%, int8pc 41.4% vs FP32 47.2%); AUC 0.87 vs 0.89.
- **Conclusion on liveness quantization:** the damage is mostly a score offset, not lost separability. A threshold tuned on data from the deployment conditions recovers FP32-level security, but then INT8's lower BPCER at the default threshold disappears (it was the same offset). The offset varies with conditions (≈+0.2 logits on the calibration window, +1 to +4.6 needed on session 2) and the fold thresholds spread widely, so with three people this is a feasibility result, not a deployable calibration. Separately, FP32 itself rejects ~half of live session-2 frames; liveness score smoothing over frames is untested.

### Liveness score smoothing: a small free APCER gain for INT8 (PC, 2026-11-17)

- `analyse_smoothing` in `eval/sequential_eval.py` (`results/processed/sequential/smoothing*.{csv,md}`): the liveness decision uses the mean score of the last W processed frames (10 frames/s, causal, cleared on a missing face, as `verify_camera --smooth-liveness` does), threshold 0, W in {1, 3, 5, 10}. No extra compute.
- Session 2, W = 1 -> 10 (1 s): int8nodw 640/K1 APCER 11.3 -> 8.8%, BPCER 30.7 -> 29.9%; 320/K5 APCER 15.0 -> 13.3%, BPCER 30.3 -> 30.2%. int8pc 320/K5 26.2 -> 20.7%. FP32 barely changes (320/K5 10.1 -> 9.4%; BPCER 47.5 -> 48.3%; 640/K1 BPCER rises 49.5 -> 53.5%). Session-1 test: every precision gains 1.0-3.5 points of APCER. It narrows the INT8 gap but does not close it (int8nodw 320/K5 13.3% vs FP32 10.1% unsmoothed).
- Caveat: these are per-frame rates. A real attacker needs one accepted frame, and smoothing mainly removes isolated spikes, so a per-attempt metric (any frame accepted within N s) would be the next thing to report; not measured.

### INT8 detector: 29% faster on the Pi, but misses faces in dim light (2026-11-17)

- `quantize/quantize_detector.py`: SCRFD-500MF fixed to a static 320x320 input, then the same PTQ as int8nodw (QDQ, S8S8, MinMax, per-channel, the 20 Conv with group > 1 left FP32), calibrated on the same 128 seed-0 frames as the other models (full frames, letterboxed as SCRFD does). File 2.52 -> 0.81 MB; hash in `configs/models.lock.json`; manifest `results/raw/quantization_manifest_detector_det320_s0.json`. `bench.py` / `power_run.py --det-model fp32,int8nodw`, `sequential_eval.py --det-model int8nodw`.
- **Pi speed (detector only, 320, incl. pre/post-processing, 100 runs x 2):** FP32 p50 38.1 / 37.7 ms, INT8 26.8 / 26.9 ms (-29%).
- **Pi power (`results/processed/power_pi_detq.csv`, runs 20261117_231157 forward + 20261117_234249 reverse, det320, 60 s per config, no throttling, no meter gaps; whole-board mJ/frame, mean [min-max]):** FP32 models: K1 583.7 [578-589] -> 539.8 [536-543] with the INT8 detector (-7.5%), K5 450.3 [450-451] -> 440.1 [438-442] (-2.3%). int8nodw models: K1 540.8 [538-544] -> 490.1 [489-491] (-9.4%), K5 398.8 [398-400] -> 390.8 [386-396] (-2.0%; ranges almost touch). fps int8nodw K1 8.47 -> 9.39, K5 11.50 -> 11.77; full-pipeline p50 at K1 117.5 -> 106.0 ms (FP32 models 123.2 -> 112.0). The saving shrinks at K5 because the detector runs on only one frame in five.
- **All three models INT8 except depthwise, det320, K5:** 390.8 mJ/frame at 11.8 fps, vs 1106 mJ/frame at 4.3 fps for the FP32 baseline (640, K1): -65%. Almost all of that comes from the detector settings; quantization of all three models contributes about 13% of the det320/K5 FP32 figure (450 -> 391 mJ).
- **Fidelity (`eval/detector_fidelity.py`, test frames at 10 fps, `results/processed/detector_fidelity_det320.csv`):** when both find the face, mean IoU 0.97 (5th pct 0.94), landmark error 2.5% of the inter-ocular distance (95th pct 4.7%). But the INT8 detector misses 155 of the 6,153 faces FP32 finds (never the reverse), concentrated in dim light and one person: p03_live_dim 126/177, p03_screen_dim 12/161, the rest <= 7 per video. p03 was not in the calibration frames (p01/p02 only) and dim frames are darker than most calibration frames: an RQ3-style concentrated loss.
- **Downstream (`results/processed/sequential_det-int8nodw/`, det320, default thresholds, vs the FP32 detector):** session 2, K5: FP32 models APCER 10.1 -> 8.7%, FRR 49.8 -> 50.6%; int8nodw models APCER 15.0 -> 14.3%, FRR 33.8 -> 35.0%; impostor accepts 0. Lower APCER is only because a missed face is a reject. Session-1 test frames with p03 dim lose many more faces (no-face frames at K1 28 -> 168).

## Status

Done: FP32 pipeline, INT8 PTQ of both models, 3 x 3 accuracy evaluation and latency/memory/CPU-time benchmark on both PC and Pi, PC-vs-Pi output comparison, figures. Team videos now include p03 (test only; calibration used p01/p02). Findings: `Version1/private/initial test.md` (local) and the Claude Docs write-up (older, PC-only).

Pi findings: INT8 is slower on the Pi too (model-only, 4 threads: recognizer 54.1 -> 71.3 ms, liveness 20.7 -> 24.4 ms; full pipeline 225 -> 249-258 ms, detector ~138 ms of that). Pi accuracy matches the PC (FP32 outputs within 1e-5; INT8 changes 0 LFW and 2/968 liveness decisions). Pi memory: the ~2,000 MB was a benchmark artifact, now fixed; see Write-up notes.

Session 2 (PC only so far): FP32 liveness rejects 49% of live frames (63% dim) but passes only 8% of paper photos; INT8 shifts live scores up by roughly +2 to +7 logits there, much more than the +0.75 to +1.5 on the older sets, so BPCER falls (49% -> 15-22%) and photo APCER rises (8% -> 16-20%). Recognizer: 0 impostor accepts in 938 cross-person attempts in every config; glasses barely move genuine cosine. p03's paper photo in bright light already passes FP32 liveness 50% of the time.

INT8 except depthwise (`int8nodw`, 2026-11-16): closest to FP32 of all INT8 variants. Recognizer fidelity (mean cos to FP32) 0.99 LFW, 0.98 team (full INT8 0.95-0.98); LFW EER 0.50% vs 0.44%. Liveness score shift mostly gone: MSU APCER 42.9% (FP32 46.7%, full INT8 60-63%), Kaggle 14.4% (16.2%), session 2 10.4% APCER / 30.9% BPCER (FP32 8.0% / 49.0%); decision flips vs FP32 about halved. Pi model-only (4 threads): recognizer 45.6 ms vs FP32 52.5 (-13%; full INT8 68), liveness 21.9 vs 20.9 (+4%). Pi full pipeline and power (`results/processed/power_pi_nodw.csv`, forward + reverse pass, 5 configs in the same runs): p50 221.5 ms vs FP32 222.8 (no measurable change; detector dominates); 4.68 W vs 4.75; 1042 mJ/frame [1033-1051] vs FP32 1065 [1062-1067], about -2% and the only INT8 config below FP32; full INT8 pc 1134 (+7%). Small gain: it is within about 2x the pass-to-pass spread.

Pi latency benchmark re-run (2026-11-16, fixed frame loader, 16 configs, 500 runs each, `results/processed/bench_pi.csv`): p50 FP32 234.0 ms, rec-int8nodw 224-232 ms, full-INT8 recognizer 250-257 ms; recognizer stage 54.5 ms FP32 vs 46.6 nodw vs 69-72 full INT8. The Pi ran at 71-81 °C (no cool-down between configs in this benchmark; `throttled` 0x0, 1.8 GHz throughout), so the detector took ~147 ms vs ~138 in the cooled power runs; compare configs within one run.

Pending: liveness score shift (calibration-window threshold tuning does not carry over to session 2; leave-one-person-out tuning on session 2 recovers FP32-level APCER; see Write-up notes), p03 glare test, calibration seeds 1 and 2.

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
  eval/sequential_eval.py      team videos in frame order: detect-every-K accuracy + liveness threshold tuning
  bench/bench.py               latency/RSS/size (--label pc|pi, --isolated); --power = sustained runs with PHASE markers
  bench/power_log.py           FNB58 meter -> CSV (PC)
  bench/power_run.py           PC: meter + Pi bench --power over SSH -> results/raw/power/<label>/<run>/
  eval/power_analysis.py       power per config (per run folder); --combine [--name] -> results/processed/power_<label>[_<name>].csv
  eval/plots.py                figures -> results/figures/
  results/{raw,processed,figures}/   committed (scores and plots only)
  models/int8/, data/, private/, external/   git-ignored
```

Run order and data placement: `Version1/README.md`. Python 3.11 venv in `Version1/.venv`; exact versions in `Version1/requirements.lock.txt`.
