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

Raspberry Pi 4 Model B, 8 GB RAM (Cortex-A72, no int8 dot-product instructions). OS, webcam, and power meter still to be recorded. The Pi is not set up yet and there is no power meter yet, so all speed numbers so far come from an x86 Windows PC: label them as PC results, never as Pi results.

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

3 x 3 grid: recognizer {FP32, INT8 per-tensor, INT8 per-channel} x liveness {same three}, one config per cell in `configs/ablation/rec-<v>_live-<v>.json` (`fp32`, `int8pt`, `int8pc`).

- ONNX Runtime 1.30 static PTQ: QDQ, S8S8 (signed INT8 weights and activations), MinMax calibration, 128 calibration frames, seed 0. `w600k_mbf` is converted from opset 11 to 13 first (needed for per-channel; FP32 outputs verified bit-identical).
- INT8 files go to `Version1/models/int8/` (git-ignored); their SHA-256 hashes are added to `configs/models.lock.json`; the calibration manifest is `results/raw/quantization_manifest_s<seed>.json`.
- Proposed next variant, not yet adopted: INT8 except depthwise convolutions (see Known risks).

## Data

All local and git-ignored. Results files hold scores only; team members appear as `p01`, `p02` (mapping in `private/pseudonyms.json`).

| Dataset | Location | Used for |
|---|---|---|
| Team videos `<name>_<live\|print\|screen>_<bright\|dim>.<mp4\|webm>` | `Version1/private/` | Calibration (first 4 s of every video only), liveness test and combined pipeline (rest), enrollment (first 4 s of `live_bright`) |
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
| Power | W | inline power meter, sustained run |
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

## Status

Done: FP32 pipeline, INT8 PTQ of both models, 3 x 3 accuracy evaluation, PC latency/memory benchmark, figures. Initial findings write-up shared with the team (Claude Docs). Pending: Pi benchmark, power, calibration seeds 1 and 2, decision on the INT8-except-depthwise variant.

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
  bench/bench.py               latency/RSS/size (--label pc|pi, --isolated)
  eval/plots.py                figures -> results/figures/
  results/{raw,processed,figures}/   committed (scores and plots only)
  models/int8/, data/, private/, external/   git-ignored
```

Run order and data placement: `Version1/README.md`. Python 3.11 venv in `Version1/.venv`; exact versions in `Version1/requirements.lock.txt`.
