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

Raspberry Pi 4 Model B, 8 GB RAM (Cortex-A72, no int8 dot-product instructions). OS, webcam, and power meter still to be recorded.

## Pipeline

```
webcam frame -> face detect + align -> liveness model ----\
                                    \-> verification model -> AND -> accept / reject
```

Accept only if the face is live **and** matches the enrolled embedding (cosine similarity above threshold).

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
| Latency | ms, p50 and p95 | end-to-end (detect, align, embed/classify), 500+ runs, warm-up excluded |
| Power | W | inline power meter, sustained run |
| Memory | MB | RSS during sustained run, not just at load |
| Model size | MB | on-disk, per precision |
| Accuracy | % | FAR/FMR, FRR/FNMR, EER, TAR @ FAR=0.1% |

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

- PTQ accuracy drop on the liveness model (relies on fine texture/frequency cues).
- Conversion pitfalls: PyTorch/ONNX to TFLite (unsupported ops, layout changes, silent output-range bugs).
- Models already fit the Pi comfortably, so efficiency gains may be small. Fallback: frame efficiency around sustained continuous inference.
- Subgroup-labeled data may be unavailable. Fallback: narrow the fairness claim to lighting.

## Status

Planned layout (adjust as the repo takes shape): `models/` (fp32, int8), `quantize/`, `bench/`, `eval/`, `results/` (raw + processed), `configs/`, `demo/`.
