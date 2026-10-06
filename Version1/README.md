What goes where:
- Combined pipeline (liveness + verification): app\pipeline.py, settings in ..\configs\pipeline_fp32.json
  - python app\pipeline.py enroll              (uses private\reference.jpg)
  - python app\pipeline.py verify --camera     (q to quit)
  - python app\pipeline.py verify --image PATH
  - any ablation config works too: --config ..\configs\ablation\rec-int8pc_live-int8pc.json
- Reference photo: private\reference.jpg
- Anti-spoofing FP32 model: inside external\face-antispoof-onnx\models\best\98.20\best_model.onnx
- InsightFace models (buffalo_sc): downloaded on first run to %USERPROFILE%\.insightface\models\buffalo_sc
- INT8 models: models\int8\ (generated, git-ignored; hashes pinned in ..\configs\models.lock.json)

Quantization ablation (3 x 3: recognizer and liveness each FP32 / INT8 per-tensor / INT8 per-channel).
Run in this order from the Version1 folder with the venv active:
  1. python eval\prepare_data.py      detect faces once, cache crops in private\cache (needs the data below)
  2. python quantize\quantize.py      INT8 models + configs\ablation\*.json (calibration seed 0)
  3. python eval\run_eval.py          accuracy -> results\processed\summary.md
  4. python bench\bench.py --label pc latency/memory/CPU time -> results\processed\bench_pc.csv (use --label pi on the Pi)
     python bench\bench.py --label pc --isolated   model-only timing
  5. python eval\plots.py --bench pc  figures -> results\figures\

On the Pi (same order, skipping 1-2: copy private\cache and models\int8 from the PC instead):
  python eval/run_eval.py --label pi            accuracy on ARM -> results/processed/pi/
  python bench/bench.py --label pi              and --label pi --isolated
  python eval/compare_runs.py --a pc --b pi     do Pi outputs match the committed PC results?

Data (all local, git-ignored, never shared):
- private\<name>_<live|print|screen>_<bright|dim>.<mp4|webm>   team videos (first 4 s = calibration only)
- data\lfw\archive.zip                     Kaggle jessicali9530/lfw-dataset (LFW deep-funneled)
- data\msu-mfsd\                            git clone of github.com/sunny3/MSU-MFSD (commit pinned)
- data\anti-spoofing-kaggle\archive.zip    Kaggle tapakah68/anti-spoofing (free 45-file sample)
