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

Quantization ablation (4 x 4: recognizer and liveness each FP32 / INT8 per-tensor / INT8 per-channel /
INT8 per-channel with depthwise convolutions left in FP32).
Run in this order from the Version1 folder with the venv active:
  1. python eval\prepare_data.py      detect faces once, cache crops in private\cache (needs the data below)
  2. python quantize\quantize.py      INT8 models + configs\ablation\*.json (calibration seed 0;
                                     reuses the manifest's samples; --variants X builds only X)
  3. python eval\run_eval.py          accuracy -> results\processed\summary.md
  4. python bench\bench.py --label pc latency/memory/CPU time -> results\processed\bench_pc.csv (use --label pi on the Pi)
     python bench\bench.py --label pc --isolated   model-only timing
  5. python eval\plots.py --bench pc  figures -> results\figures\

On the Pi (same order, skipping 1-2: copy private\cache and models\int8 from the PC instead):
  python eval/run_eval.py --label pi            accuracy on ARM -> results/processed/pi/
  python eval/sequential_eval.py                 detect-every-K on videos in frame order + liveness thresholds
                                                 -> results/processed/sequential/ (--analyse-only to re-analyse)
  python bench/bench.py --label pi              and --label pi --isolated

Power (run on the PC; meter wired PSU -> FNB58 in, FNB58 out -> Pi, micro-USB -> PC last;
the Pi needs key-based SSH and the same bench/bench.py and app/pipeline.py as the PC):
  python bench\power_run.py                     every config, 60 s sustained each, idle phases between
  python bench\power_run.py --seconds 20 --cells rec-fp32_live-fp32   quick check
  -> results\raw\power\pi\<run_id>\ (power_summary.csv, power_trace.png); nothing outside the run folder
  python eval\power_analysis.py results\raw\power\pi\<run_id>   re-analyse an existing run
  python bench\power_run.py --reverse            second pass in reverse order (order/heat check)
  python eval\power_analysis.py --combine <run1> <run2>   mean and min-max -> results\processed\power_pi.csv
  python eval\power_analysis.py --combine <run1> <run2> --name nodw   -> results\processed\power_pi_nodw.csv
  python bench\power_run.py --det-size 320 --detect-every 1,3,5 --cells <configs>   detector sweep, one chunk per size
  python eval\power_analysis.py --combine <6 runs> --name detector   -> results\processed\power_pi_detector.csv
  python eval/compare_runs.py --a pc --b pi     do Pi outputs match the committed PC results?

Data (all local, git-ignored, never shared):
- private\<name>_<live|print|screen>_<bright|dim>.<mp4|webm>   team videos (first 4 s = calibration only)
- private\session2\<name>_<live|print>_<bright|dim>_<glasses|noglasses>.mp4   second session, test only
  (python eval\prepare_data.py --only session2; every name must already be in private\pseudonyms.json)
- data\lfw\archive.zip                     Kaggle jessicali9530/lfw-dataset (LFW deep-funneled)
- data\msu-mfsd\                            git clone of github.com/sunny3/MSU-MFSD (commit pinned)
- data\anti-spoofing-kaggle\archive.zip    Kaggle tapakah68/anti-spoofing (free 45-file sample)
