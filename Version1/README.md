What goes where:
- Combined pipeline (liveness + verification): app\pipeline.py, settings in ..\configs\pipeline_fp32.json
  - python app\pipeline.py enroll              (uses private\reference.jpg)
  - python app\pipeline.py verify --camera     (q to quit)
  - python app\pipeline.py verify --image PATH
- Benchmark records: benchmarks\results.csv
- Reference photo: private\reference.jpg
- Anti-spoofing FP32 model: inside external\face-antispoof-onnx\models\best\98.20\best_model.onnx
- InsightFace models: download automatically on first run; keep them under models\ when we configure the final program.