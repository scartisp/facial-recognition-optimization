"""Post-training static INT8 quantization of the recognizer and the liveness model.

Run after eval/prepare_data.py (from the Version1 folder, venv active):
    python quantize/quantize.py            # seed 0
    python quantize/quantize.py --seed 1   # another calibration sample

For each model it produces two INT8 variants that differ in exactly one setting:
    per_tensor   one scale per weight tensor
    per_channel  one scale per output channel
Everything else is ONNX Runtime's default static PTQ: QDQ format, signed INT8 weights and
activations (S8S8), MinMax calibration. The detector is not quantized.

Calibration uses only the held-out first seconds of the team videos (split == "calib"),
never test frames. It also:
    - writes models/int8/*.onnx (git-ignored)
    - adds their SHA-256 hashes to configs/models.lock.json
    - writes the 9 ablation configs to configs/ablation/
    - records the calibration manifest in results/raw/quantization_manifest_s<seed>.json
"""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pipeline as pl  # noqa: E402

version1_folder = pl.version1_folder
repo_folder = pl.repo_folder
int8_folder = version1_folder / "models" / "int8"
ablation_folder = repo_folder / "configs" / "ablation"
results_raw = version1_folder / "results" / "raw"
cache_path = version1_folder / "private" / "cache" / "own.npz"

CALIBRATION_SIZE = 128
MIN_OPSET = 13
VARIANTS = {"per_tensor": False, "per_channel": True}
SHORT = {"fp32": "fp32", "per_tensor": "int8pt", "per_channel": "int8pc"}


class ArrayReader(CalibrationDataReader):
    """Feeds calibration samples to ONNX Runtime one at a time (batch size 1)."""

    def __init__(self, input_name, blobs):
        self.input_name = input_name
        self.blobs = blobs
        self.index = 0

    def get_next(self):
        if self.index >= len(self.blobs):
            return None
        blob = self.blobs[self.index : self.index + 1]
        self.index += 1
        return {self.input_name: blob}

    def rewind(self):
        self.index = 0


def default_opset(model):
    return next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def input_name_of(path):
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_inputs()[0].name


def quantize_model(fp32_path, blobs, out_path, per_channel):
    with tempfile.TemporaryDirectory() as temp:
        # Per-channel QDQ needs DequantizeLinear's axis attribute, added in opset 13.
        # w600k_mbf ships as opset 11; the conversion leaves FP32 outputs bit-identical.
        model = onnx.load(str(fp32_path))
        if default_opset(model) < MIN_OPSET:
            model = version_converter.convert_version(model, MIN_OPSET)
        upgraded = Path(temp) / "upgraded.onnx"
        onnx.save(model, str(upgraded))

        prepared = Path(temp) / "prepared.onnx"
        quant_pre_process(str(upgraded), str(prepared))
        quantize_static(
            str(prepared),
            str(out_path),
            ArrayReader(input_name_of(prepared), blobs),
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )


def main():
    parser = argparse.ArgumentParser(description="Static INT8 PTQ for both models")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_config = json.loads(pl.default_config.read_text())
    pl.check_pins(base_config)
    pipeline = pl.Pipeline(base_config)  # used only for its preprocessing methods

    cache = np.load(cache_path)
    meta = [json.loads(m) for m in cache["meta"]]
    calib_index = [i for i, m in enumerate(meta) if m["split"] == "calib"]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(calib_index, size=min(CALIBRATION_SIZE, len(calib_index)), replace=False).tolist())

    blobs = {
        "recognizer": np.ascontiguousarray(pipeline.recognizer_blob(list(cache["rec"][chosen]))),
        "liveness": np.ascontiguousarray(pipeline.liveness_blob(cache["live"][chosen])),
    }

    lock = json.loads(pl.lock_path.read_text())
    int8_folder.mkdir(parents=True, exist_ok=True)
    produced = {"recognizer": {"fp32": base_config["models"]["recognizer"]},
                "liveness": {"fp32": base_config["models"]["liveness"]}}

    for role in ("recognizer", "liveness"):
        fp32_path = pl.resolve_path(base_config["models"][role]["path"])
        for variant, per_channel in VARIANTS.items():
            out_path = int8_folder / f"{fp32_path.stem}_int8_{variant}_s{args.seed}.onnx"
            print(f"Quantizing {role} ({variant}) -> {out_path.name}")
            quantize_model(fp32_path, blobs[role], out_path, per_channel)
            lock_key = f"int8/{out_path.name}"
            lock["files"][lock_key] = sha256_of(out_path)
            produced[role][variant] = {
                "path": out_path.relative_to(repo_folder).as_posix(),
                "lock_key": lock_key,
            }

    pl.lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # One config per cell of the 3 x 3 grid; only the two model entries differ
    ablation_folder.mkdir(parents=True, exist_ok=True)
    for rec_variant, rec_model in produced["recognizer"].items():
        for live_variant, live_model in produced["liveness"].items():
            name = f"rec-{SHORT[rec_variant]}_live-{SHORT[live_variant]}"
            config = copy.deepcopy(base_config)
            config["precision"] = name
            config["models"]["recognizer"] = rec_model
            config["models"]["liveness"] = live_model
            config["calibration_seed"] = args.seed
            (ablation_folder / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")

    manifest = {
        "seed": args.seed,
        "calibration_size": len(chosen),
        "calibration_pool_size": len(calib_index),
        "calibration_source": "team videos, first 4 s of each video (split == calib)",
        "samples": [meta[i]["source"] + f"@{meta[i]['time_s']}s" for i in chosen],
        "quant_format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "calibrate_method": "MinMax",
        "reduce_range": False,
        "min_opset": MIN_OPSET,
        "onnxruntime_version": onnxruntime.__version__,
        "models": produced,
    }
    results_raw.mkdir(parents=True, exist_ok=True)
    (results_raw / f"quantization_manifest_s{args.seed}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done. Configs in {ablation_folder}")


if __name__ == "__main__":
    main()
