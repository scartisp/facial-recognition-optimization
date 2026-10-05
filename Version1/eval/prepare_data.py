"""Detect faces once and cache the model-ready face crops for every dataset.

Run once (from the Version1 folder, venv active):
    python eval/prepare_data.py                 # all datasets
    python eval/prepare_data.py --only own msu  # a subset

The detector is the fixed FP32 SCRFD from the pipeline, and the crops are produced by the
same Pipeline methods the live demo uses. Every precision variant is later evaluated on these
exact cached crops, so FP32 and INT8 never see different inputs.

Outputs (all under private/, never committed):
    private/cache/<dataset>.npz     uint8 crops + per-sample metadata
    private/pseudonyms.json         real name -> p01, p02 ... for the team videos
Committed summary:
    results/raw/detection_summary.csv
"""
import argparse
import csv
import io
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pipeline as pl  # noqa: E402

version1_folder = pl.version1_folder
private_folder = version1_folder / "private"
cache_folder = private_folder / "cache"
data_folder = version1_folder / "data"
results_raw = version1_folder / "results" / "raw"

# Team videos: the first CALIB_SECONDS of every video are held out for quantization
# calibration (and enrollment); only the rest is used for testing.
CALIB_SECONDS = 4.0
CALIB_INTERVAL = 0.2
TEST_INTERVAL = 0.5

# Kaggle videos: one frame per second, at most this many per video
KAGGLE_INTERVAL = 1.0
KAGGLE_MAX_FRAMES = 15


class Collector:
    """Accumulates crops and metadata for one dataset, counting detection failures."""

    def __init__(self, pipeline, need_liveness=True, metric="max"):
        self.pipeline = pipeline
        self.need_liveness = need_liveness
        self.metric = metric
        self.rec, self.live, self.meta = [], [], []
        self.attempted = 0

    def add(self, frame_bgr, **meta):
        self.attempted += 1
        face = self.pipeline.detect(frame_bgr, metric=self.metric)
        if face is None:
            return False
        bbox, keypoints = face
        self.rec.append(self.pipeline.recognizer_crop(frame_bgr, keypoints))
        if self.need_liveness:
            self.live.append(self.pipeline.liveness_crop(frame_bgr, bbox))
        meta["face_width_px"] = int(bbox[2] - bbox[0])
        self.meta.append(meta)
        return True

    def save(self, name):
        cache_folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_folder / f"{name}.npz",
            rec=np.array(self.rec, dtype=np.uint8),
            live=np.array(self.live, dtype=np.uint8),
            meta=np.array([json.dumps(m) for m in self.meta]),
        )
        print(f"{name}: {len(self.meta)}/{self.attempted} faces cached")
        return {"dataset": name, "attempted": self.attempted, "faces_found": len(self.meta)}


def sample_video(path, wanted):
    """Yield (time_s, frame) for each frame where wanted(time_s) is True.

    Uses each frame's own timestamp, so variable-frame-rate browser WebM files work.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    while capture.grab():
        time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if wanted(time_s):
            ok, frame = capture.retrieve()
            if ok:
                yield time_s, frame
    capture.release()


class EverySeconds:
    """Stateful filter: True roughly every `interval` seconds within [start, end)."""

    def __init__(self, interval, start=0.0, end=float("inf"), max_count=None):
        self.interval, self.next_t, self.end = interval, start, end
        self.max_count, self.count = max_count, 0

    def __call__(self, time_s):
        if time_s < self.next_t or time_s >= self.end:
            return False
        if self.max_count is not None and self.count >= self.max_count:
            return False
        self.next_t += self.interval
        while self.next_t <= time_s:
            self.next_t += self.interval
        self.count += 1
        return True


def prepare_own(pipeline):
    """Team videos named <person>_<live|print|screen>_<bright|dim>[_n].<ext> in private/."""
    pattern = re.compile(r"^([a-z0-9]+)_(live|print|screen)_(bright|dim)(?:_\d+)?$", re.I)
    videos = sorted(
        p for p in private_folder.iterdir()
        if p.suffix.lower() in {".mp4", ".webm", ".mov", ".avi", ".mkv"} and pattern.match(p.stem)
    )
    if not videos:
        raise FileNotFoundError(f"No team videos found in {private_folder}")

    names = sorted({pattern.match(p.stem).group(1).lower() for p in videos})
    pseudonyms = {name: f"p{i + 1:02d}" for i, name in enumerate(names)}
    (private_folder / "pseudonyms.json").write_text(json.dumps(pseudonyms, indent=2))

    collector = Collector(pipeline)
    for path in tqdm(videos, desc="own videos"):
        name, kind, lighting = pattern.match(path.stem).groups()
        base = {"dataset": "own", "person": pseudonyms[name.lower()], "kind": kind.lower(),
                "lighting": lighting.lower(), "source": f"{pseudonyms[name.lower()]}_{kind}_{lighting}"}
        calib = EverySeconds(CALIB_INTERVAL, 0.0, CALIB_SECONDS)
        test = EverySeconds(TEST_INTERVAL, CALIB_SECONDS)
        for time_s, frame in sample_video(path, lambda t: calib(t) or test(t)):
            split = "calib" if time_s < CALIB_SECONDS else "test"
            collector.add(frame, **base, split=split, time_s=round(time_s, 2))
    return collector.save("own")


def prepare_lfw(pipeline):
    """LFW (deep-funneled, Kaggle jessicali9530/lfw-dataset): every image used by pairs.csv."""
    archive = zipfile.ZipFile(data_folder / "lfw" / "archive.zip")
    image_root = "lfw-deepfunneled/lfw-deepfunneled"

    pairs = []
    rows = list(csv.reader(io.TextIOWrapper(archive.open("pairs.csv"), encoding="utf-8")))[1:]
    for row in rows:
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) == 3:
            pairs.append((cells[0], int(cells[1]), cells[0], int(cells[2]), 1))
        elif len(cells) == 4:
            pairs.append((cells[0], int(cells[1]), cells[2], int(cells[3]), 0))
    if len(pairs) != 6000:
        raise RuntimeError(f"Expected 6000 LFW pairs, parsed {len(pairs)}")

    image_ids = sorted({f"{n}_{i:04d}" for a, ia, b, ib, _ in pairs for n, i in ((a, ia), (b, ib))})

    # LFW images are centred on the subject, so prefer the face nearest the centre
    collector = Collector(pipeline, need_liveness=False, metric="default")
    for image_id in tqdm(image_ids, desc="lfw"):
        person = image_id.rsplit("_", 1)[0]
        data = np.frombuffer(archive.read(f"{image_root}/{person}/{image_id}.jpg"), np.uint8)
        collector.add(cv2.imdecode(data, cv2.IMREAD_COLOR), dataset="lfw", image_id=image_id,
                      person=person, split="test")

    with open(cache_folder.parent / "lfw_pairs.json", "w") as file:
        json.dump([{"a": f"{a}_{ia:04d}", "b": f"{b}_{ib:04d}", "same": same}
                   for a, ia, b, ib, same in pairs], file)
    return collector.save("lfw")


def prepare_msu(pipeline):
    """MSU-MFSD first-frame subset (github.com/sunny3/MSU-MFSD, commit pinned in models.lock.json)."""
    root = data_folder / "msu-mfsd" / "pics"
    collector = Collector(pipeline)
    files = sorted(p for p in root.glob("*/*.jpg") if ".ipynb_checkpoints" not in p.parts)
    for path in tqdm(files, desc="msu"):
        client = re.search(r"client(\d+)", path.name).group(1)
        if path.name.startswith("real_"):
            kind, detail = "live", "real"
        elif "printed_photo" in path.name:
            kind, detail = "print", "printed_photo"
        else:
            kind, detail = "screen", re.search(r"_(ipad|iphone)_video", path.name).group(1)
        collector.add(cv2.imread(str(path)), dataset="msu", person=f"msu{client}", kind=kind,
                      detail=detail, lighting="na", split="test", source=path.name)
    return collector.save("msu")


def prepare_kaggle(pipeline):
    """Kaggle tapakah68/anti-spoofing free sample (45 files)."""
    archive = zipfile.ZipFile(data_folder / "anti-spoofing-kaggle" / "archive.zip")
    kinds = {"live_selfie": ("live", "selfie"), "live_video": ("live", "video"),
             "printouts": ("print", "a4_printout"), "cut-out printouts": ("print", "cutout_printout"),
             "replay": ("screen", "replay")}

    collector = Collector(pipeline)
    entries = sorted(i for i in archive.namelist() if i.split("/")[0] in kinds and not i.endswith("/"))
    with tempfile.TemporaryDirectory(dir=cache_folder) as temp:
        for entry in tqdm(entries, desc="kaggle"):
            folder, filename = entry.split("/", 1)
            kind, detail = kinds[folder]
            # File names are <uploader>--<session id>[__<devices>]; the session id is the person
            person = "kg" + filename.split("--")[1].split("__")[0].split(".")[0][-8:]
            base = {"dataset": "kaggle", "person": person, "kind": kind, "detail": detail,
                    "lighting": "na", "split": "test", "source": f"{folder}/{person}"}
            if filename.lower().endswith(".jpg"):
                image = cv2.imdecode(np.frombuffer(archive.read(entry), np.uint8), cv2.IMREAD_COLOR)
                collector.add(image, **base, time_s=0.0)
                continue
            local = Path(temp) / ("clip" + Path(filename).suffix.lower())
            local.write_bytes(archive.read(entry))
            sampler = EverySeconds(KAGGLE_INTERVAL, 0.0, max_count=KAGGLE_MAX_FRAMES)
            for time_s, frame in sample_video(local, sampler):
                collector.add(frame, **base, time_s=round(time_s, 2))
    return collector.save("kaggle")


def main():
    preparers = {"own": prepare_own, "msu": prepare_msu, "kaggle": prepare_kaggle, "lfw": prepare_lfw}
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=preparers, default=list(preparers))
    args = parser.parse_args()

    config = json.loads(pl.default_config.read_text())
    pl.check_pins(config)
    pipeline = pl.Pipeline(config)
    cache_folder.mkdir(parents=True, exist_ok=True)

    summary_path = results_raw / "detection_summary.csv"
    rows = {}
    if summary_path.exists():
        rows = {r["dataset"]: r for r in csv.DictReader(open(summary_path))}
    for name in args.only:
        rows[name] = preparers[name](pipeline)

    results_raw.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset", "attempted", "faces_found"])
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))


if __name__ == "__main__":
    main()
