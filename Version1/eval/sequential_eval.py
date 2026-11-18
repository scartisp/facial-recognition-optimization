"""Accuracy of detect-every-K on the team videos played in frame order, plus liveness threshold tuning.

The cached evaluation (prepare_data.py / run_eval.py) detects on every sampled frame, so it cannot
measure what reusing a face box for K-1 frames does. This script plays each team video in order at
the rate the Pi actually processes frames (--fps, default 10: the Pi runs 8.9-11.5 fps at det320/K5),
runs the FP32 detector every K-th processed frame exactly as pipeline.verify_camera does (a missing
face forces detection on the next frame), crops the current frame with the reused box, and scores
liveness and identity with each precision.

    python eval/sequential_eval.py                 # score (slow, ~1 h on the PC) + analyse
    python eval/sequential_eval.py --analyse-only  # re-analyse results/raw/sequential/frames.csv

Splits (same rule as prepare_data.py): session-1 videos, first 4 s = calibration window, rest =
test; session 2 = test only. Each person is enrolled from the calibration window of their session-1
bright live video (K = 1 at the same detector size), per precision.

Liveness thresholds, chosen on the calibration window only (session-1 live and screen-replay
frames, never test frames), per precision and per detector setting; the rules were fixed before
looking at test results:
  default        logit 0 (facenox default; what every other result uses)
  apcer_parity   primary rule: the lowest threshold at which the precision accepts no larger a
                 share of calibration attacks than FP32 does at 0 (security parity with FP32)
  bpcer_parity   the highest threshold at which it rejects no larger a share of calibration live
                 frames than FP32 does at 0 (usability parity)
  calib_eer      each precision's own EER threshold on the calibration window, FP32 included
                 (a deployment-tuned threshold, a separate change from quantization)
A frame with no face is a reject. Verification threshold: 0.2421 (LFW FP32, FAR = 0.1%), fixed.

Writes (scores only, no images):
    results/raw/sequential/frames.csv        one row per processed frame and detector setting
    results/processed/sequential/thresholds.csv, liveness.csv, combined.csv, summary.md
"""
import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pipeline as pl  # noqa: E402
import run_eval as ev  # noqa: E402
from prepare_data import CALIB_SECONDS, EverySeconds, sample_video  # noqa: E402

version1_folder = pl.version1_folder
private_folder = version1_folder / "private"
raw_folder = version1_folder / "results" / "raw" / "sequential"
processed_folder = version1_folder / "results" / "processed" / "sequential"
# INT8 detector (quantize/quantize_detector.py), one static file per input size
INT8_DETECTOR = {320: {"path": "Version1/models/int8/det_500m_det320_int8_per_channel_nodw_s0.onnx",
                       "lock_key": "int8/det_500m_det320_int8_per_channel_nodw_s0.onnx"}}

VARIANTS = ["fp32", "int8pc", "int8nodw"]
COSINE_THRESHOLD = 0.2421
RULES = ["default", "apcer_parity", "bpcer_parity", "calib_eer"]
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def team_videos():
    """(path, metadata) for session-1 and session-2 team videos, people as pseudonyms."""
    pseudonyms = json.loads((private_folder / "pseudonyms.json").read_text())
    s1 = re.compile(r"^([a-z0-9]+)_(live|print|screen)_(bright|dim)(?:_\d+)?$", re.I)
    s2 = re.compile(r"^([a-z0-9]+)_(live|print)_(bright|dim)_(glasses|noglasses)$", re.I)
    videos = []
    for p in sorted(private_folder.iterdir()):
        if p.suffix.lower() in VIDEO_SUFFIXES and s1.match(p.stem):
            name, kind, light = (g.lower() for g in s1.match(p.stem).groups())
            videos.append((p, {"dataset": "own", "person": pseudonyms[name], "kind": kind, "lighting": light,
                               "glasses": "na", "source": f"{pseudonyms[name]}_{kind}_{light}"}))
    for p in sorted((private_folder / "session2").glob("*.mp4")):
        if s2.match(p.stem):
            name, kind, light, gl = (g.lower() for g in s2.match(p.stem).groups())
            videos.append((p, {"dataset": "session2", "person": pseudonyms[name], "kind": kind, "lighting": light,
                               "glasses": "yes" if gl == "glasses" else "no",
                               "source": f"{pseudonyms[name]}_{kind}_{light}_{gl}"}))
    return videos


# ---------- stage 1: score every processed frame ----------

def score(args):
    det_sizes = [int(d) for d in args.det_sizes.split(",")]
    every = [int(k) for k in args.every.split(",")]
    settings = [(d, k) for d in det_sizes for k in every]
    base = json.loads(pl.default_config.read_text())
    pl.check_pins(base)
    detectors = {}
    for d in det_sizes:
        cfg = json.loads(json.dumps(base))
        cfg["detector"]["input_size"] = [d, d]
        if args.det_model == "int8nodw":
            cfg["models"]["detector"] = dict(INT8_DETECTOR[d])
            pl.check_pins(cfg)
        detectors[d] = pl.Pipeline(cfg)
    configs = ev.variant_configs()
    runners = {v: ev.ModelRunner(configs[v]) for v in VARIANTS}

    rows, embeddings = [], {v: [] for v in VARIANTS}
    for path, meta in tqdm(team_videos()[:args.max_videos or None], desc="videos"):
        crops = {s: ([], []) for s in settings}
        state = {s: None for s in settings}
        frame_rows = []
        for i, (t, frame) in enumerate(sample_video(path, EverySeconds(1.0 / args.fps))):
            split = "calib" if meta["dataset"] == "own" and t < CALIB_SECONDS else "test"
            detected = {d: detectors[d].detect(frame) for d in det_sizes}
            for d, k in settings:
                fresh = state[(d, k)] is None or i % k == 0
                if fresh:
                    state[(d, k)] = detected[d]
                face = state[(d, k)]
                row = {**meta, "split": split, "time_s": round(t, 3), "frame": i, "det": d, "k": k,
                       "fresh": int(fresh), "has_face": int(face is not None)}
                if face is not None:
                    bbox, keypoints = face
                    p = detectors[d]
                    crops[(d, k)][0].append(p.recognizer_crop(frame, keypoints))
                    crops[(d, k)][1].append(p.liveness_crop(frame, bbox))
                    row["face_width_px"] = int(bbox[2] - bbox[0])
                frame_rows.append(row)
        # Run every precision on this video's crops, then attach scores in row order
        out = {}
        for s in settings:
            rec, live = np.array(crops[s][0], np.uint8), np.array(crops[s][1], np.uint8)
            out[s] = {v: (runners[v].embed(rec), runners[v].liveness_scores(live)) for v in VARIANTS}
        pos = {s: 0 for s in settings}
        for row in frame_rows:
            s = (row["det"], row["k"])
            row["emb_index"] = -1
            if row["has_face"]:
                j = pos[s]
                pos[s] += 1
                for v in VARIANTS:
                    row[f"live_{v}"] = float(out[s][v][1][j])
                row["emb_index"] = len(embeddings["fp32"])
                for v in VARIANTS:
                    embeddings[v].append(out[s][v][0][j])
            rows.append(row)

    emb = {v: np.array(embeddings[v], np.float32) for v in VARIANTS}
    people = sorted({r["person"] for r in rows})
    # Templates: calibration window of each person's session-1 bright live video, K = 1, per detector size
    templates = {}
    for d in det_sizes:
        for person in people:
            idx = [r["emb_index"] for r in rows if r["dataset"] == "own" and r["person"] == person
                   and r["kind"] == "live" and r["lighting"] == "bright" and r["split"] == "calib"
                   and r["det"] == d and r["k"] == 1 and r["has_face"]]
            for v in VARIANTS:
                t = emb[v][idx].mean(axis=0)
                templates[(d, person, v)] = t / np.linalg.norm(t)
    for r in rows:
        for person in people:
            for v in VARIANTS:
                r[f"sim_{v}_{person}"] = (float(emb[v][r["emb_index"]] @ templates[(r["det"], person, v)])
                                          if r["has_face"] else "")
        del r["emb_index"]

    fields = (["dataset", "person", "kind", "lighting", "glasses", "source", "split", "time_s", "frame", "det", "k",
               "fresh", "has_face", "face_width_px"] + [f"live_{v}" for v in VARIANTS]
              + [f"sim_{v}_{p}" for v in VARIANTS for p in people])
    raw_folder.mkdir(parents=True, exist_ok=True)
    with open(raw_folder / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.6f}" if isinstance(x, float) else x) for k, x in r.items()})
    (raw_folder / "settings.json").write_text(json.dumps({
        "fps": args.fps, "det_model": args.det_model, "det_sizes": det_sizes, "every": every, "variants": VARIANTS,
        "calib_seconds": CALIB_SECONDS, "cosine_threshold": COSINE_THRESHOLD,
        "models_lock": json.loads((pl.repo_folder / "configs" / "models.lock.json").read_text()),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")
    print(f"Wrote {raw_folder / 'frames.csv'} ({len(rows)} rows)")


# ---------- stage 2: thresholds and metrics ----------

def choose_thresholds(live, attack, fp32_live, fp32_attack, is_fp32=False):
    """Thresholds for 'live if score >= t' under each rule (calibration scores, faces only).
    FP32 is the reference of both parity rules, so it keeps 0 there by definition (the lowest/highest
    tied threshold would otherwise move it to the next calibration score)."""
    cand = np.unique(np.concatenate([live, attack, [0.0]]))
    cand = np.concatenate([cand, [np.nextafter(cand.max(), np.inf)]])
    apcer = np.array([(attack >= t).mean() for t in cand])
    bpcer = np.array([(live < t).mean() for t in cand])
    fp32_apcer, fp32_bpcer = (fp32_attack >= 0).mean(), (fp32_live < 0).mean()
    return {
        "default": 0.0,
        "apcer_parity": 0.0 if is_fp32 else float(cand[np.argmax(apcer <= fp32_apcer)]),  # lowest t meeting it
        "bpcer_parity": 0.0 if is_fp32 else float(cand[np.nonzero(bpcer <= fp32_bpcer)[0].max()]),  # highest
        "calib_eer": ev.eer(live, attack)[1],
    }


def analyse():
    rows = list(csv.DictReader(open(raw_folder / "frames.csv")))
    settings_info = json.loads((raw_folder / "settings.json").read_text())
    people = sorted({r["person"] for r in rows})
    for r in rows:
        r["has_face"] = r["has_face"] == "1"
        r["det"], r["k"] = int(r["det"]), int(r["k"])
    settings = sorted({(r["det"], r["k"]) for r in rows}, key=lambda s: (-s[0], s[1]))

    def live_scores(sel, v):
        return np.array([float(r[f"live_{v}"]) for r in sel if r["has_face"]])

    threshold_rows, thresholds = [], {}
    for d, k in settings:
        calib = [r for r in rows if r["det"] == d and r["k"] == k and r["split"] == "calib"]
        c_live = [r for r in calib if r["kind"] == "live"]
        c_att = [r for r in calib if r["kind"] != "live"]
        f_live, f_att = live_scores(c_live, "fp32"), live_scores(c_att, "fp32")
        for v in VARIANTS:
            lv, at = live_scores(c_live, v), live_scores(c_att, v)
            th = choose_thresholds(lv, at, f_live, f_att, is_fp32=(v == "fp32"))
            for rule in RULES:
                thresholds[(d, k, v, rule)] = th[rule]
                threshold_rows.append({"det": d, "k": k, "liveness": v, "rule": rule, "threshold": round(th[rule], 4),
                                       "calib_live_faces": len(lv), "calib_attack_faces": len(at),
                                       "calib_bpcer": round(float((lv < th[rule]).mean()), 4),
                                       "calib_apcer": round(float((at >= th[rule]).mean()), 4)})
    processed_folder.mkdir(parents=True, exist_ok=True)
    ev.write_csv(processed_folder / "thresholds.csv", threshold_rows)

    # Liveness and combined metrics on test frames (no face = reject)
    def is_live(r, v, t):
        return r["has_face"] and float(r[f"live_{v}"]) >= t

    def matches(r, v, person):
        return r["has_face"] and float(r[f"sim_{v}_{person}"]) >= COSINE_THRESHOLD

    groups = {"all": lambda r: True, "lighting=bright": lambda r: r["lighting"] == "bright",
              "lighting=dim": lambda r: r["lighting"] == "dim"}
    live_rows, comb_rows = [], []
    for dataset in ["own", "session2"]:
        for d, k in settings:
            test = [r for r in rows if r["dataset"] == dataset and r["det"] == d and r["k"] == k
                    and r["split"] == "test"]
            for v in VARIANTS:
                for rule in RULES:
                    t = thresholds[(d, k, v, rule)]
                    for g, in_g in groups.items():
                        sel = [r for r in test if in_g(r)]
                        lv = [r for r in sel if r["kind"] == "live"]
                        at = [r for r in sel if r["kind"] != "live"]
                        bp = sum(not is_live(r, v, t) for r in lv)
                        ap = sum(is_live(r, v, t) for r in at)
                        live_rows.append({"dataset": dataset, "det": d, "k": k, "liveness": v, "rule": rule,
                                          "threshold": round(t, 4), "group": g,
                                          "apcer_errors": ap, "attack_frames": len(at),
                                          "apcer": round(ap / len(at), 5) if at else "",
                                          "bpcer_errors": bp, "live_frames": len(lv),
                                          "bpcer": round(bp / len(lv), 5) if lv else "",
                                          "no_face_frames": sum(not r["has_face"] for r in sel)})
                    # Combined: same precision on both models, as in the power sweep
                    frr = imp = spoof = n_gen = n_imp = n_spoof = 0
                    for person in people:
                        for r in test:
                            acc = is_live(r, v, t) and matches(r, v, person)
                            if r["kind"] == "live" and r["person"] == person:
                                n_gen += 1
                                frr += not acc
                            elif r["kind"] == "live":
                                n_imp += 1
                                imp += acc
                            elif r["person"] == person:
                                n_spoof += 1
                                spoof += acc
                    comb_rows.append({"dataset": dataset, "det": d, "k": k, "precision": v, "rule": rule,
                                      "threshold": round(t, 4),
                                      "frr_errors": frr, "genuine_attempts": n_gen,
                                      "frr": round(frr / n_gen, 5) if n_gen else "",
                                      "impostor_far_errors": imp, "impostor_attempts": n_imp,
                                      "impostor_far": round(imp / n_imp, 5) if n_imp else "",
                                      "spoof_far_errors": spoof, "spoof_attempts": n_spoof,
                                      "spoof_far": round(spoof / n_spoof, 5) if n_spoof else ""})
    ev.write_csv(processed_folder / "liveness.csv", live_rows)
    ev.write_csv(processed_folder / "combined.csv", comb_rows)

    # Summary
    fr = ev.fmt_rate
    s = ["# Detect-every-K on sequential video, and liveness thresholds tuned on calibration frames", "",
         f"Generated by `eval/sequential_eval.py`. Team videos played in frame order at "
         f"{settings_info['fps']} processed frames/s; detector (FP32) every K-th processed frame, box reused in "
         "between. No face = reject. Rates: 95% Wilson intervals and counts; consecutive frames are strongly "
         "correlated, so the intervals are very optimistic. Same precision on both models.", "",
         "Session-1 test frames share videos (and the screen-replay attack type) with the calibration window; "
         "session 2 (later day, paper-photo attacks never seen in calibration) is the cleaner test.", "",
         "## Thresholds (chosen on the calibration window)", "",
         "| Detector | K | Precision | default | apcer_parity | bpcer_parity | calib_eer |", "|---|---|---|---|---|---|---|"]
    for d, k in settings:
        for v in VARIANTS:
            s.append(f"| {d} | {k} | {v} | " + " | ".join(f"{thresholds[(d, k, v, r)]:+.3f}" for r in RULES) + " |")
    s.append("")
    for dataset in ["session2", "own"]:
        s += [f"## Liveness, {dataset} test frames, group = all", "",
              "| Detector | K | Precision | Rule | Threshold | APCER | BPCER | No face |", "|---|---|---|---|---|---|---|---|"]
        for r in live_rows:
            if r["dataset"] == dataset and r["group"] == "all":
                s.append(f"| {r['det']} | {r['k']} | {r['liveness']} | {r['rule']} | {r['threshold']:+.3f} | "
                         f"{fr(r['apcer_errors'], r['attack_frames'])} | {fr(r['bpcer_errors'], r['live_frames'])} | "
                         f"{r['no_face_frames']} |")
        s.append("")
        s += [f"## Combined pipeline, {dataset} test frames", "",
              "| Detector | K | Precision | Rule | FRR | Impostor FAR | Spoof FAR |", "|---|---|---|---|---|---|---|"]
        for r in comb_rows:
            if r["dataset"] == dataset:
                s.append(f"| {r['det']} | {r['k']} | {r['precision']} | {r['rule']} | "
                         f"{fr(r['frr_errors'], r['genuine_attempts'])} | "
                         f"{fr(r['impostor_far_errors'], r['impostor_attempts'])} | "
                         f"{fr(r['spoof_far_errors'], r['spoof_attempts'])} |")
        s.append("")
    (processed_folder / "summary.md").write_text("\n".join(s) + "\n", encoding="utf-8")
    print(f"Wrote {processed_folder / 'summary.md'}")


def analyse_lopo():
    """Leave-one-person-out on session 2: tune each precision's liveness threshold on the session-2
    frames of the other people (same rules as analyse(), FP32 at 0 as the reference), apply it to
    the held-out person's session-2 frames, and pool the held-out results over all folds. Only the
    tuning data changes versus analyse(). Session 2 stops being purely test data here: each person's
    frames are used for tuning in the folds where they are not held out."""
    rows = list(csv.DictReader(open(raw_folder / "frames.csv")))
    people = sorted({r["person"] for r in rows})
    for r in rows:
        r["has_face"] = r["has_face"] == "1"
        r["det"], r["k"] = int(r["det"]), int(r["k"])
    rows = [r for r in rows if r["dataset"] == "session2"]
    settings = sorted({(r["det"], r["k"]) for r in rows}, key=lambda s: (-s[0], s[1]))

    def scores(sel, v):
        return np.array([float(r[f"live_{v}"]) for r in sel if r["has_face"]])

    def is_live(r, v, t):
        return r["has_face"] and float(r[f"live_{v}"]) >= t

    fold_rows, pooled = [], {}
    for d, k in settings:
        setting_rows = [r for r in rows if r["det"] == d and r["k"] == k]
        for held in people:
            tune = [r for r in setting_rows if r["person"] != held]
            test = [r for r in setting_rows if r["person"] == held]
            t_live = [r for r in tune if r["kind"] == "live"]
            t_att = [r for r in tune if r["kind"] != "live"]
            f_live, f_att = scores(t_live, "fp32"), scores(t_att, "fp32")
            for v in VARIANTS:
                th = choose_thresholds(scores(t_live, v), scores(t_att, v), f_live, f_att, is_fp32=(v == "fp32"))
                for rule in RULES:
                    t = th[rule]
                    lv = [r for r in test if r["kind"] == "live"]
                    at = [r for r in test if r["kind"] != "live"]
                    c = {"bp": sum(not is_live(r, v, t) for r in lv), "n_live": len(lv),
                         "ap": sum(is_live(r, v, t) for r in at), "n_att": len(at), "frr": 0, "n_gen": 0,
                         "imp": 0, "n_imp": 0, "spoof": 0, "n_spoof": 0}
                    for person in people:
                        for r in test:
                            acc = is_live(r, v, t) and r["has_face"] and float(r[f"sim_{v}_{person}"]) >= COSINE_THRESHOLD
                            if r["kind"] == "live" and person == held:
                                c["n_gen"] += 1
                                c["frr"] += not acc
                            elif r["kind"] == "live":
                                c["n_imp"] += 1
                                c["imp"] += acc
                            elif person == held:
                                c["n_spoof"] += 1
                                c["spoof"] += acc
                    fold_rows.append({"det": d, "k": k, "precision": v, "rule": rule, "held_out": held,
                                      "threshold": round(t, 4), **c})
                    agg = pooled.setdefault((d, k, v, rule), {key: 0 for key in c})
                    for key in c:
                        agg[key] += c[key]
    ev.write_csv(processed_folder / "lopo_folds.csv", fold_rows)
    pooled_rows = []
    for (d, k, v, rule), c in pooled.items():
        ts = [f["threshold"] for f in fold_rows if (f["det"], f["k"], f["precision"], f["rule"]) == (d, k, v, rule)]
        pooled_rows.append({"det": d, "k": k, "precision": v, "rule": rule,
                            "threshold_min": min(ts), "threshold_max": max(ts),
                            "apcer": round(c["ap"] / c["n_att"], 5), "apcer_errors": c["ap"], "attack_frames": c["n_att"],
                            "bpcer": round(c["bp"] / c["n_live"], 5), "bpcer_errors": c["bp"], "live_frames": c["n_live"],
                            "frr": round(c["frr"] / c["n_gen"], 5), "frr_errors": c["frr"], "genuine_attempts": c["n_gen"],
                            "impostor_far_errors": c["imp"], "impostor_attempts": c["n_imp"],
                            "spoof_far": round(c["spoof"] / c["n_spoof"], 5), "spoof_far_errors": c["spoof"],
                            "spoof_attempts": c["n_spoof"]})
    ev.write_csv(processed_folder / "lopo.csv", pooled_rows)

    fr = ev.fmt_rate
    s = ["# Session 2, leave-one-person-out liveness thresholds", "",
         "Generated by `eval/sequential_eval.py` (`analyse_lopo`). For each held-out person, every precision's "
         "threshold is tuned on the other people's session-2 frames (same rules as summary.md; FP32 stays at 0 under "
         "the parity rules) and applied to the held-out person; results pooled over the folds. Only "
         f"{len(people)} people, so each threshold rests on two people: read this as a feasibility check.", "",
         "| Detector | K | Precision | Rule | Thresholds over folds | APCER | BPCER | FRR | Impostor FAR | Spoof FAR |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in pooled_rows:
        s.append(f"| {r['det']} | {r['k']} | {r['precision']} | {r['rule']} | "
                 f"{r['threshold_min']:+.2f} to {r['threshold_max']:+.2f} | "
                 f"{fr(r['apcer_errors'], r['attack_frames'])} | {fr(r['bpcer_errors'], r['live_frames'])} | "
                 f"{fr(r['frr_errors'], r['genuine_attempts'])} | {fr(r['impostor_far_errors'], r['impostor_attempts'])} | "
                 f"{fr(r['spoof_far_errors'], r['spoof_attempts'])} |")
    s += ["", "## Per held-out person (APCER / BPCER)", "",
          "| Detector | K | Precision | Rule | Held out | Threshold | APCER | BPCER |", "|---|---|---|---|---|---|---|---|"]
    for f in fold_rows:
        s.append(f"| {f['det']} | {f['k']} | {f['precision']} | {f['rule']} | {f['held_out']} | {f['threshold']:+.2f} | "
                 f"{fr(f['ap'], f['n_att'])} | {fr(f['bp'], f['n_live'])} |")
    (processed_folder / "lopo_summary.md").write_text("\n".join(s) + "\n", encoding="utf-8")
    print(f"Wrote {processed_folder / 'lopo_summary.md'}")


SMOOTH_WINDOWS = [1, 3, 5, 10]


def analyse_smoothing():
    """Liveness score smoothing (Mabel's --smooth-liveness): the decision uses the mean liveness score of
    the last W processed frames of the same video, causal, cleared when no face is found (as
    pipeline.verify_camera does) and at the calibration/test boundary. Threshold: the default 0 for
    every precision. W = 1 is the unsmoothed result. At 10 processed frames/s, W = 10 is 1 s."""
    rows = list(csv.DictReader(open(raw_folder / "frames.csv")))
    people = sorted({r["person"] for r in rows})
    for r in rows:
        r["has_face"] = r["has_face"] == "1"
        r["det"], r["k"], r["frame"] = int(r["det"]), int(r["k"]), int(r["frame"])
    groups = {}
    for r in rows:
        groups.setdefault((r["source"], r["split"], r["det"], r["k"]), []).append(r)
    for seq in groups.values():
        seq.sort(key=lambda r: r["frame"])
        for v in VARIANTS:
            for w in SMOOTH_WINDOWS:
                window = []
                for r in seq:
                    if not r["has_face"]:
                        window.clear()
                        r[f"sm_{v}_{w}"] = None
                        continue
                    window.append(float(r[f"live_{v}"]))
                    del window[:-w]
                    r[f"sm_{v}_{w}"] = sum(window) / len(window)

    out = []
    for dataset in ["session2", "own"]:
        present = {(r["det"], r["k"]) for r in rows}
        for (d, k) in [s for s in [(640, 1), (320, 1), (320, 5)] if s in present]:
            test = [r for r in rows if r["dataset"] == dataset and r["det"] == d and r["k"] == k and r["split"] == "test"]
            for v in VARIANTS:
                for w in SMOOTH_WINDOWS:
                    live_ok = lambda r: r[f"sm_{v}_{w}"] is not None and r[f"sm_{v}_{w}"] >= 0.0
                    lv = [r for r in test if r["kind"] == "live"]
                    at = [r for r in test if r["kind"] != "live"]
                    c = {"bp": sum(not live_ok(r) for r in lv), "n_live": len(lv),
                         "ap": sum(live_ok(r) for r in at), "n_att": len(at),
                         "frr": 0, "n_gen": 0, "imp": 0, "n_imp": 0, "spoof": 0, "n_spoof": 0}
                    for person in people:
                        for r in test:
                            acc = live_ok(r) and float(r[f"sim_{v}_{person}"]) >= COSINE_THRESHOLD
                            if r["kind"] == "live" and r["person"] == person:
                                c["n_gen"] += 1
                                c["frr"] += not acc
                            elif r["kind"] == "live":
                                c["n_imp"] += 1
                                c["imp"] += acc
                            elif r["person"] == person:
                                c["n_spoof"] += 1
                                c["spoof"] += acc
                    out.append({"dataset": dataset, "det": d, "k": k, "precision": v, "window": w,
                                "apcer": round(c["ap"] / c["n_att"], 5), "bpcer": round(c["bp"] / c["n_live"], 5),
                                "frr_rate": round(c["frr"] / c["n_gen"], 5),
                                "spoof_far": round(c["spoof"] / c["n_spoof"], 5), **c})
    ev.write_csv(processed_folder / "smoothing.csv", out)
    fr = ev.fmt_rate
    s = ["# Liveness score smoothing over frames", "",
         "Generated by `eval/sequential_eval.py` (`analyse_smoothing`). Decision on the mean liveness score of the "
         "last W processed frames (10 frames/s, causal, cleared on a missing face); threshold 0 for every precision; "
         "same precision on both models. W = 1 is unsmoothed. Consecutive frames are correlated, so the Wilson "
         "intervals are very optimistic.", "",
         "| Dataset | Detector | K | Precision | W | APCER | BPCER | FRR | Impostor FAR | Spoof FAR |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        s.append(f"| {r['dataset']} | {r['det']} | {r['k']} | {r['precision']} | {r['window']} | "
                 f"{fr(r['ap'], r['n_att'])} | {fr(r['bp'], r['n_live'])} | {fr(r['frr'], r['n_gen'])} | "
                 f"{fr(r['imp'], r['n_imp'])} | {fr(r['spoof'], r['n_spoof'])} |")
    (processed_folder / "smoothing_summary.md").write_text("\n".join(s) + "\n", encoding="utf-8")
    print(f"Wrote {processed_folder / 'smoothing_summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fps", type=float, default=10.0, help="processed frames per second of video")
    ap.add_argument("--det-sizes", default="640,320")
    ap.add_argument("--every", default="1,3,5")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--max-videos", type=int, help="smoke test on the first N videos only")
    ap.add_argument("--det-model", choices=["fp32", "int8nodw"], default="fp32",
                    help="detector precision; int8nodw writes to sequential_det-int8nodw folders (320 only)")
    args = ap.parse_args()
    global raw_folder, processed_folder
    if args.det_model != "fp32":
        raw_folder = raw_folder.with_name(f"sequential_det-{args.det_model}")
        processed_folder = processed_folder.with_name(f"sequential_det-{args.det_model}")
    if not args.analyse_only:
        score(args)
    analyse()
    analyse_lopo()
    analyse_smoothing()


if __name__ == "__main__":
    main()
