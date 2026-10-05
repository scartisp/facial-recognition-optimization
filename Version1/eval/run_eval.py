"""Accuracy evaluation of the 3 x 3 quantization ablation.

Run after eval/prepare_data.py and quantize/quantize.py (from Version1, venv active):
    python eval/run_eval.py

Because the detector is fixed and the crops are cached, each model variant only has to run
once; the 9 combined-pipeline cells are then built from the per-model outputs.

Writes (no images, only scores and metrics):
    results/raw/lfw_pair_scores.csv        cosine similarity per LFW pair, per recognizer variant
    results/raw/liveness_scores.csv        liveness logit difference per face, per liveness variant
    results/raw/embedding_fidelity.csv     cosine(FP32 embedding, INT8 embedding) per face
    results/raw/combined_attempts.csv      per attempt: similarity and liveness score per variant
    results/processed/*.csv + summary.md   metrics
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pipeline as pl  # noqa: E402

version1_folder = pl.version1_folder
cache_folder = version1_folder / "private" / "cache"
results_raw = version1_folder / "results" / "raw"
results_processed = version1_folder / "results" / "processed"
ablation_folder = pl.repo_folder / "configs" / "ablation"

VARIANTS = ["fp32", "int8pt", "int8pc"]
VARIANT_LABEL = {"fp32": "FP32", "int8pt": "INT8 per-tensor", "int8pc": "INT8 per-channel"}
TARGET_FAR = 0.001
BATCH = 64


# ---------- metrics ----------

def wilson(errors, total, z=1.96):
    """95% Wilson interval for a rate. Frames from one video are correlated, so treat as optimistic."""
    if total == 0:
        return (math.nan, math.nan)
    p = errors / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centre - half), min(1.0, centre + half))


def eer(genuine, impostor):
    """Equal error rate and its threshold for 'accept if score >= t'."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thresholds[i])


def threshold_at_far(impostor, target):
    """Smallest threshold with FAR <= target for 'accept if score >= t'."""
    allowed = int(math.floor(target * len(impostor)))
    ordered = np.sort(impostor)[::-1]
    return float(np.nextafter(ordered[allowed], np.inf))


def roc_points(genuine, impostor):
    thresholds = np.unique(np.concatenate([genuine, impostor]))[::-1]
    return [((impostor >= t).mean(), (genuine >= t).mean(), t) for t in thresholds]


def auc(genuine, impostor):
    """Probability a random genuine score beats a random impostor score (ties count half)."""
    g = np.asarray(genuine)[:, None]
    i = np.asarray(impostor)[None, :]
    return float((g > i).mean() + 0.5 * (g == i).mean())


# ---------- model runs ----------

def load_cache(name):
    data = np.load(cache_folder / f"{name}.npz")
    return data["rec"], data["live"], [json.loads(m) for m in data["meta"]]


def variant_configs():
    """Map variant -> config that uses that variant for the role (other role irrelevant)."""
    configs = {}
    for v in VARIANTS:
        configs[v] = json.loads((ablation_folder / f"rec-{v}_live-{v}.json").read_text())
        pl.check_pins(configs[v])
    return configs


class ModelRunner:
    def __init__(self, config):
        models = config["models"]
        self.config = config
        self.recognizer = pl.new_session(pl.resolve_path(models["recognizer"]["path"]))
        self.liveness = pl.new_session(pl.resolve_path(models["liveness"]["path"]))
        self.rec_input = self.recognizer.get_inputs()[0].name
        self.live_input = self.liveness.get_inputs()[0].name
        # Reuse the pipeline's own normalisation code without loading a detector
        self.prep = pl.Pipeline.__new__(pl.Pipeline)
        self.prep.config = config

    def embed(self, rec_crops):
        out = []
        for start in range(0, len(rec_crops), BATCH):
            blob = self.prep.recognizer_blob(list(rec_crops[start:start + BATCH]))
            e = self.recognizer.run(None, {self.rec_input: blob})[0]
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def liveness_scores(self, live_crops):
        out = []
        for start in range(0, len(live_crops), BATCH):
            blob = pl.Pipeline.liveness_blob(live_crops[start:start + BATCH])
            logits = self.liveness.run(None, {self.live_input: blob})[0]
            out.append(logits[:, 0] - logits[:, 1])
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_rate(errors, total):
    if total == 0:
        return "n/a"
    lo, hi = wilson(errors, total)
    return f"{100 * errors / total:.1f}% [{100 * lo:.1f}-{100 * hi:.1f}] ({errors}/{total})"


# ---------- main ----------

def main():
    configs = variant_configs()
    runners = {v: ModelRunner(configs[v]) for v in VARIANTS}
    live_threshold = math.log(configs["fp32"]["liveness"]["real_prob_threshold"]
                              / (1 - configs["fp32"]["liveness"]["real_prob_threshold"]))
    summary = ["# Quantization ablation: accuracy results", ""]
    summary.append("Generated by `eval/run_eval.py`. Rates show 95% Wilson intervals and counts; frames from the "
                   "same video are correlated, so the intervals are optimistic.")
    summary.append("")

    # ---- LFW verification ----
    lfw_rec, _, lfw_meta = load_cache("lfw")
    pairs = json.loads((cache_folder.parent / "lfw_pairs.json").read_text())
    index = {m["image_id"]: i for i, m in enumerate(lfw_meta)}
    usable = [p for p in pairs if p["a"] in index and p["b"] in index]
    a_idx = np.array([index[p["a"]] for p in usable])
    b_idx = np.array([index[p["b"]] for p in usable])
    same = np.array([p["same"] for p in usable], dtype=bool)

    lfw_emb = {v: runners[v].embed(lfw_rec) for v in VARIANTS}
    sims = {v: np.einsum("ij,ij->i", lfw_emb[v][a_idx], lfw_emb[v][b_idx]) for v in VARIANTS}

    write_csv(results_raw / "lfw_pair_scores.csv", (
        {"pair": f"{p['a']}|{p['b']}", "same": int(p["same"]), **{f"sim_{v}": f"{sims[v][i]:.6f}" for v in VARIANTS}}
        for i, p in enumerate(usable)))

    fp32_threshold = threshold_at_far(sims["fp32"][~same], TARGET_FAR)
    verification_rows, roc_rows = [], []
    for v in VARIANTS:
        gen, imp = sims[v][same], sims[v][~same]
        e, e_t = eer(gen, imp)
        own_t = threshold_at_far(imp, TARGET_FAR)
        verification_rows.append({
            "recognizer": v,
            "pairs_used": len(usable), "pairs_total": len(pairs),
            "eer": round(e, 5), "eer_threshold": round(e_t, 5),
            "auc": round(auc(gen, imp), 6),
            "tar_at_far_0.1pct_own_threshold": round(float((gen >= own_t).mean()), 5),
            "own_threshold_far_0.1pct": round(own_t, 5),
            "far_at_fp32_threshold": round(float((imp >= fp32_threshold).mean()), 5),
            "frr_at_fp32_threshold": round(float((gen < fp32_threshold).mean()), 5),
            "fp32_threshold": round(fp32_threshold, 5),
        })
        roc_rows += [{"recognizer": v, "far": f"{f:.6f}", "tar": f"{t:.6f}", "threshold": f"{th:.6f}"}
                     for f, t, th in roc_points(gen, imp)]
    write_csv(results_processed / "verification_lfw.csv", verification_rows)
    write_csv(results_processed / "roc_lfw.csv", roc_rows)

    summary += ["## Verification (LFW deep-funneled, 6000 pairs)", "",
                f"Pairs with a face detected in both images: {len(usable)}/{len(pairs)} (same pairs for every variant).",
                f"Operating threshold: FP32 threshold at FAR = 0.1% on LFW = **{fp32_threshold:.4f}** "
                "(kept fixed for INT8, as a deployed system would).", "",
                "| Recognizer | EER | AUC | TAR @ FAR=0.1% (own threshold) | FAR @ FP32 threshold | FRR @ FP32 threshold |",
                "|---|---|---|---|---|---|"]
    for r in verification_rows:
        summary.append(f"| {VARIANT_LABEL[r['recognizer']]} | {100 * r['eer']:.2f}% | {r['auc']:.4f} | "
                       f"{100 * r['tar_at_far_0.1pct_own_threshold']:.2f}% | {100 * r['far_at_fp32_threshold']:.3f}% | "
                       f"{100 * r['frr_at_fp32_threshold']:.2f}% |")
    n_imp = int((~same).sum())
    summary += ["", f"FAR = 0.1% rests on about {int(TARGET_FAR * n_imp)} false accepts out of {n_imp} impostor pairs, "
                "so TAR @ FAR=0.1% is noisy.", ""]

    # ---- embedding fidelity (no labels needed) ----
    fidelity_rows, fidelity_summary = [], []
    datasets_rec = {"lfw": lfw_rec}
    own_rec, own_live, own_meta = load_cache("own")
    kg_rec, kg_live, kg_meta = load_cache("kaggle")
    datasets_rec["own"] = own_rec
    datasets_rec["kaggle"] = kg_rec
    emb = {"lfw": lfw_emb}
    for name in ("own", "kaggle"):
        emb[name] = {v: runners[v].embed(datasets_rec[name]) for v in VARIANTS}
    for name in ("lfw", "own", "kaggle"):
        for v in ("int8pt", "int8pc"):
            cos = np.einsum("ij,ij->i", emb[name]["fp32"], emb[name][v])
            fidelity_summary.append({"dataset": name, "recognizer": v, "faces": len(cos),
                                     "mean_cos_to_fp32": round(float(cos.mean()), 5),
                                     "p05_cos_to_fp32": round(float(np.percentile(cos, 5)), 5),
                                     "min_cos_to_fp32": round(float(cos.min()), 5)})
            fidelity_rows += [{"dataset": name, "index": i, "recognizer": v, "cos_to_fp32": f"{c:.6f}"}
                              for i, c in enumerate(cos)]
    write_csv(results_raw / "embedding_fidelity.csv", fidelity_rows)
    write_csv(results_processed / "embedding_fidelity.csv", fidelity_summary)
    summary += ["## Recognizer fidelity: cosine(FP32 embedding, INT8 embedding) on the same face", "",
                "| Dataset | Recognizer | Faces | Mean | 5th percentile | Min |", "|---|---|---|---|---|---|"]
    summary += [f"| {r['dataset']} | {VARIANT_LABEL[r['recognizer']]} | {r['faces']} | {r['mean_cos_to_fp32']:.4f} | "
                f"{r['p05_cos_to_fp32']:.4f} | {r['min_cos_to_fp32']:.4f} |" for r in fidelity_summary]
    summary.append("")

    # ---- liveness ----
    msu_rec, msu_live, msu_meta = load_cache("msu")
    live_sets = {"own (test split)": (own_live, own_meta, lambda m: m["split"] == "test"),
                 "msu": (msu_live, msu_meta, lambda m: True),
                 "kaggle": (kg_live, kg_meta, lambda m: True)}
    live_scores, liveness_raw = {}, []
    for name, (crops, meta, keep) in live_sets.items():
        idx = np.array([i for i, m in enumerate(meta) if keep(m)])
        scores = {v: runners[v].liveness_scores(crops[idx]) for v in VARIANTS}
        live_scores[name] = (idx, scores)
        for j, i in enumerate(idx):
            m = meta[i]
            liveness_raw.append({"dataset": name, "index": int(i), "person": m["person"], "kind": m["kind"],
                                 "detail": m.get("detail", ""), "lighting": m.get("lighting", "na"),
                                 "source": m.get("source", ""), "time_s": m.get("time_s", ""),
                                 "face_width_px": m["face_width_px"],
                                 **{f"score_{v}": f"{scores[v][j]:.6f}" for v in VARIANTS}})
    write_csv(results_raw / "liveness_scores.csv", liveness_raw)

    liveness_rows = []
    for name, (idx, scores) in live_scores.items():
        meta = live_sets[name][1]
        kinds = np.array([meta[i]["kind"] for i in idx])
        details = np.array([meta[i].get("detail", "") for i in idx])
        lights = np.array([meta[i].get("lighting", "na") for i in idx])
        is_live = kinds == "live"
        groups = [("all", np.ones(len(idx), bool))]
        groups += [(f"lighting={l}", lights == l) for l in sorted(set(lights)) if l != "na"]
        attack_groups = sorted({(k, d) for k, d in zip(kinds, details) if k != "live"})
        for v in VARIANTS:
            s = scores[v]
            says_live = s >= live_threshold
            flips = int((says_live != (scores["fp32"] >= live_threshold)).sum())
            for gname, g in groups:
                bp_err, bp_n = int((~says_live & is_live & g).sum()), int((is_live & g).sum())
                ap_err, ap_n = int((says_live & ~is_live & g).sum()), int((~is_live & g).sum())
                row = {"dataset": name, "liveness": v, "group": gname,
                       "bpcer_errors": bp_err, "live_faces": bp_n, "apcer_errors": ap_err, "attack_faces": ap_n,
                       "bpcer": round(bp_err / bp_n, 5) if bp_n else "", "apcer": round(ap_err / ap_n, 5) if ap_n else ""}
                if gname == "all":
                    row["eer"] = round(eer(s[is_live], s[~is_live])[0], 5)
                    row["auc"] = round(auc(s[is_live], s[~is_live]), 5)
                    row["decision_flips_vs_fp32"] = flips
                liveness_rows.append(row)
            for kind, detail in attack_groups:
                g = (kinds == kind) & (details == detail)
                liveness_rows.append({"dataset": name, "liveness": v, "group": f"attack={kind}/{detail}" if detail else f"attack={kind}",
                                      "apcer_errors": int((says_live & g).sum()), "attack_faces": int(g.sum()),
                                      "apcer": round(float(says_live[g].mean()), 5)})
    fields = ["dataset", "liveness", "group", "bpcer", "bpcer_errors", "live_faces", "apcer", "apcer_errors",
              "attack_faces", "eer", "auc", "decision_flips_vs_fp32"]
    write_csv(results_processed / "liveness.csv", liveness_rows, fields)

    summary += ["## Liveness", "",
                f"Threshold: real probability >= {configs['fp32']['liveness']['real_prob_threshold']} (facenox default). "
                "APCER = attacks accepted as live (the security failure); BPCER = live faces rejected as spoofs.", ""]
    for name in live_scores:
        summary += [f"### {name}", "", "| Liveness | Group | APCER | BPCER | EER | AUC | Flips vs FP32 |",
                    "|---|---|---|---|---|---|---|"]
        for r in (r for r in liveness_rows if r["dataset"] == name):
            ap = fmt_rate(r["apcer_errors"], r["attack_faces"]) if r.get("attack_faces") else "n/a"
            bp = fmt_rate(r["bpcer_errors"], r["live_faces"]) if r.get("live_faces") else "n/a"
            if r["group"].startswith("attack="):
                bp = ""
            eer_txt = f"{100 * r['eer']:.1f}%" if r.get("eer") != "" and "eer" in r else ""
            auc_txt = f"{r['auc']:.3f}" if "auc" in r else ""
            summary.append(f"| {VARIANT_LABEL[r['liveness']]} | {r['group']} | {ap} | {bp} | {eer_txt} | {auc_txt} | "
                           f"{r.get('decision_flips_vs_fp32', '')} |")
        summary.append("")

    # ---- combined pipeline ----
    combined_attempts = []

    def add_attempts(dataset, meta, emb_by_v, live_by_v, live_index, enrollments, probes):
        """enrollments: person -> list of cache indices; probes: list of cache indices (test faces)."""
        pos = {int(i): j for j, i in enumerate(live_index)}
        for person, enroll_idx in enrollments.items():
            templates = {}
            for v in VARIANTS:
                t = emb_by_v[v][enroll_idx].mean(axis=0)
                templates[v] = t / np.linalg.norm(t)
            for i in probes:
                m = meta[i]
                if m["kind"] == "live":
                    kind = "genuine" if m["person"] == person else "impostor"
                elif m["person"] == person:
                    kind = "spoof"
                else:
                    continue
                combined_attempts.append({
                    "dataset": dataset, "enrolled": person, "probe": m["person"], "attempt": kind,
                    "presentation": m["kind"], "detail": m.get("detail", ""), "lighting": m.get("lighting", "na"),
                    **{f"sim_{v}": float(emb_by_v[v][i] @ templates[v]) for v in VARIANTS},
                    **{f"live_{v}": float(live_by_v[v][pos[i]]) for v in VARIANTS},
                })

    own_idx, own_scores = live_scores["own (test split)"]
    own_enroll = {}
    for i, m in enumerate(own_meta):
        if m["split"] == "calib" and m["kind"] == "live" and m["lighting"] == "bright":
            own_enroll.setdefault(m["person"], []).append(i)
    add_attempts("own", own_meta, emb["own"], own_scores, own_idx, own_enroll, list(own_idx))

    kg_idx, kg_scores = live_scores["kaggle"]
    kg_enroll = {m["person"]: [i] for i, m in enumerate(kg_meta) if m.get("detail") == "selfie"}
    kg_probes = [int(i) for i in kg_idx if kg_meta[i].get("detail") != "selfie" and kg_meta[i]["person"] in kg_enroll]
    add_attempts("kaggle", kg_meta, emb["kaggle"], kg_scores, kg_idx, kg_enroll, kg_probes)

    write_csv(results_raw / "combined_attempts.csv", (
        {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in a.items()} for a in combined_attempts))

    combined_rows = []
    for dataset in ("own", "kaggle"):
        attempts = [a for a in combined_attempts if a["dataset"] == dataset]
        lightings = sorted({a["lighting"] for a in attempts} - {"na"})
        for rv in VARIANTS:
            for lv in VARIANTS:
                for group in ["all"] + [f"lighting={l}" for l in lightings]:
                    chosen = [a for a in attempts if group == "all" or a["lighting"] == group.split("=")[1]]
                    accept = lambda a: a[f"sim_{rv}"] >= fp32_threshold and a[f"live_{lv}"] >= live_threshold
                    row = {"dataset": dataset, "recognizer": rv, "liveness": lv, "group": group}
                    for kind, rate_name, want_accept in (("genuine", "frr", True), ("impostor", "impostor_far", False),
                                                         ("spoof", "spoof_far", False)):
                        sel = [a for a in chosen if a["attempt"] == kind]
                        errors = sum(1 for a in sel if accept(a) != want_accept)
                        row[f"{rate_name}_errors"] = errors
                        row[f"{kind}_attempts"] = len(sel)
                        row[rate_name] = round(errors / len(sel), 5) if sel else ""
                    combined_rows.append(row)
    write_csv(results_processed / "combined.csv", combined_rows)

    summary += ["## Combined pipeline (liveness AND verification)", "",
                f"Accept if cosine similarity >= {fp32_threshold:.4f} (FP32 LFW threshold at FAR=0.1%) and liveness "
                "says real. FRR = genuine user rejected; impostor FAR = a different live person accepted; "
                "spoof FAR = a photo/screen of the enrolled user accepted.", "",
                "Team videos: each person is enrolled from the held-out first 4 s of their bright live video and "
                "acts as the impostor for the other. Kaggle: each person is enrolled from their single selfie.", ""]
    for dataset in ("own", "kaggle"):
        summary += [f"### {dataset}", "", "| Recognizer | Liveness | Group | FRR | Impostor FAR | Spoof FAR |",
                    "|---|---|---|---|---|---|"]
        for r in (r for r in combined_rows if r["dataset"] == dataset):
            summary.append(f"| {VARIANT_LABEL[r['recognizer']]} | {VARIANT_LABEL[r['liveness']]} | {r['group']} | "
                           f"{fmt_rate(r['frr_errors'], r['genuine_attempts'])} | "
                           f"{fmt_rate(r['impostor_far_errors'], r['impostor_attempts'])} | "
                           f"{fmt_rate(r['spoof_far_errors'], r['spoof_attempts'])} |")
        summary.append("")

    results_processed.mkdir(parents=True, exist_ok=True)
    (results_processed / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote {results_processed / 'summary.md'}")


if __name__ == "__main__":
    main()
