"""Audit label provenance, split overlap and the actual routing contract."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .data import check_splits, load_jsonl, save_jsonl
from .policy import INDIA, PROFILES, encoder_text
from .routing import Gate
from .sparse import SparseModel
from .train import dump_json, load_splits


def label_audit(splits, teachers, minimum_confidence):
    reports = {}
    disputes = []
    for name, rows in splits.items():
        counts = Counter()
        confusions = Counter()
        for row in rows:
            teacher = teachers.get(row["id"])
            if teacher is None:
                counts["missing"] += 1
                continue
            counts["annotated"] += 1
            expected = row["label"]
            proposed = teacher["label"]
            low_confidence = teacher["confidence"] < minimum_confidence
            decision_match = INDIA.accepts(expected) == INDIA.accepts(proposed)
            counts["decision_match"] += decision_match
            counts["route_match"] += expected == proposed
            counts["low_confidence"] += low_confidence
            if not decision_match or expected != proposed or low_confidence:
                disputes.append({
                    **row, "author_label": expected, "teacher_label": proposed,
                    "teacher_confidence": teacher["confidence"], "decision_disagreement": not decision_match,
                    "needs_review": True,
                })
            if expected != proposed:
                confusions[expected, proposed] += 1
        reports[name] = {"rows": len(rows), **dict(counts), "disagreements": [
            {"authored": reference, "gemini": label, "n": count}
            for (reference, label), count in confusions.most_common()
        ]}
    disputes.sort(key=lambda row: (not row["decision_disagreement"], row["split"] == "train", row["teacher_confidence"]))
    return reports, disputes


def similarity_audit(splits):
    def text(row):
        return " ".join(encoder_text(row["input"])).strip()

    training = [row for row in splits["train"] if len(text(row)) >= 35]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, dtype=np.float32)
    train_vectors = vectorizer.fit_transform(map(text, training))
    audit = {}
    matches = []
    for key, rows in splits.items():
        if key == "train":
            continue
        heldout = [row for row in rows if len(text(row)) >= 35]
        similarity = linear_kernel(vectorizer.transform(map(text, heldout)), train_vectors)
        best = similarity.argmax(axis=1)
        maxima = similarity[np.arange(len(heldout)), best]
        audit[key] = {"compared": len(heldout), "at_or_above_0_85": int(np.sum(maxima >= 0.85)),
                      "at_or_above_0_95": int(np.sum(maxima >= 0.95)), "max_similarity": float(maxima.max())}
        matches.extend({"split": key, "evaluation_id": row["id"], "training_id": training[nearest]["id"],
                        "cosine_similarity": float(score)}
                       for row, nearest, score in zip(heldout, best, maxima, strict=True) if score >= 0.85)
    return audit, sorted(matches, key=lambda item: item["cosine_similarity"], reverse=True)


def load_model(path):
    if (path / "model.onnx").is_file():
        from .encoder_runtime import EncoderModel

        return EncoderModel(path)
    return SparseModel(path / "weights.json.gz")


def group_intervals(outcomes):
    per_group = defaultdict(lambda: np.zeros(7, dtype=int))
    for row, action, expected_action, expected_accept, decision in outcomes:
        vector = per_group[row["group"]]
        vector[0] += 1
        vector[1] += action != "review"
        vector[2] += action != "review" and action != expected_action
        vector[3] += not expected_accept
        vector[4] += decision == "accept" and not expected_accept
        vector[5] += expected_accept
        vector[6] += decision == "refuse" and expected_accept
    values = np.array(list(per_group.values()))
    rng = np.random.default_rng(187)
    draws = values[rng.integers(len(values), size=(2_000, len(values)))].sum(axis=1)
    names = (("coverage", 1, 0), ("covered_wrong_action", 2, 1),
             ("false_accept_among_refusals", 4, 3), ("false_refuse_among_accepts", 6, 5))
    return {
        name: np.quantile(draws[draws[:, denominator] > 0, numerator] / draws[draws[:, denominator] > 0, denominator],
                          (0.025, 0.975)).tolist()
        for name, numerator, denominator in names if np.any(draws[:, denominator] > 0)
    }


def serving_audit(model, datasets):
    reports = {}
    mistakes = []
    for profile in PROFILES.values():
        gate = Gate(model, profile)
        reports[profile.name] = {}
        for name, rows in datasets.items():
            if name == "train" or not rows:
                continue
            counts = Counter()
            outcomes = []
            times = []
            for row in rows[:10]:
                gate.decide({"input": row["input"]})
            for row in rows:
                started = perf_counter()
                result = gate.decide({"input": row["input"]})
                times.append((perf_counter() - started) * 1_000)
                action = result["action"]
                decision = result["answers"]["decision"]["choice"]
                expected_accept = profile.accepts(row["label"])
                expected_action = profile.action(row["label"])
                counts["expected_accept" if expected_accept else "expected_refuse"] += 1
                counts[action] += 1
                outcomes.append((row, action, expected_action, expected_accept, decision))
                if action == "review":
                    counts[f"review:{result['review_reason']}"] += 1
                    continue
                counts["covered_expected_accept" if expected_accept else "covered_expected_refuse"] += 1
                counts["auto_correct_action" if action == expected_action else "auto_wrong_action"] += 1
                if action != expected_action:
                    mistakes.append({"id": row["id"], "split": name, "theme": row["theme"], "label": row["label"],
                                     "profile": profile.name, "action": action, "expected": expected_action,
                                     "decision": decision})
                counts["false_accept"] += decision == "accept" and not expected_accept
                counts["false_refuse"] += decision == "refuse" and expected_accept
            coverage = (len(rows) - counts["review"]) / len(rows)
            covered = len(rows) - counts["review"]
            ratio = {}
            if counts["expected_accept"] and counts["expected_refuse"]:
                for accept_prevalence in (0.50, 0.75, 0.90, 0.95):
                    w_accept = accept_prevalence / counts["expected_accept"]
                    w_refuse = (1 - accept_prevalence) / counts["expected_refuse"]
                    ratio[str(accept_prevalence)] = {
                        "modeled_coverage": w_accept * counts["covered_expected_accept"] + w_refuse * counts["covered_expected_refuse"],
                        "modeled_false_refusals_per_1000": 1_000 * w_accept * counts["false_refuse"],
                        "modeled_false_accepts_per_1000": 1_000 * w_refuse * counts["false_accept"],
                    }
            reports[profile.name][name] = {
                "rows": len(rows), "counts": dict(counts), "coverage": coverage,
                "covered_action_accuracy": counts["auto_correct_action"] / covered if covered else None,
                "traffic_mix_projection": ratio,
                "group_bootstrap_95": group_intervals(outcomes),
                "warm_api_ms": {"p50": float(np.median(times)), "p95": float(np.quantile(times, 0.95))},
            }
    return reports, mistakes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--models", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.dataset)
    check_splits(splits)
    similarities, matches = similarity_audit(splits)
    dump_json(args.output / "data-audit.json", similarities)
    save_jsonl(args.output / "near-overlap.jsonl", matches)
    if args.teacher:
        teacher = {row["id"]: row for row in load_jsonl(args.teacher)}
        counts, disagreements = label_audit(splits, teacher, args.minimum_confidence)
        dump_json(args.output / "labels-audit.json", counts)
        save_jsonl(args.output / "review-queue.jsonl", disagreements)
    for path in args.models:
        reports, errors = serving_audit(load_model(path), splits)
        dump_json(args.output / f"{path.name}-serving.json", reports)
        save_jsonl(args.output / f"{path.name}-errors.jsonl", errors)
    print(json.dumps({"split_sizes": {name: len(rows) for name, rows in splits.items()}, "near_overlap": similarities}))


if __name__ == "__main__":
    main()
