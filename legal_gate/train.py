"""Train, compress and measure a CPU lexical baseline."""

import argparse
import base64
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from .data import SPLITS, load_jsonl
from .metrics import LABEL_INDEX, probabilities, report, temperature_scale, tune_thresholds
from .policy import INDIA, LABELS, POLICY_VERSION, PROFILES
from .sparse import SparseModel, features


def load_splits(directory):
    names = (*SPLITS, "challenge")
    return {name: load_jsonl(directory / f"{name}.jsonl") for name in names if (directory / f"{name}.jsonl").is_file()}


def training_labels(rows, path, minimum_confidence):
    if path is None:
        return rows, {"authored": len(rows)}
    teacher = {
        row["id"]: row for row in load_jsonl(path)
        if row.get("policy_version") == POLICY_VERSION
    }
    status = Counter()
    confirmed = []
    for row in rows:
        annotation = teacher.get(row["id"])
        if annotation is None:
            status["missing"] += 1
        elif annotation["label"] != row["label"]:
            status["disagreement"] += 1
        elif annotation["confidence"] < minimum_confidence:
            status["low_confidence"] += 1
        else:
            status["confirmed"] += 1
            confirmed.append({**row, "label": annotation["label"]})
    return confirmed, dict(status)


def matrix(rows, width):
    indices = []
    data = []
    offsets = [0]
    for row in rows:
        values = features(row["input"], width)
        indices.extend(values)
        data.extend(values.values())
        offsets.append(len(indices))
    return csr_matrix((np.array(data, dtype=np.float32), indices, offsets), shape=(len(rows), width))


def quick_scores(logits, rows):
    acceptable = [index for index, label in enumerate(LABELS) if INDIA.accepts(label)]
    soft = np.exp(logits - logits.max(axis=1, keepdims=True))
    accept_probability = soft[:, acceptable].sum(axis=1) / soft.sum(axis=1)
    gold = np.array([INDIA.accepts(row["label"]) for row in rows])
    decision_accuracy = float(np.mean((accept_probability >= 0.5) == gold))
    route_f1 = float(f1_score([LABEL_INDEX[row["label"]] for row in rows], logits.argmax(axis=1), average="macro", zero_division=0))
    return {"gate_accuracy": decision_accuracy, "route_macro_f1": route_f1}


def evaluate(all_scores, splits):
    temperature = temperature_scale(all_scores["calibration"], splits["calibration"])
    distributions = {name: probabilities(scores, temperature) for name, scores in all_scores.items()}
    profiles = {}
    thresholds = {}
    for profile in PROFILES.values():
        settings = tune_thresholds(distributions["calibration"], splits["calibration"], profile)
        thresholds[profile.jurisdiction] = asdict(settings)
        profiles[profile.name] = {
            name: report(distributions[name], splits[name], profile, settings)
            for name in distributions
        }
    return temperature, thresholds, profiles


def latency_benchmark(model, rows, max_examples=120):
    examples = [row["input"] for row in rows[:max_examples]]
    for value in examples[:10]:
        model.logits(value)
    durations = []
    for value in examples:
        start = perf_counter()
        model.logits(value)
        durations.append((perf_counter() - start) * 1_000)
    context = [duration for duration, value in zip(durations, examples, strict=True) if value["history"] or value["document"]]
    return {
        "examples": len(examples), "p50_ms": float(np.median(durations)),
        "p95_ms": float(np.quantile(durations, 0.95)),
        "context_examples": len(context), "context_p95_ms": float(np.quantile(context, 0.95)) if context else None,
    }


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quantize_coefficients(coefficients, bits):
    maximum = (1 << (bits - 1)) - 1
    scales = np.maximum(np.max(np.abs(coefficients), axis=1) / maximum, 1e-8)
    weights = np.round(coefficients / scales[:, None]).clip(-maximum, maximum).astype(np.int8)
    return weights, scales


def encode_coefficients(weights, bits):
    if bits == 4:
        nibbles = weights.astype(np.uint8) & 15
        weights = nibbles[:, ::2] | (nibbles[:, 1::2] << 4)
    return [base64.b64encode(row.tobytes()).decode() for row in weights]


def train_sparse(splits, output, candidates, teacher=None, minimum_confidence=0.55, max_iterations=250):
    training, provenance = training_labels(splits["train"], teacher, minimum_confidence)
    targets = [LABEL_INDEX[row["label"]] for row in training]
    if set(targets) != set(range(len(LABELS))):
        raise ValueError("Training rows must cover every policy label")
    best = None
    best_key = None
    trials = []
    for width in sorted({item[0] for item in candidates}):
        train_x = matrix(training, width)
        dev_x = matrix(splits["dev"], width)
        for c in [item[1] for item in candidates if item[0] == width]:
            clf = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iterations, random_state=187)
            start = perf_counter()
            with threadpool_limits(limits=2):
                clf.fit(train_x, targets)
            metrics = quick_scores(clf.decision_function(dev_x), splits["dev"])
            metrics.update({"width": width, "c": c, "fit_seconds": perf_counter() - start,
                            "iterations": int(clf.n_iter_.max())})
            trials.append(metrics)
            print(json.dumps(metrics), flush=True)
            key = (0.6 * metrics["gate_accuracy"] + 0.4 * metrics["route_macro_f1"], -width, -c)
            if best_key is None or key > best_key:
                best_key = key
                best = width, c, clf
    width, c, clf = best
    dev_x = matrix(splits["dev"], width)
    quantized = {}
    development = {}
    for bits in (4, 8):
        weights, scales = quantize_coefficients(clf.coef_, bits)
        quantized[bits] = (weights, scales)
        logits = np.asarray(dev_x @ (weights * scales[:, None]).T) + clf.intercept_
        development[bits] = quick_scores(logits, splits["dev"])
    bits = 4 if (
        development[4]["gate_accuracy"] >= development[8]["gate_accuracy"] - 0.005
        and development[4]["route_macro_f1"] >= development[8]["route_macro_f1"] - 0.01
    ) else 8
    kind = f"hashed-linear-int{bits}"
    weights, scales = quantized[bits]
    scores = {
        name: np.asarray(matrix(rows, width) @ (weights * scales[:, None]).T) + clf.intercept_
        for name, rows in splits.items() if name != "train" and rows
    }
    fp32 = clf.decision_function(matrix(splits["test"], width))
    exported = scores["test"]
    quantization = {
        "selected_bits": bits, "development": development,
        "test_route_agreement": float(np.mean(fp32.argmax(axis=1) == exported.argmax(axis=1))),
        "test_max_abs_logit_change": float(np.max(np.abs(fp32 - exported))),
    }
    temperature, thresholds, profiles = evaluate(scores, splits)
    artifact = output / "weights.json.gz"
    metadata = {
        "kind": kind, "policy_version": POLICY_VERSION, "labels": list(LABELS),
        "width": width, "weights": encode_coefficients(weights, bits),
        "scales": scales.tolist(), "biases": clf.intercept_.tolist(), "temperature": temperature,
        "thresholds": thresholds,
    }
    with artifact.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(json.dumps(metadata, separators=(",", ":")).encode())
    model = SparseModel(artifact)
    first_row = splits["dev"][0]
    if not np.allclose(model.logits(first_row["input"]), scores["dev"][0], atol=1e-5):
        raise RuntimeError("Quantized artifact scoring differs from training evaluation")
    stats = {
        "model": kind, "selected": {"width": width, "c": c}, "trials": trials,
        "training_labels": provenance, "training_rows": len(training), "temperature": temperature,
        "artifact_bytes": artifact.stat().st_size, "profiles": profiles, "quantization": quantization,
        "warm_cpu": latency_benchmark(model, splits.get("challenge", splits["test"])),
    }
    dump_json(output / "metrics.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/pilot"))
    parser.add_argument("--output", type=Path, default=Path("models/sparse"))
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    parser.add_argument("--widths", type=int, nargs="+", default=[2 ** power for power in (12, 13, 14, 15)])
    parser.add_argument("--regularization", type=float, nargs="+", default=[2.0, 8.0, 24.0])
    parser.add_argument("--max-iterations", type=int, default=250)
    args = parser.parse_args()
    if any(width < 2 or width & (width - 1) for width in args.widths):
        parser.error("feature widths must be powers of two")
    if args.max_iterations < 1 or any(not np.isfinite(c) or c <= 0 for c in args.regularization):
        parser.error("regularization values and the iteration limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.dataset)
    trials = [(width, c) for width in args.widths for c in args.regularization]
    stats = train_sparse(splits, args.output, trials, args.teacher, args.minimum_confidence, args.max_iterations)
    print(json.dumps({"artifact_bytes": stats["artifact_bytes"], "training_rows": stats["training_rows"],
                      "heldout_accuracy": stats["profiles"][INDIA.name]["test"]["gate_accuracy"]}))


if __name__ == "__main__":
    main()
