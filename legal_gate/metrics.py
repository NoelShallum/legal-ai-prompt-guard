from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import product
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score, f1_score

from .policy import LABELS
from .routing import Thresholds, candidate


LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}


def temperature_scale(logits, rows):
    targets = np.array([LABEL_INDEX[row["label"]] for row in rows])

    def loss(log_temperature):
        scaled = logits / math.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))

    result = minimize_scalar(loss, bounds=(math.log(0.2), math.log(6.0)), method="bounded")
    return math.exp(result.x)


def probabilities(logits, temperature):
    return [dict(zip(LABELS, row, strict=True)) for row in softmax(logits / temperature, axis=1)]


def estimates(probs, rows, profile):
    return [
        (candidate(scores, profile), profile.accepts(row["label"]), profile.action(row["label"]))
        for scores, row in zip(probs, rows, strict=True)
    ]


def selective_counts(estimates_by_row, thresholds):
    counts = Counter()
    for (decision, dist, action, action_score, _), expected_accept, expected_action in estimates_by_row:
        counts["actual_accept" if expected_accept else "actual_refuse"] += 1
        if dist[decision] < getattr(thresholds, decision) or action_score < thresholds.action:
            counts["review"] += 1
            continue
        counts["covered"] += 1
        counts[f"auto_{decision}"] += 1
        if (decision == "accept") != expected_accept:
            counts["false_accept" if decision == "accept" else "false_refuse"] += 1
        if action != expected_action:
            counts["wrong_action"] += 1
    return counts


def tune_thresholds(probs, rows, profile, max_false_accept=0.025, max_false_refuse=0.025):
    scored = estimates(probs, rows, profile)
    best = None
    best_key = None
    levels = (0.50, 0.60, 0.70, 0.78, 0.85, 0.89, 0.92, 0.95, 0.97, 0.985, 0.995, 1.0)
    for accept, refuse, action in product(levels, levels, (0.50, 0.60, 0.70, 0.78, 0.85, 0.92, 0.97)):
        thresholds = Thresholds(accept, refuse, action)
        c = selective_counts(scored, thresholds)
        if (
            c["false_accept"] / max(1, c["actual_refuse"]) > max_false_accept
            or c["false_refuse"] / max(1, c["actual_accept"]) > max_false_refuse
            or c["wrong_action"] / max(1, c["covered"]) > 0.05
        ):
            continue
        key = (c["covered"], -c["wrong_action"], -c["false_accept"], -c["false_refuse"])
        if best_key is None or key > best_key:
            best = thresholds
            best_key = key
    return best or Thresholds(1.0, 1.0, 1.0)


def rate(pos, total):
    return float(pos / total) if total else None


def wilson(pos, total):
    if not total:
        return None
    p = pos / total
    z = 1.95996398454
    scale = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / scale
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / scale
    return [max(0.0, center - margin), min(1.0, center + margin)]


def report(probs, rows, profile, thresholds):
    scored = estimates(probs, rows, profile)
    binary = Counter()
    family = defaultdict(list)
    slice_totals = defaultdict(Counter)
    predicted = []
    truth = []
    expected_refuse = []
    p_refuse = []
    confusion = defaultdict(Counter)
    for item, row, scores in zip(scored, rows, probs, strict=True):
        (decision, distribution, action, action_score, _), expected_accept, expected_action = item
        top_label = max(scores, key=scores.__getitem__)
        accepted = decision == "accept"
        binary["ta" if accepted else "tr"] += int(accepted == expected_accept)
        binary["fa" if accepted else "fr"] += int(accepted != expected_accept)
        expected_refuse.append(not expected_accept)
        p_refuse.append(distribution["refuse"])
        family[row["group"]].append((expected_accept, accepted == expected_accept))
        stats = slice_totals[row["theme"]]
        stats["n"] += 1
        stats["false_accept"] += int(accepted and not expected_accept)
        stats["false_refuse"] += int(not accepted and expected_accept)
        stats["wrong_action"] += int(action != expected_action)
        if distribution[decision] >= getattr(thresholds, decision) and action_score >= thresholds.action:
            stats["auto"] += 1
            stats["auto_wrong_action"] += int(action != expected_action)
        truth.append(row["label"])
        predicted.append(top_label)
        confusion[row["label"]][top_label] += 1

    fa, fr, ta, tr = (binary[key] for key in ("fa", "fr", "ta", "tr"))
    selective = selective_counts(scored, thresholds)
    contrast = [items for items in family.values() if {row[0] for row in items} == {True, False}]
    fpr = rate(fa, fa + tr)
    fnr = rate(fr, fr + ta)
    return {
        "n": len(rows),
        "gate_accuracy": rate(ta + tr, len(rows)),
        "refuse_precision": rate(tr, tr + fr),
        "refuse_recall": rate(tr, tr + fa),
        "refuse_average_precision": float(average_precision_score(expected_refuse, p_refuse)),
        "false_accept": {"n": fa, "total": fa + tr, "rate": fpr, "wilson_95": wilson(fa, fa + tr)},
        "false_refuse": {"n": fr, "total": fr + ta, "rate": fnr, "wilson_95": wilson(fr, fr + ta)},
        "modeled_error_by_accept_prevalence": {
            str(prior): (1 - prior) * fpr + prior * fnr
            for prior in (0.50, 0.75, 0.90, 0.95) if fpr is not None and fnr is not None
        },
        "gate_brier": float(np.mean((np.array(p_refuse) - np.array(expected_refuse)) ** 2)),
        "route_macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "route_confusion": {key: dict(value) for key, value in confusion.items()},
        "contrast_groups": len(contrast),
        "contrast_group_accuracy": rate(sum(all(row[1] for row in items) for items in contrast), len(contrast)),
        "thresholds": asdict(thresholds),
        "selective": {
            **dict(selective),
            "coverage": rate(selective["covered"], len(rows)),
            "selective_action_error_rate": rate(selective["wrong_action"], selective["covered"]),
            "auto_false_accept_95": wilson(selective["false_accept"], selective["actual_refuse"]),
            "auto_false_refuse_95": wilson(selective["false_refuse"], selective["actual_accept"]),
        },
        "themes": {key: dict(value) for key, value in slice_totals.items()},
    }
