"""Assemble reproducible, scenario-grouped policy datasets."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from .policy import INDIA, LABELS, POLICY_VERSION, parse_input


SPLITS = ("train", "dev", "calibration", "test")
EXPECTED_SOURCE_NAMES = ("authority", "boundaries", "documents", "regulatory", "trust")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def fingerprint(value) -> str:
    def normalized(entry):
        if isinstance(entry, dict):
            return {key: normalized(item) for key, item in entry.items()}
        if isinstance(entry, list):
            return [normalized(item) for item in entry]
        if isinstance(entry, str):
            return " ".join(WORD_RE.findall(unicodedata.normalize("NFKC", entry).casefold()))
        return entry

    payload = json.dumps(normalized(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def from_families(path, source, themes):
    groups = set()
    for family in load_jsonl(path):
        theme = family["theme"]
        group = f"{Path(path).stem}/{family['group']}"
        if theme not in themes or group in groups:
            raise ValueError(f"Invalid or repeated family {group}")
        groups.add(group)
        for sample in family["examples"]:
            label = sample["label"]
            if label not in LABELS:
                raise ValueError(f"Unknown label {label!r} in {group}")
            value = parse_input({key: sample[key] for key in ("request", "history", "document") if key in sample})
            if not (value["request"].strip() or value["document"] or value["history"]):
                raise ValueError(f"Empty example in {group}")
            if label == "file_error" and (
                not value["document"] or value["document"]["status"] == "available"
            ):
                raise ValueError(f"A file-error example needs unavailable document state in {group}")
            yield {
                "id": fingerprint(value), "input": value, "label": label, "theme": theme,
                "group": group, "source": source, "policy_version": POLICY_VERSION,
            }


def deduplicate(rows):
    seen = {}
    conflicts = set()
    for row in rows:
        match = seen.setdefault(row["id"], row)
        if match["label"] != row["label"]:
            conflicts.add(row["id"])
    unique = [row for key, row in seen.items() if key not in conflicts]
    return unique, {"duplicates": len(rows) - len(seen), "conflicting_inputs": len(conflicts)}


def holdouts(rows, seed):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["theme"]][row["group"]].append(row)
    rng = random.Random(seed)
    splits = {name: [] for name in SPLITS}
    for families in grouped.values():
        batch = list(families.values())
        rng.shuffle(batch)
        goal = max(1, round(sum(map(len, batch)) / 8))
        # Use authored inputs for development, calibration and test. The training
        # set receives the later augmentation needed to meet the theme quotas.
        for name in ("dev", "calibration", "test"):
            count = 0
            while len(batch) > 2 and count < goal:
                family = batch.pop()
                splits[name].extend(family)
                count += len(family)
        for family in batch:
            splits["train"].extend(family)
    return splits


PREFIXES = (
    "Quick question. ", "A separate query. ", "Could you help with this? ", "For my notes, ",
    "I have this request. ", "Here's what I need. ", "My question is below.\n", "One thing. ",
    "Please consider the following. ", "Here is the task.\n", "Can you take a look at this? ",
    "I would appreciate help with this. ", "Short question. ", "I need help with the following.\n",
    "I have a query about this. ", "Could you clarify something for me? ",
)
SUFFIXES = (" Thanks.", " Please.", " I need this today.", " Can you help?", " That's all.")


def augment(row, variant):
    value = dict(row["input"])
    request = value["request"].strip()
    label = row["label"]
    if request and label not in {"greeting", "courtesy", "social"}:
        if variant < len(PREFIXES):
            value["request"] = PREFIXES[variant] + request
        else:
            value["request"] = PREFIXES[variant % len(PREFIXES)] + request + SUFFIXES[
                (variant // len(PREFIXES) - 1) % len(SUFFIXES)
            ]
    elif value["document"] is not None and not request:
        value["document"] = {**value["document"], "name": f"attachment_{variant + 1:02d}.txt"}
    elif request:
        if label == "greeting":
            alternatives = ("Hello there", "Good day", "Hey there", "Morning to you", "Greetings", "Hello again")
        elif label == "courtesy":
            alternatives = ("Thank you for that", "Much appreciated", "Got it, thank you", "Thanks for your help")
        else:
            alternatives = ("Let's just hang out here", "Can we chat about our day", "I just want company", "I'll leave this now, bye")
        value["request"] = alternatives[variant % len(alternatives)]
    return {**row, "id": fingerprint(value), "input": value, "source": "augmentation", "parent": row["id"]}


def assemble(seeds, matrix, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    quotas = {}
    for theme in matrix["themes"]:
        quotas[theme["id"], True] = theme["accept"]
        quotas[theme["id"], False] = theme["refuse"]
    for row in seeds:
        buckets[row["theme"], INDIA.accepts(row["label"])].append(row)
    selected = []
    for key, quota in quotas.items():
        candidates = list(buckets[key])
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])

    splits = holdouts(selected, seed)
    counts = Counter((row["theme"], INDIA.accepts(row["label"])) for row in selected)
    used = {row["id"] for row in seeds}
    training_buckets = defaultdict(list)
    for row in splits["train"]:
        training_buckets[row["theme"], INDIA.accepts(row["label"])].append(row)
    for key, quota in quotas.items():
        pool = training_buckets[key]
        rng.shuffle(pool)
        for variant in range(48):
            for row in pool:
                if counts[key] >= quota:
                    break
                new = augment(row, variant)
                if new["id"] in used:
                    continue
                splits["train"].append(new)
                used.add(new["id"])
                counts[key] += 1
            if counts[key] >= quota:
                break
        if counts[key] != quota:
            raise ValueError(f"Insufficient distinct training examples for {key}: {counts[key]}/{quota}")
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def check_splits(splits):
    families = {}
    inputs = {}
    for name, rows in splits.items():
        for row in rows:
            if families.setdefault(row["group"], name) != name:
                raise ValueError(f"Family crosses split boundary: {row['group']}")
            if row["id"] in inputs:
                raise ValueError(f"Duplicated input: {row['id']}")
            inputs[row["id"]] = name


def summaries(splits):
    return {
        name: {
            "rows": len(rows), "groups": len({row["group"] for row in rows}),
            "labels": dict(Counter(row["label"] for row in rows)),
            "themes": dict(Counter(row["theme"] for row in rows)),
            "sources": dict(Counter(row["source"] for row in rows)),
            "indian_decision": dict(Counter("accept" if INDIA.accepts(row["label"]) else "refuse" for row in rows)),
        }
        for name, rows in splits.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--themes", type=Path, default=Path("themes.json"))
    parser.add_argument("--output", type=Path, default=Path("datasets/pilot"))
    parser.add_argument("--seed", type=int, default=187)
    args = parser.parse_args()
    matrix = json.loads(args.themes.read_text(encoding="utf-8"))
    themes = {row["id"] for row in matrix["themes"]}
    paths = [args.data / f"{name}.jsonl" for name in EXPECTED_SOURCE_NAMES]
    raw = [row for path in paths for row in from_families(path, "terra", themes)]
    supplement = args.data / "supplement.jsonl"
    if supplement.exists():
        raw.extend(from_families(supplement, "synthetic_author", themes))
    seeds, cleaning = deduplicate(raw)
    splits = assemble(seeds, matrix, args.seed)
    if sum(map(len, splits.values())) != matrix["pilot_size"]:
        raise ValueError("The theme matrix does not sum to pilot_size")
    challenge = args.data / "challenge.jsonl"
    if challenge.exists():
        challenge_rows, challenge_cleaning = deduplicate(list(from_families(challenge, "challenge", themes)))
        pilot_ids = {row["id"] for rows in splits.values() for row in rows}
        splits["challenge"] = [row for row in challenge_rows if row["id"] not in pilot_ids]
        cleaning["challenge"] = {**challenge_cleaning, "cross_split_duplicates": len(challenge_rows) - len(splits["challenge"])}
    check_splits(splits)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        save_jsonl(args.output / f"{name}.jsonl", [dict(row, split=name) for row in rows])
    report = {"seed": args.seed, "policy_version": POLICY_VERSION, "cleaning": cleaning, "splits": summaries(splits)}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in splits.items()}))


if __name__ == "__main__":
    main()
