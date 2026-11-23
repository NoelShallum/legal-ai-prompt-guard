"""Expand the pilot to 10,000 examples with new training-only scenario families."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from .data import SPLITS, check_splits, deduplicate, fingerprint, from_families, load_jsonl, save_jsonl, summaries
from .policy import INDIA, POLICY_VERSION, parse_input


STARTS_WITH_REFERENCE = re.compile(r"^(also|anyway|continue|go ahead|i meant|same|yes|no|that|then|this)\b", re.I)
UNRESOLVED_REFERENCE = re.compile(r"\b(attached|above|those files|second one|same document|last answer|earlier message)\b", re.I)


def standalone(row):
    value = row["input"]
    request = value["request"]
    return (
        not value["history"] and value["document"] is None and 6 <= len(request.split()) <= 55
        and not STARTS_WITH_REFERENCE.search(request) and not UNRESOLVED_REFERENCE.search(request)
    )


def clean(value):
    return value.strip().rstrip(".!?;")


class Composer:
    def __init__(self, rows, seed):
        self.rng = random.Random(seed)
        self.labels = defaultdict(list)
        self.documents = []
        for row in rows:
            if standalone(row):
                self.labels[row["label"]].append(row)
            doc = row["input"]["document"]
            if row["label"] == "legal" and doc and doc["status"] == "available" and doc.get("excerpt"):
                self.documents.append(row)
        self.standalone = {
            accepts: [row for label, examples in self.labels.items() if INDIA.accepts(label) == accepts for row in examples]
            for accepts in (True, False)
        }

    def combined(self, accept):
        first = self.rng.choice(self.labels["legal"])
        other_label = "legal" if accept else self.rng.choices(("unrelated", "foreign"), weights=(4, 1))[0]
        second = self.rng.choice(self.labels[other_label])
        if first["group"] == second["group"]:
            return None
        if self.rng.choice((True, False)):
            first, second = second, first
        x, y = (row["input"]["request"].strip() for row in (first, second))
        request = self.rng.choice((
            f"{x}\n\n{y}", f"{x} Separately, {y}", f"First, {x}\nNext, {y}",
            f"1. {x}\n2. {y}", f"{x} I also have a second request. {y}",
            f"I have two tasks. {x} {y}", f"{clean(x)}. Also, {y}",
        ))
        label = {"legal": "legal", "unrelated": "mixed", "foreign": "mixed_jurisdiction"}[other_label]
        return {"request": request}, label, "mixed_operations", (first, second)

    def followup(self, accept):
        labels = ("legal",) if accept else ("unrelated", "mixed", "foreign", "mixed_jurisdiction")
        label = self.rng.choice(labels)
        task = self.rng.choice(self.labels[label])
        original = task["input"]["request"].strip()
        history = self.rng.choice((
            [{"role": "user", "text": original}, {"role": "assistant", "text": "Should I proceed with the task you just described?"}],
            [{"role": "assistant", "text": f"Do you want me to address this request next?\n{original}"}],
            [{"role": "user", "text": "Here is the task I have in mind. " + original},
             {"role": "assistant", "text": "Are you asking me to act on that now?"}],
        ))
        request = self.rng.choice(("Yes", "Please go ahead", "yeah, do that", "Go on with it", "Yes, that one", "yep", "Okay, proceed", "please proceed with it"))
        return {"request": request, "history": history}, label, "followups", (task,)

    def pivot(self, accept):
        eligible = [row for row in self.standalone[accept] if row["label"] in {
            "legal", "adjacent", "unrelated", "mixed", "foreign", "mixed_jurisdiction", "protected", "private"
        }]
        current = self.rng.choice(eligible)
        previous = self.rng.choice(self.labels["legal"])
        if current["group"] == previous["group"]:
            return None
        request = current["input"]["request"]
        history = [{"role": "user", "text": previous["input"]["request"]}]
        history.append({"role": "assistant", "text": self.rng.choice((
            "Which part of the legal question should we take first?",
            "I can start by explaining the issue and the options available.",
            "What would you like to cover next?",
        ))})
        if self.rng.choice((True, False)):
            request = self.rng.choice(("Changing topics. ", "A new question. ", "Separate from that, ", "On another matter, ")) + request
        return {"request": request, "history": history}, current["label"], "followups", (current, previous)

    def evidence(self, accept):
        parent = self.rng.choice(self.documents)
        document = dict(parent["input"]["document"])
        if accept:
            label, requests = "legal", (
                "Where could this help or hurt in an Indian dispute? Explain the evidentiary points.",
                "Pull out the points from this item that counsel should verify before relying on it.",
                "Does this record support a claim or a defence under Indian law? Flag key uncertainties.",
                "What's the legal significance of this record? Give the relevant facts in order.",
                "Build a short issue-and-evidence table for a lawyer reviewing the attached record.",
                "Please prepare a legal chronology from the attachment, including any dates that would need checking.",
                "For an Indian contract or complaint, what facts in this record are useful and what facts are missing?",
                "Analyse the document as evidence for a possible legal claim. Which statements could we substantiate?",
                "Could this support a legal complaint? Extract the parts that go to rights, duties, notice or loss.",
                "Summarise this for use in discussing the legal dispute with counsel.",
            )
        else:
            label, requests = "unrelated", (
                "Rewrite this attachment as a funny country song for a family get-together.",
                "Turn the attached material into a crossword. Use only ordinary vocabulary.",
                "Can you turn the text in this file into a short cartoon scene?",
                "Use the attachment as inspiration for a café logo and come up with a colour palette.",
                "I need a birthday parody based on the attached text. Make it sound playful.",
                "Turn the document into a quiz on spelling and grammar for my class.",
                "Draft a completely fictional space adventure using a few nouns from this file.",
                "Which phrases from the attachment would make catchy names for a fantasy football team?",
                "Make a warm-up improv game from this document in a couple of bullets.",
                "Use this file to invent a dinner-party theme and a playlist.",
            )
        return {"request": self.rng.choice(requests), "document": document}, label, "document_evidence", (parent,)

    def generate(self, accept):
        recipe = self.rng.choices(
            (self.combined, self.followup, self.pivot, self.evidence), weights=(0.30, 0.25, 0.27, 0.18)
        )[0]
        case = recipe(accept)
        if case is None:
            return None
        value, label, theme, parents = case
        value = parse_input(value)
        parent_ids = sorted(row["id"] for row in parents)
        group = fingerprint({"recipe": recipe.__name__, "groups": sorted(row["group"] for row in parents)})
        return {
            "id": fingerprint(value), "input": value, "label": label, "theme": theme,
            "group": f"generated/{group}", "source": "scenario_generator", "policy_version": POLICY_VERSION,
            "recipe": recipe.__name__, "parents": parent_ids, "split": "train",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("datasets/pilot"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--themes", type=Path, default=Path("themes.json"))
    parser.add_argument("--output", type=Path, default=Path("datasets/expanded"))
    parser.add_argument("--seed", type=int, default=529)
    args = parser.parse_args()
    matrix = json.loads(args.themes.read_text(encoding="utf-8"))
    themes = {row["id"] for row in matrix["themes"]}
    splits = {key: load_jsonl(args.pilot / f"{key}.jsonl") for key in SPLITS}
    if (args.pilot / "challenge.jsonl").is_file():
        splits["challenge"] = load_jsonl(args.pilot / "challenge.jsonl")
    used = {row["id"] for rows in splits.values() for row in rows}
    paths = sorted(args.data.glob("expansion_*.jsonl"))
    raw = [row for path in paths for row in from_families(path, "terra", themes)]
    authored, cleaning = deduplicate(raw)
    fresh = [dict(row, split="train") for row in authored if row["id"] not in used]
    used.update(row["id"] for row in fresh)
    cleaning["pilot_overlap"] = len(authored) - len(fresh)
    composer = Composer([row for row in splits["train"] if row["source"] != "augmentation"] + fresh, args.seed)
    additional = matrix["expanded_size"] - matrix["pilot_size"]
    goals = {True: round(additional * 0.60), False: additional - round(additional * 0.60)}
    counts = Counter(INDIA.accepts(row["label"]) for row in fresh)
    for accept, goal in goals.items():
        for _ in range(goal * 15):
            if counts[accept] >= goal:
                break
            row = composer.generate(accept)
            if row is None or row["id"] in used:
                continue
            fresh.append(row)
            used.add(row["id"])
            counts[accept] += 1
        if counts[accept] != goal:
            raise ValueError(f"Could not fill expansion quota {accept}: {counts[accept]}/{goal}")
    random.Random(args.seed).shuffle(fresh)
    splits["train"].extend(fresh)
    check_splits(splits)
    if sum(len(splits[key]) for key in SPLITS) != matrix["expanded_size"]:
        raise ValueError("Size mismatch after expansion")
    args.output.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output / "new-train.jsonl", fresh)
    for key, rows in splits.items():
        save_jsonl(args.output / f"{key}.jsonl", rows)
    report = {"seed": args.seed, "policy_version": POLICY_VERSION, "cleaning": cleaning,
              "new_sources": dict(Counter(row["source"] for row in fresh)),
              "new_recipes": dict(Counter(row["recipe"] for row in fresh if row.get("recipe"))),
              "splits": summaries(splits)}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"new_training_rows": len(fresh), **report["new_sources"], **report["new_recipes"]}))


if __name__ == "__main__":
    main()
