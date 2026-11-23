from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Optional
import unicodedata

from .policy import GENERAL, LABELS, POLICY_VERSION, Profile, parse_input
from .sparse import scores_to_probabilities


MAX_CHARS = 12_000
DEFAULT_QUESTIONS = {"decision": {"type": "choice"}, "route": {"type": "choice"}}
QUESTION_TYPES = {"decision": ("choice", "noul"), "route": ("choice",)}


@dataclass(frozen=True)
class Thresholds:
    accept: float = 0.9
    refuse: float = 0.9
    action: float = 0.75

    def __post_init__(self):
        if not (0.5 <= self.accept <= 1 and 0.5 <= self.refuse <= 1 and 0 <= self.action <= 1):
            raise ValueError("Thresholds are out of range")


def candidate(probabilities, profile):
    actions = defaultdict(float)
    accept = 0.0
    for label in LABELS:
        probability = probabilities[label]
        actions[profile.action(label)] += probability
        if profile.accepts(label):
            accept += probability
    accept = min(1.0, accept)
    decisions = {"accept": accept, "refuse": 1 - accept}
    decision = max(decisions, key=decisions.__getitem__)
    allowed = [label for label in LABELS if profile.accepts(label) == (decision == "accept")]
    action = max({profile.action(label): actions[profile.action(label)] for label in allowed}, key=actions.__getitem__)
    route = max((label for label in allowed if profile.action(label) == action), key=probabilities.__getitem__)
    return decision, decisions, action, actions[action], route


def prediction(probabilities, profile, thresholds):
    decision, decisions, action, action_score, route = candidate(probabilities, profile)
    if decisions[decision] < getattr(thresholds, decision) or action_score < thresholds.action:
        return None, None, "review", decisions, action_score
    return decision, route, action, decisions, action_score


def input_limit(value):
    chars = len(value["request"]) + sum(len(turn["text"]) for turn in value["history"])
    if value["document"]:
        chars += len(value["document"].get("excerpt", "")) + len(value["document"].get("name", ""))
    if chars > MAX_CHARS or len(value["history"]) > 4:
        return "input_limit"
    letters = [letter for letter in value["request"] if letter.isalpha()]
    if letters:
        latin = sum(unicodedata.name(letter, "").startswith("LATIN ") for letter in letters)
        if latin * 2 < len(letters):
            return "unsupported_script"
    if not value["request"].strip() and not value["history"] and value["document"] is None:
        return "missing_request"
    return None


class Gate:
    def __init__(self, model, profile: Profile = GENERAL, thresholds: Optional[Thresholds] = None):
        if not math.isfinite(model.temperature) or model.temperature <= 0:
            raise ValueError("Model temperature must be finite and positive")
        self.model = model
        self.profile = profile
        settings = model.thresholds.get(profile.jurisdiction, {})
        self.thresholds = thresholds or Thresholds(**settings)

    def decide(self, payload):
        if not isinstance(payload, dict) or payload.keys() - {"input", "questions"} or "input" not in payload:
            raise ValueError("expected input and optional fixed decision and route questions")
        questions = payload.get("questions", DEFAULT_QUESTIONS)
        if (
            not isinstance(questions, dict) or not questions or questions.keys() - QUESTION_TYPES.keys()
            or any(not isinstance(spec, dict) or spec.keys() != {"type"} or spec["type"] not in QUESTION_TYPES[key]
                   for key, spec in questions.items())
        ):
            raise ValueError("supported questions are decision (choice or noul) and route (choice)")
        value = parse_input(payload["input"])
        probabilities = None
        reason = input_limit(value)
        if reason is None:
            if not value["request"].strip() and value["document"] and not value["history"]:
                probabilities = {label: float(label == "clarify") for label in LABELS}
            else:
                scores = self.model.logits(value)
                if scores is None:
                    reason = "input_limit"
                elif len(scores) != len(LABELS) or any(not math.isfinite(score) for score in scores):
                    reason = "model_error"
                else:
                    probabilities = scores_to_probabilities(scores, self.model.temperature)

        def answer(choice, options):
            return {
                "type": "choice", "choice": choice, "probabilities": options,
                "confidence": options[choice] if choice is not None else None,
            }

        if probabilities is None:
            action = "review"
            answers = {question: answer(None, None) for question in questions}
        else:
            decision, route, action, decisions, _ = prediction(probabilities, self.profile, self.thresholds)
            if action == "review":
                reason = "low_confidence"
            elif action == "file_error" and (not value["document"] or value["document"]["status"] == "available"):
                action, reason, decision, route = "review", "file_state_conflict", None, None
            elif action == "greeting" and value["document"] and not value["history"]:
                action = "clarify"
            complete = {"decision": answer(decision, decisions), "route": answer(route, probabilities)}
            answers = {question: complete[question] for question in questions}
        if questions.get("decision", {}).get("type") == "noul":
            decision_answer = answers["decision"]
            scores = decision_answer["probabilities"]
            answers["decision"] = {
                "type": "noul", "noul": scores["accept"] if scores else None,
                "confidence": decision_answer["confidence"],
            }
        return {
            "policy_version": POLICY_VERSION, "profile": self.profile.name, "action": action,
            "review_reason": reason, "answers": answers,
        }
