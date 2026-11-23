from dataclasses import dataclass
from typing import Literal


POLICY_VERSION = "legal-scope-v1"
LABELS = (
    "legal", "adjacent", "clarify", "file_error", "greeting", "courtesy",
    "product", "public_identity", "social", "foreign", "mixed_jurisdiction",
    "unrelated", "mixed", "protected", "private",
)
FOREIGN_LABELS = frozenset({"foreign", "mixed_jurisdiction"})
REFUSAL_ACTIONS = {
    "social": "scope_redirect",
    "foreign": "refuse_scope",
    "mixed_jurisdiction": "refuse_scope",
    "unrelated": "refuse_scope",
    "mixed": "refuse_scope",
    "protected": "refuse_internal",
    "private": "refuse_private",
}
ACCEPT_ACTIONS = {
    "legal": "legal",
    "adjacent": "legal",
    "clarify": "clarify",
    "file_error": "file_error",
    "greeting": "greeting",
    "courtesy": "courtesy",
    "product": "product",
    "public_identity": "public_identity",
    "foreign": "legal",
    "mixed_jurisdiction": "legal",
}


@dataclass(frozen=True)
class Profile:
    name: str
    jurisdiction: Literal["india", "any"] = "any"

    def __post_init__(self):
        if self.jurisdiction not in {"india", "any"}:
            raise ValueError("jurisdiction must be 'india' or 'any'")

    def accepts(self, label: str) -> bool:
        if label not in LABELS:
            raise ValueError(f"Unknown policy label {label!r}")
        return label not in REFUSAL_ACTIONS or (
            self.jurisdiction == "any" and label in FOREIGN_LABELS
        )

    def action(self, label: str) -> str:
        return (ACCEPT_ACTIONS if self.accepts(label) else REFUSAL_ACTIONS)[label]


INDIA = Profile("india_legal", "india")
GENERAL = Profile("general_legal")
PROFILES = {profile.name: profile for profile in (INDIA, GENERAL)}


def parse_input(value) -> dict:
    if isinstance(value, list):
        if not value or any(
            not isinstance(part, dict) or part.keys() != {"type", "text"}
            or part["type"] != "text" or not isinstance(part["text"], str)
            for part in value
        ):
            raise ValueError("input arrays must contain text parts")
        value = "\n".join(part["text"] for part in value)
    if isinstance(value, str):
        value = {"request": value}
    if not isinstance(value, dict) or value.keys() - {"request", "history", "document"}:
        raise ValueError("input must be text, text parts, or a request/history/document object")
    request = value.get("request", "")
    history = value.get("history", [])
    document = value.get("document")
    if not isinstance(request, str):
        raise ValueError("request must be a string")
    if not isinstance(history, list):
        raise ValueError("history must be an array")
    for turn in history:
        if (
            not isinstance(turn, dict) or turn.keys() != {"role", "text"}
            or turn["role"] not in ("user", "assistant")
            or not isinstance(turn["text"], str)
        ):
            raise ValueError("history entries need a user or assistant role and text")
    if document is not None:
        if (
            not isinstance(document, dict) or document.keys() - {"status", "name", "excerpt"}
            or document.get("status") not in ("available", "empty", "failed", "unsupported")
            or any(not isinstance(item, str) for item in document.values())
        ):
            raise ValueError("document needs a valid status and optional name and excerpt")
    return {"request": request, "history": history, "document": document}


def encoder_text(value: dict) -> tuple[str, str]:
    history = value["history"]
    document = value["document"]
    context = [f"Earlier {turn['role']}: {turn['text']}" for turn in history]
    if document is not None:
        context.append(f"Attachment status: {document['status']}")
        if document.get("name"):
            context.append(f"Attachment name: {document['name']}")
        if document.get("excerpt"):
            context.append(f"Attachment excerpt: {document['excerpt']}")
    return value["request"], "\n".join(context)
