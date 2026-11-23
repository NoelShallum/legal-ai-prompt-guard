"""Compact hashed features and a dependency-free integer-weight scorer."""

from array import array
import base64
from collections import Counter
import gzip
import json
import math
import re
import unicodedata
from zlib import crc32

from .policy import LABELS, POLICY_VERSION


WORDS = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _segment(text, prefix, width, weight):
    words = WORDS.findall(unicodedata.normalize("NFKC", text).casefold())
    counts = Counter()

    def add(value):
        digest = crc32((prefix + value).encode())
        index = digest & (width - 1)
        counts[index] += -1 if digest & 0x80000000 else 1

    for word in words:
        add("w:" + word)
        padded = f" {word} "
        for size in (3, 4, 5):
            for start in range(len(padded) - size + 1):
                add("c:" + padded[start:start + size])
    for first, second in zip(words, words[1:]):
        add("b:" + first + " " + second)
    norm = math.sqrt(sum(value ** 2 for value in counts.values())) or 1.0
    return {index: weight * count / norm for index, count in counts.items() if count}


def features(value, width):
    result = Counter(_segment(value["request"], "r/", width, 1.0))
    short = len(value["request"].split()) <= 8
    if value["history"]:
        history = "\n".join(f"{item['role']}: {item['text']}" for item in value["history"])
        result.update(_segment(history, "h/", width, 0.85 if short else 0.45))
    document = value["document"]
    status = document["status"] if document else "absent"
    result[crc32(f"attachment/{status}".encode()) & (width - 1)] += 0.4
    if document and document.get("excerpt"):
        result.update(_segment(document["excerpt"], "d/", width, 0.60 if short else 0.30))
    return result


def scores_to_probabilities(scores, temperature=1.0):
    largest = max(scores)
    exponentials = [math.exp((value - largest) / temperature) for value in scores]
    total = sum(exponentials)
    return {label: value / total for label, value in zip(LABELS, exponentials, strict=True)}


class SparseModel:
    def __init__(self, artifact):
        with gzip.open(artifact, "rt", encoding="utf-8") as handle:
            metadata = json.load(handle)
        kind = metadata.get("kind")
        if (kind not in {"hashed-linear-int8", "hashed-linear-int4"} or metadata.get("policy_version") != POLICY_VERSION
                or tuple(metadata.get("labels", [])) != LABELS):
            raise ValueError("Incompatible sparse model artifact")
        self.width = int(metadata["width"])
        self.weights = []
        for row in metadata["weights"]:
            raw = base64.b64decode(row, validate=True)
            if kind == "hashed-linear-int4":
                nibbles = (nibble for byte in raw for nibble in (byte & 15, byte >> 4))
                raw = array("b", (nibble - 16 if nibble >= 8 else nibble for nibble in nibbles)).tobytes()
            self.weights.append(array("b", raw))
        if self.width < 2 or self.width & (self.width - 1) or any(len(row) != self.width for row in self.weights):
            raise ValueError("Invalid sparse model dimensions")
        self.biases = metadata["biases"]
        self.scales = metadata["scales"]
        if len(self.weights) != len(self.biases) or len(self.weights) != len(self.scales) or len(self.weights) != len(LABELS):
            raise ValueError("Invalid sparse model labels")
        self.temperature = metadata["temperature"]
        self.thresholds = metadata["thresholds"]

    def logits(self, value):
        vector = features(value, self.width)
        return [
            bias + scale * sum(weights[index] * feature for index, feature in vector.items())
            for weights, bias, scale in zip(self.weights, self.biases, self.scales, strict=True)
        ]
