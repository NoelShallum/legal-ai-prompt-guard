"""Quantized encoder inference on a single CPU thread."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .policy import LABELS, POLICY_VERSION, encoder_text


class EncoderModel:
    def __init__(self, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if tuple(metadata["labels"]) != LABELS or metadata["policy_version"] != POLICY_VERSION:
            raise ValueError("Incompatible encoder model artifact")
        self.temperature = metadata.get("temperature", 1.0)
        self.thresholds = metadata.get("thresholds", {})
        self.max_length = metadata["max_length"]
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / "model.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {item.name for item in self.session.get_inputs()}

    def logits(self, value):
        request, context = encoder_text(value)
        encoded = self.tokenizer.encode(request, context)
        if len(encoded.ids) > self.max_length:
            return None
        columns = {
            "input_ids": [encoded.ids], "attention_mask": [encoded.attention_mask],
            "token_type_ids": [encoded.type_ids],
        }
        arrays = {name: np.asarray(columns[name], dtype=np.int64) for name in self.inputs}
        return self.session.run(None, arrays)[0][0].tolist()
