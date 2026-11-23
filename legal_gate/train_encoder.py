"""Fine-tune BERT Tiny and evaluate its quantized CPU export."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import random
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .encoder_runtime import EncoderModel
from .metrics import LABEL_INDEX
from .policy import INDIA, LABELS, POLICY_VERSION, encoder_text
from .train import dump_json, evaluate, latency_benchmark, load_splits, quick_scores, training_labels


BASE_MODEL = "google/bert_uncased_L-2_H-128_A-2"
BASE_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"


class LengthBatches:
    def __init__(self, lengths, batch_size, seed):
        ordered = sorted(range(len(lengths)), key=lengths.__getitem__)
        self.batches = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
        self.random = random.Random(seed)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        order = self.random.sample(self.batches, len(self.batches))
        return iter(order)


def encoded_batches(rows, tokenizer, batch_size, max_length, shuffle=False):
    pairs = [encoder_text(row["input"]) for row in rows]
    texts, context = zip(*pairs)
    inputs = tokenizer(list(texts), list(context), truncation=False)
    oversized = sum(len(tokens) > max_length for tokens in inputs["input_ids"])
    if oversized:
        raise ValueError(f"{oversized} training or evaluation inputs exceed max_length={max_length}")
    samples = [
        {**{key: array[index] for key, array in inputs.items()}, "labels": LABEL_INDEX[row["label"]]}
        for index, row in enumerate(rows)
    ]

    def collate(batch):
        tokens = [{key: value for key, value in example.items() if key != "labels"} for example in batch]
        encoded = tokenizer.pad(tokens, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        encoded["labels"] = torch.tensor([example["labels"] for example in batch])
        return encoded

    if shuffle:
        batches = LengthBatches([len(sample["input_ids"]) for sample in samples], batch_size, 187)
        return DataLoader(samples, batch_sampler=batches, collate_fn=collate)
    return DataLoader(samples, batch_size=batch_size, collate_fn=collate)


def model_scores(model, batches, device):
    model.eval()
    scored = []
    with torch.inference_mode():
        for batch in batches:
            tokens = {key: array.to(device) for key, array in batch.items() if key != "labels"}
            scored.append(model(**tokens).logits.cpu().numpy())
    return np.vstack(scored)


def finetune(rows, development, tokenizer, args, learning_rate, device):
    torch.manual_seed(args.seed)
    training = encoded_batches(rows, tokenizer, args.batch_size, args.max_length, shuffle=True)
    validation = encoded_batches(development, tokenizer, args.batch_size, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=args.revision, num_labels=len(LABELS), use_safetensors=True,
        id2label=dict(enumerate(LABELS)), label2id=LABEL_INDEX,
        attn_implementation="eager",
    ).to(device)
    if args.freeze_word_embeddings:
        model.get_input_embeddings().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(training) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    trials = []
    best = None
    best_key = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = perf_counter()
        for batch in training:
            batch = {key: array.to(device) for key, array in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        scores = quick_scores(model_scores(model, validation, device), development)
        scores.update({"epoch": epoch, "learning_rate": learning_rate, "seconds": perf_counter() - start})
        trials.append(scores)
        key = 0.6 * scores["gate_accuracy"] + 0.4 * scores["route_macro_f1"]
        if best_key is None or key > best_key:
            best_key = key
            best = deepcopy(model.state_dict()), scores
        print(json.dumps(scores), flush=True)
    model.load_state_dict(best[0])
    return model, best_key, best[1], trials


def export(model, tokenizer, output, max_length):
    class LogitsOnly(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids).logits

    model = LogitsOnly(model.cpu().eval()).eval()
    encoded = tokenizer("Which Indian filing date applies?", "Earlier assistant: The hearing was last week.", return_tensors="pt")
    names = ("input_ids", "attention_mask", "token_type_ids")
    axes = {name: {0: "batch", 1: "sequence"} for name in names}
    axes["logits"] = {0: "batch"}
    with TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        with torch.inference_mode():
            torch.onnx.export(model, tuple(encoded[name] for name in names), fp32,
                              input_names=list(names), output_names=["logits"],
                              dynamic_axes=axes, opset_version=17, dynamo=False)
        quantize_dynamic(fp32, output / "model.onnx", weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gather"])
    tokenizer.backend_tokenizer.save(str(output / "tokenizer.json"))
    metadata = {"labels": list(LABELS), "policy_version": POLICY_VERSION, "max_length": max_length}
    dump_json(output / "model.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/pilot"))
    parser.add_argument("--output", type=Path, default=Path("models/bert-tiny"))
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[5e-5, 2e-4])
    parser.add_argument("--freeze-word-embeddings", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=187)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    platform_info = {
        "training_device": torch.cuda.get_device_name() if device.type == "cuda" else f"CPU ({platform.machine()})",
        "torch": torch.__version__, "python": platform.python_version(),
    }
    print(json.dumps(platform_info), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.dataset)
    training, provenance = training_labels(splits["train"], args.teacher, args.minimum_confidence)
    if {row["label"] for row in training} != set(LABELS):
        raise ValueError("Training rows must cover every policy label")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=args.revision)
    best = None
    trials = []
    for lr in args.learning_rates:
        model, score, chosen, history = finetune(training, splits["dev"], tokenizer, args, lr, device)
        trials.extend(history)
        if best is None or score > best[0]:
            best = score, chosen, model.cpu()

    metadata = export(best[2], tokenizer, args.output, args.max_length)
    scorer = EncoderModel(args.output)
    scores = {}
    for name, rows in splits.items():
        if name == "train" or not rows:
            continue
        predictions = [scorer.logits(row["input"]) for row in rows]
        if any(row is None for row in predictions):
            raise ValueError(f"The encoder's input limit excludes {name} examples")
        scores[name] = np.asarray(predictions)
    fp32 = model_scores(best[2], encoded_batches(splits["test"], tokenizer, args.batch_size, args.max_length),
                        torch.device("cpu"))
    int8 = scores["test"]
    quantization = {
        "test_route_agreement": float(np.mean(fp32.argmax(axis=1) == int8.argmax(axis=1))),
        "test_max_abs_logit_change": float(np.max(np.abs(fp32 - int8))),
    }
    temperature, thresholds, profiles = evaluate(scores, splits)
    metadata.update({"temperature": temperature, "thresholds": thresholds,
                     "base_model": BASE_MODEL, "base_revision": args.revision})
    dump_json(args.output / "model.json", metadata)
    stats = {
        "model": "bert-tiny-int8", "selected": best[1], "trials": trials,
        "training_labels": provenance, "training_rows": len(training), "temperature": temperature,
        "artifact_bytes": sum((args.output / name).stat().st_size for name in ("model.onnx", "model.json", "tokenizer.json")),
        "parameters": sum(parameter.numel() for parameter in best[2].parameters()),
        "environment": platform_info, "profiles": profiles, "quantization": quantization,
        "frozen_word_embeddings": args.freeze_word_embeddings, "cpu_threads": args.cpu_threads,
        "warm_cpu": latency_benchmark(scorer, splits.get("challenge", splits["test"])),
    }
    dump_json(args.output / "metrics.json", stats)
    print(json.dumps({"artifact_bytes": stats["artifact_bytes"], "training_rows": stats["training_rows"],
                      "heldout_accuracy": profiles[INDIA.name]["test"]["gate_accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
