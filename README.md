# Legal AI Prompt Guard

A local text classifier for legal-tech request intake. The model scores the operation in the current message using relevant conversation and document context. Applications can configure jurisdiction scope, confidence thresholds, and the corresponding response path.

## Quick start

The 55.2 KB sparse model runs on Python 3.11 or newer with the standard library. From the repository checkout, download the published weights and pass a JSON request on standard input. [Hugging Face model card](https://huggingface.co/AttoFlow-AI/legal-ai-prompt-guard)

```bash
curl -fsSL https://huggingface.co/AttoFlow-AI/legal-ai-prompt-guard/resolve/6e7c45ec2de04f96b49368bc118403f261242cf1/weights.json.gz -o weights.json.gz
python -m legal_gate --model weights.json.gz <<'JSON'
{"input":"What remedies can an Indian tenant seek if the landlord withholds the security deposit?","questions":{"decision":{"type":"choice"},"route":{"type":"choice"}}}
JSON
```

The default `general_legal` profile accepts eligible legal work across jurisdictions. Set `--profile india_legal` when the application serves Indian law. In a Python service, choose the `Profile` and optional `Thresholds` at the server boundary and reuse the loaded model.

```python
from legal_gate.policy import GENERAL
from legal_gate.routing import Gate
from legal_gate.sparse import SparseModel

gate = Gate(SparseModel("weights.json.gz"), GENERAL)
result = gate.decide({
    "input": {
        "request": "Can you check the requirement in clause 8?",
        "history": [
            {"role": "assistant", "text": "We were reviewing the notice clause of your lease."}
        ],
        "document": {"status": "available", "name": "lease.txt", "excerpt": "Clause 8. The notice period is thirty days."},
    },
})
```

## Response contract

The `input` field accepts a string, an array of text parts, or the structured request object shown above. Text parts form one current turn. Use the structured object for history or document state. Use `action` for dispatch. `answers.decision` contains a choice, the accept/refuse probability distribution, and a confidence score. A `noul` decision question returns the accept probability as `noul`. `answers.route` contains the semantic category scores. This pattern follows the fixed-question portion of the Decisions API. [Google Decisions API](https://aistudio.google.com/docs/decisions-api)

| Action | Application handling |
| --- | --- |
| `legal` | Dispatch the complete request to the legal assistant with its downstream safety, source verification, and tool policy. |
| `clarify`, `file_error` | Ask for the missing legal context or use the file-error response, as appropriate. |
| `greeting`, `courtesy`, `product`, `public_identity` | Use the application reply for that route. |
| `refuse_scope` | Use the application's scope refusal. Refuse the complete mixed turn. |
| `refuse_internal`, `refuse_private`, `scope_redirect` | Use the applicable confidentiality or redirection reply. |
| `review` | Dispatch to the existing policy path. The binary `choice` is `null` and `review_reason` indicates the cause. |

Treat route scores and diagnostics as internal data. Supply the most relevant four conversation turns and a short, relevant document excerpt or extraction status from the application. Input limits and requests written chiefly in scripts outside the pilot's coverage go to review. Scores are calibrated on synthetic data. An application can set its threshold values in `Gate` and measure coverage against its own traffic mix.

## Reproduce the experiments

Prepare synthetic scenario files using the authoring guide and theme matrix. Replace `/path/to/scenarios` with the directory containing your files. [Authoring guide](data/AUTHORING.md), [theme matrix](themes.json)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[train,runtime]'
python -m legal_gate.data --data /path/to/scenarios
python -m legal_gate.expand --data /path/to/scenarios
python -m legal_gate.teacher datasets/expanded/{train,dev,calibration,test}.jsonl \
  --output datasets/expanded/gemini-labels.jsonl
python -m legal_gate.train --dataset datasets/expanded --teacher datasets/expanded/gemini-labels.jsonl \
  --output models/sparse-expanded --widths 4096 --regularization 24 --max-iterations 220
cp models/sparse-expanded/weights.json.gz weights.json.gz
python -m unittest discover -s tests -v
```

The teacher command accepts `GEMINI_API_KEY` through the environment or a hidden terminal prompt and resumes by sample ID. Use `python -m legal_gate.teacher --help` for rate limits and other label settings.

The encoder trainer supports a CUDA Colab runtime with the `train`, `runtime`, and `encoder` extras installed. It saves INT8 ONNX weights and the tokenizer next to model metadata. The command line accepts the model directory for either backend. To reproduce the pilot encoder, populate its teacher labels first.

```bash
python -m legal_gate.teacher datasets/pilot/{train,dev,calibration,test}.jsonl \
  --output datasets/pilot/gemini-labels.jsonl
python -m legal_gate.train_encoder --dataset datasets/pilot \
  --teacher datasets/pilot/gemini-labels.jsonl --output models/bert-tiny-pilot
```

The expanded comparison used the CPU command below.

```bash
python -m legal_gate.train_encoder --dataset datasets/expanded \
  --teacher datasets/expanded/gemini-labels.jsonl --output models/bert-tiny-expanded \
  --learning-rates 0.0003 --epochs 5 --batch-size 64 --freeze-word-embeddings --cpu-threads 4
```

## Results

| General profile evaluation | Binary decision accuracy | Correct actions among automatic routes | Automatic false accepts |
| --- | ---: | ---: | ---: |
| Grouped test, 238 examples | 93.7% | 203 of 213 | 3 |
| Independent challenge, 138 examples | 87.7% | 106 of 113 | 1 |

The benchmarks use synthetic requests. The calibration and error rates reflect that evaluation mix. [Results](RESULTS.md), [research notes](RESEARCH.md), [dataset manifest](datasets/expanded/manifest.json)
