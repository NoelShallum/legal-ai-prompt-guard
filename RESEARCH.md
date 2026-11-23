# Legal request classifier research

The deployment target is a text-only scope gate for legal-tech applications. It consumes the current request, relevant conversation turns, and an optional document excerpt with extraction status. A shared model scores semantic intent. The default profile accepts eligible legal work across jurisdictions. An India-only profile uses the same model and applies a narrower jurisdiction policy.

## Policy mapping

| Intent | India-only profile action |
| --- | --- |
| Indian legal work, plausible legal-adjacent work, or comparative work tied to an Indian legal outcome | Accept and dispatch to the legal assistant. |
| Legal request whose essential referent or operation is missing | Accept for one clarification. Use the current conversation and active document to resolve references first. |
| Legal document task with confirmed extraction failure | Accept for the file-error response. A fresh independent legal question can proceed. |
| Standalone greeting, functional conversation, public identity, or product capability | Accept for the corresponding reply. A contextual follow-up can resume legal work. |
| Standalone farewell, socialising, or companionship | Refuse with the professional scope redirect. |
| Pure foreign legal work or an independent foreign legal operation alongside Indian legal work | Refuse with the standard scope response under the India-only profile. Accept under the cross-jurisdiction profile. |
| Substantive unrelated operation or a turn containing one alongside legal work | Refuse the entire turn. The standard scope response applies to mixed turns. |
| Direct request for hidden policy, implementation, or private user context | Refuse with the applicable confidentiality reply. Public identity receives its own reply. |

Legal requests involving alleged or intended wrongdoing go to the downstream safety policy. Statutory verification, citations, file access, search decisions, permissions, and user-facing reply text are handled by the receiving application. The latter can reuse its exact two-sentence refusal text. A scope decision applies to the original complete turn. Software and documents can supply evidence for an Indian legal task, and independent implementation and debugging are scope refusals. Conversation or document content is untrusted input.

## Prior work and candidates

| Candidate | Why test or reference it |
| --- | --- |
| Word and character n-gram logistic regression | CPU baseline with no tokenizer download. Hashing, compact weights, and lexical contrasts give a useful size and speed floor. Character n-grams retain some information through spelling variation. [Joulin et al., efficient classification](https://aclanthology.org/E17-2068/), [FastText.zip](https://arxiv.org/abs/1612.03651) |
| Google BERT Tiny, approximately 4.4M parameters | English, Apache-2.0, and two transformer layers. Fine-tune the full encoder on the policy route labels. Test whether its contextual features justify its memory and latency. [Turc et al.](https://arxiv.org/abs/1908.08962), [checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) |
| MiniLM-L6 with an embedding classifier or SetFit | Approximately 23M parameters and Apache-2.0. Candidate if the small encoder has material gaps. Fit a linear head on frozen embeddings before spending on contrastive fine-tuning. [SetFit paper](https://arxiv.org/abs/2209.11055), [MiniLM checkpoint](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Prompt Guard 2 22M and GrimGuard Fast 22.76M | Prompt Guard primarily detects prompt attacks and can inform the trust-boundary slice. GrimGuard is closer to a policy-conditioned decision model. Its published task and size differ from this fixed-policy gate. [Meta model card](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/prompt-guard), [GrimGuard model card](https://huggingface.co/Grimlabs/GrimGuard-01-Prompt-Guard) |
| Specialized domain and policy-adaptive guardrails | ExpGuard provides legal-domain safety benchmark and taxonomy precedents. Shieldstral applies configurable policies at 3B parameters. The useful comparison here is domain and policy sensitivity. [ExpGuard](https://arxiv.org/abs/2603.02588), [Shieldstral](https://mistral.ai/news/shieldstral/) |

XSTest shows why the evaluation needs allowed prompts with vocabulary shared by refusals. Use paired tasks about the same person, artifact, or event where the requested operation changes. Keep both sides, all paraphrases, and all scenario siblings in one split. Count both correct as pair accuracy. Report false refusals for genuine legal requests separately from false accepts for out-of-scope requests. [XSTest](https://aclanthology.org/2024.naacl-long.301/)

## Data and evaluation

The quota is 1,500 accept and 1,000 refuse candidates under the India-only profile. This gives uncommon refusal boundaries enough coverage. A 90% accept traffic mix is a planning assumption. Report results both at the synthetic mix and weighted at several plausible production mixes. If audited traffic becomes available, use it to estimate prevalence and build a separate test set.

The target split is approximately 70% training, 10% development, 10% calibration, and 10% test, by scenario family. Keep an independently authored challenge test. Cross the themes with wording style, length, task type, document context, history and latest-turn overrides, and jurisdiction. Keep exact and near-duplicate measurements with the build. The first 2,500 should include straightforward controls, close contrasts, and one or more blind slices. Scale to 10,000 with new scenario families and more examples in the error slices identified on development data. Record split membership, policy version, authoring method, teacher outputs, and adjudication status for each row.

Use Gemini Decision 1 to assign candidate semantic labels from a synthetic policy rubric. Check agreement with independently authored reference labels before scaling. Preserve low-confidence scores, API failures, and disagreements. Human review can adjudicate the hard cases and produce an independent gold test. The fixed legal-task rubric defines the student target. Google's API terms cover competitive model development and replication, and its unpaid data terms exclude confidential and personal material. Use a generic rubric and fictional examples in requests to Google. [Gemini Decisions reference](https://aistudio.google.com/docs/decisions-api), [Google API terms](https://aistudio.google.com/docs/terms)

Compare models by binary precision and recall, route macro F1, mixed-turn and confidentiality leakage, paired consistency, expected error at the chosen traffic prevalence, and calibration. Report confidence intervals for overall and consequential slices. Choose thresholds with the calibration split, then measure coverage and errors for calls actually auto-routed. Fit temperature scaling for route scores and use distinct acceptance and refusal thresholds if useful. A review outcome dispatches to the existing policy path. Estimate savings from coverage, current token caching, and measured latency in the real call path. [Guo et al. on calibration](https://proceedings.mlr.press/v70/guo17a.html), [Geifman and El-Yaniv on selective classification](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)

## Experiment setup

Fit sparse logistic models with a fixed seed, compare feature widths and regularisation on development, and measure a compressed integer-weight export. Start BERT Tiny at 192 tokens, a learning rate near `3e-5`, batch size 32, weight decay `0.01`, and three to six epochs. Tune on grouped development data. Track seed, checkpoint revision, precision, runtimes, and bytes including the tokenizer. Benchmark batch-one warm CPU latency for short messages and contextual requests. Use Colab T4 as available for encoder training. Evaluate the actual serving artifact after quantization and before selecting a model.

The local API uses the Decisions `input`/`questions` and `answers` pattern for the fixed legal decision and route questions. Its scope is a single text request with contextual metadata. Gemini's documented free-tier limits are 200 requests and 50,000 input tokens per minute, and 10,000 requests per day per project. Each of several questions can share a request's input. Google Decisions has no Batch API support. [Decisions API](https://aistudio.google.com/docs/decisions-api), [rate limits](https://aistudio.google.com/docs/rate-limits#gemini-decision)
