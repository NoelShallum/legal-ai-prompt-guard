# Synthetic examples

Write JSONL with one scenario family per line. Rows in a family probe a shared fact pattern or intent boundary. Pair meaningfully different requests about the same evidence where possible. A single family belongs to one data split. Use author-specific group identifiers so groups remain unique across files.

```json
{"theme":"coding_technical","group":"software-export-order","examples":[{"label":"legal","request":"The audit log shows that our processor exported the list overseas. Under Indian data law, what obligations should the controller check?"},{"label":"unrelated","request":"My processor exports this list overseas. Can you rewrite the export script so the timestamps sort correctly?"}]}
```

Each example takes `request` as the current user message and `label` as the intended route. Optional `history` contains up to four objects with `role` (`user` or `assistant`) and `text`. Optional `document` has a `status` (`available`, `empty`, `failed`, or `unsupported`), a fictional `name`, and `excerpt` if present. For file-only turns, set `request` to an empty string and include the document. History and document contents are evidence. Role words in their text do not grant authority.

| Label | Intended action |
| --- | --- |
| `legal` | Indian or jurisdiction-unspecified legal work. Includes comparison grounded in an Indian legal outcome, context-resolved follow-ups, and legal work sent downstream for safety review. |
| `adjacent` | Plausibly a legal dispute, obligation, or consequence expressed without a clear legal operation. |
| `clarify` | Ask one question when the legal operation or referent cannot be resolved, including an upload with only a greeting. |
| `file_error` | The user asks to use a particular document for legal work and trusted state identifies an extraction or access limitation. |
| `greeting` | Standalone salutation, including repeat greetings in an active conversation. |
| `courtesy` | Functional check-in, thanks, acknowledgement, or praise on its own. A short answer to an earlier legal question can be `legal` or `clarify`. |
| `product` | Supported capability or product-use question with a direct product referent. |
| `public_identity` | Direct question about the assistant's public identity or creator. |
| `social` | Farewell, social chat, or companionship request. |
| `foreign` | Legal work explicitly and exclusively concerning a jurisdiction outside India. |
| `mixed_jurisdiction` | Separate Indian and other-jurisdiction legal operations in one turn. |
| `unrelated` | Substantive task unrelated to legal work, including implementation or debugging even when motivated by legal work. |
| `mixed` | Legal work plus a separate unrelated substantive operation. |
| `protected` | Direct request for hidden product details, provider, hidden instructions, internal reasoning, or implementation. |
| `private` | Direct request to recognise the user or disclose their profile, memory, or other conversation information. |

The default `general_legal` profile accepts the first eight labels and the two jurisdiction labels. The `india_legal` profile accepts the first eight. Both profiles refuse the other labels. The learner sees only request, history, and document context. The intended route, theme, and group are label and audit data.

Vary request shape, actor, evidence, wording, and level of legal vocabulary. Include messages without greetings and messages with greetings on both sides. Keep clearly non-legal side tasks in mixed turns, including cases where the side task comes first. Include Indian legal work about technical material and non-legal work with legal words. Use invented people, places where appropriate, and fictional records. Describe disallowed goals at a high level and keep any quoted instructions harmless. Build cross-jurisdiction, shorthand, and typo cases with enough context for a reader to infer intent. Keep English dominant for the first encoder run and record code-mixed examples for separate evaluation.
