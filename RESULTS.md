# Experiment results

## Pilot

The 2,500 examples were split into 1,786 train, 240 development, 236 calibration, and 238 test examples. The separate challenge set contains 138 examples. Scenario families stayed within one split, and the split builder removed two exact overlaps from the challenge set. The independent challenge is balanced at 69 accept and 69 refuse examples under the India-only profile.

Gemini Decision 1 labeled all 2,500 samples in 79 requests. Its scope decisions agreed with the reference labels on 2,481 samples (99.2%). Exact route agreement was 2,370 of 2,500 (94.8%). After requiring route agreement and confidence of at least 0.55, 1,686 pilot training examples were available. On the challenge set, two teacher errors disagreed with a reference scope decision. Both involved unavailable files on clearly unrelated creative requests. The review queue preserves those cases.

The pilot decision disagreements include Indian enforcement based on a foreign judgment, legal evidence attached to a request for travel writing, and an image extraction failure on a calorie estimate. These cases test which operation the user requested, whether a foreign reference serves an Indian legal outcome, and whether a file state matters to an eligible task.

| Pilot candidate | Export size | Grouped test decision accuracy | Challenge decision accuracy | Challenge auto-routing coverage |
| --- | ---: | ---: | ---: | ---: |
| Hashed linear, 8-bit | 132.4 KB | 93.7% | 88.4% | 70.3% |
| Hashed linear, 4-bit | 24.6 KB | 93.3% | 87.7% | 65.9% |
| BERT Tiny, INT8 ONNX | 5.19 MB | 93.3% | 92.8% | 76.1% |

Decision accuracy is the binary score when every example is classified. Coverage is the share the actual API dispatches automatically at the calibrated thresholds. The 4-bit pilot sent 47 of 138 challenge requests to review and automatically chose the right application action for 88 of the 91 it handled. Its three mistakes included one false accept. The pilot BERT model learned the binary boundary more strongly on the challenge set, while its specialized route macro F1 was 0.31 compared with 0.68 for the 4-bit baseline.

The BERT Tiny encoder fine-tuned on a Colab Tesla T4. It has 4,387,855 parameters with the 15-way classification head. The export size includes the tokenizer and model metadata. The Python standard-library linear model size is its compressed weights and metadata.

## Dataset expansion

The 10,000-example corpus adds 7,500 training examples and keeps the development, calibration, test, and challenge sets fixed. The additions include 1,985 newly authored prompts and 5,515 composed examples. The latter combine operations, change the conversation topic, resolve a short follow-up from context, or pair a document with legal and unrelated requests. The authored intake labels are 60% accept and 40% refuse under the India-only profile. Gemini's decisions were 59.5% accept and 40.5% refuse.

Gemini labeled the 7,500 new examples and 138 challenge examples in 160 requests, using 1,265,636 reported input tokens. Across the 10,000 examples, decision agreement was 9,765 (97.7%) and exact route agreement was 9,091 (90.9%). The final fits use 8,270 of the 9,286 training candidates after the same agreement and confidence filter as the pilot. [Dataset manifest](datasets/expanded/manifest.json), [label audit](reports/expanded/labels-audit.json)

## Compact model

The default export uses 4,096 hashed features and 8-bit weights. Development selected regularization `C=24`. The compact release was refit after replacing product names in the synthetic prompts with fictional placeholders and relinking the existing teacher labels. The 4-bit export reduced development decision accuracy by 0.8 percentage points, so the release uses the 55.2 KB 8-bit export. Quantization preserved the unquantized model's top route on 236 of the 238 grouped test examples.

An authored-label control fitted on all 9,286 training candidates reached 93.7% grouped test accuracy and 87.0% challenge accuracy. That comparison changes both the label selection rule and the number of training rows. [Authored-label control](models/sparse-expanded-authored/metrics.json)

| Profile | Evaluation | Decision accuracy | Automatic routes | Correct actions among automatic routes | Automatic false accepts | Automatic false refusals |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| General | Grouped test | 93.7% | 213 of 238 | 203 of 213 | 3 | 4 |
| General | Independent challenge | 87.7% | 113 of 138 | 106 of 113 | 1 | 2 |
| India-only | Grouped test | 93.7% | 206 of 238 | 198 of 206 | 3 | 2 |
| India-only | Independent challenge | 87.7% | 111 of 138 | 105 of 111 | 1 | 1 |

The challenge pairs test which operation was requested, including follow-ups, mixed operations, and the role of attachments. The general profile covered 81.9% of the challenge set. Weighting the challenge to the hypothetical 90% accept mix gives 79.6% coverage, assuming each class's error and coverage rates transfer. Using the same thresholds on live traffic requires measuring the actual mix and reviewing errors. [Compact model metrics](models/sparse-expanded/metrics.json), [serving audit](reports/expanded/sparse-expanded-serving.json)

## Expanded encoder comparison

Colab declined the second T4 allocation. The expanded BERT run used local CPU, froze the pretrained word embeddings, and trained for five epochs at learning rate `0.0003` with length-bucketed batches of 64. This changed the training recipe along with the corpus size.

| Expanded candidate, India-only profile | Export size | Grouped test decision accuracy | Grouped test route macro F1 | Correct challenge actions among automatic routes | Grouped test API p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hashed linear, 8-bit | 55.2 KB | 93.7% | 0.79 | 105 of 111 | 6.78 ms |
| BERT Tiny, INT8 ONNX | 5.19 MB | 90.3% | 0.42 | 95 of 108 | 3.39 ms |

Both expanded candidates achieved 87.7% binary decision accuracy on the challenge set. The linear model is the default because of its size and stronger route accuracy in this comparison. The compact model and the encoder were measured through the local API in process, after warm-up, on an Intel Core i5-8250U CPU. [Expanded encoder metrics](models/bert-tiny-expanded/metrics.json), [encoder serving audit](reports/expanded/bert-tiny-expanded-serving.json)

## Measurement

Model hyperparameters and quantization decisions use development examples. The calibration split sets temperature and routing thresholds separately for the two profiles. Test and challenge scores are reported on the frozen splits. The raw and serving metrics for each model are available in its metrics file and the corresponding audit report. The serving report includes group bootstrap intervals and actual fallback counts.

Cross-split duplicate checking uses a normalized fingerprint of the complete structured input. A second audit uses cosine similarity over character n-grams for prompts and context of at least 35 characters. That audit flags likely overlaps for inspection. It measures surface similarity, with scenario families handled by the split manifest.

These experiments use authored and composed synthetic data. The reference labels are authored policy expectations, and agreement with Gemini measures consistency with the teacher. The small challenge set and English-dominant corpus limit what these estimates say about traffic from live legal workflows. The token cost from the original policy prompt and its caching is a separate measurement in the application.

For a live comparison, log the reference route, classifier route, fallback, and cache-adjusted token expense during shadow evaluation. Estimate token savings from the scope-only tokens removed on eligible requests and the full generation calls avoided by local replies. Use the observed class mix to weight false accepts, false refusals, and fallback. The supplied 90% acceptance assumption is for planning those measurements.

## Attribution

The fine-tuned encoder is derived from Google's BERT Tiny checkpoint at revision `30b0a37ccaaa32f332884b96992754e246e48c5f`. It has been task fine-tuned and quantized for ONNX runtime. [Google checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2), [Apache license](models/bert-APACHE-LICENSE.txt)
