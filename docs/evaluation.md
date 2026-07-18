# Evaluation Contract

InkBridge evaluates offline prediction artifacts so model inference and metric calculation remain
separate, reproducible steps. The evaluator does not download a model or claim benchmark quality by
itself; it measures only the records supplied to it.

## Prediction JSONL

Each line is a JSON object with four required fields. `source_sha256` is the digest of the exact
image or page bytes used for inference:

```json
{"sample_id":"page-001","source_sha256":"<64 hex characters>","reference":"hello","prediction":"helo"}
```

Optional fields are `slices` (a list of tags), `confidence`, `latency_ms`, and
`cost_per_page_usd`. Confidence, latency, and cost must each be present for every record or for none
of them. This prevents operational and reliability metrics from being calculated from a biased
subset.

Use stable, pseudonymous `sample_id` values. Do not put student names or other personal information
in prediction artifacts.

The baseline producer always exports the model's raw decoded text in `prediction`. Abstention and
uncertainty are additional routing fields; replacing low-confidence text with a sentinel would bias
CER/WER and is therefore not allowed in an evaluation artifact.

## Produce a pretrained baseline

A schema-v2, writer-isolated, line-level dataset package can be predicted locally:

```bash
python -m scripts.run_baseline_inference \
  --dataset-dir /protected/data/gnhk-normalized \
  --output-dir /protected/predictions/trocr-base \
  --model-id microsoft/trocr-base-handwritten \
  --model-revision <commit-or-tag> \
  --split test \
  --device cpu
```

The requested Hugging Face revision is resolved once to its immutable commit SHA; that SHA, model
settings, runtime versions, manifest hashes, and output hash are written to
`predictions.meta.json`. Source-image hashes are checked before inference. Progress is fsynced to a
partial JSONL after every batch, and a compatible retry resumes without reprocessing completed
samples. `predictions.jsonl` appears only after the selected split is complete.

The producer reports line latency. The evaluator therefore derives `samples_per_minute`, where one
sample means one JSONL record; it must not be described as page throughput when records are lines.
It does not invent `cost_per_page_usd`: honest page cost requires page/essay aggregation and
infrastructure pricing, which remain a later measurement step.

## Run an evaluation

```bash
python -m scripts.run_evaluation \
  --predictions /path/to/candidate.jsonl \
  --model-version trocr-baseline-v1 \
  --test-set writer-isolated-gold-v1
```

The JSON result contains the input and configuration SHA-256 hashes, an evaluation-set hash built
from IDs/source hashes/references/slices, overall recognition and reliability metrics, latency/cost
when supplied, and metrics for every slice tag.

To compare against an existing result and make failure block the command:

```bash
python -m scripts.run_evaluation \
  --predictions /path/to/candidate.jsonl \
  --model-version trocr-candidate-v2 \
  --test-set writer-isolated-gold-v1 \
  --baseline-results /path/to/eval_trocr-baseline-v1_writer-isolated-gold-v1.json \
  --enforce-release-gate
```

The current gate covers CER improvement, maximum per-slice CER regression, false-confidence rate,
p95 latency, cost per page, schema-valid output rate, and proof that candidate and baseline used the
same evaluation set. All optional numeric fields are therefore required when enforcing this gate.
Layout fidelity and teacher correction time remain planned metrics and must not be claimed from this
artifact.

The files under `tests/fixtures/evaluation/` are synthetic regression fixtures for the evaluator;
they are not a model benchmark or a gold dataset.

The captured [OpenHand-Synth TrOCR smoke artifact](../artifacts/smoke/openhand-synth-trocr-base-v1/README.md)
is separate from those unit fixtures. It contains real pretrained-model outputs, but its six
synthetic lines remain engineering evidence rather than a student-handwriting benchmark.

The captured [CSAFE TrOCR smoke artifact](../artifacts/smoke/csafe-real-trocr-base-v1/README.md)
advances that check to real adult handwriting: five test lines from two writer-isolated
participants. Its prompt-derived references have not received independent human verification, so
it remains pipeline evidence—not a gold set or a claim about child/student handwriting.

## GitHub Actions

The manually dispatched `Baseline Predictions` workflow downloads a self-contained dataset
artifact, produces `evaluation-predictions`, and preserves resumable diagnostics if inference
fails. A subsequent `Model Evaluation` dispatch consumes that workflow run ID. A prior Model
Evaluation run and artifact can optionally be selected as the release baseline.

GitHub-hosted inference is limited to public, licensed, de-identified data by an explicit dispatch
attestation. The workflow is not an approved path for private student work. It also does not run
automatically on pull requests, so routine CI never downloads model weights or evaluation images.

## Protected student evaluation

Private student handwriting uses the separate
[protected self-hosted evaluation contract](protected_evaluation.md). It freezes disjoint
validation/test writers outside Git, requires a short-lived authorization bound to a sealed local
safetensors model, storage-control/lifecycle hashes, and exact inference settings, produces a
reference-free prediction allowlist, refuses hosted CI/external-AI execution, and persists aggregate
metrics only. The ordinary hosted workflows and `scripts.run_evaluation` command are not approved
substitutes for this protected path.
