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

## GitHub Actions

The manually dispatched `Model Evaluation` workflow consumes an artifact named
`evaluation-predictions` (configurable at dispatch) from a specified workflow run. The artifact must
contain `predictions.jsonl` at its root. A prior Model Evaluation run and artifact can optionally be
selected as the release baseline. The workflow does not run model inference: the upstream producer
of a real prediction artifact is still to be implemented with the locked gold-set access controls.
