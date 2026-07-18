# Protected Self-Hosted Evaluation

This path turns an approved student-handwriting gold candidate into a cryptographically bound,
writer-isolated shadow-evaluation set without copying private content into Git or GitHub Actions.
It is an engineering control, not proof that the contract assertions, consent, storage, or operator
attestation are factually or legally sufficient.

## Boundary

- The protected root, intake audit, manifest, predictions, attestation, and results stay outside the
  Git checkout in approved private storage.
- The split contains `validation` and locked `test` writers only. It deliberately contains no
  training split because this pilot contract authorises evaluation, not training.
- Validation writers may support error analysis. Test writers must not be used for training,
  prompt selection, threshold tuning, or repeated manual model selection.
- GitHub Actions and other detected CI environments are refused. Model weights must be preloaded;
  inference runs with network access off and no external AI service.
- Prediction JSONL contains no reference text, writer ID, slice label, or image path. References and
  controlled slices are joined from the protected manifest in memory.
- The persisted evaluation report contains aggregate and per-controlled-slice metrics, hashes, and
  run provenance. It contains no sample IDs, writer IDs, references, predictions, or images.

## 1. Validate intake

After the authorised owner has completed the deliberately invalid template in protected storage:

```bash
python -m scripts.validate_protected_pilot \
  --contract /protected/inkbridge/pilot-001/pilot_intake.json \
  --dataset-dir /protected/inkbridge/pilot-001
```

This writes `pilot_intake.audit.json`. Freezing re-runs the validator and requires every stable audit
field and source hash to match, so an old audit cannot authorise changed labels, images, reviews, or
contract settings.

## 2. Freeze the writer-isolated set

```bash
python -m scripts.freeze_protected_evaluation \
  --contract /protected/inkbridge/pilot-001/pilot_intake.json \
  --dataset-dir /protected/inkbridge/pilot-001
```

The one-time command creates:

```text
/protected/inkbridge/pilot-001/frozen_evaluation/
├── protected_evaluation.manifest.jsonl
└── protected_evaluation.manifest.meta.json
```

The contract fixes the evaluation-set version, split seed, validation/test writer ratios, and
minimum examples per writer. The manifest binds every opaque sample to its exact image hash,
protected reference, controlled slices, pseudonymous writer, and split. Metadata binds the
contract, intake audit, labels, annotation review, manifest hash, counts, and writer-isolation
result. The command will not replace an existing freeze and removes all write permission bits.
Permission bits are tamper evidence, not immutable storage: each evaluation recomputes the complete
manifest and all hashes before reading predictions.

## 3. Produce reference-free predictions

Run the approved, preloaded model locally against only the manifest's `test` images. The producer
must write `/protected/inkbridge/pilot-001/runs/<model>/predictions.jsonl` with exactly:

```json
{"sample_id":"sample-<opaque-hex>","source_sha256":"<64 hex>","prediction":"raw model text"}
```

`confidence`, `latency_ms`, and `cost_per_page_usd` are optional, but each supplied field must be
present for every record. Do not add `reference`, `writer_id`, `slices`, or image paths. The gate
requires every test sample exactly once and rejects validation samples, unknown samples, duplicate
IDs, missing rows, and source-hash drift.

## 4. Attest the execution boundary

Copy `configs/evaluation/protected_execution_attestation.template.json` beside the prediction file
as `execution_attestation.json`. The repository template is intentionally invalid. The approved
operator must bind the exact model version, evaluation-set ID, and manifest hash, then truthfully
attest the prediction-file hash and preloaded model-artifact hash. They must also truthfully attest
that protected storage was mounted, networking/external AI/GitHub Actions were unused, predictions
contain no references, and no automated educational decision was enabled.

The attestation records an asserted control. The code cannot independently prove that networking
was disabled or that the named operator had authority; infrastructure audit logs remain required.

## 5. Run aggregate-only evaluation

```bash
python -m scripts.run_protected_evaluation \
  --contract /protected/inkbridge/pilot-001/pilot_intake.json \
  --dataset-dir /protected/inkbridge/pilot-001 \
  --manifest-dir /protected/inkbridge/pilot-001/frozen_evaluation \
  --predictions /protected/inkbridge/pilot-001/runs/trocr-v1/predictions.jsonl \
  --attestation /protected/inkbridge/pilot-001/runs/trocr-v1/execution_attestation.json \
  --model-version trocr-v1
```

The read-only report is written under `protected_evaluation_results/`. A later candidate can supply
the protected baseline report with `--baseline-results` and `--enforce-release-gate`; existing CER,
slice-regression, confidence, latency, cost, validity, and evaluation-set identity checks then
apply. Optional operational fields required by the release gate must be complete for both runs.

Do not publish the result automatically. Although sample content is absent, hashes and model
performance remain protected pilot provenance and require owner review before disclosure.

## What this does not yet prove

- No real student package or model run exists in this repository.
- A three-writer synthetic rehearsal proves control flow, not statistical adequacy.
- `gold_ready` remains false; this path does not choose the final population or certify coverage.
- It does not implement storage IAM, encryption, audit-log collection, backup deletion, or consent
  withdrawal orchestration.
- It does not authorise training or production decisions.
