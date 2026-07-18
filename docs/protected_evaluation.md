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

Materialise the approved TrOCR model outside Git as a self-contained directory. It must contain
`config.json`, `preprocessor_config.json`, and safetensors weights; symbolic links, executable
files, pickle-based weights, cached Hub state, and remote Python code are refused. Seal the model
before use and record its deterministic identity:

```bash
chmod -R a-w /protected/inkbridge/models/trocr-v1
python -m scripts.hash_local_model_artifact \
  --model-dir /protected/inkbridge/models/trocr-v1
```

Copy `configs/evaluation/protected_inference_authorization.template.json` to
`/protected/inkbridge/pilot-001/protected_inference_authorization.json`. The template is
deliberately invalid. An authorised owner must set `status` to `approved`, bind the frozen
evaluation/manifest and exact model hash, approve the device and every inference setting, set
`model_artifacts_preloaded` to `true`, and give the approval a validity window of at most 30 days.
Then remove all write permission bits from the authorization file.

Run the producer on the approved self-hosted machine:

```bash
python -m scripts.run_protected_inference \
  --contract /protected/inkbridge/pilot-001/pilot_intake.json \
  --dataset-dir /protected/inkbridge/pilot-001 \
  --manifest-dir /protected/inkbridge/pilot-001/frozen_evaluation \
  --authorization /protected/inkbridge/pilot-001/protected_inference_authorization.json \
  --model-dir /protected/inkbridge/models/trocr-v1 \
  --output-dir /protected/inkbridge/pilot-001/runs/trocr-v1 \
  --device cpu \
  --batch-size 8 \
  --beam-width 4 \
  --max-length 128 \
  --confidence-threshold 0.7 \
  --abstention-threshold 0.4
```

Every CLI setting must exactly match the sealed authorization. The producer refuses CI before
opening data, revalidates the frozen manifest, reads only locked `test` images, verifies each image
hash before decoding, loads the local model with Transformers offline mode, and blocks new Python
socket connections during inference. It fsyncs resumable progress after each batch and atomically
publishes a read-only run only after every test sample is complete and all input/model hashes still
match.

`predictions.jsonl` contains exactly:

```json
{"confidence":0.8,"latency_ms":12.5,"prediction":"raw model text","sample_id":"sample-<opaque-hex>","source_sha256":"<64 hex>"}
```

It cannot contain `reference`, writer ID, slices, image paths, cost, routing decisions, or any other
field. The producer rejects validation samples, unknown or duplicate IDs, missing rows, source-hash
drift, changed authorization/model bytes, and provenance that differs from the approved settings.

## 4. Verify the generated execution evidence

The producer writes `execution_attestation.json`; operators must not hand-edit it. It binds the
approval reference and authorization hash, canonical inference-configuration hash, exact local
model artifact, frozen evaluation set/manifest, and prediction-file hash. The repository's
`protected_execution_attestation.template.json` is retained only as an intentionally invalid schema
example.

Offline library flags and the Python socket guard are defence in depth, not an independently
verified network perimeter. The attestation and permission checks cannot prove the operator's
authority, storage mount, host firewall, subprocess/native-library behaviour, or infrastructure
audit trail. Those controls still require evidence from the approved environment.

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
- A synthetic fake-backend rehearsal proves producer/evaluator control flow, not model quality or
  statistical adequacy.
- `gold_ready` remains false; this path does not choose the final population or certify coverage.
- It does not implement storage IAM, encryption, audit-log collection, backup deletion, or consent
  withdrawal orchestration.
- It does not authorise training or production decisions.
