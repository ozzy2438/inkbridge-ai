# OpenHand-Synth TrOCR smoke evidence v1

This directory is the first captured end-to-end model artifact in the repository. It proves that a
licensed source can pass through pinned selection, hash validation, writer-style isolation,
pretrained TrOCR inference, and schema-validated evaluation.

It is **not a student-handwriting benchmark**. The test set contains only six synthetic English
name/date line images from OpenHand-Synth. It is too small and too artificial to estimate classroom
accuracy, compare model releases, tune thresholds, or support a production claim.

## Captured result

| Measurement | Result |
|---|---:|
| Test samples | 6 synthetic lines / 2 isolated rendering styles |
| CER | 0.0556 |
| WER | 0.5000 |
| Normalized edit distance | 0.0613 |
| Mean confidence | 0.8628 |
| False-confidence rate | 0.6667 |
| p50 line latency | 1,208 ms |
| p95 line latency | 1,347 ms |
| Valid output rate | 1.0000 |

The CER and WER difference is expected on very short names and dates: a small spacing or character
error can make an entire word incorrect. Four of six records were exact-match errors above the
false-confidence threshold, so these confidences are not calibrated for automatic acceptance.

Latency was captured on CPU with Python 3.11.13, PyTorch 2.13.0, Transformers 4.57.6, macOS/Darwin
27.0.0, and an arm64 machine. It is line-sample latency, not page or class-set throughput, and is not
portable to production hardware.

## Artifact identity

- Dataset: `to-be/OpenHand-Synth` at `8b5027ab2dc6cc944e0ce7fe37997ce046121e66`
- Dataset licence: CC BY 4.0; selection and attribution rules are in the dataset config/contract
- Model: `microsoft/trocr-base-handwritten` at `eaacaf452b06415df8f10bb6fad3a4c11e609406`
- Manifest SHA-256: `f3fbaea6e0ad3d91ea08b7e7ab4712041c41742214c98c6b5622f0c57284959d`
- Evaluation-set SHA-256: `9b062c7b4e6bc2a496f826b8b136e6816f0c9e3c85ec659f5118937a8342a099`
- Prediction SHA-256: `b60d398794255293cab3c6075e3033a91c34c077ff4e22c33ed5fd81b1787995`

`predictions.jsonl` contains the six raw model outputs without writer IDs or images.
`predictions.meta.json` records dataset, manifest, model, inference configuration, runtime, privacy,
and output hashes. The `eval_*.json` file contains overall and per-slice metrics plus hashes of its
prediction and evaluation configuration inputs.

## Reproduce

Prepare the source package and manifest using the commands in
[`docs/data_contract.md`](../../../docs/data_contract.md), then run:

```bash
python -m scripts.run_baseline_inference \
  --dataset-dir data/processed/openhand-synth-smoke \
  --output-dir /tmp/openhand-synth-trocr-base-v1 \
  --model-id microsoft/trocr-base-handwritten \
  --model-revision eaacaf452b06415df8f10bb6fad3a4c11e609406 \
  --split test \
  --device cpu \
  --batch-size 2 \
  --beam-width 4 \
  --max-length 64 \
  --no-resume

python -m scripts.run_evaluation \
  --predictions /tmp/openhand-synth-trocr-base-v1/predictions.jsonl \
  --model-version microsoft-trocr-base-handwritten-eaacaf452b06415df8f10bb6fad3a4c11e609406 \
  --test-set openhand-synth-smoke-test-v1 \
  --config configs/evaluation_config.yaml \
  --output-dir /tmp/openhand-synth-trocr-base-v1
```

Text predictions should reproduce for the pinned model and data bytes. Latency values, timestamps,
and therefore whole-file hashes can change across hardware and runs; runtime provenance makes that
variation explicit.
