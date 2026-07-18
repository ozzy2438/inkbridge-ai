# CSAFE real-handwriting TrOCR smoke evidence v1

This directory is the first captured **real-handwriting** model artifact in the repository. It shows
that official, licensed scans can pass through pinned source validation, deterministic line crops,
writer-isolated splitting, pretrained TrOCR inference, and schema-validated evaluation.

It is **not a student-handwriting benchmark or a gold set**. CSAFE contains adult writers, this test
split has only five lines from two isolated writers, and the line references were split from the
known PHR prompt after one visual pass. The result cannot estimate classroom accuracy, compare
model releases, tune thresholds, or support a production claim. Independent human transcription
and crop-boundary review is required before promotion to gold evaluation data.

## Captured result

| Measurement | Result |
|---|---:|
| Test samples | 5 real lines / 2 isolated adult writers |
| CER | 0.0584 |
| WER | 0.3214 |
| Normalized edit distance | 0.0654 |
| Mean confidence | 0.9086 |
| False-confidence rate | 0.5000 |
| p50 line latency | 1,330 ms |
| p95 line latency | 1,763 ms |
| Valid output rate | 1.0000 |

The model transcribed one line exactly. Most remaining errors were punctuation spacing or a small
character/word substitution. Half of the exact-match errors above the configured threshold were
false-confident, so these scores are evidence that confidence calibration and human review routing
remain necessary—not evidence for automatic acceptance.

Latency was captured on CPU with Python 3.11.13, PyTorch 2.13.0, Transformers 4.57.6, macOS/Darwin
27.0.0, and an arm64 machine. It is line-sample latency, not page or class-set throughput, and is not
portable to production hardware.

## Artifact identity

- Dataset: `CSAFE Handwriting Database`, Figshare article `10062203`, version 2
- Dataset licence: CC BY 4.0; selection, citation, and attribution rules are in the dataset config
- Selection: 19 line crops from 9 adult writers; test split is 5 lines from 2 isolated writers
- Model: `microsoft/trocr-base-handwritten` at `eaacaf452b06415df8f10bb6fad3a4c11e609406`
- Manifest SHA-256: `8f6c738199da8c36798ff888a7d11464519027bb9884d7e5de84eb9f706c211d`
- Evaluation-set SHA-256: `3fc02a64ebdd9b1e2bb903f29d992104f2a55f200f5032f08a336cf9f3b8ff6d`
- Prediction SHA-256: `eb905434cf4c0512408e6ff832376999e5da137e60666bf53e8ec4d597e4e904`

`predictions.jsonl` contains the five raw model outputs without writer IDs or images.
`predictions.meta.json` records dataset, manifest, model, inference configuration, runtime, privacy,
and output hashes. The `eval_*.json` file contains overall and per-slice metrics plus hashes of its
prediction and evaluation-configuration inputs.

## Reproduce

Prepare the source package and manifest using the commands in
[`docs/data_contract.md`](../../../docs/data_contract.md), then run:

```bash
python -m scripts.run_baseline_inference \
  --dataset-dir data/processed/csafe-real-handwriting-smoke \
  --output-dir /tmp/csafe-real-trocr-base-v1 \
  --model-id microsoft/trocr-base-handwritten \
  --model-revision eaacaf452b06415df8f10bb6fad3a4c11e609406 \
  --split test \
  --device cpu \
  --batch-size 2 \
  --beam-width 4 \
  --max-length 64 \
  --no-resume

python -m scripts.run_evaluation \
  --predictions /tmp/csafe-real-trocr-base-v1/predictions.jsonl \
  --model-version microsoft-trocr-base-handwritten-eaacaf452b06415df8f10bb6fad3a4c11e609406 \
  --test-set csafe-real-handwriting-smoke-test-v1 \
  --config configs/evaluation_config.yaml \
  --output-dir /tmp/csafe-real-trocr-base-v1
```

Text predictions should reproduce for the pinned model and data bytes. Latency values, timestamps,
and therefore whole-file hashes can change across hardware and runs; runtime provenance makes that
variation explicit.
