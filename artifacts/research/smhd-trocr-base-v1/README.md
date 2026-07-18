# SMHD student-handwriting offline TrOCR research evidence v1

This is the first captured model run in the repository's target student-handwriting domain. A
sealed, local copy of `microsoft/trocr-base-handwritten` processed six writer-isolated SMHD line
images while Hugging Face/Transformers offline flags and the Python socket guard were active.

This result is deliberately narrow. SMHD is licensed CC BY-NC 4.0, its writers are a mixture of
high-school and university students, and the publisher transcriptions have not received independent
human review. The run is a **non-commercial research rehearsal**, not a gold benchmark, primary-
school accuracy estimate, protected or consented production pilot, release gate, or training result.

## Captured result

| Measurement | Result |
|---|---:|
| Test samples | 6 real lines / 2 isolated writers |
| Publisher correction-marker split | 3 marker / 3 no marker |
| CER | 0.1300 |
| WER | 0.3611 |
| Normalized edit distance | 0.1345 |
| Mean confidence | 0.8573 |
| False-confidence rate | 1.0000 |
| Uncertain / unreadable predictions | 1 / 0 |
| p50 / p95 line latency | 1,872 / 1,875 ms |
| Valid output rate | 1.0000 |

The three publisher-marker lines had CER 0.1818, versus 0.0893 on the three lines without markers.
The sample is far too small for a population estimate, but it exposes the intended failure mode:
corrections and cross-outs were harder in this selection. More importantly, the overall false-
confidence rate was 1.0 and the abstention rate was zero. Automatic acceptance is therefore not
supported; confidence calibration and human review remain required.

Latency was measured on CPU with Python 3.14.4, PyTorch 2.13.0, Transformers 4.57.6, macOS/Darwin
27.0.0, and an arm64 machine. It is per-line model latency after load/warm-up, not page or class-set
throughput. The six-line sample cannot support a production latency claim.

## Offline and artifact identity

- Dataset manifest SHA-256: `7dfae6bbfb1f5b3c6170c2d18a076ab672d6bc684a2b85a514e48defb4705824`
- Evaluation-set SHA-256: `d578516a115e29e4467168d2dbe7a51254253242f8ef52e3159b9a0cb287d098`
- Sealed model artifact SHA-256: `8215ca7b0e4536affbc01a5718abae5d572cf011b0e709953e57eb0a1be92149`
- Local prediction SHA-256: `9aa88a1f43fba2d5857dbcfa1d0816150705857640bf0cc43253d44a0e1696e6`
- Model package: 8 code-free files, 1,334,746,073 bytes, read-only, safetensors weights
- Inference: local files only, no external AI service, outbound Python sockets blocked, no CI
- Image processor: legacy/slow behavior explicitly locked with `use_fast=False`

Transformers reported that two encoder pooler parameters were newly initialized while loading the
published checkpoint. The OCR decoder consumes the encoder sequence output rather than the pooled
classification output, but the warning is retained as a compatibility observation. This evidence
is a development baseline, not a model-release certification.

`result.json` is aggregate-only. The source images, references, writer IDs, normalized package,
manifest, predictions, and full local evaluation remain in owner-only storage outside Git. No
sample-level SMHD artifact was uploaded to GitHub or processed by GitHub Actions.

## Reproduce locally

Prepare the SMHD package and writer-isolated manifest using
[`docs/data_contract.md`](../../../docs/data_contract.md), then stage and seal the pinned TrOCR model.
Run inference outside Git:

```bash
python -m scripts.run_offline_research_inference \
  --dataset-dir /private/research/smhd-line-v1/normalized-v1 \
  --output-dir /private/research/smhd-line-v1/runs/trocr-base-v1 \
  --model-dir /private/models/trocr-base-handwritten-<revision> \
  --model-version microsoft-trocr-base-handwritten-<revision>-smhd-research-v1 \
  --model-artifact-sha256 <sealed-directory-sha256> \
  --device cpu \
  --batch-size 2 \
  --beam-width 4 \
  --max-length 128
```

Evaluate the owner-only prediction file with `scripts.run_evaluation`, using a test-set name that
ends in `unreviewed`. Text predictions should reproduce for the pinned inputs and runtime. Latency,
timestamps, and whole prediction hashes can change across hardware and runs.
