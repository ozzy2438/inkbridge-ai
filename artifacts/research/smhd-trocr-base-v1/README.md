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
| p50 / p95 line latency | 1,719 / 1,862 ms |
| Valid output rate | 1.0000 |

The three publisher-marker lines had CER 0.1818, versus 0.0893 on the three lines without markers.
The sample is far too small for a population estimate, but it exposes the intended failure mode:
corrections and cross-outs were harder in this selection. More importantly, the overall false-
confidence rate was 1.0 and the abstention rate was zero. Automatic acceptance is therefore not
supported; confidence calibration and human review remain required.

## Failure atlas and calibration gate

A second offline pass used the writer-isolated validation split (9 lines / 3 writers) for
diagnostics while preserving the original 6-line / 2-writer test split as an untouched holdout.
Validation had one exact match, mean per-line CER 0.2874, mean confidence 0.6743, and exact-match ECE
0.5632. The holdout had zero exact matches and exact-match ECE 0.8573. Per-line CER is a macro
diagnostic and therefore differs from the corpus-level test CER of 0.1300 above.

The private failure atlas contains 14 failure records across the 15 validation-plus-test lines:

- 8 failures at per-line CER >= 0.25, 5 at 0.10-0.25, and 1 below 0.10;
- 8 failure records carried the publisher correction marker;
- 4 were marked uncertain by the model; and
- 1 was an exact error at confidence >= 0.90.

The predeclared calibration gate required at least 30 validation samples. Its diagnostic threshold
search also required at least 50% validation coverage (5 lines) with selective CER <= 0.10. The
9-line validation split met neither release condition: no eligible threshold was found, no
threshold was applied to the test holdout, and confidence parameters were not fitted. The resulting
decision is deliberately fail-closed: **all predictions require human review**.

Latency was measured on CPU with Python 3.14.4, PyTorch 2.13.0, Transformers 4.57.6, macOS/Darwin
27.0.0, and an arm64 machine. It is per-line model latency after load/warm-up, not page or class-set
throughput. The six-line sample cannot support a production latency claim.

## Offline and artifact identity

- Dataset manifest SHA-256: `7dfae6bbfb1f5b3c6170c2d18a076ab672d6bc684a2b85a514e48defb4705824`
- Evaluation-set SHA-256: `d578516a115e29e4467168d2dbe7a51254253242f8ef52e3159b9a0cb287d098`
- Sealed model artifact SHA-256: `8215ca7b0e4536affbc01a5718abae5d572cf011b0e709953e57eb0a1be92149`
- Validation prediction SHA-256: `d319b451b70f2d7e25dd43917c3833c7c2284d13afd57ffd1482ac11776b7fb9`
- Test prediction SHA-256: `196250d38141dbaf43cf2171806c0f23eacf7e0e7419caf524d0f495ddf9a472`
- Private failure-atlas SHA-256: `197911664dafe39aea85edfb984a2648970619132014cdab3c4034354d93abf2`
- Model package: 8 code-free files, 1,334,746,073 bytes, read-only, safetensors weights
- Inference: local files only, no external AI service, outbound Python sockets blocked, no CI
- Image processor: legacy/slow behavior explicitly locked with `use_fast=False`

Transformers reported that two encoder pooler parameters were newly initialized while loading the
published checkpoint. The OCR decoder consumes the encoder sequence output rather than the pooled
classification output, but the warning is retained as a compatibility observation. This evidence
is a development baseline, not a model-release certification.

`result.json` is aggregate-only. The source images, references, writer IDs, normalized package,
manifest, predictions, private failure atlas, and full local evaluation remain in owner-only
storage outside Git. No sample-level SMHD artifact was uploaded to GitHub or processed by GitHub
Actions.

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
  --split validation \
  --device cpu \
  --batch-size 2 \
  --beam-width 4 \
  --max-length 128
```

Repeat with `--split test` into a separate owner-only run directory, then build the diagnostic:

```bash
python -m scripts.run_offline_research_diagnostics \
  --validation-run-dir /private/research/smhd-line-v1/runs/validation-v1 \
  --test-run-dir /private/research/smhd-line-v1/runs/test-v1 \
  --output-dir /private/research/smhd-line-v1/diagnostics/v1 \
  --minimum-calibration-samples 30 \
  --minimum-policy-coverage 0.5 \
  --maximum-selective-cer 0.1
```

Evaluate the owner-only prediction file with `scripts.run_evaluation`, using a test-set name that
ends in `unreviewed`. Text predictions should reproduce for the pinned inputs and runtime. Latency,
timestamps, and whole prediction hashes can change across hardware and runs.
