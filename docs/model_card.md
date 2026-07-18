# InkBridge AI — Baseline Model Card

## Model: Pretrained TrOCR baseline

**Status:** Development baseline. No InkBridge fine-tuned checkpoint has been
trained, evaluated, or published yet.

### Model Details
- **Model type:** Vision Encoder-Decoder (ViT + RoBERTa)
- **Base model:** microsoft/trocr-base-handwritten
- **Current checkpoint:** `microsoft/trocr-base-handwritten`
- **Planned domain data:** SMHD + GNHK + IAM + controlled augmentations
- **Parameters:** ~334M
- **Model-weight license:** MIT; dataset licences remain separate

### Intended Use
- Transcribing student handwritten essays and short answers
- Camera-captured document digitization
- Batch processing of class sets
- Human-in-the-loop document workflow

### Limitations
- Trained primarily on English text
- May struggle with mathematical notation
- Requires line-level input (not full pages)
- Performance degrades on very faint pencil writing
- Not suitable for cursive scripts other than Latin

### Planned Training Data

The datasets below are candidates. Their presence in this table does not mean that they have been
approved for training or used to change the checkpoint. SMHD has been used only for a local,
non-commercial baseline inference rehearsal; it has not changed the model.

| Dataset | Published scale | Purpose |
|---------|-----------------|---------|
| IAM | 13,353 lines | Standard benchmark |
| SMHD line version | 500+ contributors | Non-commercial student-handwriting research rehearsal |
| GNHK | 9,363 lines | Camera-captured handwriting |
| Augmented | Not generated yet | Robustness |

### Evaluation Results

Two engineering smoke runs and one non-commercial research rehearsal exist for the pinned
pretrained checkpoint. None is a student-handwriting gold benchmark, model comparison, or release-
gate baseline.

| Set | Population | Test lines | CER | WER | False confidence |
|---|---|---:|---:|---:|---:|
| OpenHand-Synth smoke | Synthetic rendering styles | 6 | 0.0556 | 0.5000 | 0.6667 |
| CSAFE smoke | Real adult writers | 5 | 0.0584 | 0.3214 | 0.5000 |
| SMHD research rehearsal | Mixed student writers, unreviewed | 6 | 0.1300 | 0.3611 | 1.0000 |

Raw smoke predictions, runtime/model provenance, manifest hashes, and evaluation reports are stored
under `artifacts/smoke/`. The SMHD source, manifest, and predictions remain outside Git; only the
[aggregate research result](../artifacts/research/smhd-trocr-base-v1/README.md) is committed. Its
publisher references are not independently reviewed, and its six lines cannot estimate population
accuracy. No fine-tuned model or independently verified child/student result exists.
The protected self-hosted evaluation gate is implemented, but it has run only on synthetic control
data. Its sealed local-model producer has likewise been exercised only with a deterministic fake
backend; neither contributes a model-quality result to this table.

The protected control plane now also binds an owner-only POSIX storage check and hash-chained
consent-withdrawal/deletion evidence into intake, freeze, inference authorization, attestation, and
evaluation. This is no substitute for independently verified provider IAM/encryption/audit logs or
an authorised real-student run.

### Ethical Considerations
- No private student data is included in this public repository
- The SMHD derivative and sample-level artifacts remain outside Git under owner-only permissions;
  CC BY-NC 4.0 forbids commercial use
- A future pilot must pass the protected governance and double-annotation gate; a software pass is
  not a legal/privacy certification
- Abstention thresholds must be calibrated before operational use
- Not used for automated grading or student assessment
- Human review required for all uncertain predictions

### Environmental Impact

Not measured. Training compute, inference latency, energy use, and carbon estimates
will be reported from actual runs rather than projected values.
