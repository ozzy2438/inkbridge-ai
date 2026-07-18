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
- **License:** MIT (model weights), CC-BY-NC (SMHD data)

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

The datasets below are candidates. Their presence in this table does not mean that
they have already been downloaded, transformed, split, or used for training.

| Dataset | Published scale | Purpose |
|---------|-----------------|---------|
| IAM | 13,353 lines | Standard benchmark |
| SMHD | 500+ contributors | Student messy writing |
| GNHK | 9,363 lines | Camera-captured handwriting |
| Augmented | Not generated yet | Robustness |

### Evaluation Results

Two reproducible engineering smoke runs exist for the pinned pretrained checkpoint. Neither is a
student-handwriting gold benchmark, model comparison, or release-gate baseline.

| Set | Population | Test lines | CER | WER | False confidence |
|---|---|---:|---:|---:|---:|
| OpenHand-Synth smoke | Synthetic rendering styles | 6 | 0.0556 | 0.5000 | 0.6667 |
| CSAFE smoke | Real adult writers | 5 | 0.0584 | 0.3214 | 0.5000 |

Raw predictions, runtime/model provenance, manifest hashes, and evaluation reports are stored under
`artifacts/smoke/`. No fine-tuned model or independently verified child/student result exists.
The protected self-hosted evaluation gate is implemented, but it has run only on synthetic control
data. Its sealed local-model producer has likewise been exercised only with a deterministic fake
backend; neither contributes a model-quality result to this table.

The protected control plane now also binds an owner-only POSIX storage check and hash-chained
consent-withdrawal/deletion evidence into intake, freeze, inference authorization, attestation, and
evaluation. This is no substitute for independently verified provider IAM/encryption/audit logs or
an authorised real-student run.

### Ethical Considerations
- No private student data is included in this public repository
- A future pilot must pass the protected governance and double-annotation gate; a software pass is
  not a legal/privacy certification
- Abstention thresholds must be calibrated before operational use
- Not used for automated grading or student assessment
- Human review required for all uncertain predictions

### Environmental Impact

Not measured. Training compute, inference latency, energy use, and carbon estimates
will be reported from actual runs rather than projected values.
