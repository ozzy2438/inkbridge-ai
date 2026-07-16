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

No InkBridge evaluation run has been completed. Results will be added only with a
reproducible evaluation artifact and dataset/split manifest.

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| CER (IAM test) | Not measured | Not trained | N/A |
| CER (GNHK test) | Not measured | Not trained | N/A |
| CER (SMHD test) | Not measured | Not trained | N/A |
| False Confidence Rate | Not measured | Not trained | N/A |
| Calibration Error | Not measured | Not trained | N/A |

### Ethical Considerations
- No private student data is included in this public repository
- A future pilot requires explicit consent, de-identification, access control, and retention rules
- Abstention thresholds must be calibrated before operational use
- Not used for automated grading or student assessment
- Human review required for all uncertain predictions

### Environmental Impact

Not measured. Training compute, inference latency, energy use, and carbon estimates
will be reported from actual runs rather than projected values.
