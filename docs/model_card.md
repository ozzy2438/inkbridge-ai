# InkBridge AI — Model Card

## Model: TrOCR Fine-tuned for Student Handwriting

### Model Details
- **Model type:** Vision Encoder-Decoder (ViT + RoBERTa)
- **Base model:** microsoft/trocr-base-handwritten
- **Fine-tuned on:** SMHD + GNHK + IAM + Augmented data
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

### Training Data
| Dataset | Samples | Writers | Purpose |
|---------|---------|---------|----------|
| IAM | 9,862 lines | 500 | Standard benchmark |
| SMHD | ~2,000 lines | 500+ | Student messy writing |
| GNHK | 9,363 lines | N/A | Camera-captured |
| Augmented | ~20,000 | N/A | Robustness |

### Evaluation Results

*To be filled after training:*

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| CER (IAM test) | TBD | TBD | TBD |
| CER (GNHK test) | TBD | TBD | TBD |
| CER (SMHD test) | TBD | TBD | TBD |
| False Confidence Rate | TBD | TBD | TBD |
| Calibration Error | TBD | TBD | TBD |

### Ethical Considerations
- No student personal data in training set
- Model trained on de-identified samples only
- Abstention mechanism prevents silent hallucination
- Not used for automated grading or student assessment
- Human review required for all uncertain predictions

### Environmental Impact
- Training compute: ~8 GPU-hours (single A100)
- Inference: ~50ms per line (GPU), ~200ms per line (CPU)
- Carbon footprint: Estimated < 5 kg CO2eq
