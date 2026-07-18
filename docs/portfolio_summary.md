# InkBridge AI — portfolio summary

## What is complete

InkBridge AI is an evidence-first handwriting OCR and ModelOps demonstrator built for noisy,
real-world education documents. It covers image-quality gating, layout analysis, TrOCR inference,
VLM fallback interfaces, writer-isolated data manifests, evaluation/release gates, protected-data
governance, labeling operations, active-learning selection, API/batch scaffolding, monitoring, and
CI/CD.

The repository includes reproducible synthetic and adult-handwriting smoke evidence plus a sealed,
offline TrOCR research run on real SMHD student handwriting. The writer-isolated six-line holdout
recorded CER 0.1300 and WER 0.3611. This is a small, unreviewed, CC BY-NC research rehearsal—not a
production, primary-school, commercial, or gold-benchmark claim.

The real-data diagnostic adds:

- a private failure atlas covering 15 validation/test lines, with 14 exact-match failures;
- validation-only confidence-policy search with an untouched test holdout;
- a fail-closed decision when no threshold met 50% coverage at selective CER <= 0.10;
- a nine-task, priority-ordered validation queue requiring 18 blind annotation assignments;
- independent adjudication on disagreement and no model/reference anchoring; and
- aggregate-only public evidence while sample-level student artifacts remain outside Git/CI.

## Job-aligned evidence

| Role expectation | Repository evidence |
|---|---|
| OCR/handwriting models | Pinned TrOCR backends, real and synthetic offline baselines |
| VLM/document understanding | Structured VLM fallback and layout-analysis interfaces |
| Full model lifecycle | Immutable provenance, evaluation gates, registry and deployment scaffolding |
| Evaluation ownership | CER/WER/NED, calibration, false-confidence, slices, latency and regression locks |
| Data/labeling leadership | Annotation v2 protocol, double review, adjudication and priority queue |
| Production efficiency | FP16 path and optimization dependencies/plans; comparative benchmark still pending |
| Production monitoring | API metrics, confidence routing, batch orchestration and CI/CD checks |
| Privacy with messy data | Owner-only artifacts, CI refusal, socket blocking and hash-bound manifests |

## CV-ready bullets

- Built an evidence-first handwriting OCR/ModelOps platform in Python, PyTorch and Hugging Face,
  spanning writer-isolated data pipelines, TrOCR inference, document-quality gates, evaluation,
  monitoring, annotation operations and CI/CD.
- Executed a sealed, network-blocked TrOCR baseline on real writer-isolated student handwriting,
  recording 13.0% CER and 36.1% WER while keeping all sample-level data outside Git and hosted CI.
- Designed fail-closed confidence evaluation and a private active-learning workflow: generated a
  14-record failure atlas and a nine-task double-blind labeling queue while preserving an untouched
  test holdout and preventing model-reference anchoring.

## Honest interview boundary

Present this as a production-oriented engineering demonstrator with real research evidence. Do not
claim that the model was domain-fine-tuned, commercially validated, calibrated for automatic
acceptance, deployed to a school, or evaluated on an independently reviewed primary-school gold
set. Those are the explicit next milestones that require authorised data, human reviewers and a
real pilot rather than more repository code.
