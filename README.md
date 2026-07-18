# InkBridge AI — Handwriting Intelligence & ModelOps Prototype

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ozzy2438/inkbridge-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/ozzy2438/inkbridge-ai/actions/workflows/ci.yml)

> An evidence-first prototype for converting photographed and scanned handwriting into layout-aware, confidence-scored transcripts with human review.

## Project status

InkBridge AI is currently an **alpha research and engineering prototype**, not a
production deployment. The repository contains a baseline TrOCR inference path,
heuristic image-quality and layout components, API scaffolding, an offline evaluation
harness, a resumable pretrained TrOCR prediction producer, and a fail-closed protected-pilot
governance/double-annotation gate with sealed local-model inference and a writer-isolated,
aggregate-only self-hosted evaluation path. The protected path also enforces an owner-only POSIX
storage boundary and hash-chained consent-withdrawal/primary-and-backup-deletion evidence.

The following claims are intentionally deferred until reproducible artifacts exist:

- domain fine-tuning improvements
- calibrated auto-accept or abstention thresholds
- latency, throughput, and cost targets
- asynchronous class-set processing
- production privacy, retention, audit, and tenant-isolation controls
- reviewer-time reduction and pilot outcomes

Measured results will be published only with the dataset manifest, split report,
configuration, model version, and evaluation artifact needed to reproduce them.

## 🎯 Problem Statement

Education and assessment organizations receive handwritten student work as photos, scans, or PDFs. These documents contain:
- Phone-captured images at oblique angles
- Shadow, blur, or low-light conditions
- Crossed-out words and margin insertions
- Mixed printed questions and student handwriting
- Multiple handwriting styles and difficulty levels

**InkBridge AI** targets the real business problem: reducing the human time needed
to digitize and verify a class set. The initial 45-to-15-minute objective is a pilot
hypothesis and has not yet been validated.

## 🏗️ Target architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        InkBridge AI Platform                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐       │
│  │  Upload   │──▶│ Quality Gate │──▶│  Layout Analysis   │       │
│  │  Gateway  │   │ (blur/skew)  │   │  (segmentation)    │       │
│  └──────────┘   └──────────────┘   └────────────────────┘       │
│                                              │                    │
│                                    ┌─────────┴─────────┐         │
│                                    ▼                   ▼         │
│                          ┌──────────────┐   ┌──────────────┐    │
│                          │  TrOCR Model │   │  VLM Fallback│    │
│                          │  (fast path) │   │  (hard pages)│    │
│                          └──────────────┘   └──────────────┘    │
│                                    │                   │         │
│                                    └─────────┬─────────┘         │
│                                              ▼                    │
│                                    ┌──────────────────┐          │
│                                    │  Confidence      │          │
│                                    │  Router & Scorer │          │
│                                    └──────────────────┘          │
│                                              │                    │
│                              ┌───────────────┼───────────────┐   │
│                              ▼               ▼               ▼   │
│                    ┌──────────────┐ ┌──────────────┐ ┌──────┐   │
│                    │ Auto-Accept  │ │ Human Review │ │Export│   │
│                    │ (high conf.) │ │ (uncertain)  │ │ API  │   │
│                    └──────────────┘ └──────────────┘ └──────┘   │
│                                              │                    │
│                                              ▼                    │
│                                    ┌──────────────────┐          │
│                                    │  Active Learning │          │
│                                    │  Feedback Loop   │          │
│                                    └──────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

The diagram describes the target system. The current repository does not yet include
durable job storage, a working review queue, or the complete active-learning loop.

## 🚀 Development setup

```bash
# Clone the repository
git clone https://github.com/ozzy2438/inkbridge-ai.git
cd inkbridge-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Download public baseline model weights
python scripts/download_models.py

# Run the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run the development stack
docker compose -f docker/docker-compose.yml up --build
```

The single-image endpoint is the current integration focus. PDF conversion and the
batch workflow are not yet end-to-end complete.

## 📁 Project Structure

```
inkbridge-ai/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py             # Application entry point
│   │   ├── routes/             # API route handlers
│   │   ├── middleware/         # Request logging (auth/rate limiting planned)
│   │   └── schemas/            # Pydantic models
│   ├── pipeline/               # Core ML pipeline
│   │   ├── quality_gate.py     # Image quality assessment
│   │   ├── layout_analysis.py  # Page segmentation
│   │   ├── confidence_router.py # Route to appropriate model
│   │   ├── ocr_engine.py       # TrOCR inference
│   │   └── vlm_fallback.py     # VLM for hard pages
│   ├── models/                 # Model metadata
│   │   └── model_registry.py   # Prototype version management
│   ├── training/               # Training pipelines
│   │   ├── finetune_trocr.py   # TrOCR fine-tuning
│   │   └── active_learning.py  # Active learning loop
│   ├── evaluation/             # Evaluation framework
│   │   ├── metrics.py          # CER, WER, calibration
│   │   ├── runner.py           # Artifact validation and release gates
│   │   ├── protected_inference.py # Sealed-model protected producer
│   │   └── failure_atlas.py    # Failure categorization
│   ├── data/                   # Data processing
│   │   ├── manifest.py         # Licensed provenance and split manifests
│   │   ├── datasets.py         # Dataset loaders
│   │   ├── augmentation.py     # Data augmentation
│   │   ├── protected_storage.py # Local IAM and lifecycle evidence
│   │   └── writer_split.py     # Writer-independent splits
│   ├── labeling/               # Annotation operations
│   │   └── label_studio_config.py
│   └── export/                 # Output formatting
│       └── transcript.py       # Prototype JSON/TXT/DOCX/LMS formatting
├── configs/                    # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── evaluation_config.yaml
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/            # API lifecycle coverage
│   └── regression/             # Evaluation release-gate coverage
├── scripts/                    # Utility scripts
│   ├── download_models.py
│   ├── build_dataset_manifest.py
│   ├── prepare_datasets.py
│   ├── run_evaluation.py
│   ├── run_baseline_inference.py
│   ├── run_protected_inference.py
│   ├── prepare_protected_storage.py
│   ├── record_protected_lifecycle_event.py
│   └── export_onnx.py
├── docker/                     # Docker configuration
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── infrastructure/             # IaC and deployment
│   ├── terraform/
│   └── kubernetes/
├── notebooks/                  # Research notebooks
│   └── 01_baseline_evaluation.ipynb
├── docs/                       # Documentation
│   ├── annotation_guidelines.md
│   ├── data_contract.md
│   ├── evaluation.md
│   └── model_card.md
├── pyproject.toml
├── Makefile
└── .github/workflows/          # CI/CD
    ├── ci.yml
    ├── baseline_predictions.yml
    └── model_evaluation.yml
```

## 📊 Datasets

| Dataset | Purpose | Size | License |
|---------|---------|------|---------|
| [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | Standard benchmark & baseline | 13,353 lines, 657 writers | Research |
| [SMHD](https://doi.org/10.25439/rmt.24312715.v1) | Student essays, cross-outs, corrections | 500+ students, essays & math | CC BY-NC 4.0 |
| [GNHK](https://github.com/GoodNotes/GNHK-dataset) | Camera-captured, varied conditions | 687 images, 9,363 lines | CC BY 4.0 |
| [OpenHand-Synth](https://huggingface.co/datasets/to-be/OpenHand-Synth) | Synthetic pipeline smoke test only | Pinned 27-line subset | CC BY 4.0 |
| [CSAFE Handwriting Database](https://doi.org/10.25380/iastate.10062203.v2) | Real adult-handwriting pipeline smoke test only | Pinned 9-writer, 19-line subset | CC BY 4.0 |
| Synthetic Augmented | Blur, shadow, glare, rotation | Generated on-the-fly | N/A |
| Consented Pilot Set | Production-like evaluation | TBD (pilot phase) | Private |

See the [dataset contract](docs/data_contract.md) for normalized labels, provenance fields,
source hashing, the reproducible public smoke subsets, and deterministic writer-independent split
generation. Synthetic and adult-handwriting smoke results are plumbing evidence, not
student-handwriting benchmarks.

The [protected student-pilot gate](docs/pilot_governance.md) defines the separate path for consented
child/student data. Its template is intentionally unapproved; no private data or consent record is
included in this repository. After approval, the
[protected evaluation contract](docs/protected_evaluation.md) freezes validation/test writers and
uses a sealed, preloaded local TrOCR artifact to produce and evaluate reference-free predictions
without exporting sample-level content. The
[protected storage/lifecycle contract](docs/protected_storage_lifecycle.md) binds local IAM facts,
withdrawal state, and deletion evidence into every downstream artifact.

## 🔬 Model Architecture

### Current baseline: pretrained TrOCR
- Base: `microsoft/trocr-base-handwritten`
- Domain fine-tuning is planned; no fine-tuned checkpoint is published yet
- Normalized datasets can produce hashed, licensed manifests with deterministic writer-isolated splits
- Line-level manifests can produce resumable prediction artifacts pinned to an immutable model revision
- Confidence calibration and threshold selection remain evaluation work

### Prototype fallback: compact VLM
- Structured JSON output with bounding boxes
- Handles full-page layout understanding
- Routing and output-schema fidelity have not yet been benchmarked

### Planned optimization pipeline
- FP16 → INT8 → ONNX Runtime comparison
- Dynamic batching benchmark
- Distillation experiments gated on measured quality, latency, and cost

## 📈 Evaluation contract

The offline evaluator validates prediction JSONL artifacts, hashes its inputs, and computes the
implemented metrics below. A separate producer can now capture pretrained TrOCR predictions from a
licensed line-level manifest. The repository still has no child/student model benchmark because no
authorised, independently verified protected pilot has been run. See the
[evaluation contract](docs/evaluation.md) for the public input schema and release-gate workflow,
and the [protected evaluation contract](docs/protected_evaluation.md) for private shadow
evaluation.

The repository contains a
[captured synthetic TrOCR smoke result](artifacts/smoke/openhand-synth-trocr-base-v1/README.md)
and a [captured real adult-handwriting smoke result](artifacts/smoke/csafe-real-trocr-base-v1/README.md)
that prove the pipeline runs end to end. Both are deliberately excluded from student-handwriting
benchmark claims; the next evidence level requires independently verified child/student gold data.

| Category | Metrics |
|----------|--------|
| Text Recognition | CER, WER, Normalized Edit Distance |
| Layout | Region F1, Reading Order Accuracy **(planned)** |
| Reliability | Calibration Error, False-Confidence Rate |
| Operations | Correction Minutes/Page, Auto-Accept Rate **(planned)** |
| Production | p50/p95 Sample Latency; Page Throughput and Cost/Page **(planned)** |
| Edge Cases | Blur, Cursive, Cross-outs, Insertions |

## 🛡️ Privacy and safety requirements

- Use only licensed, synthetic, or explicitly consented and de-identified examples
- Do not use uploaded data for training without explicit authorization
- Implement and verify retention, deletion, access control, audit logging, and tenant isolation
  before a pilot
- Treat local POSIX checks and control-file assertions as engineering evidence, not independent
  proof of cloud IAM, encryption, firewall policy, WORM logs, or backup erasure
- Keep student packages outside Git and GitHub-hosted workflows; require two blind gold passes
- Calibrate abstention before enabling automatic acceptance
- Do not represent the prototype as GDPR- or production-compliant without a formal assessment

## 📋 Roadmap

- [ ] Week 1-2: Problem discovery, data governance, locked gold set, baselines
  **(governance, local-IAM/lifecycle, freeze/sealed-inference/self-hosted eval gates, and smoke
  baselines complete; real consented gold and independent infrastructure evidence pending)**
- [ ] Week 3-4: Quality gate, layout segmentation, TrOCR baseline **(prototype implemented; validation pending)**
- [ ] Week 5-6: Domain fine-tuning, failure analysis, calibration
- [ ] Week 7: VLM fallback & structured JSON SFT
- [ ] Week 8: Label Studio, annotation workflow, active learning
- [ ] Week 9: Quantization, distillation, latency benchmarks
- [ ] Week 10: API, async batch pipeline, monitoring, CI/CD
- [ ] Week 11: Shadow pilot, reviewer-time measurement
- [ ] Week 12: Champion release, case study, live demo

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 📚 References

- [TrOCR: Transformer-based OCR](https://arxiv.org/abs/2109.10282)
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [GNHK Dataset (ICDAR 2021)](https://dl.acm.org/doi/10.1007/978-3-030-86337-1_27)
- [SMHD Dataset](https://researchdata.edu.au/student-messy-handwritten-dataset-smhd/3988424)
- [Label Studio ML Pipeline](https://labelstud.io/guide/ml)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
