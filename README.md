# InkBridge AI — Production Handwriting Intelligence & ModelOps Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green.svg)](https://github.com/features/actions)

> Converts photographed, scanned, or PDF student handwriting into layout-preserved, confidence-scored, auditable digital transcripts — with human-in-the-loop review for enterprise reliability.

![Architecture Overview](docs/assets/architecture_overview.png)

## 🎯 Problem Statement

Education and assessment organizations receive handwritten student work as photos, scans, or PDFs. These documents contain:
- Phone-captured images at oblique angles
- Shadow, blur, or low-light conditions
- Crossed-out words and margin insertions
- Mixed printed questions and student handwriting
- Multiple handwriting styles and difficulty levels

**InkBridge AI** solves the real business problem: reducing the human time needed to digitize and verify a class set from ~45 minutes to ~15 minutes.

## 🏗️ Architecture

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

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ozzy2438/inkbridge-ai.git
cd inkbridge-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Download model weights
python scripts/download_models.py

# Run the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run with Docker
docker-compose up --build
```

## 📁 Project Structure

```
inkbridge-ai/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py             # Application entry point
│   │   ├── routes/             # API route handlers
│   │   ├── middleware/         # Auth, logging, rate limiting
│   │   └── schemas/            # Pydantic models
│   ├── pipeline/               # Core ML pipeline
│   │   ├── quality_gate.py     # Image quality assessment
│   │   ├── layout_analysis.py  # Page segmentation
│   │   ├── confidence_router.py # Route to appropriate model
│   │   ├── ocr_engine.py       # TrOCR inference
│   │   └── vlm_fallback.py     # VLM for hard pages
│   ├── models/                 # Model definitions
│   │   ├── trocr_finetuned.py  # Fine-tuned TrOCR
│   │   ├── vlm_structured.py   # VLM structured output
│   │   └── model_registry.py   # Version management
│   ├── training/               # Training pipelines
│   │   ├── finetune_trocr.py   # TrOCR fine-tuning
│   │   ├── sft_vlm.py          # VLM supervised fine-tuning
│   │   ├── dpo_alignment.py    # DPO preference training
│   │   └── active_learning.py  # Active learning loop
│   ├── evaluation/             # Evaluation framework
│   │   ├── metrics.py          # CER, WER, calibration
│   │   ├── failure_atlas.py    # Failure categorization
│   │   ├── regression_tests.py # Model release gate
│   │   └── benchmarks.py       # Latency/cost benchmarks
│   ├── data/                   # Data processing
│   │   ├── datasets.py         # Dataset loaders
│   │   ├── preprocessing.py    # Image preprocessing
│   │   ├── augmentation.py     # Data augmentation
│   │   └── writer_split.py     # Writer-independent splits
│   ├── labeling/               # Annotation operations
│   │   ├── label_studio_config.py
│   │   ├── annotation_guidelines.py
│   │   └── quality_metrics.py
│   └── export/                 # Output formatting
│       ├── transcript.py       # Text/JSON/DOCX export
│       └── lms_integration.py  # LMS API export
├── configs/                    # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── evaluation_config.yaml
│   └── deployment_config.yaml
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── regression/
├── scripts/                    # Utility scripts
│   ├── download_models.py
│   ├── prepare_datasets.py
│   ├── run_evaluation.py
│   └── export_onnx.py
├── docker/                     # Docker configuration
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── infrastructure/             # IaC and deployment
│   ├── terraform/
│   └── kubernetes/
├── notebooks/                  # Research notebooks
│   ├── 01_baseline_evaluation.ipynb
│   ├── 02_fine_tuning_trocr.ipynb
│   ├── 03_vlm_structured_output.ipynb
│   ├── 04_confidence_calibration.ipynb
│   └── 05_quantization_benchmark.ipynb
├── docs/                       # Documentation
│   ├── model_card.md
│   ├── evaluation_report.md
│   ├── annotation_guidelines.md
│   ├── api_documentation.md
│   └── deployment_guide.md
├── pyproject.toml
├── Makefile
└── .github/workflows/          # CI/CD
    ├── ci.yml
    ├── model_evaluation.yml
    └── deploy.yml
```

## 📊 Datasets

| Dataset | Purpose | Size | License |
|---------|---------|------|---------|
| [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | Standard benchmark & baseline | 13,353 lines, 657 writers | Research |
| [SMHD](https://github.com/hiqmatNisa/SMHD) | Student essays, cross-outs, corrections | 500+ students, essays & math | CC BY-NC |
| [GNHK](https://www.goodnotes.com/gnhk) | Camera-captured, varied conditions | 687 images, 9,363 lines | Research |
| Synthetic Augmented | Blur, shadow, glare, rotation | Generated on-the-fly | N/A |
| Consented Pilot Set | Production-like evaluation | TBD (pilot phase) | Private |

## 🔬 Model Architecture

### Primary: Fine-tuned TrOCR
- Base: `microsoft/trocr-base-handwritten`
- Fine-tuned on SMHD + GNHK + augmented data
- Writer-independent train/test split
- Confidence scoring via token-level log probabilities

### Fallback: Compact VLM
- Structured JSON output with bounding boxes
- Handles full-page layout understanding
- Routes only for hard pages (saves cost)

### Optimization Pipeline
- FP16 → INT8 → ONNX Runtime
- Dynamic batching for throughput
- Distillation experiments

## 📈 Evaluation Metrics

| Category | Metrics |
|----------|--------|
| Text Recognition | CER, WER, Normalized Edit Distance |
| Layout | Region F1, Reading Order Accuracy |
| Reliability | Calibration Error, False-Confidence Rate |
| Operations | Correction Minutes/Page, Auto-Accept Rate |
| Production | p50/p95 Latency, Pages/Min, Cost/Page |
| Edge Cases | Blur, Cursive, Cross-outs, Insertions |

## 🛡️ Privacy & Safety

- No student names logged
- Uploaded documents auto-deleted after processing
- Demo uses only licensed/synthetic examples
- Model abstains when uncertain (no silent hallucination)
- Audit trail for all predictions
- GDPR/privacy-by-design compliant

## 📋 Roadmap

- [x] Week 1-2: Problem discovery, data governance, baselines
- [x] Week 3-4: Quality gate, layout segmentation, TrOCR baseline
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
