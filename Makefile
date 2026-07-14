.PHONY: install dev test lint format clean docker-build docker-up

# Development
install:
	pip install -e ".[dev]"

dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-regression:
	pytest tests/regression/ -v

# Code quality
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	ruff check --fix src/ tests/

# Docker
docker-build:
	docker build -f docker/Dockerfile -t inkbridge-ai:latest .

docker-build-gpu:
	docker build -f docker/Dockerfile.gpu -t inkbridge-ai:gpu .

docker-up:
	cd docker && docker-compose up --build

docker-down:
	cd docker && docker-compose down

# ML Pipeline
train:
	python -m src.training.finetune_trocr

evaluate:
	python scripts/run_evaluation.py --model-version latest --test-set gold

export-onnx:
	python scripts/export_onnx.py --model-path ./checkpoints/trocr-finetuned --quantize

download-models:
	python scripts/download_models.py

# Data
prepare-data:
	python scripts/prepare_datasets.py

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete
