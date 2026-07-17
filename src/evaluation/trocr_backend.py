"""Immutable Hugging Face TrOCR backend for offline prediction production."""

from __future__ import annotations

import re
from copy import deepcopy
from importlib.metadata import version
from typing import Any

import torch
from huggingface_hub import HfApi
from PIL import Image

from src.evaluation.prediction_producer import PredictionResult
from src.pipeline.ocr_engine import TrOCREngine


class TrOCRPredictionBackend:
    """Resolve a Hub revision once, then load and report that immutable commit."""

    def __init__(
        self,
        *,
        model_id: str,
        requested_revision: str,
        device: str = "cpu",
        batch_size: int = 8,
        beam_width: int = 4,
        max_length: int = 128,
        confidence_threshold: float = 0.7,
        abstention_threshold: float = 0.4,
        use_fp16: bool = True,
        hf_api: HfApi | None = None,
    ) -> None:
        if not requested_revision.strip():
            raise ValueError("requested_revision must be non-empty")
        info = (hf_api or HfApi()).model_info(repo_id=model_id, revision=requested_revision)
        resolved_revision = info.sha
        if not isinstance(resolved_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", resolved_revision
        ):
            raise ValueError("Hugging Face revision did not resolve to an immutable commit SHA")

        model_version = f"{model_id}@{resolved_revision}"
        self._engine = TrOCREngine(
            model_path=model_id,
            revision=resolved_revision,
            model_version=model_version,
            device=device,
            batch_size=batch_size,
            beam_width=beam_width,
            max_length=max_length,
            confidence_threshold=confidence_threshold,
            abstention_threshold=abstention_threshold,
            use_fp16=use_fp16,
        )
        self._provenance = {
            "model_id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "model_version": model_version,
            "device": self._engine.device,
            "dtype": "float16" if self._engine.use_fp16 else "float32",
            "batch_size": batch_size,
            "beam_width": beam_width,
            "max_length": max_length,
            "confidence_threshold": confidence_threshold,
            "abstention_threshold": abstention_threshold,
            "runtime": {
                "torch": torch.__version__,
                "transformers": version("transformers"),
                "huggingface_hub": version("huggingface-hub"),
            },
        }

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a copy so run identity cannot be mutated by callers."""
        return deepcopy(self._provenance)

    def predict_batch(self, images: list[Image.Image]) -> list[PredictionResult]:
        """Run line recognition while preserving raw text for unbiased evaluation."""
        bboxes = [[0, 0, image.width, image.height] for image in images]
        predictions = self._engine.predict_batch(images, bboxes)
        return [
            PredictionResult(
                raw_text=prediction.raw_text,
                confidence=prediction.confidence,
                latency_ms=prediction.processing_time_ms,
                is_uncertain=prediction.is_uncertain,
                is_unreadable=prediction.is_unreadable,
                model_version=prediction.model_version,
            )
            for prediction in predictions
        ]
