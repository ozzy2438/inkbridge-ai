"""Local-files-only TrOCR backend for protected inference."""

from __future__ import annotations

import platform
import re
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.evaluation.prediction_producer import PredictionResult
from src.pipeline.ocr_engine import TrOCREngine


class ProtectedTrOCRPredictionBackend:
    """Load a materialized local TrOCR package without Hub resolution or remote code."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        model_version: str,
        model_artifact_sha256: str,
        device: str = "cpu",
        batch_size: int = 8,
        beam_width: int = 4,
        max_length: int = 128,
        confidence_threshold: float = 0.7,
        abstention_threshold: float = 0.4,
        use_fp16: bool = True,
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", model_version) is None:
            raise ValueError("model_version must contain only safe version characters")
        if len(model_artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in model_artifact_sha256
        ):
            raise ValueError("model_artifact_sha256 must be a lowercase SHA-256 digest")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self._engine = TrOCREngine(
            model_path=str(Path(model_dir).resolve(strict=True)),
            revision=None,
            model_version=model_version,
            device=device,
            batch_size=batch_size,
            beam_width=beam_width,
            max_length=max_length,
            confidence_threshold=confidence_threshold,
            abstention_threshold=abstention_threshold,
            use_fp16=use_fp16,
            local_files_only=True,
        )
        self._provenance = {
            "backend": "trocr_local_protected",
            "model_version": model_version,
            "model_artifact_sha256": model_artifact_sha256,
            "device": self._engine.device,
            "dtype": "float16" if self._engine.use_fp16 else "float32",
            "batch_size": batch_size,
            "beam_width": beam_width,
            "max_length": max_length,
            "confidence_threshold": confidence_threshold,
            "abstention_threshold": abstention_threshold,
            "local_files_only": True,
            "network_access_allowed": False,
            "external_ai_service": False,
            "remote_code_allowed": False,
            "processor_use_fast": False,
            "runtime": {
                "python": platform.python_version(),
                "os": platform.system(),
                "os_release": platform.release(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "transformers": version("transformers"),
            },
        }

    @property
    def provenance(self) -> dict[str, Any]:
        """Return immutable model/runtime evidence without exposing its local path."""
        return deepcopy(self._provenance)

    def predict_batch(self, images: list[Image.Image]) -> list[PredictionResult]:
        """Run local TrOCR and retain raw decoded text for protected evaluation."""
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
