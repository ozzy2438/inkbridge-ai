"""OCR Engine — TrOCR-based handwriting recognition.

Fine-tuned TrOCR model for student handwriting recognition.
Supports:
- Line-level transcription
- Confidence scoring via token log probabilities
- Batch inference with dynamic batching
- Abstention for uncertain predictions
"""

import time
from dataclasses import dataclass

import structlog
import torch
from PIL import Image

logger = structlog.get_logger()


@dataclass
class OCRPrediction:
    """Single line OCR prediction with metadata."""

    text: str
    raw_text: str
    confidence: float
    token_confidences: list[float]
    is_uncertain: bool
    is_unreadable: bool
    processing_time_ms: float
    model_version: str
    bbox: list[int]
    crossed_out: bool = False


class TrOCREngine:
    """Production TrOCR inference engine.

    Features:
    - Lazy model loading with warm-up
    - FP16 inference for speed
    - Beam search with configurable beam width
    - Token-level confidence extraction
    - Abstention when confidence below threshold
    - Batch processing for throughput
    """

    def __init__(
        self,
        model_path: str = "microsoft/trocr-base-handwritten",
        device: str | None = None,
        beam_width: int = 4,
        max_length: int = 128,
        confidence_threshold: float = 0.7,
        abstention_threshold: float = 0.4,
        use_fp16: bool = True,
        batch_size: int = 16,
        revision: str | None = None,
        model_version: str | None = None,
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.beam_width = beam_width
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.abstention_threshold = abstention_threshold
        self.use_fp16 = use_fp16 and self.device == "cuda"
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.revision = revision

        self._model = None
        self._processor = None
        self._model_version = model_version or f"{model_path}@{revision or 'unresolved'}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load model and processor."""
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )

        logger.info("trocr.loading", model_path=self.model_path, device=self.device)

        self._processor = TrOCRProcessor.from_pretrained(
            self.model_path, revision=self.revision
        )
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.model_path, revision=self.revision
        )

        if self.use_fp16:
            self._model = self._model.half()

        self._model = self._model.to(self.device)
        self._model.eval()

        # Warm up with dummy input
        self._warmup()

        logger.info("trocr.loaded", device=self.device, fp16=self.use_fp16)

    def _warmup(self):
        """Warm up the model with a dummy inference."""
        dummy_image = Image.new("RGB", (384, 384), color="white")
        self._predict_single(dummy_image)

    def predict_line(self, image: Image.Image, bbox: list[int]) -> OCRPrediction:
        """Predict text for a single cropped line image.

        Args:
            image: PIL Image of cropped text line
            bbox: Bounding box [x1, y1, x2, y2] on original page

        Returns:
            OCRPrediction with text, confidence, and metadata
        """
        if not self.is_loaded:
            self.load()
        start_time = time.perf_counter()

        raw_text, confidence, token_confs = self._predict_single(image)

        processing_time = (time.perf_counter() - start_time) * 1000

        # Determine if prediction is uncertain or unreadable
        is_uncertain = confidence < self.confidence_threshold
        is_unreadable = confidence < self.abstention_threshold

        text = "[UNREADABLE]" if is_unreadable else raw_text

        return OCRPrediction(
            text=text,
            raw_text=raw_text,
            confidence=confidence,
            token_confidences=token_confs,
            is_uncertain=is_uncertain,
            is_unreadable=is_unreadable,
            processing_time_ms=processing_time,
            model_version=self._model_version,
            bbox=bbox,
        )

    def predict_batch(
        self, images: list[Image.Image], bboxes: list[list[int]]
    ) -> list[OCRPrediction]:
        """Batch prediction for multiple line images.

        Uses dynamic batching for throughput optimization.
        """
        if not self.is_loaded:
            self.load()
        if len(images) != len(bboxes):
            raise ValueError("images and bboxes must contain the same number of items")
        start_time = time.perf_counter()

        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise RuntimeError("TrOCR model failed to load")

        # Process in batches
        results = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_bboxes = bboxes[i : i + self.batch_size]

            # Preprocess batch
            pixel_values = processor(images=batch_images, return_tensors="pt").pixel_values

            if self.use_fp16:
                pixel_values = pixel_values.half()

            pixel_values = pixel_values.to(self.device)

            # Generate with scores
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    max_length=self.max_length,
                    num_beams=self.beam_width,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Decode predictions
            texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            confidences = self._extract_confidences(outputs)
            if len(confidences) != len(texts):
                raise RuntimeError("TrOCR confidence output does not match decoded sequences")

            # Extract confidence scores
            for raw_text, bbox, confidence_result in zip(
                texts, batch_bboxes, confidences
            ):
                confidence, token_confs = confidence_result

                is_uncertain = confidence < self.confidence_threshold
                is_unreadable = confidence < self.abstention_threshold

                text = "[UNREADABLE]" if is_unreadable else raw_text

                results.append(
                    OCRPrediction(
                        text=text,
                        raw_text=raw_text,
                        confidence=confidence,
                        token_confidences=token_confs,
                        is_uncertain=is_uncertain,
                        is_unreadable=is_unreadable,
                        processing_time_ms=0,  # Set after batch
                        model_version=self._model_version,
                        bbox=bbox,
                    )
                )

        # Set per-item processing time
        total_time = (time.perf_counter() - start_time) * 1000
        per_item_time = total_time / len(results) if results else 0
        for r in results:
            r.processing_time_ms = per_item_time

        return results

    def _predict_single(self, image: Image.Image) -> tuple[str, float, list[float]]:
        """Single image prediction with confidence extraction."""
        if not self.is_loaded:
            self.load()

        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise RuntimeError("TrOCR model failed to load")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        if self.use_fp16:
            pixel_values = pixel_values.half()

        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.beam_width,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        confidence, token_confs = self._extract_confidences(outputs)[0]

        return text, confidence, token_confs

    def _extract_confidences(self, outputs) -> list[tuple[float, list[float]]]:
        """Compute geometric-mean token confidence along each selected beam path."""
        if not hasattr(outputs, "scores") or outputs.scores is None:
            return [(0.5, []) for _ in outputs.sequences]

        model = self._model
        if model is None:
            raise RuntimeError("TrOCR model failed to load")
        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            getattr(outputs, "beam_indices", None),
            normalize_logits=True,
        )
        generated_token_ids = outputs.sequences[:, -transition_scores.shape[1] :]
        pad_token_id = getattr(model.config, "pad_token_id", None)

        results: list[tuple[float, list[float]]] = []
        for token_ids, log_probabilities in zip(generated_token_ids, transition_scores):
            if pad_token_id is None:
                active_scores = log_probabilities[log_probabilities != 0]
            else:
                active_scores = log_probabilities[token_ids != pad_token_id]
            if active_scores.numel() == 0:
                results.append((0.5, []))
                continue
            token_confidences = active_scores.exp().tolist()
            sequence_confidence = active_scores.mean().exp().item()
            results.append((float(sequence_confidence), [float(v) for v in token_confidences]))

        return results

    def unload(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("trocr.unloaded")
