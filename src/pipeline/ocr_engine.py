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
from typing import Optional

import numpy as np
import torch
from PIL import Image
import structlog

logger = structlog.get_logger()


@dataclass
class OCRPrediction:
    """Single line OCR prediction with metadata."""
    text: str
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
        device: Optional[str] = None,
        beam_width: int = 4,
        max_length: int = 128,
        confidence_threshold: float = 0.7,
        abstention_threshold: float = 0.4,
        use_fp16: bool = True,
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.beam_width = beam_width
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.abstention_threshold = abstention_threshold
        self.use_fp16 = use_fp16 and self.device == "cuda"
        
        self._model = None
        self._processor = None
        self._model_version = "trocr-base-handwritten-v0.1"
    
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
        
        self._processor = TrOCRProcessor.from_pretrained(self.model_path)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        
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
        start_time = time.time()
        
        text, confidence, token_confs = self._predict_single(image)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Determine if prediction is uncertain or unreadable
        is_uncertain = confidence < self.confidence_threshold
        is_unreadable = confidence < self.abstention_threshold
        
        if is_unreadable:
            text = "[UNREADABLE]"
        
        return OCRPrediction(
            text=text,
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
        start_time = time.time()
        
        if not self.is_loaded:
            self.load()
        
        # Process in batches
        batch_size = 16
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_bboxes = bboxes[i:i + batch_size]
            
            # Preprocess batch
            pixel_values = self._processor(
                images=batch_images,
                return_tensors="pt"
            ).pixel_values
            
            if self.use_fp16:
                pixel_values = pixel_values.half()
            
            pixel_values = pixel_values.to(self.device)
            
            # Generate with scores
            with torch.no_grad():
                outputs = self._model.generate(
                    pixel_values,
                    max_length=self.max_length,
                    num_beams=self.beam_width,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            
            # Decode predictions
            texts = self._processor.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )
            
            # Extract confidence scores
            for j, (text, bbox) in enumerate(zip(texts, batch_bboxes)):
                confidence, token_confs = self._extract_confidence(
                    outputs, j
                )
                
                is_uncertain = confidence < self.confidence_threshold
                is_unreadable = confidence < self.abstention_threshold
                
                if is_unreadable:
                    text = "[UNREADABLE]"
                
                results.append(OCRPrediction(
                    text=text,
                    confidence=confidence,
                    token_confidences=token_confs,
                    is_uncertain=is_uncertain,
                    is_unreadable=is_unreadable,
                    processing_time_ms=0,  # Set after batch
                    model_version=self._model_version,
                    bbox=bbox,
                ))
        
        # Set per-item processing time
        total_time = (time.time() - start_time) * 1000
        per_item_time = total_time / len(results) if results else 0
        for r in results:
            r.processing_time_ms = per_item_time
        
        return results
    
    def _predict_single(self, image: Image.Image) -> tuple[str, float, list[float]]:
        """Single image prediction with confidence extraction."""
        if not self.is_loaded:
            self.load()
        
        pixel_values = self._processor(
            images=image, return_tensors="pt"
        ).pixel_values
        
        if self.use_fp16:
            pixel_values = pixel_values.half()
        
        pixel_values = pixel_values.to(self.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.beam_width,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        text = self._processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0]
        
        confidence, token_confs = self._extract_confidence(outputs, 0)
        
        return text, confidence, token_confs
    
    def _extract_confidence(
        self, outputs, batch_idx: int
    ) -> tuple[float, list[float]]:
        """Extract confidence from generation output scores.
        
        Uses average token-level log probability as confidence.
        """
        if not hasattr(outputs, 'scores') or outputs.scores is None:
            return 0.5, []
        
        token_confs = []
        for score in outputs.scores:
            probs = torch.softmax(score[batch_idx], dim=-1)
            max_prob = probs.max().item()
            token_confs.append(max_prob)
        
        if token_confs:
            avg_confidence = sum(token_confs) / len(token_confs)
        else:
            avg_confidence = 0.5
        
        return avg_confidence, token_confs
    
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
