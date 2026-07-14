"""Pipeline Orchestrator — Coordinates the full transcription pipeline.

Orchestrates:
1. Quality Gate → 2. Layout Analysis → 3. Confidence Router →
4. OCR/VLM Inference → 5. Post-processing → 6. Result Assembly
"""

import time
import uuid
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import structlog

from src.pipeline.quality_gate import ImageQualityGate
from src.pipeline.layout_analysis import LayoutAnalyzer, RegionType
from src.pipeline.confidence_router import ConfidenceRouter, ModelRoute
from src.pipeline.ocr_engine import TrOCREngine
from src.pipeline.vlm_fallback import VLMFallbackEngine

logger = structlog.get_logger()


class PipelineOrchestrator:
    """Orchestrates the complete handwriting transcription pipeline."""
    
    def __init__(self, model_registry=None):
        self.quality_gate = ImageQualityGate()
        self.layout_analyzer = LayoutAnalyzer()
        self.confidence_router = ConfidenceRouter()
        self.ocr_engine = TrOCREngine()
        self.vlm_engine = VLMFallbackEngine()
        self.model_registry = model_registry
    
    async def process_single_page(
        self,
        image_bytes: bytes,
        filename: str = "page.png",
        force_vlm: bool = False,
    ) -> dict:
        """Process a single page through the complete pipeline."""
        start_time = time.time()
        page_id = str(uuid.uuid4())
        
        logger.info("pipeline.start", page_id=page_id, filename=filename)
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_cv is None:
            raise ValueError("Failed to decode image")
        
        # Step 1: Quality Gate
        quality_report = self.quality_gate.assess(image_cv)
        
        # Auto-correct skew if detected
        if quality_report.metrics.get("skew_degrees", 0) != 0:
            skew = quality_report.metrics["skew_degrees"]
            if abs(skew) < 15:  # Only auto-correct mild skew
                image_cv = self.quality_gate.correct_skew(image_cv, skew)
        
        # Step 2: Layout Analysis
        layout = self.layout_analyzer.analyze(image_cv, page_id)
        
        # Step 3: Confidence Router
        routing = self.confidence_router.route(
            quality_report, layout, force_vlm=force_vlm
        )
        
        # Step 4: Inference based on route
        blocks = []
        uncertain_spans = []
        
        if routing.route == ModelRoute.TROCR_FAST:
            blocks, uncertain_spans = await self._run_trocr(image_cv, layout)
        elif routing.route == ModelRoute.VLM_FALLBACK:
            pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            vlm_result = await self.vlm_engine.transcribe_page(pil_image)
            blocks = [
                {
                    "type": b.block_type,
                    "reading_order": b.reading_order,
                    "text": b.text,
                    "confidence": b.confidence,
                    "bbox": b.bbox,
                    "crossed_out": b.crossed_out,
                }
                for b in vlm_result.blocks
            ]
            uncertain_spans = [
                {"text": b.text, "confidence": b.confidence, "bbox": b.bbox}
                for b in vlm_result.blocks if b.is_uncertain
            ]
        elif routing.route == ModelRoute.HYBRID:
            blocks, uncertain_spans = await self._run_hybrid(image_cv, layout)
        else:
            # Human only
            blocks = [{"type": "unreadable", "text": "", "confidence": 0.0}]
            uncertain_spans = [{"text": "", "confidence": 0.0, "bbox": [0, 0, 0, 0]}]
        
        # Assemble result
        full_text = "\n".join(
            b["text"] for b in sorted(blocks, key=lambda x: x.get("reading_order", 0))
            if not b.get("crossed_out", False) and b["text"]
        )
        
        # Calculate overall confidence
        confidences = [b["confidence"] for b in blocks if b["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        needs_review = len(uncertain_spans) > 0 or avg_confidence < 0.8
        
        result = {
            "page_id": page_id,
            "text": full_text,
            "blocks": blocks,
            "confidence_score": round(avg_confidence, 4),
            "model_version": self.ocr_engine._model_version,
            "processing_time_ms": round(processing_time, 2),
            "needs_review": needs_review,
            "uncertain_spans": uncertain_spans,
            "metadata": {
                "quality_score": quality_report.score,
                "quality_issues": [i.value for i in quality_report.issues],
                "route": routing.route.value,
                "total_lines": layout.metadata.get("total_lines", 0),
                "crossed_out_lines": layout.metadata.get("crossed_out_lines", 0),
            },
        }
        
        logger.info(
            "pipeline.complete",
            page_id=page_id,
            confidence=avg_confidence,
            route=routing.route.value,
            processing_time_ms=processing_time,
            needs_review=needs_review,
        )
        
        return result
    
    async def _run_trocr(self, image_cv: np.ndarray, layout) -> tuple[list, list]:
        """Run TrOCR on all detected lines."""
        blocks = []
        uncertain_spans = []
        
        # Crop and process each line
        pil_images = []
        bboxes = []
        line_metadata = []
        
        for line in layout.lines:
            if line.region_type == RegionType.BLANK:
                continue
            
            # Crop line from image
            cropped = image_cv[
                line.bbox.y1:line.bbox.y2,
                line.bbox.x1:line.bbox.x2
            ]
            
            if cropped.size == 0:
                continue
            
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            pil_images.append(pil_img)
            bboxes.append(line.bbox.to_list())
            line_metadata.append(line)
        
        if not pil_images:
            return blocks, uncertain_spans
        
        # Batch predict
        predictions = self.ocr_engine.predict_batch(pil_images, bboxes)
        
        for pred, meta in zip(predictions, line_metadata):
            block = {
                "type": meta.region_type.value,
                "reading_order": meta.reading_order,
                "text": pred.text,
                "confidence": pred.confidence,
                "bbox": pred.bbox,
                "crossed_out": meta.is_crossed_out or pred.crossed_out,
            }
            blocks.append(block)
            
            if pred.is_uncertain:
                uncertain_spans.append({
                    "text": pred.text,
                    "confidence": pred.confidence,
                    "bbox": pred.bbox,
                })
        
        return blocks, uncertain_spans
    
    async def _run_hybrid(self, image_cv: np.ndarray, layout) -> tuple[list, list]:
        """Hybrid: TrOCR for clear lines, VLM for uncertain ones."""
        # First pass with TrOCR
        blocks, uncertain_spans = await self._run_trocr(image_cv, layout)
        
        # If too many uncertain spans, fall back to VLM for the full page
        if len(uncertain_spans) > len(blocks) * 0.3:  # >30% uncertain
            pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            vlm_result = await self.vlm_engine.transcribe_page(pil_image)
            
            # Replace uncertain blocks with VLM results
            blocks = [
                {
                    "type": b.block_type,
                    "reading_order": b.reading_order,
                    "text": b.text,
                    "confidence": b.confidence,
                    "bbox": b.bbox,
                    "crossed_out": b.crossed_out,
                }
                for b in vlm_result.blocks
            ]
            uncertain_spans = [
                {"text": b.text, "confidence": b.confidence, "bbox": b.bbox}
                for b in vlm_result.blocks if b.is_uncertain
            ]
        
        return blocks, uncertain_spans
