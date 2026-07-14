"""VLM Fallback — Vision-Language Model for hard pages.

Uses a compact VLM to handle pages that are too complex
for line-level TrOCR:
- Full-page layout understanding
- Multi-column text
- Mixed handwriting and printed text
- Severely degraded images
- Complex reading order

Outputs structured JSON with typed blocks.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

import structlog
from PIL import Image

logger = structlog.get_logger()


@dataclass
class VLMBlock:
    """A single text block identified by the VLM."""
    block_type: str  # student_handwriting, printed_text, crossed_out, etc.
    reading_order: int
    text: str
    confidence: float
    bbox: list[int]
    crossed_out: bool = False
    is_uncertain: bool = False


@dataclass
class VLMResult:
    """Full page VLM transcription result."""
    blocks: list[VLMBlock]
    full_text: str
    avg_confidence: float
    processing_time_ms: float
    model_version: str
    raw_output: Optional[str] = None


VLM_SYSTEM_PROMPT = """You are a document transcription expert. Analyze the handwritten document image and extract all text content.

For each text region, output a JSON object with:
- "type": one of ["student_handwriting", "printed_text", "crossed_out", "margin_insertion", "header", "page_number"]
- "reading_order": integer starting from 1
- "text": the transcribed text content
- "confidence": your confidence 0.0-1.0 in the transcription accuracy
- "crossed_out": boolean, true if text has a line through it
- "bbox": [x1, y1, x2, y2] approximate bounding box as percentage of page dimensions

Rules:
1. Preserve original spelling and grammar (do not correct)
2. If text is unreadable, set text to "[UNREADABLE]" and confidence to 0.0
3. If text is crossed out, still transcribe it but mark crossed_out: true
4. Maintain correct reading order (top-to-bottom, left-to-right)
5. Separate student handwriting from printed questions/headers

Output ONLY valid JSON array. No other text."""


class VLMFallbackEngine:
    """VLM-based full-page transcription for complex documents.
    
    Supports multiple VLM backends:
    - Local: Qwen2-VL, LLaVA (via transformers)
    - API: GPT-4o, Claude 3.5 Sonnet (via httpx)
    
    The VLM produces structured JSON output that preserves:
    - Document layout and reading order
    - Block type classification
    - Per-block confidence scores
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        backend: str = "local",  # "local" or "api"
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.backend = backend
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._model_version = f"vlm-{model_name.split('/')[-1]}-v0.1"
    
    def load(self):
        """Load VLM model (local backend only)."""
        if self.backend != "local":
            return
        
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        
        logger.info("vlm.loading", model=self.model_name)
        
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        logger.info("vlm.loaded")
    
    async def transcribe_page(self, image: Image.Image) -> VLMResult:
        """Transcribe a full page using VLM.
        
        Args:
            image: PIL Image of the full page
            
        Returns:
            VLMResult with structured blocks and text
        """
        start_time = time.time()
        
        if self.backend == "api":
            raw_output = await self._call_api(image)
        else:
            raw_output = self._call_local(image)
        
        # Parse structured output
        blocks = self._parse_output(raw_output)
        
        # Assemble full text
        sorted_blocks = sorted(blocks, key=lambda b: b.reading_order)
        full_text = "\n".join(b.text for b in sorted_blocks if not b.crossed_out)
        
        # Average confidence
        confidences = [b.confidence for b in blocks]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        return VLMResult(
            blocks=blocks,
            full_text=full_text,
            avg_confidence=avg_confidence,
            processing_time_ms=processing_time,
            model_version=self._model_version,
            raw_output=raw_output,
        )
    
    async def _call_api(self, image: Image.Image) -> str:
        """Call VLM via API (OpenAI-compatible)."""
        import httpx
        import base64
        import io
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.api_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": VLM_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Transcribe this handwritten document page."
                                },
                            ],
                        },
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    def _call_local(self, image: Image.Image) -> str:
        """Call local VLM model."""
        if self._model is None:
            self.load()
        
        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Transcribe this handwritten document page."},
                ],
            },
        ]
        
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input], images=[image], return_tensors="pt"
        ).to(self._model.device)
        
        import torch
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        
        output_text = self._processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return output_text
    
    def _parse_output(self, raw_output: str) -> list[VLMBlock]:
        """Parse VLM JSON output into structured blocks."""
        try:
            # Try to extract JSON from the output
            # Handle cases where VLM wraps in markdown code blocks
            output = raw_output.strip()
            if output.startswith("```json"):
                output = output[7:]
            if output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
            
            blocks_data = json.loads(output.strip())
            
            if not isinstance(blocks_data, list):
                blocks_data = [blocks_data]
            
            blocks = []
            for item in blocks_data:
                blocks.append(VLMBlock(
                    block_type=item.get("type", "student_handwriting"),
                    reading_order=item.get("reading_order", 0),
                    text=item.get("text", ""),
                    confidence=item.get("confidence", 0.5),
                    bbox=item.get("bbox", [0, 0, 100, 100]),
                    crossed_out=item.get("crossed_out", False),
                    is_uncertain=item.get("confidence", 0.5) < 0.7,
                ))
            
            return blocks
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                "vlm.parse_error",
                error=str(e),
                raw_output_length=len(raw_output),
            )
            # Return a single block with the raw text
            return [VLMBlock(
                block_type="student_handwriting",
                reading_order=1,
                text=raw_output,
                confidence=0.3,
                bbox=[0, 0, 100, 100],
                is_uncertain=True,
            )]
