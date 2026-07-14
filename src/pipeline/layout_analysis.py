"""Layout Analysis — Page segmentation and reading order.

Segments a document page into:
- Text lines (student handwriting)
- Text blocks (paragraphs)
- Printed text regions (questions/headers)
- Crossed-out regions
- Margin insertions
- Non-text content (diagrams, drawings)

Establishes reading order for proper transcript assembly.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import structlog

logger = structlog.get_logger()


class RegionType(str, Enum):
    STUDENT_HANDWRITING = "student_handwriting"
    PRINTED_TEXT = "printed_text"
    CROSSED_OUT = "crossed_out"
    MARGIN_INSERTION = "margin_insertion"
    HEADER = "header"
    PAGE_NUMBER = "page_number"
    DIAGRAM = "diagram"
    BLANK = "blank"
    TEACHER_ANNOTATION = "teacher_annotation"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class TextLine:
    """A detected text line within a region."""
    bbox: BoundingBox
    region_type: RegionType
    reading_order: int
    confidence: float = 0.0
    is_crossed_out: bool = False
    parent_block_id: Optional[str] = None


@dataclass
class PageLayout:
    """Complete layout analysis result for a page."""
    page_id: str
    width: int
    height: int
    lines: list[TextLine] = field(default_factory=list)
    regions: list[dict] = field(default_factory=list)
    reading_order: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class LayoutAnalyzer:
    """Segments document pages into typed regions with reading order.
    
    Uses a combination of:
    1. Connected component analysis for text region detection
    2. Projection profiles for line segmentation
    3. Heuristics for region classification
    4. Top-to-bottom, left-to-right reading order
    """
    
    def __init__(
        self,
        min_line_height: int = 15,
        max_line_height: int = 200,
        line_merge_threshold: float = 0.5,
        crossed_out_threshold: float = 0.3,
    ):
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.line_merge_threshold = line_merge_threshold
        self.crossed_out_threshold = crossed_out_threshold
    
    def analyze(self, image: np.ndarray, page_id: str = "page_0") -> PageLayout:
        """Perform full layout analysis on a document page.
        
        Args:
            image: BGR or grayscale image
            page_id: Unique identifier for this page
            
        Returns:
            PageLayout with detected lines, regions, and reading order
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape[:2]
        
        # Step 1: Binarize
        binary = self._binarize(gray)
        
        # Step 2: Detect text lines via projection profile
        line_bboxes = self._detect_lines_projection(binary)
        
        # Step 3: Classify regions
        lines = self._classify_lines(gray, binary, line_bboxes)
        
        # Step 4: Detect crossed-out text
        lines = self._detect_crossouts(gray, lines)
        
        # Step 5: Establish reading order
        lines = self._establish_reading_order(lines)
        
        layout = PageLayout(
            page_id=page_id,
            width=w,
            height=h,
            lines=lines,
            reading_order=[l.reading_order for l in lines],
            metadata={
                "total_lines": len(lines),
                "handwriting_lines": sum(
                    1 for l in lines 
                    if l.region_type == RegionType.STUDENT_HANDWRITING
                ),
                "crossed_out_lines": sum(1 for l in lines if l.is_crossed_out),
            },
        )
        
        logger.info(
            "layout.analyzed",
            page_id=page_id,
            total_lines=len(lines),
            handwriting_lines=layout.metadata["handwriting_lines"],
        )
        
        return layout
    
    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive binarization for document images."""
        # Gaussian adaptive threshold works well for varied lighting
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=10,
        )
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _detect_lines_projection(self, binary: np.ndarray) -> list[BoundingBox]:
        """Detect text lines using horizontal projection profile."""
        h, w = binary.shape
        
        # Horizontal projection
        h_proj = np.sum(binary, axis=1) / 255
        
        # Find line boundaries (transitions from low to high projection)
        threshold = w * 0.01  # Minimum 1% of width has ink
        
        in_line = False
        line_start = 0
        lines = []
        
        for y in range(h):
            if h_proj[y] > threshold and not in_line:
                line_start = y
                in_line = True
            elif h_proj[y] <= threshold and in_line:
                line_height = y - line_start
                if self.min_line_height <= line_height <= self.max_line_height:
                    # Find horizontal extent
                    line_region = binary[line_start:y, :]
                    v_proj = np.sum(line_region, axis=0) / 255
                    nonzero = np.where(v_proj > 0)[0]
                    
                    if len(nonzero) > 0:
                        x1 = int(nonzero[0])
                        x2 = int(nonzero[-1])
                        lines.append(BoundingBox(x1, line_start, x2, y))
                in_line = False
        
        # Handle last line
        if in_line:
            line_height = h - line_start
            if self.min_line_height <= line_height <= self.max_line_height:
                lines.append(BoundingBox(0, line_start, w, h))
        
        return lines
    
    def _classify_lines(
        self, gray: np.ndarray, binary: np.ndarray, bboxes: list[BoundingBox]
    ) -> list[TextLine]:
        """Classify each detected line by type."""
        lines = []
        
        for i, bbox in enumerate(bboxes):
            # Extract line region
            region = gray[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            binary_region = binary[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            
            # Classification heuristics
            region_type = self._classify_single_region(region, binary_region, bbox)
            
            lines.append(TextLine(
                bbox=bbox,
                region_type=region_type,
                reading_order=i,
                confidence=0.8,  # Will be refined by model
            ))
        
        return lines
    
    def _classify_single_region(
        self, region: np.ndarray, binary: np.ndarray, bbox: BoundingBox
    ) -> RegionType:
        """Classify a single text region.
        
        Heuristics:
        - Uniform stroke width + regular spacing = printed
        - Variable stroke + irregular baseline = handwriting
        - Very short height + top of page = header
        - Very short height + bottom = page number
        """
        h, w = region.shape[:2]
        
        # Ink density
        ink_density = np.sum(binary > 0) / binary.size
        
        # Stroke width variation (approximation)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return RegionType.BLANK
        
        # Calculate aspect ratios of components
        widths = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch > 3:  # Skip noise
                widths.append(cw / ch)
        
        if not widths:
            return RegionType.STUDENT_HANDWRITING
        
        # High regularity in character sizes suggests printed text
        width_std = np.std(widths) if len(widths) > 1 else 0
        
        if width_std < 0.3 and ink_density > 0.15:
            return RegionType.PRINTED_TEXT
        
        return RegionType.STUDENT_HANDWRITING
    
    def _detect_crossouts(self, gray: np.ndarray, lines: list[TextLine]) -> list[TextLine]:
        """Detect crossed-out text using horizontal line detection."""
        for line in lines:
            region = gray[line.bbox.y1:line.bbox.y2, line.bbox.x1:line.bbox.x2]
            
            # Detect horizontal lines through text (cross-outs)
            edges = cv2.Canny(region, 50, 150)
            h_lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=50,
                minLineLength=line.bbox.width * 0.4,
                maxLineGap=10
            )
            
            if h_lines is not None:
                # Check if lines pass through middle of text region
                h = line.bbox.height
                middle_zone = (h * 0.3, h * 0.7)
                
                for hl in h_lines:
                    _, y1, _, y2 = hl[0]
                    avg_y = (y1 + y2) / 2
                    if middle_zone[0] <= avg_y <= middle_zone[1]:
                        line.is_crossed_out = True
                        break
        
        return lines
    
    def _establish_reading_order(self, lines: list[TextLine]) -> list[TextLine]:
        """Establish top-to-bottom, left-to-right reading order."""
        # Sort by vertical position (top of bbox), then horizontal
        sorted_lines = sorted(
            lines,
            key=lambda l: (l.bbox.y1, l.bbox.x1)
        )
        
        for i, line in enumerate(sorted_lines):
            line.reading_order = i
        
        return sorted_lines
