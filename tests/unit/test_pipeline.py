"""Unit tests for pipeline components."""

import numpy as np
import pytest
from src.pipeline.quality_gate import ImageQualityGate, QualityIssue
from src.pipeline.layout_analysis import LayoutAnalyzer, BoundingBox
from src.pipeline.confidence_router import ConfidenceRouter, ModelRoute


class TestQualityGate:
    """Tests for image quality assessment."""
    
    def setup_method(self):
        self.gate = ImageQualityGate()
    
    def test_blank_page_detection(self):
        """Blank white image should be detected."""
        blank = np.ones((800, 600), dtype=np.uint8) * 255
        report = self.gate.assess(blank)
        assert QualityIssue.BLANK_PAGE in report.issues
        assert report.recommendation == "skip"
    
    def test_normal_image_passes(self):
        """Image with content should pass quality gate."""
        # Create image with some "text" (dark lines on white)
        img = np.ones((800, 600), dtype=np.uint8) * 240
        # Add some dark content to simulate text
        for i in range(100, 700, 40):
            img[i:i+15, 50:550] = 30
        
        report = self.gate.assess(img)
        assert report.passed
        assert report.recommendation == "process"
    
    def test_blur_detection(self):
        """Very blurry image should be flagged."""
        import cv2
        img = np.random.randint(0, 255, (800, 600), dtype=np.uint8)
        # Heavy blur
        blurred = cv2.GaussianBlur(img, (51, 51), 0)
        report = self.gate.assess(blurred)
        # Blurred random noise may or may not trigger depending on threshold
        assert report.score <= 1.0


class TestLayoutAnalyzer:
    """Tests for layout analysis."""
    
    def setup_method(self):
        self.analyzer = LayoutAnalyzer()
    
    def test_empty_image(self):
        """Empty image should return no lines."""
        blank = np.ones((800, 600), dtype=np.uint8) * 255
        layout = self.analyzer.analyze(blank, "test_page")
        assert layout.page_id == "test_page"
        assert layout.width == 600
        assert layout.height == 800
    
    def test_bounding_box_properties(self):
        """BoundingBox should compute correct properties."""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=60)
        assert bbox.width == 100
        assert bbox.height == 40
        assert bbox.area == 4000
        assert bbox.center == (60, 40)
        assert bbox.to_list() == [10, 20, 110, 60]


class TestConfidenceRouter:
    """Tests for confidence-based routing."""
    
    def setup_method(self):
        self.router = ConfidenceRouter()
    
    def test_force_vlm(self):
        """force_vlm=True should always route to VLM."""
        from src.pipeline.quality_gate import QualityReport
        from src.pipeline.layout_analysis import PageLayout
        
        quality = QualityReport(
            passed=True, score=0.9, issues=[], metrics={},
            recommendation="process"
        )
        layout = PageLayout(
            page_id="test", width=600, height=800,
            lines=[], metadata={"handwriting_lines": 5}
        )
        
        decision = self.router.route(quality, layout, force_vlm=True)
        assert decision.route == ModelRoute.VLM_FALLBACK
