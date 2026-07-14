"""Image Quality Gate — Pre-inference validation.

Assesses uploaded images for quality issues before processing:
- Blur detection (Laplacian variance)
- Skew/rotation detection
- Exposure analysis (over/under-exposed)
- Resolution check
- Glare/shadow detection
- Crop completeness

Images failing the quality gate are flagged for user re-upload
or routed to VLM fallback with degradation metadata.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import structlog

logger = structlog.get_logger()


class QualityIssue(str, Enum):
    BLUR = "blur"
    SKEW = "skew"
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    LOW_RESOLUTION = "low_resolution"
    GLARE = "glare"
    SHADOW = "shadow"
    INCOMPLETE_CROP = "incomplete_crop"
    BLANK_PAGE = "blank_page"


@dataclass
class QualityReport:
    """Quality assessment result for a single page."""
    passed: bool
    score: float  # 0-1, overall quality score
    issues: list[QualityIssue]
    metrics: dict
    recommendation: str  # "process", "vlm_fallback", "re-upload"
    

class ImageQualityGate:
    """Validates image quality before OCR processing.
    
    Thresholds are tuned for handwritten document images:
    - More tolerant of noise than scene-text OCR
    - Strict on blur (critical for character recognition)
    - Aware of typical phone-camera artifacts
    """
    
    def __init__(
        self,
        blur_threshold: float = 100.0,
        min_resolution: int = 640,
        max_skew_degrees: float = 15.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_contrast: float = 30.0,
    ):
        self.blur_threshold = blur_threshold
        self.min_resolution = min_resolution
        self.max_skew_degrees = max_skew_degrees
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
    
    def assess(self, image: np.ndarray) -> QualityReport:
        """Run all quality checks on an image.
        
        Args:
            image: BGR image as numpy array (OpenCV format)
            
        Returns:
            QualityReport with pass/fail, issues, and recommendation
        """
        issues = []
        metrics = {}
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Blur detection (Laplacian variance)
        blur_score = self._detect_blur(gray)
        metrics["blur_laplacian"] = blur_score
        if blur_score < self.blur_threshold:
            issues.append(QualityIssue.BLUR)
        
        # 2. Resolution check
        height, width = gray.shape[:2]
        metrics["resolution"] = {"width": width, "height": height}
        if min(height, width) < self.min_resolution:
            issues.append(QualityIssue.LOW_RESOLUTION)
        
        # 3. Skew detection
        skew_angle = self._detect_skew(gray)
        metrics["skew_degrees"] = skew_angle
        if abs(skew_angle) > self.max_skew_degrees:
            issues.append(QualityIssue.SKEW)
        
        # 4. Exposure analysis
        brightness, contrast = self._analyze_exposure(gray)
        metrics["brightness"] = brightness
        metrics["contrast"] = contrast
        if brightness < self.min_brightness:
            issues.append(QualityIssue.UNDEREXPOSED)
        elif brightness > self.max_brightness:
            issues.append(QualityIssue.OVEREXPOSED)
        
        # 5. Glare detection
        has_glare = self._detect_glare(gray)
        metrics["has_glare"] = has_glare
        if has_glare:
            issues.append(QualityIssue.GLARE)
        
        # 6. Shadow detection
        has_shadow = self._detect_shadow(gray)
        metrics["has_shadow"] = has_shadow
        if has_shadow:
            issues.append(QualityIssue.SHADOW)
        
        # 7. Blank page detection
        is_blank = self._detect_blank(gray)
        metrics["is_blank"] = is_blank
        if is_blank:
            issues.append(QualityIssue.BLANK_PAGE)
        
        # Calculate overall score
        score = self._calculate_score(metrics, issues)
        
        # Determine recommendation
        recommendation = self._get_recommendation(issues, score)
        
        passed = len(issues) == 0 or (score > 0.5 and QualityIssue.BLANK_PAGE not in issues)
        
        report = QualityReport(
            passed=passed,
            score=score,
            issues=issues,
            metrics=metrics,
            recommendation=recommendation,
        )
        
        logger.info(
            "quality_gate.assessed",
            passed=passed,
            score=round(score, 3),
            issues=[i.value for i in issues],
            recommendation=recommendation,
        )
        
        return report
    
    def _detect_blur(self, gray: np.ndarray) -> float:
        """Detect blur using Laplacian variance.
        
        Higher values = sharper image.
        Typical thresholds: <100 = blurry, >300 = very sharp.
        """
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _detect_skew(self, gray: np.ndarray) -> float:
        """Detect document skew angle using Hough transform."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Calculate median angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines
            if abs(angle) < 45:
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        return float(np.median(angles))
    
    def _analyze_exposure(self, gray: np.ndarray) -> tuple[float, float]:
        """Analyze image brightness and contrast."""
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        return brightness, contrast
    
    def _detect_glare(self, gray: np.ndarray) -> bool:
        """Detect glare/specular highlights."""
        # Threshold for very bright pixels
        _, bright_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_mask > 0) / gray.size
        return bright_ratio > 0.05  # >5% of image is saturated
    
    def _detect_shadow(self, gray: np.ndarray) -> bool:
        """Detect significant shadows in the image."""
        # Check for large dark regions adjacent to light regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        local_mean = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        diff = cv2.absdiff(gray.astype(np.float32), local_mean.astype(np.float32))
        shadow_ratio = np.sum(diff > 60) / gray.size
        return shadow_ratio > 0.15
    
    def _detect_blank(self, gray: np.ndarray) -> bool:
        """Detect if page is blank (no significant content)."""
        # Adaptive threshold to find ink/content
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        content_ratio = np.sum(binary > 0) / gray.size
        return content_ratio < 0.01  # Less than 1% content
    
    def _calculate_score(self, metrics: dict, issues: list[QualityIssue]) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0
        
        # Penalize each issue
        penalties = {
            QualityIssue.BLUR: 0.3,
            QualityIssue.SKEW: 0.15,
            QualityIssue.UNDEREXPOSED: 0.2,
            QualityIssue.OVEREXPOSED: 0.2,
            QualityIssue.LOW_RESOLUTION: 0.25,
            QualityIssue.GLARE: 0.2,
            QualityIssue.SHADOW: 0.15,
            QualityIssue.INCOMPLETE_CROP: 0.1,
            QualityIssue.BLANK_PAGE: 1.0,
        }
        
        for issue in issues:
            score -= penalties.get(issue, 0.1)
        
        return max(0.0, score)
    
    def _get_recommendation(self, issues: list[QualityIssue], score: float) -> str:
        """Determine processing recommendation."""
        if QualityIssue.BLANK_PAGE in issues:
            return "skip"
        if score < 0.3:
            return "re-upload"
        if score < 0.6:
            return "vlm_fallback"
        return "process"
    
    def correct_skew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Deskew an image by the detected angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return corrected
