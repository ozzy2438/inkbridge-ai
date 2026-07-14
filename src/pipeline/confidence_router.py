"""Confidence Router — Route pages to appropriate model.

Decision logic:
- Normal pages (clear, single-column) → TrOCR (fast, cheap)
- Hard pages (multi-column, degraded, complex layout) → VLM fallback
- Very degraded pages → Flag for human review

The router considers:
- Quality gate score
- Layout complexity
- Historical performance on similar pages
- Cost/latency budgets
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import structlog

from src.pipeline.quality_gate import QualityReport
from src.pipeline.layout_analysis import PageLayout, RegionType

logger = structlog.get_logger()


class ModelRoute(str, Enum):
    TROCR_FAST = "trocr_fast"        # Fine-tuned TrOCR (line-level)
    VLM_FALLBACK = "vlm_fallback"    # Full-page VLM
    HYBRID = "hybrid"                 # TrOCR + VLM for uncertain lines
    HUMAN_ONLY = "human_only"         # Too degraded for any model


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    route: ModelRoute
    confidence: float
    reason: str
    estimated_latency_ms: float
    estimated_cost_usd: float
    metadata: dict


class ConfidenceRouter:
    """Routes pages to the most appropriate model based on complexity.
    
    The router balances accuracy, latency, and cost:
    - TrOCR: ~50ms/line, ~$0.001/page, 95%+ accuracy on clear text
    - VLM: ~2000ms/page, ~$0.01/page, better on complex layouts
    - Human: ~30s/page, ~$0.15/page, required for unreadable text
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.6,
        complexity_threshold: int = 5,
        vlm_cost_per_page: float = 0.01,
        trocr_cost_per_line: float = 0.0001,
    ):
        self.quality_threshold = quality_threshold
        self.complexity_threshold = complexity_threshold
        self.vlm_cost_per_page = vlm_cost_per_page
        self.trocr_cost_per_line = trocr_cost_per_line
    
    def route(
        self,
        quality_report: QualityReport,
        layout: PageLayout,
        force_vlm: bool = False,
    ) -> RoutingDecision:
        """Determine the best model route for a page.
        
        Args:
            quality_report: Output from quality gate
            layout: Output from layout analysis
            force_vlm: Override to always use VLM
            
        Returns:
            RoutingDecision with model, confidence, and cost estimate
        """
        if force_vlm:
            return RoutingDecision(
                route=ModelRoute.VLM_FALLBACK,
                confidence=0.9,
                reason="Forced VLM by user request",
                estimated_latency_ms=2000.0,
                estimated_cost_usd=self.vlm_cost_per_page,
                metadata={"forced": True},
            )
        
        # Check if page is too degraded for any model
        if quality_report.recommendation == "re-upload":
            return RoutingDecision(
                route=ModelRoute.HUMAN_ONLY,
                confidence=0.1,
                reason=f"Quality too low: {quality_report.issues}",
                estimated_latency_ms=0,
                estimated_cost_usd=0.15,
                metadata={"quality_score": quality_report.score},
            )
        
        # Check complexity indicators
        complexity_score = self._calculate_complexity(layout, quality_report)
        
        # Route decision
        if complexity_score < 3:
            # Simple page → TrOCR is sufficient
            num_lines = layout.metadata.get("handwriting_lines", 10)
            return RoutingDecision(
                route=ModelRoute.TROCR_FAST,
                confidence=0.85,
                reason="Clear page with simple layout",
                estimated_latency_ms=num_lines * 50.0,
                estimated_cost_usd=num_lines * self.trocr_cost_per_line,
                metadata={"complexity_score": complexity_score},
            )
        elif complexity_score < 7:
            # Moderate complexity → Hybrid approach
            return RoutingDecision(
                route=ModelRoute.HYBRID,
                confidence=0.7,
                reason="Moderate complexity, using hybrid approach",
                estimated_latency_ms=1500.0,
                estimated_cost_usd=self.vlm_cost_per_page * 0.5,
                metadata={"complexity_score": complexity_score},
            )
        else:
            # High complexity → VLM
            return RoutingDecision(
                route=ModelRoute.VLM_FALLBACK,
                confidence=0.75,
                reason="Complex layout requires full-page understanding",
                estimated_latency_ms=2500.0,
                estimated_cost_usd=self.vlm_cost_per_page,
                metadata={"complexity_score": complexity_score},
            )
    
    def _calculate_complexity(
        self, layout: PageLayout, quality: QualityReport
    ) -> float:
        """Calculate page complexity score (0-10).
        
        Factors:
        - Number of quality issues
        - Number of region types present
        - Presence of crossed-out text
        - Margin insertions
        - Mixed printed/handwritten content
        """
        score = 0.0
        
        # Quality issues add complexity
        score += len(quality.issues) * 1.5
        
        # Multiple region types
        region_types = set(l.region_type for l in layout.lines)
        score += (len(region_types) - 1) * 1.0
        
        # Crossed-out text
        crossed_out_count = sum(1 for l in layout.lines if l.is_crossed_out)
        score += crossed_out_count * 0.5
        
        # Margin insertions
        insertions = sum(
            1 for l in layout.lines
            if l.region_type == RegionType.MARGIN_INSERTION
        )
        score += insertions * 1.0
        
        # Mixed content penalty
        has_printed = RegionType.PRINTED_TEXT in region_types
        has_handwriting = RegionType.STUDENT_HANDWRITING in region_types
        if has_printed and has_handwriting:
            score += 2.0
        
        return min(10.0, score)
