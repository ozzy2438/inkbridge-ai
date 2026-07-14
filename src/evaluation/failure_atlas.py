"""Failure Atlas — Categorizes and tracks model failures.

Categories:
- Faint pencil
- Severe perspective
- Joined cursive
- Touching lines
- Crossed-out text
- Margin insertion
- Printed/handwritten mixture
- Teacher annotation
- Missing page edge
- Repeated page
- Blank page
- Diagram/non-text content
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import defaultdict
import json
from pathlib import Path
import structlog

logger = structlog.get_logger()


class FailureCategory(str, Enum):
    FAINT_PENCIL = "faint_pencil"
    SEVERE_PERSPECTIVE = "severe_perspective"
    JOINED_CURSIVE = "joined_cursive"
    TOUCHING_LINES = "touching_lines"
    CROSSED_OUT = "crossed_out"
    MARGIN_INSERTION = "margin_insertion"
    MIXED_PRINT_HANDWRITING = "mixed_print_handwriting"
    TEACHER_ANNOTATION = "teacher_annotation"
    MISSING_PAGE_EDGE = "missing_page_edge"
    REPEATED_PAGE = "repeated_page"
    BLANK_PAGE = "blank_page"
    DIAGRAM_NON_TEXT = "diagram_non_text"
    LOW_CONTRAST = "low_contrast"
    HEAVY_BLUR = "heavy_blur"
    JPEG_ARTIFACTS = "jpeg_artifacts"
    UNKNOWN = "unknown"


@dataclass
class FailureInstance:
    """A single failure instance."""
    instance_id: str
    category: FailureCategory
    image_path: str
    predicted_text: str
    reference_text: str
    confidence: float
    cer: float
    model_version: str
    notes: Optional[str] = None


@dataclass
class FailureAtlas:
    """Tracks and categorizes model failures across versions.
    
    Used for:
    - Understanding where the model fails
    - Tracking improvement across model versions
    - Prioritizing data collection and annotation
    - Regression testing
    """
    
    instances: list[FailureInstance] = field(default_factory=list)
    
    def add_failure(
        self,
        instance_id: str,
        category: FailureCategory,
        image_path: str,
        predicted_text: str,
        reference_text: str,
        confidence: float,
        cer: float,
        model_version: str,
        notes: Optional[str] = None,
    ):
        """Add a failure instance to the atlas."""
        instance = FailureInstance(
            instance_id=instance_id,
            category=category,
            image_path=image_path,
            predicted_text=predicted_text,
            reference_text=reference_text,
            confidence=confidence,
            cer=cer,
            model_version=model_version,
            notes=notes,
        )
        self.instances.append(instance)
    
    def get_summary(self) -> dict:
        """Get summary statistics by failure category."""
        by_category = defaultdict(list)
        for inst in self.instances:
            by_category[inst.category.value].append(inst)
        
        summary = {}
        for cat, instances in by_category.items():
            cers = [i.cer for i in instances]
            confs = [i.confidence for i in instances]
            summary[cat] = {
                "count": len(instances),
                "avg_cer": sum(cers) / len(cers),
                "avg_confidence": sum(confs) / len(confs),
                "max_cer": max(cers),
            }
        
        return summary
    
    def compare_versions(self, version_a: str, version_b: str) -> dict:
        """Compare failure patterns between two model versions."""
        a_instances = [i for i in self.instances if i.model_version == version_a]
        b_instances = [i for i in self.instances if i.model_version == version_b]
        
        a_by_cat = defaultdict(list)
        b_by_cat = defaultdict(list)
        
        for i in a_instances:
            a_by_cat[i.category.value].append(i.cer)
        for i in b_instances:
            b_by_cat[i.category.value].append(i.cer)
        
        comparison = {}
        all_cats = set(list(a_by_cat.keys()) + list(b_by_cat.keys()))
        
        for cat in all_cats:
            a_avg = sum(a_by_cat[cat]) / len(a_by_cat[cat]) if a_by_cat[cat] else None
            b_avg = sum(b_by_cat[cat]) / len(b_by_cat[cat]) if b_by_cat[cat] else None
            
            comparison[cat] = {
                f"{version_a}_avg_cer": a_avg,
                f"{version_b}_avg_cer": b_avg,
                "improved": (b_avg or 1.0) < (a_avg or 1.0) if a_avg and b_avg else None,
            }
        
        return comparison
    
    def save(self, path: str):
        """Save failure atlas to JSON."""
        data = [
            {
                "instance_id": i.instance_id,
                "category": i.category.value,
                "image_path": i.image_path,
                "predicted_text": i.predicted_text,
                "reference_text": i.reference_text,
                "confidence": i.confidence,
                "cer": i.cer,
                "model_version": i.model_version,
                "notes": i.notes,
            }
            for i in self.instances
        ]
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "FailureAtlas":
        """Load failure atlas from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        atlas = cls()
        for item in data:
            atlas.instances.append(FailureInstance(
                instance_id=item["instance_id"],
                category=FailureCategory(item["category"]),
                image_path=item["image_path"],
                predicted_text=item["predicted_text"],
                reference_text=item["reference_text"],
                confidence=item["confidence"],
                cer=item["cer"],
                model_version=item["model_version"],
                notes=item.get("notes"),
            ))
        
        return atlas
