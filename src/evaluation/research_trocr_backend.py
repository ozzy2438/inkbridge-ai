"""Local-files-only TrOCR backend for non-commercial research inference."""

from __future__ import annotations

from typing import Any

from src.evaluation.protected_trocr_backend import ProtectedTrOCRPredictionBackend


class ResearchTrOCRPredictionBackend(ProtectedTrOCRPredictionBackend):
    """Reuse the sealed local loader while reporting a distinct research evidence scope."""

    @property
    def provenance(self) -> dict[str, Any]:
        """Return protected-grade runtime facts without claiming protected-pilot approval."""
        value = super().provenance
        value["backend"] = "trocr_local_research"
        value["evidence_scope"] = "noncommercial_research_rehearsal_only"
        value["protected_pilot_evidence"] = False
        return value
