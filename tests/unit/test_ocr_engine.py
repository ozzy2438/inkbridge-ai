"""Focused tests for selected-beam confidence and abstention evidence."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.pipeline.ocr_engine import TrOCREngine  # noqa: E402


class FakeGenerationModel:
    """Expose known selected-beam transition scores."""

    config = SimpleNamespace(pad_token_id=1)

    def __init__(self) -> None:
        self.beam_indices = None

    def compute_transition_scores(
        self, sequences, scores, beam_indices, *, normalize_logits: bool
    ):
        self.beam_indices = beam_indices
        assert normalize_logits is True
        return torch.tensor(
            [[math.log(0.8), math.log(0.5), 0.0], [math.log(0.9), math.log(0.9), math.log(0.9)]]
        )


def test_confidence_follows_selected_beam_transition_scores() -> None:
    """Beam search confidence must use generated-token lineage, not row maxima."""
    engine = TrOCREngine(device="cpu")
    model = FakeGenerationModel()
    engine._model = model
    beam_indices = torch.tensor([[0, 1, -1], [4, 5, 5]])
    outputs = SimpleNamespace(
        sequences=torch.tensor([[2, 10, 11, 1], [2, 20, 21, 22]]),
        scores=(torch.zeros(8, 30), torch.zeros(8, 30), torch.zeros(8, 30)),
        beam_indices=beam_indices,
    )

    confidences = engine._extract_confidences(outputs)

    assert model.beam_indices is beam_indices
    assert confidences[0][0] == pytest.approx(math.sqrt(0.8 * 0.5))
    assert confidences[0][1] == pytest.approx([0.8, 0.5])
    assert confidences[1][0] == pytest.approx(0.9)


def test_unreadable_sentinel_does_not_destroy_raw_prediction(monkeypatch) -> None:
    """Routing may abstain, while evaluation still receives the model's raw text."""
    engine = TrOCREngine(device="cpu", abstention_threshold=0.4)
    engine._model = object()
    engine._processor = object()
    monkeypatch.setattr(engine, "_predict_single", lambda image: ("raw answer", 0.2, [0.2]))

    prediction = engine.predict_line(object(), [0, 0, 10, 10])

    assert prediction.text == "[UNREADABLE]"
    assert prediction.raw_text == "raw answer"
    assert prediction.is_unreadable is True
