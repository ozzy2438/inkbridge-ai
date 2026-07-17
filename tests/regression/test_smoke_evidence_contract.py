"""Regression lock for the captured synthetic TrOCR smoke evidence."""

import json
from pathlib import Path

from src.evaluation.runner import (
    evaluate_records,
    evaluation_set_sha256,
    load_evaluation_records,
    sha256_file,
)

ROOT = Path(__file__).parents[2]
EVIDENCE = ROOT / "artifacts" / "smoke" / "openhand-synth-trocr-base-v1"
EVALUATION = EVIDENCE / (
    "eval_microsoft-trocr-base-handwritten-"
    "eaacaf452b06415df8f10bb6fad3a4c11e609406_openhand-synth-smoke-test-v1.json"
)
MODEL_REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"


def test_smoke_evidence_hashes_and_metrics_remain_traceable() -> None:
    """Published predictions must continue to reproduce their captured report."""
    predictions_path = EVIDENCE / "predictions.jsonl"
    metadata = json.loads((EVIDENCE / "predictions.meta.json").read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    records = load_evaluation_records(predictions_path)
    recalculated = evaluate_records(records)

    assert len(records) == 6
    assert metadata["output"]["sha256"] == sha256_file(predictions_path)
    assert metadata["output"]["num_records"] == len(records)
    assert metadata["model"]["resolved_revision"] == MODEL_REVISION
    assert metadata["privacy"] == {
        "source_images_exported": False,
        "writer_ids_exported": False,
    }
    assert evaluation["input"]["sha256"] == sha256_file(predictions_path)
    assert evaluation["input"]["evaluation_set_sha256"] == evaluation_set_sha256(records)
    assert evaluation["config"]["sha256"] == sha256_file(ROOT / "configs/evaluation_config.yaml")
    assert evaluation["metrics"] == recalculated["metrics"]
    assert evaluation["slices"] == recalculated["slices"]
    assert "samples_per_minute" in evaluation["metrics"]
    assert "pages_per_minute" not in evaluation["metrics"]

    raw_predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all("writer_id" not in record for record in raw_predictions)
    assert all(record["model_version"].endswith(f"@{MODEL_REVISION}") for record in raw_predictions)
