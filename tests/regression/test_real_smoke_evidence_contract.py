"""Regression lock for the captured CSAFE real-handwriting TrOCR smoke evidence."""

import json
from pathlib import Path

from src.evaluation.runner import (
    evaluate_records,
    evaluation_set_sha256,
    load_evaluation_records,
    sha256_file,
)

ROOT = Path(__file__).parents[2]
EVIDENCE = ROOT / "artifacts" / "smoke" / "csafe-real-trocr-base-v1"
EVALUATION = EVIDENCE / (
    "eval_microsoft-trocr-base-handwritten-"
    "eaacaf452b06415df8f10bb6fad3a4c11e609406_"
    "csafe-real-handwriting-smoke-test-v1.json"
)
SPEC = ROOT / "configs" / "datasets" / "csafe_real_handwriting_smoke.json"
MODEL_REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"


def test_real_smoke_evidence_hashes_and_metrics_remain_traceable() -> None:
    """Published real predictions must continue to reproduce their captured report."""
    predictions_path = EVIDENCE / "predictions.jsonl"
    metadata = json.loads((EVIDENCE / "predictions.meta.json").read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    records = load_evaluation_records(predictions_path)
    recalculated = evaluate_records(records)

    assert len(records) == 5
    assert metadata["input"]["dataset"] == {
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "name": "csafe-real-handwriting-smoke",
        "sample_type": "line",
        "version": "figshare-10062203-v2-selection-v1",
    }
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
    assert spec["constraints"]["real_handwriting"] is True
    assert spec["constraints"]["contains_student_data"] is False
    assert spec["constraints"]["gold_ready"] is False

    raw_predictions = [
        json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all("writer_id" not in record for record in raw_predictions)
    assert all("real_handwriting" in record["slices"] for record in raw_predictions)
    assert all(record["model_version"].endswith(f"@{MODEL_REVISION}") for record in raw_predictions)
