"""Regression lock for aggregate-only SMHD offline research evidence."""

import json
from pathlib import Path

from src.evaluation.runner import sha256_file

ROOT = Path(__file__).parents[2]
EVIDENCE = ROOT / "artifacts" / "research" / "smhd-trocr-base-v1"
RESULT = EVIDENCE / "result.json"
SPEC = ROOT / "configs" / "datasets" / "smhd_research_rehearsal.json"
CONFIG = ROOT / "configs" / "evaluation_config.yaml"


def test_smhd_research_evidence_remains_aggregate_only_and_traceable() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    serialized = json.dumps(result, sort_keys=True)

    assert result["status"] == "captured"
    assert result["evidence_scope"] == "noncommercial_research_rehearsal_only"
    assert result["dataset"]["license_id"] == "CC-BY-NC-4.0"
    assert result["dataset"]["num_samples"] == 6
    assert result["dataset"]["num_writers"] == 2
    assert result["dataset"]["reference_status"] == "publisher_transcription_unreviewed"
    assert result["model"]["model_artifact_sha256"] == (
        "8215ca7b0e4536affbc01a5718abae5d572cf011b0e709953e57eb0a1be92149"
    )
    assert result["model"]["read_only"] is True
    assert result["model"]["processor_use_fast"] is False
    assert result["execution"]["network_access_allowed"] is False
    assert result["execution"]["github_actions_allowed"] is False
    assert result["execution"]["sample_level_artifacts_committed"] is False
    assert result["execution"]["predictions_sha256"] == (
        "196250d38141dbaf43cf2171806c0f23eacf7e0e7419caf524d0f495ddf9a472"
    )
    assert result["evaluation"]["config_sha256"] == sha256_file(CONFIG)
    assert result["evaluation"]["metrics"]["cer"] == 0.13
    assert result["evaluation"]["metrics"]["valid_output_rate"] == 1.0
    assert (
        result["evaluation"]["slices"]["publisher_hash_marker"]["cer"]
        > result["evaluation"]["slices"]["publisher_no_hash_marker"]["cer"]
    )
    diagnostics = result["diagnostics"]
    assert diagnostics["validation"]["num_samples"] == 9
    assert diagnostics["validation"]["num_writers"] == 3
    assert diagnostics["untouched_test_holdout"]["num_samples"] == 6
    assert diagnostics["failure_atlas"]["num_failure_records"] == 14
    assert diagnostics["failure_atlas"]["sample_level_artifacts_committed"] is False
    assert diagnostics["calibration_gate"]["status"] == "blocked"
    assert diagnostics["calibration_gate"]["validation_policy"]["status"] == (
        "no_eligible_threshold"
    )
    assert diagnostics["calibration_gate"]["test_threshold_applied"] is False
    assert diagnostics["calibration_gate"]["automatic_acceptance_allowed"] is False
    assert diagnostics["calibration_gate"]["human_review_required"] is True
    assert all(value is False for value in result["claims"].values())
    assert spec["dataset"]["archive"]["sha256"] == (
        "ef3c7852f61e32f4242f902da73ba1a66847bcf2c0c93ee4ed93c0299d12192a"
    )
    assert spec["constraints"]["commercial_use_allowed"] is False
    assert spec["constraints"]["gold_ready"] is False

    for forbidden_key in ('"sample_id"', '"writer_id"', '"reference"', '"prediction"'):
        assert forbidden_key not in serialized
