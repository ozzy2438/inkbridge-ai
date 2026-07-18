"""Run sealed-model, offline TrOCR inference on the SMHD research test split."""

import argparse
from pathlib import Path

import structlog

from src.evaluation.research_inference import produce_offline_research_prediction_artifact
from src.evaluation.research_trocr_backend import ResearchTrOCRPredictionBackend

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse local-only research inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--abstention-threshold", type=float, default=0.4)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Produce a private prediction artifact and aggregate-safe attestation."""
    args = parse_args()

    def backend_factory(
        model_version: str, model_artifact_sha256: str, device: str
    ) -> ResearchTrOCRPredictionBackend:
        return ResearchTrOCRPredictionBackend(
            model_dir=args.model_dir,
            model_version=model_version,
            model_artifact_sha256=model_artifact_sha256,
            device=device,
            batch_size=args.batch_size,
            beam_width=args.beam_width,
            max_length=args.max_length,
            confidence_threshold=args.confidence_threshold,
            abstention_threshold=args.abstention_threshold,
            use_fp16=device == "cuda",
        )

    result = produce_offline_research_prediction_artifact(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        model_version=args.model_version,
        expected_model_artifact_sha256=args.model_artifact_sha256,
        backend_factory=backend_factory,
        device=args.device,
        batch_size=args.batch_size,
        beam_width=args.beam_width,
        max_length=args.max_length,
        confidence_threshold=args.confidence_threshold,
        abstention_threshold=args.abstention_threshold,
        split=args.split,
        resume=not args.no_resume,
        repository_root=args.repository_root,
    )
    prediction = result["prediction_metadata"]
    logger.info(
        "offline_research_inference.complete",
        output=result["output_dir"],
        records=prediction["output"]["num_records"],
        predictions_sha256=prediction["output"]["sha256"],
        model_version=prediction["model"]["model_version"],
        network_access_allowed=False,
        protected_pilot_evidence=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
