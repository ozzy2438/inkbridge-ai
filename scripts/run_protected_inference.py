"""Produce reference-free predictions with a sealed local TrOCR artifact."""

import argparse
from pathlib import Path

import structlog

from src.evaluation.protected_inference import produce_protected_prediction_artifact
from src.evaluation.protected_trocr_backend import ProtectedTrOCRPredictionBackend

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse protected offline inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--abstention-threshold", type=float, default=0.4)
    return parser.parse_args()


def main() -> int:
    """Validate approvals and run local-files-only TrOCR inference."""
    args = parse_args()

    def backend_factory(
        model_version: str, model_artifact_sha256: str, authorized_device: str
    ) -> ProtectedTrOCRPredictionBackend:
        if args.device != authorized_device:
            raise ValueError("CLI device does not match the inference authorization")
        return ProtectedTrOCRPredictionBackend(
            model_dir=args.model_dir,
            model_version=model_version,
            model_artifact_sha256=model_artifact_sha256,
            device=args.device,
            batch_size=args.batch_size,
            beam_width=args.beam_width,
            max_length=args.max_length,
            confidence_threshold=args.confidence_threshold,
            abstention_threshold=args.abstention_threshold,
            use_fp16=args.device == "cuda",
        )

    result = produce_protected_prediction_artifact(
        contract_path=args.contract,
        dataset_dir=args.dataset_dir,
        manifest_dir=args.manifest_dir,
        authorization_path=args.authorization,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        backend_factory=backend_factory,
        batch_size=args.batch_size,
    )
    logger.info(
        "protected_inference.complete",
        status=result["status"],
        model_version=result["model_version"],
        evaluation_set_id=result["evaluation_set_id"],
        records=result["predictions"]["num_records"],
        predictions_sha256=result["predictions"]["sha256"],
        output=result["output_dir"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
