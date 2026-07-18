"""Produce a resumable, evaluator-ready TrOCR baseline prediction artifact."""

import argparse
from pathlib import Path

import structlog

from src.evaluation.prediction_producer import produce_prediction_artifact
from src.evaluation.trocr_backend import TrOCRPredictionBackend

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse baseline inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--abstention-threshold", type=float, default=0.4)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Resolve the model, validate data, and produce prediction evidence."""
    args = parse_args()
    backend = TrOCRPredictionBackend(
        model_id=args.model_id,
        requested_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        beam_width=args.beam_width,
        max_length=args.max_length,
        confidence_threshold=args.confidence_threshold,
        abstention_threshold=args.abstention_threshold,
        use_fp16=args.device == "cuda",
    )
    metadata = produce_prediction_artifact(
        args.dataset_dir,
        args.output_dir,
        backend,
        split=args.split,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    logger.info(
        "baseline_prediction.complete",
        records=metadata["output"]["num_records"],
        sha256=metadata["output"]["sha256"],
        model=metadata["model"]["model_version"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
