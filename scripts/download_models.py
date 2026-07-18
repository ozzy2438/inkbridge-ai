"""Download model weights and checkpoints."""

import argparse
from pathlib import Path

import structlog

logger = structlog.get_logger()


def download_trocr_base(cache_dir: Path) -> None:
    """Download TrOCR base handwritten model."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_name = "microsoft/trocr-base-handwritten"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading TrOCR processor...")
    TrOCRProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))

    logger.info("Downloading TrOCR model...")
    VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=str(cache_dir))

    logger.info("TrOCR download complete")


def main():
    parser = argparse.ArgumentParser(description="Download InkBridge AI models")
    parser.add_argument(
        "--version",
        choices=["base"],
        default="base",
        help="Model version to download",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./model_cache", help="Directory to cache models"
    )
    args = parser.parse_args()

    download_trocr_base(Path(args.cache_dir))
    logger.info("All models downloaded successfully", version=args.version)


if __name__ == "__main__":
    main()
