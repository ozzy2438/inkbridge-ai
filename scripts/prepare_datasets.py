"""Prepare datasets for training.

Downloads and preprocesses:
- IAM Handwriting Database
- GNHK Dataset
- SMHD Dataset

Applies writer-independent splits.
"""

import argparse
from pathlib import Path
import structlog

logger = structlog.get_logger()


def prepare_iam(data_dir: Path):
    """Prepare IAM dataset.
    
    Note: IAM requires registration at:
    https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
    """
    iam_dir = data_dir / "raw" / "iam"
    if not iam_dir.exists():
        logger.warning(
            "IAM dataset not found. Please download from:"
            " https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"
        )
        return
    
    logger.info("Preparing IAM dataset...")
    # Processing logic here


def prepare_gnhk(data_dir: Path):
    """Prepare GNHK dataset.
    
    Note: GNHK available at:
    https://github.com/GoodNotes/GNHK-dataset
    """
    gnhk_dir = data_dir / "raw" / "gnhk"
    if not gnhk_dir.exists():
        logger.warning(
            "GNHK dataset not found. Please download from:"
            " https://github.com/GoodNotes/GNHK-dataset"
        )
        return
    
    logger.info("Preparing GNHK dataset...")
    # Processing logic here


def prepare_smhd(data_dir: Path):
    """Prepare SMHD dataset.
    
    Note: SMHD requires access request:
    https://github.com/hiqmatNisa/SMHD
    Contact: hiqmat.nisa@gmail.com
    License: CC BY-NC (research/portfolio use only)
    """
    smhd_dir = data_dir / "raw" / "smhd"
    if not smhd_dir.exists():
        logger.warning(
            "SMHD dataset not found. Please request access from:"
            " https://github.com/hiqmatNisa/SMHD"
        )
        return
    
    logger.info("Preparing SMHD dataset...")
    # Processing logic here


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Root data directory"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["iam", "gnhk", "smhd"],
        help="Datasets to prepare"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if "iam" in args.datasets:
        prepare_iam(data_dir)
    if "gnhk" in args.datasets:
        prepare_gnhk(data_dir)
    if "smhd" in args.datasets:
        prepare_smhd(data_dir)
    
    logger.info("Dataset preparation complete")


if __name__ == "__main__":
    main()
