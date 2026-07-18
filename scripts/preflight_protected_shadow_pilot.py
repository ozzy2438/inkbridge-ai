"""Check protected shadow-pilot readiness without loading a model or running inference."""

import argparse
import json
from pathlib import Path

from src.evaluation.protected_inference import preflight_protected_shadow_pilot


def parse_args() -> argparse.Namespace:
    """Parse protected shadow-pilot preflight arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Print aggregate readiness and fail when external evidence is incomplete."""
    args = parse_args()
    result = preflight_protected_shadow_pilot(
        contract_path=args.contract,
        dataset_dir=args.dataset_dir,
        manifest_dir=args.manifest_dir,
        authorization_path=args.authorization,
        model_dir=args.model_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
