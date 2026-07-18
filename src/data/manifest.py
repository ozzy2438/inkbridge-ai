"""Licensed dataset manifest generation with file and writer provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from src.data.writer_split import create_writer_split


@dataclass(frozen=True)
class DatasetIdentity:
    """Versioned source and license identity attached to every manifest record."""

    name: str
    version: str
    license_id: str
    license_url: str
    sample_type: Literal["line", "page"]

    def validate(self) -> None:
        """Reject incomplete provenance before any artifact is written."""
        for field_name in ("name", "version", "license_id"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"Dataset {field_name} must be non-empty")
        parsed_url = urlparse(self.license_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("license_url must be an absolute HTTPS URL")
        if self.sample_type not in {"line", "page"}:
            raise ValueError("sample_type must be either 'line' or 'page'")


def build_manifest_records(
    dataset_dir: str | Path, identity: DatasetIdentity
) -> list[dict[str, Any]]:
    """Read normalized labels.csv and produce validated, pseudonymous records."""
    identity.validate()
    root = Path(dataset_dir)
    labels_path = root / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Dataset labels file does not exist: {labels_path}")

    records: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()
    with labels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"filename", "text", "writer_id"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("labels.csv must contain filename, text, and writer_id columns")

        for row_number, row in enumerate(reader, start=2):
            filename = _required_csv_value(row, "filename", row_number)
            reference = _required_csv_value(row, "text", row_number)
            raw_writer_id = _required_csv_value(row, "writer_id", row_number)
            slices = _parse_slices(row.get("slices"))
            relative_path = _safe_image_path(filename, row_number)
            if relative_path.as_posix() in seen_filenames:
                raise ValueError(f"labels.csv row {row_number}: duplicate filename '{filename}'")
            seen_filenames.add(relative_path.as_posix())

            image_path = root / relative_path
            _require_contained_file(root, image_path, row_number)
            source_sha256 = sha256_file(image_path)
            writer_id = _stable_digest(identity.name, identity.version, raw_writer_id)
            sample_id = _stable_digest(identity.name, identity.version, relative_path.as_posix())
            records.append(
                {
                    "schema_version": 2,
                    "sample_id": sample_id,
                    "source_sha256": source_sha256,
                    "image_path": relative_path.as_posix(),
                    "reference": reference,
                    "writer_id": writer_id,
                    "dataset": identity.name,
                    "dataset_version": identity.version,
                    "license_id": identity.license_id,
                    "license_url": identity.license_url,
                    "sample_type": identity.sample_type,
                    "slices": slices,
                }
            )

    if not records:
        raise ValueError("labels.csv contains no dataset records")
    return records


def write_split_manifest(
    records: list[dict[str, Any]],
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    identity: DatasetIdentity,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_samples_per_writer: int = 3,
) -> dict[str, Any]:
    """Assign writer-isolated splits and atomically write JSONL plus metadata."""
    identity.validate()
    train, validation, test = create_writer_split(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        min_samples_per_writer=min_samples_per_writer,
    )
    split_records = {"train": train, "validation": validation, "test": test}
    split_by_sample = {
        str(record["sample_id"]): split_name
        for split_name, split in split_records.items()
        for record in split
    }
    if len(split_by_sample) != len(records):
        raise RuntimeError("Writer split did not assign every manifest record exactly once")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.jsonl"
    temporary_path = destination / "manifest.jsonl.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda value: str(value["sample_id"])):
            output_record = {**record, "split": split_by_sample[str(record["sample_id"])]}
            json.dump(output_record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    temporary_path.replace(manifest_path)

    split_sample_counts = {split_name: len(split) for split_name, split in split_records.items()}
    split_writer_counts = {
        split_name: len({str(record["writer_id"]) for record in split})
        for split_name, split in split_records.items()
    }
    metadata = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": identity.name,
            "version": identity.version,
            "license_id": identity.license_id,
            "license_url": identity.license_url,
            "sample_type": identity.sample_type,
        },
        "source": {
            "labels_filename": "labels.csv",
            "labels_sha256": sha256_file(Path(dataset_dir) / "labels.csv"),
        },
        "manifest": {
            "filename": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "num_samples": len(records),
            "duplicate_source_hashes": _duplicate_hash_count(records),
        },
        "split": {
            "seed": seed,
            "ratios": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
            "min_samples_per_writer": min_samples_per_writer,
            "sample_counts": split_sample_counts,
            "writer_counts": split_writer_counts,
            "writer_isolation_verified": True,
        },
        "writer_id_policy": "sha256(dataset_name, dataset_version, source_writer_id)",
    }
    metadata_path = destination / "manifest.meta.json"
    metadata_temporary_path = destination / "manifest.meta.json.tmp"
    with metadata_temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    metadata_temporary_path.replace(metadata_path)
    return metadata


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a source or manifest file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_csv_value(row: dict[str, str | None], key: str, row_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"labels.csv row {row_number}: '{key}' must be non-empty")
    return value.strip()


def _safe_image_path(filename: str, row_number: int) -> Path:
    value = Path(filename)
    if value.is_absolute() or ".." in value.parts or value.name != filename:
        raise ValueError(f"labels.csv row {row_number}: filename must be a basename inside images/")
    return Path("images") / value


def _parse_slices(raw_value: str | None) -> list[str]:
    """Normalize an optional pipe-delimited slice column."""
    if raw_value is None or not raw_value.strip():
        return []
    slices = sorted({value.strip() for value in raw_value.split("|") if value.strip()})
    if not slices:
        return []
    return slices


def _require_contained_file(root: Path, image_path: Path, row_number: int) -> None:
    try:
        image_path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"labels.csv row {row_number}: image is missing or escapes the dataset directory"
        ) from exc
    if not image_path.is_file():
        raise ValueError(f"labels.csv row {row_number}: image path is not a file")


def _stable_digest(*parts: str) -> str:
    value = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _duplicate_hash_count(records: list[dict[str, Any]]) -> int:
    counts = Counter(str(record["source_sha256"]) for record in records)
    return sum(count - 1 for count in counts.values() if count > 1)
