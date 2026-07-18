"""Unit tests for licensed, writer-isolated data manifests."""

import json
from pathlib import Path

import pytest

from src.data.manifest import (
    DatasetIdentity,
    build_manifest_records,
    sha256_file,
    write_split_manifest,
)


def _normalized_dataset(
    root: Path, writers: int = 6, samples_per_writer: int = 2, *, include_slices: bool = False
) -> None:
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    rows = ["filename,text,writer_id" + (",slices" if include_slices else "")]
    for writer_index in range(writers):
        for sample_index in range(samples_per_writer):
            filename = f"w{writer_index}-s{sample_index}.png"
            (images_dir / filename).write_bytes(f"image-{writer_index}-{sample_index}".encode())
            row = f"{filename},text {writer_index} {sample_index},writer-{writer_index}"
            if include_slices:
                row += ",faint_pencil|cursive|faint_pencil"
            rows.append(row)
    (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_manifest_records_prove_source_license_and_writer_isolation(tmp_path: Path) -> None:
    """The written artifact should be hashed, licensed, complete, and leakage-free."""
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "manifest"
    _normalized_dataset(dataset_dir)
    identity = DatasetIdentity(
        name="synthetic-contract",
        version="v1",
        license_id="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        sample_type="line",
    )

    records = build_manifest_records(dataset_dir, identity)
    metadata = write_split_manifest(
        records,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        identity=identity,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        min_samples_per_writer=2,
        seed=7,
    )

    manifest_path = output_dir / "manifest.jsonl"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_records = [json.loads(line) for line in manifest_text.splitlines()]
    split_writers = {
        split: {record["writer_id"] for record in manifest_records if record["split"] == split}
        for split in ("train", "validation", "test")
    }

    assert len(manifest_records) == 12
    assert metadata["manifest"]["sha256"] == sha256_file(manifest_path)
    assert metadata["split"]["writer_isolation_verified"] is True
    assert split_writers["train"].isdisjoint(split_writers["validation"])
    assert split_writers["train"].isdisjoint(split_writers["test"])
    assert split_writers["validation"].isdisjoint(split_writers["test"])
    assert "writer-0" not in manifest_text
    assert all(record["license_id"] == "CC0-1.0" for record in manifest_records)
    assert all(record["schema_version"] == 2 for record in manifest_records)
    assert all(record["sample_type"] == "line" for record in manifest_records)
    assert metadata["dataset"]["sample_type"] == "line"


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    """A label row must never escape the normalized images directory."""
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "labels.csv").write_text(
        "filename,text,writer_id\n../outside.png,text,writer-1\n", encoding="utf-8"
    )
    identity = DatasetIdentity(
        "dataset", "v1", "CC0-1.0", "https://example.com/license", "line"
    )

    with pytest.raises(ValueError, match="basename inside images"):
        build_manifest_records(dataset_dir, identity)


def test_manifest_requires_https_license_provenance(tmp_path: Path) -> None:
    """A free-form or insecure licence reference is insufficient provenance."""
    dataset_dir = tmp_path / "dataset"
    _normalized_dataset(dataset_dir, writers=3, samples_per_writer=1)
    identity = DatasetIdentity(
        "dataset", "v1", "unknown", "http://example.com/license", "line"
    )

    with pytest.raises(ValueError, match="absolute HTTPS"):
        build_manifest_records(dataset_dir, identity)


def test_manifest_normalizes_optional_evaluation_slices(tmp_path: Path) -> None:
    """Slice tags should be deterministic and ready for regression reporting."""
    dataset_dir = tmp_path / "dataset"
    _normalized_dataset(dataset_dir, writers=3, samples_per_writer=1, include_slices=True)
    identity = DatasetIdentity(
        "dataset", "v1", "CC0-1.0", "https://example.com/license", "line"
    )

    records = build_manifest_records(dataset_dir, identity)

    assert all(record["slices"] == ["cursive", "faint_pencil"] for record in records)
