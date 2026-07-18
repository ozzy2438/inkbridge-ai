"""Tests for the private, provenance-locked SMHD research adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.data.smhd_research_subset import prepare_smhd_research_subset


def _png_bytes(color: int) -> bytes:
    image = Image.new("L", (32, 12), color=color)
    output = BytesIO()
    image.save(output, format="PNG", pnginfo=None)
    return output.getvalue()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    repository = tmp_path / "repo"
    config_dir = repository / "configs" / "datasets"
    config_dir.mkdir(parents=True)

    private_root = tmp_path / "private"
    source_dir = private_root / "source"
    source_dir.mkdir(parents=True)
    if os.name == "posix":
        private_root.chmod(0o700)
        source_dir.chmod(0o700)

    main_rows: list[str] = []
    images: dict[str, bytes] = {}
    for writer_number in range(1, 5):
        writer = f"{writer_number:04d}"
        references = (
            "ordinary student sentence",
            "a cor#rected student sentence",
            "another ordinary line",
            "an in#serted line",
        )
        for sample_number, reference in enumerate(references):
            sample_id = f"{writer}-{sample_number:03d}"
            main_rows.append(f"{sample_id},180 {reference}")
            images[f"{writer}/{sample_id}.png"] = _png_bytes(210 + writer_number)

    index_payloads = {
        "SMHD.txt": ("\n".join(main_rows) + "\n").encode(),
        "Class_Notes_SMHD.txt": b"fixture-class-note\n",
        "SMHD-Cross-outsandInsertions.txt": b"fixture-cross-out\n",
    }
    archive_path = source_dir / "SMHD-lines.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in images.items():
            archive.writestr(name, content)
        for name, content in index_payloads.items():
            archive.writestr(name, content)
    if os.name == "posix":
        archive_path.chmod(0o600)

    archive_content = archive_path.read_bytes()
    spec: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "offline_student_handwriting_research_shadow_rehearsal_only",
        "dataset": {
            "article_id": 24419986,
            "doi": "10.25439/rmt.24419986.v1",
            "title": (
                "A Messy Handwriting Dataset with Student Crossouts and Corrections (Line-version)"
            ),
            "citation": "Locked test citation",
            "api_url": "https://api.figshare.com/v2/articles/24419986",
            "source_url": "https://research-repository.rmit.edu.au/articles/dataset/example/24419986",
            "license_id": "CC-BY-NC-4.0",
            "license_name": "CC BY-NC 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "archive": {
                "file_id": 42858925,
                "filename": "SMHD-lines.zip",
                "size": len(archive_content),
                "md5": hashlib.md5(archive_content, usedforsecurity=False).hexdigest(),
                "sha256": hashlib.sha256(archive_content).hexdigest(),
                "download_url": "https://ndownloader.figshare.com/files/42858925",
            },
        },
        "constraints": {
            "language": "eng",
            "sample_type": "line",
            "population": "mixed_high_school_and_university_students",
            "contains_student_data": True,
            "real_handwriting": True,
            "source_capture": "300_dpi_grayscale_scan",
            "commercial_use_allowed": False,
            "production_pilot_evidence": False,
            "offline_only": True,
            "repository_artifacts_allowed": False,
            "reference_status": "publisher_transcription_unreviewed",
            "reference_transform": "remove_hash_markers_and_collapse_whitespace",
            "gold_ready": False,
            "max_source_image_bytes": 10_000,
        },
        "archive_inventory": {
            "entries": len(images) + len(index_payloads),
            "png_entries": len(images),
            "index_files": {
                name: {
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "records": len(content.decode().splitlines()),
                    **(
                        {"usable_records": len(main_rows), "writers": 4}
                        if name == "SMHD.txt"
                        else {}
                    ),
                }
                for name, content in index_payloads.items()
            },
        },
        "selection": {
            "algorithm": "sha256_ranked_mixed_clean_and_marker_v1",
            "seed": "fixture-selection-v1",
            "index_filename": "SMHD.txt",
            "writers": 3,
            "samples_per_writer": 3,
            "required_clean_per_writer": 1,
            "required_marked_per_writer": 1,
        },
    }
    spec_path = config_dir / "smhd.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return repository, archive_path, private_root, spec


def test_prepares_private_deterministic_mixed_subset(tmp_path: Path) -> None:
    repository, archive, private_root, _ = _fixture(tmp_path)
    first = private_root / "normalized-first"
    second = private_root / "normalized-second"

    metadata = prepare_smhd_research_subset(
        repository / "configs/datasets/smhd.json",
        archive,
        first,
        repository_root=repository,
    )
    prepare_smhd_research_subset(
        repository / "configs/datasets/smhd.json",
        archive,
        second,
        repository_root=repository,
    )

    with (first / "labels.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert len({row["writer_id"] for row in rows}) == 3
    assert all(row["writer_id"].startswith("smhd-w-") for row in rows)
    assert all("#" not in row["text"] for row in rows)
    assert any("publisher_hash_marker" in row["slices"] for row in rows)
    assert any("publisher_no_hash_marker" in row["slices"] for row in rows)
    assert (first / "labels.csv").read_bytes() == (second / "labels.csv").read_bytes()
    assert "0001" not in (first / "labels.csv").read_text(encoding="utf-8")

    image_paths = sorted((first / "images").iterdir())
    assert len(image_paths) == 9
    assert all(path.read_bytes().startswith(b"P5\n32 12\n255\n") for path in image_paths)
    assert metadata["output"]["num_samples"] == 9
    assert metadata["output"]["num_writers"] == 3
    assert metadata["reference_quality"]["gold_ready"] is False
    assert metadata["evidence_boundary"]["production_pilot_evidence"] is False
    assert metadata["privacy"]["source_writer_ids_exported"] is False

    if os.name == "posix":
        assert stat.S_IMODE(first.stat().st_mode) == 0o700
        assert stat.S_IMODE((first / "images").stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600 for path in first.iterdir() if path.is_file()
        )
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in image_paths)


def test_rejects_archive_hash_drift_before_publishing(tmp_path: Path) -> None:
    repository, archive, private_root, spec = _fixture(tmp_path)
    spec["dataset"]["archive"]["sha256"] = "0" * 64
    spec_path = repository / "configs/datasets/smhd.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = private_root / "normalized"

    with pytest.raises(ValueError, match="SHA-256 drift"):
        prepare_smhd_research_subset(
            spec_path,
            archive,
            output,
            repository_root=repository,
        )

    assert not output.exists()


def test_rejects_source_archive_inside_repository(tmp_path: Path) -> None:
    repository, archive, private_root, _ = _fixture(tmp_path)
    repository_archive = repository / "SMHD-lines.zip"
    repository_archive.write_bytes(archive.read_bytes())
    if os.name == "posix":
        repository_archive.chmod(0o600)

    with pytest.raises(ValueError, match="outside the repository"):
        prepare_smhd_research_subset(
            repository / "configs/datasets/smhd.json",
            repository_archive,
            private_root / "normalized",
            repository_root=repository,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode enforcement")
def test_rejects_group_readable_source_archive(tmp_path: Path) -> None:
    repository, archive, private_root, _ = _fixture(tmp_path)
    archive.chmod(0o640)

    with pytest.raises(PermissionError, match="group or world"):
        prepare_smhd_research_subset(
            repository / "configs/datasets/smhd.json",
            archive,
            private_root / "normalized",
            repository_root=repository,
        )


def test_rejects_gold_ready_claim(tmp_path: Path) -> None:
    repository, archive, private_root, spec = _fixture(tmp_path)
    spec["constraints"]["gold_ready"] = True
    spec_path = repository / "configs/datasets/smhd.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="gold_ready"):
        prepare_smhd_research_subset(
            spec_path,
            archive,
            private_root / "normalized",
            repository_root=repository,
        )
