"""Tests for the provenance-locked CSAFE real-handwriting smoke adapter."""

from __future__ import annotations

import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zlib import crc32

import pytest
from PIL import Image

from src.data.csafe_smoke_subset import prepare_csafe_smoke_subset

PROMPT = "The early bird may get the worm, but the second mouse gets the cheese."


def _png_bytes(color: int) -> bytes:
    image = Image.new("L", (40, 20), color=color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _pgm_hash(source: bytes, bbox: list[int]) -> str:
    with Image.open(BytesIO(source)) as image:
        crop = image.convert("L").crop(tuple(bbox))
    width, height = crop.size
    content = f"P5\n{width} {height}\n255\n".encode("ascii") + crop.tobytes()
    return hashlib.sha256(content).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, bytes], dict[str, Any]]:
    pages = {
        "Session 1/w0001_s01_pPHR_r01.png": _png_bytes(245),
        "Session 1/w0002_s01_pPHR_r01.png": _png_bytes(225),
    }
    selections = [
        {
            "writer_id": "w0001",
            "archive_path": "Session 1/w0001_s01_pPHR_r01.png",
            "width": 40,
            "height": 20,
            "lines": [
                {
                    "bbox": [0, 0, 40, 10],
                    "reference": "The early bird may get the worm, but",
                },
                {
                    "bbox": [0, 10, 40, 20],
                    "reference": "the second mouse gets the cheese.",
                },
            ],
        },
        {
            "writer_id": "w0002",
            "archive_path": "Session 1/w0002_s01_pPHR_r01.png",
            "width": 40,
            "height": 20,
            "lines": [{"bbox": [0, 0, 40, 20], "reference": PROMPT}],
        },
    ]
    for selection in selections:
        source = pages[selection["archive_path"]]
        selection["archive_size"] = len(source)
        selection["archive_crc32"] = crc32(source)
        selection["source_sha256"] = hashlib.sha256(source).hexdigest()
        for line in selection["lines"]:
            line["output_sha256"] = _pgm_hash(source, line["bbox"])

    spec = {
        "schema_version": 1,
        "purpose": "real_handwriting_pipeline_smoke_only",
        "dataset": {
            "article_id": 10062203,
            "doi": "10.25380/iastate.10062203.v2",
            "title": "CSAFE Handwriting Database",
            "citation": "Example locked citation",
            "api_url": "https://api.figshare.com/v2/articles/10062203",
            "source_url": "https://iastate.figshare.com/articles/dataset/example/10062203/2",
            "license_id": "CC-BY-4.0",
            "license_name": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "readme": {
                "file_id": 11,
                "filename": "readme.pdf",
                "size": 100,
                "md5": "1" * 32,
            },
            "archive": {
                "file_id": 22,
                "filename": "Session 1.zip",
                "size": 1000,
                "md5": "2" * 32,
                "download_url": "https://ndownloader.figshare.com/files/22",
            },
        },
        "constraints": {
            "language": "eng",
            "sample_type": "line",
            "population": "adults",
            "contains_student_data": False,
            "real_handwriting": True,
            "prompt_id": "PHR",
            "prompt": PROMPT,
            "reference_status": "prompt_derived_single_pass",
            "gold_ready": False,
            "source_pages_exported": False,
        },
        "selection": selections,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    article = {
        "id": spec["dataset"]["article_id"],
        "doi": spec["dataset"]["doi"],
        "title": spec["dataset"]["title"],
        "citation": spec["dataset"]["citation"],
        "license": {
            "name": spec["dataset"]["license_name"],
            "url": spec["dataset"]["license_url"],
        },
        "files": [
            {
                "id": spec["dataset"]["readme"]["file_id"],
                "name": spec["dataset"]["readme"]["filename"],
                "size": spec["dataset"]["readme"]["size"],
                "supplied_md5": spec["dataset"]["readme"]["md5"],
                "computed_md5": spec["dataset"]["readme"]["md5"],
            },
            {
                "id": spec["dataset"]["archive"]["file_id"],
                "name": spec["dataset"]["archive"]["filename"],
                "size": spec["dataset"]["archive"]["size"],
                "supplied_md5": spec["dataset"]["archive"]["md5"],
                "computed_md5": spec["dataset"]["archive"]["md5"],
                "download_url": spec["dataset"]["archive"]["download_url"],
            },
        ],
    }
    return spec_path, pages, article


class _FakeArchive:
    def __init__(self, pages: dict[str, bytes]) -> None:
        self.pages = pages

    def __enter__(self) -> _FakeArchive:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def getinfo(self, name: str) -> SimpleNamespace:
        content = self.pages[name]
        return SimpleNamespace(file_size=len(content), CRC=crc32(content))

    def read(self, name: str) -> bytes:
        return self.pages[name]


def test_prepare_csafe_smoke_subset_writes_real_provenance(tmp_path: Path) -> None:
    spec_path, pages, article = _fixture(tmp_path)
    output_dir = tmp_path / "normalized"

    metadata = prepare_csafe_smoke_subset(
        spec_path,
        output_dir,
        fetch_json=lambda _: article,
        archive_factory=lambda _: _FakeArchive(pages),
    )

    with (output_dir / "labels.csv").open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    assert [row["writer_id"] for row in labels] == ["w0001", "w0001", "w0002"]
    assert labels[0]["filename"] == "csafe-w0001-s01-phr-r01-line-1.pgm"
    assert labels[0]["text"] == "The early bird may get the worm, but"
    assert labels[0]["slices"] == (
        "adult|csafe|page_lines_2|prompt_phr|real_handwriting|"
        "reference_prompt_derived|repetition_1|session_1"
    )
    assert metadata["privacy"] == {
        "synthetic": False,
        "real_handwriting": True,
        "population": "adults",
        "contains_student_data": False,
        "writer_identity": "source_participant_id",
        "source_pages_exported": False,
    }
    assert metadata["reference_quality"]["gold_ready"] is False
    assert metadata["output"]["num_samples"] == 3
    assert metadata["output"]["num_writers"] == 2
    assert not any(output_dir.rglob("*.png"))
    persisted = (output_dir / "source.meta.json").read_text(encoding="utf-8")
    assert "ndownloader.figshare.com" not in persisted
    assert "not a human-verified gold benchmark" in (output_dir / "ATTRIBUTION.md").read_text(
        encoding="utf-8"
    )


def test_prepare_csafe_smoke_subset_stops_before_archive_on_license_drift(
    tmp_path: Path,
) -> None:
    spec_path, pages, article = _fixture(tmp_path)
    article["license"] = {"name": "All Rights Reserved", "url": "https://example.com/"}
    archive_calls: list[str] = []

    def archive_factory(url: str) -> _FakeArchive:
        archive_calls.append(url)
        return _FakeArchive(pages)

    with pytest.raises(ValueError, match="license drift"):
        prepare_csafe_smoke_subset(
            spec_path,
            tmp_path / "normalized",
            fetch_json=lambda _: article,
            archive_factory=archive_factory,
        )

    assert archive_calls == []
    assert not (tmp_path / "normalized").exists()


def test_prepare_csafe_smoke_subset_does_not_publish_on_crop_hash_drift(
    tmp_path: Path,
) -> None:
    spec_path, pages, article = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["selection"][0]["lines"][0]["output_sha256"] = "f" * 64
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output_dir = tmp_path / "normalized"

    with pytest.raises(ValueError, match="output hash drift"):
        prepare_csafe_smoke_subset(
            spec_path,
            output_dir,
            fetch_json=lambda _: article,
            archive_factory=lambda _: _FakeArchive(pages),
        )

    assert not output_dir.exists()


def test_prepare_csafe_smoke_subset_rejects_prompt_split_drift(tmp_path: Path) -> None:
    spec_path, pages, article = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["selection"][0]["lines"][1]["reference"] = "the second mouse gets cheese."
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="do not reconstruct"):
        prepare_csafe_smoke_subset(
            spec_path,
            tmp_path / "normalized",
            fetch_json=lambda _: article,
            archive_factory=lambda _: _FakeArchive(pages),
        )
