"""Tests for the immutable Hugging Face synthetic smoke subset adapter."""

from __future__ import annotations

import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from PIL import Image

from src.data.hf_smoke_subset import prepare_hf_smoke_subset


def _jpeg_bytes(width: int, height: int, color: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color=color).save(output, format="JPEG")
    return output.getvalue()


def _write_spec(path: Path, rows: list[dict[str, object]]) -> None:
    selection = [
        {
            "row_idx": row["row_idx"],
            "writer_id": row["writer_id"],
            "reference": row["text"],
            "width": row["image"]["width"],  # type: ignore[index]
            "height": row["image"]["height"],  # type: ignore[index]
            "image_sha256": hashlib.sha256(row["image_bytes"]).hexdigest(),  # type: ignore[arg-type]
        }
        for row in rows
    ]
    spec = {
        "schema_version": 1,
        "purpose": "synthetic_pipeline_smoke_only",
        "dataset": {
            "id": "owner/dataset",
            "revision": "a" * 40,
            "config": "default",
            "split": "train",
            "license_id": "cc-by-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_url": "https://huggingface.co/datasets/owner/dataset",
            "attribution": "Synthetic Dataset by Example Author",
            "allowed_sources": ["faker-name", "faker-date"],
        },
        "constraints": {"language": "eng", "max_height": 200, "sample_type": "line"},
        "selection": selection,
    }
    path.write_text(json.dumps(spec), encoding="utf-8")


def _fixture_rows() -> list[dict[str, object]]:
    first_image = _jpeg_bytes(120, 40, "white")
    second_image = _jpeg_bytes(140, 50, "lightgray")
    return [
        {
            "row_idx": 3,
            "writer_id": 10,
            "text": "Jane Example",
            "language": "eng",
            "source": "faker-name",
            "neatness": 90,
            "color": "black",
            "augmented_noise": False,
            "image": {
                "src": "https://datasets-server.huggingface.co/assets/3.jpg",
                "width": 120,
                "height": 40,
            },
            "image_bytes": first_image,
        },
        {
            "row_idx": 8,
            "writer_id": 11,
            "text": "1 January 2000",
            "language": "eng",
            "source": "faker-date",
            "neatness": 68,
            "color": "blue",
            "augmented_noise": True,
            "image": {
                "src": "https://datasets-server.huggingface.co/assets/8.jpg",
                "width": 140,
                "height": 50,
            },
            "image_bytes": second_image,
        },
    ]


def _fake_fetchers(
    rows: list[dict[str, object]], *, revision: str = "a" * 40
) -> tuple[object, object, list[str], list[str]]:
    json_calls: list[str] = []
    byte_calls: list[str] = []

    def fetch_json(url: str) -> dict[str, object]:
        json_calls.append(url)
        if url.startswith("https://huggingface.co/api/datasets/"):
            return {"sha": revision, "cardData": {"license": "cc-by-4.0"}}
        offset = int(parse_qs(urlparse(url).query)["offset"][0])
        source_row = next(row for row in rows if row["row_idx"] == offset)
        public_row = {key: value for key, value in source_row.items() if key != "image_bytes"}
        return {"rows": [{"row_idx": offset, "row": public_row}]}

    def fetch_bytes(url: str) -> bytes:
        byte_calls.append(url)
        source_row = next(row for row in rows if row["image"]["src"] == url)  # type: ignore[index]
        return source_row["image_bytes"]  # type: ignore[return-value]

    return fetch_json, fetch_bytes, json_calls, byte_calls


def test_prepare_hf_smoke_subset_writes_normalized_provenance(tmp_path: Path) -> None:
    rows = _fixture_rows()
    spec_path = tmp_path / "spec.json"
    output_dir = tmp_path / "normalized"
    _write_spec(spec_path, rows)
    fetch_json, fetch_bytes, _, byte_calls = _fake_fetchers(rows)

    metadata = prepare_hf_smoke_subset(
        spec_path,
        output_dir,
        fetch_json=fetch_json,  # type: ignore[arg-type]
        fetch_bytes=fetch_bytes,  # type: ignore[arg-type]
    )

    with (output_dir / "labels.csv").open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    assert labels == [
        {
            "filename": "hf-train-3.jpg",
            "text": "Jane Example",
            "writer_id": "style-10",
            "slices": "ink_color_black|neatness_high|no_noise|source_faker_name|synthetic",
        },
        {
            "filename": "hf-train-8.jpg",
            "text": "1 January 2000",
            "writer_id": "style-11",
            "slices": "ink_color_blue|neatness_low|noise_augmented|source_faker_date|synthetic",
        },
    ]
    assert (output_dir / "images" / "hf-train-3.jpg").read_bytes() == rows[0]["image_bytes"]
    assert metadata["dataset"]["revision"] == "a" * 40
    assert metadata["privacy"] == {
        "synthetic": True,
        "contains_real_student_data": False,
        "writer_identity": "synthetic_style_id",
    }
    assert metadata["output"]["num_samples"] == 2
    persisted_metadata = (output_dir / "source.meta.json").read_text(encoding="utf-8")
    assert "datasets-server.huggingface.co/assets" not in persisted_metadata
    assert len(byte_calls) == 2
    assert "selected and normalized derivative" in (output_dir / "ATTRIBUTION.md").read_text(
        encoding="utf-8"
    )


def test_prepare_hf_smoke_subset_stops_before_rows_on_revision_drift(tmp_path: Path) -> None:
    rows = _fixture_rows()
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, rows)
    fetch_json, fetch_bytes, json_calls, byte_calls = _fake_fetchers(rows, revision="b" * 40)

    with pytest.raises(ValueError, match="Dataset revision drift"):
        prepare_hf_smoke_subset(
            spec_path,
            tmp_path / "normalized",
            fetch_json=fetch_json,  # type: ignore[arg-type]
            fetch_bytes=fetch_bytes,  # type: ignore[arg-type]
        )

    assert len(json_calls) == 1
    assert byte_calls == []
    assert not (tmp_path / "normalized").exists()


def test_prepare_hf_smoke_subset_does_not_publish_index_on_hash_drift(
    tmp_path: Path,
) -> None:
    rows = _fixture_rows()
    spec_path = tmp_path / "spec.json"
    output_dir = tmp_path / "normalized"
    _write_spec(spec_path, rows)
    fetch_json, _, _, _ = _fake_fetchers(rows)

    def corrupt_bytes(_: str) -> bytes:
        return b"not-the-selected-image"

    with pytest.raises(ValueError, match="image hash drift"):
        prepare_hf_smoke_subset(
            spec_path,
            output_dir,
            fetch_json=fetch_json,  # type: ignore[arg-type]
            fetch_bytes=corrupt_bytes,
        )

    assert not (output_dir / "labels.csv").exists()
    assert not (output_dir / "source.meta.json").exists()
