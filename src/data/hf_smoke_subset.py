"""Prepare a hash-pinned public Hugging Face subset for pipeline smoke tests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from PIL import Image

JsonFetcher = Callable[[str], Mapping[str, Any]]
BytesFetcher = Callable[[str], bytes]

_HF_DATASET_API = "https://huggingface.co/api/datasets/"
_HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
_ALLOWED_IMAGE_HOST = "datasets-server.huggingface.co"


def prepare_hf_smoke_subset(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    fetch_json: JsonFetcher | None = None,
    fetch_bytes: BytesFetcher | None = None,
) -> dict[str, Any]:
    """Download and normalize an explicitly selected, immutable public subset."""
    spec_file = Path(spec_path)
    spec = _load_spec(spec_file)
    get_json = fetch_json or _fetch_json
    get_bytes = fetch_bytes or _fetch_bytes

    dataset = spec["dataset"]
    dataset_id = _required_string(dataset, "id", "dataset")
    revision = _required_sha(dataset, "revision", "dataset")
    config = _required_string(dataset, "config", "dataset")
    split = _required_string(dataset, "split", "dataset")
    expected_license = _required_string(dataset, "license_id", "dataset")
    allowed_sources = _required_string_list(dataset, "allowed_sources", "dataset")

    repo_metadata = get_json(f"{_HF_DATASET_API}{dataset_id}")
    current_revision = repo_metadata.get("sha")
    if current_revision != revision:
        raise ValueError(
            f"Dataset revision drift: expected {revision}, received {current_revision!r}"
        )
    card_license = _card_license(repo_metadata)
    if card_license.casefold() != expected_license.casefold():
        raise ValueError(
            f"Dataset license drift: expected {expected_license}, received {card_license!r}"
        )

    constraints = spec["constraints"]
    language = _required_string(constraints, "language", "constraints")
    max_height = _required_positive_int(constraints, "max_height", "constraints")
    if _required_string(constraints, "sample_type", "constraints") != "line":
        raise ValueError("constraints.sample_type must be 'line'")

    normalized_rows: list[dict[str, Any]] = []
    image_payloads: list[tuple[str, bytes, str]] = []
    seen_rows: set[int] = set()
    seen_filenames: set[str] = set()

    for position, selection in enumerate(spec["selection"]):
        context = f"selection[{position}]"
        row_idx = _required_nonnegative_int(selection, "row_idx", context)
        if row_idx in seen_rows:
            raise ValueError(f"{context}.row_idx duplicates {row_idx}")
        seen_rows.add(row_idx)

        payload = get_json(
            f"{_HF_ROWS_API}?"
            + urlencode(
                {
                    "dataset": dataset_id,
                    "config": config,
                    "split": split,
                    "offset": row_idx,
                    "length": 1,
                }
            )
        )
        row = _extract_row(payload, row_idx, context)
        _validate_row(
            row,
            selection,
            language=language,
            allowed_sources=allowed_sources,
            max_height=max_height,
            context=context,
        )

        image = row["image"]
        image_url = _validated_image_url(image, context)
        image_bytes = get_bytes(image_url)
        expected_hash = _required_sha(selection, "image_sha256", context)
        actual_hash = _sha256_bytes(image_bytes)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{context} image hash drift: expected {expected_hash}, received {actual_hash}"
            )
        width = _required_positive_int(selection, "width", context)
        height = _required_positive_int(selection, "height", context)
        _validate_jpeg(image_bytes, width=width, height=height, context=context)

        filename = f"hf-train-{row_idx}.jpg"
        if filename in seen_filenames:
            raise ValueError(f"Generated duplicate filename: {filename}")
        seen_filenames.add(filename)
        source = _required_string(row, "source", context)
        normalized_rows.append(
            {
                "filename": filename,
                "text": _required_string(row, "text", context),
                "writer_id": f"style-{_required_writer_id(row, context)}",
                "slices": "|".join(_row_slices(row, source)),
            }
        )
        image_payloads.append((filename, image_bytes, actual_hash))

    destination = Path(output_dir)
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for filename, image_bytes, image_hash in image_payloads:
        image_path = images_dir / filename
        if image_path.exists() and _sha256_file(image_path) == image_hash:
            continue
        _atomic_write_bytes(image_path, image_bytes)

    labels_path = destination / "labels.csv"
    _atomic_write_csv(labels_path, normalized_rows)
    attribution_path = destination / "ATTRIBUTION.md"
    _atomic_write_text(attribution_path, _attribution_text(spec))

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": _required_string(spec, "purpose", "spec"),
        "dataset": {
            "id": dataset_id,
            "revision": revision,
            "config": config,
            "split": split,
            "license_id": expected_license,
            "license_url": _required_https_url(dataset, "license_url", "dataset"),
            "source_url": _required_https_url(dataset, "source_url", "dataset"),
            "attribution": _required_string(dataset, "attribution", "dataset"),
        },
        "privacy": {
            "synthetic": True,
            "contains_real_student_data": False,
            "writer_identity": "synthetic_style_id",
        },
        "source_spec": {
            "filename": spec_file.name,
            "sha256": _sha256_file(spec_file),
        },
        "output": {
            "labels_filename": labels_path.name,
            "labels_sha256": _sha256_file(labels_path),
            "attribution_filename": attribution_path.name,
            "attribution_sha256": _sha256_file(attribution_path),
            "num_samples": len(normalized_rows),
            "images": [
                {"filename": filename, "sha256": image_hash}
                for filename, _, image_hash in image_payloads
            ],
        },
    }
    _atomic_write_text(
        destination / "source.meta.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return metadata


def _load_spec(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Smoke subset spec does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Smoke subset spec is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Smoke subset spec must be a JSON object")
    if raw.get("schema_version") != 1:
        raise ValueError("Smoke subset spec schema_version must be 1")
    for key in ("dataset", "constraints"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"Smoke subset spec '{key}' must be an object")
    selection = raw.get("selection")
    if not isinstance(selection, list) or not selection:
        raise ValueError("Smoke subset spec 'selection' must be a non-empty list")
    if any(not isinstance(value, dict) for value in selection):
        raise ValueError("Every smoke subset selection must be an object")
    return raw


def _extract_row(payload: Mapping[str, Any], row_idx: int, context: str) -> Mapping[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{context} Dataset Viewer response must contain exactly one row")
    wrapper = rows[0]
    if wrapper.get("row_idx") != row_idx:
        raise ValueError(
            f"{context} row index drift: expected {row_idx}, received {wrapper.get('row_idx')!r}"
        )
    row = wrapper.get("row")
    if not isinstance(row, dict):
        raise ValueError(f"{context} Dataset Viewer row payload is missing")
    return row


def _validate_row(
    row: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    language: str,
    allowed_sources: list[str],
    max_height: int,
    context: str,
) -> None:
    expected_reference = _required_string(selection, "reference", context)
    reference = _required_string(row, "text", context)
    if reference != expected_reference:
        raise ValueError(
            f"{context} reference drift: expected {expected_reference!r}, received {reference!r}"
        )
    if "\n" in reference or "\\n" in reference:
        raise ValueError(f"{context} is not a single text line")

    expected_writer = selection.get("writer_id")
    actual_writer = _required_writer_id(row, context)
    if actual_writer != expected_writer:
        raise ValueError(
            f"{context} writer drift: expected {expected_writer!r}, received {actual_writer!r}"
        )
    if _required_string(row, "language", context) != language:
        raise ValueError(f"{context} language does not match {language!r}")
    source = _required_string(row, "source", context)
    if source not in allowed_sources:
        raise ValueError(f"{context} source {source!r} is outside the allowed source list")

    image = row.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"{context}.image must be an object")
    width = _required_positive_int(image, "width", f"{context}.image")
    height = _required_positive_int(image, "height", f"{context}.image")
    if width != _required_positive_int(selection, "width", context):
        raise ValueError(f"{context} image width drift")
    if height != _required_positive_int(selection, "height", context):
        raise ValueError(f"{context} image height drift")
    if height > max_height:
        raise ValueError(f"{context} image height exceeds the configured line limit")


def _validated_image_url(image: Mapping[str, Any], context: str) -> str:
    source = image.get("src")
    if not isinstance(source, str):
        raise ValueError(f"{context}.image.src must be a string")
    parsed = urlparse(source)
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_IMAGE_HOST:
        raise ValueError(f"{context}.image.src must use the trusted Dataset Viewer host")
    return source


def _validate_jpeg(image_bytes: bytes, *, width: int, height: int, context: str) -> None:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            actual_size = image.size
            actual_format = image.format
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"{context} image is not decodable") from exc
    if actual_format != "JPEG":
        raise ValueError(f"{context} image format must be JPEG, received {actual_format!r}")
    if actual_size != (width, height):
        raise ValueError(
            f"{context} decoded size drift: expected {(width, height)}, received {actual_size}"
        )


def _row_slices(row: Mapping[str, Any], source: str) -> list[str]:
    slices = {"synthetic", f"source_{source.replace('-', '_')}"}
    noise = row.get("augmented_noise")
    if not isinstance(noise, bool):
        raise ValueError("augmented_noise must be boolean")
    slices.add("noise_augmented" if noise else "no_noise")

    neatness = row.get("neatness")
    if isinstance(neatness, bool) or not isinstance(neatness, (int, float)):
        raise ValueError("neatness must be numeric")
    if neatness < 70:
        slices.add("neatness_low")
    elif neatness < 85:
        slices.add("neatness_mid")
    else:
        slices.add("neatness_high")

    color = _required_string(row, "color", "row").casefold().replace(" ", "_")
    if not color.replace("_", "").isalnum():
        raise ValueError("color must contain only alphanumeric characters, spaces, or underscores")
    slices.add(f"ink_color_{color}")
    return sorted(slices)


def _attribution_text(spec: Mapping[str, Any]) -> str:
    dataset = spec["dataset"]
    return (
        "# Dataset attribution\n\n"
        f"{_required_string(dataset, 'attribution', 'dataset')}. "
        f"Source: {_required_https_url(dataset, 'source_url', 'dataset')} at revision "
        f"`{_required_sha(dataset, 'revision', 'dataset')}`.\n\n"
        f"License: {_required_string(dataset, 'license_id', 'dataset')} "
        f"({_required_https_url(dataset, 'license_url', 'dataset')}).\n\n"
        "This is a selected and normalized derivative: files were renamed, rows were filtered "
        "to synthetic Faker-generated names/dates, and pipeline slices were added. It is a "
        "synthetic engineering smoke set, not evidence of performance on student handwriting.\n"
    )


def _card_license(repo_metadata: Mapping[str, Any]) -> str:
    card_data = repo_metadata.get("cardData")
    if not isinstance(card_data, dict):
        raise ValueError("Hugging Face dataset metadata has no cardData object")
    license_value = card_data.get("license")
    if not isinstance(license_value, str) or not license_value.strip():
        raise ValueError("Hugging Face dataset card has no single license value")
    return license_value.strip()


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return result.strip()


def _required_sha(value: Mapping[str, Any], key: str, context: str) -> str:
    result = _required_string(value, key, context)
    if len(result) != 64 and len(result) != 40:
        raise ValueError(f"{context}.{key} must be a 40- or 64-character SHA")
    if any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{context}.{key} must be lowercase hexadecimal")
    return result


def _required_string_list(value: Mapping[str, Any], key: str, context: str) -> list[str]:
    result = value.get(key)
    if not isinstance(result, list) or not result:
        raise ValueError(f"{context}.{key} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{context}.{key} values must be non-empty strings")
    return [item.strip() for item in result]


def _required_positive_int(value: Mapping[str, Any], key: str, context: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return result


def _required_nonnegative_int(value: Mapping[str, Any], key: str, context: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(f"{context}.{key} must be a non-negative integer")
    return result


def _required_writer_id(row: Mapping[str, Any], context: str) -> int | str:
    writer_id = row.get("writer_id")
    if isinstance(writer_id, bool) or not isinstance(writer_id, (int, str)):
        raise ValueError(f"{context}.writer_id must be an integer or string")
    if isinstance(writer_id, str) and not writer_id.strip():
        raise ValueError(f"{context}.writer_id must be non-empty")
    return writer_id


def _required_https_url(value: Mapping[str, Any], key: str, context: str) -> str:
    result = _required_string(value, key, context)
    parsed = urlparse(result)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{context}.{key} must be an absolute HTTPS URL")
    return result


def _fetch_json(url: str) -> Mapping[str, Any]:
    payload = _fetch_bytes(url)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Remote response is not valid JSON: {url}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Remote response must be a JSON object: {url}")
    return parsed


def _fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "inkbridge-ai-smoke-subset/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - URLs are validated/configured.
        return cast(bytes, response.read())


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "text", "writer_id", "slices"])
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
