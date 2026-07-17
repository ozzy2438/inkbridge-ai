"""Prepare a provenance-locked CSAFE real-handwriting smoke subset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

JsonFetcher = Callable[[str], Mapping[str, Any]]
ArchiveFactory = Callable[[str], Any]

_FIGSHARE_API_HOST = "api.figshare.com"
_FIGSHARE_DOWNLOAD_HOST = "ndownloader.figshare.com"
_WRITER_ID = re.compile(r"w[0-9]{4}")


def prepare_csafe_smoke_subset(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    fetch_json: JsonFetcher | None = None,
    archive_factory: ArchiveFactory | None = None,
) -> dict[str, Any]:
    """Fetch selected source pages by range request and publish verified line crops."""
    spec_file = Path(spec_path)
    spec = _load_spec(spec_file)
    dataset = cast(Mapping[str, Any], spec["dataset"])
    constraints = cast(Mapping[str, Any], spec["constraints"])
    selection = cast(list[Mapping[str, Any]], spec["selection"])

    api_url = _trusted_url(dataset, "api_url", "dataset", _FIGSHARE_API_HOST)
    article = (fetch_json or _fetch_json)(api_url)
    _validate_article(article, dataset)

    archive_spec = _required_mapping(dataset, "archive", "dataset")
    archive_url = _trusted_url(
        archive_spec, "download_url", "dataset.archive", _FIGSHARE_DOWNLOAD_HOST
    )
    prompt = _required_string(constraints, "prompt", "constraints")
    _validate_constraints(constraints)

    rows: list[dict[str, str]] = []
    image_payloads: list[tuple[str, bytes, str]] = []
    source_pages: list[dict[str, Any]] = []
    seen_writers: set[str] = set()
    seen_archive_paths: set[str] = set()
    seen_filenames: set[str] = set()
    make_archive = archive_factory or _remote_zip

    with make_archive(archive_url) as archive:
        for page_index, page in enumerate(selection):
            context = f"selection[{page_index}]"
            writer_id = _writer_id(page, context)
            if writer_id in seen_writers:
                raise ValueError(f"{context}.writer_id duplicates {writer_id!r}")
            seen_writers.add(writer_id)

            archive_path = _archive_path(page, writer_id, context)
            if archive_path in seen_archive_paths:
                raise ValueError(f"{context}.archive_path duplicates {archive_path!r}")
            seen_archive_paths.add(archive_path)
            try:
                archive_info = archive.getinfo(archive_path)
            except KeyError as exc:
                raise ValueError(
                    f"{context} source page is missing from the locked archive"
                ) from exc
            expected_size = _required_positive_int(page, "archive_size", context)
            expected_crc = _required_nonnegative_int(page, "archive_crc32", context)
            if archive_info.file_size != expected_size:
                raise ValueError(f"{context} archive entry size drift")
            if archive_info.CRC != expected_crc:
                raise ValueError(f"{context} archive entry CRC drift")

            source_bytes = cast(bytes, archive.read(archive_path))
            source_hash = _sha256_bytes(source_bytes)
            expected_source_hash = _required_sha256(page, "source_sha256", context)
            if source_hash != expected_source_hash:
                raise ValueError(
                    f"{context} source hash drift: expected {expected_source_hash}, "
                    f"received {source_hash}"
                )
            width = _required_positive_int(page, "width", context)
            height = _required_positive_int(page, "height", context)
            grayscale = _decode_png(source_bytes, width=width, height=height, context=context)
            lines = _required_mapping_list(page, "lines", context)
            references = [
                _required_string(line, "reference", f"{context}.lines[{index}]")
                for index, line in enumerate(lines)
            ]
            if " ".join(references) != prompt:
                raise ValueError(f"{context} line references do not reconstruct the locked prompt")

            previous_bottom = 0
            page_outputs: list[dict[str, Any]] = []
            for line_index, line in enumerate(lines, start=1):
                line_context = f"{context}.lines[{line_index - 1}]"
                bbox = _bbox(line, width=width, height=height, context=line_context)
                if bbox[1] < previous_bottom:
                    raise ValueError(f"{line_context}.bbox overlaps or reorders the previous line")
                previous_bottom = bbox[3]
                output_bytes = _encode_pgm(grayscale.crop(bbox))
                output_hash = _sha256_bytes(output_bytes)
                expected_output_hash = _required_sha256(line, "output_sha256", line_context)
                if output_hash != expected_output_hash:
                    raise ValueError(
                        f"{line_context} output hash drift: expected {expected_output_hash}, "
                        f"received {output_hash}"
                    )
                filename = f"csafe-{writer_id}-s01-phr-r01-line-{line_index}.pgm"
                if filename in seen_filenames:
                    raise ValueError(f"Generated duplicate filename: {filename}")
                seen_filenames.add(filename)
                slices = sorted(
                    {
                        "adult",
                        "csafe",
                        f"page_lines_{len(lines)}",
                        "prompt_phr",
                        "real_handwriting",
                        "reference_prompt_derived",
                        "repetition_1",
                        "session_1",
                    }
                )
                rows.append(
                    {
                        "filename": filename,
                        "text": references[line_index - 1],
                        "writer_id": writer_id,
                        "slices": "|".join(slices),
                    }
                )
                image_payloads.append((filename, output_bytes, output_hash))
                page_outputs.append(
                    {
                        "line": line_index,
                        "bbox": list(bbox),
                        "filename": filename,
                        "sha256": output_hash,
                    }
                )
            source_pages.append(
                {
                    "archive_path": archive_path,
                    "archive_size": expected_size,
                    "archive_crc32": expected_crc,
                    "source_sha256": source_hash,
                    "writer_id": writer_id,
                    "outputs": page_outputs,
                }
            )

    destination = Path(output_dir)
    images_dir = destination / "images"
    for filename, content, expected_hash in image_payloads:
        image_path = images_dir / filename
        if image_path.exists() and _sha256_file(image_path) == expected_hash:
            continue
        _atomic_write_bytes(image_path, content)

    labels_path = destination / "labels.csv"
    _atomic_write_csv(labels_path, rows)
    attribution_path = destination / "ATTRIBUTION.md"
    _atomic_write_text(attribution_path, _attribution_text(dataset))

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": _required_string(spec, "purpose", "spec"),
        "dataset": {
            "article_id": _required_positive_int(dataset, "article_id", "dataset"),
            "doi": _required_string(dataset, "doi", "dataset"),
            "title": _required_string(dataset, "title", "dataset"),
            "citation": _required_string(dataset, "citation", "dataset"),
            "source_url": _required_https_url(dataset, "source_url", "dataset"),
            "license_id": _required_string(dataset, "license_id", "dataset"),
            "license_url": _required_https_url(dataset, "license_url", "dataset"),
            "archive_file_id": _required_positive_int(archive_spec, "file_id", "dataset.archive"),
            "archive_filename": _required_string(archive_spec, "filename", "dataset.archive"),
            "archive_md5": _required_md5(archive_spec, "md5", "dataset.archive"),
        },
        "privacy": {
            "synthetic": False,
            "real_handwriting": True,
            "population": "adults",
            "contains_student_data": False,
            "writer_identity": "source_participant_id",
            "source_pages_exported": False,
        },
        "reference_quality": {
            "status": "prompt_derived_single_pass",
            "gold_ready": False,
            "required_next_step": "independent human transcription and crop-boundary review",
        },
        "source_spec": {
            "filename": spec_file.name,
            "sha256": _sha256_file(spec_file),
        },
        "source_pages": source_pages,
        "output": {
            "labels_filename": labels_path.name,
            "labels_sha256": _sha256_file(labels_path),
            "attribution_filename": attribution_path.name,
            "attribution_sha256": _sha256_file(attribution_path),
            "num_samples": len(rows),
            "num_writers": len(seen_writers),
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
        raise FileNotFoundError(f"CSAFE smoke subset spec does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"CSAFE smoke subset spec is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("CSAFE smoke subset spec must be a JSON object")
    if raw.get("schema_version") != 1:
        raise ValueError("CSAFE smoke subset spec schema_version must be 1")
    for key in ("dataset", "constraints"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"CSAFE smoke subset spec '{key}' must be an object")
    selection = raw.get("selection")
    if not isinstance(selection, list) or not selection:
        raise ValueError("CSAFE smoke subset spec 'selection' must be a non-empty list")
    if any(not isinstance(page, dict) for page in selection):
        raise ValueError("Every CSAFE smoke subset selection must be an object")
    return raw


def _validate_article(article: Mapping[str, Any], dataset: Mapping[str, Any]) -> None:
    expected_id = _required_positive_int(dataset, "article_id", "dataset")
    if article.get("id") != expected_id:
        raise ValueError(f"Figshare article ID drift: expected {expected_id}")
    for key, label in (("doi", "DOI"), ("title", "title"), ("citation", "citation")):
        expected = _required_string(dataset, key, "dataset")
        if article.get(key) != expected:
            raise ValueError(f"Figshare article {label} drift: expected {expected!r}")

    license_value = article.get("license")
    if not isinstance(license_value, dict):
        raise ValueError("Figshare article license metadata is missing")
    expected_name = _required_string(dataset, "license_name", "dataset")
    expected_url = _required_https_url(dataset, "license_url", "dataset")
    if license_value.get("name") != expected_name or license_value.get("url") != expected_url:
        raise ValueError("Figshare article license drift")

    files = article.get("files")
    if not isinstance(files, list):
        raise ValueError("Figshare article files metadata is missing")
    for field in ("readme", "archive"):
        locked = _required_mapping(dataset, field, "dataset")
        file_id = _required_positive_int(locked, "file_id", f"dataset.{field}")
        current = next(
            (item for item in files if isinstance(item, dict) and item.get("id") == file_id), None
        )
        if current is None:
            raise ValueError(f"Figshare {field} file is missing")
        expected_values = {
            "name": _required_string(locked, "filename", f"dataset.{field}"),
            "size": _required_positive_int(locked, "size", f"dataset.{field}"),
            "supplied_md5": _required_md5(locked, "md5", f"dataset.{field}"),
            "computed_md5": _required_md5(locked, "md5", f"dataset.{field}"),
        }
        if any(current.get(key) != value for key, value in expected_values.items()):
            raise ValueError(f"Figshare {field} file metadata drift")
        if field == "archive":
            expected_download = _trusted_url(
                locked, "download_url", "dataset.archive", _FIGSHARE_DOWNLOAD_HOST
            )
            if current.get("download_url") != expected_download:
                raise ValueError("Figshare archive download URL drift")


def _validate_constraints(constraints: Mapping[str, Any]) -> None:
    expected = {
        "language": "eng",
        "sample_type": "line",
        "population": "adults",
        "contains_student_data": False,
        "real_handwriting": True,
        "prompt_id": "PHR",
        "reference_status": "prompt_derived_single_pass",
        "gold_ready": False,
        "source_pages_exported": False,
    }
    for key, value in expected.items():
        if constraints.get(key) != value:
            raise ValueError(f"constraints.{key} must be {value!r}")


def _decode_png(content: bytes, *, width: int, height: int, context: str) -> Image.Image:
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            actual_format = image.format
            actual_size = image.size
            grayscale = image.convert("L")
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"{context} source page is not decodable") from exc
    if actual_format != "PNG":
        raise ValueError(f"{context} source format must be PNG, received {actual_format!r}")
    if actual_size != (width, height):
        raise ValueError(
            f"{context} decoded size drift: expected {(width, height)}, received {actual_size}"
        )
    return grayscale


def _encode_pgm(image: Image.Image) -> bytes:
    width, height = image.size
    return f"P5\n{width} {height}\n255\n".encode("ascii") + image.tobytes()


def _bbox(
    line: Mapping[str, Any], *, width: int, height: int, context: str
) -> tuple[int, int, int, int]:
    value = line.get("bbox")
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{context}.bbox must contain four integers")
    left, top, right, bottom = cast(list[int], value)
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f"{context}.bbox is empty or has negative coordinates")
    if right > width or bottom > height:
        raise ValueError(f"{context}.bbox exceeds the source page")
    return left, top, right, bottom


def _archive_path(page: Mapping[str, Any], writer_id: str, context: str) -> str:
    value = _required_string(page, "archive_path", context)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context}.archive_path must be a safe relative path")
    expected = f"Session 1/{writer_id}_s01_pPHR_r01.png"
    if value != expected:
        raise ValueError(f"{context}.archive_path must be {expected!r}")
    return value


def _writer_id(page: Mapping[str, Any], context: str) -> str:
    value = _required_string(page, "writer_id", context)
    if _WRITER_ID.fullmatch(value) is None:
        raise ValueError(f"{context}.writer_id must match w followed by four digits")
    return value


def _attribution_text(dataset: Mapping[str, Any]) -> str:
    return (
        "# Dataset attribution\n\n"
        f"{_required_string(dataset, 'citation', 'dataset')}\n\n"
        f"Source: {_required_https_url(dataset, 'source_url', 'dataset')}\n\n"
        f"License: {_required_string(dataset, 'license_name', 'dataset')} "
        f"({_required_https_url(dataset, 'license_url', 'dataset')}).\n\n"
        "This selected derivative contains deterministic grayscale line crops from nine adult "
        "writers' Session 1 PHR pages. Files were cropped, converted to binary PGM, renamed, and "
        "tagged. Source pages are not exported. References were split from the known prompt after "
        "one visual pass; this is a real-handwriting engineering smoke set, not a human-verified "
        "gold benchmark and not evidence of performance on student handwriting.\n"
    )


def _remote_zip(url: str) -> Any:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific.
        raise RuntimeError("remotezip is required to prepare the CSAFE smoke subset") from exc
    return RemoteZip(url)


def _fetch_json(url: str) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": "inkbridge-ai-csafe-smoke/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - URL host is allow-listed.
        payload = cast(bytes, response.read())
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Remote response is not valid JSON: {url}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Remote response must be a JSON object: {url}")
    return value


def _required_mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return result


def _required_mapping_list(
    value: Mapping[str, Any], key: str, context: str
) -> list[Mapping[str, Any]]:
    result = value.get(key)
    if (
        not isinstance(result, list)
        or not result
        or any(not isinstance(item, dict) for item in result)
    ):
        raise ValueError(f"{context}.{key} must be a non-empty list of objects")
    return cast(list[Mapping[str, Any]], result)


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return result.strip()


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


def _required_sha256(value: Mapping[str, Any], key: str, context: str) -> str:
    return _required_hex(value, key, context, length=64)


def _required_md5(value: Mapping[str, Any], key: str, context: str) -> str:
    return _required_hex(value, key, context, length=32)


def _required_hex(value: Mapping[str, Any], key: str, context: str, *, length: int) -> str:
    result = _required_string(value, key, context)
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{context}.{key} must be {length}-character lowercase hexadecimal")
    return result


def _required_https_url(value: Mapping[str, Any], key: str, context: str) -> str:
    result = _required_string(value, key, context)
    parsed = urlparse(result)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{context}.{key} must be an absolute HTTPS URL")
    return result


def _trusted_url(value: Mapping[str, Any], key: str, context: str, host: str) -> str:
    result = _required_https_url(value, key, context)
    if urlparse(result).hostname != host:
        raise ValueError(f"{context}.{key} must use the trusted {host} host")
    return result


def _atomic_write_csv(path: Path, rows: list[dict[str, str]]) -> None:
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


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
