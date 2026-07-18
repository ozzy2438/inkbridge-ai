"""Prepare a private, provenance-locked SMHD student-handwriting research subset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse

from PIL import Image

_INDEX_RECORD = re.compile(r"^(?P<sample>[0-9]{4}-[0-9]{3}),(?P<threshold>[0-9]+)\s+(?P<text>.*)$")
_FIGSHARE_API_HOST = "api.figshare.com"
_FIGSHARE_DOWNLOAD_HOST = "ndownloader.figshare.com"


@dataclass(frozen=True)
class _SourceSample:
    sample_id: str
    writer_id: str
    threshold: int
    reference: str
    normalized_reference: str
    marked: bool
    archive_path: str


def prepare_smhd_research_subset(
    spec_path: str | Path,
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize a deterministic SMHD subset without exporting raw source identifiers."""
    spec_file = Path(spec_path)
    spec = _load_spec(spec_file)
    dataset = _required_mapping(spec, "dataset", "spec")
    constraints = _required_mapping(spec, "constraints", "spec")
    inventory = _required_mapping(spec, "archive_inventory", "spec")
    selection = _required_mapping(spec, "selection", "spec")
    _validate_constraints(constraints)
    _validate_selection(selection)

    repository = Path(repository_root or Path.cwd()).resolve(strict=True)
    source_archive = Path(archive_path)
    destination = Path(output_dir)
    _validate_local_boundaries(
        spec_file=spec_file,
        source_archive=source_archive,
        destination=destination,
        repository=repository,
    )

    archive_spec = _required_mapping(dataset, "archive", "dataset")
    actual_size, actual_md5, actual_sha256 = _file_digests(source_archive)
    if actual_size != _required_positive_int(archive_spec, "size", "dataset.archive"):
        raise ValueError("SMHD archive size drift")
    if actual_md5 != _required_md5(archive_spec, "md5", "dataset.archive"):
        raise ValueError("SMHD archive MD5 drift")
    if actual_sha256 != _required_sha256(archive_spec, "sha256", "dataset.archive"):
        raise ValueError("SMHD archive SHA-256 drift")

    with zipfile.ZipFile(source_archive) as archive:
        index_payloads = _validate_archive_inventory(archive, inventory)
        index_name = _required_string(selection, "index_filename", "selection")
        try:
            main_index = index_payloads[index_name]
        except KeyError as exc:
            raise ValueError("selection.index_filename is not locked in archive_inventory") from exc
        source_samples = _parse_main_index(
            archive,
            main_index,
            constraints=constraints,
            inventory=inventory,
            index_name=index_name,
        )
        selected = _select_samples(source_samples, selection)
        payloads = _normalize_images(archive, selected, constraints, selection)

    rows = [payload["row"] for payload in payloads]
    image_records = [payload["metadata"] for payload in payloads]
    metadata = _build_metadata(
        spec=spec,
        spec_file=spec_file,
        dataset=dataset,
        archive_spec=archive_spec,
        constraints=constraints,
        selection=selection,
        archive_sha256=actual_sha256,
        rows=rows,
        image_records=image_records,
    )
    _publish_output(
        destination,
        rows=rows,
        image_payloads=[
            (cast(str, payload["row"]["filename"]), cast(bytes, payload["content"]))
            for payload in payloads
        ],
        attribution=_attribution_text(dataset, len(rows), metadata["selection"]["writers"]),
        metadata=metadata,
    )
    return metadata


def _load_spec(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SMHD research subset spec does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"SMHD research subset spec is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("SMHD research subset spec must be a JSON object")
    if raw.get("schema_version") != 1:
        raise ValueError("SMHD research subset spec schema_version must be 1")
    for key in ("dataset", "constraints", "archive_inventory", "selection"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"SMHD research subset spec '{key}' must be an object")
    _validate_dataset(cast(Mapping[str, Any], raw["dataset"]))
    _required_string(raw, "purpose", "spec")
    return raw


def _validate_dataset(dataset: Mapping[str, Any]) -> None:
    _required_positive_int(dataset, "article_id", "dataset")
    _required_string(dataset, "doi", "dataset")
    _required_string(dataset, "title", "dataset")
    _required_string(dataset, "citation", "dataset")
    _trusted_url(dataset, "api_url", "dataset", _FIGSHARE_API_HOST)
    _required_https_url(dataset, "source_url", "dataset")
    if _required_string(dataset, "license_id", "dataset") != "CC-BY-NC-4.0":
        raise ValueError("dataset.license_id must be 'CC-BY-NC-4.0'")
    if _required_string(dataset, "license_name", "dataset") != "CC BY-NC 4.0":
        raise ValueError("dataset.license_name must be 'CC BY-NC 4.0'")
    license_url = _required_https_url(dataset, "license_url", "dataset")
    if license_url != "https://creativecommons.org/licenses/by-nc/4.0/":
        raise ValueError("dataset.license_url must identify CC BY-NC 4.0")
    archive = _required_mapping(dataset, "archive", "dataset")
    _required_positive_int(archive, "file_id", "dataset.archive")
    _required_string(archive, "filename", "dataset.archive")
    _required_positive_int(archive, "size", "dataset.archive")
    _required_md5(archive, "md5", "dataset.archive")
    _required_sha256(archive, "sha256", "dataset.archive")
    _trusted_url(archive, "download_url", "dataset.archive", _FIGSHARE_DOWNLOAD_HOST)


def _validate_constraints(constraints: Mapping[str, Any]) -> None:
    expected = {
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
    }
    for key, value in expected.items():
        if constraints.get(key) != value:
            raise ValueError(f"constraints.{key} must be {value!r}")
    _required_positive_int(constraints, "max_source_image_bytes", "constraints")


def _validate_selection(selection: Mapping[str, Any]) -> None:
    algorithm = _required_string(selection, "algorithm", "selection")
    if algorithm != "sha256_ranked_mixed_clean_and_marker_v1":
        raise ValueError("selection.algorithm must be 'sha256_ranked_mixed_clean_and_marker_v1'")
    _required_string(selection, "seed", "selection")
    _required_string(selection, "index_filename", "selection")
    writers = _required_positive_int(selection, "writers", "selection")
    samples_per_writer = _required_positive_int(selection, "samples_per_writer", "selection")
    clean = _required_positive_int(selection, "required_clean_per_writer", "selection")
    marked = _required_positive_int(selection, "required_marked_per_writer", "selection")
    if writers < 3:
        raise ValueError("selection.writers must be at least 3 for writer-isolated splits")
    if samples_per_writer < 3:
        raise ValueError("selection.samples_per_writer must be at least 3")
    if clean + marked > samples_per_writer:
        raise ValueError("Required clean and marked samples exceed samples_per_writer")


def _validate_local_boundaries(
    *,
    spec_file: Path,
    source_archive: Path,
    destination: Path,
    repository: Path,
) -> None:
    spec_resolved = spec_file.resolve(strict=True)
    if not _is_within(spec_resolved, repository):
        raise ValueError("SMHD subset spec must be inside the repository")
    if source_archive.is_symlink() or not source_archive.is_file():
        raise FileNotFoundError(
            f"SMHD source archive must be a regular local file: {source_archive}"
        )
    source_resolved = source_archive.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if _is_within(source_resolved, repository):
        raise ValueError("SMHD source archive must remain outside the repository")
    if _is_within(destination_resolved, repository):
        raise ValueError("SMHD normalized output must remain outside the repository")
    if destination.exists():
        raise FileExistsError(f"SMHD normalized output already exists: {destination}")
    if destination.is_symlink():
        raise ValueError("SMHD normalized output cannot be a symlink")
    if source_resolved == destination_resolved:
        raise ValueError("SMHD source archive and normalized output must differ")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("SMHD normalized output parent must be an existing regular directory")
    _require_owner_only(source_archive, "SMHD source archive")
    _require_owner_only(source_archive.parent, "SMHD source archive directory")
    _require_owner_only(destination.parent, "SMHD normalized output parent")


def _validate_archive_inventory(
    archive: zipfile.ZipFile, inventory: Mapping[str, Any]
) -> dict[str, bytes]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("SMHD archive contains duplicate paths")
    if len(infos) != _required_positive_int(inventory, "entries", "archive_inventory"):
        raise ValueError("SMHD archive entry-count drift")
    png_infos = [info for info in infos if info.filename.lower().endswith(".png")]
    if len(png_infos) != _required_positive_int(inventory, "png_entries", "archive_inventory"):
        raise ValueError("SMHD archive PNG-count drift")

    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("SMHD archive contains an unsafe path")
        if info.flag_bits & 1:
            raise ValueError("SMHD archive contains an encrypted entry")
        mode = info.external_attr >> 16
        if mode and stat.S_IFMT(mode) == stat.S_IFLNK:
            raise ValueError("SMHD archive contains a symbolic link")

    index_specs = _required_mapping(inventory, "index_files", "archive_inventory")
    expected_other_files = set(index_specs)
    actual_other_files = {
        info.filename
        for info in infos
        if not info.is_dir() and not info.filename.lower().endswith(".png")
    }
    if actual_other_files != expected_other_files:
        raise ValueError("SMHD archive non-image inventory drift")

    payloads: dict[str, bytes] = {}
    for name, raw_spec in index_specs.items():
        if not isinstance(name, str) or not name or not isinstance(raw_spec, dict):
            raise ValueError("archive_inventory.index_files must map filenames to objects")
        path = PurePosixPath(name)
        if path.name != name:
            raise ValueError("SMHD index filenames must be root-level basenames")
        index_spec = cast(Mapping[str, Any], raw_spec)
        try:
            payload = archive.read(name)
        except KeyError as exc:
            raise ValueError(f"SMHD archive index is missing: {name}") from exc
        if len(payload) != _required_positive_int(
            index_spec, "size", f"archive_inventory.index_files.{name}"
        ):
            raise ValueError(f"SMHD index size drift: {name}")
        if _sha256_bytes(payload) != _required_sha256(
            index_spec, "sha256", f"archive_inventory.index_files.{name}"
        ):
            raise ValueError(f"SMHD index SHA-256 drift: {name}")
        line_count = len(payload.decode("utf-8-sig").splitlines())
        if line_count != _required_positive_int(
            index_spec, "records", f"archive_inventory.index_files.{name}"
        ):
            raise ValueError(f"SMHD index record-count drift: {name}")
        payloads[name] = payload
    return payloads


def _parse_main_index(
    archive: zipfile.ZipFile,
    payload: bytes,
    *,
    constraints: Mapping[str, Any],
    inventory: Mapping[str, Any],
    index_name: str,
) -> list[_SourceSample]:
    records: list[_SourceSample] = []
    seen_samples: set[str] = set()
    max_image_bytes = _required_positive_int(constraints, "max_source_image_bytes", "constraints")
    for row_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), start=1):
        match = _INDEX_RECORD.fullmatch(line)
        if match is None:
            raise ValueError(f"SMHD main index row {row_number} has invalid syntax")
        sample_id = match.group("sample")
        if sample_id in seen_samples:
            raise ValueError(f"SMHD main index duplicates sample {sample_id}")
        seen_samples.add(sample_id)
        writer_id = sample_id.split("-", 1)[0]
        threshold = int(match.group("threshold"))
        if threshold < 0 or threshold > 255:
            raise ValueError(f"SMHD main index row {row_number} has invalid threshold")
        reference = match.group("text")
        normalized = _normalize_reference(reference)
        archive_path = f"{writer_id}/{sample_id}.png"
        try:
            image_info = archive.getinfo(archive_path)
        except KeyError as exc:
            raise ValueError(f"SMHD main index image is missing for row {row_number}") from exc
        if image_info.file_size <= 0 or image_info.file_size > max_image_bytes:
            raise ValueError(f"SMHD main index image size is invalid for row {row_number}")
        records.append(
            _SourceSample(
                sample_id=sample_id,
                writer_id=writer_id,
                threshold=threshold,
                reference=reference,
                normalized_reference=normalized,
                marked="#" in reference,
                archive_path=archive_path,
            )
        )

    index_specs = _required_mapping(inventory, "index_files", "archive_inventory")
    main_spec = _required_mapping(index_specs, index_name, "archive_inventory.index_files")
    usable = [record for record in records if record.normalized_reference]
    expected_usable = _required_positive_int(
        main_spec, "usable_records", f"archive_inventory.index_files.{index_name}"
    )
    if len(usable) != expected_usable:
        raise ValueError("SMHD usable main-index record-count drift")
    expected_writers = _required_positive_int(
        main_spec, "writers", f"archive_inventory.index_files.{index_name}"
    )
    if len({record.writer_id for record in records}) != expected_writers:
        raise ValueError("SMHD main-index writer-count drift")
    return usable


def _select_samples(
    samples: list[_SourceSample], selection: Mapping[str, Any]
) -> list[_SourceSample]:
    seed = _required_string(selection, "seed", "selection")
    writer_count = _required_positive_int(selection, "writers", "selection")
    samples_per_writer = _required_positive_int(selection, "samples_per_writer", "selection")
    required_clean = _required_positive_int(selection, "required_clean_per_writer", "selection")
    required_marked = _required_positive_int(selection, "required_marked_per_writer", "selection")
    grouped: dict[str, list[_SourceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.writer_id].append(sample)

    eligible = {
        writer: writer_samples
        for writer, writer_samples in grouped.items()
        if len(writer_samples) >= samples_per_writer
        and sum(not sample.marked for sample in writer_samples) >= required_clean
        and sum(sample.marked for sample in writer_samples) >= required_marked
    }
    if len(eligible) < writer_count:
        raise ValueError(
            f"SMHD selection requires {writer_count} eligible writers, found {len(eligible)}"
        )

    writers = sorted(
        eligible,
        key=lambda writer: _rank(seed, "writer", writer),
    )[:writer_count]
    selected: list[_SourceSample] = []
    for writer in writers:
        candidates = eligible[writer]
        clean = sorted(
            (sample for sample in candidates if not sample.marked),
            key=lambda sample: _rank(seed, "clean", sample.sample_id),
        )[:required_clean]
        marked = sorted(
            (sample for sample in candidates if sample.marked),
            key=lambda sample: _rank(seed, "marked", sample.sample_id),
        )[:required_marked]
        chosen = [*clean, *marked]
        chosen_ids = {sample.sample_id for sample in chosen}
        remaining = sorted(
            (sample for sample in candidates if sample.sample_id not in chosen_ids),
            key=lambda sample: _rank(seed, "remaining", sample.sample_id),
        )
        chosen.extend(remaining[: samples_per_writer - len(chosen)])
        if len(chosen) != samples_per_writer:
            raise RuntimeError("SMHD deterministic selection produced an incomplete writer set")
        selected.extend(chosen)
    return sorted(selected, key=lambda sample: _opaque_filename(seed, sample.sample_id))


def _normalize_images(
    archive: zipfile.ZipFile,
    selected: list[_SourceSample],
    constraints: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seed = _required_string(selection, "seed", "selection")
    max_image_bytes = _required_positive_int(constraints, "max_source_image_bytes", "constraints")
    payloads: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()
    for sample in selected:
        source_content = archive.read(sample.archive_path)
        if not source_content or len(source_content) > max_image_bytes:
            raise ValueError("Selected SMHD source image exceeds the locked size limit")
        output_content, width, height = _decode_and_encode_pgm(source_content)
        filename = _opaque_filename(seed, sample.sample_id)
        if filename in seen_filenames:
            raise RuntimeError(f"SMHD opaque filename collision: {filename}")
        seen_filenames.add(filename)
        pseudonymous_writer = _pseudonymous_writer(seed, sample.writer_id)
        source_slice = "publisher_hash_marker" if sample.marked else "publisher_no_hash_marker"
        slices = sorted(
            {
                "age_band_unknown",
                "cc_by_nc_research_only",
                "real_handwriting",
                "reference_hash_marker_removed" if sample.marked else "reference_unchanged",
                "scanned",
                "smhd",
                "student",
                source_slice,
            }
        )
        payloads.append(
            {
                "row": {
                    "filename": filename,
                    "text": sample.normalized_reference,
                    "writer_id": pseudonymous_writer,
                    "slices": "|".join(slices),
                },
                "content": output_content,
                "metadata": {
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "source_marker_present": sample.marked,
                    "source_sha256": _sha256_bytes(source_content),
                    "output_sha256": _sha256_bytes(output_content),
                },
            }
        )
    return payloads


def _decode_and_encode_pgm(content: bytes) -> tuple[bytes, int, int]:
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            actual_format = image.format
            width, height = image.size
            grayscale = image.convert("L")
    except (OSError, SyntaxError) as exc:
        raise ValueError("Selected SMHD source image is not decodable") from exc
    if actual_format != "PNG":
        raise ValueError(f"Selected SMHD source format must be PNG, received {actual_format!r}")
    if width <= 0 or height <= 0 or width > 10_000 or height > 10_000:
        raise ValueError("Selected SMHD source dimensions are invalid")
    payload = f"P5\n{width} {height}\n255\n".encode("ascii") + cast(bytes, grayscale.tobytes())
    return payload, width, height


def _build_metadata(
    *,
    spec: Mapping[str, Any],
    spec_file: Path,
    dataset: Mapping[str, Any],
    archive_spec: Mapping[str, Any],
    constraints: Mapping[str, Any],
    selection: Mapping[str, Any],
    archive_sha256: str,
    rows: list[dict[str, str]],
    image_records: list[dict[str, Any]],
) -> dict[str, Any]:
    writers = len({row["writer_id"] for row in rows})
    marker_count = sum(bool(record["source_marker_present"]) for record in image_records)
    labels_content = _csv_bytes(rows)
    attribution = _attribution_text(dataset, len(rows), writers)
    return {
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
            "commercial_use_allowed": constraints["commercial_use_allowed"],
            "archive_file_id": _required_positive_int(archive_spec, "file_id", "dataset.archive"),
            "archive_filename": _required_string(archive_spec, "filename", "dataset.archive"),
            "archive_md5": _required_md5(archive_spec, "md5", "dataset.archive"),
            "archive_sha256": archive_sha256,
        },
        "privacy": {
            "synthetic": False,
            "real_handwriting": True,
            "contains_student_data": True,
            "population": constraints["population"],
            "source_writer_ids_exported": False,
            "writer_identity": "deterministic_pseudonym_linkable_to_public_source",
            "repository_artifacts_allowed": False,
            "offline_only": True,
        },
        "reference_quality": {
            "status": constraints["reference_status"],
            "transform": constraints["reference_transform"],
            "gold_ready": False,
            "required_next_step": "independent human transcription and crop-boundary review",
        },
        "evidence_boundary": {
            "research_rehearsal_only": True,
            "production_pilot_evidence": False,
            "consented_protected_pilot_substitute": False,
        },
        "source_spec": {
            "filename": spec_file.name,
            "sha256": _sha256_file(spec_file),
        },
        "selection": {
            "algorithm": "sha256_ranked_mixed_clean_and_marker_v1",
            "seed": _required_string(selection, "seed", "selection"),
            "writers": writers,
            "samples_per_writer": _required_positive_int(
                selection, "samples_per_writer", "selection"
            ),
            "samples": len(rows),
            "source_marker_present": marker_count,
            "source_marker_absent": len(rows) - marker_count,
        },
        "output": {
            "labels_filename": "labels.csv",
            "labels_sha256": _sha256_bytes(labels_content),
            "attribution_filename": "ATTRIBUTION.md",
            "attribution_sha256": _sha256_bytes(attribution.encode("utf-8")),
            "num_samples": len(rows),
            "num_writers": writers,
            "images": image_records,
        },
    }


def _publish_output(
    destination: Path,
    *,
    rows: list[dict[str, str]],
    image_payloads: list[tuple[str, bytes]],
    attribution: str,
    metadata: dict[str, Any],
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _set_private_directory(staging)
        images_dir = staging / "images"
        images_dir.mkdir()
        _set_private_directory(images_dir)
        for filename, content in image_payloads:
            _write_private_file(images_dir / filename, content)
        labels_content = _csv_bytes(rows)
        _write_private_file(staging / "labels.csv", labels_content)
        _write_private_file(staging / "ATTRIBUTION.md", attribution.encode("utf-8"))
        metadata_content = (
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _write_private_file(staging / "source.meta.json", metadata_content)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["filename", "text", "writer_id", "slices"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _attribution_text(dataset: Mapping[str, Any], samples: int, writers: int) -> str:
    return (
        "# Dataset attribution and use boundary\n\n"
        f"{_required_string(dataset, 'citation', 'dataset')}\n\n"
        f"Source: {_required_https_url(dataset, 'source_url', 'dataset')}\n\n"
        f"License: {_required_string(dataset, 'license_name', 'dataset')} "
        f"({_required_https_url(dataset, 'license_url', 'dataset')}).\n\n"
        f"This private derivative contains {samples} deterministic grayscale line images from "
        f"{writers} pseudonymous student writers. Source images were decoded, stripped of image "
        "metadata, converted to binary PGM, and renamed. The publisher's '#' annotation markers "
        "were removed from references and whitespace was collapsed; no other transcription edits "
        "were made. References have not received independent human review.\n\n"
        "CC BY-NC 4.0 forbids commercial use. This package is an offline research rehearsal only: "
        "it is not a gold benchmark, production-pilot evidence, or a substitute for an approved "
        "and consented protected student pilot. Do not commit or upload this derivative or its "
        "sample-level outputs to GitHub.\n"
    )


def _normalize_reference(reference: str) -> str:
    return " ".join(reference.replace("#", "").split())


def _opaque_filename(seed: str, sample_id: str) -> str:
    digest = _rank(seed, "sample", sample_id)
    return f"smhd-line-{digest[:24]}.pgm"


def _pseudonymous_writer(seed: str, writer_id: str) -> str:
    digest = _rank(seed, "writer-pseudonym", writer_id)
    return f"smhd-w-{digest[:16]}"


def _rank(seed: str, domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}\0{seed}\0{value}".encode()).hexdigest()


def _file_digests(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return size, md5.hexdigest(), sha256.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_private_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    if os.name == "posix":
        path.chmod(0o600)


def _set_private_directory(path: Path) -> None:
    if os.name == "posix":
        path.chmod(0o700)


def _require_owner_only(path: Path, label: str) -> None:
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError(f"{label} must not grant group or world permissions: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _required_mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return result


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
