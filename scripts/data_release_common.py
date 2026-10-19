"""Standard-library helpers for the versioned experimental input archive."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import BinaryIO
from urllib.parse import urlparse
import zipfile

CHUNK_SIZE = 1024 * 1024
SCHEMA_VERSION = 1
DEFAULT_PATHS = (
    "data/energy_scenarios.csv",
    "data/R05-D333.sample.json",
    "data/representative_tasks_eight_types.csv.gz",
)


class DataReleaseError(Exception):
    """An invalid manifest, incomplete dataset, or unsafe archive."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_data_path(value: str) -> str:
    """Accept only canonical, portable file paths underneath data/."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise DataReleaseError(f"Unsafe data path: {value!r}")
    parts = value.split("/")
    if len(parts) < 2 or parts[0] != "data" or any(part in ("", ".", "..") for part in parts):
        raise DataReleaseError(f"Unsafe data path: {value!r}")
    # These names cannot be represented safely on Windows, the project's authoring platform.
    for part in parts:
        stem = part.split(".", 1)[0].upper()
        if part.endswith((" ", ".")) or any(char in part for char in '<>"|?*') or any(ord(char) < 32 for char in part):
            raise DataReleaseError(f"Unsafe data path: {value!r}")
        if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            raise DataReleaseError(f"Unsafe data path: {value!r}")
    return PurePosixPath(value).as_posix()


def _validate_record(record: dict) -> None:
    if not isinstance(record, dict):
        raise DataReleaseError("Each manifest file record must be an object.")
    validate_data_path(record.get("path"))
    if type(record.get("size")) is not int or record["size"] < 0:
        raise DataReleaseError(f"Invalid file size for {record['path']}.")
    value = record.get("sha256")
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DataReleaseError(f"Invalid SHA-256 for {record['path']}.")


def validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise DataReleaseError("Unsupported data manifest schema; update the download scripts.")
    for name in ("dataset_version", "release_tag"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise DataReleaseError(f"Missing manifest field: {name}.")
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise DataReleaseError("Missing archive metadata in the data manifest.")
    name = archive.get("name")
    if not isinstance(name, str) or not name or name in (".", "..") or any(char in name for char in "/\\:"):
        raise DataReleaseError("Invalid archive name in the data manifest.")
    if type(archive.get("size")) is not int or archive["size"] < 0:
        raise DataReleaseError("Invalid archive size in the data manifest.")
    digest = archive.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise DataReleaseError("Invalid archive SHA-256 in the data manifest.")
    url = archive.get("url")
    if not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).netloc:
        raise DataReleaseError("The manifest must declare an HTTPS archive URL.")
    seen = set()
    for group in ("default_files", "files"):
        records = manifest.get(group)
        if not isinstance(records, list):
            raise DataReleaseError(f"Missing file list in the data manifest: {group}.")
        for record in records:
            _validate_record(record)
            key = record["path"].casefold()
            if key in seen:
                raise DataReleaseError(f"Duplicate manifest path: {record['path']}.")
            seen.add(key)
    # A file cannot also be a parent directory of another declared file.
    for name in seen:
        if any("/".join(name.split("/")[:index]) in seen for index in range(1, len(name.split("/")))):
            raise DataReleaseError(f"Conflicting manifest paths under {name}.")
    return manifest


def load_manifest(path: Path | str) -> dict:
    path = Path(path)
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise DataReleaseError("Data manifest is unexpectedly large.")
        with path.open("r", encoding="utf-8") as source:
            return validate_manifest(json.load(source))
    except (OSError, ValueError) as exc:
        raise DataReleaseError(f"Cannot read data manifest {path}: {exc}") from exc


def _safe_target(destination: Path, name: str) -> Path:
    validate_data_path(name)
    target = destination.joinpath(*name.split("/"))
    # Refuse symlinks even when their current target happens to stay inside the repo.
    cursor = target
    while cursor != destination:
        if cursor.is_symlink():
            raise DataReleaseError(f"Refusing a symbolic link in destination path: {cursor}")
        cursor = cursor.parent
    if not target.resolve().is_relative_to(destination):
        raise DataReleaseError(f"Destination escapes the repository: {name}")
    return target


def check_records(records: list[dict], destination: Path | str) -> list[str]:
    destination = Path(destination).resolve()
    problems = []
    for record in records:
        try:
            target = _safe_target(destination, record["path"])
            if not target.is_file():
                problems.append(f"Missing: {record['path']}")
            elif target.stat().st_size != record["size"] or sha256_file(target) != record["sha256"]:
                problems.append(f"Modified or corrupt: {record['path']}")
        except (DataReleaseError, OSError) as exc:
            problems.append(str(exc))
    return problems


def check_dataset(manifest: dict, destination: Path | str) -> list[str]:
    validate_manifest(manifest)
    return check_records(manifest["default_files"] + manifest["files"], destination)


def _archive_entries(archive: zipfile.ZipFile, manifest: dict) -> dict[str, zipfile.ZipInfo]:
    expected = {record["path"]: record for record in manifest["files"]}
    entries = archive.infolist()
    if len(entries) != len(expected):
        raise DataReleaseError("Archive entries do not match the manifest file list.")
    seen = set()
    result = {}
    for entry in entries:
        if entry.orig_filename != entry.filename:
            raise DataReleaseError("Archive entry contains an invalid or truncated filename.")
        name = validate_data_path(entry.filename)
        if name.casefold() in seen:
            raise DataReleaseError(f"Duplicate archive path: {name}")
        seen.add(name.casefold())
        file_type = stat.S_IFMT(entry.external_attr >> 16)
        if entry.is_dir() or file_type not in (0, stat.S_IFREG):
            raise DataReleaseError(f"Archive entry is not a regular input file: {name}")
        if entry.flag_bits & 1:
            raise DataReleaseError(f"Encrypted archive entry is unsupported: {name}")
        if name not in expected:
            raise DataReleaseError(f"Unexpected archive entry: {name}")
        if entry.file_size != expected[name]["size"]:
            raise DataReleaseError(f"Wrong uncompressed size for {name}.")
        result[name] = entry
    if result.keys() != expected.keys():
        raise DataReleaseError("Archive is missing declared input files.")
    return result


def _copy_verified(source: BinaryIO, record: dict, output: BinaryIO | None = None) -> None:
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = source.read(min(CHUNK_SIZE, record["size"] - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > record["size"]:
            raise DataReleaseError(f"Input exceeds its declared size: {record['path']}")
        digest.update(chunk)
        if output is not None:
            output.write(chunk)
    if total != record["size"] or digest.hexdigest() != record["sha256"]:
        raise DataReleaseError(f"SHA-256 or size mismatch for {record['path']}.")


def verify_archive(archive_path: Path | str, manifest: dict) -> None:
    validate_manifest(manifest)
    archive_path = Path(archive_path)
    try:
        if archive_path.stat().st_size != manifest["archive"]["size"] or sha256_file(archive_path) != manifest["archive"]["sha256"]:
            raise DataReleaseError("Archive SHA-256 or size does not match the pinned manifest; download it again.")
        with zipfile.ZipFile(archive_path) as archive:
            entries = _archive_entries(archive, manifest)
            for record in manifest["files"]:
                with archive.open(entries[record["path"]]) as source:
                    _copy_verified(source, record)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DataReleaseError(f"Cannot validate input archive: {exc}") from exc


def install_archive(archive_path: Path | str, manifest: dict, destination: Path | str, force: bool = False) -> tuple[int, int]:
    """Fully validate, preflight conflicts, then install inputs; return (installed, skipped)."""
    verify_archive(archive_path, manifest)
    destination = Path(destination).resolve()
    if destination.exists() and not destination.is_dir():
        raise DataReleaseError(f"Destination is not a directory: {destination}")
    pending = []
    skipped = 0
    for record in manifest["files"]:
        target = _safe_target(destination, record["path"])
        if target.exists():
            if not target.is_file():
                raise DataReleaseError(f"A directory blocks an input file: {record['path']}")
            if target.stat().st_size == record["size"] and sha256_file(target) == record["sha256"]:
                skipped += 1
                continue
            if not force:
                raise DataReleaseError(f"Existing file differs: {record['path']}. Use --force only to replace modified experiment inputs.")
        cursor = target.parent
        while cursor != destination:
            if cursor.exists() and not cursor.is_dir():
                raise DataReleaseError(f"A file blocks an input directory: {cursor}")
            cursor = cursor.parent
        pending.append((record, target))
    if not pending:
        return 0, skipped
    # Stage a second verified read before touching the destination. This also detects
    # an archive changed between the initial validation and the installation read.
    try:
        with tempfile.TemporaryDirectory(prefix="contract-inputs-") as temporary:
            stage = Path(temporary)
            with zipfile.ZipFile(archive_path) as archive:
                entries = _archive_entries(archive, manifest)
                for index, (record, _) in enumerate(pending):
                    with archive.open(entries[record["path"]]) as source, (stage / str(index)).open("wb") as output:
                        _copy_verified(source, record, output)
            for index, (_, target) in enumerate(pending):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(prefix=".data-download-", dir=target.parent, delete=False) as output:
                        temporary_path = Path(output.name)
                        with (stage / str(index)).open("rb") as source:
                            shutil.copyfileobj(source, output, CHUNK_SIZE)
                    os.replace(temporary_path, target)
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DataReleaseError(f"Cannot install input archive: {exc}") from exc
    return len(pending), skipped
