#!/usr/bin/env python3
"""Build the deterministic experiment input ZIP and its pinned data manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
import zipfile

try:
    from .data_release_common import CHUNK_SIZE, DEFAULT_PATHS, DataReleaseError, file_record, load_manifest, sha256_file, validate_data_path, validate_manifest, verify_archive
except ImportError:
    from data_release_common import CHUNK_SIZE, DEFAULT_PATHS, DataReleaseError, file_record, load_manifest, sha256_file, validate_data_path, validate_manifest, verify_archive

ARCHIVE_NAME = "experiment-data-v1.zip"
RELEASE_TAG = "data-v1"
ARCHIVE_URL = f"https://github.com/Zijun0819/contract_ai_data_center/releases/download/{RELEASE_TAG}/{ARCHIVE_NAME}"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRECTORIES = {"results", "output", "outputs", "tmp", "__pycache__", ".cache"}


def is_metadata(path: Path) -> bool:
    return path.suffix.lower() == ".md" or path.name.lower().endswith("manifest.json")


def is_generated_output(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in {"verification.json", "workload_validation.json", "validation.json", "audit.json", "results.json"}
        or name.endswith(("_audit.json", "_validation.json", "_results.json", ".log", ".tmp", ".pyc", ".pyo", ".bak", ".cache"))
    )


def _validate_source_completeness(root: Path) -> None:
    """Catch a fresh clone or partial restore before replacing the published file map."""
    data = root / "data"
    missing = set()

    def require(reference: str, parent: Path) -> None:
        if not isinstance(reference, str):
            return
        path = (parent / reference).resolve()
        if path.is_relative_to(data) and not path.is_file():
            missing.add(path.relative_to(root).as_posix())

    path_fields = {"recipe", "original_recipe", "source_file", "anchor_recipe", "anchor_source", "anchor_horizon_manifest"}

    def walk(value: object, parent: Path) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "files" and isinstance(item, dict):
                    for reference in item:
                        require(reference, parent)
                elif key in path_fields:
                    require(item, parent)
                else:
                    walk(item, parent)
        elif isinstance(value, list):
            for item in value:
                walk(item, parent)

    for path in sorted(data.rglob("*manifest.json")):
        if path.name == "dataset_manifest.json":
            prior = load_manifest(path)
            for record in prior["default_files"] + prior["files"]:
                require(record["path"], root)
            continue
        try:
            with path.open("r", encoding="utf-8") as source:
                walk(json.load(source), path.parent)
        except (OSError, ValueError) as exc:
            raise DataReleaseError(f"Cannot read source metadata {path}: {exc}") from exc
    if missing:
        raise DataReleaseError(
            "Source data is incomplete; refusing to replace the release manifest. "
            "Run python scripts/download_data.py before rebuilding, or restore all original inputs. Missing:\n"
            + "\n".join(sorted(missing))
        )


def build_release(root: Path | str, output_dir: Path | str | None = None, manifest_path: Path | str | None = None) -> dict:
    root = Path(root).resolve()
    data = root / "data"
    output_dir = Path(output_dir).resolve() if output_dir is not None else root / "release_assets"
    manifest_path = Path(manifest_path).resolve() if manifest_path is not None else data / "dataset_manifest.json"
    if not data.is_dir() or data.is_symlink():
        raise DataReleaseError(f"Missing or linked source data directory: {data}")
    if output_dir.is_relative_to(data):
        raise DataReleaseError("Choose an archive output directory outside data/ to prevent the archive including itself.")
    default_files = []
    for name in DEFAULT_PATHS:
        path = root.joinpath(*name.split("/"))
        if not path.is_file() or path.is_symlink():
            raise DataReleaseError(f"Required default input is missing or linked: {name}")
        default_files.append(file_record(path, root))
    sources = []
    for path in sorted(data.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise DataReleaseError(f"Refusing linked source input: {path}")
        if path.is_dir() and path.name.lower() in FORBIDDEN_DIRECTORIES:
            raise DataReleaseError(f"Generated output/cache directory cannot be included in a data release: {path}. Move it outside data/ before rebuilding.")
        if not path.is_file() or is_metadata(path) or is_generated_output(path) or path.relative_to(root).as_posix() in DEFAULT_PATHS:
            continue
        validate_data_path(path.relative_to(root).as_posix())
        sources.append(path)
    if not sources:
        raise DataReleaseError("No experiment inputs are present. Run python scripts/download_data.py before building a release from a fresh clone.")
    _validate_source_completeness(root)
    files = [file_record(path, root) for path in sources]
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME
    temporary_path = output_dir / (ARCHIVE_NAME + ".tmp")
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
            for path, record in zip(sources, files):
                entry = zipfile.ZipInfo(record["path"], date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = (stat.S_IFREG | 0o644) << 16
                entry._compresslevel = 9
                with path.open("rb") as source, archive.open(entry, "w", force_zip64=record["size"] >= zipfile.ZIP64_LIMIT) as output:
                    while chunk := source.read(CHUNK_SIZE):
                        output.write(chunk)
        # Fail if source bytes changed during packaging, instead of publishing an inconsistent manifest.
        manifest = {
            "schema_version": 1,
            "dataset_version": "v1",
            "release_tag": RELEASE_TAG,
            "archive": {"name": ARCHIVE_NAME, "url": ARCHIVE_URL, "sha256": sha256_file(temporary_path), "size": temporary_path.stat().st_size},
            "default_files": default_files,
            "files": files,
        }
        validate_manifest(manifest)
        verify_archive(temporary_path, manifest)
        temporary_path.replace(archive_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        (output_dir / "SHA256SUMS").write_text(f"{manifest['archive']['sha256']}  {ARCHIVE_NAME}\n", encoding="ascii", newline="\n")
        return manifest
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT, help="Repository containing the complete local data/ tree.")
    parser.add_argument("--output-dir", type=Path, help="Archive destination (default: <root>/release_assets).")
    parser.add_argument("--manifest", type=Path, help="Manifest destination (default: <root>/data/dataset_manifest.json).")
    args = parser.parse_args(argv)
    try:
        manifest = build_release(args.root, args.output_dir, args.manifest)
    except (DataReleaseError, OSError, zipfile.BadZipFile) as exc:
        print(f"Cannot build experiment data release: {exc}", file=sys.stderr)
        return 1
    output = args.output_dir.resolve() if args.output_dir else args.root.resolve() / "release_assets"
    print(f"Built {output / ARCHIVE_NAME} ({manifest['archive']['size']:,} bytes; {len(manifest['files'])} experiment input files).")
    print(f"SHA-256: {manifest['archive']['sha256']}")
    print(f"Upload {ARCHIVE_NAME} and SHA256SUMS as assets of GitHub Release {RELEASE_TAG}.")
    print("No local input files were removed. Keep this archive version immutable after publication.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
