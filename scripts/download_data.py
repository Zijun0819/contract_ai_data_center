#!/usr/bin/env python3
"""Restore or verify versioned experiment inputs, without third-party dependencies."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from .data_release_common import CHUNK_SIZE, DataReleaseError, check_dataset, check_records, install_archive, load_manifest
except ImportError:
    from data_release_common import CHUNK_SIZE, DataReleaseError, check_dataset, check_records, install_archive, load_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def download_archive(url: str, output: Path, archive_metadata: dict) -> None:
    try:
        parsed_url = urlparse(url)
    except ValueError as exc:
        raise DataReleaseError(f"Invalid download URL: {exc}. Use --archive for a local file.") from exc
    if parsed_url.scheme not in {"https", "http"} or not parsed_url.netloc:
        raise DataReleaseError("--url must be an HTTP or HTTPS download URL; use --archive for a local file.")
    digest = hashlib.sha256()
    size = 0
    request = Request(url, headers={"User-Agent": "contract-ai-data-center-data/1"})
    try:
        with urlopen(request, timeout=60) as response, output.open("wb") as target:
            while chunk := response.read(CHUNK_SIZE):
                size += len(chunk)
                if size > archive_metadata["size"]:
                    raise DataReleaseError("Downloaded archive exceeds its pinned size; refusing the unexpected response.")
                target.write(chunk)
                digest.update(chunk)
    except HTTPError as exc:
        if exc.code == 404:
            raise DataReleaseError("The data release asset is unavailable (HTTP 404). It may not have been published yet. Ask the maintainer for experiment-data-v1.zip, then run: python scripts/download_data.py --archive /path/to/experiment-data-v1.zip") from exc
        raise DataReleaseError(f"Data download failed (HTTP {exc.code}). You can supply a local ZIP with --archive.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise DataReleaseError(f"Data download failed: {exc}. You can supply a local ZIP with --archive.") from exc
    if size != archive_metadata["size"] or digest.hexdigest() != archive_metadata["sha256"]:
        raise DataReleaseError("Downloaded archive SHA-256 or size does not match the pinned manifest; retry or obtain the original release asset.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "data" / "dataset_manifest.json", help="Pinned dataset manifest (default: this repository's data/dataset_manifest.json).")
    parser.add_argument("--destination", type=Path, default=REPOSITORY_ROOT, help="Repository root to restore or check (default: this repository).")
    parser.add_argument("--archive", type=Path, help="Use a local release ZIP instead of downloading.")
    parser.add_argument("--url", help="Alternate download URL; the same pinned checksum remains required.")
    parser.add_argument("--check", action="store_true", help="Verify all declared default and experiment input files, without downloading or changing files.")
    parser.add_argument("--force", action="store_true", help="Allow replacement of experiment input files that differ from the manifest.")
    args = parser.parse_args(argv)
    if args.archive and args.url:
        parser.error("--archive and --url are alternatives; supply only one.")
    try:
        manifest = load_manifest(args.manifest)
        problems = check_dataset(manifest, args.destination)
        if args.check:
            if problems:
                for problem in problems:
                    print(problem, file=sys.stderr)
                print(f"Dataset is incomplete or modified ({len(problems)} files). Run python scripts/download_data.py to restore experiment inputs.", file=sys.stderr)
                return 1
            print(f"Verified all {len(manifest['default_files']) + len(manifest['files'])} declared input files.")
            return 0
        if not problems:
            print("All declared input files already match the manifest; no download needed.")
            return 0
        default_problems = check_records(manifest["default_files"], args.destination)
        if default_problems:
            raise DataReleaseError("Default inputs are missing or modified. Restore them from the Git checkout before downloading experiment inputs:\n" + "\n".join(default_problems))
        if args.archive:
            installed, skipped = install_archive(args.archive, manifest, args.destination, args.force)
        else:
            print(f"Downloading {manifest['archive']['name']} ({manifest['archive']['size']:,} bytes)...")
            with tempfile.TemporaryDirectory(prefix="contract-data-download-") as temporary:
                archive_path = Path(temporary) / manifest["archive"]["name"]
                download_archive(args.url or manifest["archive"]["url"], archive_path, manifest["archive"])
                installed, skipped = install_archive(archive_path, manifest, args.destination, args.force)
        remaining = check_dataset(manifest, args.destination)
        if remaining:
            raise DataReleaseError("Inputs changed during restoration:\n" + "\n".join(remaining))
        print(f"Experiment inputs ready: installed {installed} files; retained {skipped} matching files.")
        return 0
    except (DataReleaseError, OSError) as exc:
        print(f"Data setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
