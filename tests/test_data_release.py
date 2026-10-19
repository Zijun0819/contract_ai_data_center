"""Small, offline checks for the experiment-data release and restoration tools."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.build_data_release import build_release
from scripts.download_data import main as download_main
from scripts.data_release_common import (
    DataReleaseError,
    check_dataset,
    install_archive,
    load_manifest,
    sha256_file,
    verify_archive,
)


class DataReleaseTests(unittest.TestCase):
    """Use a few bytes per input; real research inputs are never loaded."""

    DEFAULTS = {
        "data/energy_scenarios.csv": b"slot,price\n0,1\n",
        "data/R05-D333.sample.json": b'{"recipe": "default"}\n',
        "data/representative_tasks_eight_types.csv.gz": b"small default fixture\n",
    }
    EXPERIMENTS = {
        "data/R02-D246.sample.json": b'{"recipe": "experiment"}\n',
        "data/planning_horizons/window.csv.gz": b"small experiment fixture\n",
    }

    def setUp(self):
        # Keep replacement operations within the writable workspace on Windows.
        temporary_root = PROJECT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="data-release-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        for name, contents in {**self.DEFAULTS, **self.EXPERIMENTS}.items():
            self.write(self.source / name, contents)
        self.write(self.source / "data/README.md", b"Input documentation\n")
        self.write(self.source / "data/planning_horizons/experiment_manifest.json", b"{}\n")
        self.manifest = build_release(self.source)
        self.archive = self.source / "release_assets" / self.manifest["archive"]["name"]
        self.destination = self.base / "clone"
        for name, contents in self.DEFAULTS.items():
            self.write(self.destination / name, contents)

    @staticmethod
    def write(path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    @staticmethod
    def snapshot(root):
        if not root.exists():
            return {}
        return {path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    def rewrite_archive(self, entries):
        """Give tampered ZIPs valid outer hashes to reach member validation."""
        archive = self.base / "tampered.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for name, contents in entries:
                    package.writestr(name, contents)
        manifest = deepcopy(self.manifest)
        manifest["archive"]["sha256"] = sha256_file(archive)
        manifest["archive"]["size"] = archive.stat().st_size
        return archive, manifest

    def test_builder_excludes_defaults_and_documentation_and_is_deterministic(self):
        with zipfile.ZipFile(self.archive) as package:
            self.assertEqual(package.namelist(), sorted(self.EXPERIMENTS))
            for name, contents in self.EXPERIMENTS.items():
                self.assertEqual(package.read(name), contents)
        self.assertEqual({item["path"] for item in self.manifest["default_files"]}, set(self.DEFAULTS))
        self.assertEqual({item["path"] for item in self.manifest["files"]}, set(self.EXPERIMENTS))
        second_output = self.base / "second-release"
        second = build_release(self.source, output_dir=second_output)
        self.assertEqual(self.archive.read_bytes(),
                         (second_output / second["archive"]["name"]).read_bytes())
        self.assertEqual(self.manifest, second)
        saved = load_manifest(self.source / "data/dataset_manifest.json")
        self.assertEqual(saved, second)

    def test_local_restore_check_and_repeat_skip_unchanged_inputs(self):
        self.assertTrue(check_dataset(self.manifest, self.destination))
        verify_archive(self.archive, self.manifest)
        installed, skipped = install_archive(self.archive, self.manifest, self.destination)
        self.assertEqual((installed, skipped), (len(self.EXPERIMENTS), 0))
        self.assertEqual(check_dataset(self.manifest, self.destination), [])
        self.assertEqual(self.snapshot(self.destination), {**self.DEFAULTS, **self.EXPERIMENTS})
        self.assertEqual(install_archive(self.archive, self.manifest, self.destination),
                         (0, len(self.EXPERIMENTS)))
        changed = next(iter(self.EXPERIMENTS))
        (self.destination / changed).write_bytes(b"corrupt")
        issues = check_dataset(self.manifest, self.destination)
        self.assertTrue(any(changed in message for message in issues), issues)

    def test_wrong_archive_checksum_causes_no_writes(self):
        manifest = deepcopy(self.manifest)
        manifest["archive"]["sha256"] = "0" * 64
        before = self.snapshot(self.destination)
        with self.assertRaises(DataReleaseError):
            install_archive(self.archive, manifest, self.destination)
        self.assertEqual(self.snapshot(self.destination), before)

    def test_download_command_restores_local_archive_without_network_access(self):
        arguments = ["--manifest", str(self.source / "data/dataset_manifest.json"),
                     "--destination", str(self.destination)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), patch(
            "scripts.download_data.urlopen", side_effect=AssertionError("Local restoration must remain offline")
        ):
            self.assertEqual(download_main([*arguments, "--check"]), 1)
            self.assertEqual(download_main([*arguments, "--archive", str(self.archive)]), 0)
            self.assertEqual(download_main([*arguments, "--check"]), 0)
        self.assertIn("Run python scripts/download_data.py", stderr.getvalue())
        self.assertIn("Experiment inputs ready", stdout.getvalue())
        self.assertEqual(check_dataset(self.manifest, self.destination), [])

    def test_builder_refuses_default_only_data_in_a_clean_clone(self):
        for name in self.EXPERIMENTS:
            (self.source / name).unlink()
        before = self.archive.read_bytes()
        with self.assertRaises(DataReleaseError):
            build_release(self.source)
        self.assertEqual(self.archive.read_bytes(), before)

    def test_builder_omits_generated_output_files_and_audits(self):
        generated = (
            "data/workload_validation.json", "data/type_counts/verification.json",
            "data/selection_audit.json", "data/run_results.json", "data/progress.log",
            "data/scratch.tmp", "data/generated.pyc",
        )
        for name in generated:
            self.write(self.source / name, b"generated output, not a source input\n")
        before = self.archive.read_bytes()
        manifest = build_release(self.source)
        self.assertEqual({record["path"] for record in manifest["files"]}, set(self.EXPERIMENTS))
        self.assertEqual(self.archive.read_bytes(), before)

    def test_builder_refuses_generated_output_or_cache_directories(self):
        before = self.archive.read_bytes()
        for directory in ("results", "output", "outputs", "tmp", "__pycache__", ".cache"):
            with self.subTest(directory=directory):
                path = self.source / "data" / directory / "generated.csv"
                self.write(path, b"generated output\n")
                with self.assertRaises(DataReleaseError):
                    build_release(self.source)
                self.assertEqual(self.archive.read_bytes(), before)
                path.unlink()
                path.parent.rmdir()

    def test_builder_refuses_missing_input_declared_by_experiment_manifest(self):
        metadata = self.source / "data/planning_horizons/experiment_manifest.json"
        metadata.write_text(json.dumps({"files": {"missing.csv": "0" * 64}}), encoding="utf-8")
        before = self.archive.read_bytes()
        with self.assertRaises(DataReleaseError):
            build_release(self.source)
        self.assertEqual(self.archive.read_bytes(), before)

    def test_incorrect_member_bytes_cause_no_partial_restoration(self):
        entries = list(self.EXPERIMENTS.items())
        entries[-1] = (entries[-1][0], b"altered payload")
        archive, manifest = self.rewrite_archive(entries)
        before = self.snapshot(self.destination)
        with self.assertRaises(DataReleaseError):
            install_archive(archive, manifest, self.destination)
        self.assertEqual(self.snapshot(self.destination), before)

    def test_unsafe_and_unexpected_archive_members_are_rejected(self):
        unsafe_names = (
            "../outside.csv", "data/../../outside.csv", "/outside.csv",
            "C:/outside.csv", "data\\outside.csv", "data/unexpected.csv",
        )
        before = self.snapshot(self.destination)
        for name in unsafe_names:
            with self.subTest(name=name):
                entries = list(self.EXPERIMENTS.items())
                entries[0] = (name, entries[0][1])
                archive, manifest = self.rewrite_archive(entries)
                with self.assertRaises(DataReleaseError):
                    install_archive(archive, manifest, self.destination)
                self.assertEqual(self.snapshot(self.destination), before)
                self.assertFalse((self.base / "outside.csv").exists())

    def test_symbolic_link_and_duplicate_members_are_rejected(self):
        name, contents = next(iter(self.EXPERIMENTS.items()))
        symlink = zipfile.ZipInfo(name)
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        others = [(path, data) for path, data in self.EXPERIMENTS.items() if path != name]
        before = self.snapshot(self.destination)
        variants = (
            [(symlink, contents), *others],
            [(name, contents), (name, contents)],
            [(name, contents), ("data/" + name[5:].upper(), contents)],
        )
        for entries in variants:
            with self.subTest(entries=[str(entry[0]) for entry in entries]):
                archive, manifest = self.rewrite_archive(entries)
                with self.assertRaises(DataReleaseError):
                    install_archive(archive, manifest, self.destination)
                self.assertEqual(self.snapshot(self.destination), before)

    def test_modified_local_input_is_preserved_unless_force_is_requested(self):
        name = next(iter(self.EXPERIMENTS))
        self.write(self.destination / name, b"my local changes\n")
        before = self.snapshot(self.destination)
        with self.assertRaises(DataReleaseError):
            install_archive(self.archive, self.manifest, self.destination)
        self.assertEqual(self.snapshot(self.destination), before)
        installed, skipped = install_archive(self.archive, self.manifest, self.destination, force=True)
        self.assertEqual((installed, skipped), (len(self.EXPERIMENTS), 0))
        self.assertEqual(self.snapshot(self.destination), {**self.DEFAULTS, **self.EXPERIMENTS})

    def test_manifest_rejects_paths_outside_data_directory(self):
        for invalid in ("../outside.csv", "config/default.yaml", "data/../outside.csv"):
            with self.subTest(path=invalid):
                manifest = deepcopy(self.manifest)
                manifest["files"][0]["path"] = invalid
                path = self.base / "invalid_manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(DataReleaseError):
                    load_manifest(path)


if __name__ == "__main__":
    unittest.main()
