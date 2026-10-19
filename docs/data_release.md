# Data distribution and releases

Git retains three inputs needed for the default `R05-D333` / `D02` case (about 29 MiB):

- `data/energy_scenarios.csv`
- `data/R05-D333.sample.json`
- `data/representative_tasks_eight_types.csv.gz`

Markdown documentation and provenance/experiment manifests also remain in Git. Additional prepared inputs are distributed in `experiment-data-v1.zip`, a GitHub Release asset. The complete prepared dataset is about 260 MiB on disk. Downloaded inputs and local release assets are ignored by Git; results and intermediate outputs are excluded from the archive.

Default validation, solving, comparison, and the default `representative-day` plot work from a clone without downloading the archive. Other workloads (including `R02-D246`), profit sweeps, experiment sweeps, optional preparation/input-visualization workflows, and the full test suite require the additional inputs. Rebuilding from original raw sources also requires external BurstGPT and CAISO-derived files, which are not distributed here.

## Use the data

The configured release tag is `data-v1`, with this pinned asset URL:

[`experiment-data-v1.zip`](https://github.com/Zijun0819/contract_ai_data_center/releases/download/data-v1/experiment-data-v1.zip)

The maintainer must publish that release and upload its assets before the URL is available. A missing release asset does not affect default-case use.

After the release is published, a fresh Windows clone can be checked as follows:

```sh
git clone https://github.com/Zijun0819/contract_ai_data_center.git
cd contract_ai_data_center
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e .
./.venv/Scripts/python.exe -m llm_contract_optimizer validate
./.venv/Scripts/python.exe scripts/download_data.py
./.venv/Scripts/python.exe scripts/download_data.py --check
./.venv/Scripts/python.exe -m llm_contract_optimizer compare-horizons --validate-only
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

On macOS/Linux, use `.venv/bin/python` in place of `./.venv/Scripts/python.exe`. The remaining commands assume an active Python environment and run from the repository root.

The downloader verifies the archive against the tracked `data/dataset_manifest.json` before installing it. `--check` verifies all default and additional inputs against their recorded sizes and SHA-256 checksums without downloading. The existing optimization commands continue to apply their own recipe and experiment consistency checks.

For a local archive or another destination:

```sh
python scripts/download_data.py --archive release_assets/experiment-data-v1.zip
python scripts/download_data.py --archive release_assets/experiment-data-v1.zip --destination E:/path/to/another/contract_ai_data_center
```

`--destination` selects a repository root, with inputs installed beneath its `data/` directory. Use `--force` only when deliberately replacing conflicting local dataset files. Keep custom data in separate locations.

## Build and publish a release

1. Start with the complete prepared inputs locally under `data/`. Keep the matching configurations and input provenance manifests. Build the distribution:

   ```sh
   python scripts/build_data_release.py
   python scripts/download_data.py --check
   ```

   The builder creates ignored `release_assets/experiment-data-v1.zip` and `release_assets/SHA256SUMS`, and updates tracked `data/dataset_manifest.json`. It leaves the original inputs in place. The distribution manifest records the asset location, archive checksum, and per-file checksums.

2. Verify that a clean copy of the tracked repository passes default validation, then install the local archive there and run the full checks shown above. Review and commit the code, documentation, ignore rules, byte-preserving attributes, and distribution manifest. Do not force-add the archive or ignored experiment inputs to Git.

3. Push the reviewed repository changes. On the repository's GitHub page, open **Releases**, choose **Draft a new release**, and create tag `data-v1` at the corresponding commit. Attach both `release_assets/experiment-data-v1.zip` and `release_assets/SHA256SUMS`, then publish the release. This workflow needs no GitHub CLI.

   The [prepared release notes](data_release_notes.md) can be used as the release description.

4. Test the published download from a fresh clone with `python scripts/download_data.py`, followed by `python scripts/download_data.py --check` and experiment validation. Retain the local build assets until publication is verified.

`.gitattributes` disables automatic text conversion for data inputs and configurations so Git checkouts preserve the exact bytes used in checksums, including CSV, JSON, and YAML inputs on Windows. Preserve these attributes when transferring or updating the dataset.

For a changed dataset, create a new release version and rebuild its matching distribution manifest; keep published versioned assets stable so earlier repository revisions remain reproducible.
