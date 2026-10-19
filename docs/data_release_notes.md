# Prepared experiment inputs v1

This release supplies the additional prepared workload and energy inputs for the LLM contract-design experiments. The three inputs required for the default case remain in the source repository.

The archive includes the other daily workloads, planning-horizon windows, participating-type and immediate-service-share variants, computational-time inputs, and preparation source tables. It contains no optimizer results or raw upstream downloads.

From a matching repository checkout, install and verify the data with:

```sh
python scripts/download_data.py
python scripts/download_data.py --check
```

The downloader verifies the archive and individual input files against `data/dataset_manifest.json`, preserves matching files, and refuses to replace modified inputs unless explicitly requested with `--force`.

Attach `experiment-data-v1.zip` and `SHA256SUMS` to the `data-v1` release. Keep the published assets unchanged so this repository revision remains reproducible. Workload and energy scenarios are prepared research inputs; upstream data retain their original licenses and terms.
