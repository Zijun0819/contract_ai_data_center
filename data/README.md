# Prepared input data

This directory holds prepared optimizer inputs. Git includes three default-case inputs (about 29 MiB), Markdown documentation, and provenance/experiment manifests. The remaining prepared inputs are excluded from Git and distributed as the versioned `experiment-data-v1.zip` GitHub Release asset. Solver results, generated figures, and historical optimization checks are omitted from both.

## Download additional inputs

Run from the repository root:

```sh
python scripts/download_data.py
python scripts/download_data.py --check
```

The download requires the maintainer to publish the configured `data-v1` release and upload its asset first. A locally built archive can be used with `python scripts/download_data.py --archive release_assets/experiment-data-v1.zip`. See the [release guide](../docs/data_release.md) for build, publication, and verification instructions.

Downloaded inputs remain on disk under `data/` and are ignored by Git. The tracked `dataset_manifest.json` records file sizes and SHA-256 checksums for distribution checks. Existing experiment manifests retain their model-specific provenance and validation checks. `.gitattributes` preserves data-file bytes across checkouts so hashes remain valid on Windows and other platforms.

## Default scenarios

Git retains `R05-D333.sample.json`, `representative_tasks_eight_types.csv.gz`, and `energy_scenarios.csv`. They are sufficient for default `validate`, `solve`, and `compare` commands. The other files listed below arrive with the archive:

- `R01-D308.sample.json` through `R12-D320.sample.json`: twelve workload recipes selected from the BurstGPT v2.0 `BurstGPT_3.csv` trace. Each records exact integer request-copy counts and fixed task-type labels.
- `representative_tasks_eight_types.csv.gz`: shared request-level source referenced by the default recipes. Requests retain their arrival slots and input/output token sizes. Copies are represented by recipes rather than expanded into individual rows.
- `representative_tasks_raw.csv.gz`: selected source requests with the four-group preparation weights used before splitting into eight participating types.
- `energy_scenarios.csv`: twelve prepared power/price scenarios, with 96 fifteen-minute slots per scenario.
- `representative_slot_inputs.csv`: slot-level calibration inputs; actual sampled arrivals come from recipes and their referenced request table.
- `workload_selection.csv`, `workload_candidate_metrics.csv`, and `workload_selection_manifest.json`: input-selection metrics, criteria, and source hashes. These describe workload preparation rather than optimization outcomes.

The default case is workload `R05-D333` and power profile `D02`, selected in `config/default.yaml`. It uses eight participating types plus immediate-service type 0, with 50% immediate service measured by sampled task occurrences. Selection uses density, burst intensity, and necessary capacity screens; the twelve cases are selected stress scenarios. Passing an input screen does not certify a feasible or optimal menu.

Request timing retains the circular -900-second alignment within each source day. Task preference weights and sensitivities are assumed model inputs. Energy generation and battery-discharge trajectories are prescribed inputs.

## Experiment packages

| Folder | Prepared variation |
|---|---|
| [planning_horizons](planning_horizons/README.md) | Consecutive, nested 12-168 hour windows |
| [type_counts](type_counts/README.md) | 2, 4, 6, 8, or 10 participating types |
| [immediate_shares](immediate_shares/README.md) | 10%, 20%, 30%, 40%, or 50% immediate task share |
| `computational_time/` | Joint horizon/type-count grid for timing experiments |

Manifests record relative file paths, frozen settings, and checksums. Keep recipes, source tables, configurations, and their manifest hashes together. A deliberate input change requires regenerating its experiment package.

After installing the project, run from the repository root. Default validation works immediately; download the archive before the experiment commands:

```powershell
llm-contract-opt validate --config config/default.yaml
python scripts/download_data.py
llm-contract-opt compare-horizons --validate-only
llm-contract-opt compare-type-counts --validate-only
llm-contract-opt compare-immediate-shares --validate-only
llm-contract-opt compare-computational-time --validate-only
```

The archive also supplies the other daily workloads, profit-sweep inputs, optional preparation inputs, and input-visualization data. The full test suite requires it. The complete BurstGPT download and original all-days power source are not required for the prepared experiments and are not included in this release. Source-release links and original hashes are retained in the selection manifest. `reselect_workload_scenarios.py --input PATH` can prepare replacement workload scenarios from an original trace; it replaces active inputs and creates a backup. Use a separate copy when preparing a new dataset.
