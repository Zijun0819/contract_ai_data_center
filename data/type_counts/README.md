# Participating-type-count inputs

This package's prepared inputs are distributed in the GitHub Release archive. Before using the commands below, run `python scripts/download_data.py` from the repository root. Documentation and manifests are retained in Git; downloaded inputs are ignored. See the [data release guide](../../docs/data_release.md) if the release has not yet been published or you have a local archive.

These cases use 2, 4, 6, 8, or 10 participating types plus a separate immediate-service type 0. Configurations are in `config/type_counts/`. All retain workload `R05-D333`, power profile `D02`, 96 fifteen-minute slots, and the same request sizes, arrivals, sampled copy counts, and immediate-service copy identities.

For N participating types, sensitivities are evenly spaced:

```text
lambda_i = 0.1 + (i - 1) * (2.5 - 0.1) / (N - 1), i = 1, ..., N
```

Participating labels are obtained by merging or splitting the original eight ordered probability bins using interval overlap on [0, 1]. Balanced integer allocation preserves each source request's occurrence count and scenario-level quotas. Type 0 remains exactly 50% of sampled task occurrences. Participating workload shares are nonuniform; changing N also changes the discrete sensitivity mixture. N=8 preserves the original labels but uses an evenly spaced sensitivity grid, which differs from the default grid.

- `Nxx/tasks.csv.gz` and `Nxx/R05-D333.sample.json`: matching request tables and fixed-label recipes.
- `energy_scenarios.csv`: unchanged power inputs shared by the cases.
- `type_distributions.csv`: target/actual task and token shares, sensitivities, and input capacity screens.
- `experiment_manifest.json`: frozen controls, case paths, and file hashes.

`N02/` additionally contains converted recipes and request tables for all twelve workload scenarios. Original types 1-4 and 5-8 merge into two participating types while retaining each request's copy count and immediate copies. `workload_type_distributions.csv` and `workload_scenario_manifest.json` document these inputs for workload/power sweeps; they do not change the five-case fixed-workload experiment.

After installation, run from the repository root:

```powershell
llm-contract-opt compare-type-counts --validate-only
llm-contract-opt compare-type-counts
llm-contract-opt compare-type-counts --type-counts 2 6 10 --no-group-schedule
llm-contract-opt compare --config config/type_counts/N06.yaml
llm-contract-opt compare --config config/type_counts/N02.yaml --profit-sweeps
```

New outputs are written under `results/type_counts/` and are omitted and ignored. Comparisons include both fixed-policy benchmarks, financial metrics, type distributions, runtime, and solver status/gap information. These are paired sensitivity cases on one workload/power pair rather than independent population estimates.

To prepare replacement inputs after changing the anchor, choose separate destinations:

```powershell
llm-contract-opt prepare-type-counts --config config/default.yaml --data-dir data/type_counts_new --config-dir config/type_counts_new
llm-contract-opt compare-type-counts --manifest data/type_counts_new/experiment_manifest.json
```

The anchor must be a validated 24-hour fixed-label recipe with 50% immediate task share. Editing only the sensitivity-list length is insufficient: source probability columns, fixed labels, and configurations must agree. `python prepare_two_type_scenarios.py` can prepare or validate the additional N02 workload recipes.
