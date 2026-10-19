# Immediate-service-share inputs

This package's prepared inputs are distributed in the GitHub Release archive. Before using the commands below, run `python scripts/download_data.py` from the repository root. Documentation and manifests are retained in Git; downloaded inputs are ignored. See the [data release guide](../../docs/data_release.md) if the release has not yet been published or you have a local archive.

These cases use 10%, 20%, 30%, 40%, or 50% immediate service, measured by sampled task occurrences. Configurations are in `config/immediate_shares/`. Actual immediate-token shares differ because request sizes vary.

All cases retain default workload `R05-D333`, power profile `D02`, 24-hour horizon, source requests, sampled copy counts, sizes, arrival times, tariffs, capacity, energy settings, and solver controls. The eight participating sensitivities remain 0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, and 2.5.

The 50% case preserves the default fixed-label table. Other cases calibrate source-level immediate probabilities and use balanced integer assignment to preserve each request's occurrence count. Immediate-task quotas change while conditional participating-type proportions remain fixed, subject to integer rounding. The preparation seed is shared across cases.

- `source_tasks.csv.gz`: common request-level source table.
- `energy_scenarios.csv`: unchanged power inputs.
- `Sxx/R05-D333.sample.json`: each case's fixed-label recipe.
- `type_distributions.csv`: target/actual task shares, token shares, and input-load statistics.
- `experiment_manifest.json`: frozen settings, case paths, source hashes, and input/configuration checksums.

After installation, run from the repository root:

```powershell
llm-contract-opt compare-immediate-shares --validate-only
llm-contract-opt compare-immediate-shares
llm-contract-opt compare-immediate-shares --shares 10 30 50 --no-group-schedule
```

Each run compares the proposed method and both fixed-policy benchmarks and writes timestamped outputs under `results/immediate_shares/`. Financial metrics include profit, served-work discount expenditure, and day-ahead electricity cost, together with hourly values and admission/status diagnostics. Generated results are omitted and ignored.

The proposed method admits immediate work up to same-slot capacity, dropping unavoidable immediate overflow pro rata and retaining flexible work for scheduling. Benchmarks apply their own admission and scheduling policies. Inspect dropped work and solver certificates alongside financial metrics. These are paired sensitivity cases on one workload/power pair.

To prepare replacement inputs after changing the anchor, choose new destinations:

```powershell
llm-contract-opt prepare-immediate-shares --config config/default.yaml --data-dir data/immediate_shares_new --config-dir config/immediate_shares_new
llm-contract-opt compare-immediate-shares --manifest data/immediate_shares_new/experiment_manifest.json
```

The anchor must have a validated 50% fixed-label recipe and 24-hour horizon. Editing only `immediate_execution_share` does not relabel tasks; matching prepared recipes are required.
