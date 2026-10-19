# Planning-horizon inputs

This package's prepared inputs are distributed in the GitHub Release archive. Before using the commands below, run `python scripts/download_data.py` from the repository root. Documentation and manifests are retained in Git; downloaded inputs are ignored. See the [data release guide](../../docs/data_release.md) if the release has not yet been published or you have a local archive.

This package supplies 12, 24, 48, 72, 96, 120, 144, and 168 hour cases, with matching configurations in `config/planning_horizons/`. All use fifteen-minute slots and share default workload/power anchor `R05-D333` / `D02`.

Cases are nested suffixes of consecutive workload trace days D327-D333 and CAISO-derived power days January 15-21, 2025. The 24-hour case reproduces the default anchor; the 12-hour case uses its last half-day. Longer cases prepend preceding days. Workload trace days and power dates are paired by relative day and clock slot; they are not asserted to be contemporaneous observations.

Complete source days are sampled and labeled once. Overlapping windows preserve request identities, token sizes, copy counts, and labels. Complete days target 50% immediate service by task count; cropping a day inherits its realized labels. The circular -900-second daily workload alignment is retained.

- `daily/`: seven daily sampling recipes and their shared request table.
- `Hxxx/workload.window.json`: daily recipe references and selected window.
- `Hxxx/energy_scenarios.csv`: matching power and price inputs.
- `consecutive_power_source.csv`: shared prepared seven-day power trajectory.
- `horizon_data_summary.csv`: input workload, type-share, and power summaries.
- `experiment_manifest.json`: case paths, frozen non-horizon settings, source provenance, and input/configuration hashes.

Initial backlog is empty, all admitted work must finish within the horizon, and work can cross midnight in a multi-day window. Menu candidates and the flat benchmark's nominal 24-hour deferral remain fixed; execution deadlines are clipped at the horizon end.

After installation, run from the repository root:

```powershell
llm-contract-opt compare-horizons --validate-only
llm-contract-opt compare-horizons
llm-contract-opt compare-horizons --horizons 12 24 168 --time-limit 600 --no-group-schedule
llm-contract-opt compare --config config/planning_horizons/H048.yaml
```

The runner compares the proposed method with both fixed-policy benchmarks and writes new timestamped outputs under `results/planning_horizons/`. Generated results are omitted and ignored. Windows overlap and vary in realized workload and power conditions; they describe this selected trajectory rather than independent replications or a rolling-horizon forecast study.

To visualize the input timeline without solving:

```powershell
python data_visualization/plot_planning_horizon_inputs.py
```

Rebuilding requires the original BurstGPT trace, the external all-days power CSV, and data dependencies. Use `prepare_planning_horizon_data.py --help` for required paths, and choose new data/configuration directories when preparing a replacement package. Editing only the horizon field does not produce matching workload and power windows.
