# Input visualization scripts

These scripts visualize prepared workload and power inputs without solving the contract optimization problem. Generated figures, tables, and visualization metadata are omitted and ignored.

These scripts need the additional experiment data archive; the three inputs retained in Git cover only the default optimization case. Install the project with its data dependencies, download the prepared inputs, then run from the repository root:

```powershell
python -m pip install -e ".[data]"
python scripts/download_data.py
python data_visualization/plot_scenarios.py
python data_visualization/plot_planning_horizon_inputs.py
```

The configured `data-v1` GitHub Release must be published with its asset before the download is available. Maintainers can instead supply a locally built archive with `python scripts/download_data.py --archive release_assets/experiment-data-v1.zip`. See the [data release guide](../docs/data_release.md) for details.

`plot_scenarios.py` reads the twelve default workload recipes, their referenced request table, `data/energy_scenarios.csv`, and processing capacity from `config/default.yaml`. It aggregates copy counts and tokens by slot without expanding sampled requests. It checks source hashes, request identities, and recipe totals, and compares active inputs with selection metrics when available.

It creates six PNG/PDF figures: representative workloads, a workload heatmap, all workload profiles, workload stress statistics, representative power/price profiles, and power/price heatmaps. It also writes aggregated CSV tables, `visualization_manifest.json`, and an output-specific `README.md`. The default output directory is `output/input_visualizations/`, which Git ignores. Use `--output-dir PATH` to choose another location.

The daily plotting script requires NumPy, pandas, Matplotlib, and PyYAML. It uses Times New Roman when available and falls back to DejaVu Serif. Pass `--font-file PATH_TO_FONT.ttf` to supply a font. Other options include `--data-dir` and `--config`; daily plots require 96 slots of 900 seconds.

`plot_planning_horizon_inputs.py` reads `config/planning_horizons/H168.yaml` and the horizon manifest. It plots the shared seven-day workload arrival rate and capacity, prescribed power components, and day-ahead price, marking nested horizon starts. PNG/PDF outputs are written to ignored `output/input_visualizations/planning_horizons/`. Use `--config`, `--manifest`, and `--output-dir` to change inputs or outputs. It uses Times New Roman when available and otherwise falls back to DejaVu Serif.
