"""Plot energy and LLM arrival scenarios without expanding sampled task copies.

Run: python data_visualization/plot_scenarios.py
Requires numpy, pandas, matplotlib and PyYAML. Inputs are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
SLOTS = np.arange(1, 97)
TICKS = [1, 16, 32, 48, 64, 80, 96]
COLORS = ["#23749B", "#D07826", "#77589B"]
ENERGY_COLS = ["ppa_kwh", "gt_kwh", "pv_kwh", "bess_discharge_kwh"]
SAVED_FIGURES = {}


def configure_style(font_file: Path | None) -> str:
    if font_file:
        font_manager.fontManager.addfont(str(font_file))
        family = font_manager.FontProperties(fname=str(font_file)).get_name()
    else:
        windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
            if (windows_fonts / name).exists():
                font_manager.fontManager.addfont(str(windows_fonts / name))
        try:
            font_manager.findfont("Times New Roman", fallback_to_default=False)
            family = "Times New Roman"
        except ValueError:
            family = "DejaVu Serif"
            print("Times New Roman is unavailable; using bundled DejaVu Serif.", flush=True)
    resolved = font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": family, "font.size": 16,
        "axes.labelsize": 18, "axes.titlesize": 19, "figure.titlesize": 22,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })
    return resolved


def check_grid(frame: pd.DataFrame, scenario: str, slot: str) -> None:
    if frame.duplicated([scenario, slot]).any():
        raise ValueError(f"Duplicate scenario/slot rows: {scenario}")
    for name, group in frame.groupby(scenario):
        if not np.array_equal(np.sort(group[slot].to_numpy()), SLOTS):
            raise ValueError(f"{name}: expected exactly slots 1..96")


def load_workloads(data: Path, slot_seconds: int):
    source_cache, hash_cache, aggregates, provenance = {}, {}, [], []
    for path in sorted(data.glob("*.sample.json")):
        recipe = json.loads(path.read_text(encoding="utf-8"))
        if recipe["schema"] != "individual_task_sampling_v1":
            raise ValueError(f"Unsupported recipe schema: {path.name}")
        source_path = (path.parent / recipe["source_file"]).resolve()
        if source_path not in source_cache:
            hash_cache[source_path] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            source_cache[source_path] = pd.read_csv(source_path, usecols=[
                "scenario_id", "task_id", "arrival_slot", "input_tokens", "output_tokens", "total_tokens"
            ])
        if hash_cache[source_path] != recipe["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {path.name}")
        scenario = recipe["scenario_id"]
        source = source_cache[source_path]
        group = source.loc[source.scenario_id.eq(scenario)].copy()
        draws = pd.DataFrame(recipe["draws"], columns=["task_id", "copies"])
        if group.task_id.duplicated().any() or draws.task_id.duplicated().any():
            raise ValueError(f"Duplicate task IDs: {scenario}")
        if set(group.task_id) != set(draws.task_id):
            raise ValueError(f"Recipe and source task IDs disagree: {scenario}")
        group = group.merge(draws, on="task_id", validate="one_to_one")
        if (group.copies < 1).any() or not pd.api.types.is_integer_dtype(group.copies):
            raise ValueError(f"Invalid copy counts: {scenario}")
        if not group.arrival_slot.isin(SLOTS).all():
            raise ValueError(f"Arrival outside 96-slot day: {scenario}")
        if not (group.input_tokens + group.output_tokens).equals(group.total_tokens):
            raise ValueError(f"Input/output token identity failed: {scenario}")
        group["sampled_tokens"] = group.total_tokens * group.copies
        aggregated = group.groupby("arrival_slot").agg(
            total_tokens=("sampled_tokens", "sum"), task_count=("copies", "sum")
        ).reindex(SLOTS, fill_value=0).rename_axis("slot").reset_index()
        if int(aggregated.total_tokens.sum()) != recipe["target_tokens"]:
            raise ValueError(f"Sampled token total mismatch: {scenario}")
        if int(aggregated.task_count.sum()) != recipe["task_count"]:
            raise ValueError(f"Sampled task count mismatch: {scenario}")
        aggregated["scenario_id"] = scenario
        aggregated["million_tokens_per_s"] = aggregated.total_tokens / slot_seconds / 1e6
        aggregates.append(aggregated)
        provenance.append({"scenario_id": scenario, "recipe": path.name,
                           "source": source_path.name, "source_sha256": hash_cache[source_path]})
    if not aggregates:
        raise ValueError("No sampled workload recipes found")
    workload = pd.concat(aggregates, ignore_index=True)
    check_grid(workload, "scenario_id", "slot")
    return workload, provenance


def save(fig, output: Path, stem: str):
    fig.canvas.draw()
    for label in fig.findobj(match=Text):
        if label.get_text() and label.get_visible():
            if label.get_fontsize() < 16:
                raise ValueError(f"Font smaller than 16 pt: {label.get_text()}")
            if font_manager.FontProperties(fname=font_manager.findfont(
                label.get_fontproperties(), fallback_to_default=False
            )).get_name() != plt.rcParams["font.family"][0]:
                raise ValueError(f"Unexpected font: {label.get_text()}")
    for extension in ("png", "pdf"):
        target = output / f"{stem}.{extension}"
        temporary = output / f".render-{stem}-{uuid.uuid4().hex}.{extension}"
        try:
            fig.savefig(temporary, dpi=300, bbox_inches="tight", pad_inches=0.15)
            try:
                temporary.replace(target)
            except PermissionError:
                # A Windows PDF viewer can deny replacement of an open file.
                # Keep the complete refreshed artifact in the requested folder.
                target = output / f"{stem}_updated.{extension}"
                temporary.replace(target)
                print(f"Open file could not be replaced; refreshed figure saved as {target.name}", flush=True)
            SAVED_FIGURES.setdefault(stem, {})[extension] = target.name
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def slot_axis(ax):
    ax.set_xlim(1, 96)
    ax.set_xticks(TICKS)
    ax.set_xlabel("Slot (15 minutes)")
    ax.grid(axis="y", color="#DFE3E6", linewidth=0.7)
    ax.set_axisbelow(True)


def workload_summary(workload, capacity_million_tokens_per_s):
    summary = workload.groupby("scenario_id").agg(
        daily_tokens=("total_tokens", "sum"), daily_tasks=("task_count", "sum"),
        mean_million_tokens_per_s=("million_tokens_per_s", "mean"),
        peak_million_tokens_per_s=("million_tokens_per_s", "max"),
        std_million_tokens_per_s=("million_tokens_per_s", "std"),
    )
    summary["coefficient_of_variation"] = summary.std_million_tokens_per_s / summary.mean_million_tokens_per_s
    capacity_tokens = capacity_million_tokens_per_s * 1e6 * 900
    augmented = workload.assign(
        overloaded=workload.total_tokens > capacity_tokens,
        dense=workload.total_tokens >= .8 * capacity_tokens,
        overflow_tokens=np.maximum(workload.total_tokens - capacity_tokens, 0),
    )
    stress = augmented.groupby("scenario_id").agg(
        overload_slots=("overloaded", "sum"), dense_slots=("dense", "sum"),
        immediate_service_overflow_tokens=("overflow_tokens", "sum"),
    )
    summary = summary.join(stress)
    summary["immediate_service_overflow_fraction"] = summary.immediate_service_overflow_tokens / summary.daily_tokens
    summary["overload_hours"] = summary.overload_slots / 4
    summary["peak_to_capacity_ratio"] = summary.peak_million_tokens_per_s / capacity_million_tokens_per_s
    for scenario, group in augmented.groupby("scenario_id"):
        over = group.sort_values("slot").overloaded.to_numpy()
        runs = np.split(over, np.flatnonzero(np.diff(over)) + 1)
        summary.loc[scenario, "longest_overload_run_slots"] = max((len(run) for run in runs if run[0]), default=0)
        summary.loc[scenario, "top_four_slots_token_share"] = group.total_tokens.nlargest(4).sum() / group.total_tokens.sum()
    # Show the strongest sustained profile alongside both user-specified anchors.
    dense = summary.sort_values(
        ["overload_slots", "immediate_service_overflow_fraction"], ascending=False, kind="stable"
    ).index[0]
    sharp = "R05-D333" if "R05-D333" in summary.index else summary.drop(index=dense).peak_million_tokens_per_s.idxmax()
    extended = "R11-D228" if "R11-D228" in summary.index else summary.drop(index=[dense, sharp]).immediate_service_overflow_fraction.idxmax()
    if dense in (sharp, extended):
        dense = summary.drop(index=[sharp, extended]).sort_values(
            ["overload_slots", "immediate_service_overflow_fraction"], ascending=False, kind="stable"
        ).index[0]
    selected = [(dense, "Sustained overload"), (sharp, "Sharp burst"), (extended, "Extended burst")]
    summary["selection_reason"] = ""
    summary["representative_panel_reason"] = ""
    for scenario, reason in selected:
        summary.loc[scenario, "representative_panel_reason"] = reason
    return summary, selected


def plot_workloads(workload, selected, output, capacity_million_tokens_per_s):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey="row", layout="constrained")
    fig.suptitle("Dense and bursty LLM workload arrivals: three representative scenarios")
    for column, ((scenario, reason), color) in enumerate(zip(selected, COLORS)):
        group = workload.loc[workload.scenario_id.eq(scenario)].sort_values("slot")
        top, bottom = axes[:, column]
        top.plot(group.slot, group.million_tokens_per_s, color=color)
        top.fill_between(group.slot, group.million_tokens_per_s, color=color, alpha=0.12)
        top.fill_between(group.slot, capacity_million_tokens_per_s, group.million_tokens_per_s,
                         where=group.million_tokens_per_s > capacity_million_tokens_per_s,
                         color="#C8463D", alpha=0.22)
        capacity_line = top.axhline(
            capacity_million_tokens_per_s, color="red", linestyle="--", linewidth=2,
            label=f"AI data center capacity: {capacity_million_tokens_per_s:g} million tokens/s",
        )
        top.set_title(f"{scenario}\n{reason}")
        bottom.bar(group.slot, group.task_count / 1000, width=0.9, color=color, alpha=0.85)
        for ax in (top, bottom):
            slot_axis(ax)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        top.set_xlabel("")
        bottom.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0, 0].set_ylabel("Workload (million tokens/s)")
    axes[1, 0].set_ylabel("LLM tasks (thousands/slot)")
    # Set shared limits only after all panels are populated; otherwise the first
    # panel's explicit limit disables autoscaling for subsequent scenarios.
    displayed = workload.loc[workload.scenario_id.isin([s for s, _ in selected])]
    axes[0, 0].set_ylim(0, max(displayed.million_tokens_per_s.max(), capacity_million_tokens_per_s) * 1.08)
    axes[1, 0].set_ylim(0, displayed.task_count.max() / 1000 * 1.08)
    fig.legend(handles=[capacity_line], loc="outside lower center", frameon=False)
    save(fig, output, "llm_workload_representative")


def heatmap(ax, frame, scenario, slot, value, color_label, cmap, row_labels=None, norm=None):
    matrix = frame.pivot(index=scenario, columns=slot, values=value).sort_index()
    im = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap,
                   extent=(0.5, 96.5, len(matrix) - 0.5, -0.5), norm=norm)
    ax.set_yticks(range(len(matrix)), row_labels or matrix.index.tolist())
    ax.set_xticks(TICKS)
    ax.set_xlabel("Slot (15 minutes)")
    ax.set_ylabel("Scenario")
    for boundary in np.arange(0.5, len(matrix), 1):
        ax.axhline(boundary, color="white", linewidth=0.6, alpha=0.65)
    colorbar = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    colorbar.set_label(color_label, labelpad=14)
    return im


def plot_workload_heatmap(workload, output):
    # Use one linear scale across all 12 sampled days, starting at zero.
    fig, ax = plt.subplots(figsize=(16, 8), layout="constrained")
    ax.set_title("Dense and bursty LLM workload arrivals: all 12 sampled scenarios", pad=18)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=workload.million_tokens_per_s.max())
    heatmap(ax, workload, "scenario_id", "slot", "million_tokens_per_s",
            "Workload (million tokens/s)", "YlGnBu", norm=norm)
    save(fig, output, "llm_workload_heatmap")


def plot_all_workloads(workload, summary, output, capacity):
    fig, axes = plt.subplots(4, 3, figsize=(20, 17), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("All 12 selected workload scenarios: sustained overload and bursts")
    maximum = max(float(workload.million_tokens_per_s.max()), capacity) * 1.08
    capacity_line = None
    for ax, scenario in zip(axes.flat, summary.index):
        group = workload.loc[workload.scenario_id.eq(scenario)].sort_values("slot")
        ax.plot(group.slot, group.million_tokens_per_s, color="#23749B")
        ax.fill_between(group.slot, group.million_tokens_per_s, color="#23749B", alpha=.12)
        ax.fill_between(group.slot, capacity, group.million_tokens_per_s,
                        where=group.million_tokens_per_s > capacity, color="#C8463D", alpha=.25)
        capacity_line = ax.axhline(capacity, color="red", linestyle="--", linewidth=1.8,
                                   label=f"Processing capacity: {capacity:g} million tokens/s")
        count = int(summary.loc[scenario, "overload_slots"])
        ax.set_title(f"{scenario}\n{count} slots above capacity")
        ax.set_ylim(0, maximum)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        slot_axis(ax)
    for ax in axes[:-1].flat:
        ax.set_xlabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("Workload (million tokens/s)")
    fig.legend(handles=[capacity_line], loc="outside lower center", frameon=False)
    save(fig, output, "llm_workload_all_scenarios")


def plot_stress_summary(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=True, layout="constrained")
    fig.suptitle("Selected workload stress: overload duration and immediate-service overflow")
    positions = np.arange(len(summary))
    fields = [("overload_slots", "Overloaded slots (of 96)", "Overload duration"),
              ("immediate_service_overflow_fraction", "Overflow (% of daily tokens)", "Immediate-service overflow")]
    for ax, (field, label, title), color in zip(axes, fields, COLORS):
        values = summary[field].to_numpy(dtype=float)
        if field.endswith("fraction"):
            values = values * 100
        bars = ax.barh(positions, values, color=color, alpha=.9)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=5, fontsize=16)
        ax.set_xlim(0, values.max() * 1.22)
        ax.set_yticks(positions, summary.index)
        ax.set_xlabel(label)
        ax.set_title(title)
        ax.grid(axis="x", color="#DFE3E6", linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    save(fig, output, "llm_workload_stress_summary")


def write_workload_readme(output, summary, selected):
    table = "| Scenario | Mean | Peak | Overloaded slots | Overflow (%) |\n|---|---:|---:|---:|---:|\n"
    table += "\n".join(
        f"| {scenario} | {row.mean_million_tokens_per_s:.3f} | {row.peak_million_tokens_per_s:.3f} | {int(row.overload_slots)} | {row.immediate_service_overflow_fraction * 100:.2f} |"
        for scenario, row in summary.iterrows()
    )
    panels = ", ".join(f"{scenario} ({reason.lower()})" for scenario, reason in selected)
    updated = [name for formats in SAVED_FIGURES.values() for name in formats.values() if "_updated." in name]
    replacement_note = ""
    if updated:
        replacement_note = "\nFiles open in a Windows viewer could not be overwritten. The complete refreshed figures are " + ", ".join(f"`{name}`" for name in updated) + ". Their corresponding PNGs and the manifest reference the new data. Close the original PDFs and rerun the plotting command to refresh their original filenames.\n"
    content = f"""# Energy and dense/bursty LLM workload visualizations

All six figures are exported as 300-dpi PNGs and PDFs with embedded TrueType fonts. Every visible title, axis label, tick and legend uses {plt.rcParams['font.family'][0]} at 16 pt or larger. Use the intended physical figure size when inserting PDFs into a document.
{replacement_note}

| Figure stem | Contents |
|---|---|
| `llm_workload_representative` | Three workload examples: {panels}. Top row: sampled arrivals in million tokens/s, with overload shaded and the red dashed processing-capacity line. Bottom row: sampled task arrivals in thousands per slot. |
| `llm_workload_heatmap` | All twelve actual sampled workload profiles with one shared linear color scale starting at zero. |
| `llm_workload_all_scenarios` | A 4-by-3 overview of all twelve workload profiles, common axes, capacity lines, and overloaded-slot counts. |
| `llm_workload_stress_summary` | Overloaded-slot counts and the hypothetical immediate-service overflow fraction for all twelve scenarios. |
| `energy_representative` | Three energy days showing fixed supply, battery charging, net available energy and day-ahead prices. |
| `energy_heatmaps` | All twelve energy days: net available energy and electricity price. |

The workload set now targets sustained overload and concentrated bursts. It retains R05-D333 and R11-D228 as anchors. The data folder's `workload_selection_manifest.json` records the selection criteria, all-day screening results, source hashes and backups. These are purposively selected stress scenarios; no optimizer-profit outcome was used in their selection. Retained scenarios preserve their original sampled occurrences and fixed labels.

## Actual sampled workload statistics

Mean and peak rates below are in million tokens/s. Overload means slot-average demand exceeds the installed processing capacity. Overflow is `100 * sum(max(arrival_tokens - capacity_tokens, 0)) / sum(arrival_tokens)`: a hypothetical same-slot service shortfall, not workload dropped by the proposed scheduler.

{table}

Each day contains 96 slots of 900 seconds. The red dashed capacity is read from `config/default.yaml`: `installed_replicas * throughput_tokens_per_second_per_replica / 1e6`. Dense slots, also recorded in the summary CSV, have arrivals at least 80% of that capacity. Workload rates and counts describe arrivals, not measured serving throughput or scheduled completions.

Plots use the active `.sample.json` recipes and their referenced `representative_tasks_eight_types.csv.gz` source. Per slot, task arrivals are `sum(copies)`, token arrivals are `sum(copies * (input_tokens + output_tokens))`, and the rate is token arrivals divided by 900 seconds. All task types, including immediate service, are included. Copied requests are aggregated without materializing individual copies; no smoothing or synthetic burst is added.

Energy inputs are unchanged. PPA, gas turbine, PV and battery discharge form the displayed supply; battery charging is subtracted to obtain net available energy. Prices are converted from USD/kWh to USD/MWh. Energy trajectories are scenario inputs, not optimized dispatch. Workload trace-day identifiers and energy calendar dates are independent, and the days are not asserted to be paired observations or a consecutive time series.

## Reproduce and validate

From the repository root:

```powershell
python data_visualization/plot_scenarios.py
```

Options include `--data-dir`, `--output-dir`, `--config`, and `--font-file`. The script resolves default paths from its own location and saves generated files under `output/input_visualizations/` by default. It requires NumPy, pandas, Matplotlib and PyYAML. It uses the supplied font or installed Times New Roman, falling back to bundled DejaVu Serif when needed; the manifest records the selected font.

`llm_sampled_slot_metrics.csv` contains plotted slot-level arrivals; `llm_scenario_summary.csv` contains actual workload rates, overload counts, overflow fractions and selection reasons. The energy CSVs contain the unchanged energy aggregates. `visualization_manifest.json` records formulas, provenance, selected panels and font settings. Checks cover 96-slot grids, hashes, unique task IDs, token accounting, exact sampled task/token totals, agreement with the data-selection audit, and font family/minimum size.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def plot_energy(data, output):
    energy = pd.read_csv(data / "energy_scenarios.csv")
    check_grid(energy, "energy_scenario_id", "slot")
    numeric = ENERGY_COLS + ["bess_charge_kwh", "day_ahead_price_usd_per_kwh"]
    if not np.isfinite(energy[numeric].to_numpy()).all():
        raise ValueError("Non-finite energy values")
    if (energy[ENERGY_COLS + ["bess_charge_kwh"]] < 0).any().any():
        raise ValueError("Negative energy quantities")
    energy["net_available_kwh"] = energy[ENERGY_COLS].sum(axis=1) - energy.bess_charge_kwh
    energy["price_usd_per_mwh"] = energy.day_ahead_price_usd_per_kwh * 1000
    summary = energy.groupby("energy_scenario_id").agg(
        date=("date", "first"), pv_kwh=("pv_kwh", "sum"),
        net_available_kwh=("net_available_kwh", "sum"),
        mean_price_usd_per_mwh=("price_usd_per_mwh", "mean"),
    )
    ordered = summary.sort_values(["pv_kwh"], kind="stable")
    median = float(summary.pv_kwh.median())
    middle = (summary.pv_kwh - median).abs().sort_values(kind="stable").index[0]
    selected = [(ordered.index[0], "Lowest daily PV"), (middle, "Near-median daily PV"),
                (ordered.index[-1], "Highest daily PV")]
    summary["selection_reason"] = ""
    fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True, sharey="col", layout="constrained")
    fig.suptitle("Energy scenarios: supply composition and day-ahead electricity price")
    component_colors = ["#337DA1", "#9EACB7", "#E9B847", "#509D79"]
    for row, (scenario, reason) in enumerate(selected):
        summary.loc[scenario, "selection_reason"] = reason
        group = energy.loc[energy.energy_scenario_id.eq(scenario)].sort_values("slot")
        left, right = axes[row]
        left.stackplot(group.slot, *[group[c] for c in ENERGY_COLS],
                       labels=["PPA", "Gas turbine", "PV", "Battery discharge"], colors=component_colors, alpha=0.9)
        left.fill_between(group.slot, -group.bess_charge_kwh, 0, color="#BB7079", label="Battery charge")
        left.plot(group.slot, group.net_available_kwh, color="#242A30", linewidth=1.7, label="Net available")
        left.axhline(0, color="#555555", linewidth=0.8)
        left.set_ylabel("Energy (kWh/slot)")
        left.set_title(f"{scenario} | {group.date.iloc[0]} | {reason}")
        right.plot(group.slot, group.price_usd_per_mwh, color="#77589B", drawstyle="steps-mid")
        right.set_ylabel("Price (USD/MWh)")
        right.set_title(f"{scenario} | Day-ahead price")
        for ax in (left, right):
            slot_axis(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if row < 2:
                ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    save(fig, output, "energy_representative")
    fig, axes = plt.subplots(2, 1, figsize=(17, 13), layout="constrained")
    row_labels = [f"{scenario}  {summary.loc[scenario, 'date']}" for scenario in sorted(summary.index)]
    for ax, value, title, label, cmap in [
        (axes[0], "net_available_kwh", "Net available energy: all 12 scenarios", "Energy (kWh/slot)", "YlGnBu"),
        (axes[1], "price_usd_per_mwh", "Day-ahead electricity price: all 12 scenarios", "Price (USD/MWh)", "YlOrRd"),
    ]:
        ax.set_title(title, pad=15)
        heatmap(ax, energy, "energy_scenario_id", "slot", value, label, cmap, row_labels)
    save(fig, output, "energy_heatmaps")
    energy.to_csv(output / "energy_slot_metrics.csv", index=False)
    summary.to_csv(output / "energy_scenario_summary.csv")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "output/input_visualizations")
    parser.add_argument("--config", type=Path, default=HERE.parent / "config" / "default.yaml",
                        help="Optimizer configuration supplying the processing capacity")
    parser.add_argument("--font-file", type=Path, help="Optional TrueType font file")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = configure_style(args.font_file)
    parameters = yaml.safe_load(args.config.read_text(encoding="utf-8"))["parameters"]
    slot_seconds = parameters["slot_seconds"]
    if slot_seconds != 900 or parameters["horizon_slots"] != 96:
        raise ValueError("These daily plots require 96 fifteen-minute slots")
    replicas = parameters["installed_replicas"]
    per_replica_rate = parameters["throughput_tokens_per_second_per_replica"]
    if not np.isfinite([replicas, per_replica_rate]).all() or min(replicas, per_replica_rate) <= 0:
        raise ValueError("Replica count and per-replica throughput must be finite and positive")
    capacity = replicas * per_replica_rate / 1e6
    workload, provenance = load_workloads(args.data_dir, slot_seconds)
    summary, selected = workload_summary(workload, capacity)
    selection_path = args.data_dir / "workload_selection.csv"
    selection_manifest_path = args.data_dir / "workload_selection_manifest.json"
    selection_manifest = None
    if selection_path.exists():
        selection = pd.read_csv(selection_path).set_index("scenario_id")
        if selection.index.has_duplicates or set(selection.index) != set(summary.index):
            raise ValueError("Selection audit and active workload scenarios disagree")
        for field in ("daily_tokens", "daily_tasks", "peak_million_tokens_per_s",
                      "overload_slots", "immediate_service_overflow_fraction"):
            if not np.allclose(summary[field], selection.loc[summary.index, field], rtol=1e-12, atol=1e-9):
                raise ValueError(f"Selection audit differs from actual sampled plots: {field}")
        summary["selection_reason"] = selection.loc[summary.index, "selection_reason"]
    if selection_manifest_path.exists():
        selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    workload.to_csv(args.output_dir / "llm_sampled_slot_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "llm_scenario_summary.csv")
    plot_workloads(workload, selected, args.output_dir, capacity)
    plot_workload_heatmap(workload, args.output_dir)
    plot_all_workloads(workload, summary, args.output_dir, capacity)
    plot_stress_summary(summary, args.output_dir)
    energy_selected = plot_energy(args.data_dir, args.output_dir)
    manifest = {
        "slot_seconds": slot_seconds, "font": plt.rcParams["font.family"][0], "font_file": font,
        "minimum_font_pt": 16, "formats": ["png (300 dpi)", "pdf (embedded TrueType)"],
        "optimizer_config": str(args.config.resolve()),
        "capacity_million_tokens_per_s": capacity,
        "capacity_formula": "installed_replicas * throughput_tokens_per_second_per_replica / 1e6",
        "workload_source": "sample recipe copies multiplied by source input_tokens + output_tokens",
        "rate_formula": "sum(copies * total_tokens) / 900 / 1e6",
        "count_formula": "sum(copies) by arrival_slot, including all task types",
        "workload_selected": selected, "energy_selected": energy_selected,
        "workload_selection": selection_manifest,
        "workload_figures": ["llm_workload_representative", "llm_workload_heatmap",
                             "llm_workload_all_scenarios", "llm_workload_stress_summary"],
        "figure_files": SAVED_FIGURES,
        "provenance": provenance,
        "validation": "96 unique slots per day; source hashes, task IDs, daily task/token totals against recipes, workload-selection audit, font family and minimum sizes checked",
    }
    (args.output_dir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_workload_readme(args.output_dir, summary, selected)
    print(summary.to_string())
    print(f"Saved six figures in PNG and PDF to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
