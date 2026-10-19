"""Plot the shared consecutive workload and power trajectory for the horizon sweep.

Run: python data_visualization/plot_planning_horizon_inputs.py
"""
import argparse
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.solve import load_case
from plot_scenarios import configure_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config/planning_horizons/H168.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/planning_horizons/experiment_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "output/input_visualizations/planning_horizons")
    parser.add_argument("--font-file", type=Path, help="Optional TrueType font file")
    args = parser.parse_args()
    config, workload, energy = load_case(load_config(args.config))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    arrivals = workload.opt_out_tokens_by_arrival + workload.participating_tokens_by_type_arrival.sum(axis=0)
    x = np.arange(config.parameters.horizon_slots) * config.parameters.slot_hours
    capacity = config.parameters.token_capacity_per_slot / config.parameters.slot_seconds / 1e6
    rate = arrivals / config.parameters.slot_seconds / 1e6
    configure_style(args.font_file)
    plt.rcParams.update({"font.size": 16,
                         "axes.titlesize": 18, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True, constrained_layout=True)
    axes[0].step(x, rate, where="post", color="#2563eb", linewidth=1.6, label="Arrival demand")
    axes[0].axhline(capacity, color="#dc2626", linestyle="--", label="Installed capacity")
    axes[0].fill_between(x, capacity, rate, where=rate > capacity, step="post", alpha=.25, color="#dc2626")
    axes[0].set(title="Consecutive workload; identical task copies and labels in overlapping windows", ylabel="Million tokens/s")
    axes[0].legend(loc="upper left", fontsize=16)
    top = axes[0].secondary_xaxis("top")
    starts = [168 - hours for hours in reversed(manifest["horizons_hours"])]
    top.set_xticks(starts, [f"{hours}h" for hours in reversed(manifest["horizons_hours"])])
    top.set_xlabel("Window start for each horizon; all windows end at hour 168", labelpad=12)
    for values, label, color in ((energy.ppa_kwh, "PPA", "#2563eb"),
            (energy.gas_turbine_kwh, "Gas turbine", "#7c3aed"), (energy.pv_kwh, "PV", "#d97706"),
            (energy.bess_discharge_kwh, "Battery discharge", "#0f766e")):
        axes[1].step(x, values / config.parameters.slot_hours, where="post", label=label, color=color)
    axes[1].set(title="Consecutive default-anchored power profiles", ylabel="Supply power (kW)")
    axes[1].legend(loc="upper left", ncol=4, fontsize=16)
    axes[2].step(x, energy.day_ahead_price_usd_per_kwh, where="post", color="#0f766e", linewidth=1.8)
    axes[2].set(title="Day-ahead electricity price", ylabel="USD/kWh", xlabel="Hours since start of the 168-hour window")
    for ax in axes:
        ax.axvspan(144, 168, color="#64748b", alpha=.10)
        for boundary in range(24, 168, 24):
            ax.axvline(boundary, color="#64748b", alpha=.25, linewidth=.8)
        ax.grid(alpha=.2)
        ax.set_xlim(0, 168)
    axes[-1].set_xticks(np.arange(0, 169, 24))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"consecutive_horizon_inputs.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output / "consecutive_horizon_inputs.pdf")


if __name__ == "__main__":
    main()
