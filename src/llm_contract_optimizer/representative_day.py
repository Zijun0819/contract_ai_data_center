"""Four aligned figures for a single 24-hour case and both existing benchmarks."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from llm_contract_optimizer.benchmarks import run_benchmarks
from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.exceptions import DataValidationError, NoFeasibleSolutionError
from llm_contract_optimizer.horizon_experiment import _check_proposed
from llm_contract_optimizer.reporting import (
    _atomic_text, _csv, print_method_comparison, write_method_comparison, write_result_bundle,
)
from llm_contract_optimizer.solve import load_case, solve_aggregate

LOGGER = logging.getLogger(__name__)
FIGURES = (
    "llm_workload_arrival",
    "llm_workload_scheduling",
    "day_ahead_market_price",
    "grid_imports",
)
METHODS = (
    ("proposed", "Proposed method", "#2563eb", "-"),
    ("flat_50_24", "Voluntary Flat-50/24", "#0f766e", "--"),
    ("zero_discount", "Immediate 0%; overflow dropped", "#dc2626", "-."),
)
BENCHMARK_IDS = ("voluntary_flat_50_24", "immediate_zero_discount_drop_overflow")


def day_series(config, proposed, benchmarks):
    """Return the exact plotted slot data, with imports also converted to mean MW."""
    p = config.parameters
    slots = p.horizon_slots
    tolerance = max(1e-3, config.solver.token_scale * config.solver.feasibility_tolerance * 10)

    def array(values, name):
        values = np.asarray(values, dtype=float)
        if values.shape != (slots,) or not np.isfinite(values).all():
            raise DataValidationError(f"Invalid representative-day series: {name}")
        if np.any(values < -1e-5):
            raise DataValidationError(f"Negative representative-day series: {name}")
        return values

    immediate = array(proposed.workload.opt_out_tokens_by_arrival, "immediate arrivals")
    participating = array(proposed.workload.participating_tokens_by_type_arrival.sum(axis=0),
                          "participating arrivals")
    arrivals = immediate + participating
    data = {
        "arrival_tokens": arrivals,
        "immediate_arrival_tokens": immediate,
        "participating_arrival_tokens": participating,
        "capacity_tokens": np.full(slots, p.token_capacity_per_slot),
        "day_ahead_price_usd_per_kwh": array(proposed.energy.day_ahead_price_usd_per_kwh, "price"),
        "proposed_executed_tokens": array(proposed.inner.total_tokens_by_slot, "proposed execution"),
        "proposed_grid_import_kwh": array(proposed.inner.day_ahead_energy_kwh, "proposed imports"),
        "proposed_dropped_tokens": np.zeros(slots),
    }
    by_id = {result.benchmark_id: result for result in benchmarks}
    if len(benchmarks) != 2 or set(by_id) != set(BENCHMARK_IDS):
        raise DataValidationError("Representative day requires both existing benchmarks.")
    for prefix, benchmark_id in zip(("flat_50_24", "zero_discount"), BENCHMARK_IDS):
        result = by_id[benchmark_id]
        if not result.feasible:
            raise DataValidationError(f"Representative-day benchmark failed: {benchmark_id}: {result.message}")
        if not np.allclose(result.arrival_tokens_by_slot, arrivals, atol=tolerance, rtol=0):
            raise DataValidationError(f"Benchmark arrivals differ: {benchmark_id}")
        data[f"{prefix}_executed_tokens"] = array(result.executed_tokens_by_slot, f"{prefix} execution")
        data[f"{prefix}_grid_import_kwh"] = array(result.day_ahead_energy_kwh, f"{prefix} imports")
        data[f"{prefix}_dropped_tokens"] = array(result.metadata.get("dropped_tokens_by_slot"),
                                                  f"{prefix} dropped arrivals")

    costs = (proposed.day_ahead_cost_usd, by_id[BENCHMARK_IDS[0]].day_ahead_cost_usd,
             by_id[BENCHMARK_IDS[1]].day_ahead_cost_usd)
    for (prefix, _, _, _), cost in zip(METHODS, costs):
        executed = data[f"{prefix}_executed_tokens"]
        dropped = data[f"{prefix}_dropped_tokens"]
        imports = data[f"{prefix}_grid_import_kwh"]
        admitted = arrivals - dropped
        if (abs(executed.sum() - admitted.sum()) > tolerance
                or np.max(executed - p.token_capacity_per_slot) > tolerance
                or np.max(np.cumsum(executed) - np.cumsum(admitted)) > tolerance
                or np.min(admitted) < -tolerance):
            raise DataValidationError(f"Workload conservation/capacity failed: {prefix}")
        expected_imports = np.maximum(
            0, p.idle_facility_energy_kwh_per_slot + p.facility_kwh_per_token * executed
            + proposed.energy.bess_charge_kwh - proposed.energy.fixed_supply_kwh,
        )
        if not np.allclose(imports, expected_imports, atol=1e-5, rtol=0):
            raise DataValidationError(f"Grid imports disagree with energy balance: {prefix}")
        if not np.isclose(np.dot(imports, data["day_ahead_price_usd_per_kwh"]), cost,
                          atol=1e-5, rtol=0):
            raise DataValidationError(f"Day-ahead cost disagrees with plotted imports: {prefix}")
        data[f"{prefix}_grid_import_mw"] = imports / (1000 * p.slot_hours)
    return data


def write_day_plots(output, config, data, proposed_converged):
    """Write exactly four figure identities, each in PNG and vector PDF."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    p = config.parameters
    edges = np.arange(p.horizon_slots + 1) * p.slot_hours
    horizon = float(edges[-1])
    context = f"{config.inputs.workload_scenario_id} | {config.inputs.energy_scenario_id} | {horizon:g} h"

    def setup(title, ylabel):
        fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
        fig.suptitle(f"{title}\n{context}", fontsize=16)
        ax.set(xlabel="Hour within planning horizon", ylabel=ylabel, xlim=(0, horizon))
        ax.set_xticks(np.arange(0, horizon + .01, 3))
        ax.grid(alpha=.22)
        ax.set_axisbelow(True)
        return fig, ax

    def finish(fig, ax, name):
        ax.set_ylim(bottom=0)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=2,
                       frameon=False, fontsize=14)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    slot_unit = f"Million tokens / {p.slot_seconds / 60:g}-min slot"
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 16,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = setup("LLM workload arrival", slot_unit)
        ax.stairs(data["immediate_arrival_tokens"] / 1e6, edges, baseline=0, fill=True,
                  color="#bfdbfe", label="Immediate arrivals")
        ax.stairs(data["arrival_tokens"] / 1e6, edges,
                  baseline=data["immediate_arrival_tokens"] / 1e6, fill=True,
                  color="#99f6e4", label="Participating arrivals")
        ax.stairs(data["arrival_tokens"] / 1e6, edges, baseline=None,
                  color="#111827", linewidth=1.8, label="Total arrivals")
        ax.axhline(p.token_capacity_per_slot / 1e6, color="#6b7280", linestyle=":",
                   linewidth=1.5, label="Execution capacity")
        finish(fig, ax, FIGURES[0])

        status = "" if proposed_converged else " (proposed: best feasible; optimality not certified)"
        fig, ax = setup("LLM workload scheduling" + status, slot_unit)
        for prefix, label, color, style in METHODS:
            ax.stairs(data[f"{prefix}_executed_tokens"] / 1e6, edges, baseline=None,
                      color=color, linestyle=style, linewidth=2, label=label)
        ax.axhline(p.token_capacity_per_slot / 1e6, color="#6b7280", linestyle=":",
                   linewidth=1.5, label="Execution capacity")
        finish(fig, ax, FIGURES[1])

        fig, ax = setup("Day-ahead market price", "USD/kWh")
        ax.stairs(data["day_ahead_price_usd_per_kwh"], edges, baseline=None,
                  color="#7c3aed", linewidth=2)
        finish(fig, ax, FIGURES[2])

        fig, ax = setup("Grid imports" + status, "Average grid import (MW)")
        for prefix, label, color, style in METHODS:
            ax.stairs(data[f"{prefix}_grid_import_mw"], edges, baseline=None,
                      color=color, linestyle=style, linewidth=2, label=label)
        finish(fig, ax, FIGURES[3])


def run_representative_day(args):
    config = load_config(args.config)
    if not np.isclose(config.parameters.horizon_slots * config.parameters.slot_hours, 24):
        raise DataValidationError("representative-day requires a 24-hour configuration.")
    root = args.output_dir.resolve() if args.output_dir is not None else config.output.root / "representative_day"
    output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    # Reuse the numeric result writers while suppressing all their standard plots.
    config = replace(config, output=replace(config.output, root=output,
                                           make_plot=False, write_group_schedule=False))
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        LOGGER.info("Representative day: %s / %s; outputs: %s",
                    config.inputs.workload_scenario_id, config.inputs.energy_scenario_id, output)
        config, workload, energy = load_case(config)
        proposed = solve_aggregate(config, workload, energy)
        audit = _check_proposed(proposed, config)
        benchmarks = run_benchmarks(config, workload, energy)
        data = day_series(config, proposed, benchmarks)
        write_result_bundle(proposed, config, output)
        write_method_comparison(output, proposed, benchmarks, config)
        rows = [{"slot": s + 1, "hour_start": s * config.parameters.slot_hours,
                 "hour_end": (s + 1) * config.parameters.slot_hours,
                 **{name: float(values[s]) for name, values in data.items()}}
                for s in range(config.parameters.horizon_slots)]
        _csv(output / "representative_day_slots.csv", rows)
        write_day_plots(output, config, data, proposed.converged)
        manifest = {
            "experiment": "representative_day_v1",
            "workload_scenario_id": workload.scenario_id, "energy_scenario_id": energy.scenario_id,
            "energy_date": energy.date, "horizon_hours": 24,
            "slot_seconds": config.parameters.slot_seconds, "algorithm": proposed.algorithm,
            "proposed_converged": proposed.converged, "proposed_validation": audit,
            "time_alignment": "Slot indices align within the planning horizon; workload and power are independently selected traces.",
            "grid_import_unit": "Mean MW per slot = imported kWh / (1000 * slot hours).",
            "schedule_unit": "Divisible tokens executed within each slot; not whole-task completion counts.",
            "source_data": "representative_day_slots.csv",
            "figure_files": [f"{name}.{extension}" for name in FIGURES for extension in ("png", "pdf")],
            "methods": {
                prefix: {"executed_tokens": float(data[f"{prefix}_executed_tokens"].sum()),
                         "dropped_tokens": float(data[f"{prefix}_dropped_tokens"].sum()),
                         "dropped_fraction": float(data[f"{prefix}_dropped_tokens"].sum() / workload.total_tokens)
                         if workload.total_tokens else 0.0,
                         "grid_import_kwh": float(data[f"{prefix}_grid_import_kwh"].sum())}
                for prefix, _, _, _ in METHODS
            },
        }
        actual = {path.name for path in output.iterdir() if path.suffix in (".png", ".pdf")}
        if actual != set(manifest["figure_files"]):
            raise DataValidationError("Representative-day run must contain exactly the four requested figures in PNG/PDF.")
        _atomic_text(output / "representative_day.json",
                     lambda stream: json.dump(manifest, stream, indent=2, allow_nan=False))
        print_method_comparison(proposed, benchmarks, output)
        print(f"Four representative-day figures saved as PNG/PDF in: {output}", flush=True)
        return 0
    except Exception as exc:
        _atomic_text(output / "failure.json", lambda stream: json.dump({
            "status": "infeasible" if isinstance(exc, NoFeasibleSolutionError) else "error",
            "error_type": type(exc).__name__, "message": str(exc),
        }, stream, indent=2))
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
