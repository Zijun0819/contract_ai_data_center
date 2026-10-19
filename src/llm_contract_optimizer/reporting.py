"""Contract-first, reproducible outputs for the divisible-workload model."""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import numpy as np
import yaml

from llm_contract_optimizer.config import AppConfig
from llm_contract_optimizer.domain import BenchmarkResult, OptimizationResult
from llm_contract_optimizer.inner import FluidInnerSolver


def _atomic_text(path: Path, writer: Callable[[Any], None], newline: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", newline=newline,
                                dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _effective_config(config: AppConfig) -> dict[str, Any]:
    data = asdict(config)
    data.pop("source_path")
    data["inputs"]["root"] = str(config.inputs.root)
    data["inputs"]["workload_file"] = (
        str(config.inputs.workload_file) if config.inputs.workload_file is not None else None)
    data["output"]["root"] = str(config.output.root)
    # Use ordinary YAML lists, so the saved file can be loaded without Python tags.
    for section in data.values():
        for name, value in section.items():
            if isinstance(value, tuple):
                section[name] = list(value)
    return data


def _csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    def write(stream: Any) -> None:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_text(path, write, newline="")


def create_run_directory(config: AppConfig) -> Path:
    case = f"{config.inputs.workload_scenario_id}__{config.inputs.energy_scenario_id}"
    case = re.sub(r"[^A-Za-z0-9_.-]", "_", case)
    output = config.output.root / case / (
        datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + config.solver.algorithm)
    output.mkdir(parents=True, exist_ok=False)
    return output


def menu_rows(result: OptimizationResult, config: AppConfig) -> list[dict]:
    p, w, menu = config.parameters, result.workload, result.menu
    rows = []
    for i in range(len(p.type_lambdas) + 1):
        deadline = menu.deadlines_slots[i - 1] if i else 0
        discount = menu.discounts[i - 1] if i else 0.0
        penalty = menu.penalties[i - 1] if i else 0.0
        sensitivity = p.type_lambdas[i - 1] if i else None
        bill = w.participating_bill_usd_by_type[i - 1] if i else w.opt_out_bill_usd
        tokens = (w.participating_tokens_by_type_arrival[i - 1].sum()
                  if i else w.opt_out_tokens_by_arrival.sum())
        rows.append({
            "type": i,
            "service": "immediate" if i == 0 else "participating",
            "delay_sensitivity": sensitivity,
            "max_deferral_slots": deadline,
            "max_deferral_hours": deadline * p.slot_hours,
            "delay_penalty": penalty,
            "discount_fraction": discount,
            "discount_percent": 100 * discount,
            "input_price_usd_per_million_tokens": p.input_tariff_usd_per_million_tokens * (1 - discount),
            "output_price_usd_per_million_tokens": p.output_tariff_usd_per_million_tokens * (1 - discount),
            "task_count": int(w.type_counts[i]),
            "tokens": float(tokens),
            "undiscounted_bill_usd": float(bill),
            "revenue_usd": float(bill * (1 - discount)),
            "normalized_ir_margin": discount - sensitivity * penalty if i else 0.0,
        })
    return rows


def result_validation(result: OptimizationResult, config: AppConfig) -> dict:
    p, w = config.parameters, result.workload
    v = result.inner.scheduled_tokens_by_type_slot
    arrivals = np.cumsum(w.participating_tokens_by_type_arrival, axis=1)
    executed = np.cumsum(v, axis=1)
    due = np.zeros_like(arrivals)
    for i, deadline in enumerate(result.menu.deadlines_slots):
        for s in range(p.horizon_slots - 1):
            due[i, s] = arrivals[i, s - deadline] if s >= deadline else 0.0
        due[i, -1] = arrivals[i, -1]
    penalties = np.asarray(result.menu.penalties)
    discounts = np.asarray(result.menu.discounts)
    lambdas = np.asarray(p.type_lambdas)
    utilities = discounts[None, :] - lambdas[:, None] * penalties[None, :]
    assigned = np.diag(utilities)
    energy = result.energy
    balance = (energy.fixed_supply_kwh + result.inner.day_ahead_energy_kwh
               - result.inner.facility_energy_kwh - energy.bess_charge_kwh
               - result.inner.spill_energy_kwh)
    return {
        "minimum_ir_margin": float(assigned.min()),
        "minimum_ic_margin": float((assigned[:, None] - utilities).min()),
        "maximum_discount_cap_excess": float(max(0, discounts.max() - p.maximum_discount)),
        "maximum_execution_before_arrival_tokens": float(np.maximum(executed - arrivals, 0).max()),
        "maximum_deadline_shortfall_tokens": float(np.maximum(due - executed, 0).max()),
        "maximum_terminal_mass_error_tokens": float(np.abs(executed[:, -1] - arrivals[:, -1]).max()),
        "maximum_capacity_excess_tokens": float(np.maximum(
            result.inner.total_tokens_by_slot - p.token_capacity_per_slot, 0).max()),
        "maximum_energy_balance_error_kwh": float(np.abs(balance).max()),
        "solver_feasibility_tolerance_scaled": config.solver.feasibility_tolerance,
        "token_scale": config.solver.token_scale,
    }


def _summary(result: OptimizationResult, config: AppConfig) -> dict[str, Any]:
    return {
        "schema_version": "4.0",
        "status": "optimal_within_tolerance" if result.converged else "best_feasible",
        "solution_scope": "fixed_types_aggregated_divisible_workload",
        "horizon_policy": "finish_within_horizon",
        "algorithm": result.algorithm,
        "termination_reason": result.termination_reason,
        "workload_scenario_id": result.workload_scenario_id,
        "energy_scenario_id": result.energy_scenario_id,
        "energy_date": result.energy_date,
        "contract_menu": menu_rows(result, config),
        "objective": {
            "profit_usd": result.profit_usd,
            "revenue_usd": result.revenue_usd,
            "fixed_energy_cost_usd": result.fixed_energy_cost_usd,
            "day_ahead_cost_usd": result.day_ahead_cost_usd,
        },
        "certificate": {
            "loss_lower_bound_usd": result.loss_lower_bound_usd,
            "loss_upper_bound_usd": result.loss_upper_bound_usd,
            "profit_lower_bound_usd": result.lower_bound_profit_usd,
            "profit_upper_bound_usd": result.upper_bound_profit_usd,
            "absolute_gap_usd": result.absolute_gap_usd,
            "relative_gap": result.relative_gap,
            "converged": result.converged,
        },
        "workload": {
            "task_count": result.workload.task_count,
            "source_record_count": result.workload.source_record_count,
            "type_counts": result.workload.type_counts.tolist(),
            "total_tokens": result.workload.total_tokens,
            "input_provenance": result.workload.input_provenance,
        },
        "iterations": len(result.history),
        "optimality_cuts": result.optimality_cuts,
        "feasibility_cuts": result.feasibility_cuts,
        "elapsed_seconds": result.elapsed_seconds,
        "validation": result_validation(result, config),
    }


def print_result(result: OptimizationResult, config: AppConfig) -> None:
    print("\nDERIVED CONTRACT MENU")
    print("Type  Deferral (h)  Discount (%)  Input USD/M tokens  Output USD/M tokens")
    for row in menu_rows(result, config):
        print(f"{row['type']:>4}  {row['max_deferral_hours']:>12.2f}  "
              f"{row['discount_percent']:>12.4f}  "
              f"{row['input_price_usd_per_million_tokens']:>18.6f}  "
              f"{row['output_price_usd_per_million_tokens']:>19.6f}")
    print("Type 0: immediate service. All work finishes within the supplied horizon.")
    status = "Optimal within tolerance" if result.converged else "Best feasible; optimality not established"
    print(f"\n{status} ({result.algorithm}; {result.termination_reason})")
    print(f"Profit: USD {result.profit_usd:,.2f}; certified upper bound: "
          f"USD {result.upper_bound_profit_usd:,.2f}; gap: USD {result.absolute_gap_usd:.6g}")
    if result.output_directory:
        print(f"Contract menu: {result.output_directory / 'contract_menu.md'}")
        print(f"All outputs:   {result.output_directory}")


def _write_menu_document(path: Path, result: OptimizationResult, config: AppConfig) -> None:
    status = "Optimal within the configured tolerance" if result.converged else "Best feasible menu; optimality is not established"
    lines = [
        "# Derived contract menu", "",
        f"{result.workload_scenario_id} / {result.energy_scenario_id} | {result.energy_date}", "",
        f"**{status}.** Algorithm: {result.algorithm}. Termination: {result.termination_reason}.", "",
        "| Type | Maximum deferral (slots) | Maximum deferral (hours) | Discount | Input price (USD/M tokens) | Output price (USD/M tokens) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in menu_rows(result, config):
        lines.append(f"| {row['type']} | {row['max_deferral_slots']} | "
                     f"{row['max_deferral_hours']:.2f} | {row['discount_percent']:.6f}% | "
                     f"{row['input_price_usd_per_million_tokens']:.8f} | "
                     f"{row['output_price_usd_per_million_tokens']:.8f} |")
    lines.extend([
        "", "Type 0 is the fixed immediate-service group. Participating types retain their assigned labels.",
        "Deferrals are nominal contract promises. An arrival in slot a must finish by "
        "min(a + D_i, S); no workload is moved beyond the supplied horizon.",
        "Workload is divisible at the arrival/type aggregate level. IR and IC are weak inequalities; "
        "indifferent types are assumed to retain their designated menu option.",
        "", f"- Revenue: USD {result.revenue_usd:,.6f}",
        f"- Fixed energy cost: USD {result.fixed_energy_cost_usd:,.6f}",
        f"- Day-ahead energy cost: USD {result.day_ahead_cost_usd:,.6f}",
        f"- Profit: USD {result.profit_usd:,.6f}",
        f"- Certified profit interval: [{result.lower_bound_profit_usd:,.6f}, "
        f"{result.upper_bound_profit_usd:,.6f}] USD",
        f"- Absolute optimality gap: USD {result.absolute_gap_usd:.8g}",
        f"- Relative loss gap: {100 * result.relative_gap:.8f}%",
        "", "contract_menu.csv and contract_menu.json retain full numerical precision. "
        "summary.json records feasibility residuals, provenance, and solver bounds.", "",
    ])
    _atomic_text(path, lambda stream: stream.write("\n".join(lines)))


def _write_schedule(path: Path, result: OptimizationResult, config: AppConfig) -> None:
    rows = []
    e, inner, p = result.energy, result.inner, config.parameters
    for s in range(p.horizon_slots):
        rows.append({
            "slot": s + 1, "hour_start": s * p.slot_hours,
            "immediate_tokens": float(result.workload.opt_out_tokens_by_arrival[s]),
            **{f"scheduled_type_{i + 1}_tokens": float(value)
               for i, value in enumerate(inner.scheduled_tokens_by_type_slot[:, s])},
            "total_tokens": float(inner.total_tokens_by_slot[s]),
            "capacity_tokens": p.token_capacity_per_slot,
            "facility_energy_kwh": float(inner.facility_energy_kwh[s]),
            "ppa_kwh": float(e.ppa_kwh[s]), "gas_turbine_kwh": float(e.gas_turbine_kwh[s]),
            "pv_kwh": float(e.pv_kwh[s]), "bess_discharge_kwh": float(e.bess_discharge_kwh[s]),
            "bess_charge_kwh": float(e.bess_charge_kwh[s]),
            "day_ahead_energy_kwh": float(inner.day_ahead_energy_kwh[s]),
            "spill_energy_kwh": float(inner.spill_energy_kwh[s]),
            "day_ahead_price_usd_per_kwh": float(e.day_ahead_price_usd_per_kwh[s]),
            "day_ahead_cost_usd": float(e.day_ahead_price_usd_per_kwh[s] * inner.day_ahead_energy_kwh[s]),
        })
    _csv(path, rows)


def _write_plots(output: Path, result: OptimizationResult, config: AppConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = menu_rows(result, config)
    types = [row["type"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key, label, color in [
        (axes[0], "max_deferral_hours", "Maximum deferral (hours)", "#2563eb"),
        (axes[1], "discount_percent", "Discount (%)", "#0f766e"),
    ]:
        values = [row[key] for row in rows]
        bars = ax.bar(types, values, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set(xlabel="Contract type (0 = immediate)", ylabel=label)
        ax.set_xticks(types)
        ax.set_ylim(0, max(max(values) * 1.18, 1))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Derived contract menu" + ("" if result.converged else " | best feasible"))
    fig.savefig(output / "contract_menu.png", dpi=160)
    plt.close(fig)

    p, inner, e = config.parameters, result.inner, result.energy
    hours = np.arange(p.horizon_slots) * p.slot_hours
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(hours, inner.total_tokens_by_slot / 1e6, label="Executed")
    ax.plot(hours, (result.workload.opt_out_tokens_by_arrival
                   + result.workload.participating_tokens_by_type_arrival.sum(axis=0)) / 1e6,
            alpha=.6, label="Arrivals")
    ax.axhline(p.token_capacity_per_slot / 1e6, linestyle="--", color="red", label="Capacity")
    ax.set(title="Workload schedule", xlabel="Hour", ylabel="Million tokens / slot")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(hours, inner.facility_energy_kwh + e.bess_charge_kwh, label="Demand incl. BESS charge")
    ax.plot(hours, e.fixed_supply_kwh, label="Fixed supply")
    ax.fill_between(hours, inner.day_ahead_energy_kwh, alpha=.3, label="Grid import")
    ax.plot(hours, inner.spill_energy_kwh, linestyle=":", label="Spill")
    ax.set(title="Energy balance", xlabel="Hour", ylabel="kWh / slot")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.step(hours, e.day_ahead_price_usd_per_kwh, where="post")
    ax.set(title="Day-ahead price", xlabel="Hour", ylabel="USD / kWh")
    ax = axes[1, 1]
    ax.plot([h.iteration for h in result.history],
            [h.lower_bound_loss_usd for h in result.history], label="Certified lower bound")
    ax.plot([h.iteration for h in result.history],
            [h.incumbent_loss_usd for h in result.history], label="Feasible upper bound")
    ax.set(title=f"Optimization gap: USD {result.absolute_gap_usd:.6f}",
           xlabel="Iteration", ylabel="Discount cost + grid cost (USD)")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"{result.algorithm} | {result.workload_scenario_id} / {result.energy_scenario_id}")
    fig.savefig(output / "solution_dashboard.png", dpi=160)
    plt.close(fig)


def write_result_bundle(result: OptimizationResult, config: AppConfig,
                        output_directory: Path | None = None) -> OptimizationResult:
    output = output_directory if output_directory is not None else create_run_directory(config)
    output.mkdir(parents=True, exist_ok=True)
    rows = menu_rows(result, config)
    _csv(output / "contract_menu.csv", rows)
    _atomic_text(output / "contract_menu.json", lambda f: json.dump(rows, f, indent=2, allow_nan=False))
    _write_menu_document(output / "contract_menu.md", result, config)
    _atomic_text(output / "summary.json",
                 lambda f: json.dump(_summary(result, config), f, indent=2, allow_nan=False))
    _atomic_text(output / "effective_config.yaml",
                 lambda f: yaml.safe_dump(_effective_config(config), f, sort_keys=False))
    _write_schedule(output / "slot_schedule.csv", result, config)
    _csv(output / "iteration_history.csv", [asdict(item) for item in result.history],
         list(result.history[0].__dataclass_fields__) if result.history else ["iteration"])
    if config.output.write_group_schedule:
        inner = FluidInnerSolver(result.workload, result.energy, config.parameters, config.solver)
        groups = inner.recover_groups(result.inner, result.menu.deadlines_slots)
        group_rows = []
        for i, a, s in np.argwhere(groups > 0):
            mass = result.workload.participating_tokens_by_type_arrival[i, a]
            group_rows.append({
                "type": int(i + 1), "arrival_slot": int(a + 1),
                "deadline_slot": int(min(a + 1 + result.menu.deadlines_slots[i], inner.S)),
                "execution_slot": int(s + 1), "tokens": float(groups[i, a, s]),
                "group_fraction": float(groups[i, a, s] / mass) if mass else 0.0,
            })
        _csv(output / "group_schedule.csv", group_rows,
             ["type", "arrival_slot", "deadline_slot", "execution_slot", "tokens", "group_fraction"])
    if config.output.make_plot:
        _write_plots(output, result, config)
    return replace(result, output_directory=output)


def _benchmark_summary(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "benchmark_id": result.benchmark_id,
        "description": result.description,
        "status": "feasible" if result.feasible else "infeasible",
        "message": result.message,
        "objective": {
            "profit_usd": result.profit_usd,
            "revenue_usd": result.revenue_usd,
            "discount_cost_usd": result.discount_cost_usd,
            "fixed_energy_cost_usd": result.fixed_energy_cost_usd,
            "day_ahead_cost_usd": result.day_ahead_cost_usd,
        },
        "service": {
            "shifted_tokens": result.shifted_tokens,
            "mean_delay_hours": result.mean_delay_hours,
            "maximum_delay_hours": result.maximum_delay_hours,
        },
        "policy": result.metadata,
    }


def _write_benchmark_schedule(
    path: Path, result: BenchmarkResult, config: AppConfig, energy
) -> None:
    if not result.feasible or result.executed_tokens_by_slot is None:
        return
    rows = []
    p = config.parameters
    dropped = np.asarray(
        result.metadata.get("dropped_tokens_by_slot", np.zeros(p.horizon_slots)),
        dtype=float,
    )
    for s in range(p.horizon_slots):
        rows.append({
            "slot": s + 1,
            "hour_start": s * p.slot_hours,
            "arrival_tokens": float(result.arrival_tokens_by_slot[s]),
            "executed_tokens": float(result.executed_tokens_by_slot[s]),
            "dropped_tokens": float(dropped[s]),
            "capacity_tokens": p.token_capacity_per_slot,
            "facility_energy_kwh": float(result.facility_energy_kwh[s]),
            "day_ahead_energy_kwh": float(result.day_ahead_energy_kwh[s]),
            "spill_energy_kwh": float(result.spill_energy_kwh[s]),
            "day_ahead_price_usd_per_kwh": float(energy.day_ahead_price_usd_per_kwh[s]),
            "day_ahead_cost_usd": float(
                energy.day_ahead_price_usd_per_kwh[s] * result.day_ahead_energy_kwh[s]
            ),
        })
    _csv(path, rows)


def _write_workload_comparison_plot(
    output: Path,
    proposed: OptimizationResult,
    benchmarks: tuple[BenchmarkResult, ...],
    config: AppConfig,
) -> None:
    """Compare arrivals and executed workload for all three methods."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    p = config.parameters
    hours = np.arange(p.horizon_slots) * p.slot_hours
    arrivals = (
        proposed.workload.opt_out_tokens_by_arrival
        + proposed.workload.participating_tokens_by_type_arrival.sum(axis=0)
    )
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    ax.plot(hours, proposed.inner.total_tokens_by_slot / 1e6,
            color="#2563eb", linewidth=2.0, label="Proposed method")
    colors = ("#0f766e", "#dc2626")
    labels = {
        "voluntary_flat_50_24": "Voluntary Flat-50/24",
        "immediate_zero_discount_drop_overflow": "0%-discount drop-overflow",
    }
    for result, color in zip(benchmarks, colors):
        if result.feasible and result.executed_tokens_by_slot is not None:
            ax.plot(hours, result.executed_tokens_by_slot / 1e6,
                    color=color, linewidth=1.8,
                    label=labels.get(result.benchmark_id, result.benchmark_id))
    # Draw arrivals last so the curve remains visible when same-slot service is unconstrained.
    ax.plot(hours, arrivals / 1e6, color="#111827", linewidth=1.8,
            linestyle="--", label="Arrivals", zorder=5)
    ax.axhline(p.token_capacity_per_slot / 1e6, color="#6b7280", linewidth=1.2,
               linestyle=":", label="Capacity")
    ax.set(
        title=("Workload schedule comparison | "
               f"{proposed.workload_scenario_id} / {proposed.energy_scenario_id}"),
        xlabel="Hour of day",
        ylabel="Million tokens per slot",
    )
    ax.grid(alpha=.2)
    ax.legend(ncol=2, frameon=False)
    fig.savefig(output / "workload_schedule_comparison.png", dpi=180)
    fig.savefig(output / "workload_schedule_comparison.pdf")
    plt.close(fig)


def write_profit_scenario_comparisons(
    output: Path,
    workload_rows: list[dict],
    energy_rows: list[dict],
) -> None:
    """Write financial tables and six figures, each covering both scenario sweeps."""
    _csv(output / "profit_by_workload_scenario.csv", workload_rows)
    _csv(output / "profit_by_energy_scenario.csv", energy_rows)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import logging
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    methods = (
        ("proposed", "Proposed method", "#2563eb", "o"),
        ("flat_50_24", "Voluntary Flat-50/24", "#0f766e", "s"),
        ("zero_discount", "0% discount; overflow dropped", "#dc2626", "^"),
    )
    metrics = (
        ("profit", "AI data center profit"),
        ("discount_expenditure", "Discount expenditure"),
        ("day_ahead_electricity_cost", "Day-ahead electricity cost"),
    )

    def draw(ax, rows: list[dict], scenario_key: str, metric: str,
             suffix: str, unit: str, title: str) -> bool:
        x = np.arange(len(rows))
        has_limited = False
        has_missing = False
        for prefix, label, color, marker in methods:
            key = f"{prefix}_{metric}_{suffix}"
            values = np.array([
                np.nan if row.get(key) in (None, "") else float(row[key])
                for row in rows
            ])
            has_missing |= bool(np.isnan(values).any())
            ax.plot(x, values, label=label, color=color, marker=marker,
                    linewidth=2.0, markersize=5.5)
            limited = [i for i, row in enumerate(rows) if prefix == "proposed"
                       and row.get("proposed_converged") in (False, "False")
                       and np.isfinite(values[i])]
            if limited:
                has_limited = True
                ax.scatter(x[limited], values[limited], facecolors="white",
                           edgecolors=color, s=80, zorder=4)
        ax.set_xticks(x, [str(row[scenario_key]) for row in rows], rotation=35, ha="right")
        xlabel = "Workload scenario" if scenario_key == "workload_scenario_id" else "Power profile"
        ax.set(title=title, xlabel=xlabel, ylabel=unit)
        ax.grid(axis="y", alpha=.22)
        if has_missing:
            ax.text(
                .01, .01, "Missing values: infeasible case or evaluation error.",
                transform=ax.transAxes, fontsize=12, color="#4b5563",
            )
        return has_limited

    fixed_energy = workload_rows[0]["energy_scenario_id"]
    fixed_workload = energy_rows[0]["workload_scenario_id"]
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 16,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        for suffix, unit in (("usd", "USD"), ("usd_per_hour", "USD/hour")):
            for metric, title in metrics:
                fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)
                workload_limited = draw(axes[0], workload_rows, "workload_scenario_id",
                                        metric, suffix, unit,
                                        f"Workload scenarios (power profile: {fixed_energy})")
                energy_limited = draw(axes[1], energy_rows, "energy_scenario_id",
                                      metric, suffix, unit,
                                      f"Power profiles (workload: {fixed_workload})")
                figure_title = title + (" over planning horizon" if suffix == "usd" else " per hour")
                if workload_limited or energy_limited:
                    figure_title += "\nOpen markers: best feasible values; optimality not certified"
                fig.suptitle(figure_title, fontsize=16)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=16)
                for extension in ("png", "pdf"):
                    fig.savefig(output / f"scenario_{metric}_{suffix}.{extension}",
                                dpi=300, bbox_inches="tight")
                plt.close(fig)


def write_method_comparison(
    output: Path,
    proposed: OptimizationResult,
    benchmarks: tuple[BenchmarkResult, ...],
    config: AppConfig,
) -> None:
    """Write operator-profit comparison and transparent benchmark diagnostics."""
    proposed_discount = proposed.workload.total_bill_usd - proposed.revenue_usd
    rows = [{
        "method": "proposed_method",
        "status": "feasible",
        "profit_usd": proposed.profit_usd,
        "proposed_minus_method_profit_usd": 0.0,
        "revenue_usd": proposed.revenue_usd,
        "discount_cost_usd": proposed_discount,
        "fixed_energy_cost_usd": proposed.fixed_energy_cost_usd,
        "day_ahead_cost_usd": proposed.day_ahead_cost_usd,
        "shifted_tokens": "",
        "mean_delay_hours": "",
        "maximum_delay_hours": "",
        "message": proposed.termination_reason,
    }]
    for result in benchmarks:
        difference = (
            proposed.profit_usd - result.profit_usd
            if result.profit_usd is not None else None
        )
        rows.append({
            "method": result.benchmark_id,
            "status": "feasible" if result.feasible else "infeasible",
            "profit_usd": result.profit_usd,
            "proposed_minus_method_profit_usd": difference,
            "revenue_usd": result.revenue_usd,
            "discount_cost_usd": result.discount_cost_usd,
            "fixed_energy_cost_usd": result.fixed_energy_cost_usd,
            "day_ahead_cost_usd": result.day_ahead_cost_usd,
            "shifted_tokens": result.shifted_tokens,
            "mean_delay_hours": result.mean_delay_hours,
            "maximum_delay_hours": result.maximum_delay_hours,
            "message": result.message,
        })
    _csv(output / "method_comparison.csv", rows)
    payload = {
        "schema_version": "1.0",
        "workload_scenario_id": proposed.workload_scenario_id,
        "energy_scenario_id": proposed.energy_scenario_id,
        "primary_metric": "AI data center operator profit in USD",
        "proposed": rows[0],
        "benchmarks": [_benchmark_summary(result) for result in benchmarks],
    }
    _atomic_text(
        output / "benchmark_summary.json",
        lambda stream: json.dump(payload, stream, indent=2, allow_nan=False),
    )
    for result in benchmarks:
        _write_benchmark_schedule(
            output / f"{result.benchmark_id}_slot_schedule.csv", result, config, proposed.energy
        )
        outcomes = result.metadata.get("type_outcomes")
        if outcomes:
            _csv(output / f"{result.benchmark_id}_type_acceptance.csv", outcomes)
    if config.output.make_plot:
        _write_workload_comparison_plot(output, proposed, benchmarks, config)

    lines = [
        "# AI data center operator profit comparison",
        "",
        f"{proposed.workload_scenario_id} / {proposed.energy_scenario_id} | {proposed.energy_date}",
        "",
        "| Method | Status | Revenue (USD) | Discount cost (USD) | Day-ahead cost (USD) | Profit (USD) | Proposed minus method (USD) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def number(value):
            return "N/A" if value is None or value == "" else f"{float(value):,.6f}"
        lines.append(
            f"| {row['method']} | {row['status']} | {number(row['revenue_usd'])} | "
            f"{number(row['discount_cost_usd'])} | {number(row['day_ahead_cost_usd'])} | "
            f"{number(row['profit_usd'])} | {number(row['proposed_minus_method_profit_usd'])} |"
        )
    lines.extend([
        "",
        "An infeasible benchmark receives no imputed profit. Its diagnostic is retained in "
        "benchmark_summary.json and method_comparison.csv.",
        "",
    ])
    _atomic_text(output / "method_comparison.md", lambda stream: stream.write("\n".join(lines)))


def print_method_comparison(
    proposed: OptimizationResult, benchmarks: tuple[BenchmarkResult, ...], output: Path
) -> None:
    print("\nAI DATA CENTER OPERATOR PROFIT COMPARISON")
    print(f"{'Method':<39} {'Status':<11} {'Profit (USD)':>18}")
    print(f"{'proposed_method':<39} {'feasible':<11} {proposed.profit_usd:>18,.6f}")
    for result in benchmarks:
        profit = f"{result.profit_usd:,.6f}" if result.profit_usd is not None else "N/A"
        status = "feasible" if result.feasible else "infeasible"
        print(f"{result.benchmark_id:<39} {status:<11} {profit:>18}")
    print(f"Comparison: {output / 'method_comparison.md'}")
