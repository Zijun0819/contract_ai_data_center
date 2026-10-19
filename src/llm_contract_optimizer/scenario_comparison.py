"""Cross-scenario financial comparisons for the proposed and fixed policies."""
from __future__ import annotations

import csv
import logging
from dataclasses import replace
from pathlib import Path

from llm_contract_optimizer.benchmarks import run_benchmarks
from llm_contract_optimizer.config import AppConfig
from llm_contract_optimizer.data import load_energy_scenario, load_workload_aggregate
from llm_contract_optimizer.domain import BenchmarkResult, OptimizationResult
from llm_contract_optimizer.exceptions import OptimizerError
from llm_contract_optimizer.menu import presolve_parameters
from llm_contract_optimizer.reporting import _csv, write_profit_scenario_comparisons
from llm_contract_optimizer.solve import solve_aggregate

LOGGER = logging.getLogger(__name__)


def _workload_scenario_ids(config: AppConfig) -> list[str]:
    current = config.inputs.workload_file
    if current is None or not current.name.endswith(".sample.json"):
        raise OptimizerError(
            "Profit sweeps require sampled workload recipe files named <scenario>.sample.json."
        )
    scenario_ids = sorted(
        path.name.removesuffix(".sample.json")
        for path in current.parent.glob("*.sample.json")
    )
    if not scenario_ids:
        raise OptimizerError(f"No sampled workload recipes found in {current.parent}.")
    return scenario_ids


def _energy_scenario_ids(config: AppConfig) -> list[str]:
    path = config.inputs.root / "energy_scenarios.csv"
    if not path.is_file():
        raise OptimizerError(f"Energy input file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        ids = list(dict.fromkeys(row["energy_scenario_id"] for row in csv.DictReader(stream)))
    if not ids:
        raise OptimizerError(f"No power scenarios found in {path}.")
    return ids


def _case_config(config: AppConfig, workload_id: str, energy_id: str) -> AppConfig:
    workload_file = config.inputs.workload_file
    if workload_file is not None and workload_file.name.endswith(".sample.json"):
        workload_file = workload_file.with_name(f"{workload_id}.sample.json")
    return replace(
        config,
        inputs=replace(
            config.inputs,
            workload_scenario_id=workload_id,
            energy_scenario_id=energy_id,
            workload_file=workload_file,
        ),
    )


def _row(
    workload_id: str,
    energy_id: str,
    proposed: OptimizationResult | None,
    benchmarks: tuple[BenchmarkResult, BenchmarkResult] | None,
    error: str = "",
    *,
    horizon_hours: float,
) -> dict:
    flat = benchmarks[0] if benchmarks else None
    zero = benchmarks[1] if benchmarks else None
    row = {
        "workload_scenario_id": workload_id,
        "energy_scenario_id": energy_id,
        "proposed_status": "feasible" if proposed is not None else "infeasible_or_error",
        "proposed_profit_usd": proposed.profit_usd if proposed is not None else None,
        "proposed_message": proposed.termination_reason if proposed is not None else error,
        "flat_50_24_status": (
            "feasible" if flat is not None and flat.feasible else "infeasible_or_error"
        ),
        "flat_50_24_profit_usd": flat.profit_usd if flat is not None else None,
        "flat_50_24_message": flat.message if flat is not None else error,
        "flat_50_24_dropped_tokens": (
            flat.metadata.get("dropped_tokens") if flat is not None else None
        ),
        "flat_50_24_drop_fraction": (
            flat.metadata.get("drop_fraction") if flat is not None else None
        ),
        "zero_discount_status": (
            "feasible" if zero is not None and zero.feasible else "infeasible_or_error"
        ),
        "zero_discount_profit_usd": zero.profit_usd if zero is not None else None,
        "zero_discount_message": zero.message if zero is not None else error,
        "zero_discount_dropped_tokens": (
            zero.metadata.get("dropped_tokens") if zero is not None else None
        ),
        "zero_discount_drop_fraction": (
            zero.metadata.get("drop_fraction") if zero is not None else None
        ),
        "error": error,
    }
    row["horizon_hours"] = horizon_hours
    row["proposed_converged"] = proposed.converged if proposed is not None else None
    for prefix, result in (("proposed", proposed), ("flat_50_24", flat), ("zero_discount", zero)):
        feasible = result is not None and (prefix == "proposed" or result.feasible)
        if feasible:
            # Benchmarks may drop work: use their served-work discounts directly,
            # rather than treating dropped-work revenue loss as a discount.
            discount = (proposed.workload.total_bill_usd - proposed.revenue_usd
                        if prefix == "proposed" else result.discount_cost_usd)
            day_ahead = result.day_ahead_cost_usd
        else:
            discount = day_ahead = None
            row[f"{prefix}_profit_usd"] = None
        row[f"{prefix}_discount_expenditure_usd"] = discount
        row[f"{prefix}_day_ahead_electricity_cost_usd"] = day_ahead
        for metric in ("profit", "discount_expenditure", "day_ahead_electricity_cost"):
            total = row[f"{prefix}_{metric}_usd"]
            row[f"{prefix}_{metric}_usd_per_hour"] = (
                total / horizon_hours if total is not None else None
            )
    return row


def run_profit_scenario_sweeps(
    config: AppConfig,
    base_proposed: OptimizationResult,
    base_benchmarks: tuple[BenchmarkResult, BenchmarkResult],
    output: Path,
) -> tuple[list[dict], list[dict]]:
    """Vary workload and power one at a time, holding the other base scenario fixed."""
    workload_ids = _workload_scenario_ids(config)
    energy_ids = _energy_scenario_ids(config)
    base_key = (config.inputs.workload_scenario_id, config.inputs.energy_scenario_id)
    horizon_hours = config.parameters.horizon_slots * config.parameters.slot_hours
    cache: dict[
        tuple[str, str],
        tuple[OptimizationResult | None, tuple[BenchmarkResult, BenchmarkResult] | None, str],
    ] = {base_key: (base_proposed, base_benchmarks, "")}
    workload_cache = {base_key[0]: base_proposed.workload}
    energy_cache = {base_key[1]: base_proposed.energy}

    def evaluate(workload_id: str, energy_id: str):
        key = (workload_id, energy_id)
        if key in cache:
            return cache[key]
        LOGGER.info("Profit sweep case %s / %s", workload_id, energy_id)
        proposed = None
        benchmarks = None
        errors: list[str] = []
        try:
            case = _case_config(config, workload_id, energy_id)
            effective = replace(case, parameters=presolve_parameters(case.parameters))
            if workload_id not in workload_cache:
                inputs = effective.inputs
                workload_cache[workload_id] = load_workload_aggregate(
                    inputs.root,
                    workload_id,
                    effective.parameters,
                    inputs.workload_file,
                    inputs.immediate_execution_share,
                    inputs.type_distribution,
                )
            if energy_id not in energy_cache:
                energy_cache[energy_id] = load_energy_scenario(
                    effective.inputs.root, energy_id, effective.parameters.horizon_slots
                )
            workload = workload_cache[workload_id]
            energy = energy_cache[energy_id]
            try:
                proposed = solve_aggregate(effective, workload, energy)
            except OptimizerError as exc:
                errors.append(f"proposed: {exc}")
            benchmarks = run_benchmarks(effective, workload, energy)
        except (OptimizerError, OSError, ValueError) as exc:
            errors.append(str(exc))
        value = (proposed, benchmarks, " | ".join(errors))
        cache[key] = value
        return value

    workload_rows = []
    for workload_id in workload_ids:
        proposed, benchmarks, error = evaluate(workload_id, base_key[1])
        workload_rows.append(_row(workload_id, base_key[1], proposed, benchmarks, error,
                                  horizon_hours=horizon_hours))

    energy_rows = []
    for energy_id in energy_ids:
        proposed, benchmarks, error = evaluate(base_key[0], energy_id)
        energy_rows.append(_row(base_key[0], energy_id, proposed, benchmarks, error,
                                horizon_hours=horizon_hours))

    if config.output.make_plot:
        write_profit_scenario_comparisons(output, workload_rows, energy_rows)
    else:
        _csv(output / "profit_by_workload_scenario.csv", workload_rows)
        _csv(output / "profit_by_energy_scenario.csv", energy_rows)
    return workload_rows, energy_rows
