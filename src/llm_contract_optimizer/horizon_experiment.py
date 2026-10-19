"""Paired planning-horizon comparisons with frozen non-horizon settings."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from llm_contract_optimizer.benchmarks import voluntary_flat_50_24, immediate_zero_discount_with_overflow
from llm_contract_optimizer.config import load_config, validate_config
from llm_contract_optimizer.data import load_energy_scenario
from llm_contract_optimizer.exceptions import DataValidationError, NoFeasibleSolutionError, OptimizerError
from llm_contract_optimizer.horizon_data import load_consecutive_workload
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.menu import audit_menu_domain
from llm_contract_optimizer.reporting import (
    _atomic_text, _csv, _effective_config, result_validation, write_method_comparison, write_result_bundle,
)
from llm_contract_optimizer.sampling import source_digest
from llm_contract_optimizer.solve import solve_aggregate

LOGGER = logging.getLogger(__name__)
METHODS = ("proposed_method", "voluntary_flat_50_24", "immediate_zero_discount_drop_overflow")


def controlled_settings(config):
    raw = _effective_config(config)
    return {"parameters": {k: v for k, v in raw["parameters"].items() if k != "horizon_slots"},
            "solver": raw["solver"],
            "inputs": {k: v for k, v in raw["inputs"].items() if k not in ("root", "workload_file")}}


def validate_suite(manifest_path, hours):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "planning_horizon_experiment_v1":
        raise DataValidationError("Unsupported planning-horizon experiment manifest.")
    if not hours or len(set(hours)) != len(hours) or not set(hours).issubset(manifest["horizons_hours"]):
        raise DataValidationError("Select distinct horizons from the prepared experiment.")
    for name, digest in manifest["files"].items():
        path = (manifest_path.parent / name).resolve()
        if not path.is_file() or source_digest(path) != digest:
            raise DataValidationError(f"Prepared experiment file changed or is missing: {path}. Rebuild a new data package to change settings.")
    cases = []
    cache = {}
    for entry in manifest["cases"]:
        if entry["hours"] not in hours:
            continue
        config = load_config(manifest_path.parent / entry["config"])
        if controlled_settings(config) != manifest["controlled_settings"]:
            raise DataValidationError("Only horizon_slots may differ among experiment model settings.")
        if config.parameters.horizon_slots * config.parameters.slot_seconds != entry["hours"] * 3600:
            raise DataValidationError("Configured horizon does not match its experiment case.")
        workload = load_consecutive_workload(config.inputs.workload_file, config.inputs.workload_scenario_id,
            config.parameters, config.inputs.immediate_execution_share, config.inputs.type_distribution, cache)
        energy = load_energy_scenario(config.inputs.root, config.inputs.energy_scenario_id, config.parameters.horizon_slots)
        screen = FluidInnerSolver(workload, energy, config.parameters, config.solver).screen_infeasibility()
        if screen:
            raise DataValidationError(f"{entry['hours']}h capacity screen failed: {screen}")
        if not audit_menu_domain(config.parameters)["contract_domain_feasible"]:
            raise DataValidationError(f"{entry['hours']}h contract domain is infeasible.")
        cases.append((entry, config, workload, energy))
    if len(cases) != len(hours) or {case[0]["hours"] for case in cases} != set(hours):
        raise DataValidationError("The manifest is missing a selected horizon configuration.")
    return manifest, sorted(cases, key=lambda case: case[0]["hours"])


def _row(hours, method, workload):
    return {"horizon_hours": hours, "method": method, "status": "error", "converged": None,
            "profit_usd": None, "discount_expenditure_usd": None, "day_ahead_electricity_cost_usd": None,
            "revenue_usd": None, "fixed_energy_cost_usd": None, "total_arrival_tokens": workload.total_tokens,
            "dropped_tokens": None, "drop_fraction": None,
            "profit_usd_per_hour": None, "discount_expenditure_usd_per_hour": None,
            "day_ahead_electricity_cost_usd_per_hour": None, "profit_usd_per_million_arrival_tokens": None,
            "profit_lower_bound_usd": None, "profit_upper_bound_usd": None,
            "absolute_gap_usd": None, "proposed_minus_method_profit_usd": None,
            "elapsed_seconds": None, "message": ""}


def _normalize(row):
    pairs = (("profit_usd", "profit_usd_per_hour"),
             ("discount_expenditure_usd", "discount_expenditure_usd_per_hour"),
             ("day_ahead_electricity_cost_usd", "day_ahead_electricity_cost_usd_per_hour"))
    for total, rate in pairs:
        if row[total] is not None:
            row[rate] = row[total] / row["horizon_hours"]
    if row["profit_usd"] is not None and row["total_arrival_tokens"]:
        row["profit_usd_per_million_arrival_tokens"] = row["profit_usd"] / (row["total_arrival_tokens"] / 1e6)


def _check_proposed(result, config):
    audit = result_validation(result, config)
    token_tolerance = max(1e-3, config.solver.token_scale * config.solver.feasibility_tolerance * 10)
    for key, value in audit.items():
        if not np.isfinite(value):
            raise DataValidationError(f"Nonfinite solution validation: {key}")
        if key.startswith("minimum_") and value < -1e-7:
            raise DataValidationError(f"IR/IC validation failed: {key}={value}")
        if key.startswith("maximum_"):
            tolerance = token_tolerance if key.endswith("_tokens") else 1e-5
            if value > tolerance:
                raise DataValidationError(f"Solution validation failed: {key}={value}")
    return audit


def write_horizon_plots(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    styles = ((METHODS[0], "Proposed method", "#2563eb", "o"),
              (METHODS[1], "Voluntary Flat-50/24", "#0f766e", "s"),
              (METHODS[2], "0% discount; overflow dropped", "#dc2626", "^"))
    metrics = (("profit_usd", "Profit"), ("discount_expenditure_usd", "Discount expenditure"),
               ("day_ahead_electricity_cost_usd", "Day-ahead electricity cost"))
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    for column, (key, title) in enumerate(metrics):
        for row_index in range(2):
            ax = axes[row_index, column]
            field = key if row_index == 0 else {"profit_usd": "profit_usd_per_hour",
                "discount_expenditure_usd": "discount_expenditure_usd_per_hour",
                "day_ahead_electricity_cost_usd": "day_ahead_electricity_cost_usd_per_hour"}[key]
            for method, label, color, marker in styles:
                selected = sorted((r for r in rows if r["method"] == method), key=lambda r: r["horizon_hours"])
                ax.plot([r["horizon_hours"] for r in selected],
                        [np.nan if r[field] is None else r[field] for r in selected],
                        color=color, marker=marker, linewidth=2, label=label)
                # Open circles identify feasible proposed values without an optimality certificate.
                limited = [r for r in selected if method == METHODS[0] and r["profit_usd"] is not None and not r["converged"]]
                if limited:
                    ax.scatter([r["horizon_hours"] for r in limited], [r[field] for r in limited],
                               facecolors="white", edgecolors=color, s=80, zorder=4)
            ax.set(title=title + (" over horizon" if row_index == 0 else " per hour"),
                   xlabel="Planning horizon (hours)", ylabel="USD" if row_index == 0 else "USD/hour")
            ax.set_xticks([12, 24, 48, 72, 96, 120, 144, 168])
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=16)
    if any(r["method"] == METHODS[0] and r["profit_usd"] is not None and not r["converged"] for r in rows):
        fig.suptitle("Open markers: best feasible values; optimality not certified", fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"planning_horizon_metrics.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_horizon_experiment(args):
    manifest, cases = validate_suite(args.manifest, args.horizons)
    if args.validate_only:
        print(json.dumps({"status": "validated", "horizons_hours": [case[0]["hours"] for case in cases],
                          "controlled_settings_identical": True, "capacity_screens": "all passed",
                          "data_summary": [case[0] for case in cases]}, indent=2))
        return 0
    base = args.output_dir.resolve() if args.output_dir else cases[0][1].output.root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = base / stamp
    output.mkdir(parents=True)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    rows, errors = [], []
    try:
        # Resolve provenance paths so the copied manifest remains usable from the run directory.
        snapshot = json.loads(json.dumps(manifest))
        source_root = args.manifest.resolve().parent
        snapshot["files"] = {str((source_root / name).resolve()): digest for name, digest in manifest["files"].items()}
        for entry in snapshot["cases"]:
            entry["config"] = str((source_root / entry["config"]).resolve())
        _atomic_text(output / "experiment_manifest.json", lambda stream: json.dump(snapshot, stream, indent=2))
        for entry, original_config, workload, energy in cases:
            hours = entry["hours"]
            case_dir = output / f"H{hours:03d}"
            case_dir.mkdir()
            config = replace(original_config,
                solver=replace(original_config.solver,
                    total_time_limit_seconds=args.time_limit if args.time_limit is not None else original_config.solver.total_time_limit_seconds),
                output=replace(original_config.output, root=case_dir,
                    make_plot=original_config.output.make_plot and not args.no_plot,
                    write_group_schedule=original_config.output.write_group_schedule and not args.no_group_schedule))
            validate_config(config)
            LOGGER.info("Planning horizon %sh: proposed method and both fixed-policy benchmarks", hours)
            print(f"Running {hours}h comparison...", flush=True)
            proposed = None
            proposed_row = _row(hours, METHODS[0], workload)
            try:
                proposed = solve_aggregate(config, workload, energy)
                _check_proposed(proposed, config)
                write_result_bundle(proposed, config, case_dir)
                proposed_row.update(status="optimal_within_tolerance" if proposed.converged else "best_feasible",
                    converged=proposed.converged, profit_usd=proposed.profit_usd,
                    discount_expenditure_usd=workload.total_bill_usd - proposed.revenue_usd,
                    day_ahead_electricity_cost_usd=proposed.day_ahead_cost_usd,
                    revenue_usd=proposed.revenue_usd, fixed_energy_cost_usd=proposed.fixed_energy_cost_usd,
                    dropped_tokens=0., drop_fraction=0., profit_lower_bound_usd=proposed.lower_bound_profit_usd,
                    profit_upper_bound_usd=proposed.upper_bound_profit_usd, absolute_gap_usd=proposed.absolute_gap_usd,
                    elapsed_seconds=proposed.elapsed_seconds, message=proposed.termination_reason)
            except (OptimizerError, OSError, ValueError) as exc:
                proposed = None
                proposed_row.update(status="infeasible" if isinstance(exc, NoFeasibleSolutionError) else "error", message=str(exc))
                errors.append(f"{hours}h proposed: {exc}")
                _atomic_text(case_dir / "failure.json", lambda stream: json.dump(proposed_row, stream, indent=2))
            case_rows = [proposed_row]
            benchmarks = []
            for method, function in zip(METHODS[1:], (voluntary_flat_50_24, immediate_zero_discount_with_overflow)):
                row = _row(hours, method, workload)
                started = time.perf_counter()
                try:
                    result = function(config, workload, energy)
                    benchmarks.append(result)
                    row.update(status="feasible" if result.feasible else "infeasible", profit_usd=result.profit_usd,
                        discount_expenditure_usd=result.discount_cost_usd,
                        day_ahead_electricity_cost_usd=result.day_ahead_cost_usd,
                        revenue_usd=result.revenue_usd, fixed_energy_cost_usd=result.fixed_energy_cost_usd,
                        dropped_tokens=result.metadata.get("dropped_tokens"), drop_fraction=result.metadata.get("drop_fraction"),
                        message=result.message)
                    if not result.feasible:
                        errors.append(f"{hours}h {method}: {result.message}")
                except (OptimizerError, OSError, ValueError) as exc:
                    row["message"] = str(exc)
                    errors.append(f"{hours}h {method}: {exc}")
                row["elapsed_seconds"] = time.perf_counter() - started
                case_rows.append(row)
            if proposed is not None:
                write_method_comparison(case_dir, proposed, tuple(benchmarks), config)
            for row in case_rows:
                _normalize(row)
                if proposed_row["profit_usd"] is not None and row["profit_usd"] is not None:
                    row["proposed_minus_method_profit_usd"] = proposed_row["profit_usd"] - row["profit_usd"]
            rows.extend(case_rows)
            _csv(output / "planning_horizon_metrics.csv", rows)
            _atomic_text(case_dir / "horizon_metrics.json", lambda stream: json.dump(case_rows, stream, indent=2, allow_nan=False))
            print(f"{hours}h: {proposed_row['status']}; profit={proposed_row['profit_usd']}; discount={proposed_row['discount_expenditure_usd']}; day-ahead cost={proposed_row['day_ahead_electricity_cost_usd']}", flush=True)
        if not args.no_plot:
            write_horizon_plots(output, rows)
        summary = {"status": "completed" if not errors else "completed_with_errors", "errors": errors,
                   "horizons_hours": [case[0]["hours"] for case in cases],
                   "all_proposed_converged": all(row["converged"] for row in rows if row["method"] == METHODS[0]),
                   "uniform_time_limit_override_seconds": args.time_limit,
                   "metrics_file": "planning_horizon_metrics.csv",
                   "implementation_sha256": {path.name: source_digest(path) for path in (
                       Path(__file__), Path(__file__).with_name("horizon_data.py"),
                       Path(__file__).with_name("inner.py"), Path(__file__).with_name("master.py"),
                       Path(__file__).with_name("solve.py"), Path(__file__).with_name("benchmarks.py"))}}
        _atomic_text(output / "experiment_summary.json", lambda stream: json.dump(summary, stream, indent=2))
        print(f"Results: {output}\nAll proposed runs converged: {summary['all_proposed_converged']}", flush=True)
        return 2 if errors else 0
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
