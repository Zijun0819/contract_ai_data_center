"""Fixed-workload experiments with different participating-type resolutions."""
from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import io
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import yaml

from llm_contract_optimizer.benchmarks import voluntary_flat_50_24, immediate_zero_discount_with_overflow
from llm_contract_optimizer.config import load_config, validate_config
from llm_contract_optimizer.data import load_energy_scenario, load_workload_aggregate
from llm_contract_optimizer.exceptions import DataValidationError, NoFeasibleSolutionError, OptimizerError
from llm_contract_optimizer.horizon_experiment import METHODS, _row, _normalize, _check_proposed
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.menu import audit_menu_domain
from llm_contract_optimizer.preference_audit import audit_rows
from llm_contract_optimizer.preferences import balanced_integer_labels, largest_remainder
from llm_contract_optimizer.reporting import (
    _atomic_text, _csv, _effective_config, write_method_comparison, write_result_bundle,
)
from llm_contract_optimizer.sampling import read_sampling_source, source_digest, validate_known_types
from llm_contract_optimizer.solve import solve_aggregate

TYPE_COUNTS = (2, 4, 6, 8, 10)
LOGGER = logging.getLogger(__name__)


def ordered_bin_mapping(old_count, new_count):
    """Overlap of equal ordered bins; each old bin distributes all its mass.

    Integer endpoints avoid rounding ambiguities at common bin boundaries.
    This includes exact pair merging and the existing equal pair splitting.
    """
    if (isinstance(old_count, bool) or isinstance(new_count, bool)
            or not isinstance(old_count, int) or not isinstance(new_count, int)
            or old_count < 1 or new_count < 1):
        raise DataValidationError("Type counts must be positive integers.")
    return np.array([[max(0, min((i + 1) * new_count, (j + 1) * old_count)
                            - max(i * new_count, j * old_count)) / new_count
                      for j in range(new_count)] for i in range(old_count)])


def rebin_fixed_types(known, new_count):
    """Keep immediate-copy identities; redistribute only participating labels."""
    original = np.asarray(known["counts_by_source_type"], dtype=np.int64)
    old_count = original.shape[1] - 1
    mapping = ordered_bin_mapping(old_count, new_count)
    if new_count == old_count:
        return json.loads(json.dumps(known))
    expected = original[:, 1:] @ mapping
    row_counts = original[:, 1:].sum(axis=1)
    weights = expected.sum(axis=0)
    distribution = np.concatenate(([original[:, 0].sum()], weights)) / int(original.sum())
    immediate_share = known.get("target_immediate_share", .5)
    # An odd task total rounds the immediate quota by half a task. The target
    # distribution must retain its exact share, rather than the realized quota.
    if not np.isclose(distribution[0], immediate_share, rtol=0, atol=1e-10):
        distribution = np.r_[immediate_share, (1 - immediate_share) * weights / weights.sum()]
        target = largest_remainder(int(row_counts.sum()), distribution[1:])
    else:
        target = largest_remainder(int(row_counts.sum()), weights)
    participating = balanced_integer_labels(expected, row_counts, target, known["seed"])
    labels = np.column_stack((original[:, 0], participating))
    totals = labels.sum(axis=0)
    total = int(totals.sum())
    return {"schema": "fixed_task_types_v1", "method": "ordered_bin_rebin_balanced_integer_rounding",
            "seed": known["seed"], "target_immediate_share": immediate_share,
            "target_distribution": distribution.tolist(), "scenario_counts": totals.tolist(),
            "actual_distribution": (totals / total).tolist(), "counts_by_source_type": labels.tolist(),
            "copy_mapping": "Zero-based copy IDs partitioned into consecutive type blocks 0 through I; original immediate-copy prefix preserved."}


def _relative(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


def _write_source(path, raw, new_count):
    matrix = ordered_bin_mapping(raw.tasks[0].probabilities.size - 1, new_count)
    names = ["scenario_id", "task_id", "arrival_slot", "input_tokens", "output_tokens", "total_tokens",
             "p_opt_out", *[f"p_type_{i}" for i in range(1, new_count + 1)]]
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(names)
                for task in raw.tasks:
                    probabilities = np.r_[task.probabilities[0], task.probabilities[1:] @ matrix]
                    writer.writerow([raw.scenario_id, task.task_id, task.arrival_slot, task.input_tokens,
                                     task.output_tokens, task.total_tokens, *[repr(float(v)) for v in probabilities]])


def controlled_settings(config):
    raw = _effective_config(config)
    return {"parameters": {k: v for k, v in raw["parameters"].items() if k != "type_lambdas"},
            "solver": raw["solver"],
            "inputs": {k: v for k, v in raw["inputs"].items() if k not in ("root", "workload_file")}}


def prepare_type_count_experiment(args):
    config = load_config(args.config)
    if config.parameters.horizon_slots * config.parameters.slot_seconds != 86400:
        raise DataValidationError("Type-count preparation requires the default 24-hour horizon.")
    if config.inputs.immediate_execution_share != .5:
        raise DataValidationError("The default must use the requested 50% immediate-task target.")
    if config.inputs.type_distribution is not None:
        raise DataValidationError("This experiment extends the default source-weight distribution; remove a separately forced type_distribution in a new anchor configuration.")
    if config.inputs.workload_file is None:
        raise DataValidationError("A fixed-label default sampling recipe is required.")
    original_path = config.inputs.workload_file
    original = json.loads(original_path.read_text(encoding="utf-8"))
    raw, counts = read_sampling_source(original_path, config.inputs.workload_scenario_id, config.parameters)
    validate_known_types(original, counts, len(config.parameters.type_lambdas), .5)
    baseline = load_workload_aggregate(config.inputs.root, config.inputs.workload_scenario_id,
        config.parameters, original_path, .5)
    output, configs = args.data_dir.resolve(), args.config_dir.resolve()
    if (output / "experiment_manifest.json").exists() or any(configs.glob("N*.yaml")):
        raise DataValidationError("Prepared cases already exist. Choose new --data-dir and --config-dir destinations to preserve them.")
    output.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    energy_path = output / "energy_scenarios.csv"
    # Copy bytes, including the default power trajectory, without changing any values.
    energy_path.write_bytes((config.inputs.root / "energy_scenarios.csv").read_bytes())
    base_raw = _effective_config(config)
    cases, distribution_rows = [], []
    for number in TYPE_COUNTS:
        case_dir = output / f"N{number:02d}"
        case_dir.mkdir(exist_ok=True)
        source = case_dir / "tasks.csv.gz"
        _write_source(source, raw, number)
        payload = json.loads(json.dumps(original))
        payload.update(source_file=source.name, source_sha256=source_digest(source),
                       known_types=rebin_fixed_types(original["known_types"], number))
        recipe = case_dir / original_path.name
        recipe.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        settings = json.loads(json.dumps(base_raw))
        settings["parameters"]["type_lambdas"] = np.linspace(.1, 2.5, number).tolist()
        settings["inputs"]["root"] = _relative(output, configs)
        settings["inputs"]["workload_file"] = _relative(recipe, configs)
        settings["output"]["root"] = _relative(config.source_path.parents[1] / "results/type_counts", configs)
        config_path = configs / f"N{number:02d}.yaml"
        config_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        case = load_config(config_path)
        workload = load_workload_aggregate(output, case.inputs.workload_scenario_id, case.parameters, recipe, .5)
        energy = load_energy_scenario(output, case.inputs.energy_scenario_id, case.parameters.horizon_slots)
        if (not np.array_equal(workload.opt_out_tokens_by_arrival, baseline.opt_out_tokens_by_arrival)
                or not np.array_equal(workload.opt_out_bill_usd_by_arrival, baseline.opt_out_bill_usd_by_arrival)
                or workload.task_count != baseline.task_count or workload.total_tokens != baseline.total_tokens
                or not np.allclose(workload.total_bill_usd_by_arrival, baseline.total_bill_usd_by_arrival, rtol=0, atol=1e-8)
                or not np.allclose(workload.participating_tokens_by_type_arrival.sum(axis=0),
                                   baseline.participating_tokens_by_type_arrival.sum(axis=0), rtol=0, atol=1e-3)):
            raise DataValidationError("Type preparation changed fixed arrivals, immediate service or billing.")
        screen = FluidInnerSolver(workload, energy, case.parameters, case.solver).screen_infeasibility()
        if screen:
            raise DataValidationError(f"{number} types fail a necessary capacity screen: {screen}")
        new_raw, new_counts = read_sampling_source(recipe, case.inputs.workload_scenario_id, case.parameters)
        for row in audit_rows(new_raw, new_counts, payload, case.parameters):
            distribution_rows.append({"participating_type_count": number,
                "lambda": 0. if row["task_type"] == 0 else case.parameters.type_lambdas[row["task_type"] - 1], **row})
        cases.append({"type_count": number, "total_classes_including_immediate": number + 1,
                      "config": _relative(config_path, output), "type_lambdas": list(case.parameters.type_lambdas),
                      "task_count": workload.task_count, "total_tokens": workload.total_tokens,
                      "type_counts": workload.type_counts.tolist(),
                      "actual_immediate_task_share": float(workload.type_counts[0] / workload.task_count),
                      "capacity_screen": "passed"})
        print(f"Prepared {number} participating types; lambda range 0.1..2.5; immediate tasks {workload.type_counts[0]:,}/{workload.task_count:,}", flush=True)
    _csv(output / "type_distributions.csv", distribution_rows)
    files = [*output.rglob("*.json"), *output.rglob("*.gz"), *output.rglob("*.csv"), *configs.glob("N*.yaml")]
    source_path = (original_path.parent / original["source_file"]).resolve()
    manifest = {"schema": "type_count_experiment_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "type_counts": list(TYPE_COUNTS), "cases": cases, "controlled_settings": controlled_settings(config),
                "anchor_config_sha256": source_digest(config.source_path),
                "anchor_recipe": str(original_path), "anchor_recipe_sha256": source_digest(original_path),
                "anchor_source": str(source_path), "anchor_source_sha256": source_digest(source_path),
                "label_rule": "Ordered-bin overlap maps the existing participating groups to N groups; balanced integer rounding preserves each source count and predefined type totals. Immediate-copy identities are unchanged. Equal-count refinement is identity, so the existing eight-type labels are retained.",
                "lambda_rule": "linspace(0.1, 2.5, N); N excludes immediate type 0",
                "immediate_share_definition": "50% of sampled task occurrences, as in the default; the exact immediate-token trajectory is also held fixed, without imposing a 50% token share.",
                "files": {_relative(path, output): source_digest(path) for path in files}}
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared all five cases: {output / 'experiment_manifest.json'}", flush=True)
    return 0


def validate_type_count_suite(manifest_path, numbers):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "type_count_experiment_v1":
        raise DataValidationError("Unsupported type-count experiment manifest.")
    if not numbers or len(set(numbers)) != len(numbers) or not set(numbers).issubset(manifest["type_counts"]):
        raise DataValidationError("Select distinct prepared type counts.")
    for name, digest in manifest["files"].items():
        file = (path.parent / name).resolve()
        if not file.is_file() or source_digest(file) != digest:
            raise DataValidationError(f"Prepared experiment file changed or is missing: {file}")
    cases = []
    fixed = None
    fixed_immediate_labels = None
    fixed_draws = None
    for entry in manifest["cases"]:
        number = entry["type_count"]
        if number not in numbers:
            continue
        config = load_config(path.parent / entry["config"])
        if controlled_settings(config) != manifest["controlled_settings"]:
            raise DataValidationError("Non-type-count model or scenario settings differ among cases.")
        if not np.array_equal(config.parameters.type_lambdas, np.linspace(.1, 2.5, number)):
            raise DataValidationError("Lambdas must be evenly spaced from 0.1 to 2.5.")
        recipe = json.loads(config.inputs.workload_file.read_text(encoding="utf-8"))
        immediate_labels = [row[0] for row in recipe["known_types"]["counts_by_source_type"]]
        workload = load_workload_aggregate(config.inputs.root, config.inputs.workload_scenario_id,
            config.parameters, config.inputs.workload_file, .5)
        energy = load_energy_scenario(config.inputs.root, config.inputs.energy_scenario_id, config.parameters.horizon_slots)
        if fixed is None:
            fixed = workload
            fixed_immediate_labels, fixed_draws = immediate_labels, recipe["draws"]
        if (recipe["draws"] != fixed_draws or immediate_labels != fixed_immediate_labels
                or workload.task_count != fixed.task_count or workload.total_tokens != fixed.total_tokens
                or not np.array_equal(workload.opt_out_tokens_by_arrival, fixed.opt_out_tokens_by_arrival)
                or not np.array_equal(workload.opt_out_bill_usd_by_arrival, fixed.opt_out_bill_usd_by_arrival)
                or not np.allclose(workload.total_bill_usd_by_arrival, fixed.total_bill_usd_by_arrival, rtol=0, atol=1e-8)):
            raise DataValidationError("Task draws, immediate identities or bills changed across type counts.")
        screen = FluidInnerSolver(workload, energy, config.parameters, config.solver).screen_infeasibility()
        if screen or not audit_menu_domain(config.parameters)["contract_domain_feasible"]:
            raise DataValidationError(f"{number} types fail a feasibility screen: {screen}")
        cases.append((entry, config, workload, energy))
    if len(cases) != len(numbers) or {case[0]["type_count"] for case in cases} != set(numbers):
        raise DataValidationError("Selected type-count configurations are missing.")
    return manifest, sorted(cases, key=lambda case: case[0]["type_count"])


def _type_count_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _plot_type_count_metric(ax, rows, field, title, unit):
    styles = ((METHODS[0], "Proposed method", "#2563eb", "o"),
              (METHODS[1], "Voluntary Flat-50/24", "#0f766e", "s"),
              (METHODS[2], "0% discount; overflow dropped", "#dc2626", "^"))
    has_limited = False
    for method, label, color, marker in styles:
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["participating_type_count"])
        ax.plot([row["participating_type_count"] for row in selected],
                [np.nan if row[field] is None else row[field] for row in selected],
                color=color, marker=marker, linewidth=2, label=label)
        limited = [row for row in selected if method == METHODS[0] and row[field] is not None and not row["converged"]]
        if limited:
            has_limited = True
            ax.scatter([row["participating_type_count"] for row in limited], [row[field] for row in limited],
                       facecolors="white", edgecolors=color, s=80, zorder=4)
    ax.set(title=title, xlabel="Number of participating types", ylabel=unit)
    ax.set_xticks(TYPE_COUNTS)
    ax.grid(alpha=.25)
    return has_limited


def write_type_count_hourly_plots(output, rows):
    """Export three separate figures from the already normalized USD/hour fields.

    This can also refresh completed results without rerunning the optimizer.
    """
    plt = _type_count_pyplot()
    metrics = (("profit_usd_per_hour", "Profit per hour", "profit"),
               ("discount_expenditure_usd_per_hour", "Discount expenditure per hour", "discount_expenditure"),
               ("day_ahead_electricity_cost_usd_per_hour", "Day-ahead electricity cost per hour", "day_ahead_electricity_cost"))
    for field, title, name in metrics:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        limited = _plot_type_count_metric(ax, rows, field, title, "USD/hour")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=1, fontsize=16)
        if limited:
            fig.suptitle("Open markers: best feasible values; optimality not certified", fontsize=16)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"type_count_{name}_per_hour.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_type_count_plots(output, rows, distribution_rows):
    plt = _type_count_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for ax, (field, title) in zip(axes, (("profit_usd", "Profit over 24 hours"),
            ("discount_expenditure_usd", "Discount expenditure"),
            ("day_ahead_electricity_cost_usd", "Day-ahead electricity cost"))):
        _plot_type_count_metric(ax, rows, field, title, "USD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=16)
    if any(row["method"] == METHODS[0] and row["profit_usd"] is not None and not row["converged"] for row in rows):
        fig.suptitle("Open markers: best feasible values; optimality not certified", fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"type_count_metrics.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for number in TYPE_COUNTS:
        selected = sorted((row for row in distribution_rows if int(row["participating_type_count"]) == number
                            and int(row["task_type"]) > 0), key=lambda row: int(row["task_type"]))
        if not selected:
            continue
        lambdas = [float(row["lambda"]) for row in selected]
        axes[0].plot(lambdas, [100 * float(row["actual_share"]) for row in selected], marker="o", label=f"{number} types")
        axes[1].plot(lambdas, [100 * float(row["token_share"]) for row in selected], marker="o", label=f"{number} types")
    axes[0].set(title="Participating-task distribution", ylabel="Share of all tasks (%)", xlabel="Sensitivity lambda")
    axes[1].set(title="Participating-token distribution", ylabel="Share of all tokens (%)", xlabel="Sensitivity lambda")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"type_count_distributions.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_type_count_hourly_plots(output, rows)


def run_type_count_experiment(args):
    manifest, cases = validate_type_count_suite(args.manifest, args.type_counts)
    if args.validate_only:
        print(json.dumps({"status": "validated", "type_counts": [case[0]["type_count"] for case in cases],
                          "fixed_workload_power_horizon": True, "fixed_immediate_copy_identities": True,
                          "data_summary": [case[0] for case in cases]}, indent=2))
        return 0
    base = args.output_dir.resolve() if args.output_dir else cases[0][1].output.root
    output = base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    rows, errors = [], []
    try:
        snapshot = json.loads(json.dumps(manifest))
        source_root = args.manifest.resolve().parent
        snapshot["files"] = {str((source_root / name).resolve()): digest for name, digest in manifest["files"].items()}
        for entry in snapshot["cases"]:
            entry["config"] = str((source_root / entry["config"]).resolve())
        _atomic_text(output / "experiment_manifest.json", lambda stream: json.dump(snapshot, stream, indent=2))
        for entry, original_config, workload, energy in cases:
            number = entry["type_count"]
            case_dir = output / f"N{number:02d}"
            case_dir.mkdir()
            config = replace(original_config,
                solver=replace(original_config.solver,
                    total_time_limit_seconds=args.time_limit if args.time_limit is not None else original_config.solver.total_time_limit_seconds),
                output=replace(original_config.output, root=case_dir,
                    make_plot=original_config.output.make_plot and not args.no_plot,
                    write_group_schedule=original_config.output.write_group_schedule and not args.no_group_schedule))
            validate_config(config)
            print(f"Running {number} participating types...", flush=True)
            proposed = None
            proposed_row = {"participating_type_count": number, **_row(24, METHODS[0], workload)}
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
                errors.append(f"{number} types proposed: {exc}")
                _atomic_text(case_dir / "failure.json", lambda stream: json.dump(proposed_row, stream, indent=2))
            case_rows, benchmarks = [proposed_row], []
            for method, function in zip(METHODS[1:], (voluntary_flat_50_24, immediate_zero_discount_with_overflow)):
                row = {"participating_type_count": number, **_row(24, method, workload)}
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
                        errors.append(f"{number} types {method}: {result.message}")
                except (OptimizerError, OSError, ValueError) as exc:
                    row["message"] = str(exc)
                    errors.append(f"{number} types {method}: {exc}")
                row["elapsed_seconds"] = time.perf_counter() - started
                case_rows.append(row)
            if proposed is not None:
                write_method_comparison(case_dir, proposed, tuple(benchmarks), config)
            for row in case_rows:
                _normalize(row)
                if proposed_row["profit_usd"] is not None and row["profit_usd"] is not None:
                    row["proposed_minus_method_profit_usd"] = proposed_row["profit_usd"] - row["profit_usd"]
            rows.extend(case_rows)
            _csv(output / "type_count_metrics.csv", rows)
            _atomic_text(case_dir / "type_count_metrics.json", lambda stream: json.dump(case_rows, stream, indent=2, allow_nan=False))
            print(f"{number} types: {proposed_row['status']}; profit={proposed_row['profit_usd']}; discount={proposed_row['discount_expenditure_usd']}; day-ahead cost={proposed_row['day_ahead_electricity_cost_usd']}", flush=True)
        # Read the checksummed input audit, also when using a copied run manifest.
        distribution_name = next(name for name in manifest["files"] if Path(name).name == "type_distributions.csv")
        distributions_path = (args.manifest.resolve().parent / distribution_name).resolve()
        with distributions_path.open(encoding="utf-8", newline="") as stream:
            distributions = list(csv.DictReader(stream))
        _csv(output / "type_distributions.csv", distributions)
        if not args.no_plot:
            write_type_count_plots(output, rows, distributions)
        summary = {"status": "completed" if not errors else "completed_with_errors", "errors": errors,
                   "type_counts": [case[0]["type_count"] for case in cases],
                   "all_proposed_converged": all(row["converged"] for row in rows if row["method"] == METHODS[0]),
                   "uniform_time_limit_override_seconds": args.time_limit,
                   "metrics_file": "type_count_metrics.csv",
                   "implementation_sha256": {path.name: source_digest(path) for path in (
                       Path(__file__), Path(__file__).with_name("preferences.py"),
                       Path(__file__).with_name("inner.py"), Path(__file__).with_name("solve.py"),
                       Path(__file__).with_name("benchmarks.py"))}}
        _atomic_text(output / "experiment_summary.json", lambda stream: json.dump(summary, stream, indent=2))
        print(f"Results: {output}\nAll proposed runs converged: {summary['all_proposed_converged']}", flush=True)
        return 2 if errors else 0
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
