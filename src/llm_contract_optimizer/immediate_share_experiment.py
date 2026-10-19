"""Paired immediate-service-share experiments with frozen arrivals and power."""
from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

import numpy as np
import yaml

from llm_contract_optimizer.benchmarks import voluntary_flat_50_24, immediate_zero_discount_with_overflow
from llm_contract_optimizer.config import load_config, validate_config
from llm_contract_optimizer.data import load_energy_scenario, load_workload_aggregate
from llm_contract_optimizer.exceptions import DataValidationError, NoFeasibleSolutionError, OptimizerError
from llm_contract_optimizer.horizon_experiment import METHODS, _row, _normalize, _check_proposed
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.preference_audit import audit_rows
from llm_contract_optimizer.preferences import calibrate_probabilities, balanced_integer_labels, largest_remainder
from llm_contract_optimizer.reporting import (
    _atomic_text, _csv, _effective_config, _benchmark_summary, _write_benchmark_schedule,
    write_method_comparison, write_result_bundle,
)
from llm_contract_optimizer.sampling import read_sampling_source, source_digest, validate_known_types
from llm_contract_optimizer.solve import solve_aggregate

SHARE_PERCENTAGES = (10, 20, 30, 40, 50)


def _relative(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


def _label_digest(known):
    return hashlib.sha256(json.dumps(known, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def controlled_settings(config):
    raw = _effective_config(config)
    return {"parameters": raw["parameters"], "solver": raw["solver"],
            "inputs": {k: v for k, v in raw["inputs"].items()
                       if k not in ("root", "workload_file", "immediate_execution_share")}}


def assign_share_types(tasks, counts, anchor_known, share):
    """Change the immediate quota and keep the anchor's conditional type-count mix.

    The existing common-log-odds calibration supplies source-level weights.
    Balanced integer assignment preserves every source occurrence count and
    imposes the anchor's relative participating-type quotas at the new share.
    """
    if anchor_known.get("target_immediate_share") != .5:
        raise DataValidationError("Immediate-share experiments require a 50% default anchor.")
    if isinstance(share, bool) or not np.isfinite(share) or not 0 <= share < 1:
        raise DataValidationError("Immediate share must lie in [0, 1).")
    original = np.asarray(anchor_known["counts_by_source_type"], dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if (original.shape != (len(tasks), len(tasks[0].probabilities))
            or not np.array_equal(original.sum(axis=1), counts)):
        raise DataValidationError("The default label table does not match the fixed task copies.")
    if share == .5:
        return json.loads(json.dumps(anchor_known))
    source = np.asarray([task.probabilities for task in tasks])
    calibrated = calibrate_probabilities(source, share, counts)
    expected = counts[:, None] * calibrated
    total = int(counts.sum())
    immediate = int(np.floor(total * share + .5))
    participating = original[:, 1:].sum(axis=0)
    mix = participating / participating.sum()
    target = np.r_[immediate, largest_remainder(total - immediate, mix)]
    labels = balanced_integer_labels(expected, counts, target, anchor_known["seed"])
    return {"schema": "fixed_task_types_v1", "method": "calibrated_immediate_share_fixed_participating_mix",
            "seed": anchor_known["seed"], "target_immediate_share": share,
            "target_distribution": np.r_[share, (1 - share) * mix].tolist(),
            "scenario_counts": target.tolist(), "actual_distribution": (target / total).tolist(),
            "counts_by_source_type": labels.tolist(),
            "copy_mapping": "Zero-based copy IDs partitioned into consecutive type blocks 0 through I."}


def admit_immediate_overflow(workload, capacity):
    """Drop unavoidable type-0 slot overflow; retain flexible arrivals for scheduling.

    Bills and tokens receive the same admission fraction. Type counts continue
    to describe the input task cohort, as in the existing fluid benchmarks.
    """
    if not np.isfinite(capacity) or capacity <= 0:
        raise DataValidationError("Admission requires positive finite slot capacity.")
    arrivals = workload.opt_out_tokens_by_arrival
    admitted = np.minimum(arrivals, capacity)
    dropped = arrivals - admitted
    admission = {"policy": "pro_rata_immediate_overflow_drop_keep_flexible_work",
                 "dropped_tokens": float(dropped.sum()),
                 "drop_fraction": float(dropped.sum() / workload.total_tokens) if workload.total_tokens else 0.,
                 "dropped_tokens_by_slot": dropped.tolist(), "dropped_undiscounted_bill_usd": 0.,
                 "task_count_definition": "Input task occurrences before admission; token/bill admission is fluid."}
    if not np.any(dropped):
        return workload, admission
    fractions = np.divide(admitted, arrivals, out=np.ones_like(admitted), where=arrivals > 0)
    bills = workload.opt_out_bill_usd_by_arrival
    if bills is None:
        bills = workload.opt_out_bill_usd * arrivals / arrivals.sum()
    served_bills = bills * fractions
    admission["dropped_undiscounted_bill_usd"] = float((bills * (1 - fractions)).sum())
    return replace(workload, opt_out_tokens_by_arrival=admitted,
                   opt_out_bill_usd=float(served_bills.sum()), opt_out_bill_usd_by_arrival=served_bills,
                   input_provenance={**workload.input_provenance, "admission": admission}), admission


def prepare_immediate_share_experiment(args):
    config = load_config(args.config)
    if config.parameters.horizon_slots * config.parameters.slot_seconds != 86400:
        raise DataValidationError("Immediate-share preparation requires the default 24-hour horizon.")
    if config.inputs.immediate_execution_share != .5 or config.inputs.type_distribution is not None:
        raise DataValidationError("Use the 50% default anchor without a separately forced type_distribution.")
    recipe_path = config.inputs.workload_file
    if recipe_path is None or not recipe_path.name.endswith(".sample.json"):
        raise DataValidationError("The default needs an existing fixed-label sampling recipe.")
    original = json.loads(recipe_path.read_text(encoding="utf-8"))
    raw, counts = read_sampling_source(recipe_path, config.inputs.workload_scenario_id, config.parameters)
    validate_known_types(original, counts, len(config.parameters.type_lambdas), .5)
    output, configs = args.data_dir.resolve(), args.config_dir.resolve()
    if (output / "experiment_manifest.json").exists() or any(configs.glob("S*.yaml")):
        raise DataValidationError("Prepared cases already exist. Use new data/config directories to preserve them.")
    output.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    source_path = (recipe_path.parent / original["source_file"]).resolve()
    source = output / "source_tasks.csv.gz"
    shutil.copyfile(source_path, source)
    energy_path = output / "energy_scenarios.csv"
    shutil.copyfile(config.inputs.root / "energy_scenarios.csv", energy_path)
    settings = _effective_config(config)
    cases, distributions, files = [], [], [source, energy_path]
    for percentage in SHARE_PERCENTAGES:
        share = percentage / 100
        case_dir = output / f"S{percentage:02d}"
        case_dir.mkdir(exist_ok=True)
        payload = json.loads(json.dumps(original))
        payload.update(source_file=_relative(source, case_dir), source_sha256=source_digest(source),
                       known_types=assign_share_types(raw.tasks, counts, original["known_types"], share))
        recipe = case_dir / recipe_path.name
        _atomic_text(recipe, lambda stream: json.dump(payload, stream, separators=(",", ":")))
        case_settings = json.loads(json.dumps(settings))
        case_settings["inputs"].update(root=_relative(output, configs),
                                       workload_file=_relative(recipe, configs),
                                       immediate_execution_share=share)
        case_settings["output"]["root"] = _relative(config.source_path.parents[1] / "results/immediate_shares", configs)
        config_path = configs / f"S{percentage:02d}.yaml"
        _atomic_text(config_path, lambda stream: yaml.safe_dump(case_settings, stream, sort_keys=False))
        case = load_config(config_path)
        workload = load_workload_aggregate(output, case.inputs.workload_scenario_id, case.parameters, recipe, share)
        energy = load_energy_scenario(output, case.inputs.energy_scenario_id, case.parameters.horizon_slots)
        reason = FluidInnerSolver(workload, energy, case.parameters, case.solver).screen_infeasibility()
        for row in audit_rows(raw, counts, payload, config.parameters):
            distributions.append({"immediate_share_percent": percentage, **row})
        immediate = workload.opt_out_tokens_by_arrival
        cases.append({"immediate_share_percent": percentage, "immediate_service_share": share,
                      "config": _relative(config_path, output),
                      "task_count": workload.task_count, "total_tokens": workload.total_tokens,
                      "type_counts": workload.type_counts.tolist(),
                      "actual_immediate_task_share": float(workload.type_counts[0] / workload.task_count),
                      "actual_immediate_token_share": float(immediate.sum() / workload.total_tokens),
                      "peak_immediate_capacity_fraction": float(immediate.max() / case.parameters.token_capacity_per_slot),
                      "peak_immediate_slot": int(immediate.argmax()) + 1,
                      "capacity_screen": "proved_infeasible" if reason else "passed",
                      "capacity_screen_reason": reason or ""})
        files.extend((recipe, config_path))
        print(f"Prepared {percentage}% immediate tasks; capacity screen: {reason or 'passed'}", flush=True)
    distribution_path = output / "type_distributions.csv"
    _csv(distribution_path, distributions)
    files.append(distribution_path)
    known = original["known_types"]
    manifest = {"schema": "immediate_share_experiment_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "share_percentages": list(SHARE_PERCENTAGES), "cases": cases,
                "controlled_settings": controlled_settings(config), "label_seed": known["seed"],
                "anchor_config_sha256": source_digest(config.source_path),
                "anchor_recipe": str(recipe_path), "anchor_recipe_sha256": source_digest(recipe_path),
                "anchor_source_sha256": source_digest(source_path),
                "anchor_known_types_sha256": _label_digest(known),
                "anchor_participating_type_counts": known["scenario_counts"][1:],
                "share_definition": "Fraction of sampled task occurrences; token shares are reported separately.",
                "label_rule": "Keep draws and source rows unchanged. Preserve the exact default labels at 50%. For other shares, calibrate source immediate log odds using the existing method, then balance integer labels to the immediate quota and the default conditional participating-type task proportions with the same seed.",
                "admission_rule": "Proposed: drop type-0 arrival-slot overflow pro rata, retain flexible work for scheduling. Benchmarks: preserve their existing pro-rata arrival-slot admission/drop policies. Financial metrics exclude dropped work from revenue and discount expenditure.",
                "files": {_relative(path, output): source_digest(path) for path in files}}
    _atomic_text(output / "experiment_manifest.json", lambda stream: json.dump(manifest, stream, indent=2))
    print(f"Prepared all five cases: {output / 'experiment_manifest.json'}", flush=True)
    return 0


def validate_immediate_share_suite(manifest_path, percentages):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "immediate_share_experiment_v1":
        raise DataValidationError("Unsupported immediate-share experiment manifest.")
    if (not percentages or len(set(percentages)) != len(percentages)
            or not set(percentages).issubset(manifest["share_percentages"])):
        raise DataValidationError("Select distinct prepared immediate-share percentages.")
    for name, digest in manifest["files"].items():
        file = (path.parent / name).resolve()
        if not file.is_file() or source_digest(file) != digest:
            raise DataValidationError(f"Prepared experiment file changed or is missing: {file}")
    entries = {entry["immediate_share_percent"]: entry for entry in manifest["cases"]}
    if set(entries) != set(SHARE_PERCENTAGES):
        raise DataValidationError("Prepared immediate-share configurations are missing.")
    anchor_config = load_config(path.parent / entries[50]["config"])
    anchor_payload = json.loads(anchor_config.inputs.workload_file.read_text(encoding="utf-8"))
    if _label_digest(anchor_payload["known_types"]) != manifest["anchor_known_types_sha256"]:
        raise DataValidationError("The 50% case no longer has the exact default labels.")
    baseline = load_workload_aggregate(anchor_config.inputs.root, anchor_config.inputs.workload_scenario_id,
        anchor_config.parameters, anchor_config.inputs.workload_file, .5)
    energy = load_energy_scenario(anchor_config.inputs.root, anchor_config.inputs.energy_scenario_id,
                                  anchor_config.parameters.horizon_slots)
    participating = np.asarray(manifest["anchor_participating_type_counts"], dtype=np.int64)
    cases = []
    for percentage in sorted(percentages):
        entry = entries[percentage]
        config = load_config(path.parent / entry["config"])
        if controlled_settings(config) != manifest["controlled_settings"]:
            raise DataValidationError("Non-share model, solver or scenario settings differ among cases.")
        if config.inputs.immediate_execution_share != percentage / 100:
            raise DataValidationError("The configured immediate share differs from the selected case.")
        payload = json.loads(config.inputs.workload_file.read_text(encoding="utf-8"))
        if (payload["draws"] != anchor_payload["draws"] or payload["source_sha256"] != anchor_payload["source_sha256"]
                or payload["known_types"]["seed"] != manifest["label_seed"]):
            raise DataValidationError("Task draws, source or type-assignment seed changed across shares.")
        workload = baseline if percentage == 50 else load_workload_aggregate(
            config.inputs.root, config.inputs.workload_scenario_id, config.parameters,
            config.inputs.workload_file, percentage / 100)
        if (workload.task_count != baseline.task_count or workload.total_tokens != baseline.total_tokens
                or not np.allclose(workload.opt_out_tokens_by_arrival + workload.participating_tokens_by_type_arrival.sum(axis=0),
                                   baseline.opt_out_tokens_by_arrival + baseline.participating_tokens_by_type_arrival.sum(axis=0),
                                   rtol=0, atol=1e-3)
                or not np.allclose(workload.total_bill_usd_by_arrival, baseline.total_bill_usd_by_arrival, rtol=0, atol=1e-8)):
            raise DataValidationError("Task counts, arrival tokens or undiscounted bills changed across shares.")
        remaining = workload.task_count - int(workload.type_counts[0])
        # Match preparation's normalized weights, including tied remainder order.
        if not np.array_equal(workload.type_counts[1:], largest_remainder(remaining, participating / participating.sum())):
            raise DataValidationError("Conditional participating-type task proportions changed.")
        reason = FluidInnerSolver(workload, energy, config.parameters, config.solver).screen_infeasibility()
        if (reason or "") != entry["capacity_screen_reason"]:
            raise DataValidationError("The saved capacity screen differs from the prepared inputs.")
        cases.append((entry, config, workload, energy))
    return manifest, cases


def write_immediate_share_plots(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    styles = ((METHODS[0], "Proposed method", "#2563eb", "o"),
              (METHODS[1], "Voluntary Flat-50/24", "#0f766e", "s"),
              (METHODS[2], "0% discount; overflow dropped", "#dc2626", "^"))
    metrics = (("profit", "AI data center profit"), ("discount_expenditure", "Discount expenditure"),
               ("day_ahead_electricity_cost", "Day-ahead electricity cost"))
    notes = []
    for status, label in (("infeasible", "Proposed method infeasible"), ("error", "Proposed evaluation failed")):
        shares = [str(row["immediate_share_percent"]) + "%" for row in rows
                  if row["method"] == METHODS[0] and row["status"] == status]
        if shares:
            notes.append(label + " at " + ", ".join(shares))
    if any(row["method"] == METHODS[0] and row["profit_usd"] is not None and not row["converged"] for row in rows):
        notes.append("Open markers: best feasible values; optimality not certified")
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 16,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        for suffix, unit in (("usd", "USD"), ("usd_per_hour", "USD/hour")):
            for metric, title in metrics:
                field = f"{metric}_{suffix}"
                fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
                for method, label, color, marker in styles:
                    selected = sorted((row for row in rows if row["method"] == method),
                                      key=lambda row: row["immediate_share_percent"])
                    ax.plot([row["immediate_share_percent"] for row in selected],
                            [np.nan if row[field] is None else row[field] for row in selected],
                            color=color, marker=marker, linewidth=2, label=label)
                    limited = [row for row in selected if method == METHODS[0]
                               and row[field] is not None and not row["converged"]]
                    if limited:
                        ax.scatter([row["immediate_share_percent"] for row in limited], [row[field] for row in limited],
                                   facecolors="white", edgecolors=color, s=80, zorder=4)
                ax.set(title=title + (" over 24 hours" if suffix == "usd" else " per hour"),
                       xlabel="Immediate service share of tasks (%)", ylabel=unit)
                ax.set_xticks(SHARE_PERCENTAGES)
                ax.grid(alpha=.25)
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc="outside lower center", ncol=1, fontsize=16)
                if notes:
                    fig.suptitle("\n".join(notes), fontsize=13)
                for extension in ("png", "pdf"):
                    fig.savefig(output / f"immediate_share_{metric}_{suffix}.{extension}", dpi=300, bbox_inches="tight")
                plt.close(fig)


def run_immediate_share_experiment(args):
    manifest, cases = validate_immediate_share_suite(args.manifest, args.shares)
    if args.validate_only:
        print(json.dumps({"status": "validated", "only_immediate_share_varies": True,
                          "data_summary": [case[0] for case in cases]}, indent=2))
        return 0
    base = args.output_dir.resolve() if args.output_dir else cases[0][1].output.root
    output = base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    rows, errors, infeasible = [], [], []
    try:
        snapshot = json.loads(json.dumps(manifest))
        source_root = args.manifest.resolve().parent
        snapshot["files"] = {str((source_root / name).resolve()): digest for name, digest in manifest["files"].items()}
        for entry in snapshot["cases"]:
            entry["config"] = str((source_root / entry["config"]).resolve())
        _atomic_text(output / "experiment_manifest.json", lambda stream: json.dump(snapshot, stream, indent=2))
        for entry, original_config, workload, energy in cases:
            percentage = entry["immediate_share_percent"]
            case_dir = output / f"S{percentage:02d}"
            case_dir.mkdir()
            config = replace(original_config,
                solver=replace(original_config.solver, total_time_limit_seconds=(args.time_limit
                    if args.time_limit is not None else original_config.solver.total_time_limit_seconds)),
                output=replace(original_config.output, root=case_dir,
                    make_plot=original_config.output.make_plot and not args.no_plot,
                    write_group_schedule=original_config.output.write_group_schedule and not args.no_group_schedule))
            validate_config(config)
            _atomic_text(case_dir / "effective_config.yaml",
                         lambda stream: yaml.safe_dump(_effective_config(config), stream, sort_keys=False))
            def metric_row(method):
                return {"immediate_share_percent": percentage, "immediate_service_share": percentage / 100,
                        "actual_immediate_task_share": entry["actual_immediate_task_share"],
                        "actual_immediate_token_share": entry["actual_immediate_token_share"],
                        "peak_immediate_capacity_fraction": entry["peak_immediate_capacity_fraction"],
                        "dropped_undiscounted_bill_usd": None,
                        **_row(config.parameters.horizon_slots * config.parameters.slot_hours, method, workload)}
            print(f"Running {percentage}% immediate-service comparison...", flush=True)
            proposed, proposed_row = None, metric_row(METHODS[0])
            started = time.perf_counter()
            try:
                admitted, admission = admit_immediate_overflow(workload, config.parameters.token_capacity_per_slot)
                _atomic_text(case_dir / "admission.json", lambda stream: json.dump(admission, stream, indent=2))
                proposed = solve_aggregate(config, admitted, energy)
                _check_proposed(proposed, config)
                write_result_bundle(proposed, config, case_dir)
                proposed_row.update(status="optimal_within_tolerance" if proposed.converged else "best_feasible",
                    converged=proposed.converged, profit_usd=proposed.profit_usd,
                    discount_expenditure_usd=admitted.total_bill_usd - proposed.revenue_usd,
                    day_ahead_electricity_cost_usd=proposed.day_ahead_cost_usd,
                    revenue_usd=proposed.revenue_usd, fixed_energy_cost_usd=proposed.fixed_energy_cost_usd,
                    dropped_tokens=admission["dropped_tokens"], drop_fraction=admission["drop_fraction"],
                    dropped_undiscounted_bill_usd=admission["dropped_undiscounted_bill_usd"],
                    profit_lower_bound_usd=proposed.lower_bound_profit_usd,
                    profit_upper_bound_usd=proposed.upper_bound_profit_usd, absolute_gap_usd=proposed.absolute_gap_usd,
                    message=proposed.termination_reason)
            except (OptimizerError, OSError, ValueError) as exc:
                proposed = None
                proven = isinstance(exc, NoFeasibleSolutionError)
                proposed_row.update(status="infeasible" if proven else "error", message=str(exc))
                if proven:
                    infeasible.append(percentage)
                else:
                    errors.append(f"{percentage}% proposed: {exc}")
                _atomic_text(case_dir / "failure.json", lambda stream: json.dump(proposed_row, stream, indent=2))
            proposed_row["elapsed_seconds"] = time.perf_counter() - started
            case_rows, benchmarks = [proposed_row], []
            for method, function in zip(METHODS[1:], (voluntary_flat_50_24, immediate_zero_discount_with_overflow)):
                row = metric_row(method)
                started = time.perf_counter()
                try:
                    result = function(config, workload, energy)
                    benchmarks.append(result)
                    row.update(status="feasible" if result.feasible else "infeasible",
                        dropped_tokens=result.metadata.get("dropped_tokens"), drop_fraction=result.metadata.get("drop_fraction"),
                        message=result.message)
                    if result.feasible:
                        row.update(profit_usd=result.profit_usd, discount_expenditure_usd=result.discount_cost_usd,
                            day_ahead_electricity_cost_usd=result.day_ahead_cost_usd,
                            revenue_usd=result.revenue_usd, fixed_energy_cost_usd=result.fixed_energy_cost_usd,
                            dropped_undiscounted_bill_usd=max(0., workload.total_bill_usd - result.revenue_usd - result.discount_cost_usd))
                    else:
                        errors.append(f"{percentage}% {method}: {result.message}")
                except (OptimizerError, OSError, ValueError) as exc:
                    row["message"] = str(exc)
                    errors.append(f"{percentage}% {method}: {exc}")
                row["elapsed_seconds"] = time.perf_counter() - started
                case_rows.append(row)
            if proposed is not None:
                write_method_comparison(case_dir, proposed, tuple(benchmarks), config)
            else:
                _atomic_text(case_dir / "benchmark_summary.json",
                    lambda stream: json.dump({"schema_version": "1.0", "proposed_status": proposed_row["status"],
                        "benchmarks": [_benchmark_summary(result) for result in benchmarks]},
                        stream, indent=2, allow_nan=False))
                for result in benchmarks:
                    _write_benchmark_schedule(case_dir / f"{result.benchmark_id}_slot_schedule.csv", result, config, energy)
                    if result.metadata.get("type_outcomes"):
                        _csv(case_dir / f"{result.benchmark_id}_type_acceptance.csv", result.metadata["type_outcomes"])
            for row in case_rows:
                _normalize(row)
                if proposed_row["profit_usd"] is not None and row["profit_usd"] is not None:
                    row["proposed_minus_method_profit_usd"] = proposed_row["profit_usd"] - row["profit_usd"]
            rows.extend(case_rows)
            _csv(output / "immediate_share_metrics.csv", rows)
            _atomic_text(case_dir / "immediate_share_metrics.json",
                         lambda stream: json.dump(case_rows, stream, indent=2, allow_nan=False))
            print(f"{percentage}%: {proposed_row['status']}; profit={proposed_row['profit_usd']}; {proposed_row['message']}", flush=True)
        distribution_name = next(name for name in manifest["files"] if Path(name).name == "type_distributions.csv")
        with (source_root / distribution_name).open(encoding="utf-8", newline="") as stream:
            _csv(output / "type_distributions.csv", list(csv.DictReader(stream)))
        if not args.no_plot:
            write_immediate_share_plots(output, rows)
        feasible_rows = [row for row in rows if row["method"] == METHODS[0] and row["profit_usd"] is not None]
        summary = {"status": "completed_with_errors" if errors else "completed_with_infeasible_cases" if infeasible else "completed",
                   "errors": errors, "share_percentages": [case[0]["immediate_share_percent"] for case in cases],
                   "proposed_infeasible_share_percentages": infeasible,
                   "proposed_feasible_share_percentages": [row["immediate_share_percent"] for row in feasible_rows],
                   "all_feasible_proposed_converged": all(row["converged"] for row in feasible_rows),
                   "uniform_time_limit_override_seconds": args.time_limit, "metrics_file": "immediate_share_metrics.csv",
                   "implementation_sha256": {path.name: source_digest(path) for path in (
                       Path(__file__), Path(__file__).with_name("preferences.py"), Path(__file__).with_name("inner.py"),
                       Path(__file__).with_name("solve.py"), Path(__file__).with_name("benchmarks.py"))}}
        _atomic_text(output / "experiment_summary.json", lambda stream: json.dump(summary, stream, indent=2))
        print(f"Results: {output}\nProposed infeasible shares (%): {infeasible}", flush=True)
        return 2 if errors else 0
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
