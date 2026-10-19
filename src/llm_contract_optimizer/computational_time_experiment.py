"""Joint type/horizon runtime experiment on exact fixed-label consecutive inputs."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import yaml

from llm_contract_optimizer.config import load_config, validate_config
from llm_contract_optimizer.data import load_energy_scenario
from llm_contract_optimizer.exceptions import DataValidationError, NoFeasibleSolutionError, OptimizerError
from llm_contract_optimizer.horizon_data import load_consecutive_workload
from llm_contract_optimizer.horizon_experiment import _check_proposed, validate_suite
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.menu import audit_menu_domain, presolve_parameters
from llm_contract_optimizer.reporting import _atomic_text, _csv, _effective_config, _summary
from llm_contract_optimizer.sampling import read_sampling_source, source_digest, validate_known_types
from llm_contract_optimizer.solve import solve_aggregate
from llm_contract_optimizer.type_count_experiment import TYPE_COUNTS, _relative, _write_source, rebin_fixed_types

LOGGER = logging.getLogger(__name__)
HORIZONS = (12, 24, 48, 72, 96, 120, 144, 168)
SCHEMA = "computational_time_experiment_v1"
TIMING_SCOPE = ("Elapsed wall-clock seconds for a fresh solve_aggregate call: configuration validation, "
                "menu presolve, model construction, feasibility screening and the proposed algorithm. "
                "Excludes input loading/preparation, warmups, post-solve validation and all output writing.")


def controlled_settings(config):
    raw = _effective_config(config)
    return {"parameters": {k: v for k, v in raw["parameters"].items()
                           if k not in ("horizon_slots", "type_lambdas")},
            "solver": raw["solver"],
            "inputs": {k: v for k, v in raw["inputs"].items() if k not in ("root", "workload_file")}}


def prepare_timing_experiment(args):
    _, anchors = validate_suite(args.horizon_manifest, list(HORIZONS))
    default = load_config(Path(__file__).resolve().parents[2] / "config/default.yaml")
    base = next(case[1] for case in anchors if case[0]["hours"] == 24)
    if (controlled_settings(default) != controlled_settings(base)
            or default.parameters.type_lambdas != base.parameters.type_lambdas
            or len(base.parameters.type_lambdas) != 8
            or base.inputs.immediate_execution_share != .5 or base.inputs.type_distribution is not None):
        raise DataValidationError("Timing preparation needs the existing eight-type, 50%-immediate default horizon package.")
    output, configs = args.data_dir.resolve(), args.config_dir.resolve()
    if any(path.exists() and any(path.iterdir()) for path in (output, configs)):
        raise DataValidationError("Timing destinations must be empty. Use new destinations to preserve prepared inputs.")
    output.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    longest_config = anchors[-1][1]
    window_path = longest_config.inputs.workload_file
    longest_window = json.loads(window_path.read_text(encoding="utf-8"))
    day_parameters = replace(base.parameters, horizon_slots=longest_window["day_slots"])
    daily_entries = {number: [] for number in TYPE_COUNTS}
    distributions, files = [], []
    for day in longest_window["days"]:
        original_path = (window_path.parent / day["recipe"]).resolve()
        original = json.loads(original_path.read_text(encoding="utf-8"))
        raw, counts = read_sampling_source(original_path, day["scenario_id"], day_parameters)
        validate_known_types(original, counts, 8, .5)
        for number in TYPE_COUNTS:
            directory = output / "daily" / f"N{number:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            source = directory / f"{day['scenario_id']}.tasks.csv.gz"
            recipe = directory / f"{day['scenario_id']}.sample.json"
            _write_source(source, raw, number)
            payload = json.loads(json.dumps(original))
            payload.update(source_file=source.name, source_sha256=source_digest(source),
                           known_types=rebin_fixed_types(original["known_types"], number))
            _atomic_text(recipe, lambda stream: json.dump(payload, stream, separators=(",", ":")))
            daily_entries[number].append({**day, "recipe": str(recipe), "recipe_sha256": source_digest(recipe)})
            known = payload["known_types"]
            for index, count in enumerate(known["scenario_counts"]):
                distributions.append({"trace_day": day["trace_day"], "scenario_id": day["scenario_id"],
                    "participating_type_count": number, "task_type": index, "task_count": count,
                    "lambda": 0.0 if index == 0 else float(np.linspace(.1, 2.5, number)[index - 1])})
            files.extend((source, recipe))
        print(f"Prepared fixed type labels for consecutive workload day {day['trace_day']}.", flush=True)

    cases = []
    for entry, original_config, workload, _ in anchors:
        hours = entry["hours"]
        horizon_dir = output / f"H{hours:03d}"
        horizon_dir.mkdir()
        energy_path = horizon_dir / "energy_scenarios.csv"
        shutil.copyfile(original_config.inputs.root / "energy_scenarios.csv", energy_path)
        files.append(energy_path)
        original_window = json.loads(original_config.inputs.workload_file.read_text(encoding="utf-8"))
        for number in TYPE_COUNTS:
            case_dir = horizon_dir / f"N{number:02d}"
            case_dir.mkdir()
            window = json.loads(json.dumps(original_window))
            window["days"] = [{**day, "recipe": _relative(Path(day["recipe"]), case_dir)}
                              for day in daily_entries[number]]
            recipe = case_dir / "workload.window.json"
            _atomic_text(recipe, lambda stream: json.dump(window, stream, indent=2))
            settings = _effective_config(original_config)
            settings["parameters"]["type_lambdas"] = np.linspace(.1, 2.5, number).tolist()
            settings["inputs"].update(root=_relative(horizon_dir, configs), workload_file=_relative(recipe, configs))
            settings["output"].update(root=_relative(default.output.root / "computational_time", configs),
                                       make_plot=False, write_group_schedule=False)
            config_path = configs / f"H{hours:03d}_N{number:02d}.yaml"
            _atomic_text(config_path, lambda stream: yaml.safe_dump(settings, stream, sort_keys=False))
            files.extend((recipe, config_path))
            cases.append({"hours": hours, "type_count": number, "config": _relative(config_path, output),
                          "task_count": workload.task_count, "total_tokens": workload.total_tokens})
    distribution_path = output / "daily_type_counts.csv"
    _csv(distribution_path, distributions)
    files.append(distribution_path)
    manifest = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons_hours": list(HORIZONS), "type_counts": list(TYPE_COUNTS), "cases": cases,
        "controlled_settings": controlled_settings(base),
        "workload_scenario_id": base.inputs.workload_scenario_id, "energy_scenario_id": base.inputs.energy_scenario_id,
        "anchor_horizon_manifest": str(args.horizon_manifest.resolve()),
        "anchor_horizon_manifest_sha256": source_digest(args.horizon_manifest),
        "anchor_default_config_sha256": source_digest(default.source_path),
        "label_rule": "Apply the existing ordered-bin rebinning and balanced integer rounding once to each complete source day. Preserve each source's immediate-copy prefix and total sampled copies. Eight-type labels remain identical to the anchor.",
        "lambda_rule": "linspace(0.1, 2.5, N); N counts participating types and excludes immediate type 0.",
        "horizon_rule": "Exact consecutive suffixes of the existing default-anchored 168h window. Shared occurrences retain their labels; the 12h slice inherits its realized immediate share.",
        "files": {_relative(path, output): source_digest(path) for path in files}}
    manifest_path = output / "experiment_manifest.json"
    _atomic_text(manifest_path, lambda stream: json.dump(manifest, stream, indent=2))
    print(f"Prepared all 40 cases: {manifest_path}", flush=True)
    return 0


def validate_timing_suite(manifest_path, hours, numbers):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise DataValidationError("Unsupported computational-time manifest.")
    for selected, allowed in ((hours, manifest["horizons_hours"]), (numbers, manifest["type_counts"])):
        if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(allowed):
            raise DataValidationError("Select distinct prepared horizons and type counts.")
    for name, digest in manifest["files"].items():
        file = (path.parent / name).resolve()
        if not file.is_file() or source_digest(file) != digest:
            raise DataValidationError(f"Prepared timing input changed or is missing: {file}")
    cases, cache, by_horizon, power_by_horizon = [], {}, {}, {}
    for entry in manifest["cases"]:
        horizon, number = entry["hours"], entry["type_count"]
        if horizon not in hours or number not in numbers:
            continue
        config = load_config(path.parent / entry["config"])
        if (controlled_settings(config) != manifest["controlled_settings"]
                or config.parameters.horizon_slots * config.parameters.slot_seconds != horizon * 3600
                or not np.array_equal(config.parameters.type_lambdas, np.linspace(.1, 2.5, number))):
            raise DataValidationError("Only planning horizon and participating-type resolution may change.")
        workload = load_consecutive_workload(config.inputs.workload_file, config.inputs.workload_scenario_id,
            config.parameters, config.inputs.immediate_execution_share, config.inputs.type_distribution, cache)
        energy = load_energy_scenario(config.inputs.root, config.inputs.energy_scenario_id, config.parameters.horizon_slots)
        if workload.task_count != entry["task_count"] or abs(workload.total_tokens - entry["total_tokens"]) > 1e-3:
            raise DataValidationError("Timing data changed its sampled task or token totals.")
        baseline = by_horizon.setdefault(horizon, workload)
        baseline_power = power_by_horizon.setdefault(horizon, energy)
        if (workload.task_count != baseline.task_count
                or not np.array_equal(workload.opt_out_tokens_by_arrival, baseline.opt_out_tokens_by_arrival)
                or not np.array_equal(workload.opt_out_bill_usd_by_arrival, baseline.opt_out_bill_usd_by_arrival)
                or not np.allclose(workload.participating_tokens_by_type_arrival.sum(axis=0),
                                   baseline.participating_tokens_by_type_arrival.sum(axis=0), rtol=0, atol=1e-3)
                or not np.allclose(workload.total_bill_usd_by_arrival, baseline.total_bill_usd_by_arrival, rtol=0, atol=1e-7)):
            raise DataValidationError("Type resolution changed immediate work, total arrivals or bills.")
        for field in ("ppa_kwh", "gas_turbine_kwh", "pv_kwh", "bess_discharge_kwh",
                      "bess_charge_kwh", "day_ahead_price_usd_per_kwh"):
            if not np.array_equal(getattr(energy, field), getattr(baseline_power, field)):
                raise DataValidationError("Type resolution changed the fixed power profile.")
        reason = FluidInnerSolver(workload, energy, config.parameters, config.solver).screen_infeasibility()
        if reason or not audit_menu_domain(config.parameters)["contract_domain_feasible"]:
            raise DataValidationError(f"{horizon}h/{number} types fails a feasibility screen: {reason}")
        cases.append((entry, config, workload, energy))
    expected = {(h, n) for h in hours for n in numbers}
    if len(cases) != len(expected) or {(e["hours"], e["type_count"]) for e, _, _, _ in cases} != expected:
        raise DataValidationError("The timing manifest must contain exactly one case per selected pair.")
    for number in numbers:
        selected = sorted((c for c in cases if c[0]["type_count"] == number), key=lambda c: c[0]["hours"])
        _, _, longest, long_energy = selected[-1]
        for _, config, workload, energy in selected:
            slots = config.parameters.horizon_slots
            if (not np.array_equal(workload.opt_out_tokens_by_arrival, longest.opt_out_tokens_by_arrival[-slots:])
                    or not np.array_equal(workload.participating_tokens_by_type_arrival,
                                          longest.participating_tokens_by_type_arrival[:, -slots:])
                    or not np.array_equal(workload.total_bill_usd_by_arrival,
                                          longest.total_bill_usd_by_arrival[-slots:])):
                raise DataValidationError("Workload horizons must be exact fixed-label consecutive suffixes.")
            for field in ("ppa_kwh", "gas_turbine_kwh", "pv_kwh", "bess_discharge_kwh",
                          "bess_charge_kwh", "day_ahead_price_usd_per_kwh"):
                if not np.array_equal(getattr(energy, field), getattr(long_energy, field)[-slots:]):
                    raise DataValidationError("Power horizons must be exact consecutive suffixes.")
    return manifest, sorted(cases, key=lambda c: (c[0]["hours"], c[0]["type_count"]))


def summarize_timings(rows, hours, numbers, repeats):
    summaries = []
    for horizon in sorted(hours):
        for number in sorted(numbers):
            selected = [r for r in rows if r["horizon_hours"] == horizon and r["participating_type_count"] == number]
            values = [r["wall_seconds"] for r in selected]
            failures = sum(r["status"] not in ("optimal_within_tolerance", "best_feasible") for r in selected)
            converged = sum(r["converged"] is True for r in selected)
            status = ("failed" if failures else "not_run" if not selected else "incomplete" if len(selected) < repeats
                      else "optimal_within_tolerance" if converged == repeats else "best_feasible")
            usable = status in ("optimal_within_tolerance", "best_feasible")
            summaries.append({"horizon_hours": horizon, "participating_type_count": number,
                "status": status, "attempts": len(selected), "expected_repeats": repeats,
                "converged_repeats": converged, "failed_repeats": failures,
                "median_seconds": float(np.median(values)) if usable else None,
                "attempt_median_seconds": float(np.median(values)) if values else None,
                "min_seconds": min(values) if values else None, "max_seconds": max(values) if values else None,
                "sample_std_seconds": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return summaries


def table_cells(summaries, hours, numbers):
    lookup = {(r["horizon_hours"], r["participating_type_count"]): r for r in summaries}
    cells = []
    for horizon in sorted(hours):
        row = [str(horizon)]
        for number in sorted(numbers):
            result = lookup[horizon, number]
            value = result["median_seconds"]
            row.append("ERR" if result["status"] == "failed" else "—" if value is None
                       else f"{value:.3f}" + ("†" if result["status"] == "best_feasible" else ""))
        cells.append(row)
    return cells


def write_timing_tables(output, rows, hours, numbers, repeats, *, plot=False):
    summaries = summarize_timings(rows, hours, numbers, repeats)
    _csv(output / "timing_runs.csv", rows)
    _csv(output / "timing_summary.csv", summaries)
    lookup = {(r["horizon_hours"], r["participating_type_count"]): r for r in summaries}
    values = [{"planning_horizon_hours": h, **{str(n): lookup[h, n]["median_seconds"] for n in sorted(numbers)}}
              for h in sorted(hours)]
    statuses = [{"planning_horizon_hours": h, **{str(n): lookup[h, n]["status"] for n in sorted(numbers)}}
                for h in sorted(hours)]
    _csv(output / "computational_time_table.csv", values)
    _csv(output / "computational_time_status.csv", statuses)
    cells = table_cells(summaries, hours, numbers)
    header = ["Planning horizon (h)", *map(str, sorted(numbers))]
    text = ["# Proposed-method computation time", "", f"Median wall-clock seconds over {repeats} measured solve(s) per pair.",
            "Columns count participating user types; immediate type 0 is additional.", "",
            "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |",
            *["| " + " | ".join(row) + " |" for row in cells], "",
            "† At least one solve returned a feasible solution without a convergence certificate. ERR: at least one attempt failed; no completed runtime is substituted. —: not yet fully measured.",
            "", TIMING_SCOPE, "", "Full-precision timings, status, gaps and iteration counts are in timing_runs.csv; computational_time_status.csv records the matrix status.", ""]
    _atomic_text(output / "computational_time_table.md", lambda stream: stream.write("\n".join(text)))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        with plt.rc_context({"font.family": "Times New Roman", "font.size": 16,
                             "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig, ax = plt.subplots(figsize=(11, max(4, .58 * len(cells) + 2)))
            ax.axis("off")
            fig.suptitle("Proposed-method computation time (seconds)", fontsize=18, y=.96)
            fig.text(.5, .89, f"Columns: participating user types | Median of {repeats} measured solve(s)",
                     ha="center", fontsize=14)
            table = ax.table(cellText=cells, colLabels=header, cellLoc="center",
                             colWidths=[.30] + [.70 / len(numbers)] * len(numbers),
                             bbox=[0, .07, 1, .90])
            table.auto_set_font_size(False)
            table.set_fontsize(16)
            for (row, column), cell in table.get_celld().items():
                cell.set_edgecolor("#94a3b8")
                cell.set_linewidth(.7)
                if row == 0:
                    cell.set_facecolor("#dbeafe")
                    cell.set_text_props(weight="bold")
                elif column == 0:
                    cell.set_facecolor("#f1f5f9")
                    cell.set_text_props(weight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#f8fafc")
            footnote = "Fresh model construction and solving; input loading and output writing excluded."
            if any(r["status"] == "best_feasible" for r in summaries):
                footnote += "\n† Feasible result; optimality not certified in every repetition."
            if any(r["status"] == "failed" for r in summaries):
                footnote += "\nERR: failed attempt; see timing_runs.csv."
            fig.text(.5, .035, footnote, ha="center", fontsize=12)
            fig.subplots_adjust(top=.84, bottom=.13, left=.04, right=.96)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"computational_time_table.{extension}", dpi=300, bbox_inches="tight")
            plt.close(fig)
    return summaries


def measure_case(config, workload, energy, repetition, order):
    row = {"horizon_hours": config.parameters.horizon_slots * config.parameters.slot_hours,
        "participating_type_count": len(config.parameters.type_lambdas), "repeat": repetition,
        "execution_order": order, "algorithm": config.solver.algorithm,
        "wall_seconds": None, "algorithm_elapsed_seconds": None, "status": "error", "converged": False,
        "termination_reason": "", "absolute_gap_usd": None, "relative_gap": None,
        "iterations": None, "optimality_cuts": None, "feasibility_cuts": None,
        "task_count": workload.task_count, "source_record_count": workload.source_record_count,
        "total_tokens": workload.total_tokens, "horizon_slots": config.parameters.horizon_slots,
        "continuous_variables": (len(config.parameters.type_lambdas) + 1) * config.parameters.horizon_slots,
        "menu_binaries": len(config.parameters.type_lambdas) * len(presolve_parameters(config.parameters).candidate_deferrals_slots),
        "message": ""}
    started = time.perf_counter()
    try:
        result = solve_aggregate(config, workload, energy)
    except (OptimizerError, OSError, ValueError) as exc:
        row.update(wall_seconds=time.perf_counter() - started,
                   status="infeasible" if isinstance(exc, NoFeasibleSolutionError) else "error", message=str(exc))
        return row, None
    row["wall_seconds"] = time.perf_counter() - started
    try:
        _check_proposed(result, config)
    except DataValidationError as exc:
        row.update(status="validation_failed", message=str(exc))
        return row, None
    row.update(algorithm_elapsed_seconds=result.elapsed_seconds,
        status="optimal_within_tolerance" if result.converged else "best_feasible",
        converged=result.converged, termination_reason=result.termination_reason,
        absolute_gap_usd=result.absolute_gap_usd, relative_gap=result.relative_gap,
        iterations=len(result.history), optimality_cuts=result.optimality_cuts, feasibility_cuts=result.feasibility_cuts)
    return row, result


def environment_info():
    processor = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": processor, "logical_cpu_count": os.cpu_count(),
            "packages": {name: version(name) for name in ("numpy", "scipy", "matplotlib", "PyYAML")},
            "thread_environment": {name: os.environ.get(name) for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "HIGHS_THREADS")},
            "clock": {"name": "time.perf_counter", "resolution_seconds": time.get_clock_info("perf_counter").resolution,
                      "monotonic": time.get_clock_info("perf_counter").monotonic}}


def run_timing_experiment(args):
    if args.repeats < 1 or args.warmups < 0:
        raise DataValidationError("--repeats must be positive and --warmups must be nonnegative.")
    if args.time_limit is not None and (not np.isfinite(args.time_limit) or args.time_limit <= 0):
        raise DataValidationError("--time-limit must be finite and positive.")
    manifest, cases = validate_timing_suite(args.manifest, args.horizons, args.type_counts)
    if args.validate_only:
        print(json.dumps({"status": "validated", "pairs": len(cases), "horizons_hours": sorted(args.horizons),
            "type_counts": sorted(args.type_counts), "controlled_settings_identical": True,
            "arrivals_and_power_shared_across_types": True, "consecutive_fixed_label_suffixes": True,
            "capacity_screens": "all passed"}, indent=2))
        return 0
    root = args.output_dir.resolve() if args.output_dir else cases[0][1].output.root
    output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    solve_logger = logging.getLogger("llm_contract_optimizer.solve")
    old_level = solve_logger.level
    solve_logger.setLevel(logging.WARNING)
    rows, warmups, case_inputs = [], [], []
    try:
        snapshot = json.loads(json.dumps(manifest))
        base = args.manifest.resolve().parent
        snapshot["files"] = {str((base / name).resolve()): digest for name, digest in manifest["files"].items()}
        for entry in snapshot["cases"]:
            entry["config"] = str((base / entry["config"]).resolve())
        _atomic_text(output / "experiment_manifest.json", lambda stream: json.dump(snapshot, stream, indent=2))
        for entry, config, workload, energy in cases:
            directory = output / f"H{entry['hours']:03d}_N{entry['type_count']:02d}"
            directory.mkdir()
            config = replace(config, solver=replace(config.solver,
                total_time_limit_seconds=args.time_limit if args.time_limit is not None else config.solver.total_time_limit_seconds),
                output=replace(config.output, root=directory, make_plot=False, write_group_schedule=False))
            validate_config(config)
            _atomic_text(directory / "effective_config.yaml", lambda stream: yaml.safe_dump(_effective_config(config), stream, sort_keys=False))
            case_inputs.append((entry, config, workload, energy))
        chosen = next((case for case in case_inputs if case[0]["hours"] == 24 and case[0]["type_count"] == 8), case_inputs[0])
        for index in range(args.warmups):
            print(f"Unmeasured warmup {index + 1}/{args.warmups}...", flush=True)
            warmup, _ = measure_case(*chosen[1:], 0, 0)
            warmups.append(warmup)
            if warmup["status"] not in ("optimal_within_tolerance", "best_feasible"):
                LOGGER.warning("Warmup failed: %s", warmup["message"])
                break
        environment = environment_info()
        _atomic_text(output / "timing_protocol.json", lambda stream: json.dump({
            "timing_scope": TIMING_SCOPE, "unit": "seconds", "statistic": "median",
            "measured_repeats": args.repeats, "requested_warmups": args.warmups, "warmups": warmups,
            "seed": args.seed, "execution": "Sequential, shuffled case order per repetition; fresh model and solve for every attempt.",
            "uniform_time_limit_override_seconds": args.time_limit,
            "solver_settings": _effective_config(case_inputs[0][1])["solver"], "environment": environment,
            "implementation_sha256": {path.name: source_digest(path) for path in (
                Path(__file__), Path(__file__).with_name("solve.py"), Path(__file__).with_name("inner.py"),
                Path(__file__).with_name("master.py"), Path(__file__).with_name("menu.py"))},
        }, stream, indent=2))
        rng = random.Random(args.seed)
        for repetition in range(1, args.repeats + 1):
            ordered = list(case_inputs)
            rng.shuffle(ordered)
            for entry, config, workload, energy in ordered:
                label = f"{entry['hours']}h / {entry['type_count']} types / repeat {repetition}"
                print(f"Timing {label}...", flush=True)
                row, result = measure_case(config, workload, energy, repetition, len(rows) + 1)
                rows.append(row)
                detail = config.output.root / f"repeat_{repetition:02d}.json"
                payload = {"timing": row, "solution": _summary(result, config) if result is not None else None}
                _atomic_text(detail, lambda stream: json.dump(payload, stream, indent=2, allow_nan=False))
                write_timing_tables(output, rows, args.horizons, args.type_counts, args.repeats)
                LOGGER.info("%s: %.6fs, %s", label, row["wall_seconds"], row["status"])
        summaries = write_timing_tables(output, rows, args.horizons, args.type_counts, args.repeats, plot=not args.no_plot)
        failed = [r for r in summaries if r["status"] == "failed"]
        summary = {"status": "completed_with_errors" if failed else "completed",
            "pair_count": len(cases), "measured_attempts": len(rows), "repeats": args.repeats,
            "all_runs_converged": all(r["converged"] for r in rows), "failed_pairs": failed,
            "table": "computational_time_table.csv", "timing_scope": TIMING_SCOPE}
        _atomic_text(output / "experiment_summary.json", lambda stream: json.dump(summary, stream, indent=2))
        print(f"Timing table: {output / 'computational_time_table.md'}\nAll measured runs converged: {summary['all_runs_converged']}", flush=True)
        return 2 if failed else 0
    finally:
        solve_logger.setLevel(old_level)
        logging.getLogger().removeHandler(handler)
        handler.close()
