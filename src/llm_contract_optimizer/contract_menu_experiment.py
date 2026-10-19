"""Grouped contract-menu bars for randomly selected workload and horizon cases."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random

import numpy as np

from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.exceptions import DataValidationError, NoFeasibleSolutionError, OptimizerError
from llm_contract_optimizer.horizon_experiment import _check_proposed, controlled_settings, validate_suite
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.menu import audit_menu_domain
from llm_contract_optimizer.reporting import _atomic_text, _csv, _effective_config, write_result_bundle
from llm_contract_optimizer.sampling import source_digest
from llm_contract_optimizer.scenario_comparison import _case_config, _workload_scenario_ids
from llm_contract_optimizer.solve import load_case, solve_aggregate

LOGGER = logging.getLogger(__name__)
FIGURES = ("contract_deferral_by_workload", "contract_discount_by_workload",
           "contract_deferral_by_horizon", "contract_discount_by_horizon")


def select_four(population, seed):
    pool = sorted(population)
    if len(pool) < 4 or len(set(pool)) != len(pool):
        raise DataValidationError("Contract-menu comparisons need at least four distinct candidate cases.")
    return sorted(random.Random(seed).sample(pool, 4))


def load_menu_cases(args):
    base = load_config(args.config)
    if len(base.parameters.type_lambdas) != 8 or base.parameters.horizon_slots * base.parameters.slot_seconds != 86400:
        raise DataValidationError("Contract-menu comparisons require the default eight participating types and 24-hour anchor.")
    if args.time_limit is not None and (not np.isfinite(args.time_limit) or args.time_limit <= 0):
        raise DataValidationError("--time-limit must be finite and positive.")
    population = _workload_scenario_ids(base)
    horizon_manifest = json.loads(args.horizon_manifest.read_text(encoding="utf-8"))
    workloads = select_four(population, args.seed)
    hours = select_four(horizon_manifest["horizons_hours"], args.seed + 1)
    _, horizon_cases = validate_suite(args.horizon_manifest, hours)
    for _, config, _, _ in horizon_cases:
        if controlled_settings(config) != controlled_settings(base):
            raise DataValidationError("The horizon suite must match the configured default in every non-horizon setting, including all eight lambdas.")
    cases = []
    for workload_id in workloads:
        config, workload, energy = load_case(_case_config(base, workload_id, base.inputs.energy_scenario_id))
        screen = FluidInnerSolver(workload, energy, config.parameters, config.solver).screen_infeasibility()
        if screen or not audit_menu_domain(config.parameters)["contract_domain_feasible"]:
            raise DataValidationError(f"Selected workload {workload_id} fails a feasibility screen: {screen}")
        cases.append(({"comparison": "workload", "case_label": workload_id,
                       "case_directory": f"W_{workload_id}"}, config, workload, energy))
    for entry, config, workload, energy in horizon_cases:
        cases.append(({"comparison": "horizon", "case_label": f"{entry['hours']} h",
                       "case_directory": f"H{entry['hours']:03d}"}, config, workload, energy))
    selection = {"schema": "contract_menu_comparison_v1", "seed": args.seed,
        "selection_rule": "Uniform sampling without replacement before solving; sorted candidate pools. Workload seed = seed; horizon seed = seed + 1.",
        "workload_population": population, "horizon_population_hours": sorted(horizon_manifest["horizons_hours"]),
        "selected_workload_scenarios": workloads, "selected_horizons_hours": hours,
        "participating_user_types": list(range(1, 9)), "fixed_type_lambdas": list(base.parameters.type_lambdas),
        "workload_comparison": {"fixed_power_profile": base.inputs.energy_scenario_id, "horizon_hours": 24},
        "horizon_comparison": {"fixed_workload_anchor": base.inputs.workload_scenario_id,
                               "fixed_power_anchor": base.inputs.energy_scenario_id,
                               "data_rule": "Existing consecutive observed workload/power windows; shared occurrences retain fixed labels."},
        "anchor_config": str(base.source_path), "anchor_config_sha256": source_digest(base.source_path),
        "anchor_settings": _effective_config(base),
        "horizon_manifest": str(args.horizon_manifest.resolve()),
        "horizon_manifest_sha256": source_digest(args.horizon_manifest),
        "workload_recipe_sha256": {str(base.inputs.workload_file.with_name(f"{label}.sample.json")):
            source_digest(base.inputs.workload_file.with_name(f"{label}.sample.json")) for label in workloads},
        "uniform_time_limit_override_seconds": args.time_limit,
        "deferral_definition": "Offered nominal maximum deferral in hours, not measured execution delay. Late arrivals retain the existing horizon-end clipping policy.",
        "discount_definition": "Offered proportional bill discount multiplied by 100 for percent.",
    }
    return base, selection, cases


def contract_rows(meta, config, result=None, *, error=None):
    common = {"comparison": meta["comparison"], "case_label": meta["case_label"],
        "workload_scenario_id": config.inputs.workload_scenario_id, "energy_scenario_id": config.inputs.energy_scenario_id,
        "horizon_hours": config.parameters.horizon_slots * config.parameters.slot_hours,
        "status": ("optimal_within_tolerance" if result.converged else "best_feasible") if result is not None
                  else "infeasible" if isinstance(error, NoFeasibleSolutionError) else "error",
        "converged": result.converged if result is not None else False,
        "termination_reason": result.termination_reason if result is not None else "",
        "absolute_gap_usd": result.absolute_gap_usd if result is not None else None,
        "message": str(error) if error is not None else ""}
    return [{**common, "user_type": i + 1, "lambda": float(sensitivity),
             "deferral_slots": int(result.menu.deadlines_slots[i]) if result is not None else None,
             "deferral_hours": float(result.menu.deadlines_slots[i] * config.parameters.slot_hours) if result is not None else None,
             "discount_fraction": float(result.menu.discounts[i]) if result is not None else None,
             "discount_percent": float(result.menu.discounts[i] * 100) if result is not None else None}
            for i, sensitivity in enumerate(config.parameters.type_lambdas)]


def write_contract_plots(output, selection, rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    colors = ("#2563eb", "#0f766e", "#ea580c", "#7c3aed")
    hatches = ("", "//", "xx", "..")
    specs = (("workload", selection["selected_workload_scenarios"], "workload scenarios"),
             ("horizon", [f"{h} h" for h in selection["selected_horizons_hours"]], "planning horizons"))
    metrics = (("deferral_hours", "Deferral window (hours)", "Contract deferral windows", "deferral"),
               ("discount_percent", "Discount (%)", "Contract discounts", "discount"))
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 16,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        for comparison, labels, description in specs:
            if len(labels) != 4:
                raise DataValidationError("Each contract-menu figure must compare four selected cases.")
            selected = {label: sorted((r for r in rows if r["comparison"] == comparison and r["case_label"] == label),
                                     key=lambda r: r["user_type"]) for label in labels}
            if any([r["user_type"] for r in selected[label]] != list(range(1, 9)) for label in labels):
                raise DataValidationError("Every selected contract menu must have exactly user types 1 through 8.")
            for field, ylabel, title, metric in metrics:
                fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
                limited = False
                missing = False
                for index, (label, color, hatch) in enumerate(zip(labels, colors, hatches)):
                    case = selected[label]
                    no_menu = case[0]["status"] in ("error", "infeasible")
                    uncertified = case[0]["status"] == "best_feasible"
                    limited |= uncertified
                    missing |= no_menu
                    legend = label + (" (no menu)" if no_menu else " †" if uncertified else "")
                    values = [np.nan if r[field] is None else r[field] for r in case]
                    ax.bar(np.arange(1, 9) + (index - 1.5) * .2, values, width=.2,
                           label=legend, color=color, hatch=hatch, edgecolor="#334155", linewidth=.5)
                if comparison == "workload":
                    fixed = selection["workload_comparison"]
                    context = f"Power profile {fixed['fixed_power_profile']} | {fixed['horizon_hours']} h"
                else:
                    fixed = selection["horizon_comparison"]
                    context = f"Workload {fixed['fixed_workload_anchor']} | Power profile {fixed['fixed_power_anchor']}"
                fig.suptitle(f"{title} across {description}\n{context}", fontsize=16)
                ax.set(xlabel="User type", ylabel=ylabel, xlim=(.4, 8.6))
                ax.set_xticks(np.arange(1, 9))
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=.22)
                ax.set_axisbelow(True)
                handles, legend_labels = ax.get_legend_handles_labels()
                fig.legend(handles, legend_labels, loc="outside lower center", ncol=4,
                           frameon=False, fontsize=14)
                notes = []
                if limited:
                    notes.append("† Best feasible menu; optimality not certified.")
                if missing:
                    notes.append("Missing bars: no feasible menu returned; see case diagnostics.")
                if notes:
                    ax.set_title("\n".join(notes), fontsize=12, color="#4b5563")
                for extension in ("png", "pdf"):
                    fig.savefig(output / f"contract_{metric}_by_{comparison}.{extension}",
                                dpi=300, bbox_inches="tight")
                plt.close(fig)


def run_contract_menu_experiment(args):
    base, selection, cases = load_menu_cases(args)
    if args.validate_only:
        print(json.dumps({"status": "validated", **selection}, indent=2))
        return 0
    root = args.output_dir.resolve() if args.output_dir is not None else base.output.root / "contract_menus"
    output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    rows, diagnostics = [], []
    try:
        _atomic_text(output / "experiment_selection.json", lambda stream: json.dump(selection, stream, indent=2))
        print(f"Selected workloads: {', '.join(selection['selected_workload_scenarios'])}", flush=True)
        print(f"Selected horizons (hours): {selection['selected_horizons_hours']}", flush=True)
        for meta, original_config, workload, energy in cases:
            directory = output / meta["case_directory"]
            directory.mkdir()
            config = replace(original_config,
                solver=replace(original_config.solver, total_time_limit_seconds=args.time_limit
                    if args.time_limit is not None else original_config.solver.total_time_limit_seconds),
                output=replace(original_config.output, root=directory, make_plot=False, write_group_schedule=False))
            LOGGER.info("Contract menu: %s comparison, %s", meta["comparison"], meta["case_label"])
            result = None
            try:
                result = solve_aggregate(config, workload, energy)
                _check_proposed(result, config)
            except (OptimizerError, OSError, ValueError) as exc:
                result = None
                case_rows = contract_rows(meta, config, error=exc)
                _atomic_text(directory / "failure.json", lambda stream: json.dump(case_rows[0], stream, indent=2))
            else:
                case_rows = contract_rows(meta, config, result)
                write_result_bundle(result, config, directory)
            rows.extend(case_rows)
            diagnostics.append({**meta, "status": case_rows[0]["status"], "converged": case_rows[0]["converged"],
                "absolute_gap_usd": case_rows[0]["absolute_gap_usd"], "message": case_rows[0]["message"]})
            for comparison in ("workload", "horizon"):
                selected = [row for row in rows if row["comparison"] == comparison]
                if selected:
                    _csv(output / f"contract_menus_by_{comparison}.csv", selected)
        write_contract_plots(output, selection, rows)
        figure_files = [f"{name}.{ext}" for name in FIGURES for ext in ("png", "pdf")]
        actual_figures = {path.name for path in output.rglob("*") if path.suffix in (".png", ".pdf")}
        if actual_figures != set(figure_files):
            raise DataValidationError("Contract-menu experiment must produce exactly four figure identities in PNG/PDF.")
        failed = [case for case in diagnostics if case["status"] in ("infeasible", "error")]
        summary = {"status": "completed_with_errors" if failed else "completed", "case_count": len(cases),
            "all_proposed_converged": all(case["converged"] for case in diagnostics),
            "selected_workload_scenarios": selection["selected_workload_scenarios"],
            "selected_horizons_hours": selection["selected_horizons_hours"],
            "figure_files": figure_files, "cases": diagnostics,
            "implementation_sha256": {path.name: source_digest(path) for path in
                (Path(__file__), Path(__file__).with_name("solve.py"), Path(__file__).with_name("menu.py"))}}
        _atomic_text(output / "experiment_summary.json", lambda stream: json.dump(summary, stream, indent=2, allow_nan=False))
        print(f"Four contract-menu bar figures saved as PNG/PDF in: {output}\nAll selected solves converged: {summary['all_proposed_converged']}", flush=True)
        return 2 if failed else 0
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
