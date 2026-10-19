from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from llm_contract_optimizer.config import AppConfig, load_config, validate_config
from llm_contract_optimizer.exceptions import NoFeasibleSolutionError, OptimizerError
from llm_contract_optimizer.reporting import (
    _atomic_text, create_run_directory, print_method_comparison, print_result,
    write_method_comparison, write_result_bundle,
)
from llm_contract_optimizer.solve import solve_case, validate_case

CONFIG_DIRECTORY = Path(__file__).resolve().parents[2] / "config"
DATA_DIRECTORY = CONFIG_DIRECTORY.parent / "data"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-contract-opt",
        description="Solve the LLM contract-menu and workload-allocation problem.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate configuration and selected inputs.")
    validate.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")

    menu_audit = commands.add_parser("audit-menu", help="Check contract feasibility without loading or scheduling tasks.")
    menu_audit.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")

    horizons = commands.add_parser("compare-horizons", help="Compare nested consecutive planning horizons against both benchmarks.")
    horizons.add_argument("--manifest", type=Path, default=DATA_DIRECTORY / "planning_horizons/experiment_manifest.json")
    horizons.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 48, 72, 96, 120, 144, 168],
                          choices=[12, 24, 48, 72, 96, 120, 144, 168])
    horizons.add_argument("--validate-only", action="store_true")
    horizons.add_argument("--time-limit", type=float, help="Uniform proposed-method solver budget in seconds for every horizon.")
    horizons.add_argument("--output-dir", type=Path)
    horizons.add_argument("--no-plot", action="store_true")
    horizons.add_argument("--no-group-schedule", action="store_true")

    type_prepare = commands.add_parser("prepare-type-counts", help="Prepare the fixed-workload participating-type-count sweep.")
    type_prepare.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")
    type_prepare.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY / "type_counts")
    type_prepare.add_argument("--config-dir", type=Path, default=CONFIG_DIRECTORY / "type_counts")
    types = commands.add_parser("compare-type-counts", help="Compare 2, 4, 6, 8 and 10 participating types against both benchmarks.")
    types.add_argument("--manifest", type=Path, default=DATA_DIRECTORY / "type_counts/experiment_manifest.json")
    types.add_argument("--type-counts", type=int, nargs="+", default=[2, 4, 6, 8, 10], choices=[2, 4, 6, 8, 10])
    types.add_argument("--validate-only", action="store_true")
    types.add_argument("--time-limit", type=float, help="Uniform proposed-method solver budget in seconds for every type count.")
    types.add_argument("--output-dir", type=Path)
    types.add_argument("--no-plot", action="store_true")
    types.add_argument("--no-group-schedule", action="store_true")

    share_prepare = commands.add_parser("prepare-immediate-shares", help="Prepare fixed-workload immediate-service-share cases.")
    share_prepare.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")
    share_prepare.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY / "immediate_shares")
    share_prepare.add_argument("--config-dir", type=Path, default=CONFIG_DIRECTORY / "immediate_shares")
    shares = commands.add_parser("compare-immediate-shares", help="Compare 10%, 20%, 30%, 40% and 50% immediate service against both benchmarks.")
    shares.add_argument("--manifest", type=Path, default=DATA_DIRECTORY / "immediate_shares/experiment_manifest.json")
    shares.add_argument("--shares", type=int, nargs="+", default=[10, 20, 30, 40, 50], choices=[10, 20, 30, 40, 50],
                        help="Immediate-task share percentages.")
    shares.add_argument("--validate-only", action="store_true")
    shares.add_argument("--time-limit", type=float, help="Uniform proposed-method solver budget in seconds for every share.")
    shares.add_argument("--output-dir", type=Path)
    shares.add_argument("--no-plot", action="store_true")
    shares.add_argument("--no-group-schedule", action="store_true")

    day = commands.add_parser("representative-day", help="Plot arrivals, scheduling, day-ahead price and grid imports for one default 24-hour case.")
    day.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")
    day.add_argument("--output-dir", type=Path, help="Parent directory for a new timestamped representative-day run.")

    timing_prepare = commands.add_parser("prepare-computational-time", help="Prepare the joint type-count and consecutive-horizon timing experiment.")
    timing_prepare.add_argument("--horizon-manifest", type=Path, default=DATA_DIRECTORY / "planning_horizons/experiment_manifest.json")
    timing_prepare.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY / "computational_time")
    timing_prepare.add_argument("--config-dir", type=Path, default=CONFIG_DIRECTORY / "computational_time")
    timing = commands.add_parser("compare-computational-time", help="Time the proposed method over the 8-by-5 horizon/type grid and write a table in seconds.")
    timing.add_argument("--manifest", type=Path, default=DATA_DIRECTORY / "computational_time/experiment_manifest.json")
    timing.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 48, 72, 96, 120, 144, 168], choices=[12, 24, 48, 72, 96, 120, 144, 168])
    timing.add_argument("--type-counts", type=int, nargs="+", default=[2, 4, 6, 8, 10], choices=[2, 4, 6, 8, 10])
    timing.add_argument("--repeats", type=int, default=1, help="Measured fresh solves per pair; table entries are their median runtime.")
    timing.add_argument("--warmups", type=int, default=1, help="Unmeasured fresh solves of the selected 24h/eight-type case, or the smallest selected case.")
    timing.add_argument("--seed", type=int, default=20261001, help="Seed for the sequential case order in each repetition.")
    timing.add_argument("--time-limit", type=float, help="Uniform proposed-method total solver budget in seconds per solve.")
    timing.add_argument("--validate-only", action="store_true")
    timing.add_argument("--output-dir", type=Path)
    timing.add_argument("--no-plot", action="store_true", help="Write CSV and Markdown tables without the PNG/PDF table.")

    contract = commands.add_parser("compare-contract-menus", help="Plot default eight-type contract menus for four random workloads and four random horizons.")
    contract.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")
    contract.add_argument("--horizon-manifest", type=Path, default=DATA_DIRECTORY / "planning_horizons/experiment_manifest.json")
    contract.add_argument("--seed", type=int, default=20261001, help="Reproducible selection seed; four cases are drawn without replacement per comparison.")
    contract.add_argument("--time-limit", type=float, help="Uniform proposed-method total solve budget in seconds.")
    contract.add_argument("--output-dir", type=Path)
    contract.add_argument("--validate-only", action="store_true", help="Show and validate the selected inputs without solving or plotting.")

    for name, help_text in (
        ("solve", "Run menu search and write a result bundle."),
        ("compare", "Run the proposed method and both fixed-policy profit benchmarks."),
    ):
        run = commands.add_parser(name, help=help_text)
        run.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")
        run.add_argument("--workload-scenario")
        run.add_argument("--energy-scenario")
        run.add_argument("--algorithm", choices=["benders", "direct"])
        run.add_argument("--time-limit", type=float, help="Total solver time budget in seconds.")
        run.add_argument("--max-iterations", type=int)
        run.add_argument("--output-dir", type=Path)
        run.add_argument("--no-plot", action="store_true")
        run.add_argument("--no-group-schedule", action="store_true")
        if name == "compare":
            run.add_argument(
                "--profit-sweeps",
                action="store_true",
                help=("Also vary every sampled workload and power scenario one at a time "
                      "and write profit, discount expenditure and day-ahead electricity "
                      "cost comparisons in USD and USD/hour (six figures)."),
            )
    prepare = commands.add_parser("prepare-workload", help="Resample individual tasks while preserving trace arrival proportions.")
    prepare.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")
    prepare.add_argument(
        "--source-file",
        type=Path,
        default=DATA_DIRECTORY / "representative_tasks_eight_types.csv.gz",
        help="Original individual-task table used for sampling.",
    )
    prepare.add_argument("--targets", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--all-scenarios", action="store_true")
    prepare.add_argument("--audit-dir", type=Path, help="Optional directory for generated sampling audit files.")
    prepare.add_argument("--write-config", type=Path, help="Optional path for a generated configuration.")
    prepare.add_argument("--seed", type=int, default=20260910)
    prepare.add_argument("--immediate-execution-share", type=float, default=0.5,
                         help="Target fixed type-0 task share; integer counts are assigned before solving.")
    for command in ("assign-types", "audit-types"):
        audit = commands.add_parser(command, help="Assign or audit one fixed type per sampled task occurrence.")
        audit.add_argument("--config", type=Path, default=CONFIG_DIRECTORY / "default.yaml")
        audit.add_argument("--all-scenarios", action="store_true")
        audit.add_argument("--seed", type=int, default=20260910)
        audit.add_argument("--output-file", type=Path, required=True)
    return parser


def _overrides(args: argparse.Namespace, config: AppConfig) -> AppConfig:
    inputs = replace(
        config.inputs,
        workload_scenario_id=args.workload_scenario or config.inputs.workload_scenario_id,
        energy_scenario_id=args.energy_scenario or config.inputs.energy_scenario_id,
    )
    if args.workload_scenario and inputs.workload_file is not None and inputs.workload_file.name.endswith(".sample.json"):
        candidate = inputs.workload_file.with_name(f"{inputs.workload_scenario_id}.sample.json")
        if not candidate.is_file():
            raise OptimizerError(f"Sampled workload recipe not found: {candidate}")
        inputs = replace(inputs, workload_file=candidate)
    solver = replace(
        config.solver,
        algorithm=args.algorithm or config.solver.algorithm,
        total_time_limit_seconds=(args.time_limit if args.time_limit is not None
                                  else config.solver.total_time_limit_seconds),
        max_iterations=(
            args.max_iterations
            if args.max_iterations is not None
            else config.solver.max_iterations
        ),
    )
    output_root = (
        args.output_dir.resolve() if args.output_dir is not None else config.output.root
    )
    output = replace(
        config.output,
        root=output_root,
        make_plot=config.output.make_plot and not args.no_plot,
        write_group_schedule=config.output.write_group_schedule and not args.no_group_schedule,
    )
    resolved = replace(config, inputs=inputs, solver=solver, output=output)
    validate_config(resolved)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments or ["solve"])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    output = None
    file_handler = None
    try:
        if args.command == "compare-contract-menus":
            from llm_contract_optimizer.contract_menu_experiment import run_contract_menu_experiment
            return run_contract_menu_experiment(args)
        if args.command == "prepare-computational-time":
            from llm_contract_optimizer.computational_time_experiment import prepare_timing_experiment
            return prepare_timing_experiment(args)
        if args.command == "compare-computational-time":
            from llm_contract_optimizer.computational_time_experiment import run_timing_experiment
            return run_timing_experiment(args)
        if args.command == "representative-day":
            from llm_contract_optimizer.representative_day import run_representative_day
            return run_representative_day(args)
        if args.command == "prepare-type-counts":
            from llm_contract_optimizer.type_count_experiment import prepare_type_count_experiment
            return prepare_type_count_experiment(args)
        if args.command == "prepare-immediate-shares":
            from llm_contract_optimizer.immediate_share_experiment import prepare_immediate_share_experiment
            return prepare_immediate_share_experiment(args)
        if args.command == "compare-immediate-shares":
            from llm_contract_optimizer.immediate_share_experiment import run_immediate_share_experiment
            return run_immediate_share_experiment(args)
        if args.command == "compare-type-counts":
            from llm_contract_optimizer.type_count_experiment import run_type_count_experiment
            return run_type_count_experiment(args)
        if args.command == "compare-horizons":
            from llm_contract_optimizer.horizon_experiment import run_horizon_experiment
            return run_horizon_experiment(args)
        config = load_config(args.config)
        if args.command == "audit-menu":
            from llm_contract_optimizer.menu import audit_menu_domain
            print(json.dumps(audit_menu_domain(config.parameters), indent=2))
            return 0
        if args.command in ("assign-types", "audit-types"):
            from llm_contract_optimizer.preference_audit import audit_recipes
            audit_recipes(config, args.output_file, args.all_scenarios, args.command == "assign-types", args.seed)
            return 0
        if args.command == "prepare-workload":
            config = replace(config, inputs=replace(config.inputs,
                immediate_execution_share=args.immediate_execution_share))
            validate_config(config)
            from llm_contract_optimizer.data import load_source_pool
            from llm_contract_optimizer.sampling import daily_targets, plan_sampling, write_plan, write_slot_audit, assign_recipe_types
            from llm_contract_optimizer.reporting import _effective_config
            import yaml
            targets = daily_targets(args.targets)
            scenarios = sorted(targets) if args.all_scenarios else [config.inputs.workload_scenario_id]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            summaries = []
            source = args.source_file.resolve()
            if source.suffix == ".json":
                raise OptimizerError("--source-file must be an original individual-task table, not a sampled recipe.")
            for scenario in scenarios:
                if scenario not in targets:
                    raise OptimizerError(f"Missing daily token target for {scenario}.")
                workload = load_source_pool(config.inputs.root, scenario, config.parameters, source)
                plan = plan_sampling(workload, targets[scenario], args.seed)
                recipe = (args.output_dir / f"{scenario}.sample.json").resolve()
                write_plan(recipe, source, workload, plan)
                assign_recipe_types(recipe, scenario, config.parameters, config.inputs.immediate_execution_share, args.seed)
                if args.audit_dir is not None:
                    write_slot_audit(args.audit_dir / f"{scenario}.slots.csv", workload, plan, config.parameters.horizon_slots)
                summary = {"scenario":scenario, "original_tasks":workload.task_count,
                           "sampled_tasks":plan.task_count, "count_multiplier":plan.multiplier,
                           "target_tokens":plan.target_tokens, "actual_tokens":sum(
                               c*t.total_tokens for c,t in zip(plan.counts,workload.tasks)),
                           "arrival_proportions_exact":True, "recipe":str(recipe)}
                summaries.append(summary)
                if args.audit_dir is not None:
                    _atomic_text(args.audit_dir / "sampling_summary.json",lambda f:json.dump(summaries,f,indent=2))
                print(json.dumps(summary), flush=True)
            # One configuration serves every recipe through --workload-scenario.
            sampled_config = replace(config, inputs=replace(config.inputs,
                workload_scenario_id=scenarios[0],
                workload_file=(args.output_dir / f"{scenarios[0]}.sample.json").resolve()))
            if args.write_config is not None:
                _atomic_text(args.write_config,
                    lambda f: yaml.safe_dump(_effective_config(sampled_config), f, sort_keys=False))
            return 0
        if args.command == "validate":
            print(json.dumps(validate_case(config), indent=2, sort_keys=True))
            return 0
        config = _overrides(args, config)
        output = create_run_directory(config)
        file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
        logging.getLogger().addHandler(file_handler)
        logging.getLogger(__name__).info("Configuration: %s; outputs: %s", config.source_path, output)
        result = write_result_bundle(solve_case(config), config, output)
        if args.command == "compare":
            from llm_contract_optimizer.benchmarks import run_benchmark_case
            benchmarks = run_benchmark_case(config)
            write_method_comparison(output, result, benchmarks, config)
            print_method_comparison(result, benchmarks, output)
            if args.profit_sweeps:
                from llm_contract_optimizer.scenario_comparison import run_profit_scenario_sweeps
                run_profit_scenario_sweeps(config, result, benchmarks, output)
        logging.getLogger(__name__).info(
            "Finished: %s | profit USD %.6f | absolute gap USD %.8g | contract menu: %s",
            result.termination_reason, result.profit_usd, result.absolute_gap_usd,
            output / "contract_menu.md",
        )
        print_result(result, config)
        return 0
    except (OptimizerError, OSError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        if output is not None:
            _atomic_text(output / "failure.json", lambda stream: json.dump({
                "status": "infeasible" if isinstance(exc, NoFeasibleSolutionError) else "error",
                "error_type": type(exc).__name__, "message": str(exc),
            }, stream, indent=2))
        return 2
    finally:
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
