"""Prepare nested consecutive stress windows anchored to the current 24h default."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "src"))

import prepare_burstgpt_task_inputs as prep
from reselect_workload_scenarios import source_pool, write_gzip
from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.data import load_energy_scenario, load_workload_aggregate
from llm_contract_optimizer.horizon_data import WINDOW_SCHEMA, load_consecutive_workload
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.preferences import assign_known_types
from llm_contract_optimizer.reporting import _effective_config
from llm_contract_optimizer.sampling import plan_sampling, source_digest

HORIZONS = (12, 24, 48, 72, 96, 120, 144, 168)


def relative(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


def metrics(workload, energy, parameters):
    arrivals = workload.opt_out_tokens_by_arrival + workload.participating_tokens_by_type_arrival.sum(axis=0)
    cap = parameters.token_capacity_per_slot
    return {
        "total_tokens": workload.total_tokens, "task_count": workload.task_count,
        "active_slots": int((arrivals > 0).sum()),
        "dense_slots": int((arrivals >= .8 * cap).sum()),
        "overload_slots": int((arrivals > cap).sum()),
        "peak_million_tokens_per_s": float(arrivals.max() / parameters.slot_seconds / 1e6),
        "overflow_fraction": float(np.maximum(arrivals - cap, 0).sum() / arrivals.sum()),
        "actual_immediate_task_share": float(workload.type_counts[0] / workload.task_count),
        "mean_day_ahead_price_usd_per_kwh": float(energy.day_ahead_price_usd_per_kwh.mean()),
        "fixed_supply_kwh": float(energy.fixed_supply_kwh.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config/default.yaml")
    parser.add_argument("--burstgpt-file", type=Path, required=True,
                        help="Original BurstGPT v2.0 BurstGPT_3.csv trace")
    parser.add_argument("--power-file", type=Path, required=True,
                        help="Prepared all-days power table (slot_inputs_all_days.csv)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data/planning_horizons")
    parser.add_argument("--config-dir", type=Path, default=PROJECT / "config/planning_horizons")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.parameters.horizon_slots * config.parameters.slot_seconds != 86400:
        parser.error("Preparation requires a 24-hour anchor configuration.")
    if config.parameters.slot_seconds != prep.SLOT_SECONDS:
        parser.error("Preparation retains the established 900-second source alignment and slot size.")
    if len(config.parameters.type_lambdas) != 8:
        parser.error("The established source preparation requires eight participating types.")
    if config.inputs.workload_file is None:
        parser.error("A sampled default workload recipe is required.")
    original = json.loads(config.inputs.workload_file.read_text(encoding="utf-8"))
    if original.get("schema") != "individual_task_sampling_v1":
        parser.error("The default workload must be an individual-task sampling recipe.")
    anchor = int(config.inputs.workload_scenario_id.split("-D")[-1])
    days = list(range(anchor - 6, anchor + 1))
    day_slots = config.parameters.horizon_slots
    if any(hours * 3600 % config.parameters.slot_seconds for hours in HORIZONS):
        parser.error("Every horizon must be an integer number of slots.")
    source = args.burstgpt_file.resolve()
    power_source = args.power_file.resolve()
    if not source.is_file() or not power_source.is_file():
        parser.error("Original BurstGPT and all-days power input files must be available locally.")
    output, configs = args.output_dir.resolve(), args.config_dir.resolve()
    if (output / "experiment_manifest.json").exists():
        parser.error("Prepared data already exist. Choose a new --output-dir and --config-dir to rebuild.")
    if any(configs.glob("H*.yaml")):
        parser.error("The configuration directory already contains horizon cases. Choose a new --config-dir to preserve the earlier package.")
    output.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    daily_dir = output / "daily"
    daily_dir.mkdir(exist_ok=True)
    print(f"Extracting consecutive trace days {days[0]}..{days[-1]}...", flush=True)
    tasks = prep.extract_representative_tasks(source, days, {d: ["consecutive horizon window"] for d in days}, 400_000)
    if set(tasks.trace_day) != set(days):
        raise ValueError("Trace does not contain every required consecutive day.")
    mapping = {d: (config.inputs.workload_scenario_id if d == anchor else f"C-D{d}") for d in days}
    tasks["scenario_id"] = tasks.trace_day.map(mapping)
    tasks = tasks.sort_values(["scenario_id", "arrival_slot", "arrival_second", "source_row"], kind="stable").reset_index(drop=True)
    if not tasks.task_id.is_unique or (tasks.input_tokens + tasks.output_tokens != tasks.total_tokens).any():
        raise ValueError("Original request identities or token totals are invalid.")
    refined = tasks.drop(columns=[f"p_type_{i}" for i in range(1, 5)]).copy()
    for i in range(1, 5):
        refined[f"p_type_{2*i-1}"] = tasks[f"p_type_{i}"] / 2
        refined[f"p_type_{2*i}"] = tasks[f"p_type_{i}"] / 2
    source_table = daily_dir / "tasks_eight_types.csv.gz"
    write_gzip(refined, source_table)
    digest = source_digest(source_table)
    entries = []
    for day in days:
        scenario = mapping[day]
        group = refined.loc[refined.trace_day.eq(day)]
        pool = source_pool(group, scenario)
        raw_slots = group.groupby("arrival_slot").total_tokens.sum().reindex(range(1, day_slots + 1), fill_value=0)
        target = int(np.rint(prep.QWEN_SHARE * (prep.AFFINE_SCALE * raw_slots.to_numpy()
                                              + prep.AFFINE_INTERCEPT_TPS) * prep.SLOT_SECONDS).sum())
        if day == anchor:
            payload = json.loads(json.dumps(original))
            if (payload["target_tokens"] != target
                    or [task.task_id for task in pool.tasks] != [row[0] for row in payload["draws"]]):
                raise ValueError("The anchor trace or calibration differs from the default recipe.")
        else:
            print(f"Sampling {scenario} once; labels will be shared by all horizons...", flush=True)
            plan = plan_sampling(pool, target, original["seed"])
            payload = {"schema": "individual_task_sampling_v1", "scenario_id": scenario,
                       "seed": plan.seed, "multiplier": plan.multiplier,
                       "target_tokens": plan.target_tokens, "task_count": plan.task_count,
                       "draws": [[task.task_id, count] for task, count in zip(pool.tasks, plan.counts)],
                       "known_types": assign_known_types(pool.tasks, plan.counts,
                           original["known_types"]["target_immediate_share"], original["known_types"]["seed"])}
        payload.update(source_file=source_table.name, source_sha256=digest)
        path = daily_dir / f"{scenario}.sample.json"
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        entries.append({"trace_day": day, "scenario_id": scenario, "recipe": relative(path, output),
                        "recipe_sha256": source_digest(path)})

    # The default selected power day fixes the end date; prepend the six real preceding days.
    original_power = pd.read_csv(config.inputs.root / "energy_scenarios.csv")
    anchor_power = original_power.loc[original_power.energy_scenario_id.eq(config.inputs.energy_scenario_id)]
    if len(anchor_power) != day_slots or anchor_power.date.nunique() != 1:
        raise ValueError("Power anchor must contain exactly one complete day.")
    anchor_date = datetime.strptime(anchor_power.date.iloc[0], "%Y-%m-%d").date()
    dates = [(anchor_date - timedelta(days=6-index)).isoformat() for index in range(7)]
    power = pd.read_csv(power_source)
    power = power.loc[power.date.isin(dates)].sort_values(["date", "slot"]).reset_index(drop=True)
    if (len(power) != 7 * day_slots or set(power.date) != set(dates)
            or any(group.slot.tolist() != list(range(1, day_slots + 1)) for _, group in power.groupby("date"))):
        raise ValueError("Power data must have complete consecutive dates and slots; gaps are not filled.")
    columns = ["ppa_kwh", "gt_kwh", "pv_kwh", "bess_discharge_kwh", "bess_charge_kwh"]
    anchor_power = anchor_power.sort_values("slot")
    end_power = power.iloc[-day_slots:]
    if (not np.allclose(anchor_power[columns], end_power[columns], rtol=0, atol=1e-8)
            or not np.allclose(anchor_power.day_ahead_price_usd_per_kwh, end_power.c_da_usd_per_kwh,
                               rtol=0, atol=1e-8)):
        raise ValueError("All-days power data do not reproduce the selected default power day.")
    # Keep the exact default CSV values on the shared final day.
    power.loc[power.index[-day_slots:], columns] = anchor_power[columns].to_numpy()
    power.loc[power.index[-day_slots:], "c_da_usd_per_kwh"] = anchor_power.day_ahead_price_usd_per_kwh.to_numpy()
    power.to_csv(output / "consecutive_power_source.csv", index=False)
    base_raw = _effective_config(config)
    rows, case_entries = [], []
    cache = {}
    for hours in HORIZONS:
        slots = hours * 3600 // config.parameters.slot_seconds
        start = 7 * day_slots - slots
        case_dir = output / f"H{hours:03d}"
        case_dir.mkdir(exist_ok=True)
        window_path = case_dir / "workload.window.json"
        window = {"schema": WINDOW_SCHEMA, "scenario_id": config.inputs.workload_scenario_id,
                  "horizon_slots": slots, "slot_seconds": config.parameters.slot_seconds,
                  "day_slots": day_slots, "start_slot_zero_based": start,
                  "days": [{**entry, "recipe": relative(output / entry["recipe"], case_dir)} for entry in entries]}
        window_path.write_text(json.dumps(window, indent=2), encoding="utf-8")
        selected_power = power.iloc[start:].copy()
        selected_power["source_slot"] = selected_power["slot"]
        selected_power["slot"] = np.arange(1, slots + 1)
        selected_power["energy_scenario_id"] = config.inputs.energy_scenario_id
        selected_power = selected_power.rename(columns={"c_da_usd_per_kwh": "day_ahead_price_usd_per_kwh"})
        energy_columns = ["energy_scenario_id", "date", "slot", *columns,
                          "day_ahead_price_usd_per_kwh", "source_slot", "timestamp_local"]
        selected_power[energy_columns].to_csv(case_dir / "energy_scenarios.csv", index=False)
        raw = json.loads(json.dumps(base_raw))
        raw["parameters"]["horizon_slots"] = slots
        raw["inputs"]["root"] = relative(case_dir, configs)
        raw["inputs"]["workload_file"] = relative(window_path, configs)
        raw["output"]["root"] = relative(PROJECT / "results/planning_horizons", configs)
        config_path = configs / f"H{hours:03d}.yaml"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        case = load_config(config_path)
        workload = load_consecutive_workload(window_path, case.inputs.workload_scenario_id,
            case.parameters, case.inputs.immediate_execution_share, case.inputs.type_distribution, cache)
        energy = load_energy_scenario(case_dir, case.inputs.energy_scenario_id, slots)
        screen = FluidInnerSolver(workload, energy, case.parameters, case.solver).screen_infeasibility()
        if screen:
            raise ValueError(f"{hours}h consecutive workload fails a necessary capacity screen: {screen}")
        if hours == 24:
            default_workload = load_workload_aggregate(config.inputs.root, config.inputs.workload_scenario_id,
                config.parameters, config.inputs.workload_file, config.inputs.immediate_execution_share,
                config.inputs.type_distribution)
            for field in ("opt_out_tokens_by_arrival", "participating_tokens_by_type_arrival", "type_counts",
                          "opt_out_bill_usd_by_arrival", "participating_bill_usd_by_type_arrival"):
                if not np.array_equal(getattr(workload, field), getattr(default_workload, field)):
                    raise ValueError(f"24h anchor does not exactly reproduce default workload field {field}.")
        row = {"horizon_hours": hours, "horizon_slots": slots,
               "workload_start_trace_day": days[start // day_slots],
               "workload_start_day_slot": start % day_slots + 1, "workload_end_trace_day": anchor,
               "power_start": selected_power.timestamp_local.iloc[0],
               "power_end_exclusive": (datetime.combine(anchor_date + timedelta(days=1), datetime.min.time())).isoformat(),
               "necessary_capacity_screen": "passed", **metrics(workload, energy, case.parameters)}
        rows.append(row)
        case_entries.append({"hours": hours, "config": relative(config_path, output), **row})
        print(f"{hours:3d}h: {row['overload_slots']} overload slots; peak {row['peak_million_tokens_per_s']:.3f} M tokens/s; screen passed", flush=True)
    pd.DataFrame(rows).to_csv(output / "horizon_data_summary.csv", index=False)
    frozen = {"parameters": {k: v for k, v in base_raw["parameters"].items() if k != "horizon_slots"},
              "solver": base_raw["solver"],
              "inputs": {k: v for k, v in base_raw["inputs"].items() if k not in ("root", "workload_file")}}
    files = list(output.rglob("*.json")) + list(output.rglob("*.csv")) + list(output.rglob("*.gz")) + list(configs.glob("H*.yaml"))
    manifest = {"schema": "planning_horizon_experiment_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "horizons_hours": list(HORIZONS), "workload_scenario_id": config.inputs.workload_scenario_id,
                "energy_scenario_id": config.inputs.energy_scenario_id, "cases": case_entries,
                "controlled_settings": frozen, "anchor_config_sha256": source_digest(config.source_path),
                "source_provenance": {"burstgpt_file": str(source), "burstgpt_sha256": source_digest(source),
                                      "power_file": str(power_source), "power_sha256": source_digest(power_source)},
                "selection_rule": "Nested suffixes of seven consecutive days ending on the default workload/power day; preserve the exact 24h default and its burst, without optimization-based selection.",
                "time_alignment": "Retain the established per-day circular -900s BurstGPT alignment to reproduce the default; trace-day IDs are consecutive. Workload and power dates are paired by relative day and clock slot, not claimed to be contemporaneous.",
                "boundary_policy": "Empty initial backlog; finish within each horizon; cross-midnight execution allowed; nominal menu candidates and Flat-50/24 nominal deferral fixed across horizons.",
                "labels": "Sample each complete day once; keep fixed labels in every overlapping window. Partial-day type shares are inherited, never rebalanced.",
                "files": {relative(path, output): source_digest(path) for path in files}}
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared all eight horizons. Manifest: {output / 'experiment_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
