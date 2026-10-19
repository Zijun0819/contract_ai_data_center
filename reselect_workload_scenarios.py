"""Rebuild twelve dense/bursty BurstGPT workload scenarios from the original trace.

Selection uses workload characteristics only. Existing retained scenarios keep
their IDs, sample copies, and fixed labels. Run with --input BurstGPT_3.csv.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "src"))

import prepare_burstgpt_task_inputs as prep
from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.domain import SourceTask, TaskPool, WorkloadAggregate
from llm_contract_optimizer.data import load_energy_scenario, validate_type_counts
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.preferences import assign_known_types
from llm_contract_optimizer.sampling import plan_sampling, source_digest, validate_known_types

ANCHORS = {333: "R05-D333", 228: "R11-D228"}
SEED = 20260910


def candidate_metrics(daily, slots, capacity):
    rows = []
    for day, group in slots.groupby("day", sort=True):
        raw = group.raw_total_tokens.to_numpy(dtype=float)
        targets = np.rint(prep.QWEN_SHARE * (
            prep.AFFINE_SCALE * raw + prep.AFFINE_INTERCEPT_TPS
        ) * prep.SLOT_SECONDS)
        # Analytical screening profile; final metrics use the actual recipes.
        predicted = raw / raw.sum() * targets.sum()
        over = predicted > capacity
        runs = np.split(over, np.flatnonzero(np.diff(over)) + 1)
        rows.append({
            "trace_day": int(day), "target_tokens": int(targets.sum()),
            "estimated_mean_million_tokens_per_s": float(predicted.mean() / 900 / 1e6),
            "estimated_peak_million_tokens_per_s": float(predicted.max() / 900 / 1e6),
            "estimated_overload_slots": int(over.sum()),
            "estimated_dense_slots": int((predicted >= .8 * capacity).sum()),
            "estimated_overflow_fraction": float(np.maximum(predicted - capacity, 0).sum() / predicted.sum()),
            "estimated_top_hour_token_share": float(np.sort(predicted)[-4:].sum() / predicted.sum()),
            "estimated_longest_overload_run": max((len(run) for run in runs if run[0]), default=0),
            "estimated_max_suffix_capacity_ratio": float(np.max(
                np.cumsum(predicted[::-1])[::-1] / (capacity * np.arange(96, 0, -1))
            )),
        })
    return daily.merge(pd.DataFrame(rows), on="trace_day", validate="one_to_one")


def select_days(candidates, old_recipes):
    eligible = candidates.loc[
        (candidates.nonzero_slots >= 92) & (candidates.estimated_overload_slots > 0)
        & (candidates.estimated_max_suffix_capacity_ratio <= 1)
        & candidates.necessary_capacity_screen.isin(["not_evaluated", "passed"])
    ].copy()
    if not set(ANCHORS).issubset(set(eligible.trace_day)):
        raise ValueError("Both required anchor days must be eligible in this trace.")
    reasons = {333: "Retained anchor: concentrated burst", 228: "Retained anchor: extended burst"}
    selected = list(ANCHORS)
    dense = eligible.loc[~eligible.trace_day.isin(selected)].sort_values(
        ["estimated_overload_slots", "estimated_overflow_fraction", "trace_day"],
        ascending=[False, False, True], kind="stable",
    ).head(6)
    for day in dense.trace_day.astype(int):
        selected.append(day)
        reasons[day] = "Sustained overload: ranked by overloaded-slot count"
    bursts = eligible.loc[~eligible.trace_day.isin(selected)].sort_values(
        ["estimated_overflow_fraction", "estimated_peak_million_tokens_per_s", "trace_day"],
        ascending=[False, False, True], kind="stable",
    ).head(4)
    for day in bursts.trace_day.astype(int):
        selected.append(day)
        reasons[day] = "Concentrated burst: ranked by overflow-token fraction"
    if len(selected) != 12 or len(set(selected)) != 12:
        raise ValueError("The trace does not provide twelve distinct eligible stress days.")
    previous = {int(scenario.split("-D")[1]): scenario for scenario in old_recipes}
    mapping = {day: previous[day] for day in selected if day in previous}
    for day, scenario in ANCHORS.items():
        if mapping.get(day) != scenario:
            raise ValueError(f"Required original anchor is missing: {scenario}")
    used_numbers = {int(scenario[1:3]) for scenario in mapping.values()}
    available = iter(number for number in range(1, 13) if number not in used_numbers)
    for day in selected:
        if day not in mapping:
            mapping[day] = f"R{next(available):02d}-D{day}"
    ordered = sorted(selected, key=lambda day: mapping[day])
    return ordered, mapping, reasons, len(eligible)


def write_gzip(frame, path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                frame.to_csv(stream, index=False)


def source_pool(group, scenario):
    probability_names = ["p_opt_out"] + [f"p_type_{i}" for i in range(1, 9)]
    arrays = group[probability_names].to_numpy(dtype=float)
    tasks = tuple(SourceTask(
        str(row.task_id), int(row.arrival_slot), int(row.input_tokens),
        int(row.output_tokens), int(row.total_tokens), probabilities,
    ) for row, probabilities in zip(group.itertuples(index=False), arrays))
    return TaskPool(scenario, tasks)


def aggregate_and_validate(payload, pool, config, energy):
    counts = np.asarray([count for _, count in payload["draws"]], dtype=np.int64)
    if [task.task_id for task in pool.tasks] != [task_id for task_id, _ in payload["draws"]]:
        raise ValueError("Source IDs and ordering differ from the sampling recipe.")
    labels = validate_known_types(payload, counts, 8, config.inputs.immediate_execution_share)
    known = payload["known_types"]
    validate_type_counts(labels.sum(axis=0), known["target_distribution"], known["target_immediate_share"])
    sizes = np.asarray([task.total_tokens for task in pool.tasks], dtype=np.int64)
    slots = np.asarray([task.arrival_slot - 1 for task in pool.tasks], dtype=np.int64)
    if sum(int(count) * int(size) for count, size in zip(counts, sizes)) != payload["target_tokens"]:
        raise ValueError("Sampled tokens do not match the daily target.")
    raw_count = np.bincount(slots, minlength=96)
    sampled_count = np.bincount(slots, weights=counts, minlength=96).astype(np.int64)
    if not np.array_equal(sampled_count, payload["multiplier"] * raw_count):
        raise ValueError("Original arrival-count proportions were changed.")
    loads = np.zeros((9, 96))
    bills_by_arrival = np.zeros_like(loads)
    p = config.parameters
    bill = np.asarray([
        (p.input_tariff_usd_per_million_tokens * task.input_tokens
         + p.output_tariff_usd_per_million_tokens * task.output_tokens) / 1e6
        for task in pool.tasks
    ])
    for label in range(9):
        np.add.at(loads[label], slots, labels[:, label] * sizes)
        np.add.at(bills_by_arrival[label], slots, labels[:, label] * bill)
    workload = WorkloadAggregate(
        pool.scenario_id, loads[0], loads[1:], float(bills_by_arrival[0].sum()),
        bills_by_arrival[1:].sum(axis=1), labels.sum(axis=0), len(pool.tasks),
        opt_out_bill_usd_by_arrival=bills_by_arrival[0],
        participating_bill_usd_by_type_arrival=bills_by_arrival[1:],
    )
    inner = FluidInnerSolver(workload, energy, p, config.solver)
    screen = inner.screen_infeasibility()
    arrival = loads.sum(axis=0)
    capacity = p.token_capacity_per_slot
    overloaded = arrival > capacity
    runs = np.split(overloaded, np.flatnonzero(np.diff(overloaded)) + 1)
    metrics = {
        "daily_tokens": int(payload["target_tokens"]), "daily_tasks": int(payload["task_count"]),
        "source_records": len(pool.tasks),
        "mean_million_tokens_per_s": float(arrival.mean() / 900 / 1e6),
        "peak_million_tokens_per_s": float(arrival.max() / 900 / 1e6),
        "coefficient_of_variation": float(arrival.std(ddof=1) / arrival.mean()),
        "overload_slots": int(overloaded.sum()),
        "dense_slots": int((arrival >= .8 * capacity).sum()),
        "longest_overload_run_slots": max((len(run) for run in runs if run[0]), default=0),
        "immediate_service_overflow_fraction": float(np.maximum(arrival - capacity, 0).sum() / arrival.sum()),
        "top_four_slots_token_share": float(np.sort(arrival)[-4:].sum() / arrival.sum()),
        "peak_immediate_capacity_fraction": float(loads[0].max() / capacity),
        "necessary_capacity_screen": screen or "passed",
        "labels_fixed": True, "arrival_count_proportions_exact": True,
    }
    if metrics["overload_slots"] == 0:
        raise ValueError("A selected stress scenario has no actual capacity-overload slot.")
    return metrics, workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Original BurstGPT v2.0 BurstGPT_3.csv trace")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tmp/workload_reselection",
                        help="Staging and backup directory inside this repository's tmp directory")
    parser.add_argument("--baseline-data", type=Path,
                        help="Existing inputs supplying stable IDs and retained recipes; defaults to data/")
    parser.add_argument("--capacity-audit", type=Path,
                        help="Optional failed staging audit from this identical trace and preparation")
    args = parser.parse_args()
    source = args.input.resolve()
    work = args.work_dir.resolve()
    if not source.is_file():
        parser.error(f"Original BurstGPT trace does not exist: {source}")
    if (not work.is_relative_to(ROOT.resolve())
            or not work.is_relative_to((ROOT / "tmp").resolve())):
        parser.error("The working/backup directory must stay inside this repository's tmp directory.")
    work.mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT / "config/default.yaml")
    if len(config.parameters.type_lambdas) != 8 or config.parameters.horizon_slots != 96 or config.parameters.slot_seconds != 900:
        raise ValueError("This preparation requires eight types and 96 fifteen-minute slots.")
    if config.inputs.immediate_execution_share != .5:
        raise ValueError("The retained examples require the established 50% immediate-task convention.")
    data = PROJECT / "data"
    baseline_data = args.baseline_data.resolve() if args.baseline_data else data
    old_recipes = {path.stem.removesuffix(".sample"): json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted(baseline_data.glob("*.sample.json"))}
    if len(old_recipes) != 12:
        raise ValueError("Expected twelve existing workload recipes before replacement.")
    print("Scanning the original BurstGPT trace...", flush=True)
    daily, slots = prep.scan_trace(source, 400_000)
    candidates = candidate_metrics(daily, slots, config.parameters.token_capacity_per_slot)
    raw_digest = source_digest(source)
    candidates["necessary_capacity_screen"] = "not_evaluated"
    # Reuse only capacity audits from the identical source, capacity and type
    # preparation assumptions. This retains rejected days in the audit trail.
    prior_manifest_path = data / "workload_selection_manifest.json"
    if prior_manifest_path.exists():
        prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        if (prior.get("source_sha256") == raw_digest
                and prior.get("capacity_tokens_per_slot") == config.parameters.token_capacity_per_slot
                and prior.get("immediate_task_share") == .5 and prior.get("seed") == SEED):
            audited = {}
            for name in ("workload_candidate_metrics.csv", "workload_selection.csv"):
                path = data / name
                if path.exists():
                    frame = pd.read_csv(path, keep_default_na=False)
                    if "necessary_capacity_screen" in frame:
                        audited.update(zip(frame.trace_day.astype(int), frame.necessary_capacity_screen))
            candidates["necessary_capacity_screen"] = candidates.trace_day.map(audited).fillna("not_evaluated")
    if args.capacity_audit:
        audited = pd.read_csv(args.capacity_audit, keep_default_na=False).set_index("trace_day")
        for day, row in audited.iterrows():
            if row.necessary_capacity_screen != "not_evaluated":
                candidates.loc[candidates.trace_day.eq(day), "necessary_capacity_screen"] = row.necessary_capacity_screen
    days, mapping, reasons, eligible_count = select_days(candidates, old_recipes)
    print("Selected:", ", ".join(mapping[day] for day in days), flush=True)
    revision = work / f"revision_{time.time_ns()}"
    backup = revision / "backup"
    shutil.copytree(data, backup / "data")
    shutil.copytree(PROJECT / "data_visualization", backup / "data_visualization")
    # The shipped inputs are a complete baseline. Point provenance at their
    # snapshot so a later rebuild never depends on files it has replaced.
    if baseline_data == data.resolve():
        baseline_data = backup / "data"
    stage = revision / "staged_data"
    stage.mkdir()
    shutil.copy2(data / "energy_scenarios.csv", stage / "energy_scenarios.csv")
    print("Extracting original requests and building daily calibration targets...", flush=True)
    tasks = prep.extract_representative_tasks(source, days, {day: [reasons[day]] for day in days}, 400_000)
    if set(tasks.scenario_id) != set(mapping.values()):
        raise ValueError("Extracted scenario IDs differ from the stable scenario mapping.")
    if not tasks.task_id.is_unique or (tasks.input_tokens + tasks.output_tokens != tasks.total_tokens).any():
        raise ValueError("Invalid original task identities or token accounting.")
    write_gzip(tasks, stage / "representative_tasks_raw.csv.gz")
    # Reuse the historical daily calibration exactly; no artificial bursts are added.
    cohorts, targets = prep.build_scaled_cohorts(tasks)
    targets = prep.add_raw_slot_probabilities(tasks, targets)
    del cohorts
    targets.to_csv(stage / "representative_slot_inputs.csv", index=False)
    refined = tasks.drop(columns=[f"p_type_{i}" for i in range(1, 5)]).copy()
    for i in range(1, 5):
        refined[f"p_type_{2*i-1}"] = tasks[f"p_type_{i}"] / 2
        refined[f"p_type_{2*i}"] = tasks[f"p_type_{i}"] / 2
    write_gzip(refined, stage / "representative_tasks_eight_types.csv.gz")
    digest = source_digest(stage / "representative_tasks_eight_types.csv.gz")
    energy = load_energy_scenario(stage, config.inputs.energy_scenario_id, 96)
    target_by_scenario = targets.groupby("scenario_id").scaled_qwen_total_tokens.sum().astype("int64").to_dict()
    rows = []
    retained = []
    for day in days:
        scenario = mapping[day]
        print(f"Preparing and validating {scenario}...", flush=True)
        pool = source_pool(refined.loc[refined.scenario_id.eq(scenario)], scenario)
        if scenario in old_recipes:
            payload = old_recipes[scenario]
            if payload["target_tokens"] != target_by_scenario[scenario]:
                raise ValueError(f"Retained scenario's daily calibration changed: {scenario}")
            retained.append(scenario)
        else:
            plan = plan_sampling(pool, target_by_scenario[scenario], SEED)
            payload = {
                "schema": "individual_task_sampling_v1", "scenario_id": scenario,
                "seed": plan.seed, "multiplier": plan.multiplier,
                "target_tokens": plan.target_tokens, "task_count": plan.task_count,
                "draws": [[task.task_id, count] for task, count in zip(pool.tasks, plan.counts)],
                "known_types": assign_known_types(pool.tasks, plan.counts, .5, SEED),
            }
        payload.update(source_file="representative_tasks_eight_types.csv.gz", source_sha256=digest)
        metrics, workload = aggregate_and_validate(payload, pool, config, energy)
        row = {"scenario_id": scenario, "trace_day": day,
               "selection_reason": reasons[day], "retained_original_recipe": scenario in retained,
               **metrics}
        rows.append(row)
        candidates.loc[candidates.trace_day.eq(day), "necessary_capacity_screen"] = metrics["necessary_capacity_screen"]
        (stage / f"{scenario}.sample.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"  peak={metrics['peak_million_tokens_per_s']:.3f} M tokens/s; "
              f"overloaded slots={metrics['overload_slots']}; capacity screen={metrics['necessary_capacity_screen']}", flush=True)
    failed = [row for row in rows if row["necessary_capacity_screen"] != "passed"]
    if failed:
        candidates.to_csv(revision / "capacity_screen_audit.csv", index=False)
        raise ValueError(f"Selected days fail necessary capacity screens; active inputs were not replaced. Audit: {revision / 'capacity_screen_audit.csv'}")
    selection = pd.DataFrame(rows)
    selection.to_csv(stage / "workload_selection.csv", index=False)
    candidates["selected"] = candidates.trace_day.isin(days)
    candidates["selected_scenario_id"] = candidates.trace_day.map(mapping).fillna("")
    candidates["selection_reason"] = candidates.trace_day.map(reasons).fillna("")
    candidates.to_csv(stage / "workload_candidate_metrics.csv", index=False)
    manifest = {
        "schema": "dense_bursty_workload_selection_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_release": "https://github.com/HPMLL/BurstGPT/releases/tag/v2.0",
        "source_download": "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv",
        "source_sha256": raw_digest, "source_bytes": source.stat().st_size,
        "total_trace_days": len(daily), "eligible_stress_days": eligible_count,
        "selection_policy": {
            "eligibility": "At least 92 nonzero raw-token slots and one predicted overload; every estimated suffix must fit remaining same-day capacity; exclude audited immediate-load or same-day-capacity failures; final samples must pass necessary capacity screens",
            "anchors": ANCHORS,
            "density": "Six other days ranked by overloaded-slot count, then overflow fraction, then ascending trace day",
            "burst": "Four remaining days ranked by overflow fraction, then peak rate, then ascending trace day",
            "screening_profile": "Raw slot token proportions multiplied by the existing calibrated daily token target",
            "uses_optimizer_profit": False,
        },
        "capacity_tokens_per_slot": config.parameters.token_capacity_per_slot,
        "slot_seconds": 900, "dense_threshold_capacity_fraction": .8,
        "seed": SEED, "immediate_task_share": .5,
        "retained_original_recipes": retained,
        "retained_fields": "Task IDs, draws, copy counts, token targets and known_types; only shared-source path/hash refreshed",
        "active_source_sha256": digest,
        "backup_directory": backup.relative_to(ROOT).as_posix(),
        "baseline_data_directory": (baseline_data.relative_to(ROOT).as_posix()
                                    if baseline_data.is_relative_to(ROOT) else str(baseline_data)),
        "capacity_excluded_days": candidates.loc[
            ~candidates.necessary_capacity_screen.isin(["not_evaluated", "passed"]),
            ["trace_day", "necessary_capacity_screen"],
        ].to_dict("records"),
        "scenarios": rows,
        "interpretation": "Purposively selected workload stress cases; not an unbiased sample of ordinary operating days",
        "validation": "12 distinct scenarios; 96 slots each; exact daily tokens and slot task-count proportions; fixed type quotas; all actual profiles overload capacity; necessary capacity screens recorded",
    }
    (stage / "workload_selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    columns = ["scenario_id", "peak_million_tokens_per_s", "overload_slots", "immediate_service_overflow_fraction"]
    table = "| Scenario | Peak (million tokens/s) | Overloaded slots | Overflow fraction |\n|---|---:|---:|---:|\n"
    table += "\n".join(
        f"| {row.scenario_id} | {row.peak_million_tokens_per_s:.4f} | {row.overload_slots} | {row.immediate_service_overflow_fraction:.4f} |"
        for row in selection[columns].itertuples(index=False)
    )
    readme = f"""# Dense and bursty workload inputs

The twelve active recipes are workload stress scenarios from the original BurstGPT v2.0 `BurstGPT_3.csv` trace. R05-D333 and R11-D228 retain their original IDs, sampled copies, daily token targets, and fixed labels. Other previously selected days are retained when they meet the new selection rule. Replacement IDs keep the R01 through R12 positions.

Selection uses workload characteristics before optimization: retain the two requested anchors; select six other days by the number of overloaded 15-minute slots; select four remaining days by the fraction of daily tokens exceeding same-slot capacity. Ties use overflow fraction or peak rate, then trace day. Eligible days have at least 92 active raw-token slots and at least one predicted overload slot. Days failing audited mandatory-immediate or same-day-capacity conditions are excluded; every final selected sample passes those necessary screens. `workload_candidate_metrics.csv` records all {len(daily)} trace days, the exclusions and the selection decisions. `workload_selection.csv` reports actual sampled metrics rather than screening estimates. Passing these screens alone is not a complete IR/IC menu-feasibility certificate.

Capacity is {config.parameters.token_capacity_per_slot / 900 / 1e6:g} million tokens/s, read from the optimizer configuration at preparation time. Dense slots have demand at least 80% of capacity. Overflow fraction is `sum(max(arrival_tokens - capacity, 0)) / sum(arrival_tokens)`: a diagnostic for immediate service, not dropped workload under the proposed scheduler. Every selected sampled profile exceeds capacity in at least one slot.

{table}

Source request tokens and arrival timing are retained, including the established -900-second alignment. The original affine calibration supplies the daily token target; the sampled recipes preserve original per-slot request-count proportions and do not impose the affine background as the solver's slot profile. Eight fixed participating types and the established 50% immediate-task share are preserved. Preferences remain assumed inputs.

`representative_tasks_raw.csv.gz` contains the selected original requests and four-type preparation weights. `representative_tasks_eight_types.csv.gz` splits those weights into eight participating types. The twelve `.sample.json` recipes contain exact integer copy counts and fixed labels; their shared-source hashes have been refreshed. `representative_slot_inputs.csv` retains the historical calibration-target schema. `energy_scenarios.csv` is unchanged.

Reproduce from the repository root:

```powershell
python reselect_workload_scenarios.py --input path/to/BurstGPT_3.csv
python data_visualization/plot_scenarios.py
```

The preparation makes a full backup before replacing active inputs. This revision's backup is `{manifest['backup_directory']}`. The baseline supplying retained recipes is `{manifest['baseline_data_directory']}`. The inputs shipped in this repository provide the initial baseline; no historical workspace backup is required. `workload_selection_manifest.json` records hashes, criteria, and validation. These selected stress cases should be identified as such when reporting experimental results.
"""
    (stage / "README.md").write_text(readme, encoding="utf-8")
    expected = {f"{mapping[day]}.sample.json" for day in days}
    if {path.name for path in stage.glob("*.sample.json")} != expected:
        raise ValueError("The staged package does not contain exactly the twelve selected recipes.")
    if source_digest(data / "energy_scenarios.csv") != source_digest(stage / "energy_scenarios.csv"):
        raise ValueError("Energy inputs changed during preparation.")
    # Only individual obsolete recipe files are removed, after the backup and
    # staged-package checks; directories and historical result files are untouched.
    for path in stage.iterdir():
        if path.is_file():
            shutil.copy2(path, data / path.name)
    for path in data.glob("*.sample.json"):
        if path.name not in expected:
            if path.resolve().parent != data.resolve():
                raise ValueError("Refusing to remove a recipe outside the data directory.")
            path.unlink()
    print(selection[["scenario_id", "peak_million_tokens_per_s", "overload_slots", "immediate_service_overflow_fraction"]].to_string(index=False), flush=True)
    print("Backup:", backup.resolve(), flush=True)


if __name__ == "__main__":
    main()
