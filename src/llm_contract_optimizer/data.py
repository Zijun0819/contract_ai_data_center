from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from llm_contract_optimizer.config import StaticParameters
from llm_contract_optimizer.domain import Task, EnergyScenario, WorkloadAggregate
from llm_contract_optimizer.exceptions import DataValidationError

ENERGY_COLUMNS = {
    "energy_scenario_id",
    "date",
    "slot",
    "ppa_kwh",
    "gt_kwh",
    "pv_kwh",
    "bess_discharge_kwh",
    "bess_charge_kwh",
    "day_ahead_price_usd_per_kwh",
}

def _require_columns(actual: set[str], required: set[str], file_path: Path) -> None:
    missing = sorted(required - actual)
    if missing:
        raise DataValidationError(f"Missing columns in {file_path}: {', '.join(missing)}")


def load_energy_scenario(
    input_root: Path, scenario_id: str, horizon_slots: int
) -> EnergyScenario:
    path = input_root / "energy_scenarios.csv"
    if not path.is_file():
        raise DataValidationError(f"Energy input file not found: {path}")

    selected: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require_columns(set(reader.fieldnames or ()), ENERGY_COLUMNS, path)
        for row in reader:
            if row["energy_scenario_id"] != scenario_id:
                continue
            try:
                slot = int(row["slot"])
            except ValueError as exc:
                raise DataValidationError(f"Invalid slot in {path}: {row['slot']!r}") from exc
            if slot in selected:
                raise DataValidationError(
                    f"Duplicate slot {slot} for energy scenario {scenario_id}."
                )
            selected[slot] = row

    expected = set(range(1, horizon_slots + 1))
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise DataValidationError(
            f"Energy scenario {scenario_id!r} must contain slots 1..{horizon_slots}; "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )

    def values(column: str) -> np.ndarray:
        try:
            result = np.array(
                [float(selected[slot][column]) for slot in range(1, horizon_slots + 1)],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise DataValidationError(
                f"Non-numeric value in energy column {column!r} for scenario {scenario_id}."
            ) from exc
        if not np.all(np.isfinite(result)) or np.any(result < 0):
            raise DataValidationError(
                f"Energy column {column!r} must be finite and nonnegative for {scenario_id}."
            )
        return result

    dates = list(dict.fromkeys(selected[slot]["date"] for slot in range(1, horizon_slots + 1)))
    date_label = dates[0] if len(dates) == 1 else f"{dates[0]}..{dates[-1]}"

    return EnergyScenario(
        scenario_id=scenario_id,
        date=date_label,
        ppa_kwh=values("ppa_kwh"),
        gas_turbine_kwh=values("gt_kwh"),
        pv_kwh=values("pv_kwh"),
        bess_discharge_kwh=values("bess_discharge_kwh"),
        bess_charge_kwh=values("bess_charge_kwh"),
        day_ahead_price_usd_per_kwh=values("day_ahead_price_usd_per_kwh"),
    )


def _iter_task_rows(path, scenario_id, parameters, source_pool):
    from llm_contract_optimizer.domain import SourceTask, Task
    if not path.is_file():
        raise DataValidationError(f"Individual-task input not found: {path}")
    probability_names = ["p_opt_out"] + [f"p_type_{i}" for i in range(1, len(parameters.type_lambdas)+1)]
    required = {"task_id", "arrival_slot", "input_tokens", "output_tokens", "total_tokens"}
    required.update(probability_names if source_pool else ["task_type"])
    opener = gzip.open if path.suffix == ".gz" else open
    seen = set()
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        _require_columns(columns, required, path)
        if "cohort_count" in columns:
            raise DataValidationError("Cohort scheduling inputs are not supported.")
        scenario_column = "workload_scenario_id" if "workload_scenario_id" in columns else "scenario_id"
        _require_columns(columns, {scenario_column}, path)
        for row_number, row in enumerate(reader, start=2):
            if row[scenario_column] != scenario_id:
                continue
            try:
                arrival = int(row["arrival_slot"])
                inp, out, total = (int(row[k]) for k in ("input_tokens", "output_tokens", "total_tokens"))
                task_id = row["task_id"].strip()
                label = None if source_pool else int(row["task_type"])
                probabilities = np.array([float(row[k]) for k in probability_names]) if source_pool else None
            except (ValueError, TypeError, AttributeError) as exc:
                raise DataValidationError(f"Invalid task row {row_number}: {exc}") from exc
            if not task_id or task_id in seen:
                raise DataValidationError(f"Missing or duplicate task ID at row {row_number}.")
            if inp < 0 or out < 0 or inp + out != total:
                raise DataValidationError(f"Token identity failed at row {row_number}.")
            if not 1 <= arrival <= parameters.horizon_slots:
                raise DataValidationError(f"Arrival outside horizon at row {row_number}.")
            if source_pool:
                if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
                        or not np.isclose(probabilities.sum(), 1., rtol=0., atol=1e-8)):
                    raise DataValidationError(f"Invalid source probabilities at row {row_number}.")
            elif not 0 <= label <= len(parameters.type_lambdas):
                raise DataValidationError(f"Invalid task_type at row {row_number}.")
            seen.add(task_id)
            yield (SourceTask(task_id, arrival, inp, out, total, probabilities) if source_pool
                   else Task(task_id, arrival, inp, out, total, label, row.get("source_task_id") or None))
    if not seen:
        raise DataValidationError(f"Workload scenario not found: {scenario_id!r}")


def load_source_pool(input_root, scenario_id, parameters, workload_file=None):
    from llm_contract_optimizer.domain import TaskPool
    path = workload_file or input_root / "workload_tasks.csv.gz"
    return TaskPool(scenario_id, tuple(_iter_task_rows(path, scenario_id, parameters, True)))


def validate_type_counts(counts, distribution=None, immediate_share=None):
    from llm_contract_optimizer.preferences import largest_remainder
    counts = np.asarray(counts)
    if counts.ndim != 1 or not len(counts) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise DataValidationError("Type counts must be a nonnegative integer vector.")
    total = int(np.sum(counts))
    if immediate_share is not None and abs(int(counts[0]) - total * immediate_share) > .5000001:
        raise DataValidationError("Fixed immediate-task count differs from the target; solving never relabels tasks.")
    if distribution is not None:
        shares = np.asarray(distribution, dtype=float)
        if (shares.shape != np.shape(counts) or not np.all(np.isfinite(shares))
                or np.any(shares < 0) or not np.isclose(shares.sum(), 1., atol=1e-10, rtol=0)):
            raise DataValidationError("Invalid predefined type distribution.")
        if immediate_share is not None and not np.isclose(shares[0], immediate_share, rtol=0, atol=1e-10):
            raise DataValidationError("Predefined distribution and immediate-service target disagree.")
        target = largest_remainder(total, shares)
        if immediate_share is not None:
            target[0] = int(np.floor(total * immediate_share + .5))
            remaining = total - int(target[0])
            target[1:] = largest_remainder(remaining, shares[1:]) if remaining else 0
        if not np.array_equal(counts, target):
            raise DataValidationError("Fixed labels do not meet the predefined integer type quotas.")


def load_workload_aggregate(input_root, scenario_id, parameters, workload_file=None,
                            immediate_execution_share=None, type_distribution=None):
    """Account for every occurrence once; optimization never uses preference weights."""
    path = workload_file or input_root / "workload_tasks.csv.gz"
    if not path.is_file():
        raise DataValidationError(f"Workload input not found: {path}")
    if path.suffix == ".json":
        import json
        from llm_contract_optimizer.horizon_data import WINDOW_SCHEMA, load_consecutive_workload
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == WINDOW_SCHEMA:
            # Validate quotas on complete source days, then preserve labels in the slice.
            return load_consecutive_workload(path, scenario_id, parameters,
                                              immediate_execution_share, type_distribution)
        from llm_contract_optimizer.sampling import load_sampled_workload
        result = load_sampled_workload(path, scenario_id, parameters, immediate_execution_share)
    else:
        result = account_tasks(scenario_id, _iter_task_rows(path, scenario_id, parameters, False), parameters)
    validate_type_counts(result.type_counts, type_distribution, immediate_execution_share)
    return result


def account_tasks(scenario_id, tasks, parameters):
    """Stream exact token and bill totals without retaining task decision records."""
    loads = np.zeros((len(parameters.type_lambdas) + 1, parameters.horizon_slots))
    bills = np.zeros(len(loads))
    bills_by_arrival = np.zeros_like(loads)
    counts = np.zeros(len(loads), dtype=np.int64)
    records = 0
    for t in tasks:
        if (isinstance(t.task_type, bool) or not isinstance(t.task_type, (int, np.integer))
                or not 0 <= t.task_type < len(bills)):
            raise DataValidationError("Every task must have one valid integer task_type.")
        if (not 1 <= t.arrival_slot <= parameters.horizon_slots or t.input_tokens < 0
                or t.output_tokens < 0 or t.total_tokens != t.input_tokens + t.output_tokens):
            raise DataValidationError("Task arrival or token identity is invalid.")
        loads[t.task_type, t.arrival_slot-1] += t.total_tokens
        bill = (parameters.input_tariff_usd_per_million_tokens * t.input_tokens
                + parameters.output_tariff_usd_per_million_tokens * t.output_tokens) / 1e6
        bills[t.task_type] += bill
        bills_by_arrival[t.task_type, t.arrival_slot-1] += bill
        counts[t.task_type] += 1
        records += 1
    return WorkloadAggregate(
        scenario_id, loads[0], loads[1:], float(bills[0]), bills[1:], counts, records,
        opt_out_bill_usd_by_arrival=bills_by_arrival[0],
        participating_bill_usd_by_type_arrival=bills_by_arrival[1:],
    )
