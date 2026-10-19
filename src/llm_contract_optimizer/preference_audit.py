"""Audit fixed task labels without expanding the sampled workload."""
from __future__ import annotations
import csv
import json
import numpy as np
from llm_contract_optimizer.exceptions import DataValidationError
from llm_contract_optimizer.reporting import _atomic_text
from llm_contract_optimizer.sampling import read_sampling_source, validate_known_types, assign_recipe_types


def audit_rows(raw, counts, payload, parameters):
    labels = validate_known_types(payload, counts, len(parameters.type_lambdas))
    total = sum(counts)
    tokens = np.array([t.total_tokens for t in raw.tasks], dtype=np.int64)
    arrivals = np.array([t.arrival_slot-1 for t in raw.tasks])
    daily_tokens = sum(int(c)*int(x) for c,x in zip(counts,tokens))
    rows = []
    for j in range(labels.shape[1]):
        task_count = int(labels[:,j].sum())
        typed_tokens = labels[:,j]*tokens
        loads = np.bincount(arrivals, weights=typed_tokens, minlength=parameters.horizon_slots)
        token_count = sum(int(x) for x in typed_tokens)
        rows.append({"scenario":raw.scenario_id, "task_type":j, "task_count":task_count,
                     "target_share":payload["known_types"]["target_distribution"][j],
                     "actual_share":task_count/total, "tokens":token_count,
                     "token_share":token_count/daily_tokens if daily_tokens else 0.,
                     "peak_arrival_capacity_fraction":float(loads.max()/parameters.token_capacity_per_slot),
                     "immediate_capacity_check_passed":bool(loads.max() <= parameters.token_capacity_per_slot+1e-6) if j == 0 else ""})
    assert sum(r["task_count"] for r in rows) == total
    assert sum(r["tokens"] for r in rows) == payload["target_tokens"]
    return rows


def audit_recipes(config, output_file, all_scenarios=False, assign=False, seed=20260910):
    recipe = config.inputs.workload_file
    if recipe is None or not recipe.name.endswith(".sample.json"):
        raise DataValidationError("Type assignment/audit requires a .sample.json recipe.")
    paths = sorted(recipe.parent.glob("*.sample.json")) if all_scenarios else [recipe]
    rows = []
    for path in paths:
        scenario = path.name.removesuffix(".sample.json")
        if assign:
            raw, counts, payload = assign_recipe_types(path, scenario, config.parameters,
                                                       config.inputs.immediate_execution_share, seed)
        else:
            raw, counts = read_sampling_source(path, scenario, config.parameters)
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_known_types(payload, counts, len(config.parameters.type_lambdas), config.inputs.immediate_execution_share)
        scenario_rows = audit_rows(raw, counts, payload, config.parameters)
        rows.extend(scenario_rows)
        print(json.dumps({"scenario":scenario, "known_type_counts":[r["task_count"] for r in scenario_rows],
                          "immediate_capacity_check_passed":scenario_rows[0]["immediate_capacity_check_passed"]}), flush=True)
    def writer(stream):
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    _atomic_text(output_file, writer, newline="")
    return rows
