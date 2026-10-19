"""Lossless windows of consecutive, independently sampled and fixed-label days."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from llm_contract_optimizer.domain import WorkloadAggregate
from llm_contract_optimizer.exceptions import DataValidationError
from llm_contract_optimizer.sampling import (
    read_sampling_source, source_digest, validate_known_types,
)

WINDOW_SCHEMA = "consecutive_task_window_v1"


def _day_arrays(path, scenario_id, parameters, immediate_share, distribution):
    from llm_contract_optimizer.data import validate_type_counts

    raw, copies = read_sampling_source(path, scenario_id, parameters)
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = validate_known_types(payload, copies, len(parameters.type_lambdas), immediate_share)
    known = payload["known_types"]
    if known.get("target_distribution") is None:
        raise DataValidationError("A consecutive day must have predefined fixed type quotas.")
    validate_type_counts(labels.sum(axis=0), known["target_distribution"],
                         known.get("target_immediate_share"))
    validate_type_counts(labels.sum(axis=0), distribution, immediate_share)
    shape = (len(parameters.type_lambdas) + 1, parameters.horizon_slots)
    tokens = np.zeros(shape)
    bills = np.zeros(shape)
    counts = np.zeros(shape, dtype=np.int64)
    records = np.zeros(parameters.horizon_slots, dtype=np.int64)
    for task, row in zip(raw.tasks, labels):
        slot = task.arrival_slot - 1
        tokens[:, slot] += row * task.total_tokens
        bill = (parameters.input_tariff_usd_per_million_tokens * task.input_tokens
                + parameters.output_tariff_usd_per_million_tokens * task.output_tokens) / 1e6
        bills[:, slot] += row * bill
        counts[:, slot] += row
        records[slot] += 1
    if not np.isclose(tokens.sum(), payload["target_tokens"], rtol=0, atol=1e-3):
        raise DataValidationError("Consecutive-day token conservation failed.")
    return tokens, bills, counts, records


def load_consecutive_workload(path, scenario_id, parameters, immediate_share=None,
                              distribution=None, cache=None):
    """Slice fixed occurrences; boundary slices never resample or relabel tasks.

    Quotas apply to each source day. A partial day has its inherited realized
    distribution, which need not equal the whole-day target to the last task.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != WINDOW_SCHEMA or payload.get("scenario_id") != scenario_id:
        raise DataValidationError("Consecutive workload schema or scenario does not match.")
    if (payload.get("horizon_slots") != parameters.horizon_slots
            or payload.get("slot_seconds") != parameters.slot_seconds):
        raise DataValidationError("Consecutive workload horizon does not match the configuration.")
    day_slots = payload.get("day_slots")
    if (isinstance(day_slots, bool) or not isinstance(day_slots, int)
            or day_slots * parameters.slot_seconds != 86400):
        raise DataValidationError("Consecutive recipes must describe complete 24-hour source days.")
    entries = payload.get("days", [])
    trace_days = [entry.get("trace_day") for entry in entries]
    if (not entries or any(isinstance(d, bool) or not isinstance(d, int) for d in trace_days)
            or any(b != a + 1 for a, b in zip(trace_days, trace_days[1:]))):
        raise DataValidationError("Workload source days must be unique and consecutive.")
    start = payload.get("start_slot_zero_based")
    if (isinstance(start, bool) or not isinstance(start, int) or start < 0
            or start + parameters.horizon_slots != len(entries) * day_slots):
        raise DataValidationError("Horizon must be an exact suffix of the fixed consecutive sequence.")
    day_parameters = replace(parameters, horizon_slots=day_slots)
    cache = {} if cache is None else cache
    token_parts, bill_parts, count_parts, record_parts = [], [], [], []
    provenance = []
    for index, entry in enumerate(entries):
        day_start = index * day_slots
        lo, hi = max(start - day_start, 0), day_slots
        if lo >= hi:
            continue
        recipe = (path.parent / entry["recipe"]).resolve()
        digest = source_digest(recipe)
        if digest != entry.get("recipe_sha256"):
            raise DataValidationError(f"Consecutive-day recipe changed: {recipe}")
        # Include every setting used in accounting or quota validation in the cache key.
        key = (recipe, digest, entry["scenario_id"], day_parameters, immediate_share, distribution)
        if key not in cache:
            cache[key] = _day_arrays(recipe, entry["scenario_id"], day_parameters,
                                     immediate_share, distribution)
        tokens, bills, counts, records = cache[key]
        token_parts.append(tokens[:, lo:hi])
        bill_parts.append(bills[:, lo:hi])
        count_parts.append(counts[:, lo:hi])
        record_parts.append(records[lo:hi])
        provenance.append({**entry, "included_day_slots": [lo + 1, hi]})
    tokens = np.concatenate(token_parts, axis=1)
    bills = np.concatenate(bill_parts, axis=1)
    counts = np.concatenate(count_parts, axis=1).sum(axis=1)
    if tokens.shape[1] != parameters.horizon_slots:
        raise DataValidationError("Consecutive workload has the wrong number of slots.")
    return WorkloadAggregate(
        scenario_id, tokens[0], tokens[1:], float(bills[0].sum()), bills[1:].sum(axis=1),
        counts, int(sum(records.sum() for records in record_parts)),
        {"window_recipe": str(path.resolve()), "labels_fixed": True,
         "quota_scope": "each_complete_source_day; partial windows inherit fixed labels",
         "actual_distribution": (counts / counts.sum()).tolist(), "days": provenance},
        opt_out_bill_usd_by_arrival=bills[0],
        participating_bill_usd_by_type_arrival=bills[1:],
    )
