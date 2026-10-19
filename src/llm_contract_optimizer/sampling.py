"""Reproducible task resampling; count records are preprocessing provenance, not solver variables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import os

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from llm_contract_optimizer.domain import TaskPool
from llm_contract_optimizer.exceptions import DataValidationError, SolveError


@dataclass(frozen=True)
class SamplingPlan:
    counts: tuple[int, ...]
    multiplier: int
    target_tokens: int
    seed: int

    @property
    def task_count(self) -> int:
        return sum(self.counts)


def plan_sampling(workload: TaskPool, target_tokens: int, seed: int,
                  time_limit_seconds: float = 30.0) -> SamplingPlan:
    """Retain originals, bootstrap within slots, then reconcile the exact daily token total."""
    tasks = workload.tasks
    raw_tokens = sum(t.total_tokens for t in tasks)
    if raw_tokens <= 0 or target_tokens < raw_tokens:
        raise DataValidationError("Sampling requires positive source demand and a target at least as large as the source.")
    multiplier = max(1, (target_tokens + raw_tokens // 2) // raw_tokens)
    if target_tokens > raw_tokens:
        multiplier = max(2, multiplier)
    by_slot: dict[int, list[int]] = {}
    for i, task in enumerate(tasks):
        by_slot.setdefault(task.arrival_slot, []).append(i)
    rng = np.random.default_rng(seed)
    counts = np.ones(len(tasks), dtype=np.int64)
    for indices in by_slot.values():
        extra = (multiplier - 1) * len(indices)
        counts[indices] += rng.multinomial(extra, np.full(len(indices), 1 / len(indices)))
    sizes = np.array([t.total_tokens for t in tasks], dtype=np.int64)
    residual = int(target_tokens - sum(int(c) * int(x) for c, x in zip(counts, sizes)))
    if residual:
        # Restrict repair to a deterministic small set of real task templates per slot.
        # Include the largest available donor count for each size, extrema and adjacent sizes.
        candidates = []
        for indices in by_slot.values():
            size_index: dict[int, int] = {}
            for i in indices:
                x = int(sizes[i])
                if x not in size_index or counts[i] > counts[size_index[x]]:
                    size_index[x] = i
            ordered = sorted(size_index)
            if len(ordered) <= 32:
                selected = ordered
            else:
                selected = set(ordered[int(k)] for k in np.linspace(0, len(ordered)-1, 24))
                # Small differences improve the chances of exact integer reconciliation.
                adjacent = sorted(zip(ordered, ordered[1:]), key=lambda pair: pair[1]-pair[0])[:8]
                selected.update(x for pair in adjacent for x in pair)
            candidates.extend(size_index[x] for x in sorted(selected))
        # First use existing within-slot task substitutions, largest token difference first.
        # These preserve every slot count and never remove the last original occurrence.
        candidate_slots = {}
        for i in candidates:
            candidate_slots.setdefault(tasks[i].arrival_slot, []).append(i)
        moves = []
        for indices in candidate_slots.values():
            for a, low in enumerate(indices):
                for high in indices[a+1:]:
                    difference = int(sizes[high]-sizes[low])
                    if difference > 0:
                        moves.append((difference,low,high))
        for difference,low,high in sorted(moves,reverse=True):
            if not residual:
                break
            donor,recipient = (low,high) if residual>0 else (high,low)
            copies = min(abs(residual)//difference, int(counts[donor])-1)
            if copies:
                counts[donor]-=copies
                counts[recipient]+=copies
                residual -= copies*difference if residual>0 else -copies*difference
        if residual:
            J = len(candidates)
            slot_rows = {slot: r for r, slot in enumerate(sorted(by_slot))}
            token_row = len(slot_rows)
            rr, cc, vv = [], [], []
            lb = [-np.inf] * (token_row + 1 + 2 * J)
            ub = [np.inf] * len(lb)
            for r in range(token_row):
                lb[r] = ub[r] = 0.
            lb[token_row] = ub[token_row] = float(residual)
            for j, i in enumerate(candidates):
                rr.extend([slot_rows[tasks[i].arrival_slot], token_row])
                cc.extend([j, j]); vv.extend([1., float(sizes[i])])
                # t_j >= |delta_j|; minimize changes to the bootstrap draw.
                for sign, offset in ((1., 0), (-1., 1)):
                    row = token_row + 1 + 2*j + offset
                    rr.extend([row, row]); cc.extend([j, J+j]); vv.extend([sign, -1.])
                    ub[row] = 0.
            matrix = coo_matrix((vv,(rr,cc)), shape=(len(lb),2*J)).tocsr()
            lower = np.concatenate((1-counts[candidates], np.zeros(J)))
            upper = np.concatenate((np.array([multiplier*len(by_slot[tasks[i].arrival_slot])-counts[i]
                                              for i in candidates]), np.full(J,np.inf)))
            result = milp(np.concatenate((np.zeros(J),np.ones(J))),
                          integrality=np.concatenate((np.ones(J),np.zeros(J))),
                          bounds=Bounds(lower,upper), constraints=LinearConstraint(matrix,lb,ub),
                          options={"time_limit":time_limit_seconds,"mip_rel_gap":0.01})
            # Repair optimality is irrelevant; only exact integer feasibility is required.
            if result.x is None:
                raise SolveError("No exact token reconciliation found in the bounded sampling repair. "
                                 "No task size or arrival count was changed; this is not a proof that a broader repair is impossible.")
            delta = np.rint(result.x[:J]).astype(np.int64)
            if np.any(np.abs(result.x[:J]-delta)>1e-5):
                raise SolveError("Sampling repair did not return integer counts.")
            counts[candidates] += delta
    if np.any(counts < 1):
        raise SolveError("Sampling repair removed an original task.")
    if sum(int(c)*int(x) for c,x in zip(counts,sizes)) != target_tokens:
        raise SolveError("Exact daily token reconciliation failed; no approximate plan is accepted.")
    for indices in by_slot.values():
        if sum(int(counts[i]) for i in indices) != multiplier*len(indices):
            raise SolveError("Sampling changed the original arrival proportions.")
    return SamplingPlan(tuple(int(c) for c in counts), multiplier, target_tokens, seed)


def source_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write_plan(path: Path, source: Path, workload: TaskPool, plan: SamplingPlan) -> None:
    from llm_contract_optimizer.reporting import _atomic_text
    payload = {"schema":"individual_task_sampling_v1", "source_file":Path(os.path.relpath(source.resolve(), path.resolve().parent)).as_posix(),
               "source_sha256":source_digest(source), "scenario_id":workload.scenario_id,
               "seed":plan.seed, "multiplier":plan.multiplier, "target_tokens":plan.target_tokens,
               "task_count":plan.task_count,
               "draws":[[t.task_id,c] for t,c in zip(workload.tasks,plan.counts)]}
    _atomic_text(path,lambda stream:json.dump(payload,stream,separators=(",",":")))


def read_sampling_source(path: Path, scenario_id: str, parameters):
    from llm_contract_optimizer.data import load_source_pool
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "individual_task_sampling_v1" or payload.get("scenario_id") != scenario_id:
        raise DataValidationError("Sampling recipe schema or scenario does not match.")
    draws = payload.get("draws", [])
    if not draws or any(not isinstance(c,int) or isinstance(c,bool) or c<1 for _,c in draws):
        raise DataValidationError("Invalid sampling counts.")
    count = sum(c for _,c in draws)
    if count != payload["task_count"]:
        raise DataValidationError("Sampling task-count identity failed.")
    source = Path(payload["source_file"])
    if not source.is_absolute():
        source = (path.parent / source).resolve()
    if source.suffix == ".json" or source_digest(source) != payload["source_sha256"]:
        raise DataValidationError("Sampling source changed or is not an individual-task table.")
    raw = load_source_pool(source.parent, scenario_id, parameters, source)
    if [t.task_id for t in raw.tasks] != [task_id for task_id,_ in draws]:
        raise DataValidationError("Sampling source task IDs do not match.")
    counts = [c for _,c in draws]
    if sum(c*t.total_tokens for c,t in zip(counts,raw.tasks)) != payload["target_tokens"]:
        raise DataValidationError("Sampling daily token total does not match.")
    by_slot = {}
    for t,c in zip(raw.tasks,counts):
        before,after=by_slot.get(t.arrival_slot,(0,0))
        by_slot[t.arrival_slot]=(before+1,after+c)
    if any(after != payload["multiplier"]*before for before,after in by_slot.values()):
        raise DataValidationError("Sampling arrival distribution does not match.")
    return raw, counts


def validate_known_types(payload, counts, type_count, immediate_execution_share=None):
    known = payload.get("known_types")
    if not known or known.get("schema") != "fixed_task_types_v1":
        raise DataValidationError("Recipe has no fixed task labels; run assign-types before solving.")
    rows = known.get("counts_by_source_type", [])
    if (len(rows) != len(counts) or any(len(r) != type_count+1 for r in rows)
            or any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for r in rows for c in r)):
        raise DataValidationError("Invalid integer source/type counts.")
    labels = np.asarray(rows, dtype=np.int64)
    if not np.array_equal(labels.sum(axis=1), counts):
        raise DataValidationError("Known-type labeling changed source occurrence counts.")
    totals = labels.sum(axis=0)
    if totals.tolist() != known.get("scenario_counts"):
        raise DataValidationError("Known-type scenario count identity failed.")
    if not np.allclose(totals/sum(counts), known.get("actual_distribution", []), rtol=0, atol=1e-12):
        raise DataValidationError("Known-type distribution identity failed.")
    if immediate_execution_share is not None and abs(int(totals[0])-sum(counts)*immediate_execution_share) > .5000001:
        raise DataValidationError("Recipe immediate share differs from configuration; run assign-types explicitly. Solving does not relabel tasks.")
    return labels


def assign_recipe_types(path, scenario_id, parameters, target_share=.5, seed=20260910):
    from llm_contract_optimizer.preferences import assign_known_types
    from llm_contract_optimizer.reporting import _atomic_text
    raw, counts = read_sampling_source(path, scenario_id, parameters)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["known_types"] = assign_known_types(raw.tasks, counts, target_share, seed)
    validate_known_types(payload, counts, len(parameters.type_lambdas), target_share)
    _atomic_text(path, lambda f: json.dump(payload, f, separators=(",", ":")))
    return raw, counts, payload


def load_sampled_workload(path, scenario_id, parameters, immediate_execution_share=None):
    """Aggregate source/type counts exactly; do not instantiate sampled copies."""
    from llm_contract_optimizer.data import validate_type_counts
    from llm_contract_optimizer.domain import WorkloadAggregate
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw, counts = read_sampling_source(path, scenario_id, parameters)
    labels = validate_known_types(payload, counts, len(parameters.type_lambdas), immediate_execution_share)
    known = payload["known_types"]
    if known.get("target_distribution") is None:
        raise DataValidationError("Recipe must record its predefined type distribution.")
    totals = labels.sum(axis=0)
    validate_type_counts(totals, known.get("target_distribution"), known.get("target_immediate_share"))
    loads = np.zeros((len(parameters.type_lambdas)+1, parameters.horizon_slots))
    bills = np.zeros(len(loads))
    bills_by_arrival = np.zeros_like(loads)
    for source, row in zip(raw.tasks, labels):
        loads[:, source.arrival_slot-1] += row * source.total_tokens
        bill = (parameters.input_tariff_usd_per_million_tokens * source.input_tokens
                + parameters.output_tariff_usd_per_million_tokens * source.output_tokens) / 1e6
        bills += row * bill
        bills_by_arrival[:, source.arrival_slot-1] += row * bill
    if not np.isclose(loads.sum(), payload["target_tokens"], rtol=0, atol=1e-3):
        raise DataValidationError("Aggregated token conservation failed.")
    return WorkloadAggregate(
        scenario_id, loads[0], loads[1:], float(bills[0]), bills[1:], totals, len(raw.tasks),
        {"recipe": str(path.resolve()), "source_sha256": payload["source_sha256"],
         "target_distribution": known.get("target_distribution"),
         "actual_distribution": known["actual_distribution"], "labels_fixed": True},
        opt_out_bill_usd_by_arrival=bills_by_arrival[0],
        participating_bill_usd_by_type_arrival=bills_by_arrival[1:],
    )


def daily_targets(path: Path) -> dict[str,int]:
    result = {}
    seen = set()
    with path.open(encoding="utf-8-sig",newline="") as stream:
        for r in csv.DictReader(stream):
            scenario = r["scenario_id"]
            slot = int(r["arrival_slot"])
            key=(scenario,slot)
            if key in seen:
                raise DataValidationError("Duplicate target scenario/slot.")
            seen.add(key)
            target=int(r["scaled_qwen_total_tokens"])
            if target<0:
                raise DataValidationError("Negative target tokens.")
            result[scenario]=result.get(scenario,0)+target
    return result


def write_slot_audit(path: Path, workload: TaskPool, plan: SamplingPlan,
                     horizon_slots: int) -> None:
    rows = [[s,0,0,0,0] for s in range(1,horizon_slots+1)]
    for t,c in zip(workload.tasks,plan.counts):
        row=rows[t.arrival_slot-1]
        row[1]+=1; row[2]+=c; row[3]+=t.total_tokens; row[4]+=c*t.total_tokens
    from llm_contract_optimizer.reporting import _atomic_text
    def writer(stream):
        out=csv.writer(stream)
        out.writerow(["arrival_slot","original_tasks","sampled_tasks","original_tokens","sampled_tokens"])
        out.writerows(rows)
    _atomic_text(path,writer,newline="")
