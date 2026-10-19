"""Prepare fixed task labels from scenario distributions; report realized workloads."""
from __future__ import annotations


import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from llm_contract_optimizer.exceptions import DataValidationError


def calibrate_probabilities(probabilities, target_share: float, counts=None) -> np.ndarray:
    """Match expected task-count share by a common shift of immediate-service log odds.

    Counts are sampling occurrences, used only for input calibration. Structural zero/one
    probabilities and relative participating-type probabilities are preserved.
    """
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2
            or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-8)):
        raise DataValidationError("Calibration needs normalized task probability rows.")
    if isinstance(target_share, bool) or not np.isfinite(target_share) or not 0 <= target_share <= 1:
        raise DataValidationError("Immediate execution share must lie in [0, 1].")
    weights = np.ones(len(p)) if counts is None else np.asarray(counts, dtype=float)
    if (weights.shape != (len(p),) or not np.all(np.isfinite(weights))
            or np.any(weights <= 0) or not np.isfinite(weights.sum())):
        raise DataValidationError("Calibration counts must be finite and positive.")
    weights = weights / weights.sum()
    old = p[:, 0]
    lower = float(weights[old == 1].sum())
    upper = float(weights[old > 0].sum())
    if target_share < lower - 1e-12 or target_share > upper + 1e-12:
        raise DataValidationError("Immediate share is incompatible with fixed zero/one probabilities.")
    if abs(target_share - lower) <= 1e-12:
        new = (old == 1).astype(float)
    elif abs(target_share - upper) <= 1e-12:
        new = (old > 0).astype(float)
    else:
        odds = logit(old)
        shift = brentq(lambda b: float(weights @ expit(odds + b)) - target_share,
                       -1024., 1024., xtol=1e-13)
        new = expit(odds + shift)
    result = p.copy()
    result[:, 0] = new
    flexible = old < 1
    # Normalize by the stored participating sum to tolerate source rounding.
    result[flexible, 1:] *= ((1 - new[flexible])
                            / p[flexible, 1:].sum(axis=1))[:, None]
    result[~flexible, 1:] = 0.
    if abs(float(weights @ new) - target_share) > 1e-10:
        raise DataValidationError("Immediate-share calibration did not reach its target.")
    return result


def largest_remainder(total, shares):
    shares = np.asarray(shares, dtype=float)
    if total < 0 or not np.all(np.isfinite(shares)) or np.any(shares < 0) or shares.sum() <= 0:
        raise DataValidationError("Invalid type-count target.")
    expected = total * shares / shares.sum()
    counts = np.floor(expected).astype(np.int64)
    missing = int(total - counts.sum())
    order = np.argsort(-(expected-counts), kind="stable")
    counts[order[:missing]] += 1
    return counts


def assign_known_types(tasks, counts, target_share, seed):
    """Prepare integer labels with fixed row counts and scenario type totals.

    Source weights guide deterministic balanced rounding; they never enter the solver.
    Count entries encode consecutive uniquely identified copies, not scheduling groups.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (len(tasks),) or np.any(counts < 1):
        raise DataValidationError("Each source task needs a positive occurrence count.")
    source = np.asarray([t.probabilities for t in tasks])
    probabilities = source if target_share is None else calibrate_probabilities(source, target_share, counts)
    expected = counts[:, None] * probabilities
    total = int(counts.sum())
    target_distribution = expected.sum(axis=0) / total
    target = largest_remainder(total, target_distribution)
    if target_share is not None:
        target[0] = int(np.floor(total*target_share+.5))
        remaining = total-int(target[0])
        target[1:] = largest_remainder(remaining, target_distribution[1:]) if remaining else 0
    labels = balanced_integer_labels(expected, counts, target, seed)
    return {"schema": "fixed_task_types_v1", "method": "balanced_integer_rounding",
            "seed": int(seed), "target_immediate_share": target_share,
            "target_distribution": target_distribution.tolist(), "scenario_counts": target.tolist(),
            "actual_distribution": (target/total).tolist(), "counts_by_source_type": labels.tolist(),
            "copy_mapping": "Zero-based copy IDs partitioned into consecutive type blocks 0 through I."}


def balanced_integer_labels(expected, counts, target, seed):
    """Round a fixed occurrence table while preserving every row and type quota."""
    expected = np.asarray(expected, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if (expected.shape != (len(counts), len(target)) or np.any(counts < 0)
            or np.any(target < 0) or target.sum() != counts.sum()
            or not np.all(np.isfinite(expected)) or np.any(expected < 0)
            or not np.allclose(expected.sum(axis=1), counts, rtol=0, atol=1e-7)):
        raise DataValidationError("Invalid expected labels, row counts or type quotas.")
    labels = np.floor(expected).astype(np.int64)
    missing = counts-labels.sum(axis=1)
    rng = np.random.default_rng(seed)
    tie = rng.random(expected.shape)*1e-12
    ranking = np.argsort(-(expected-labels+tie), axis=1, kind="stable")
    for rank in range(expected.shape[1]):
        rows = np.flatnonzero(missing > rank)
        labels[rows, ranking[rows, rank]] += 1
    discrepancy = labels.sum(axis=0)-target
    # Transfer one occurrence per chosen row; preserve row totals throughout.
    # Prefer transfers with the smallest increase in squared rounding error.
    while np.any(discrepancy):
        donor, recipient = int(np.argmax(discrepancy)), int(np.argmin(discrepancy))
        eligible = np.flatnonzero(labels[:, donor] > 0)
        amount = min(int(discrepancy[donor]), int(-discrepancy[recipient]), len(eligible))
        if amount <= 0:
            raise DataValidationError("Could not reconcile known-type counts.")
        score = ((labels[eligible, recipient]-expected[eligible, recipient])
                 -(labels[eligible, donor]-expected[eligible, donor])+tie[eligible, recipient])
        chosen = eligible[np.argsort(score, kind="stable")[:amount]]
        labels[chosen, donor] -= 1
        labels[chosen, recipient] += 1
        discrepancy[donor] -= amount
        discrepancy[recipient] += amount
    assert np.array_equal(labels.sum(axis=1), counts)
    assert np.array_equal(labels.sum(axis=0), target)
    return labels


def workload_statistics(tasks, parameters):
    counts = np.bincount([t.task_type for t in tasks], minlength=len(parameters.type_lambdas)+1)
    total = sum(t.total_tokens for t in tasks)
    loads = np.zeros(parameters.horizon_slots)
    for task in tasks:
        if task.task_type == 0:
            loads[task.arrival_slot-1] += task.total_tokens
    return {"type_counts": counts.tolist(), "type_distribution": (counts/len(tasks)).tolist(),
            "immediate_task_share": float(counts[0]/len(tasks)),
            "immediate_token_share": float(loads.sum()/total) if total else 0.,
            "peak_immediate_capacity_fraction": float(loads.max()/parameters.token_capacity_per_slot),
            "peak_immediate_slot": int(loads.argmax())+1}
