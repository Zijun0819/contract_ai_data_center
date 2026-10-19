"""Prepare task-level and full-scale cohort inputs from BurstGPT v2.0.

The raw task table preserves one observed request per row.  The scaled cohort
table implements the affine, Qwen-share-adjusted reconstruction defined by
the calibration constants below without materializing the roughly
hundred-million individual tasks implied by a full campus day.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SLOT_SECONDS = 900
SLOTS_PER_DAY = 96
DAY_SECONDS = 86_400
TIME_SHIFT_SECONDS = -900
AFFINE_SCALE = 0.8507008955943308
AFFINE_INTERCEPT_TPS = 1_459_538.739117371
QWEN_SHARE = 0.55255
SOURCE_COPY_MULTIPLIER = AFFINE_SCALE * SLOT_SECONDS * QWEN_SHARE
BASELINE_TOKENS_PER_SLOT = AFFINE_INTERCEPT_TPS * SLOT_SECONDS * QWEN_SHARE
MAX_TASK_TOKENS = 1_000 * SLOT_SECONDS
RANDOM_SEED = 20260905

P_NAMES = ["p_opt_out", "p_type_1", "p_type_2", "p_type_3", "p_type_4"]
BASE_PROBABILITIES = np.array([0.15, 0.30, 0.25, 0.18, 0.12], dtype=float)
SIZE_BINS = np.array([0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, np.inf])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the original BurstGPT v2.0 BurstGPT_3.csv trace",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "burstgpt_source_preparation",
    )
    parser.add_argument("--representative-days", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=400_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probability_matrix(total_tokens: np.ndarray, model: np.ndarray, log_type: np.ndarray) -> np.ndarray:
    """Create transparent, assumed type probabilities conditional on task traits."""
    total = np.maximum(np.asarray(total_tokens, dtype=float), 1.0)
    size_score = np.clip((np.log1p(total) - math.log(1024.0)) / 1.25, -2.0, 2.0)
    model_text = pd.Series(model, dtype="string").fillna("").str.upper().to_numpy()
    log_text = pd.Series(log_type, dtype="string").fillna("").str.upper().to_numpy()
    gpt4 = np.char.find(model_text.astype(str), "GPT-4") >= 0
    api = np.char.find(log_text.astype(str), "API") >= 0
    conversation = np.char.find(log_text.astype(str), "CONVERSATION") >= 0

    logits = np.tile(np.log(BASE_PROBABILITIES), (len(total), 1))
    logits[:, 0] += -0.15 * size_score + 0.12 * conversation + 0.12 * gpt4
    logits[:, 1] += 0.45 * size_score + 0.30 * api
    logits[:, 2] += 0.20 * size_score + 0.10 * api
    logits[:, 3] += -0.05 * size_score + 0.10 * conversation
    logits[:, 4] += -0.35 * size_score + 0.20 * conversation + 0.20 * gpt4
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs = np.round(probs, 8)
    probs[:, -1] = 1.0 - probs[:, :-1].sum(axis=1)
    return probs


def scan_trace(path: Path, chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = ["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type", "Elapsed time"]
    stats: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sizes: dict[int, list[np.ndarray]] = defaultdict(list)
    elapsed: dict[int, list[np.ndarray]] = defaultdict(list)
    slot_rows: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=fields, chunksize=chunk_size):
        for column in ["Timestamp", "Request tokens", "Response tokens", "Total tokens", "Elapsed time"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=["Timestamp", "Request tokens", "Response tokens", "Total tokens"])
        chunk["day"] = np.floor(chunk["Timestamp"] / DAY_SECONDS).astype(int)
        second = np.mod(chunk["Timestamp"].to_numpy(dtype=float), DAY_SECONDS)
        chunk["arrival_slot"] = (np.floor(np.mod(second + TIME_SHIFT_SECONDS, DAY_SECONDS) / SLOT_SECONDS).astype(int) + 1)
        chunk["is_gpt4"] = chunk["Model"].astype("string").str.upper().str.contains("GPT-4", regex=False, na=False).astype(int)
        chunk["is_api"] = chunk["Log Type"].astype("string").str.upper().str.contains("API", regex=False, na=False).astype(int)
        chunk["is_zero_response"] = (chunk["Response tokens"] <= 0).astype(int)

        grouped = chunk.groupby("day", observed=True)
        for day, group in grouped:
            day = int(day)
            record = stats[day]
            record["request_count"] += len(group)
            record["request_tokens"] += group["Request tokens"].sum()
            record["response_tokens"] += group["Response tokens"].sum()
            record["total_tokens"] += group["Total tokens"].sum()
            record["gpt4_count"] += group["is_gpt4"].sum()
            record["api_count"] += group["is_api"].sum()
            record["zero_response_count"] += group["is_zero_response"].sum()
            record["max_total_tokens"] = max(record["max_total_tokens"], float(group["Total tokens"].max()))
            sizes[day].append(group["Total tokens"].to_numpy(dtype=np.int64))
            elapsed_values = group["Elapsed time"].dropna().to_numpy(dtype=float)
            if len(elapsed_values):
                elapsed[day].append(elapsed_values)

        slot = (
            chunk.groupby(["day", "arrival_slot"], observed=True)
            .agg(
                raw_request_count=("Total tokens", "size"),
                raw_request_tokens=("Request tokens", "sum"),
                raw_response_tokens=("Response tokens", "sum"),
                raw_total_tokens=("Total tokens", "sum"),
            )
            .reset_index()
        )
        slot_rows.append(slot)

    slot_summary = pd.concat(slot_rows, ignore_index=True)
    slot_summary = (
        slot_summary.groupby(["day", "arrival_slot"], observed=True, as_index=False)
        .sum(numeric_only=True)
    )
    complete = pd.MultiIndex.from_product(
        [sorted(stats), range(1, SLOTS_PER_DAY + 1)], names=["day", "arrival_slot"]
    ).to_frame(index=False)
    slot_summary = complete.merge(slot_summary, how="left", on=["day", "arrival_slot"]).fillna(0)

    daily_rows: list[dict[str, float | int]] = []
    for day in sorted(stats):
        record = stats[day]
        count = int(record["request_count"])
        tokens = float(record["total_tokens"])
        day_slots = slot_summary.loc[slot_summary["day"] == day, "raw_total_tokens"].to_numpy(dtype=float)
        size_values = np.concatenate(sizes[day]) if sizes[day] else np.array([], dtype=float)
        elapsed_values = np.concatenate(elapsed[day]) if elapsed[day] else np.array([], dtype=float)
        mean_slot = day_slots.mean()
        daily_rows.append(
            {
                "trace_day": day,
                "request_count": count,
                "request_tokens": int(record["request_tokens"]),
                "response_tokens": int(record["response_tokens"]),
                "total_tokens": int(tokens),
                "mean_total_tokens_per_task": tokens / max(count, 1),
                "p95_total_tokens_per_task": float(np.quantile(size_values, 0.95)) if len(size_values) else np.nan,
                "max_total_tokens_per_task": int(record["max_total_tokens"]),
                "response_token_share": record["response_tokens"] / max(tokens, 1.0),
                "gpt4_request_share": record["gpt4_count"] / max(count, 1),
                "api_request_share": record["api_count"] / max(count, 1),
                "zero_response_share": record["zero_response_count"] / max(count, 1),
                "p95_elapsed_seconds": float(np.quantile(elapsed_values, 0.95)) if len(elapsed_values) else np.nan,
                "nonzero_slots": int(np.count_nonzero(day_slots)),
                "peak_slot_tokens": int(day_slots.max()),
                "peak_to_mean_slot_tokens": float(day_slots.max() / mean_slot) if mean_slot > 0 else np.nan,
                "slot_token_cv": float(day_slots.std(ddof=0) / mean_slot) if mean_slot > 0 else np.nan,
            }
        )
    return pd.DataFrame(daily_rows), slot_summary


def select_representative_days(daily: pd.DataFrame, number: int) -> tuple[list[int], dict[int, list[str]]]:
    eligible = daily.loc[daily["nonzero_slots"] >= 92].copy()
    reasons: dict[int, list[str]] = defaultdict(list)
    selected: list[int] = []

    def add(day: int, reason: str) -> None:
        day = int(day)
        if day not in selected:
            selected.append(day)
        reasons[day].append(reason)

    if 308 in set(eligible["trace_day"]):
        add(308, "paper-profile match")
    anchors = [
        ("total_tokens", "highest daily token volume", "max"),
        ("total_tokens", "lowest daily token volume", "min"),
        ("request_count", "highest request count", "max"),
        ("peak_to_mean_slot_tokens", "most bursty slot profile", "max"),
        ("p95_total_tokens_per_task", "largest upper-tail tasks", "max"),
        ("response_token_share", "output-heavy task mix", "max"),
        ("gpt4_request_share", "highest GPT-4 share", "max"),
        ("api_request_share", "highest API share", "max"),
    ]
    for metric, reason, direction in anchors:
        ordered = eligible.sort_values(metric, ascending=(direction == "min"), kind="stable")
        for day in ordered["trace_day"]:
            if int(day) not in selected:
                add(int(day), reason)
                break
        if len(selected) >= number:
            break

    feature_cols = [
        "request_count", "total_tokens", "p95_total_tokens_per_task", "response_token_share",
        "gpt4_request_share", "api_request_share", "peak_to_mean_slot_tokens", "slot_token_cv",
    ]
    values = eligible[feature_cols].astype(float)
    standardized = (values - values.mean()) / values.std(ddof=0).replace(0, 1)
    feature_by_day = dict(zip(eligible["trace_day"].astype(int), standardized.to_numpy()))
    while len(selected) < min(number, len(eligible)):
        candidates = [int(day) for day in eligible["trace_day"] if int(day) not in selected]
        distances = {}
        for day in candidates:
            distances[day] = min(np.linalg.norm(feature_by_day[day] - feature_by_day[s]) for s in selected)
        best = max(candidates, key=lambda d: (distances[d], -d))
        add(best, "diversity coverage")
    return selected[:number], reasons


def extract_representative_tasks(
    path: Path, selected_days: list[int], reasons: dict[int, list[str]], chunk_size: int
) -> pd.DataFrame:
    usecols = ["Timestamp", "Session ID", "Elapsed time", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"]
    scenario_map = {day: f"R{idx:02d}-D{day}" for idx, day in enumerate(selected_days, start=1)}
    pieces: list[pd.DataFrame] = []
    row_offset = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, dtype={"Session ID": "string"}):
        row_ids = np.arange(row_offset + 2, row_offset + len(chunk) + 2, dtype=np.int64)
        row_offset += len(chunk)
        timestamp = pd.to_numeric(chunk["Timestamp"], errors="coerce")
        day = np.floor(timestamp / DAY_SECONDS)
        mask = day.isin(selected_days) & timestamp.notna()
        if not mask.any():
            continue
        part = chunk.loc[mask].copy()
        part["source_row"] = row_ids[mask.to_numpy()]
        for column in ["Timestamp", "Elapsed time", "Request tokens", "Response tokens", "Total tokens"]:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.dropna(subset=["Timestamp", "Request tokens", "Response tokens", "Total tokens"])
        part["trace_day"] = np.floor(part["Timestamp"] / DAY_SECONDS).astype(int)
        raw_second = np.mod(part["Timestamp"].to_numpy(dtype=float), DAY_SECONDS)
        aligned_second = np.mod(raw_second + TIME_SHIFT_SECONDS, DAY_SECONDS)
        part["arrival_second"] = aligned_second.astype(int)
        part["arrival_slot"] = np.floor(aligned_second / SLOT_SECONDS).astype(int) + 1
        part["scenario_id"] = part["trace_day"].map(scenario_map)
        part["task_id"] = "BG3-D" + part["trace_day"].astype(str) + "-L" + part["source_row"].astype(str)
        probs = probability_matrix(part["Total tokens"].to_numpy(), part["Model"].to_numpy(), part["Log Type"].to_numpy())
        for idx, name in enumerate(P_NAMES):
            part[name] = probs[:, idx]
        part["prefill_tokens"] = part["Request tokens"].astype(np.int64)
        part["decode_passes"] = np.maximum(part["Response tokens"].astype(np.int64) - 1, 0)
        part["response_zero_flag"] = (part["Response tokens"] <= 0).astype(int)
        pieces.append(
            part.rename(
                columns={
                    "Timestamp": "source_timestamp_s",
                    "Elapsed time": "elapsed_seconds",
                    "Model": "source_model",
                    "Request tokens": "input_tokens",
                    "Response tokens": "output_tokens",
                    "Total tokens": "total_tokens",
                    "Log Type": "log_type",
                    "Session ID": "session_id",
                }
            )[
                [
                    "scenario_id", "trace_day", "task_id", "source_row", "source_timestamp_s", "arrival_second",
                    "arrival_slot", "source_model", "log_type", "session_id", "elapsed_seconds", "input_tokens",
                    "output_tokens", "total_tokens", "prefill_tokens", "decode_passes", "response_zero_flag", *P_NAMES,
                ]
            ]
        )
    tasks = pd.concat(pieces, ignore_index=True)
    tasks = tasks.sort_values(["scenario_id", "arrival_slot", "arrival_second", "source_row"], kind="stable").reset_index(drop=True)
    return tasks


def allocate_background_cohorts(slot_tasks: pd.DataFrame, remaining_tokens: int, prefix: str) -> list[dict[str, object]]:
    if remaining_tokens <= 0:
        return []
    work = slot_tasks.copy()
    work["size_bin"] = pd.cut(work["total_tokens"], bins=SIZE_BINS, right=False, include_lowest=True).astype(str)
    grouped = (
        work.groupby(["size_bin", "source_model", "log_type"], observed=True)
        .agg(
            observed_count=("task_id", "size"),
            input_tokens=("input_tokens", "median"),
            output_tokens=("output_tokens", "median"),
        )
        .reset_index()
    )
    if grouped.empty:
        grouped = pd.DataFrame(
            [{"size_bin": "fallback", "source_model": "Unknown", "log_type": "Unknown", "observed_count": 1,
              "input_tokens": 1000, "output_tokens": 256}]
        )
    grouped["input_tokens"] = grouped["input_tokens"].round().clip(lower=0).astype(int)
    grouped["output_tokens"] = grouped["output_tokens"].round().clip(lower=0).astype(int)
    grouped["total_tokens"] = (grouped["input_tokens"] + grouped["output_tokens"]).clip(lower=1, upper=MAX_TASK_TOKENS)
    grouped["share"] = grouped["observed_count"] / grouped["observed_count"].sum()
    mean_tokens = float((grouped["share"] * grouped["total_tokens"]).sum())
    estimated_tasks = max(1, int(remaining_tokens // max(mean_tokens, 1.0)))
    exact = estimated_tasks * grouped["share"]
    grouped["cohort_count"] = np.floor(exact).astype(np.int64)
    leftover_count = estimated_tasks - int(grouped["cohort_count"].sum())
    if leftover_count > 0:
        order = np.argsort(-(exact - np.floor(exact)).to_numpy())[:leftover_count]
        grouped.loc[grouped.index[order], "cohort_count"] += 1
    assigned = int((grouped["cohort_count"] * grouped["total_tokens"]).sum())
    while assigned > remaining_tokens:
        candidates = grouped.index[grouped["cohort_count"] > 0]
        idx = max(candidates, key=lambda i: int(grouped.at[i, "total_tokens"]))
        grouped.at[idx, "cohort_count"] -= 1
        assigned -= int(grouped.at[idx, "total_tokens"])

    rows: list[dict[str, object]] = []
    for idx, row in grouped.loc[grouped["cohort_count"] > 0].iterrows():
        inp = int(row["input_tokens"])
        out = int(row["output_tokens"])
        total = max(1, inp + out)
        probs = probability_matrix(np.array([total]), np.array([row["source_model"]]), np.array([row["log_type"]]))[0]
        rows.append(
            {
                "cohort_id": f"{prefix}-B{len(rows)+1:02d}",
                "source_kind": "affine_intercept_background",
                "source_task_id": "",
                "source_model": row["source_model"],
                "log_type": row["log_type"],
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": total,
                "prefill_tokens": inp,
                "decode_passes": max(out - 1, 0),
                "cohort_count": int(row["cohort_count"]),
                **{name: float(probs[i]) for i, name in enumerate(P_NAMES)},
            }
        )
    residual = remaining_tokens - sum(int(row["cohort_count"]) * int(row["total_tokens"]) for row in rows)
    prompt_share = float(slot_tasks["input_tokens"].sum() / max(slot_tasks["total_tokens"].sum(), 1))
    correction_index = 1
    while residual > 0:
        token_size = int(min(residual, MAX_TASK_TOKENS))
        inp = int(round(token_size * prompt_share))
        out = token_size - inp
        probs = probability_matrix(np.array([token_size]), np.array(["Synthetic Qwen"]), np.array(["Background"]))[0]
        rows.append(
            {
                "cohort_id": f"{prefix}-C{correction_index:02d}",
                "source_kind": "rounding_correction_background",
                "source_task_id": "",
                "source_model": "Synthetic Qwen",
                "log_type": "Background",
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": token_size,
                "prefill_tokens": inp,
                "decode_passes": max(out - 1, 0),
                "cohort_count": 1,
                **{name: float(probs[i]) for i, name in enumerate(P_NAMES)},
            }
        )
        residual -= token_size
        correction_index += 1
    return rows


def build_scaled_cohorts(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    slot_rows: list[dict[str, object]] = []
    base_count = int(math.floor(SOURCE_COPY_MULTIPLIER))
    fractional = SOURCE_COPY_MULTIPLIER - base_count

    for scenario_id, scenario_tasks in tasks.groupby("scenario_id", sort=True):
        trace_day = int(scenario_tasks["trace_day"].iloc[0])
        for arrival_slot in range(1, SLOTS_PER_DAY + 1):
            group = scenario_tasks.loc[scenario_tasks["arrival_slot"] == arrival_slot]
            ids = group["source_row"].to_numpy(dtype=np.uint64)
            hashed = (ids * np.uint64(1103515245) + np.uint64(12345 + RANDOM_SEED + int(trace_day))) & np.uint64(0xFFFFFFFF)
            extra = (hashed.astype(np.float64) / float(2**32) < fractional).astype(np.int64)
            counts = base_count + extra
            source_tokens = int(np.dot(counts, group["total_tokens"].to_numpy(dtype=np.int64)))
            raw_slot_tokens = int(group["total_tokens"].sum())
            target_tokens = int(round(QWEN_SHARE * (AFFINE_SCALE * raw_slot_tokens + AFFINE_INTERCEPT_TPS) * SLOT_SECONDS))
            remaining = target_tokens - source_tokens
            if remaining < 0:
                raise ValueError(f"Scaled source component exceeds target in {scenario_id} slot {arrival_slot}")

            source = pd.DataFrame(
                {
                    "scenario_id": scenario_id,
                    "trace_day": int(trace_day),
                    "arrival_slot": int(arrival_slot),
                    "cohort_id": "C-" + group["task_id"].astype(str),
                    "source_kind": "scaled_observed_task",
                    "source_task_id": group["task_id"].astype(str).to_numpy(),
                    "source_model": group["source_model"].astype(str).to_numpy(),
                    "log_type": group["log_type"].astype(str).to_numpy(),
                    "input_tokens": group["input_tokens"].astype(np.int64).to_numpy(),
                    "output_tokens": group["output_tokens"].astype(np.int64).to_numpy(),
                    "total_tokens": group["total_tokens"].astype(np.int64).to_numpy(),
                    "prefill_tokens": group["prefill_tokens"].astype(np.int64).to_numpy(),
                    "decode_passes": group["decode_passes"].astype(np.int64).to_numpy(),
                    "cohort_count": counts,
                }
            )
            for name in P_NAMES:
                source[name] = group[name].to_numpy(dtype=float)
            if not source.empty:
                rows.append(source)

            background_records = allocate_background_cohorts(group, remaining, f"{scenario_id}-S{int(arrival_slot):02d}")
            if background_records:
                background = pd.DataFrame(background_records)
                background.insert(0, "arrival_slot", int(arrival_slot))
                background.insert(0, "trace_day", int(trace_day))
                background.insert(0, "scenario_id", scenario_id)
                rows.append(background)

            implied_count = int(counts.sum()) + sum(int(r["cohort_count"]) for r in background_records)
            type_counts = {}
            source_p = source[P_NAMES].to_numpy(dtype=float)
            for idx, name in enumerate(P_NAMES):
                type_counts[f"expected_{name[2:]}_tasks"] = float(np.dot(counts, source_p[:, idx])) + sum(
                    int(r["cohort_count"]) * float(r[name]) for r in background_records
                )
            slot_rows.append(
                {
                    "scenario_id": scenario_id,
                    "trace_day": int(trace_day),
                    "arrival_slot": int(arrival_slot),
                    "hour_start": (int(arrival_slot) - 1) / 4.0,
                    "raw_request_count": len(group),
                    "raw_request_tokens": int(group["input_tokens"].sum()),
                    "raw_response_tokens": int(group["output_tokens"].sum()),
                    "raw_total_tokens": raw_slot_tokens,
                    "scaled_observed_component_tokens": source_tokens,
                    "scaled_background_tokens": remaining,
                    "scaled_qwen_total_tokens": target_tokens,
                    "scaled_qwen_token_rate_per_s": target_tokens / SLOT_SECONDS,
                    "implied_full_scale_task_count": implied_count,
                    **type_counts,
                }
            )
    cohorts = pd.concat(rows, ignore_index=True)
    slots = pd.DataFrame(slot_rows).sort_values(["scenario_id", "arrival_slot"]).reset_index(drop=True)
    return cohorts, slots


def add_raw_slot_probabilities(tasks: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    expected = tasks.groupby(["scenario_id", "arrival_slot"], observed=True)[P_NAMES].sum().reset_index()
    expected = expected.rename(columns={name: f"raw_expected_{name[2:]}_tasks" for name in P_NAMES})
    return slots.merge(expected, on=["scenario_id", "arrival_slot"], how="left")


def plot_overview(daily: pd.DataFrame, reps: pd.DataFrame, slots: pd.DataFrame, tasks: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])

    ax = fig.add_subplot(grid[0, 0])
    ax.scatter(daily["request_count"] / 1000, daily["total_tokens"] / 1e6, s=22, alpha=0.45, color="#64748b", label="All 110 days")
    rep_daily = daily[daily["representative"]]
    ax.scatter(rep_daily["request_count"] / 1000, rep_daily["total_tokens"] / 1e6, s=52, color="#d97706", label="Selected days")
    for _, row in rep_daily.iterrows():
        ax.annotate(str(int(row["trace_day"])), (row["request_count"] / 1000, row["total_tokens"] / 1e6), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_title("Representative days cover workload volume and request intensity")
    ax.set_xlabel("Requests per day (thousand)")
    ax.set_ylabel("Raw total tokens per day (million)")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[0, 1])
    scenario_order = reps["scenario_id"].tolist()
    matrix = slots.pivot(index="scenario_id", columns="arrival_slot", values="scaled_qwen_token_rate_per_s").reindex(scenario_order).to_numpy() / 1e6
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Affine-scaled Qwen workload across 15-minute slots")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Representative scenario")
    ax.set_xticks([0, 16, 32, 48, 64, 80, 95], ["0", "4", "8", "12", "16", "20", "24"])
    ax.set_yticks(range(len(scenario_order)), scenario_order)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Million tokens/s")

    ax = fig.add_subplot(grid[1, 0])
    type_means = tasks.groupby("scenario_id", observed=True)[P_NAMES].mean().reindex(scenario_order)
    bottom = np.zeros(len(type_means))
    colors = ["#475569", "#2563eb", "#0891b2", "#16a34a", "#d97706"]
    labels = ["Opt out", "Type 1", "Type 2", "Type 3", "Type 4"]
    for name, label, color in zip(P_NAMES, labels, colors):
        values = type_means[name].to_numpy()
        ax.bar(range(len(values)), values, bottom=bottom, color=color, label=label, width=0.78)
        bottom += values
    ax.set_title("Assumed task-type probability mix")
    ax.set_xlabel("Representative scenario")
    ax.set_ylabel("Mean probability")
    ax.set_xticks(range(len(scenario_order)), scenario_order, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper center")

    ax = fig.add_subplot(grid[1, 1])
    positive = tasks.loc[tasks["total_tokens"] > 0, "total_tokens"].to_numpy(dtype=float)
    bins = np.logspace(0, math.ceil(math.log10(max(positive.max(), 10))), 45)
    ax.hist(positive, bins=bins, color="#2563eb", alpha=0.8)
    ax.set_xscale("log")
    ax.set_title("Observed task-size distribution retained in solver inputs")
    ax.set_xlabel("Total tokens per task (log scale)")
    ax.set_ylabel("Observed tasks")

    fig.suptitle("BurstGPT v2.0 task-level dynamic inputs", fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily, all_slots = scan_trace(args.input, args.chunk_size)
    selected_days, reasons = select_representative_days(daily, args.representative_days)
    scenario_map = {day: f"R{idx:02d}-D{day}" for idx, day in enumerate(selected_days, start=1)}
    daily["representative"] = daily["trace_day"].isin(selected_days)
    daily["scenario_id"] = daily["trace_day"].map(scenario_map).fillna("")
    daily["selection_reason"] = daily["trace_day"].map(lambda d: "; ".join(reasons.get(int(d), [])))
    daily = daily.sort_values("trace_day").reset_index(drop=True)

    tasks = extract_representative_tasks(args.input, selected_days, reasons, args.chunk_size)
    cohorts, scaled_slots = build_scaled_cohorts(tasks)
    scaled_slots = add_raw_slot_probabilities(tasks, scaled_slots)

    representative_days = daily[daily["representative"]].copy()
    representative_days["selection_order"] = representative_days["trace_day"].map({d: i for i, d in enumerate(selected_days, 1)})
    representative_days = representative_days.sort_values("selection_order")

    weighted_type_totals = np.array([
        np.dot(cohorts["cohort_count"].to_numpy(dtype=float), cohorts[name].to_numpy(dtype=float)) for name in P_NAMES
    ])
    weighted_type_share = weighted_type_totals / weighted_type_totals.sum()
    raw_type_share = tasks[P_NAMES].mean().to_numpy(dtype=float)

    daily.to_csv(args.output_dir / "daily_summary_all_110_days.csv", index=False)
    representative_days.to_csv(args.output_dir / "representative_days.csv", index=False)
    scaled_slots.to_csv(args.output_dir / "representative_slot_inputs.csv", index=False)
    tasks.to_csv(args.output_dir / "representative_tasks_raw.csv.gz", index=False, compression="gzip")

    assumptions = pd.DataFrame(
        [
            ["slot_seconds", SLOT_SECONDS, "seconds", "15-minute contract slots"],
            ["time_shift_seconds", TIME_SHIFT_SECONDS, "seconds", "Matches np.roll(raw, 95) in the replication fit"],
            ["affine_scale", AFFINE_SCALE, "token/s per raw token/slot", "Replication affine slope"],
            ["affine_intercept", AFFINE_INTERCEPT_TPS, "tokens/s", "Replication affine intercept; inferred, not author disclosed"],
            ["qwen_share", QWEN_SHARE, "fraction", "Single-model share inferred in the replication"],
            ["source_copy_multiplier", SOURCE_COPY_MULTIPLIER, "copies per observed task", "affine_scale * slot_seconds * qwen_share"],
            ["baseline_tokens_per_slot", BASELINE_TOKENS_PER_SLOT, "tokens/slot", "affine_intercept * slot_seconds * qwen_share"],
            ["max_task_tokens", MAX_TASK_TOKENS, "tokens", "1000 tokens/s/replica * 900 seconds"],
        ], columns=["parameter", "value", "unit", "interpretation"]
    )
    assumptions.to_csv(args.output_dir / "scaling_assumptions.csv", index=False)

    type_table = pd.DataFrame(
        {
            "selection": ["opt_out", "type_1", "type_2", "type_3", "type_4"],
            "ordering": [0, 1, 2, 3, 4],
            "interpretation": [
                "Immediate service, no discount",
                "Participates; least delay-sensitive",
                "Participates; low-to-medium delay sensitivity",
                "Participates; medium-to-high delay sensitivity",
                "Participates; most delay-sensitive",
            ],
            "base_probability": BASE_PROBABILITIES,
            "realized_raw_mean_probability": raw_type_share,
            "realized_full_scale_task_share": weighted_type_share,
        }
    )
    type_table.to_csv(args.output_dir / "type_probability_assumptions.csv", index=False)

    probability_error = np.abs(tasks[P_NAMES].sum(axis=1) - 1.0)
    cohort_reconciliation = cohorts.assign(weighted_tokens=cohorts["cohort_count"] * cohorts["total_tokens"]).groupby(
        ["scenario_id", "arrival_slot"], observed=True
    )["weighted_tokens"].sum().reset_index()
    compare = scaled_slots.merge(cohort_reconciliation, on=["scenario_id", "arrival_slot"], how="left")
    reconcile_error = compare["weighted_tokens"] - compare["scaled_qwen_total_tokens"]
    report = {
        "source": {
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
            "sha256": sha256(args.input),
            "release_url": "https://github.com/HPMLL/BurstGPT/releases/tag/v2.0",
        },
        "coverage": {
            "trace_days": int(len(daily)),
            "trace_day_min": int(daily["trace_day"].min()),
            "trace_day_max": int(daily["trace_day"].max()),
            "representative_days": selected_days,
            "representative_raw_tasks": int(len(tasks)),
            "scaled_cohort_rows": int(len(cohorts)),
            "implied_full_scale_tasks": int(cohorts["cohort_count"].sum()),
        },
        "checks": {
            "probability_sum_max_abs_error": float(probability_error.max()),
            "probability_min": float(tasks[P_NAMES].min().min()),
            "probability_max": float(tasks[P_NAMES].max().max()),
            "arrival_slot_min": int(tasks["arrival_slot"].min()),
            "arrival_slot_max": int(tasks["arrival_slot"].max()),
            "task_id_unique": bool(tasks["task_id"].is_unique),
            "token_accounting_rows_mismatch": int((tasks["input_tokens"] + tasks["output_tokens"] != tasks["total_tokens"]).sum()),
            "max_observed_task_tokens": int(tasks["total_tokens"].max()),
            "one_slot_feasibility_violations": int((tasks["total_tokens"] > MAX_TASK_TOKENS).sum()),
            "scaled_slot_reconciliation_max_abs_tokens": int(np.abs(reconcile_error).max()),
            "scaled_slot_reconciliation_mismatch_count": int((reconcile_error != 0).sum()),
        },
        "assumptions": {
            "type_probability_model": "softmax around [0.15,0.30,0.25,0.18,0.12], conditioned on log task size, model, and log type",
            "background_mix": "The undisclosed affine-intercept workload uses the observed model/log-type/task-size mix within each slot.",
            "scaling_representation": "Integer cohort counts avoid materializing the full-scale request population.",
        },
    }
    with (args.output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    plot_overview(daily, representative_days, scaled_slots, tasks, args.output_dir / "task_inputs_overview.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
