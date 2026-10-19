"""Fixed-policy profit benchmarks for the contract-menu experiments."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from llm_contract_optimizer.config import AppConfig
from llm_contract_optimizer.domain import BenchmarkResult, EnergyScenario, WorkloadAggregate
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.menu import delay_penalty
from llm_contract_optimizer.solve import load_case


FLAT_DISCOUNT = 0.50
FLAT_DEFERRAL_HOURS = 24.0


def _fixed_energy_cost(config: AppConfig, energy: EnergyScenario) -> float:
    p = config.parameters
    return float(
        p.ppa_cost_usd_per_kwh * energy.ppa_kwh.sum()
        + p.gas_turbine_cost_usd_per_kwh * energy.gas_turbine_kwh.sum()
        + p.pv_cost_usd_per_kwh * energy.pv_kwh.sum()
        + p.bess_cost_usd_per_kwh_discharged * energy.bess_discharge_kwh.sum()
    )


def _energy_accounting(
    executed_tokens: np.ndarray, config: AppConfig, energy: EnergyScenario
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    p = config.parameters
    facility = p.idle_facility_energy_kwh_per_slot + p.facility_kwh_per_token * executed_tokens
    net = facility + energy.bess_charge_kwh - energy.fixed_supply_kwh
    imports = np.maximum(net, 0.0)
    spill = np.maximum(-net, 0.0)
    return facility, imports, spill, float(imports @ energy.day_ahead_price_usd_per_kwh)


def _pro_rata_slot_admission(
    arrivals: np.ndarray, capacity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Admit at most one slot of capacity and drop overflow pro rata."""
    admitted = np.minimum(arrivals, capacity)
    served_fraction = np.divide(
        admitted, arrivals, out=np.ones_like(admitted), where=arrivals > 0
    )
    return admitted, arrivals - admitted, served_fraction


def _bill_components_by_arrival(
    workload: WorkloadAggregate,
) -> tuple[np.ndarray, np.ndarray]:
    """Return opt-out and per-type bills by arrival, using token-proportional fallbacks."""
    if workload.opt_out_bill_usd_by_arrival is not None:
        opt_out = np.asarray(workload.opt_out_bill_usd_by_arrival, dtype=float)
    else:
        opt_out = np.zeros_like(workload.opt_out_tokens_by_arrival, dtype=float)
        total = float(workload.opt_out_tokens_by_arrival.sum())
        if total > 0:
            opt_out = (
                workload.opt_out_bill_usd * workload.opt_out_tokens_by_arrival / total
            )

    if workload.participating_bill_usd_by_type_arrival is not None:
        participating = np.asarray(
            workload.participating_bill_usd_by_type_arrival, dtype=float
        )
    else:
        participating = np.zeros_like(
            workload.participating_tokens_by_type_arrival, dtype=float
        )
        for i, (tokens, bill) in enumerate(
            zip(
                workload.participating_tokens_by_type_arrival,
                workload.participating_bill_usd_by_type,
            )
        ):
            total = float(tokens.sum())
            if total > 0:
                participating[i] = float(bill) * tokens / total
    return opt_out, participating


def _infeasible(
    benchmark_id: str,
    description: str,
    message: str,
    discount_cost_usd: float,
    fixed_energy_cost_usd: float,
    arrivals: np.ndarray,
    metadata: dict,
    revenue_usd: float | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        description=description,
        feasible=False,
        message=message,
        revenue_usd=revenue_usd,
        discount_cost_usd=discount_cost_usd,
        fixed_energy_cost_usd=fixed_energy_cost_usd,
        day_ahead_cost_usd=None,
        profit_usd=None,
        arrival_tokens_by_slot=arrivals,
        executed_tokens_by_slot=None,
        facility_energy_kwh=None,
        day_ahead_energy_kwh=None,
        spill_energy_kwh=None,
        shifted_tokens=None,
        mean_delay_hours=None,
        maximum_delay_hours=None,
        metadata=metadata,
    )


def voluntary_flat_50_24(
    config: AppConfig, workload: WorkloadAggregate, energy: EnergyScenario
) -> BenchmarkResult:
    """Offer a flat 50%/24-hour contract and drop arrival-slot overflow pro rata."""
    p = config.parameters
    benchmark_id = "voluntary_flat_50_24"
    description = (
        "Voluntary flat 50% discount with a 24-hour nominal deferral; "
        "arrival-slot overflow dropped"
    )
    fixed_cost = _fixed_energy_cost(config, energy)
    arrivals = workload.opt_out_tokens_by_arrival + workload.participating_tokens_by_type_arrival.sum(axis=0)

    exact_slots = FLAT_DEFERRAL_HOURS / p.slot_hours
    deferral_slots = int(round(exact_slots))
    if not np.isclose(exact_slots, deferral_slots, rtol=0.0, atol=1e-12):
        return _infeasible(
            benchmark_id,
            description,
            "The slot duration cannot represent an exact 24-hour deferral.",
            0.0,
            fixed_cost,
            arrivals,
            {"requested_deferral_hours": FLAT_DEFERRAL_HOURS},
        )

    penalty = delay_penalty(deferral_slots, p)
    margins = FLAT_DISCOUNT - np.asarray(p.type_lambdas) * penalty
    tolerance = 1e-12
    accepted = margins >= -tolerance

    _, dropped, served_fraction = _pro_rata_slot_admission(
        arrivals, p.token_capacity_per_slot
    )
    admitted_opt_out = workload.opt_out_tokens_by_arrival * served_fraction
    admitted_by_type = (
        workload.participating_tokens_by_type_arrival * served_fraction[None, :]
    )
    rejected_mass = (
        admitted_by_type[~accepted].sum(axis=0)
        if np.any(~accepted)
        else np.zeros(p.horizon_slots)
    )
    participating_mass = admitted_by_type.copy()
    participating_mass[~accepted] = 0.0
    immediate = admitted_opt_out + rejected_mass

    opt_out_bills_by_arrival, participating_bills_by_arrival = (
        _bill_components_by_arrival(workload)
    )
    admitted_opt_out_bills_by_arrival = opt_out_bills_by_arrival * served_fraction
    admitted_participating_bills_by_arrival = (
        participating_bills_by_arrival * served_fraction[None, :]
    )
    accepted_bills_by_arrival = (
        admitted_participating_bills_by_arrival * accepted[:, None]
    )
    rejected_bill_by_arrival = admitted_participating_bills_by_arrival[~accepted].sum(
        axis=0
    )
    accepted_bills = accepted_bills_by_arrival.sum(axis=1)
    rejected_bill = float(rejected_bill_by_arrival.sum())

    scheduled_workload = WorkloadAggregate(
        scenario_id=workload.scenario_id,
        opt_out_tokens_by_arrival=immediate,
        participating_tokens_by_type_arrival=participating_mass,
        opt_out_bill_usd=float(admitted_opt_out_bills_by_arrival.sum()) + rejected_bill,
        participating_bill_usd_by_type=accepted_bills,
        type_counts=workload.type_counts,
        source_record_count=workload.source_record_count,
        input_provenance=workload.input_provenance,
        opt_out_bill_usd_by_arrival=(
            admitted_opt_out_bills_by_arrival + rejected_bill_by_arrival
        ),
        participating_bill_usd_by_type_arrival=accepted_bills_by_arrival,
    )
    # The benchmark deadline is fixed independently of the proposed menu domain.
    benchmark_parameters = replace(p, candidate_deferrals_slots=(deferral_slots,))
    inner = FluidInnerSolver(scheduled_workload, energy, benchmark_parameters, config.solver)
    type_outcomes = [
        {
            "type": i + 1,
            "delay_sensitivity": float(p.type_lambdas[i]),
            "normalized_ir_margin": float(margins[i]),
            "accepts_flat_contract": bool(accepted[i]),
            "task_count": int(workload.type_counts[i + 1]),
            "tokens": float(workload.participating_tokens_by_type_arrival[i].sum()),
            "admitted_tokens": float(admitted_by_type[i].sum()),
            "dropped_tokens": float(
                workload.participating_tokens_by_type_arrival[i].sum()
                - admitted_by_type[i].sum()
            ),
            "undiscounted_bill_usd": float(workload.participating_bill_usd_by_type[i]),
            "admitted_undiscounted_bill_usd": float(
                admitted_participating_bills_by_arrival[i].sum()
            ),
        }
        for i in range(len(p.type_lambdas))
    ]
    metadata = {
        "discount_fraction": FLAT_DISCOUNT,
        "nominal_deferral_slots": deferral_slots,
        "nominal_deferral_hours": FLAT_DEFERRAL_HOURS,
        "delay_penalty": float(penalty),
        "ir_tolerance": tolerance,
        "accepted_types": [i + 1 for i, value in enumerate(accepted) if value],
        "rejected_types_routed_to_immediate": [i + 1 for i, value in enumerate(accepted) if not value],
        "accepted_task_count": int(workload.type_counts[1:][accepted].sum()),
        "rejected_task_count": int(workload.type_counts[1:][~accepted].sum()),
        "type_outcomes": type_outcomes,
        "admission_rule": "arrival-slot admission up to capacity; excess is dropped pro rata",
        "dropped_tokens": float(dropped.sum()),
        "drop_fraction": float(dropped.sum() / arrivals.sum()) if arrivals.sum() else 0.0,
        "dropped_tokens_by_slot": dropped.tolist(),
        "served_fraction_by_slot": served_fraction.tolist(),
        "horizon_policy": config.solver.horizon_policy,
    }
    discount_cost = float(FLAT_DISCOUNT * accepted_bills.sum())
    revenue = scheduled_workload.total_bill_usd - discount_cost
    screen = inner.screen_infeasibility()
    if screen:
        return _infeasible(
            benchmark_id, description, screen, discount_cost, fixed_cost, arrivals, metadata,
            revenue_usd=float(revenue),
        )
    solution = inner.solve((deferral_slots,) * len(p.type_lambdas), config.solver.subproblem_time_limit_seconds)
    if solution.scheduled_tokens_by_type_slot is None:
        return _infeasible(
            benchmark_id,
            description,
            solution.message,
            discount_cost,
            fixed_cost,
            arrivals,
            metadata,
            revenue_usd=float(revenue),
        )
    metadata["scheduler_status"] = solution.status
    metadata["scheduler_message"] = solution.message

    groups = inner.recover_groups(solution, (deferral_slots,) * len(p.type_lambdas))
    slot_indices = np.arange(p.horizon_slots)
    delays = slot_indices[None, :] - slot_indices[:, None]
    delay_mass = groups * delays[None, :, :]
    accepted_tokens = float(groups.sum())
    shifted_tokens = float(groups[:, delays > 0].sum())
    mean_delay_hours = (
        float(delay_mass.sum() / accepted_tokens * p.slot_hours) if accepted_tokens else 0.0
    )
    positive = np.argwhere(groups > max(1e-7, config.solver.feasibility_tolerance * config.solver.token_scale))
    maximum_delay_slots = int(np.max(positive[:, 2] - positive[:, 1])) if len(positive) else 0
    profit = revenue - fixed_cost - solution.objective_day_ahead_cost_usd
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        description=description,
        feasible=True,
        message=(
            f"{solution.status} fixed-policy schedule; "
            f"dropped {float(dropped.sum()):,.3f} excess tokens"
        ),
        revenue_usd=float(revenue),
        discount_cost_usd=discount_cost,
        fixed_energy_cost_usd=fixed_cost,
        day_ahead_cost_usd=solution.objective_day_ahead_cost_usd,
        profit_usd=float(profit),
        arrival_tokens_by_slot=arrivals,
        executed_tokens_by_slot=solution.total_tokens_by_slot,
        facility_energy_kwh=solution.facility_energy_kwh,
        day_ahead_energy_kwh=solution.day_ahead_energy_kwh,
        spill_energy_kwh=solution.spill_energy_kwh,
        shifted_tokens=shifted_tokens,
        mean_delay_hours=mean_delay_hours,
        maximum_delay_hours=maximum_delay_slots * p.slot_hours,
        metadata=metadata,
    )


def immediate_zero_discount_with_overflow(
    config: AppConfig, workload: WorkloadAggregate, energy: EnergyScenario,
) -> BenchmarkResult:
    """Charge full price for same-slot service and drop all capacity overflow."""
    p = config.parameters
    benchmark_id = "immediate_zero_discount_drop_overflow"
    description = "0% discount; same-slot service with excess workload dropped"
    arrivals = workload.opt_out_tokens_by_arrival + workload.participating_tokens_by_type_arrival.sum(axis=0)
    fixed_cost = _fixed_energy_cost(config, energy)
    executed, dropped, served_fraction = _pro_rata_slot_admission(
        arrivals, p.token_capacity_per_slot
    )
    revenue_by_arrival = workload.total_bill_usd_by_arrival * served_fraction
    revenue = float(revenue_by_arrival.sum())
    dropped_tokens = float(dropped.sum())
    total_tokens = float(arrivals.sum())
    metadata = {
        "discount_fraction": 0.0,
        "nominal_deferral_slots": 0,
        "scheduling_rule": "same-slot admission up to capacity; excess is dropped pro rata",
        "dropped_tokens": dropped_tokens,
        "drop_fraction": dropped_tokens / total_tokens if total_tokens else 0.0,
        "dropped_tokens_by_slot": dropped.tolist(),
        "served_fraction_by_slot": served_fraction.tolist(),
        "revenue_usd_by_arrival": revenue_by_arrival.tolist(),
        "horizon_policy": config.solver.horizon_policy,
    }
    facility, imports, spill, day_ahead_cost = _energy_accounting(executed, config, energy)
    profit = revenue - fixed_cost - day_ahead_cost
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        description=description,
        feasible=True,
        message=f"same-slot admission; dropped {dropped_tokens:,.3f} excess tokens",
        revenue_usd=float(revenue),
        discount_cost_usd=0.0,
        fixed_energy_cost_usd=fixed_cost,
        day_ahead_cost_usd=day_ahead_cost,
        profit_usd=float(profit),
        arrival_tokens_by_slot=arrivals,
        executed_tokens_by_slot=executed,
        facility_energy_kwh=facility,
        day_ahead_energy_kwh=imports,
        spill_energy_kwh=spill,
        shifted_tokens=0.0,
        mean_delay_hours=0.0,
        maximum_delay_hours=0.0,
        metadata=metadata,
    )


def run_benchmarks(
    config: AppConfig, workload: WorkloadAggregate, energy: EnergyScenario
) -> tuple[BenchmarkResult, BenchmarkResult]:
    return (
        voluntary_flat_50_24(config, workload, energy),
        immediate_zero_discount_with_overflow(config, workload, energy),
    )


def run_benchmark_case(config: AppConfig) -> tuple[BenchmarkResult, BenchmarkResult]:
    effective, workload, energy = load_case(config)
    return run_benchmarks(effective, workload, energy)
