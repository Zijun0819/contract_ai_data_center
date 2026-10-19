"""Solve the same-day fluid model with LP-dual Benders or the compact MILP."""
from __future__ import annotations

import logging
import time
from dataclasses import replace
import numpy as np

from llm_contract_optimizer.config import AppConfig, validate_config
from llm_contract_optimizer.data import load_energy_scenario, load_workload_aggregate
from llm_contract_optimizer.domain import BendersIteration, OptimizationResult
from llm_contract_optimizer.exceptions import NoFeasibleSolutionError, SolveError
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.master import BendersMaster
from llm_contract_optimizer.menu import audit_menu_domain, menu_from_deadlines, presolve_parameters

LOGGER = logging.getLogger(__name__)


def load_case(config):
    validate_config(config)
    config = replace(config, parameters=presolve_parameters(config.parameters))
    p, inputs = config.parameters, config.inputs
    energy = load_energy_scenario(inputs.root, inputs.energy_scenario_id, p.horizon_slots)
    workload = load_workload_aggregate(inputs.root, inputs.workload_scenario_id, p,
                                      inputs.workload_file, inputs.immediate_execution_share,
                                      inputs.type_distribution)
    return config, workload, energy


def validate_case(config: AppConfig) -> dict:
    config, workload, energy = load_case(config)
    inner = FluidInnerSolver(workload, energy, config.parameters, config.solver)
    total = workload.task_count
    return {
        "workload_scenario_id": workload.scenario_id, "energy_scenario_id": energy.scenario_id,
        "task_count": total, "source_record_count": workload.source_record_count,
        "total_tokens": workload.total_tokens, "type_counts": workload.type_counts.tolist(),
        "type_distribution": (workload.type_counts/total).tolist() if total else [],
        "horizon_slots": inner.S, "horizon_hours": inner.S*config.parameters.slot_hours,
        "group_count": int(np.count_nonzero(inner.mass)), "service_variables": inner.nv,
        "continuous_variables": inner.nx, "menu_binaries": inner.I*inner.K,
        "assignment_binaries": 0, "horizon_policy": config.solver.horizon_policy,
        "necessary_capacity_screen": inner.screen_infeasibility() or "passed",
        "contract_domain": audit_menu_domain(config.parameters),
        "algorithm": config.solver.algorithm,
    }


def _discount_cost(menu, workload):
    return float(np.asarray(menu.discounts) @ workload.participating_bill_usd_by_type)


def _fixed_energy_cost(p, energy):
    return float(p.ppa_cost_usd_per_kwh * energy.ppa_kwh.sum()
                 + p.gas_turbine_cost_usd_per_kwh * energy.gas_turbine_kwh.sum()
                 + p.pv_cost_usd_per_kwh * energy.pv_kwh.sum()
                 + p.bess_cost_usd_per_kwh_discharged * energy.bess_discharge_kwh.sum())


def _limit(settings, started, per_call=None):
    remaining = (None if settings.total_time_limit_seconds is None else
                 settings.total_time_limit_seconds - (time.perf_counter()-started))
    if remaining is not None and remaining <= 0:
        return 0.
    if per_call is None:
        return remaining
    return per_call if remaining is None else min(per_call, remaining)


def _within_gap(settings, lower, upper):
    return upper-lower <= settings.absolute_gap_usd + settings.relative_gap*max(1., abs(upper))


def solve_case(config: AppConfig) -> OptimizationResult:
    config, workload, energy = load_case(config)
    return solve_aggregate(config, workload, energy)


def solve_aggregate(config, workload, energy) -> OptimizationResult:
    """Public in-memory entry point, also used for independently generated validation cases."""
    validate_config(config)
    config = replace(config, parameters=presolve_parameters(config.parameters))
    p, settings = config.parameters, config.solver
    started = time.perf_counter()
    inner = FluidInnerSolver(workload, energy, p, settings)
    reason = inner.screen_infeasibility()
    if reason:
        raise NoFeasibleSolutionError(reason)
    if not audit_menu_domain(p)["contract_domain_feasible"]:
        raise NoFeasibleSolutionError("Even the shortest menu violates the IR/IC discount cap.")
    master = BendersMaster(inner)
    shortest = menu_from_deadlines((p.candidate_deferrals_slots[0],)*inner.I, p)
    lower = _discount_cost(shortest, workload) + inner.q0
    incumbent, incumbent_inner, upper = None, None, float("inf")
    history, cuts, infeasible_menus = [], [], []
    reason = "iteration_limit"
    LOGGER.info("Loaded %s fixed-label tasks (%s source records), %s tokens; "
                "%d menu binaries and %d continuous scheduling variables.",
                f"{workload.task_count:,}", f"{workload.source_record_count:,}",
                f"{workload.total_tokens:,.0f}", master.z_count, inner.nx)

    if settings.algorithm == "direct":
        tick = time.perf_counter()
        limit = _limit(settings, started)
        if limit == 0:
            raise SolveError("Time limit reached before the compact MILP could start.")
        result = master.solve_direct(limit, settings.master_mip_relative_gap)
        if result.status == 2:
            raise NoFeasibleSolutionError("The compact fluid model has no feasible contract menu.")
        if result.x is None:
            raise SolveError(f"No feasible menu was returned; this is not proof of infeasibility. {result.message}")
        incumbent = master.decode_menu(result.x[:master.z_count])
        incumbent_inner = inner.decode_solution(result.x[master.z_count:], incumbent.deadlines_slots,
                                               "optimal" if result.status == 0 else "feasible",
                                               str(result.message))
        upper = _discount_cost(incumbent, workload) + incumbent_inner.objective_day_ahead_cost_usd
        bound = getattr(result, "mip_dual_bound", None)
        if bound is not None and np.isfinite(bound):
            lower = max(lower, float(bound))
        reason = "gap_reached" if _within_gap(settings, lower, upper) else "direct_solver_limit"
        history.append(BendersIteration(1, incumbent.deadlines_slots, str(result.status),
                                       incumbent_inner.status, lower, upper,
                                       incumbent_inner.objective_day_ahead_cost_usd, 0,
                                       time.perf_counter()-tick, 0.))
    else:
        visited = set()
        for iteration in range(1, settings.max_iterations+1):
            limit = _limit(settings, started, settings.master_time_limit_seconds)
            if limit == 0:
                reason = "total_time_limit"; break
            tick = time.perf_counter()
            ms = master.solve(cuts, infeasible_menus, limit, settings.master_mip_relative_gap)
            master_seconds = time.perf_counter()-tick
            if ms.lower_bound is not None:
                lower = max(lower, ms.lower_bound)
            if ms.status == "infeasible":
                if incumbent is not None:
                    raise SolveError("Master infeasibility contradicts a saved feasible menu; check numerical certificates.")
                raise NoFeasibleSolutionError("No menu satisfies the contract and same-day capacity requirements.")
            if incumbent is not None and _within_gap(settings, lower, upper):
                history.append(BendersIteration(iteration, incumbent.deadlines_slots, ms.status,
                                               "bound_converged", lower, upper, None, len(cuts),
                                               master_seconds, 0.))
                reason = "gap_reached"; break
            if ms.menu is None:
                reason = "master_limit_without_candidate" if ms.status == "limit" else "master_error"
                LOGGER.warning("%s", ms.message)
                break
            menu = ms.menu
            limit = _limit(settings, started, settings.subproblem_time_limit_seconds)
            if limit == 0:
                reason = "total_time_limit"; break
            tick = time.perf_counter()
            solution = inner.solve(menu.deadlines_slots, limit)
            sub_seconds = time.perf_counter()-tick
            q = None
            if solution.scheduled_tokens_by_type_slot is not None:
                q = solution.objective_day_ahead_cost_usd
                loss = _discount_cost(menu, workload) + q
                if loss < upper:
                    incumbent, incumbent_inner, upper = menu, solution, loss
                if solution.cut is not None:
                    if menu.deadlines_slots in visited:
                        reason = "master_limit" if ms.status == "limit" else "numerical_stall"
                    else:
                        cuts.append(solution.cut)
                        visited.add(menu.deadlines_slots)
                else:
                    reason = "subproblem_limit"
            elif solution.status == "infeasible":
                if menu.deadlines_slots in infeasible_menus:
                    raise SolveError("Master repeated a previously certified infeasible menu.")
                infeasible_menus.append(menu.deadlines_slots)
            else:
                reason = "subproblem_limit"
            history.append(BendersIteration(iteration, menu.deadlines_slots, ms.status,
                                           solution.status, lower, upper if incumbent else None,
                                           q, len(cuts), master_seconds, sub_seconds))
            LOGGER.info("Iteration %d | deferrals=%s | loss lower=%.6f upper=%s | cuts=%d",
                        iteration, menu.deadlines_slots, lower,
                        f"{upper:.6f}" if incumbent else "none", len(cuts))
            if incumbent is not None and _within_gap(settings, lower, upper):
                reason = "gap_reached"; break
            if reason in ("subproblem_limit", "master_limit", "numerical_stall"):
                break

    if incumbent is None or incumbent_inner is None:
        raise SolveError(f"Stopped ({reason}) without a feasible menu; infeasibility has not been established.")
    roundoff = max(1e-7, 1e-9*abs(upper))
    if lower > upper + roundoff:
        raise SolveError(f"Lower bound {lower} exceeds incumbent loss {upper}.")
    lower = min(lower, upper)
    gap = max(0., upper-lower)
    converged = _within_gap(settings, lower, upper)
    # Validate constructive recovery before marking the aggregate plan ready for output.
    inner.recover_groups(incumbent_inner, incumbent.deadlines_slots)
    fixed_cost = _fixed_energy_cost(p, energy)
    revenue = workload.total_bill_usd - _discount_cost(incumbent, workload)
    constant = workload.total_bill_usd-fixed_cost
    return OptimizationResult(
        workload.scenario_id, energy.scenario_id, energy.date, settings.algorithm,
        incumbent, incumbent_inner, revenue, fixed_cost, incumbent_inner.objective_day_ahead_cost_usd,
        constant-upper, lower, upper, constant-upper, constant-lower, gap, gap/max(1., abs(upper)),
        converged, "gap_reached" if converged else reason, time.perf_counter()-started,
        workload, energy, tuple(history), len(cuts), master.interval_count+len(infeasible_menus))
