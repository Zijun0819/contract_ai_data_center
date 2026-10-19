from __future__ import annotations

from llm_contract_optimizer.config import StaticParameters
from llm_contract_optimizer.domain import Menu


def delay_penalty(deferral_slots: int, parameters: StaticParameters) -> float:
    hours = deferral_slots * parameters.slot_hours
    scaled = parameters.delay_curvature_per_hour * hours
    return scaled / (1.0 + scaled)


def minimal_discounts(
    penalties: tuple[float, ...], lambdas: tuple[float, ...]
) -> tuple[float, ...]:
    discounts = [0.0] * len(lambdas)
    discounts[-1] = lambdas[-1] * penalties[-1]
    for index in range(len(lambdas) - 2, -1, -1):
        discounts[index] = (
            discounts[index + 1]
            + lambdas[index] * (penalties[index] - penalties[index + 1])
        )
    return tuple(discounts)


def all_pairs_ic_holds(menu: Menu, lambdas: tuple[float, ...], tolerance: float = 1e-10) -> bool:
    for true_type, sensitivity in enumerate(lambdas):
        truthful = menu.discounts[true_type] - sensitivity * menu.penalties[true_type]
        if truthful < -tolerance:
            return False
        for item in range(len(lambdas)):
            alternative = menu.discounts[item] - sensitivity * menu.penalties[item]
            if truthful + tolerance < alternative:
                return False
    return True


def menu_from_deadlines(
    deadlines_slots: tuple[int, ...], parameters: StaticParameters
) -> Menu:
    if len(deadlines_slots) != len(parameters.type_lambdas):
        raise ValueError("Deadline vector length must equal the number of types.")
    penalties = tuple(delay_penalty(value, parameters) for value in deadlines_slots)
    discounts = minimal_discounts(penalties, parameters.type_lambdas)
    menu = Menu(deadlines_slots, penalties, discounts)
    if not all_pairs_ic_holds(menu, parameters.type_lambdas):
        raise RuntimeError(f"Analytical discount construction failed IC for {deadlines_slots}.")
    return menu


def audit_menu_domain(parameters: StaticParameters) -> dict[str, object]:
    """Check IR/IC/cap geometry without loading tasks or solving a schedule.

    For increasing sensitivities and decreasing deadlines, the largest minimum
    discount is delta_1. Its coefficients in the penalty vector are lambda_1
    and successive sensitivity differences, all nonnegative. The cheapest
    nonpooled menu therefore raises only D_1 to the second candidate.
    """
    candidates = parameters.candidate_deferrals_slots
    count = len(parameters.type_lambdas)
    shortest = candidates[0]
    longest = candidates[-1]
    shortest_penalty = delay_penalty(shortest, parameters)
    longest_penalty = delay_penalty(longest, parameters)
    feasibility_cap = (parameters.maximum_discount / shortest_penalty
                       if shortest_penalty > 0 else None)
    full_domain_cap = (parameters.maximum_discount / longest_penalty
                       if longest_penalty > 0 else None)
    pooled = menu_from_deadlines((shortest,) * count, parameters)
    nonpooled = (menu_from_deadlines((candidates[1],) + (shortest,) * (count - 1), parameters)
                 if count > 1 and len(candidates) > 1 else None)
    feasible = pooled.discounts[0] <= parameters.maximum_discount
    differentiated = nonpooled is not None and nonpooled.discounts[0] <= parameters.maximum_discount
    return {
        "scope": "Contract IR/IC/discount-cap feasibility only; scheduling is not evaluated.",
        "candidate_deferrals_slots": list(candidates),
        "minimum_deferral_hours": shortest * parameters.slot_hours,
        "configured_maximum_sensitivity": parameters.type_lambdas[-1],
        "maximum_feasible_sensitivity_at_minimum_deferral": feasibility_cap,
        "maximum_sensitivity_supporting_every_candidate_deferral": full_domain_cap,
        "configured_fraction_of_feasibility_cap": (
            parameters.type_lambdas[-1] / feasibility_cap if feasibility_cap else None
        ),
        "minimum_pooled_discount": pooled.discounts[0],
        "discount_cap_slack_at_shortest_menu": parameters.maximum_discount - pooled.discounts[0],
        "contract_domain_feasible": feasible,
        "differentiated_menu_feasible": differentiated,
        "pooling_forced": feasible and not differentiated,
        "minimum_nonpooled_max_discount": nonpooled.discounts[0] if nonpooled else None,
        "nonpooled_witness_deadlines_slots": list(nonpooled.deadlines_slots) if differentiated else None,
        "nonpooled_witness_discounts": list(nonpooled.discounts) if differentiated else None,
        "master_binary_variables_after_presolve": count * len(presolve_parameters(parameters).candidate_deferrals_slots),
    }


def presolve_parameters(parameters: StaticParameters) -> StaticParameters:
    """At the exact IR/cap boundary, positive sensitivities force the shortest menu."""
    from dataclasses import replace
    shortest = min(parameters.candidate_deferrals_slots)
    # Equality is intentional: a tolerance must not remove a genuinely feasible menu.
    if (parameters.type_lambdas[-1]*delay_penalty(shortest, parameters) == parameters.maximum_discount
            and parameters.type_lambdas[0] > 0):
        return replace(parameters, candidate_deferrals_slots=(shortest,))
    return parameters
