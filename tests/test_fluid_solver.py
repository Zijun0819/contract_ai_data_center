"""Independent flow-LP and exhaustive-menu checks for the cumulative model."""
import csv
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from llm_contract_optimizer.benchmarks import (
    immediate_zero_discount_with_overflow,
    run_benchmarks,
    voluntary_flat_50_24,
)
from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.data import load_workload_aggregate, validate_type_counts
from llm_contract_optimizer.domain import EnergyScenario, WorkloadAggregate
from llm_contract_optimizer.exceptions import DataValidationError, NoFeasibleSolutionError
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.master import BendersMaster, MasterSolution
from llm_contract_optimizer.menu import menu_from_deadlines
from llm_contract_optimizer.reporting import write_method_comparison, write_result_bundle
from llm_contract_optimizer.scenario_comparison import _row as scenario_metric_row
from llm_contract_optimizer.solve import solve_aggregate

PROJECT = Path(__file__).resolve().parents[1]


def toy_case(mass=None, immediate=None, prices=None):
    config = load_config(PROJECT / "config/default.yaml")
    mass = np.array([[3, 3, 0, 2], [1, 0, 2, 1]] if mass is None else mass, dtype=float)
    I, S = mass.shape
    immediate = np.zeros(S) if immediate is None else np.asarray(immediate, dtype=float)
    prices = np.array([3, 1, 4, 2][:S] if prices is None else prices, dtype=float)
    p = replace(config.parameters, horizon_slots=S, slot_seconds=3600, installed_replicas=1,
                throughput_tokens_per_second_per_replica=5 / 3600,
                dynamic_energy_j_per_token=3.6e6, pue=1., idle_power_w_per_replica=0.,
                type_lambdas=tuple(.15 * (i + 1) for i in range(I)),
                candidate_deferrals_slots=(0, 1, 2), maximum_discount=.5)
    settings = replace(config.solver, token_scale=1., absolute_gap_usd=1e-7,
                       relative_gap=1e-9, total_time_limit_seconds=30.,
                       master_time_limit_seconds=10., subproblem_time_limit_seconds=10.)
    config = replace(config, parameters=p, solver=settings)
    w = WorkloadAggregate("toy", immediate, mass, float(immediate.sum()),
                          mass.sum(axis=1), np.array([1] * (I + 1)), 3)
    e = EnergyScenario("energy", "test", *[np.zeros(S) for _ in range(5)], prices)
    return config, w, e


def flow_reference(config, workload, energy, deadlines):
    """Original y[i,a,s] conservation/capacity LP, with no cumulative formulas."""
    p = config.parameters
    I, S = workload.participating_tokens_by_type_arrival.shape
    arcs = [(i, a, s) for i in range(I) for a in range(S)
            for s in range(a, min(a + deadlines[i] + 1, S))]
    n = len(arcs)
    eq = np.zeros((I * S, n + S))
    ub = np.zeros((2 * S, n + S))
    for j, (i, a, s) in enumerate(arcs):
        eq[i * S + a, j] = 1.
        ub[s, j] = 1.
        ub[S + s, j] = p.facility_kwh_per_token
    ub[S:, n:] = -np.eye(S)
    rhs = np.r_[
        p.token_capacity_per_slot - workload.opt_out_tokens_by_arrival,
        energy.fixed_supply_kwh - energy.bess_charge_kwh - p.idle_facility_energy_kwh_per_slot
        - p.facility_kwh_per_token * workload.opt_out_tokens_by_arrival,
    ]
    return linprog(np.r_[np.zeros(n), energy.day_ahead_price_usd_per_kwh],
                   A_ub=ub, b_ub=rhs, A_eq=eq,
                   b_eq=workload.participating_tokens_by_type_arrival.ravel(),
                   bounds=(0, None), method="highs")


def enumerate_menus(config, workload, energy):
    best = float("inf")
    for deadlines in itertools.product(config.parameters.candidate_deferrals_slots,
                                       repeat=len(config.parameters.type_lambdas)):
        if any(a < b for a, b in zip(deadlines, deadlines[1:])):
            continue
        menu = menu_from_deadlines(deadlines, config.parameters)
        if max(menu.discounts) > config.parameters.maximum_discount:
            continue
        lp = flow_reference(config, workload, energy, deadlines)
        if lp.success:
            best = min(best, lp.fun + np.dot(menu.discounts, workload.participating_bill_usd_by_type))
    return best


class FluidSchedulingTests(unittest.TestCase):
    def test_cumulative_flow_equivalence_recovery_and_interval_feasibility(self):
        rng = np.random.default_rng(92)
        feasible_count = infeasible_count = 0
        for case in range(12):
            mass = rng.integers(0, 5, (2, 4)).astype(float) / 2
            if case == 0:
                mass[:] = 0
            if case == 1:
                mass[:, -1] = 4
            config, w, e = toy_case(mass, rng.integers(0, 3, 4), rng.integers(0, 5, 4))
            e = replace(e, pv_kwh=rng.uniform(0, 2, 4), bess_charge_kwh=rng.uniform(0, .5, 4))
            inner = FluidInnerSolver(w, e, config.parameters, config.solver)
            intervals, capacity = inner.interval_inequalities()
            for deadlines in itertools.product((0, 1, 2), repeat=2):
                with self.subTest(case=case, deadlines=deadlines):
                    reference = flow_reference(config, w, e, deadlines)
                    result = inner.solve(deadlines)
                    self.assertEqual(result.status == "optimal", reference.success)
                    z = np.zeros((2, 3))
                    for i, d in enumerate(deadlines):
                        z[i, d] = 1
                    passes = np.all(intervals @ z.ravel() <= capacity + 1e-8)
                    self.assertEqual(bool(passes), reference.success)
                    if not reference.success:
                        infeasible_count += 1
                        continue
                    feasible_count += 1
                    self.assertAlmostEqual(result.objective_day_ahead_cost_usd, reference.fun, places=6)
                    y = inner.recover_groups(result, deadlines)
                    np.testing.assert_allclose(y.sum(axis=2), w.participating_tokens_by_type_arrival, atol=1e-7)
                    np.testing.assert_allclose(y.sum(axis=1), result.scheduled_tokens_by_type_slot, atol=1e-7)
                    for i, a, s in np.argwhere(y > 1e-7):
                        self.assertGreaterEqual(s, a)
                        self.assertLessEqual(s, min(a + deadlines[i], inner.S - 1))
        self.assertGreater(feasible_count, 0)
        self.assertGreater(infeasible_count, 0)

    def test_dual_cuts_are_tight_and_valid_across_other_menus(self):
        config, w, e = toy_case()
        inner = FluidInnerSolver(w, e, config.parameters, config.solver)
        menus = list(itertools.product((0, 1, 2), repeat=2))
        solutions = {d: inner.solve(d) for d in menus}
        for d, solution in solutions.items():
            if solution.cut is None:
                continue
            self.assertAlmostEqual(solution.cut.value(d, inner.candidates),
                                   solution.objective_day_ahead_cost_usd, places=7)
            for other, reference in solutions.items():
                if reference.status == "optimal":
                    self.assertLessEqual(solution.cut.value(other, inner.candidates),
                                         reference.objective_day_ahead_cost_usd + 1e-7)

    def test_benders_and_direct_match_exhaustive_flow_optimum(self):
        rng = np.random.default_rng(410)
        for case in range(8):
            config, w, e = toy_case(rng.integers(0, 4, (2, 4)), prices=rng.integers(0, 6, 4))
            optimum = enumerate_menus(config, w, e)
            for algorithm in ("direct", "benders"):
                for monotonic in (False, True) if algorithm == "benders" else (False,):
                    cfg = replace(config, solver=replace(config.solver, algorithm=algorithm,
                                                        add_monotonic_cuts=monotonic))
                    with self.subTest(case=case, algorithm=algorithm, monotonic=monotonic):
                        if not np.isfinite(optimum):
                            with self.assertRaises(NoFeasibleSolutionError):
                                solve_aggregate(cfg, w, e)
                        else:
                            result = solve_aggregate(cfg, w, e)
                            self.assertTrue(result.converged)
                            self.assertAlmostEqual(result.loss_upper_bound_usd, optimum, places=6)
                            self.assertLessEqual(result.loss_lower_bound_usd, optimum + 1e-7)

    def test_final_arrival_cannot_borrow_capacity_from_tomorrow(self):
        config, w, e = toy_case([[0, 0, 0, 6], [0, 0, 0, 0]])
        with self.assertRaisesRegex(NoFeasibleSolutionError, "Same-day completion"):
            solve_aggregate(config, w, e)
        config, w, e = toy_case([[0, 0, 0, 5], [0, 0, 0, 0]])
        result = FluidInnerSolver(w, e, config.parameters, config.solver).solve((2, 2))
        np.testing.assert_allclose(result.scheduled_tokens_by_type_slot[0], [0, 0, 0, 5])

    def test_immediate_overload_and_single_slot_horizon(self):
        config, w, e = toy_case([[0]], [6], [1])
        with self.assertRaisesRegex(NoFeasibleSolutionError, "Immediate"):
            solve_aggregate(config, w, e)
        config, w, e = toy_case([[2]], [1], [0])
        for algorithm in ("benders", "direct"):
            result = solve_aggregate(replace(config, solver=replace(config.solver, algorithm=algorithm)), w, e)
            self.assertTrue(result.converged)
            np.testing.assert_allclose(result.inner.total_tokens_by_slot, [3])

    def test_time_limited_master_uses_dual_bound_not_its_incumbent(self):
        config, w, e = toy_case()
        menu = menu_from_deadlines((2, 2), config.parameters)
        config = replace(config, solver=replace(config.solver, max_iterations=1))
        with patch.object(BendersMaster, "solve", return_value=MasterSolution("limit", menu, 0., "timeout")):
            result = solve_aggregate(config, w, e)
        self.assertFalse(result.converged)
        self.assertEqual(result.loss_lower_bound_usd, 0.)
        self.assertGreater(result.absolute_gap_usd, 0.)
        self.assertEqual(result.menu, menu)


class PolicyBenchmarkTests(unittest.TestCase):
    def test_voluntary_flat_contract_routes_negative_ir_type_to_immediate(self):
        config, w, e = toy_case([[2, 0, 0, 0], [3, 0, 0, 0]], immediate=[1, 0, 0, 0])
        parameters = replace(config.parameters, type_lambdas=(0.5, 0.8))
        config = replace(config, parameters=parameters)

        result = voluntary_flat_50_24(config, w, e)

        self.assertTrue(result.feasible, result.message)
        self.assertEqual(result.metadata["accepted_types"], [1])
        self.assertEqual(result.metadata["rejected_types_routed_to_immediate"], [2])
        self.assertAlmostEqual(result.metadata["dropped_tokens"], 1.0)
        self.assertAlmostEqual(result.discount_cost_usd, 0.5 * (2.0 * 5.0 / 6.0))
        self.assertAlmostEqual(result.executed_tokens_by_slot.sum(), 5.0)
        self.assertAlmostEqual(result.revenue_usd, 5.0 - result.discount_cost_usd)

    def test_voluntary_flat_contract_drops_overflow_and_is_always_schedulable(self):
        rng = np.random.default_rng(806)
        for case in range(20):
            mass = rng.uniform(0.0, 12.0, (2, 4))
            immediate = rng.uniform(0.0, 8.0, 4)
            config, w, e = toy_case(mass, immediate, rng.uniform(0.0, 5.0, 4))

            result = voluntary_flat_50_24(config, w, e)

            with self.subTest(case=case):
                self.assertTrue(result.feasible, result.message)
                arrivals = immediate + mass.sum(axis=0)
                admitted = np.minimum(arrivals, config.parameters.token_capacity_per_slot)
                np.testing.assert_allclose(
                    result.metadata["dropped_tokens_by_slot"], arrivals - admitted
                )
                self.assertAlmostEqual(result.executed_tokens_by_slot.sum(), admitted.sum())
                self.assertLessEqual(
                    float(result.executed_tokens_by_slot.max()),
                    config.parameters.token_capacity_per_slot + 1e-7,
                )

    def test_voluntary_flat_contract_excludes_dropped_work_from_revenue_and_discount(self):
        config, _, e = toy_case([[0, 0, 0, 0], [0, 0, 0, 0]])
        config = replace(
            config,
            parameters=replace(config.parameters, type_lambdas=(0.5, 0.8)),
        )
        w = WorkloadAggregate(
            "priced-overflow",
            np.array([2.0, 0.0, 0.0, 0.0]),
            np.array([[2.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]]),
            10.0,
            np.array([20.0, 60.0]),
            np.array([1, 1, 1]),
            3,
            opt_out_bill_usd_by_arrival=np.array([10.0, 0.0, 0.0, 0.0]),
            participating_bill_usd_by_type_arrival=np.array(
                [[20.0, 0.0, 0.0, 0.0], [60.0, 0.0, 0.0, 0.0]]
            ),
        )

        result = voluntary_flat_50_24(config, w, e)

        self.assertTrue(result.feasible, result.message)
        self.assertAlmostEqual(result.metadata["dropped_tokens"], 5.0)
        self.assertAlmostEqual(result.discount_cost_usd, 5.0)
        self.assertAlmostEqual(result.revenue_usd, 40.0)

        zero = immediate_zero_discount_with_overflow(config, w, e)
        row = scenario_metric_row(w.scenario_id, e.scenario_id, None, (result, zero),
                                  horizon_hours=4.)
        # Half the arriving work is dropped. Its USD 45 bill is not a discount;
        # only the USD 5 reduction on the accepted, served work is counted.
        self.assertAlmostEqual(row["flat_50_24_discount_expenditure_usd"], 5.)
        self.assertAlmostEqual(row["flat_50_24_day_ahead_electricity_cost_usd"], 13.)
        self.assertAlmostEqual(row["flat_50_24_profit_usd"], 27.)
        self.assertAlmostEqual(row["flat_50_24_discount_expenditure_usd_per_hour"], 1.25)
        self.assertAlmostEqual(row["flat_50_24_day_ahead_electricity_cost_usd_per_hour"], 3.25)
        self.assertAlmostEqual(row["flat_50_24_profit_usd_per_hour"], 6.75)
        self.assertEqual(row["zero_discount_discount_expenditure_usd_per_hour"], 0.)

    def test_sweep_metrics_normalize_actual_half_hour_horizon_and_negative_profit(self):
        config, w, e = toy_case([[1], [0]], prices=[2])
        config = replace(config, parameters=replace(config.parameters, slot_seconds=1800))
        proposed = solve_aggregate(config, w, e)
        row = scenario_metric_row(w.scenario_id, e.scenario_id, proposed, run_benchmarks(config, w, e),
                                  horizon_hours=config.parameters.horizon_slots * config.parameters.slot_hours)
        self.assertEqual(row["horizon_hours"], .5)
        self.assertTrue(row["proposed_converged"])
        self.assertAlmostEqual(row["proposed_profit_usd"], -1.)
        self.assertAlmostEqual(row["proposed_profit_usd_per_hour"], -2.)
        self.assertAlmostEqual(row["proposed_day_ahead_electricity_cost_usd_per_hour"], 4.)
        self.assertEqual(row["proposed_discount_expenditure_usd_per_hour"], 0.)

    def test_sweep_missing_or_infeasible_metrics_stay_missing_but_valid_zero_survives(self):
        config, w, e = toy_case()
        flat, zero = run_benchmarks(config, w, e)
        failed = replace(flat, feasible=False, discount_cost_usd=123., profit_usd=456.)
        row = scenario_metric_row(w.scenario_id, e.scenario_id, None, (failed, zero),
                                  "proposed failed", horizon_hours=4.)
        self.assertEqual(row["error"], "proposed failed")
        for prefix in ("proposed", "flat_50_24"):
            for metric in ("profit", "discount_expenditure", "day_ahead_electricity_cost"):
                self.assertIsNone(row[f"{prefix}_{metric}_usd"])
                self.assertIsNone(row[f"{prefix}_{metric}_usd_per_hour"])
        self.assertEqual(row["zero_discount_discount_expenditure_usd"], 0.)
        self.assertEqual(row["zero_discount_discount_expenditure_usd_per_hour"], 0.)

    def test_zero_discount_benchmark_drops_overflow_without_postponement(self):
        config, _, e = toy_case([[0, 0, 0, 0], [0, 0, 0, 0]], prices=[4, 1, 1, 1])
        w = WorkloadAggregate(
            "overflow", np.array([8.0, 2.0, 0.0, 0.0]), np.zeros((2, 4)),
            10.0, np.zeros(2), np.array([2, 0, 0]), 2,
            opt_out_bill_usd_by_arrival=np.array([8.0, 2.0, 0.0, 0.0]),
            participating_bill_usd_by_type_arrival=np.zeros((2, 4)),
        )

        result = immediate_zero_discount_with_overflow(config, w, e)

        self.assertTrue(result.feasible, result.message)
        np.testing.assert_allclose(result.executed_tokens_by_slot, [5.0, 2.0, 0.0, 0.0])
        np.testing.assert_allclose(result.metadata["dropped_tokens_by_slot"], [3.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(result.metadata["dropped_tokens"], 3.0)
        self.assertAlmostEqual(result.shifted_tokens, 0.0)
        self.assertAlmostEqual(result.maximum_delay_hours, 0.0)
        self.assertAlmostEqual(result.discount_cost_usd, 0.0)
        self.assertAlmostEqual(result.revenue_usd, 7.0)
        self.assertAlmostEqual(
            result.profit_usd,
            result.revenue_usd - result.fixed_energy_cost_usd - result.day_ahead_cost_usd,
        )

    def test_comparison_writes_arrival_and_three_method_schedule_plot(self):
        config, w, e = toy_case()
        proposed = solve_aggregate(config, w, e)
        benchmarks = run_benchmarks(config, w, e)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_method_comparison(output, proposed, benchmarks, config)
            self.assertTrue((output / "workload_schedule_comparison.png").is_file())
            with (output / "voluntary_flat_50_24_slot_schedule.csv").open() as stream:
                flat_rows = list(csv.DictReader(stream))
            self.assertIn("dropped_tokens", flat_rows[0])
            with (output / "immediate_zero_discount_drop_overflow_slot_schedule.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertIn("dropped_tokens", rows[0])


class InputOutputTests(unittest.TestCase):
    def test_real_recipe_aggregates_millions_without_task_expansion(self):
        config = load_config(PROJECT / "config/default.yaml")
        w = load_workload_aggregate(config.inputs.root, config.inputs.workload_scenario_id,
                                    config.parameters, config.inputs.workload_file,
                                    config.inputs.immediate_execution_share)
        self.assertGreater(w.task_count, 0)
        self.assertGreater(w.total_tokens, 0)
        self.assertEqual(
            w.participating_tokens_by_type_arrival.shape,
            (len(config.parameters.type_lambdas), config.parameters.horizon_slots),
        )
        self.assertFalse(hasattr(w, "tasks"))
        self.assertAlmostEqual(
            w.type_counts[0] / w.task_count,
            config.inputs.immediate_execution_share,
            delta=1 / w.task_count,
        )
        self.assertLess(w.source_record_count, w.task_count)

    def test_predefined_distribution_is_validated(self):
        validate_type_counts(np.array([5, 3, 2]), (.5, .3, .2), .5)
        with self.assertRaises(DataValidationError):
            validate_type_counts(np.array([5, 4, 1]), (.5, .3, .2), .5)

    def test_result_bundle_prices_conservation_and_config_round_trip(self):
        config, w, e = toy_case()
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(config, output=replace(config.output, root=Path(tmp), make_plot=False))
            result = write_result_bundle(solve_aggregate(config, w, e), config)
            output = result.output_directory
            rows = json.loads((output / "contract_menu.json").read_text())
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertAlmostEqual(row["input_price_usd_per_million_tokens"],
                                       config.parameters.input_tariff_usd_per_million_tokens
                                       * (1 - row["discount_fraction"]))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "optimal_within_tolerance")
            self.assertLess(summary["validation"]["maximum_terminal_mass_error_tokens"], 1e-7)
            loaded = load_config(output / "effective_config.yaml")
            self.assertEqual(loaded.parameters, config.parameters)
            with (output / "group_schedule.csv").open() as stream:
                allocations = list(csv.DictReader(stream))
            self.assertAlmostEqual(sum(float(row["tokens"]) for row in allocations),
                                   w.participating_tokens_by_type_arrival.sum())
            self.assertTrue(all(int(row["execution_slot"]) <= 4 for row in allocations))
            self.assertTrue(all(int(row["arrival_slot"]) <= int(row["execution_slot"])
                                <= int(row["deadline_slot"]) for row in allocations))

    def test_fixed_csv_allows_a_divisible_group_larger_than_one_slot(self):
        config, _, _ = toy_case()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.csv"
            path.write_text(
                "scenario_id,task_id,arrival_slot,input_tokens,output_tokens,total_tokens,task_type\n"
                "case,one,1,7,1,8,1\ncase,two,4,1,1,2,0\n", encoding="utf-8")
            w = load_workload_aggregate(Path(tmp), "case", config.parameters, path, .5, (.5, .5, 0.))
            self.assertEqual(w.total_tokens, 10)
            self.assertEqual(w.participating_tokens_by_type_arrival[0, 0], 8)
            self.assertAlmostEqual(w.participating_bill_usd_by_type[0],
                                   (7 * config.parameters.input_tariff_usd_per_million_tokens
                                    + config.parameters.output_tariff_usd_per_million_tokens) / 1e6)

    def test_nonconverged_result_bundle_retains_gap_and_feasible_menu(self):
        config, w, e = toy_case()
        config = replace(config, solver=replace(config.solver, max_iterations=1))
        menu = menu_from_deadlines((2, 2), config.parameters)
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(config, output=replace(config.output, root=Path(tmp), make_plot=False))
            with patch.object(BendersMaster, "solve", return_value=MasterSolution("limit", menu, 0., "timeout")):
                result = write_result_bundle(solve_aggregate(config, w, e), config)
            summary = json.loads((result.output_directory / "summary.json").read_text())
            self.assertEqual(summary["status"], "best_feasible")
            self.assertGreater(summary["certificate"]["absolute_gap_usd"], 0.)
            self.assertEqual(summary["contract_menu"][1]["max_deferral_slots"], 2)

    def test_script_runs_outside_project_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            completed = subprocess.run([sys.executable, str(PROJECT / "src/llm_contract_optimizer/__main__.py"),
                                        "audit-menu"], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(completed.stdout)["contract_domain_feasible"])


if __name__ == "__main__":
    unittest.main()
