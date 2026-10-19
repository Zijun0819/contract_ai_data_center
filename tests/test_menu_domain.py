"""Contract-domain checks independent of the full-campus scheduling size."""
import itertools
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import linprog

from llm_contract_optimizer.config import load_config, validate_config
from llm_contract_optimizer.domain import EnergyScenario, WorkloadAggregate
from llm_contract_optimizer.config import SolverConfig
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.master import BendersMaster
from llm_contract_optimizer.exceptions import ConfigurationError
from llm_contract_optimizer.menu import audit_menu_domain, delay_penalty, menu_from_deadlines, presolve_parameters


PROJECT = Path(__file__).resolve().parents[1]


def direct_ic_ir_lp(deadlines, parameters):
    """Minimize discounts with every original IC and IR inequality explicitly."""
    lambdas = parameters.type_lambdas
    hours = np.array(deadlines) * parameters.slot_hours
    penalties = parameters.delay_curvature_per_hour * hours / (1 + parameters.delay_curvature_per_hour * hours)
    rows, bounds = [], []
    for i, sensitivity in enumerate(lambdas):
        row = np.zeros(len(lambdas))
        row[i] = -1
        rows.append(row)
        bounds.append(-sensitivity * penalties[i])
        for j in range(len(lambdas)):
            if i == j:
                continue
            row = np.zeros(len(lambdas))
            row[i], row[j] = -1, 1
            rows.append(row)
            bounds.append(sensitivity * (penalties[j] - penalties[i]))
    return linprog(np.ones(len(lambdas)), A_ub=rows, b_ub=bounds,
                   bounds=(0, parameters.maximum_discount), method="highs")


class MenuDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These regression cases have analytical values for the historical
        # calibration; the user-editable experiment defaults may use another one.
        cls.parameters = replace(
            load_config(PROJECT / "config/default.yaml").parameters,
            type_lambdas=tuple(.000375 * i for i in range(1, 9)),
            delay_curvature_per_hour=1 / 6,
        )

    def test_single_canonical_config_keeps_full_candidate_grid(self):
        expected = (1, 2, 3) + tuple(range(4, 97, 4))
        self.assertEqual(load_config(PROJECT / "config/default.yaml").parameters.candidate_deferrals_slots,
                         expected)

    def test_original_boundary_still_presolve_correctly(self):
        original_lambdas = (0.00160037, 0.0048011, 0.01440329, 0.04320988,
                            0.12962963, 0.38888889, 1.16666667, 3.5)
        original = replace(self.parameters, type_lambdas=original_lambdas,
                           candidate_deferrals_slots=tuple(range(4, 97, 4)))
        self.assertEqual(presolve_parameters(original).candidate_deferrals_slots, (4,))
        self.assertTrue(audit_menu_domain(original)["pooling_forced"])
        self.assertFalse(direct_ic_ir_lp((8,) + (4,) * 7, original).success)

    def test_new_domain_and_eight_distinct_items(self):
        audit = audit_menu_domain(self.parameters)
        self.assertEqual(self.parameters.maximum_discount, .5)
        self.assertFalse(audit["pooling_forced"])
        self.assertTrue(audit["differentiated_menu_feasible"])
        self.assertAlmostEqual(audit["minimum_pooled_discount"], .00012)
        self.assertAlmostEqual(audit["discount_cap_slack_at_shortest_menu"], .49988)
        self.assertAlmostEqual(audit["maximum_feasible_sensitivity_at_minimum_deferral"], 12.5)
        self.assertAlmostEqual(audit["maximum_sensitivity_supporting_every_candidate_deferral"], .625)
        self.assertAlmostEqual(audit["configured_fraction_of_feasibility_cap"], .003 / 12.5)
        self.assertEqual(audit["master_binary_variables_after_presolve"], 216)
        deadlines = (96, 80, 64, 48, 32, 16, 8, 1)
        result = direct_ic_ir_lp(deadlines, self.parameters)
        self.assertTrue(result.success)
        menu = menu_from_deadlines(deadlines, self.parameters)
        np.testing.assert_allclose(menu.discounts, result.x, atol=1e-8, rtol=0)
        self.assertLess(max(menu.discounts), .5)

    def test_lambda_feasibility_cap_is_enforced(self):
        cap = self.parameters.maximum_discount / delay_penalty(
            self.parameters.candidate_deferrals_slots[0], self.parameters)
        invalid_parameters = replace(
            self.parameters,
            type_lambdas=self.parameters.type_lambdas[:-1] + (cap + 1e-6,),
        )
        config = load_config(PROJECT / "config/default.yaml")
        with self.assertRaisesRegex(ConfigurationError, "analytical feasibility cap"):
            validate_config(replace(config, parameters=invalid_parameters))

    def test_positive_slack_alone_does_not_guarantee_differentiation(self):
        p = replace(self.parameters, type_lambdas=(1., 3.5),
                    candidate_deferrals_slots=(1, 4), maximum_discount=.15)
        audit = audit_menu_domain(p)
        self.assertGreater(audit["discount_cap_slack_at_shortest_menu"], 0)
        self.assertTrue(audit["pooling_forced"])

    def test_audit_matches_exhaustive_original_ic_ir_constraints(self):
        for cap in (.13, .14, .15, .2, .5):
            p = replace(self.parameters, type_lambdas=(.1, 1., 3.5),
                        candidate_deferrals_slots=(1, 2, 4), maximum_discount=cap)
            feasible = []
            for deadlines in itertools.product(p.candidate_deferrals_slots, repeat=3):
                if any(a < b for a, b in zip(deadlines, deadlines[1:])):
                    continue
                if direct_ic_ir_lp(deadlines, p).success:
                    feasible.append(deadlines)
            audit = audit_menu_domain(p)
            self.assertEqual(audit["contract_domain_feasible"], bool(feasible))
            self.assertEqual(audit["differentiated_menu_feasible"], any(len(set(d)) > 1 for d in feasible))

    def test_master_can_select_nonpooled_menu(self):
        p = replace(self.parameters, type_lambdas=(.1, 3.5), candidate_deferrals_slots=(1, 2, 4))
        workload = WorkloadAggregate("menu-test", np.zeros(96), np.zeros((2, 96)),
                                     0., np.array([100., 100.]), np.array([0, 0, 0]))
        energy = EnergyScenario("test", "test", *[np.zeros(96) for _ in range(6)])
        inner = FluidInnerSolver(workload, energy, p, SolverConfig())
        # Represent a subproblem that has ruled out the all-shortest menu.
        result = BendersMaster(inner).solve([], [(1, 1)], 10., 0.)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.menu.deadlines_slots, (2, 1))
        reference = direct_ic_ir_lp((2, 1), p)
        self.assertTrue(reference.success)
        self.assertAlmostEqual(result.lower_bound, 100 * reference.fun)


if __name__ == "__main__":
    unittest.main()
