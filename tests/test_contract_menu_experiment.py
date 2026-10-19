"""Selection reproducibility and the four grouped contract-menu figures."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.contract_menu_experiment import (
    FIGURES, contract_rows, select_four, write_contract_plots,
)
from llm_contract_optimizer.exceptions import DataValidationError, SolveError
from test_fluid_solver import PROJECT


class ContractMenuExperimentTests(unittest.TestCase):
    def test_selection_is_reproducible_without_replacement_or_enumeration_bias(self):
        population = [f"R{i:02d}" for i in range(1, 13)]
        selected = select_four(population, 20261001)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertEqual(selected, select_four(list(reversed(population)), 20261001))
        self.assertTrue(set(selected).issubset(population))
        with self.assertRaises(DataValidationError):
            select_four([12, 24, 48], 1)
        with self.assertRaises(DataValidationError):
            select_four([12, 24, 24, 48], 1)

    def test_failed_case_preserves_all_type_positions_as_missing_values(self):
        config = load_config(PROJECT / "config/default.yaml")
        rows = contract_rows({"comparison": "workload", "case_label": "failed"}, config,
                             error=SolveError("no incumbent"))
        self.assertEqual([r["user_type"] for r in rows], list(range(1, 9)))
        self.assertTrue(all(r["deferral_hours"] is None and r["discount_percent"] is None for r in rows))
        self.assertTrue(all(r["status"] == "error" for r in rows))

    def test_four_figures_have_eight_type_groups_four_cases_and_correct_units(self):
        config = load_config(PROJECT / "config/default.yaml")
        selection = {"selected_workload_scenarios": ["A", "B", "C", "D"],
                     "selected_horizons_hours": [24, 48, 96, 120],
                     "workload_comparison": {"fixed_power_profile": "D02", "horizon_hours": 24},
                     "horizon_comparison": {"fixed_workload_anchor": "R05-D333", "fixed_power_anchor": "D02"}}
        menu = SimpleNamespace(deadlines_slots=(8, 4, 3, 2, 1, 1, 1, 1),
                               discounts=(.20, .19, .18, .17, .16, .15, .14, .13))
        result = SimpleNamespace(menu=menu, converged=True, termination_reason="gap_reached", absolute_gap_usd=0.)
        rows = []
        for comparison, labels in (("workload", ["A", "B", "C", "D"]),
                                    ("horizon", ["24 h", "48 h", "96 h", "120 h"])):
            for label in labels:
                rows.extend(contract_rows({"comparison": comparison, "case_label": label}, config, result))
        captured = set()

        def capture(fig, path, **kwargs):
            name = Path(path).name
            ax = fig.axes[0]
            self.assertEqual(len(ax.patches), 32)
            np.testing.assert_array_equal(ax.get_xticks(), np.arange(1, 9))
            labels = ax.get_legend_handles_labels()[1]
            self.assertEqual(labels, ["A", "B", "C", "D"] if "workload" in name
                             else ["24 h", "48 h", "96 h", "120 h"])
            if "deferral" in name:
                expected = [2., 1., .75, .5, .25, .25, .25, .25]
                self.assertEqual(ax.get_ylabel(), "Deferral window (hours)")
            else:
                expected = [20., 19., 18., 17., 16., 15., 14., 13.]
                self.assertEqual(ax.get_ylabel(), "Discount (%)")
            np.testing.assert_allclose([bar.get_height() for bar in ax.patches], expected * 4)
            self.assertGreater(ax.get_ylim()[1], max(expected))
            for index in range(4):
                centers = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches[index * 8:(index + 1) * 8]]
                np.testing.assert_allclose(centers, np.arange(1, 9) + (index - 1.5) * .2)
            captured.add(name)

        with patch("matplotlib.figure.Figure.savefig", new=capture):
            write_contract_plots(Path("unused"), selection, rows)
        self.assertEqual(captured, {f"{name}.{ext}" for name in FIGURES for ext in ("png", "pdf")})


if __name__ == "__main__":
    unittest.main()
