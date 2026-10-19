"""Representative-day units, paired inputs and complete figure coverage."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from llm_contract_optimizer.benchmarks import run_benchmarks
from llm_contract_optimizer.exceptions import DataValidationError
from llm_contract_optimizer.representative_day import FIGURES, METHODS, day_series, write_day_plots
from llm_contract_optimizer.solve import solve_aggregate
from test_fluid_solver import toy_case


class RepresentativeDayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config, workload, energy = toy_case(mass=[[6, 0, 1, 0], [1, 0, 0, 1]])
        cls.config = replace(config, parameters=replace(config.parameters, slot_seconds=900,
                             throughput_tokens_per_second_per_replica=5 / 900))
        cls.proposed = solve_aggregate(cls.config, workload, energy)
        cls.benchmarks = run_benchmarks(cls.config, workload, energy)
        cls.data = day_series(cls.config, cls.proposed, cls.benchmarks)

    def test_slot_energy_power_conversion_and_dropped_work(self):
        data = self.data
        self.assertGreater(data["zero_discount_dropped_tokens"].sum(), 0)
        self.assertEqual(data["proposed_dropped_tokens"].sum(), 0)
        for prefix, _, _, _ in METHODS:
            imported_kwh = data[f"{prefix}_grid_import_kwh"]
            np.testing.assert_allclose(data[f"{prefix}_grid_import_mw"] * .25 * 1000, imported_kwh)
            self.assertAlmostEqual(data[f"{prefix}_executed_tokens"].sum()
                                   + data[f"{prefix}_dropped_tokens"].sum(),
                                   data["arrival_tokens"].sum())

    def test_inconsistent_energy_and_benchmark_inputs_are_rejected(self):
        bad = replace(self.proposed, inner=replace(self.proposed.inner,
                      day_ahead_energy_kwh=self.proposed.inner.day_ahead_energy_kwh + 1))
        with self.assertRaisesRegex(DataValidationError, "energy balance"):
            day_series(self.config, bad, self.benchmarks)
        bad_benchmark = replace(self.benchmarks[0],
                                arrival_tokens_by_slot=self.benchmarks[0].arrival_tokens_by_slot + 1)
        with self.assertRaisesRegex(DataValidationError, "arrivals differ"):
            day_series(self.config, self.proposed, (bad_benchmark, self.benchmarks[1]))

    def test_only_four_figures_and_no_peak_or_final_slot_clipping(self):
        captured = {}

        def capture(fig, path, **kwargs):
            ax = fig.axes[0]
            self.assertEqual(ax.get_xlim(), (0, 1))
            for stairs in ax.patches:
                stair_data = stairs.get_data()
                self.assertEqual(stair_data.edges[-1], 1)
                self.assertGreaterEqual(ax.get_ylim()[1], np.max(stair_data.values))
            captured[Path(path).name] = ax.get_ylabel()

        with patch("matplotlib.figure.Figure.savefig", new=capture):
            write_day_plots(Path("unused"), self.config, self.data, True)
        self.assertEqual(set(captured), {f"{name}.{ext}" for name in FIGURES for ext in ("png", "pdf")})
        self.assertEqual(captured["grid_imports.png"], "Average grid import (MW)")
        self.assertEqual(captured["day_ahead_market_price.png"], "USD/kWh")


if __name__ == "__main__":
    unittest.main()
