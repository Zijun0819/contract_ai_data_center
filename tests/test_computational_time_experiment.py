"""Timing boundaries, matrix orientation, failures and joint input controls."""
import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from llm_contract_optimizer.computational_time_experiment import (
    measure_case, summarize_timings, table_cells, validate_timing_suite, write_timing_tables,
)
from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.data import load_workload_aggregate
from llm_contract_optimizer.data import validate_type_counts
from llm_contract_optimizer.exceptions import DataValidationError, SolveError
from llm_contract_optimizer.solve import solve_aggregate
from llm_contract_optimizer.type_count_experiment import rebin_fixed_types
from test_fluid_solver import PROJECT, toy_case


def timing_row(hours, number, seconds, status="optimal_within_tolerance"):
    return {"horizon_hours": hours, "participating_type_count": number, "wall_seconds": seconds,
            "status": status, "converged": status == "optimal_within_tolerance"}


class TimingProtocolTests(unittest.TestCase):
    def test_odd_daily_count_preserves_exact_share_target_and_rounded_immediate_quota(self):
        known = {"counts_by_source_type": [[6, 3, 2]], "seed": 91, "target_immediate_share": .5}
        for number in (3, 4, 6, 8, 10):
            result = rebin_fixed_types(known, number)
            self.assertEqual(result["target_distribution"][0], .5)
            self.assertEqual(result["scenario_counts"][0], 6)
            validate_type_counts(np.array(result["scenario_counts"]), result["target_distribution"], .5)

    def test_outer_clock_includes_solve_and_excludes_validation(self):
        config, workload, energy = toy_case()
        result = solve_aggregate(config, workload, energy)
        clock = [10.0]

        def solve(*args):
            clock[0] += 3
            return result

        def validate(*args):
            clock[0] += 100
            return {}

        with patch("llm_contract_optimizer.computational_time_experiment.time.perf_counter", side_effect=lambda: clock[0]), \
             patch("llm_contract_optimizer.computational_time_experiment.solve_aggregate", side_effect=solve), \
             patch("llm_contract_optimizer.computational_time_experiment._check_proposed", side_effect=validate):
            row, actual = measure_case(config, workload, energy, 1, 1)
        self.assertEqual(row["wall_seconds"], 3)
        self.assertEqual(row["algorithm_elapsed_seconds"], result.elapsed_seconds)
        self.assertIs(actual, result)

    def test_failed_attempt_keeps_elapsed_time_without_claiming_convergence(self):
        config, workload, energy = toy_case()
        with patch("llm_contract_optimizer.computational_time_experiment.time.perf_counter", side_effect=[7., 11.]), \
             patch("llm_contract_optimizer.computational_time_experiment.solve_aggregate", side_effect=SolveError("no incumbent")):
            row, result = measure_case(config, workload, energy, 1, 1)
        self.assertEqual(row["wall_seconds"], 4)
        self.assertEqual(row["status"], "error")
        self.assertFalse(row["converged"])
        self.assertIsNone(result)

    def test_repeat_median_and_dispersion_do_not_hide_failed_or_incomplete_pairs(self):
        rows = [timing_row(24, 2, v) for v in (4., 8., 6.)]
        summary = summarize_timings(rows, [24], [2], 3)[0]
        self.assertEqual(summary["median_seconds"], 6)
        self.assertEqual(summary["sample_std_seconds"], 2)
        self.assertIsNone(summarize_timings(rows[:2], [24], [2], 3)[0]["median_seconds"])
        rows[-1] = timing_row(24, 2, 2., "error")
        summary = summarize_timings(rows, [24], [2], 3)[0]
        self.assertEqual(summary["status"], "failed")
        self.assertIsNone(summary["median_seconds"])
        self.assertEqual(summary["attempt_median_seconds"], 4)

    def test_numeric_matrix_has_horizon_rows_type_columns_and_status_markers(self):
        rows = [timing_row(24, 4, 9., "best_feasible"), timing_row(12, 2, 1.),
                timing_row(12, 4, 3., "error"), timing_row(24, 2, 5.)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summaries = write_timing_tables(output, rows, [24, 12], [4, 2], 1)
            with (output / "computational_time_table.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["planning_horizon_hours", "2", "4"])
                matrix = list(reader)
            self.assertEqual([r["planning_horizon_hours"] for r in matrix], ["12", "24"])
            self.assertEqual(matrix[0]["2"], "1.0")
            self.assertEqual(matrix[0]["4"], "")
            self.assertEqual(matrix[1]["4"], "9.0")
            cells = table_cells(summaries, [24, 12], [4, 2])
            self.assertEqual(cells, [["12", "1.000", "ERR"], ["24", "5.000", "9.000†"]])


class PreparedTimingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = PROJECT / "data/computational_time/experiment_manifest.json"
        _, cases = validate_timing_suite(cls.manifest, [12, 24, 168], [2, 8, 10])
        cls.cases = {(entry["hours"], entry["type_count"]): (config, workload, energy)
                     for entry, config, workload, energy in cases}

    def test_joint_grid_keeps_default_arrivals_immediate_labels_and_power(self):
        default = load_config(PROJECT / "config/default.yaml")
        original = load_workload_aggregate(default.inputs.root, default.inputs.workload_scenario_id,
                   default.parameters, default.inputs.workload_file, .5)
        anchor = self.cases[24, 8][1]
        np.testing.assert_array_equal(anchor.participating_tokens_by_type_arrival,
                                      original.participating_tokens_by_type_arrival)
        np.testing.assert_array_equal(anchor.type_counts, original.type_counts)
        for (hours, number), (config, workload, energy) in self.cases.items():
            np.testing.assert_array_equal(config.parameters.type_lambdas, np.linspace(.1, 2.5, number))
            fixed, power = self.cases[hours, 8][1:]
            np.testing.assert_array_equal(workload.opt_out_tokens_by_arrival, fixed.opt_out_tokens_by_arrival)
            np.testing.assert_array_equal(energy.day_ahead_price_usd_per_kwh, power.day_ahead_price_usd_per_kwh)
            longest = self.cases[168, number][1]
            np.testing.assert_array_equal(workload.participating_tokens_by_type_arrival,
                                          longest.participating_tokens_by_type_arrival[:, -config.parameters.horizon_slots:])

    def test_unrequested_model_change_is_rejected(self):
        original = self.cases[24, 2][0]
        changed = replace(original, parameters=replace(original.parameters, pue=original.parameters.pue + .1))
        with patch("llm_contract_optimizer.computational_time_experiment.load_config", return_value=changed):
            with self.assertRaisesRegex(DataValidationError, "Only planning horizon"):
                validate_timing_suite(self.manifest, [24], [2])


if __name__ == "__main__":
    unittest.main()
