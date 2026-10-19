"""Task-cohort controls, conditional type quotas and infeasibility reporting."""
import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.domain import SourceTask
from llm_contract_optimizer.exceptions import DataValidationError
from llm_contract_optimizer.immediate_share_experiment import (
    SHARE_PERCENTAGES, assign_share_types, admit_immediate_overflow, controlled_settings,
    validate_immediate_share_suite, run_immediate_share_experiment,
)
from llm_contract_optimizer.solve import solve_aggregate
from test_fluid_solver import toy_case

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "data/immediate_shares/experiment_manifest.json"


class ShareLabelTests(unittest.TestCase):
    def test_fixed_copy_counts_conditional_mix_and_exact_default_identity(self):
        tasks = (SourceTask("a", 1, 1, 1, 2, np.array([.2, .5, .3])),
                 SourceTask("b", 2, 1, 1, 2, np.array([.6, .2, .2])))
        known = {"counts_by_source_type": [[2, 6, 2], [8, 1, 1]],
                 "scenario_counts": [10, 7, 3], "target_immediate_share": .5, "seed": 18}
        self.assertEqual(assign_share_types(tasks, [10, 10], known, .5), known)
        expected_quotas = {10: [2, 13, 5], 20: [4, 11, 5], 30: [6, 10, 4], 40: [8, 8, 4]}
        for percentage, quotas in expected_quotas.items():
            result = assign_share_types(tasks, [10, 10], known, percentage / 100)
            labels = np.asarray(result["counts_by_source_type"])
            np.testing.assert_array_equal(labels.sum(axis=0), quotas)
            np.testing.assert_array_equal(labels.sum(axis=1), [10, 10])
            self.assertEqual(result, assign_share_types(tasks, [10, 10], known, percentage / 100))
        self.assertEqual(known["counts_by_source_type"], [[2, 6, 2], [8, 1, 1]])

    def test_rejects_invalid_share_and_mismatched_copy_counts(self):
        tasks = (SourceTask("a", 1, 1, 1, 2, np.array([.5, .5])),)
        known = {"counts_by_source_type": [[1, 1]], "target_immediate_share": .5, "seed": 1}
        for share in (-.1, 1., float("nan"), True):
            with self.assertRaises(DataValidationError):
                assign_share_types(tasks, [2], known, share)
        with self.assertRaises(DataValidationError):
            assign_share_types(tasks, [3], known, .1)


class PreparedImmediateShareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, entries = validate_immediate_share_suite(MANIFEST, list(SHARE_PERCENTAGES))
        cls.cases = {entry["immediate_share_percent"]: (entry, config, workload, energy)
                     for entry, config, workload, energy in entries}
        cls.default = load_config(PROJECT / "config/default.yaml")

    def test_only_share_changes_and_50_percent_is_the_exact_default(self):
        original = json.loads(self.default.inputs.workload_file.read_text())
        for percentage, (_, config, workload, energy) in self.cases.items():
            self.assertEqual(controlled_settings(config), controlled_settings(self.default))
            self.assertEqual(config.inputs.immediate_execution_share, percentage / 100)
            self.assertEqual(config.parameters.horizon_slots * config.parameters.slot_seconds, 86400)
            payload = json.loads(config.inputs.workload_file.read_text())
            self.assertEqual(payload["draws"], original["draws"])
            self.assertEqual(payload["source_sha256"], original["source_sha256"])
            if percentage == 50:
                self.assertEqual(payload["known_types"], original["known_types"])

    def test_same_arrivals_bills_and_power_and_rounding_bounded_type_mix(self):
        _, _, reference, reference_energy = self.cases[50]
        mix = reference.type_counts[1:] / reference.type_counts[1:].sum()
        for percentage, (_, _, workload, energy) in self.cases.items():
            self.assertEqual(workload.task_count, reference.task_count)
            self.assertEqual(workload.total_tokens, reference.total_tokens)
            np.testing.assert_array_equal(workload.opt_out_tokens_by_arrival + workload.participating_tokens_by_type_arrival.sum(axis=0),
                                          reference.opt_out_tokens_by_arrival + reference.participating_tokens_by_type_arrival.sum(axis=0))
            np.testing.assert_allclose(workload.total_bill_usd_by_arrival, reference.total_bill_usd_by_arrival, rtol=0, atol=1e-8)
            self.assertLessEqual(abs(workload.type_counts[0] - percentage / 100 * workload.task_count), .5)
            remaining = workload.task_count - workload.type_counts[0]
            self.assertTrue(np.all(np.abs(workload.type_counts[1:] - remaining * mix) <= 1.))
            for field in ("ppa_kwh", "gas_turbine_kwh", "pv_kwh", "bess_discharge_kwh", "bess_charge_kwh", "day_ahead_price_usd_per_kwh"):
                np.testing.assert_array_equal(getattr(energy, field), getattr(reference_energy, field))

    def test_revised_range_passes_screens_and_immediate_copy_prefixes_are_nested(self):
        self.assertEqual(set(self.cases), set(SHARE_PERCENTAGES))
        previous = None
        for percentage in SHARE_PERCENTAGES:
            entry, config, workload, energy = self.cases[percentage]
            self.assertEqual(entry["capacity_screen"], "passed")
            self.assertLessEqual(entry["peak_immediate_capacity_fraction"], 1.)
            labels = np.asarray(json.loads(config.inputs.workload_file.read_text())["known_types"]["counts_by_source_type"])
            if previous is not None:
                self.assertTrue(np.all(labels[:, 0] >= previous))
            previous = labels[:, 0]

    def test_rejects_other_parameter_changes_and_incorrect_share(self):
        original = self.cases[10][1]
        changes = (replace(original, parameters=replace(original.parameters, installed_replicas=8192)),
                   replace(original, inputs=replace(original.inputs, immediate_execution_share=.3)))
        for changed in changes:
            def lookup(path):
                return changed if Path(path).resolve() == original.source_path else load_config(path)
            with patch("llm_contract_optimizer.immediate_share_experiment.load_config", side_effect=lookup):
                with self.assertRaises(DataValidationError):
                    validate_immediate_share_suite(MANIFEST, [10])


class OverflowAdmissionTests(unittest.TestCase):
    def test_drop_scales_immediate_bills_retains_flexible_work_and_does_not_mutate_input(self):
        config, workload, energy = toy_case([[1, 0, 0, 0], [1, 0, 0, 0]], immediate=[8, 2, 0, 0])
        admitted, info = admit_immediate_overflow(workload, 5.)
        np.testing.assert_array_equal(admitted.opt_out_tokens_by_arrival, [5, 2, 0, 0])
        np.testing.assert_array_equal(admitted.participating_tokens_by_type_arrival, workload.participating_tokens_by_type_arrival)
        np.testing.assert_array_equal(workload.opt_out_tokens_by_arrival, [8, 2, 0, 0])
        self.assertEqual(admitted.opt_out_bill_usd, 7.)
        self.assertEqual(info["dropped_tokens"], 3.)
        self.assertEqual(info["dropped_undiscounted_bill_usd"], 3.)
        self.assertEqual(info["drop_fraction"], .25)
        same, info = admit_immediate_overflow(admitted, 5.)
        self.assertIs(same, admitted)
        self.assertEqual(info["dropped_tokens"], 0.)

    def test_overflow_is_dropped_and_served_work_metrics_are_reported_for_all_methods(self):
        config, workload, energy = toy_case([[0, 0, 0, 0], [0, 0, 0, 0]], immediate=[8, 2, 0, 0])
        entry = {"immediate_share_percent": 50, "actual_immediate_task_share": .5,
                 "actual_immediate_token_share": 1., "peak_immediate_capacity_fraction": 1.6}
        # The real solver and both real benchmark policies run on an overloaded
        # toy instance; only the prepared-package lookup is replaced.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest_path = output / "manifest.json"
            (output / "type_distributions.csv").write_text("task_type,task_count\n0,10\n")
            manifest = {"files": {"type_distributions.csv": "unused"}, "cases": []}
            args = SimpleNamespace(manifest=manifest_path, shares=[50], validate_only=False,
                                   output_dir=output / "results", time_limit=None, no_plot=True, no_group_schedule=True)
            with patch("llm_contract_optimizer.immediate_share_experiment.validate_immediate_share_suite",
                       return_value=(manifest, [(entry, config, workload, energy)])):
                self.assertEqual(run_immediate_share_experiment(args), 0)
            run = next((output / "results").iterdir())
            with (run / "immediate_share_metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["status"], "optimal_within_tolerance")
            self.assertEqual(float(rows[0]["dropped_tokens"]), 3.)
            self.assertEqual(float(rows[0]["dropped_undiscounted_bill_usd"]), 3.)
            self.assertEqual(float(rows[0]["discount_expenditure_usd"]), 0.)
            self.assertEqual(float(rows[0]["revenue_usd"]), 7.)
            self.assertEqual([row["status"] for row in rows[1:]], ["feasible", "feasible"])
            summary = json.loads((run / "experiment_summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["errors"], [])
            self.assertEqual(summary["proposed_infeasible_share_percentages"], [])
            self.assertFalse(list(run.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
