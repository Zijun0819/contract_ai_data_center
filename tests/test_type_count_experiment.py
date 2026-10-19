"""Type-count mapping, fixed-copy controls, and experiment integrity."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.exceptions import DataValidationError
from llm_contract_optimizer.preferences import balanced_integer_labels
from llm_contract_optimizer.sampling import read_sampling_source
from llm_contract_optimizer.type_count_experiment import (
    TYPE_COUNTS, ordered_bin_mapping, rebin_fixed_types, validate_type_count_suite,
)

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "data/type_counts/experiment_manifest.json"


class OrderedTypeMappingTests(unittest.TestCase):
    def test_generalizes_existing_equal_split_and_exact_merge(self):
        np.testing.assert_array_equal(ordered_bin_mapping(4, 8), np.repeat(np.eye(4), 2, axis=1) / 2)
        np.testing.assert_array_equal(ordered_bin_mapping(8, 8), np.eye(8))
        np.testing.assert_array_equal(ordered_bin_mapping(8, 4), np.repeat(np.eye(4), 2, axis=0))
        # Independent two-to-three-bin example: overlap lengths are 1/3 and 1/6.
        np.testing.assert_allclose(np.array([.3, .7]) @ ordered_bin_mapping(2, 3), [.2, 1 / 3, 7 / 15])
        for number in TYPE_COUNTS:
            matrix = ordered_bin_mapping(8, number)
            np.testing.assert_allclose(matrix.sum(axis=1), 1.)
            self.assertTrue(np.all(matrix >= 0))

    def test_rounding_keeps_zero_rows_immediate_prefix_and_quotas(self):
        known = {"counts_by_source_type": [[5, 0, 0, 0, 0], [3, 2, 4, 0, 3], [2, 1, 0, 3, 1]], "seed": 91}
        result = rebin_fixed_types(known, 6)
        labels = np.array(result["counts_by_source_type"])
        original = np.array(known["counts_by_source_type"])
        np.testing.assert_array_equal(labels[:, 0], original[:, 0])
        np.testing.assert_array_equal(labels.sum(axis=1), original.sum(axis=1))
        np.testing.assert_array_equal(labels.sum(axis=0), result["scenario_counts"])
        np.testing.assert_array_equal(labels[0, 1:], 0)
        self.assertEqual(result, rebin_fixed_types(known, 6))

    def test_rounding_rejects_inconsistent_row_or_global_totals(self):
        with self.assertRaises(DataValidationError):
            balanced_integer_labels([[.4, .6]], [2], [1, 1], 1)
        with self.assertRaises(DataValidationError):
            balanced_integer_labels([[.4, .6]], [1], [1, 1], 1)


class PreparedTypeCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, entries = validate_type_count_suite(MANIFEST, list(TYPE_COUNTS))
        cls.cases = {entry["type_count"]: (config, workload, energy) for entry, config, workload, energy in entries}
        cls.default = load_config(PROJECT / "config/default.yaml")
        cls.baseline = json.loads(cls.default.inputs.workload_file.read_text(encoding="utf-8"))

    def test_all_grids_share_endpoints_and_even_spacing_at_24h(self):
        for number, (config, workload, energy) in self.cases.items():
            self.assertEqual(len(config.parameters.type_lambdas), number)
            self.assertEqual(config.parameters.type_lambdas[0], .1)
            self.assertEqual(config.parameters.type_lambdas[-1], 2.5)
            np.testing.assert_allclose(np.diff(config.parameters.type_lambdas), 2.4 / (number - 1))
            self.assertEqual(config.parameters.horizon_slots * config.parameters.slot_seconds, 86400)
            self.assertEqual(int(workload.type_counts[0]) * 2, workload.task_count)
            self.assertEqual(workload.total_tokens, self.baseline["target_tokens"])

    def test_same_tasks_arrivals_immediate_identities_and_power_in_every_case(self):
        original_raw, _ = read_sampling_source(self.default.inputs.workload_file,
            self.default.inputs.workload_scenario_id, self.default.parameters)
        reference_tasks = [(t.task_id, t.arrival_slot, t.input_tokens, t.output_tokens, t.total_tokens) for t in original_raw.tasks]
        reference_labels = np.array(self.baseline["known_types"]["counts_by_source_type"])
        reference_energy = self.cases[8][2]
        for number, (config, workload, energy) in self.cases.items():
            payload = json.loads(config.inputs.workload_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["draws"], self.baseline["draws"])
            np.testing.assert_array_equal(np.array(payload["known_types"]["counts_by_source_type"])[:, 0], reference_labels[:, 0])
            raw, _ = read_sampling_source(config.inputs.workload_file, config.inputs.workload_scenario_id, config.parameters)
            self.assertEqual([(t.task_id, t.arrival_slot, t.input_tokens, t.output_tokens, t.total_tokens) for t in raw.tasks], reference_tasks)
            for field in ("ppa_kwh", "gas_turbine_kwh", "pv_kwh", "bess_discharge_kwh", "bess_charge_kwh", "day_ahead_price_usd_per_kwh"):
                np.testing.assert_array_equal(getattr(energy, field), getattr(reference_energy, field))

    def test_eight_types_keep_original_labels_and_lower_counts_merge_exactly(self):
        original = np.array(self.baseline["known_types"]["counts_by_source_type"])
        for number in (2, 4, 8):
            config = self.cases[number][0]
            payload = json.loads(config.inputs.workload_file.read_text(encoding="utf-8"))
            labels = np.array(payload["known_types"]["counts_by_source_type"])
            expected = original[:, 1:].reshape(len(original), number, 8 // number).sum(axis=2)
            np.testing.assert_array_equal(labels[:, 1:], expected)
        self.assertEqual(json.loads(self.cases[8][0].inputs.workload_file.read_text())["known_types"], self.baseline["known_types"])

    def test_rejects_non_type_parameters_and_uneven_lambdas(self):
        config = self.cases[2][0]
        changed = replace(config, parameters=replace(config.parameters, horizon_slots=48))
        with patch("llm_contract_optimizer.type_count_experiment.load_config", return_value=changed):
            with self.assertRaisesRegex(DataValidationError, "Non-type-count"):
                validate_type_count_suite(MANIFEST, [2])
        changed = replace(config, parameters=replace(config.parameters, type_lambdas=(.2, 2.5)))
        with patch("llm_contract_optimizer.type_count_experiment.load_config", return_value=changed):
            with self.assertRaisesRegex(DataValidationError, "evenly spaced"):
                validate_type_count_suite(MANIFEST, [2])


if __name__ == "__main__":
    unittest.main()
