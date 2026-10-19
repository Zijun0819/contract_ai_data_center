"""Consecutive-window integrity, unchanged controls, and cross-midnight service."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.data import load_workload_aggregate, load_energy_scenario
from llm_contract_optimizer.domain import WorkloadAggregate, EnergyScenario
from llm_contract_optimizer.exceptions import DataValidationError
from llm_contract_optimizer.horizon_data import load_consecutive_workload
from llm_contract_optimizer.horizon_experiment import validate_suite, controlled_settings, _normalize, _row
from llm_contract_optimizer.inner import FluidInnerSolver

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "data/planning_horizons/experiment_manifest.json"
HOURS = [12, 24, 48, 72, 96, 120, 144, 168]


class ConsecutiveHorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, entries = validate_suite(MANIFEST, HOURS)
        cls.cases = {entry["hours"]: (config, workload, energy) for entry, config, workload, energy in entries}

    def test_every_window_is_exact_suffix_in_workload_and_power(self):
        _, longest, power = self.cases[168]
        for hours, (config, workload, energy) in self.cases.items():
            slots = config.parameters.horizon_slots
            np.testing.assert_array_equal(workload.opt_out_tokens_by_arrival, longest.opt_out_tokens_by_arrival[-slots:])
            np.testing.assert_array_equal(workload.participating_tokens_by_type_arrival,
                                          longest.participating_tokens_by_type_arrival[:, -slots:])
            np.testing.assert_array_equal(workload.total_bill_usd_by_arrival, longest.total_bill_usd_by_arrival[-slots:])
            for field in ("ppa_kwh", "gas_turbine_kwh", "pv_kwh", "bess_discharge_kwh",
                          "bess_charge_kwh", "day_ahead_price_usd_per_kwh"):
                np.testing.assert_array_equal(getattr(energy, field), getattr(power, field)[-slots:])
            self.assertEqual(controlled_settings(config), self.manifest["controlled_settings"])

    def test_24h_is_exact_default_and_partial_day_labels_are_inherited(self):
        config, workload, energy = self.cases[24]
        default = load_config(PROJECT / "config/default.yaml")
        original = load_workload_aggregate(default.inputs.root, default.inputs.workload_scenario_id,
            default.parameters, default.inputs.workload_file, default.inputs.immediate_execution_share)
        original_energy = load_energy_scenario(default.inputs.root, default.inputs.energy_scenario_id, 96)
        np.testing.assert_array_equal(workload.type_counts, original.type_counts)
        np.testing.assert_array_equal(workload.participating_tokens_by_type_arrival, original.participating_tokens_by_type_arrival)
        np.testing.assert_array_equal(energy.day_ahead_price_usd_per_kwh, original_energy.day_ahead_price_usd_per_kwh)
        partial = self.cases[12][1]
        self.assertTrue(partial.input_provenance["labels_fixed"])
        # Rebalancing the cropped sample to 50% would change the experiment's fixed occurrences.
        self.assertGreater(abs(partial.type_counts[0] / partial.task_count - .5), .001)

    def test_reject_gapped_days_and_modified_recipe_digest(self):
        config = self.cases[24][0]
        path = config.inputs.workload_file
        original = json.loads(path.read_text(encoding="utf-8"))
        for entry in original["days"]:
            entry["recipe"] = str((path.parent / entry["recipe"]).resolve())
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "window.json"
            gapped = json.loads(json.dumps(original))
            gapped["days"][1]["trace_day"] += 1
            copy.write_text(json.dumps(gapped), encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "consecutive"):
                load_consecutive_workload(copy, config.inputs.workload_scenario_id, config.parameters, .5)
            original["days"][-1]["recipe_sha256"] = "0" * 64
            copy.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "recipe changed"):
                load_consecutive_workload(copy, config.inputs.workload_scenario_id, config.parameters, .5)

    def test_reject_non_horizon_parameter_changes(self):
        config = self.cases[12][0]
        changed = replace(config, parameters=replace(config.parameters, pue=config.parameters.pue + .1))
        with patch("llm_contract_optimizer.horizon_experiment.load_config", return_value=changed):
            with self.assertRaisesRegex(DataValidationError, "Only horizon_slots"):
                validate_suite(MANIFEST, [12])

    def test_reporting_normalizes_totals_without_imputing_failures(self):
        row = _row(12, "proposed_method", self.cases[12][1])
        _normalize(row)
        self.assertIsNone(row["profit_usd_per_hour"])
        row.update(profit_usd=120., discount_expenditure_usd=24., day_ahead_electricity_cost_usd=36.)
        _normalize(row)
        self.assertEqual(row["profit_usd_per_hour"], 10.)
        self.assertEqual(row["discount_expenditure_usd_per_hour"], 2.)
        self.assertEqual(row["day_ahead_electricity_cost_usd_per_hour"], 3.)

    def test_longer_horizon_allows_service_across_midnight(self):
        base = load_config(PROJECT / "config/default.yaml")
        parameters = replace(base.parameters, horizon_slots=48, slot_seconds=3600,
            installed_replicas=1, throughput_tokens_per_second_per_replica=5 / 3600,
            type_lambdas=(.1,), candidate_deferrals_slots=(1,), pue=1., idle_power_w_per_replica=0.)
        mass = np.zeros((1, 48))
        mass[0, 23] = 8.
        workload = WorkloadAggregate("midnight", np.zeros(48), mass, 0., np.array([1.]), np.array([0, 1]))
        energy = EnergyScenario("power", "test", *[np.zeros(48) for _ in range(5)], np.ones(48))
        inner = FluidInnerSolver(workload, energy, parameters, base.solver)
        solution = inner.solve((1,))
        self.assertEqual(solution.status, "optimal")
        self.assertGreater(solution.scheduled_tokens_by_type_slot[0, 24], 0.)
        self.assertAlmostEqual(solution.scheduled_tokens_by_type_slot.sum(), 8.)
        short_workload = replace(workload, opt_out_tokens_by_arrival=np.zeros(24),
                                 participating_tokens_by_type_arrival=mass[:, :24])
        short_energy = EnergyScenario("power", "test", *[np.zeros(24) for _ in range(5)], np.ones(24))
        short = FluidInnerSolver(short_workload, short_energy, replace(parameters, horizon_slots=24), base.solver)
        self.assertIsNotNone(short.screen_infeasibility())


if __name__ == "__main__":
    unittest.main()
