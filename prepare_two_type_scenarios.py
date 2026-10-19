"""Extend N02 with all sampled scenarios without changing the original experiment."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.data import load_energy_scenario, load_workload_aggregate
from llm_contract_optimizer.exceptions import DataValidationError
from llm_contract_optimizer.inner import FluidInnerSolver
from llm_contract_optimizer.preference_audit import audit_rows
from llm_contract_optimizer.reporting import _atomic_text, _csv
from llm_contract_optimizer.sampling import read_sampling_source, source_digest, validate_known_types
from llm_contract_optimizer.scenario_comparison import _workload_scenario_ids
from llm_contract_optimizer.type_count_experiment import _write_source, rebin_fixed_types


def relative(path, root):
    return Path(os.path.relpath(path, root)).as_posix()


def prepare(source_config, target_config):
    original = load_config(source_config)
    target = load_config(target_config)
    if len(original.parameters.type_lambdas) != 8 or len(target.parameters.type_lambdas) != 2:
        raise DataValidationError("Preparation requires eight source types and two target types.")
    if replace(original.parameters, type_lambdas=target.parameters.type_lambdas) != target.parameters:
        raise DataValidationError("Source and target model parameters may differ only in type_lambdas.")
    if original.inputs.workload_file is None or target.inputs.workload_file is None:
        raise DataValidationError("Both configurations must select fixed-label sampling recipes.")
    if (original.inputs.immediate_execution_share != target.inputs.immediate_execution_share
            or target.inputs.type_distribution is not None):
        raise DataValidationError("Preserve the source immediate share without forcing a new distribution.")
    output = target.inputs.workload_file.parent
    source_directory = original.inputs.workload_file.parent
    if output.resolve() == source_directory.resolve():
        raise DataValidationError("Keep the original eight-type recipes in a separate directory.")
    scenario_ids = _workload_scenario_ids(original)
    before = {path: source_digest(path) for path in output.iterdir()
              if path.is_file() and (path.name.endswith(".sample.json") or path.suffix == ".gz")}
    energy = load_energy_scenario(target.inputs.root, target.inputs.energy_scenario_id,
                                  target.parameters.horizon_slots)
    rows, cases, pending = [], [], []
    with TemporaryDirectory(prefix=".prepare_workloads_", dir=output) as temporary:
        staging = Path(temporary)
        for scenario in scenario_ids:
            original_path = source_directory / f"{scenario}.sample.json"
            payload = json.loads(original_path.read_text(encoding="utf-8"))
            raw, counts = read_sampling_source(original_path, scenario, original.parameters)
            old_labels = validate_known_types(payload, counts, 8,
                                             original.inputs.immediate_execution_share)
            destination = output / original_path.name
            if destination.exists():
                recipe = destination
            else:
                source_name = f"{scenario}.tasks.csv.gz"
                if (output / source_name).exists():
                    raise DataValidationError(f"Source file already exists without its recipe: {source_name}")
                source = staging / source_name
                _write_source(source, raw, 2)
                converted = {**payload, "source_file": source_name,
                             "source_sha256": source_digest(source),
                             "known_types": rebin_fixed_types(payload["known_types"], 2)}
                recipe = staging / original_path.name
                recipe.write_text(json.dumps(converted, separators=(",", ":")), encoding="utf-8")
                pending.extend((source, recipe))
            converted = json.loads(recipe.read_text(encoding="utf-8"))
            new_raw, new_counts = read_sampling_source(recipe, scenario, target.parameters)
            labels = validate_known_types(converted, new_counts, 2,
                                          target.inputs.immediate_execution_share)
            expected = np.column_stack((old_labels[:, 0], old_labels[:, 1:5].sum(axis=1),
                                        old_labels[:, 5:9].sum(axis=1)))
            if converted["draws"] != payload["draws"] or not np.array_equal(labels, expected):
                raise DataValidationError(f"Sampled occurrences or exact type merges changed: {scenario}")
            if len(new_raw.tasks) != len(raw.tasks):
                raise DataValidationError(f"Source request count changed: {scenario}")
            fields = ("task_id", "arrival_slot", "input_tokens", "output_tokens", "total_tokens")
            old_requests = [tuple(getattr(task, field) for field in fields) for task in raw.tasks]
            new_requests = [tuple(getattr(task, field) for field in fields) for task in new_raw.tasks]
            probabilities = np.array([task.probabilities for task in raw.tasks])
            expected_probabilities = np.column_stack((probabilities[:, 0],
                                                     probabilities[:, 1:5].sum(axis=1),
                                                     probabilities[:, 5:9].sum(axis=1)))
            actual_probabilities = np.array([task.probabilities for task in new_raw.tasks])
            if (old_requests != new_requests
                    or not np.allclose(actual_probabilities, expected_probabilities, rtol=0, atol=1e-12)):
                raise DataValidationError(f"Source request or merged probabilities changed: {scenario}")
            baseline = load_workload_aggregate(original.inputs.root, scenario, original.parameters,
                                               original_path, original.inputs.immediate_execution_share)
            workload = load_workload_aggregate(target.inputs.root, scenario, target.parameters,
                                               recipe, target.inputs.immediate_execution_share)
            if (workload.task_count != baseline.task_count or workload.total_tokens != baseline.total_tokens
                    or not np.array_equal(workload.opt_out_tokens_by_arrival, baseline.opt_out_tokens_by_arrival)
                    or not np.array_equal(workload.opt_out_bill_usd_by_arrival, baseline.opt_out_bill_usd_by_arrival)
                    or not np.array_equal(workload.participating_tokens_by_type_arrival.sum(axis=0),
                                          baseline.participating_tokens_by_type_arrival.sum(axis=0))
                    or not np.allclose(workload.total_bill_usd_by_arrival, baseline.total_bill_usd_by_arrival,
                                       rtol=0, atol=1e-8)):
                raise DataValidationError(f"Arrival tokens, bills or immediate identities changed: {scenario}")
            reason = FluidInnerSolver(workload, energy, target.parameters, target.solver).screen_infeasibility()
            if reason:
                raise DataValidationError(f"{scenario} capacity screen failed: {reason}")
            rows.extend(audit_rows(new_raw, new_counts, converted, target.parameters))
            source_path = (recipe.parent / converted["source_file"]).resolve()
            cases.append({"scenario_id": scenario, "task_count": workload.task_count,
                          "total_tokens": workload.total_tokens, "type_counts": workload.type_counts.tolist(),
                          "original_recipe": relative(original_path, output),
                          "original_recipe_sha256": source_digest(original_path),
                          "recipe": destination.name, "recipe_sha256": source_digest(recipe),
                          "source_file": relative(source_path, recipe.parent),
                          "source_sha256": source_digest(source_path), "capacity_screen": "passed",
                          "draws_preserved": True, "immediate_identities_preserved": True,
                          "slot_tokens_and_bills_preserved": True, "exact_type_merge_verified": True})
            print(f"Validated {scenario}: {workload.task_count:,} tasks; {workload.total_tokens:,.0f} tokens", flush=True)
        for path in pending:
            with (output / path.name).open("xb") as stream:
                stream.write(path.read_bytes())
    preserved = all(source_digest(path) == digest for path, digest in before.items())
    if not preserved:
        raise DataValidationError("A pre-existing N02 file changed during preparation.")
    _csv(output / "workload_type_distributions.csv", rows)
    manifest = {"schema": "two_type_workload_scenarios_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_config": relative(original.source_path, output),
                "source_config_sha256": source_digest(original.source_path),
                "target_config": relative(target.source_path, output),
                "target_config_sha256": source_digest(target.source_path),
                "type_lambdas": list(target.parameters.type_lambdas),
                "type_rule": "Type 0 unchanged; original types 1-4 merge into type 1, types 5-8 into type 2.",
                "scenario_count": len(cases), "cases": cases,
                "pre_existing_input_files_preserved": preserved}
    _atomic_text(output / "workload_scenario_manifest.json",
                 lambda stream: json.dump(manifest, stream, indent=2))
    print(f"Ready: {len(cases)} two-type workload scenarios in {output}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=PROJECT / "config/default.yaml")
    parser.add_argument("--target-config", type=Path, default=PROJECT / "config/type_counts/N02.yaml")
    args = parser.parse_args()
    raise SystemExit(prepare(args.source_config, args.target_config))
