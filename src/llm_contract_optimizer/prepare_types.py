"""Reproduce the project's eight-type source and recipe provenance."""
import argparse
import json
import os
from pathlib import Path

from llm_contract_optimizer.config import load_config
from llm_contract_optimizer.data import load_energy_scenario
from llm_contract_optimizer.reporting import _atomic_text
from llm_contract_optimizer.sampling import source_digest
from llm_contract_optimizer.type_calibration import (
    refine_probability_table, sensitivity_calibration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--source-file", type=Path,
        default=project / "data/representative_tasks_raw.csv.gz",
        help="Four-type probability table to refine.",
    )
    parser.add_argument(
        "--destination-file", type=Path,
        default=project / "data/representative_tasks_eight_types.csv.gz",
        help="Eight-type probability table to write.",
    )
    parser.add_argument("--report-file", type=Path, help="Optional calibration report outside the input package.")
    args = parser.parse_args()
    baseline = load_config(project / "config/default.yaml")
    energy = load_energy_scenario(baseline.inputs.root, baseline.inputs.energy_scenario_id,
                                  baseline.parameters.horizon_slots)
    calibration = sensitivity_calibration(baseline.parameters, energy)
    data = project / "data"
    destination = args.destination_file.resolve()
    source = args.source_file.resolve()
    count = refine_probability_table(source, destination)
    digest = source_digest(destination)
    calibration.update({"source_file": str(source), "source_sha256": source_digest(source),
                        "refined_file": str(destination), "refined_sha256": digest,
                        "source_task_count": count,
                        "probability_rule": "p_new[2*i-1] = p_new[2*i] = p_source[i]/2; p_opt_out unchanged"})
    recipe_dir = data
    for path in sorted(recipe_dir.glob("*.sample.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Preserve occurrence counts, arrivals, task identities and token targets exactly.
        if payload.get("source_sha256") != digest:
            # Existing labels depend on source identities and counts. Require reassignment
            # whenever preparation changes the source instead of silently retaining them.
            payload.pop("known_types", None)
        payload.update(source_file=Path(os.path.relpath(destination, path.parent)).as_posix(), source_sha256=digest)
        _atomic_text(path, lambda f: json.dump(payload, f, separators=(",", ":")))
    if args.report_file is not None:
        _atomic_text(args.report_file, lambda f: json.dump(calibration, f, indent=2))
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
