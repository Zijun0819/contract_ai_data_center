"""Eight-type input refinement and sensitivity-reference reporting."""
from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import numpy as np

from llm_contract_optimizer.exceptions import DataValidationError

def sensitivity_calibration(parameters, energy) -> dict:
    """Report configured sensitivities and a six-hour energy-price reference.

    Assumes equal input/output token weights and grid imports at both endpoints.
    Fixed-source energy and actual scheduling are handled by the optimization model.
    """
    tariff = (parameters.input_tariff_usd_per_million_tokens
              + parameters.output_tariff_usd_per_million_tokens) / 2
    spread = float(np.ptp(energy.day_ahead_price_usd_per_kwh))
    if tariff <= 0 or spread <= 0:
        raise DataValidationError("Sensitivity reference requires positive tariffs and a positive price spread.")
    saving = 1e6 * parameters.facility_kwh_per_token * spread
    reference_discount = saving / tariff
    hours = 6.
    penalty = parameters.delay_curvature_per_hour * hours / (1 + parameters.delay_curvature_per_hour * hours)
    center = reference_discount / penalty
    lambdas = parameters.type_lambdas
    if len(lambdas) != 8:
        raise DataValidationError("Eight-type calibration requires exactly eight configured sensitivities.")
    return {
        "reference_energy_scenario": energy.scenario_id,
        "reference_price_spread_usd_per_kwh": spread,
        "reference_tariff_usd_per_million_tokens": tariff,
        "reference_grid_saving_usd_per_million_tokens": saving,
        "reference_discount_fraction": reference_discount,
        "reference_delay_hours": hours,
        "adjacent_sensitivity_ratios": [b / a for a, b in zip(lambdas, lambdas[1:])],
        "maximum_sensitivity": lambdas[-1],
        "maximum_sensitivity_basis": (
            "Cap-safe linear grid selected from the default scenario's observed "
            "deferral-level/discount-spread frontier. The energy reference is comparison-only."
        ),
        "energy_reference_maximum_sensitivity": round(16 * center, 8),
        "sensitivity_multipliers": [value / center for value in lambdas],
        "type_lambdas": list(lambdas),
        "interpretation": (
            "Exogenous contract-design calibration, not estimated preferences or an optimality result."
        ),
    }


def refine_probability_table(source: Path, destination: Path) -> int:
    """Split each original participating probability equally between two adjacent types.

    Keep all source fields other than participating probabilities unchanged, including
    immediate probabilities. Write a separate deterministic gzip source table.
    """
    if source.resolve() == destination.resolve():
        raise DataValidationError("Type refinement must preserve the original source table.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    original = [f"p_type_{i}" for i in range(1, 5)]
    count = 0
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as inp:
        reader = csv.DictReader(inp)
        columns = reader.fieldnames or []
        if {name for name in columns if name.startswith("p_type_")} != set(original):
            raise DataValidationError("Eight-type refinement requires exactly four source participating types.")
        names = [name for name in columns if name not in original] + [f"p_type_{i}" for i in range(1, 9)]
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=names)
                    writer.writeheader()
                    for row in reader:
                        probabilities = [float(row.pop(name)) for name in original]
                        if any(not np.isfinite(p) or p < 0 for p in probabilities):
                            raise DataValidationError("Invalid source participating probabilities.")
                        for i, probability in enumerate(probabilities):
                            # repr retains round-trip precision, including source values beyond eight digits.
                            row[f"p_type_{2*i+1}"] = repr(probability / 2)
                            row[f"p_type_{2*i+2}"] = repr(probability / 2)
                        writer.writerow(row)
                        count += 1
    temporary.replace(destination)
    return count
