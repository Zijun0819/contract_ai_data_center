from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from llm_contract_optimizer.exceptions import ConfigurationError

Algorithm = Literal["benders", "direct"]


@dataclass(frozen=True, slots=True)
class InputConfig:
    root: Path
    workload_scenario_id: str
    energy_scenario_id: str
    workload_file: Path | None = None
    immediate_execution_share: float | None = None
    type_distribution: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class StaticParameters:
    horizon_slots: int
    slot_seconds: int
    throughput_tokens_per_second_per_replica: float
    installed_replicas: int
    pue: float
    dynamic_energy_j_per_token: float
    idle_power_w_per_replica: float
    input_tariff_usd_per_million_tokens: float
    output_tariff_usd_per_million_tokens: float
    type_lambdas: tuple[float, ...]
    candidate_deferrals_slots: tuple[int, ...]
    delay_curvature_per_hour: float
    maximum_discount: float
    ppa_cost_usd_per_kwh: float
    gas_turbine_cost_usd_per_kwh: float
    pv_cost_usd_per_kwh: float
    bess_cost_usd_per_kwh_discharged: float

    @property
    def slot_hours(self) -> float:
        return self.slot_seconds / 3600.0

    @property
    def token_capacity_per_slot(self) -> float:
        return (
            self.throughput_tokens_per_second_per_replica
            * self.slot_seconds
            * self.installed_replicas
        )

    @property
    def idle_facility_energy_kwh_per_slot(self) -> float:
        return (
            self.pue
            * self.idle_power_w_per_replica
            * self.slot_seconds
            * self.installed_replicas
            / 3.6e6
        )

    @property
    def facility_kwh_per_token(self) -> float:
        return self.pue * self.dynamic_energy_j_per_token / 3.6e6


@dataclass(frozen=True, slots=True)
class SolverConfig:
    algorithm: Algorithm = "benders"
    max_iterations: int = 500
    absolute_gap_usd: float = 0.01
    relative_gap: float = 1e-6
    master_time_limit_seconds: float | None = 30.0
    subproblem_time_limit_seconds: float | None = 30.0
    total_time_limit_seconds: float | None = 300.0
    master_mip_relative_gap: float = 0.0
    token_scale: float = 1e6
    feasibility_tolerance: float = 1e-7
    add_monotonic_cuts: bool = False
    horizon_policy: str = "finish_within_horizon"


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: Path
    make_plot: bool = True
    write_group_schedule: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    source_path: Path
    inputs: InputConfig
    parameters: StaticParameters
    solver: SolverConfig
    output: OutputConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping.")
    return value


def _resolve(base: Path, raw: Any, name: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(f"'{name}' must be a non-empty path string.")
    path = Path(raw)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path) -> AppConfig:
    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigurationError(f"Configuration file not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(raw, "root")
    input_raw = _mapping(root.get("inputs"), "inputs")
    parameter_raw = _mapping(root.get("parameters"), "parameters")
    solver_raw = _mapping(root.get("solver"), "solver")
    output_raw = _mapping(root.get("output"), "output")
    base = source.parent

    try:
        normalized_parameters = dict(parameter_raw)
        if any(isinstance(x, bool) or int(x) != float(x)
               for x in parameter_raw["candidate_deferrals_slots"]):
            raise ValueError("Candidate deferrals must be integer slot counts.")
        normalized_parameters["type_lambdas"] = tuple(
            float(x) for x in parameter_raw["type_lambdas"]
        )
        normalized_parameters["candidate_deferrals_slots"] = tuple(
            int(x) for x in parameter_raw["candidate_deferrals_slots"]
        )
        parameters = StaticParameters(**normalized_parameters)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"Invalid parameters section: {exc}") from exc

    try:
        inputs = InputConfig(
            immediate_execution_share=input_raw.get("immediate_execution_share"),
            type_distribution=(tuple(float(v) for v in input_raw["type_distribution"])
                               if input_raw.get("type_distribution") is not None else None),
            root=_resolve(base, input_raw["root"], "inputs.root"),
            workload_scenario_id=str(input_raw["workload_scenario_id"]),
            energy_scenario_id=str(input_raw["energy_scenario_id"]),
            workload_file=(
                _resolve(base, input_raw["workload_file"], "inputs.workload_file")
                if input_raw.get("workload_file") is not None else None
            ),
        )
        solver = SolverConfig(**solver_raw)
        output = OutputConfig(
            root=_resolve(base, output_raw["root"], "output.root"),
            make_plot=output_raw.get("make_plot", True),
            write_group_schedule=output_raw.get("write_group_schedule", True),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc

    config = AppConfig(source, inputs, parameters, solver, output)
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    share = config.inputs.immediate_execution_share
    if share is not None and (isinstance(share, bool) or not isinstance(share, (int, float))
                              or not math.isfinite(share) or not 0 <= share <= 1):
        raise ConfigurationError("inputs.immediate_execution_share must be null or a number in [0, 1].")
    p = config.parameters
    s = config.solver
    for name in p.__slots__:
        raw_value = getattr(p, name)
        values = raw_value if isinstance(raw_value, tuple) else (raw_value,)
        if any(not math.isfinite(value) for value in values):
            raise ConfigurationError(f"Parameter {name} must be finite.")
    if p.input_tariff_usd_per_million_tokens < 0 or p.output_tariff_usd_per_million_tokens < 0:
        raise ConfigurationError("Token tariffs must be nonnegative.")
    for name in ("horizon_slots", "slot_seconds", "installed_replicas"):
        value = getattr(p, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"{name} must be an integer.")
    for name in ("max_iterations",):
        value = getattr(s, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"{name} must be an integer.")
    for name in ("absolute_gap_usd", "relative_gap", "master_mip_relative_gap",
                 "token_scale", "feasibility_tolerance", "master_time_limit_seconds",
                 "subproblem_time_limit_seconds", "total_time_limit_seconds"):
        value = getattr(s, name)
        if value is not None and not math.isfinite(value):
            raise ConfigurationError(f"{name} must be finite.")
    if p.horizon_slots <= 0 or p.slot_seconds <= 0:
        raise ConfigurationError("Horizon and slot duration must be positive.")
    if p.installed_replicas <= 0 or p.throughput_tokens_per_second_per_replica <= 0:
        raise ConfigurationError("Replica count and throughput must be positive.")
    if p.pue < 1 or p.dynamic_energy_j_per_token < 0 or p.idle_power_w_per_replica < 0:
        raise ConfigurationError("PUE must be >= 1 and energy/power coefficients nonnegative.")
    if not p.type_lambdas or any(b <= a for a, b in zip(p.type_lambdas, p.type_lambdas[1:])):
        raise ConfigurationError("type_lambdas must be nonempty and strictly increasing.")
    if any(x < 0 for x in p.type_lambdas):
        raise ConfigurationError("type_lambdas must be nonnegative.")
    if (
        not p.candidate_deferrals_slots
        or any(d < 0 for d in p.candidate_deferrals_slots)
        or any(
            b <= a
            for a, b in zip(
                p.candidate_deferrals_slots, p.candidate_deferrals_slots[1:]
            )
        )
    ):
        raise ConfigurationError("candidate_deferrals_slots must be sorted and unique.")
    if p.delay_curvature_per_hour <= 0:
        raise ConfigurationError("delay_curvature_per_hour must be positive.")
    if not 0 <= p.maximum_discount <= 1:
        raise ConfigurationError("maximum_discount must lie in [0, 1].")
    minimum_deferral_hours = p.candidate_deferrals_slots[0] * p.slot_hours
    scaled_minimum_delay = p.delay_curvature_per_hour * minimum_deferral_hours
    minimum_delay_penalty = scaled_minimum_delay / (1.0 + scaled_minimum_delay)
    if minimum_delay_penalty > 0:
        maximum_feasible_lambda = p.maximum_discount / minimum_delay_penalty
        cap_tolerance = 1e-12 * max(1.0, abs(maximum_feasible_lambda))
        if p.type_lambdas[-1] > maximum_feasible_lambda + cap_tolerance:
            raise ConfigurationError(
                "The largest type_lambdas value exceeds the analytical feasibility cap "
                f"{maximum_feasible_lambda:.12g} implied by maximum_discount and the "
                "minimum candidate deferral."
            )
    if any(
        cost < 0
        for cost in (
            p.ppa_cost_usd_per_kwh,
            p.gas_turbine_cost_usd_per_kwh,
            p.pv_cost_usd_per_kwh,
            p.bess_cost_usd_per_kwh_discharged,
        )
    ):
        raise ConfigurationError("Energy costs must be nonnegative.")
    if s.algorithm not in ("benders", "direct"):
        raise ConfigurationError("solver.algorithm must be benders or direct.")
    if s.max_iterations <= 0 or s.token_scale <= 0:
        raise ConfigurationError("Iteration limit and token scale must be positive.")
    if s.absolute_gap_usd < 0 or s.relative_gap < 0:
        raise ConfigurationError("Convergence tolerances must be nonnegative.")
    if not 0 <= s.master_mip_relative_gap < 1:
        raise ConfigurationError("master_mip_relative_gap must lie in [0, 1).")
    if not 1e-10 <= s.feasibility_tolerance <= 1e-4:
        raise ConfigurationError("feasibility_tolerance must lie in [1e-10, 1e-4].")
    for name in ("master_time_limit_seconds", "subproblem_time_limit_seconds", "total_time_limit_seconds"):
        limit = getattr(s, name)
        if limit is not None and limit <= 0:
            raise ConfigurationError(f"{name} must be null or positive.")
    if s.horizon_policy != "finish_within_horizon":
        raise ConfigurationError("Use horizon_policy=finish_within_horizon for divisible scheduling.")
    for name, value in (("add_monotonic_cuts", s.add_monotonic_cuts),
                        ("make_plot", config.output.make_plot),
                        ("write_group_schedule", config.output.write_group_schedule)):
        if not isinstance(value, bool):
            raise ConfigurationError(f"{name} must be a boolean.")
    distribution = config.inputs.type_distribution
    if distribution is not None:
        if (len(distribution) != len(p.type_lambdas) + 1
                or any(not math.isfinite(v) or v < 0 for v in distribution)
                or not math.isclose(sum(distribution), 1., abs_tol=1e-10, rel_tol=0)):
            raise ConfigurationError("type_distribution must contain normalized shares for types 0 through I.")
        if share is not None and not math.isclose(distribution[0], share, abs_tol=1e-10):
            raise ConfigurationError("type_distribution[0] and immediate_execution_share disagree.")
    if not config.inputs.root.is_dir():
        raise ConfigurationError(f"Input root not found: {config.inputs.root}")
