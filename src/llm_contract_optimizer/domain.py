from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnergyScenario:
    scenario_id: str
    date: str
    ppa_kwh: FloatArray
    gas_turbine_kwh: FloatArray
    pv_kwh: FloatArray
    bess_discharge_kwh: FloatArray
    bess_charge_kwh: FloatArray
    day_ahead_price_usd_per_kwh: FloatArray

    @property
    def slots(self) -> int:
        return int(self.ppa_kwh.size)

    @property
    def fixed_supply_kwh(self) -> FloatArray:
        return self.ppa_kwh + self.gas_turbine_kwh + self.pv_kwh + self.bess_discharge_kwh


@dataclass(frozen=True, slots=True)
class WorkloadAggregate:
    """Exact fixed-label group totals; sampled task copies are never expanded."""
    scenario_id: str
    opt_out_tokens_by_arrival: FloatArray
    participating_tokens_by_type_arrival: FloatArray
    opt_out_bill_usd: float
    participating_bill_usd_by_type: FloatArray
    type_counts: NDArray[np.int64]
    source_record_count: int = 0
    input_provenance: dict = field(default_factory=dict)
    opt_out_bill_usd_by_arrival: FloatArray | None = None
    participating_bill_usd_by_type_arrival: FloatArray | None = None

    @property
    def task_count(self) -> int:
        return int(self.type_counts.sum())

    @property
    def total_tokens(self) -> float:
        return float(self.opt_out_tokens_by_arrival.sum() + self.participating_tokens_by_type_arrival.sum())

    @property
    def total_bill_usd(self) -> float:
        return float(self.opt_out_bill_usd + self.participating_bill_usd_by_type.sum())

    @property
    def total_bill_usd_by_arrival(self) -> FloatArray:
        """Undiscounted bill by arrival slot, exact when supplied by the loader."""
        if (self.opt_out_bill_usd_by_arrival is not None
                and self.participating_bill_usd_by_type_arrival is not None):
            return (
                np.asarray(self.opt_out_bill_usd_by_arrival, dtype=float)
                + np.asarray(self.participating_bill_usd_by_type_arrival, dtype=float).sum(axis=0)
            )

        # Backward-compatible fallback for programmatically constructed aggregates.
        # Allocate each group's bill in proportion to that group's token arrivals.
        result = np.zeros_like(self.opt_out_tokens_by_arrival, dtype=float)
        opt_tokens = float(self.opt_out_tokens_by_arrival.sum())
        if opt_tokens > 0:
            result += self.opt_out_bill_usd * self.opt_out_tokens_by_arrival / opt_tokens
        for tokens, bill in zip(
            self.participating_tokens_by_type_arrival,
            self.participating_bill_usd_by_type,
        ):
            total = float(tokens.sum())
            if total > 0:
                result += float(bill) * tokens / total
        return result


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    arrival_slot: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    task_type: int
    source_task_id: str | None = None


@dataclass(frozen=True, slots=True)
class SourceTask:
    """Source weights are used only in explicit input preparation."""
    task_id: str
    arrival_slot: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    probabilities: FloatArray


@dataclass(frozen=True, slots=True)
class TaskPool:
    scenario_id: str
    tasks: tuple[SourceTask, ...]

    @property
    def task_count(self) -> int:
        return len(self.tasks)


@dataclass(frozen=True, slots=True)
class Menu:
    deadlines_slots: tuple[int, ...]
    penalties: tuple[float, ...]
    discounts: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OptimalityCut:
    constant: float
    coefficients: FloatArray
    reference_deadlines: tuple[int, ...]
    reference_cost: float

    def value(self, deadlines: tuple[int, ...], candidates: tuple[int, ...]) -> float:
        return self.constant + sum(self.coefficients[i, candidates.index(d)] for i, d in enumerate(deadlines))


@dataclass(frozen=True, slots=True)
class InnerSolution:
    status: str
    objective_day_ahead_cost_usd: float
    message: str
    scheduled_tokens_by_type_slot: FloatArray | None = None
    total_tokens_by_slot: FloatArray | None = None
    facility_energy_kwh: FloatArray | None = None
    day_ahead_energy_kwh: FloatArray | None = None
    spill_energy_kwh: FloatArray | None = None
    cut: OptimalityCut | None = None
    max_constraint_violation_scaled: float = 0.0


@dataclass(frozen=True, slots=True)
class BendersIteration:
    iteration: int
    deadlines_slots: tuple[int, ...]
    master_status: str
    subproblem_status: str
    lower_bound_loss_usd: float
    incumbent_loss_usd: float | None
    subproblem_cost_usd: float | None
    optimality_cuts: int
    master_seconds: float
    subproblem_seconds: float


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    workload_scenario_id: str
    energy_scenario_id: str
    energy_date: str
    algorithm: str
    menu: Menu
    inner: InnerSolution
    revenue_usd: float
    fixed_energy_cost_usd: float
    day_ahead_cost_usd: float
    profit_usd: float
    loss_lower_bound_usd: float
    loss_upper_bound_usd: float
    lower_bound_profit_usd: float
    upper_bound_profit_usd: float
    absolute_gap_usd: float
    relative_gap: float
    converged: bool
    termination_reason: str
    elapsed_seconds: float
    workload: WorkloadAggregate
    energy: EnergyScenario
    history: tuple[BendersIteration, ...]
    optimality_cuts: int
    feasibility_cuts: int
    output_directory: Path | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One fixed-policy benchmark evaluated on the same workload and energy case."""

    benchmark_id: str
    description: str
    feasible: bool
    message: str
    revenue_usd: float | None
    discount_cost_usd: float
    fixed_energy_cost_usd: float
    day_ahead_cost_usd: float | None
    profit_usd: float | None
    arrival_tokens_by_slot: FloatArray
    executed_tokens_by_slot: FloatArray | None
    facility_energy_kwh: FloatArray | None
    day_ahead_energy_kwh: FloatArray | None
    spill_energy_kwh: FloatArray | None
    shifted_tokens: float | None
    mean_delay_hours: float | None
    maximum_delay_hours: float | None
    metadata: dict = field(default_factory=dict)
