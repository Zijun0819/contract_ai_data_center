"""Cumulative fluid scheduling, LP dual certificates, and group recovery."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from llm_contract_optimizer.config import StaticParameters, SolverConfig
from llm_contract_optimizer.domain import EnergyScenario, InnerSolution, OptimalityCut, WorkloadAggregate
from llm_contract_optimizer.exceptions import DataValidationError, SolveError


class FluidInnerSolver:
    def __init__(self, workload: WorkloadAggregate, energy: EnergyScenario,
                 parameters: StaticParameters, solver: SolverConfig):
        self.workload, self.energy, self.p, self.settings = workload, energy, parameters, solver
        self.I, self.S = len(parameters.type_lambdas), parameters.horizon_slots
        self.candidates = parameters.candidate_deferrals_slots
        self.K = len(self.candidates)
        self.scale = solver.token_scale
        self.nv = self.I * self.S
        self.nx = self.nv + self.S
        self.tol = solver.feasibility_tolerance
        mass = np.asarray(workload.participating_tokens_by_type_arrival, dtype=float)
        immediate = np.asarray(workload.opt_out_tokens_by_arrival, dtype=float)
        if mass.shape != (self.I, self.S) or immediate.shape != (self.S,):
            raise DataValidationError("Workload group array dimensions do not match the configuration.")
        if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(immediate)) or np.any(mass < 0) or np.any(immediate < 0):
            raise DataValidationError("Workload must contain finite nonnegative token masses.")
        for field in ("ppa_kwh", "gas_turbine_kwh", "pv_kwh", "bess_discharge_kwh",
                      "bess_charge_kwh", "day_ahead_price_usd_per_kwh"):
            values = np.asarray(getattr(energy, field))
            if values.shape != (self.S,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise DataValidationError(f"Invalid energy input: {field}.")
        if (workload.participating_bill_usd_by_type.shape != (self.I,)
                or not np.all(np.isfinite(workload.participating_bill_usd_by_type))
                or np.any(workload.participating_bill_usd_by_type < 0)
                or not np.isfinite(workload.opt_out_bill_usd) or workload.opt_out_bill_usd < 0):
            raise DataValidationError("Bills must be finite, nonnegative, and indexed by type.")

        self.mass = mass / self.scale
        self.A = np.column_stack((np.zeros(self.I), np.cumsum(self.mass, axis=1)))
        self.M = self.A[:, -1]
        self.r = (parameters.token_capacity_per_slot - immediate) / self.scale
        self.alpha = parameters.facility_kwh_per_token * self.scale
        self.b = (parameters.idle_facility_energy_kwh_per_slot
                  + parameters.facility_kwh_per_token * immediate
                  + energy.bess_charge_kwh - energy.fixed_supply_kwh)
        self.prices = energy.day_ahead_price_usd_per_kwh
        self.q0 = float(self.prices @ np.maximum(self.b, 0.))
        self.c = np.concatenate((np.zeros(self.nv), self.prices))

        prefix = sparse.csr_matrix(np.tril(np.ones((self.S - 1, self.S))))
        P = sparse.kron(sparse.eye(self.I), prefix, format="csr")
        C = sparse.hstack([sparse.eye(self.S)] * self.I, format="csr")
        Z = sparse.csr_matrix((self.I * (self.S - 1), self.S))
        self.A_ub = sparse.vstack((
            sparse.hstack((P, Z)), sparse.hstack((-P, Z)),
            sparse.hstack((C, sparse.csr_matrix((self.S, self.S)))),
            sparse.hstack((self.alpha * C, -sparse.eye(self.S))),
        ), format="csr")
        self.A_eq = sparse.hstack((
            sparse.kron(sparse.eye(self.I), np.ones((1, self.S))),
            sparse.csr_matrix((self.I, self.S))), format="csr")
        self.prefix_count = self.I * (self.S - 1)
        self.b_base = np.concatenate((self.A[:, 1:-1].ravel(), np.zeros(self.prefix_count),
                                      self.r, -self.b))
        indices = np.maximum(np.arange(1, self.S)[:, None] - np.array(self.candidates), 0)
        self.H = np.stack([self.A[i, indices] for i in range(self.I)])
        self.H_matrix = sparse.block_diag([sparse.csr_matrix(h) for h in self.H], format="csr")

    def screen_infeasibility(self) -> str | None:
        bad = np.flatnonzero(self.r < -self.tol)
        if len(bad):
            return f"Immediate workload alone exceeds capacity in slot {int(bad[0]) + 1}."
        remaining = (self.M[:, None] - self.A[:, :-1]).sum(axis=0)
        capacity = np.cumsum(self.r[::-1])[::-1]
        bad = np.flatnonzero(remaining > capacity + self.tol)
        if len(bad):
            u = int(bad[-1])
            return (f"Same-day completion is impossible for arrivals in slots {u+1}..{self.S}: "
                    f"{remaining[u]*self.scale:,.3f} tokens exceed "
                    f"{capacity[u]*self.scale:,.3f} remaining tokens of capacity.")
        return None

    def rhs(self, deadlines: tuple[int, ...]) -> np.ndarray:
        if len(deadlines) != self.I or any(d not in self.candidates for d in deadlines):
            raise DataValidationError("The menu must select one configured deferral per type.")
        due = np.stack([self.A[i, np.maximum(np.arange(1, self.S) - d, 0)]
                        for i, d in enumerate(deadlines)])
        result = self.b_base.copy()
        result[self.prefix_count:2*self.prefix_count] = -due.ravel()
        return result

    def interval_inequalities(self):
        """Necessary and sufficient inequalities, omitting provably redundant rows.

        Each type chooses one candidate. The shortest candidate maximizes its
        mandatory interval mass. If even that maximum fits, the interval holds
        for every menu (including fractional one-hot combinations).
        """
        blocks, bounds = [], []
        cap = np.concatenate(([0.], np.cumsum(self.r)))
        candidates = np.asarray(self.candidates)
        for end in range(1, self.S + 1):
            before = self.A[:, :end].T
            if end == self.S:
                coeff = np.repeat((self.M[None, :] - before)[:, :, None], self.K, axis=2)
            else:
                idx = np.maximum(end - candidates, 0)
                coeff = np.maximum(self.A[:, idx][None, :, :] - before[:, :, None], 0.)
            available = cap[end] - cap[:end]
            keep = coeff[:, :, 0].sum(axis=1) > available
            if np.any(keep):
                blocks.append(sparse.csr_matrix(coeff[keep].reshape(-1, self.I * self.K)))
                bounds.append(available[keep])
        if not blocks:
            return sparse.csr_matrix((0, self.I * self.K)), np.empty(0)
        return sparse.vstack(blocks, format="csr"), np.concatenate(bounds)

    def decode_solution(self, x, deadlines, status="optimal", message="") -> InnerSolution:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.nx,) or not np.all(np.isfinite(x)):
            raise SolveError("Scheduling solver returned an invalid primal vector.")
        rhs = self.rhs(deadlines)
        violation = max(0., float(np.max(self.A_ub @ x - rhs)),
                        float(np.max(np.abs(self.A_eq @ x - self.M))), float(-np.min(x)))
        if violation > 10 * self.tol:
            raise SolveError(f"Scheduling primal residual {violation:g} exceeds tolerance.")
        service = np.maximum(x[:self.nv].reshape(self.I, self.S), 0.) * self.scale
        total = self.workload.opt_out_tokens_by_arrival + service.sum(axis=0)
        facility = self.p.idle_facility_energy_kwh_per_slot + self.p.facility_kwh_per_token * total
        net = facility + self.energy.bess_charge_kwh - self.energy.fixed_supply_kwh
        imports, spill = np.maximum(net, 0.), np.maximum(-net, 0.)
        return InnerSolution(status, float(imports @ self.prices), message, service,
                             total, facility, imports, spill, max_constraint_violation_scaled=violation)

    def _dual_cut(self, result, deadlines, cost):
        n = self.prefix_count
        marg = np.asarray(result.ineqlin.marginals)
        nu = np.maximum(-marg[:n].reshape(self.I, self.S-1), 0.)
        beta = np.maximum(-marg[n:2*n].reshape(self.I, self.S-1), 0.)
        rho = np.maximum(-marg[2*n:2*n+self.S], 0.)
        theta = np.clip(-marg[2*n+self.S:], 0., self.prices)
        # Repair numerical dual feasibility by decreasing the free terminal multiplier.
        suffix = np.column_stack((np.cumsum((nu-beta)[:, ::-1], axis=1)[:, ::-1],
                                  np.zeros(self.I)))
        upper_phi = np.min(suffix + rho + self.alpha*theta, axis=1)
        guard = 1e-12 * max(1., float(np.max(np.abs(upper_phi))))
        phi = np.minimum(np.asarray(result.eqlin.marginals), upper_phi) - guard
        constant = float(self.M @ phi - np.sum(nu*self.A[:, 1:-1])
                         - self.r @ rho + self.b @ theta)
        coeff = np.einsum("is,isk->ik", beta, self.H)
        cut = OptimalityCut(constant, coeff, deadlines, cost)
        if cut.value(deadlines, self.candidates) > cost + max(1e-7, abs(cost)*1e-9):
            raise SolveError("Dual certificate exceeds the verified primal cost.")
        return cut

    def solve(self, deadlines: tuple[int, ...], time_limit: float | None = None) -> InnerSolution:
        from dataclasses import replace
        reason = self.screen_infeasibility()
        if reason:
            return InnerSolution("infeasible", float("inf"), reason)
        options = {"primal_feasibility_tolerance": self.tol, "dual_feasibility_tolerance": self.tol}
        if time_limit is not None:
            options["time_limit"] = time_limit
        result = linprog(self.c, A_ub=self.A_ub, b_ub=self.rhs(deadlines),
                         A_eq=self.A_eq, b_eq=self.M, bounds=(0, None), method="highs",
                         options=options)
        if result.status == 2:
            return InnerSolution("infeasible", float("inf"), str(result.message))
        if result.x is None:
            return InnerSolution("not_solved", float("inf"), str(result.message))
        solution = self.decode_solution(result.x, deadlines,
                                        "optimal" if result.status == 0 else "feasible",
                                        str(result.message))
        if result.status == 0:
            solution = replace(solution, cut=self._dual_cut(
                result, deadlines, solution.objective_day_ahead_cost_usd))
        return solution

    def recover_groups(self, solution: InnerSolution, deadlines: tuple[int, ...]) -> np.ndarray:
        """Return [type, arrival, execution] token masses, all inside this horizon."""
        service = solution.scheduled_tokens_by_type_slot / self.scale
        V = np.column_stack((np.zeros(self.I), np.cumsum(service, axis=1)))
        y = np.maximum(np.minimum(self.A[:, 1:, None], V[:, None, 1:])
                       - np.maximum(self.A[:, :-1, None], V[:, None, :-1]), 0.)
        arrival = np.arange(self.S)[:, None]
        execution = np.arange(self.S)[None, :]
        for i, d in enumerate(deadlines):
            invalid = (execution < arrival) | (execution > np.minimum(arrival+d, self.S-1))
            if np.max(y[i][invalid], initial=0.) > 10*self.tol:
                raise SolveError("Recovered group allocation violates its execution window.")
            # Remove only tolerated floating-point leakage; exported rows always
            # respect the actual arrival/deadline window.
            y[i][invalid] = 0.
        if (np.max(np.abs(y.sum(axis=2)-self.mass)) > 10*self.tol
                or np.max(np.abs(y.sum(axis=1)-service)) > 10*self.tol):
            raise SolveError("Recovered group allocations do not conserve workload.")
        return y * self.scale
