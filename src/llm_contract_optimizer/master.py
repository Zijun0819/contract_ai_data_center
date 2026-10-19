"""Discrete contract master and direct compact MILP benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from llm_contract_optimizer.domain import Menu, OptimalityCut
from llm_contract_optimizer.menu import delay_penalty, menu_from_deadlines
from llm_contract_optimizer.exceptions import SolveError


@dataclass(frozen=True, slots=True)
class MasterSolution:
    status: str
    menu: Menu | None
    lower_bound: float | None
    message: str


class BendersMaster:
    def __init__(self, inner):
        self.inner, self.p = inner, inner.p
        self.I, self.K = inner.I, inner.K
        self.z_count = self.I*self.K
        self.candidates = inner.candidates
        penalties = np.array([delay_penalty(d, self.p) for d in self.candidates])
        bills = inner.workload.participating_bill_usd_by_type
        lambdas = np.array(self.p.type_lambdas)
        h = lambdas*bills
        if self.I > 1:
            h[1:] += np.diff(lambdas)*np.cumsum(bills)[:-1]
        self.menu_cost = (h[:, None]*penalties).ravel()
        rows, cols, vals, lb, ub = [], [], [], [], []
        def add(coeff, lower, upper):
            nz = np.flatnonzero(coeff)
            rows.extend([len(lb)]*len(nz)); cols.extend(nz.tolist()); vals.extend(coeff[nz].tolist())
            lb.append(lower); ub.append(upper)
        for i in range(self.I):
            row = np.zeros((self.I, self.K)); row[i] = 1
            add(row.ravel(), 1., 1.)
        for i in range(self.I-1):
            row = np.zeros((self.I, self.K))
            row[i] = -np.array(self.candidates); row[i+1] = self.candidates
            add(row.ravel(), -np.inf, 0.)
        cap_coeff = np.concatenate(([lambdas[0]], np.diff(lambdas)))[:, None]*penalties
        add(cap_coeff.ravel(), -np.inf, self.p.maximum_discount)
        menu_matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(len(lb), self.z_count)).tocsr()
        interval_matrix, interval_rhs = inner.interval_inequalities()
        self.interval_count = len(interval_rhs)
        self.base_matrix = sparse.vstack((menu_matrix, interval_matrix), format="csr")
        self.base_lb = np.concatenate((lb, np.full(len(interval_rhs), -np.inf)))
        self.base_ub = np.concatenate((ub, interval_rhs))

    def decode_menu(self, z) -> Menu:
        z = np.asarray(z).reshape(self.I, self.K)
        if not np.all(np.isfinite(z)) or np.max(np.abs(z-np.rint(z))) > 1e-5 or np.any(np.abs(z.sum(axis=1)-1)>1e-5):
            raise SolveError("Master returned an invalid one-hot menu.")
        deadlines = tuple(self.candidates[int(i)] for i in z.argmax(axis=1))
        menu = menu_from_deadlines(deadlines, self.p)
        if any(a < b for a, b in zip(deadlines, deadlines[1:])) or max(menu.discounts) > self.p.maximum_discount + 1e-8:
            raise SolveError("Master menu violates monotonicity or the discount cap.")
        return menu

    def solve(self, cuts: list[OptimalityCut], infeasible_menus=(), time_limit=None, mip_gap=0.) -> MasterSolution:
        blocks = [sparse.hstack((self.base_matrix, sparse.csr_matrix((len(self.base_lb), 1))))]
        lower, upper = [self.base_lb], [self.base_ub]
        extra, rhs = [], []
        for cut in cuts:
            extra.append(np.concatenate((cut.coefficients.ravel(), [-1.])))
            rhs.append(-cut.constant)
            if self.inner.settings.add_monotonic_cuts:
                q = max(self.inner.q0, cut.value(cut.reference_deadlines, self.candidates))
                margin = q-self.inner.q0
                chi = (np.array(self.candidates)[None, :] <= np.array(cut.reference_deadlines)[:, None]).astype(float)
                extra.append(np.concatenate(((margin*chi).ravel(), [-1.])))
                rhs.append(margin*self.I-q)
        for deadlines in infeasible_menus:
            chi = (np.array(self.candidates)[None, :] <= np.array(deadlines)[:, None]).astype(float)
            extra.append(np.concatenate((chi.ravel(), [0.])))
            rhs.append(self.I-1.)
        if extra:
            blocks.append(sparse.csr_matrix(np.asarray(extra)))
            lower.append(np.full(len(rhs), -np.inf)); upper.append(np.asarray(rhs))
        options = {"mip_rel_gap": mip_gap}
        if time_limit is not None:
            options["time_limit"] = time_limit
        result = milp(np.concatenate((self.menu_cost, [1.])),
                      integrality=np.concatenate((np.ones(self.z_count), [0.])),
                      bounds=Bounds(np.concatenate((np.zeros(self.z_count), [self.inner.q0])),
                                    np.concatenate((np.ones(self.z_count), [np.inf]))),
                      constraints=LinearConstraint(sparse.vstack(blocks, format="csc"),
                                                   np.concatenate(lower), np.concatenate(upper)),
                      options=options)
        status = {0: "optimal", 1: "limit", 2: "infeasible"}.get(result.status, "error")
        bound = getattr(result, "mip_dual_bound", None)
        bound = float(bound) if bound is not None and np.isfinite(bound) else None
        menu = self.decode_menu(result.x[:self.z_count]) if result.x is not None else None
        return MasterSolution(status, menu, bound, str(result.message))

    def solve_direct(self, time_limit=None, mip_gap=0.):
        inner = self.inner
        coupling = sparse.vstack((
            sparse.csr_matrix((inner.prefix_count, self.z_count)), inner.H_matrix,
            sparse.csr_matrix((2*inner.S, self.z_count))), format="csr")
        matrix = sparse.vstack((
            sparse.hstack((self.base_matrix, sparse.csr_matrix((len(self.base_lb), inner.nx)))),
            sparse.hstack((coupling, inner.A_ub)),
            sparse.hstack((sparse.csr_matrix((inner.I, self.z_count)), inner.A_eq)),
        ), format="csc")
        lower = np.concatenate((self.base_lb, np.full(inner.A_ub.shape[0], -np.inf), inner.M))
        upper = np.concatenate((self.base_ub, inner.b_base, inner.M))
        options = {"mip_rel_gap": mip_gap}
        if time_limit is not None:
            options["time_limit"] = time_limit
        return milp(np.concatenate((self.menu_cost, inner.c)),
                    integrality=np.concatenate((np.ones(self.z_count), np.zeros(inner.nx))),
                    bounds=Bounds(np.zeros(self.z_count+inner.nx),
                                  np.concatenate((np.ones(self.z_count), np.full(inner.nx, np.inf)))),
                    constraints=LinearConstraint(matrix, lower, upper), options=options)
