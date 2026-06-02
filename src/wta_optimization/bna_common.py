from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence

import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import broyden1

from .models import WTAInstance, WTASolution

_EPS = 1e-10
_MIN_EXP = 1e-320

_GRB_STATUS_MAP: dict[int, str] = {
    GRB.OPTIMAL: "optimal",
    GRB.TIME_LIMIT: "time_limit",
    GRB.INFEASIBLE: "infeasible",
    GRB.INF_OR_UNBD: "inf_or_unbd",
    GRB.UNBOUNDED: "unbounded",
    GRB.INTERRUPTED: "interrupted",
    GRB.NODE_LIMIT: "node_limit",
    GRB.SOLUTION_LIMIT: "solution_limit",
    GRB.ITERATION_LIMIT: "iteration_limit",
    GRB.MEM_LIMIT: "mem_limit",
}


def compute_integer_solution(
    x_keys: list[tuple[int, int]],
    x_values: list[float],
    mu_values: list[int],
    instance: WTAInstance,
) -> tuple[list[list[int]], list[float], list[float], list[float], float]:
    """Build x*, true survival, log-survival, per-target costs, and total objective."""
    n_w, n_t = instance.weapons, instance.targets
    current_assignment = [[0] * n_t for _ in range(n_w)]
    true_survival = [1.0] * n_t
    true_log_survival = [0.0] * n_t
    for idx, (i, j) in enumerate(x_keys):
        value = max(0, min(mu_values[i], int(round(x_values[idx]))))
        current_assignment[i][j] = value
        if value > 0:
            p_ij = instance.destruction_probabilities[i][j]
            log_q = math.log(max(1.0 - p_ij, _EPS))
            true_survival[j] *= (1.0 - p_ij) ** value
            true_log_survival[j] += value * log_q
    true_target_cost = [
        instance.target_values[j] * true_survival[j] for j in range(n_t)
    ]
    return (
        current_assignment,
        true_survival,
        true_log_survival,
        true_target_cost,
        sum(true_target_cost),
    )


def add_tangent_cuts(
    cb_model,
    z_values: list[float],
    true_target_cost: list[float],
    true_log_survival: list[float],
) -> int:
    """Add tangent lazy cuts where z underestimates true per-target cost."""
    inst = cb_model._instance
    n_t = inst.targets
    needs_adjustment = [
        j for j in range(n_t) if z_values[j] < true_target_cost[j] - 1e-6
    ]
    if not needs_adjustment:
        return 0
    if len(needs_adjustment) > 50:
        needs_adjustment = sorted(
            needs_adjustment,
            key=lambda jj: true_target_cost[jj] - z_values[jj],
            reverse=True,
        )[:50]
    for j in needs_adjustment:
        c_j = true_target_cost[j]
        y_j_star = true_log_survival[j]
        cb_model.cbLazy(
            cb_model._z[j]
            >= c_j * (1.0 - y_j_star)
            + gp.quicksum(
                c_j
                * math.log(max(1.0 - inst.destruction_probabilities[i][j], _EPS))
                * cb_model._x[i, j]
                for i in range(inst.weapons)
            )
        )
    return len(needs_adjustment)


def integer_assignment_objective(
    instance: WTAInstance,
    assignment: Sequence[Sequence[int]],
) -> float:
    """True nonlinear WTA objective for integer assignments x_ij."""
    total = 0.0
    for j, target_value in enumerate(instance.target_values):
        survived_value = float(target_value)
        for i in range(instance.weapons):
            count = int(assignment[i][j])
            if count > 0:
                survived_value *= (
                    1.0 - instance.destruction_probabilities[i][j]
                ) ** count
        total += survived_value
    return total


def finalize_solution(
    model,
    instance: WTAInstance,
    x: dict[tuple[int, int], gp.Var],
) -> tuple[tuple[tuple[int, ...], ...], float]:
    """Extract best assignment and objective after optimize()."""
    if model._best_assignment is not None:
        return model._best_assignment, model._best_true_obj
    if model.SolCount > 0:
        assignment = tuple(
            tuple(int(round(x[i, j].X)) for j in range(instance.targets))
            for i in range(instance.weapons)
        )
        return assignment, integer_assignment_objective(instance, assignment)
    empty = tuple(
        tuple(0 for _ in range(instance.targets)) for _ in range(instance.weapons)
    )
    return empty, float("inf")


def compute_breakpoints(
    destruction_probs: tuple[tuple[float, ...], ...],
    mu: list[int],
    j: int,
    delta: float,
) -> list[float]:
    """Delta-based breakpoint construction from Andersen et al."""
    weapons = len(destruction_probs)

    prod = 1.0
    for i in range(weapons):
        q = max(1.0 - destruction_probs[i][j], _EPS)
        prod *= q ** mu[i]
    prod = max(prod, _MIN_EXP)

    b_t = math.log(prod)
    b_list: list[float] = [b_t]

    while b_t < -_EPS:
        current_b = b_t

        def f1(x_val, _b=current_b):
            xf = float(x_val)
            return math.exp(xf) - (math.exp(_b) + math.exp(_b) * (xf - _b)) - delta

        try:
            x1 = float(broyden1(f1, 1.0, f_tol=1e-15))
        except Exception:
            x1 = 0.0

        b_next = x1 if x1 < 0 else 0.0
        b_list.append(b_next)
        b_t = b_next

        if b_t < -_EPS:
            current_b = b_t

            def f2(x_val, _b=current_b):
                xf = float(x_val)
                return math.exp(xf) - (math.exp(_b) + math.exp(xf) * (xf - _b)) + delta

            try:
                x2 = float(broyden1(f2, 1.0, f_tol=1e-15))
            except Exception:
                x2 = 0.0

            b_t = x2 if x2 < 0 else 0.0
            if abs(b_t) < _EPS:
                b_list.append(0.0)

    return b_list


def resolve_mu(
    instance: WTAInstance,
    mu: Sequence[int] | None = None,
) -> list[int]:
    """Resolve weapon availabilities mu_i for the integer WTA model."""
    raw_mu = mu
    if raw_mu is None:
        raw_mu = getattr(instance, "mu", None)
    if raw_mu is None:
        raw_mu = getattr(instance, "weapon_availabilities", None)
    if raw_mu is None:
        raw_mu = getattr(instance, "availabilities", None)
    if raw_mu is None:
        raw_mu = [1] * instance.weapons

    resolved = [int(v) for v in raw_mu]
    if len(resolved) != instance.weapons:
        raise ValueError(
            "mu must have one value for each weapon type / probability row"
        )
    if any(v < 0 for v in resolved):
        raise ValueError("all mu values must be non-negative integers")
    return resolved


def normalize_integer_warm_start(
    instance: WTAInstance,
    warm_start: WTASolution | Sequence[Sequence[int]] | None,
    mu: Sequence[int],
) -> tuple[tuple[int, ...], ...] | None:
    """Normalize and validate integer warm start assignment."""
    if warm_start is None:
        return None

    assignment = (
        warm_start.assignment if isinstance(warm_start, WTASolution) else warm_start
    )
    if len(assignment) != instance.weapons:
        raise ValueError(
            "warm start assignment row count must match the number of weapon types"
        )

    normalized: list[tuple[int, ...]] = []
    for i, row in enumerate(assignment):
        if len(row) != instance.targets:
            raise ValueError("each warm start row must match the number of targets")

        normalized_row: list[int] = []
        row_sum = 0
        for value in row:
            rounded = int(round(float(value)))
            if abs(float(value) - rounded) > 1e-7:
                raise ValueError(
                    "warm start values for Branch-and-Adjust must be integers"
                )
            if rounded < 0 or rounded > mu[i]:
                raise ValueError("warm start value outside [0, mu_i]")
            normalized_row.append(rounded)
            row_sum += rounded

        if row_sum > mu[i]:
            raise ValueError(
                "warm start uses more weapons of a type than allowed by mu_i"
            )
        normalized.append(tuple(normalized_row))

    return tuple(normalized)
