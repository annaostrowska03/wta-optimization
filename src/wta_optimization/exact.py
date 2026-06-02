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


def _compute_integer_solution(
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


def _add_tangent_cuts(
    cb_model,
    z_values: list[float],
    true_target_cost: list[float],
    true_log_survival: list[float],
) -> int:
    """Add tangent lazy cuts for targets where z underestimates the true cost.

    Returns the number of cuts added.
    """
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


def _finalize_solution(
    model,
    instance: WTAInstance,
    x: dict[tuple[int, int], gp.Var],
) -> tuple[tuple[tuple[int, ...], ...], float]:
    """Extract the best assignment and objective after model.optimize().

    Falls back to model variables if callback tracking missed a solution.
    Returns all-zeros with inf objective if no feasible solution was found.
    """
    if model._best_assignment is not None:
        return model._best_assignment, model._best_true_obj
    if model.SolCount > 0:
        assignment = tuple(
            tuple(int(round(x[i, j].X)) for j in range(instance.targets))
            for i in range(instance.weapons)
        )
        return assignment, _integer_assignment_objective(instance, assignment)
    empty = tuple(
        tuple(0 for _ in range(instance.targets)) for _ in range(instance.weapons)
    )
    return empty, float("inf")


def _compute_breakpoints(
    destruction_probs: tuple[tuple[float, ...], ...],
    mu: list[int],
    j: int,
    delta: float,
) -> list[float]:
    """
    Iterative delta-based procedure from Section 5 of Andersen (equations 5.2-5.4).

    Returns a list of breakpoints b[j,0], b[j,1], ..., b[j,T] = 0
    guaranteeing a maximum approximation error of exactly delta.

    Parameters
    ----------
    destruction_probs : matrix p[i][j]
    mu               : list mu[i] — availability of each weapon type
    j                : target index
    delta            : maximum allowed approximation error
    """
    weapons = len(destruction_probs)

    # Equation (3.14): smallest breakpoint = ln(product of (1-p_ij)^mu_i for i)
    # Corresponds to the scenario "all weapons fired at target j".
    prod = 1.0
    for i in range(weapons):
        q = max(1.0 - destruction_probs[i][j], _EPS)
        prod *= q ** mu[i]  # ← ** mu[i] required by eq (3.14)
    prod = max(prod, _MIN_EXP)

    b_t = math.log(prod)
    b_list: list[float] = [b_t]

    # Iteratively add breakpoints until we reach 0
    while b_t < -_EPS:
        current_b = b_t  # remember before entering the closure

        # Step 1 (eq. 5.3):
        #   Find x > current_b where the tangent drawn at current_b reaches an error of delta.
        #   Tangent at current_b: L(x) = exp(current_b) + exp(current_b)*(x - current_b)
        #   Condition: exp(x) - L(x) = delta
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

            # Step 2 (eq. 5.4):
            #   Find the tangent point x* > current_b for the next segment.
            #   Condition: exp(x) - (exp(current_b) + exp(x)*(x - current_b)) + delta = 0
            def f2(x_val, _b=current_b):
                xf = float(x_val)
                return math.exp(xf) - (math.exp(_b) + math.exp(xf) * (xf - _b)) + delta

            try:
                x2 = float(broyden1(f2, 1.0, f_tol=1e-15))
            except Exception:
                x2 = 0.0

            b_t = x2 if x2 < 0 else 0.0

            # Add to list only when we reach 0 (last breakpoint)
            if abs(b_t) < _EPS:
                b_list.append(0.0)

    return b_list

def _resolve_mu(
    instance: WTAInstance,
    mu: Sequence[int] | None = None,
) -> list[int]:
    """
    Resolve weapon-type availabilities for the full Andersen WTA model.

    The original WTAInstance used in this project stores only a probability row
    per weapon/weapon type, but it does not store the Andersen parameter mu_i.
    Therefore Branch-and-Adjust accepts mu explicitly. For backwards
    compatibility, if mu is not provided and the instance has no mu-like
    attribute, the solver uses mu_i = 1 for every row.
    """
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


def _normalize_integer_warm_start(
    instance: WTAInstance,
    warm_start: WTASolution | Sequence[Sequence[int]] | None,
    mu: Sequence[int],
) -> tuple[tuple[int, ...], ...] | None:
    """
    Normalize a warm start for the full integer WTA model.

    Unlike _normalize_warm_start(), this function deliberately preserves counts
    greater than 1. It is used only by Branch-and-Adjust.
    """
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


def _integer_assignment_objective(
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


def solve_branch_and_adjust(
    instance: WTAInstance,
    delta: float = 1e-4,
    warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
    time_limit_seconds: float | None = None,
    mu: Sequence[int] | None = None,
) -> WTASolution:
    """
    Branch-and-Adjust for the full integer WTA model from Andersen et al. (2022).

    This implementation supports x[i,j] > 1, i.e. assigning multiple weapons of
    the same type to the same target. The Andersen parameter mu_i is resolved as:

        1. the explicit `mu=` argument, if provided;
        2. instance.mu / instance.weapon_availabilities / instance.availabilities,
           if such an attribute exists;
        3. [1, ..., 1] for backwards compatibility with the original project.

    Mathematical model:
        x[i,j] integer in {0, ..., mu[i]}
        sum_j x[i,j] <= mu[i]
        sum_t lambda[j,t] = 1
        sum_t b[j,t] lambda[j,t] = sum_i log(1-p[i,j]) x[i,j]
        z[j] >= compact under-approximation of w[j] * exp(log-survival[j])
        min sum_j z[j]

    Adjustment step:
        Whenever Gurobi finds an integer assignment x*, the callback computes
        the true nonlinear WTA value. If z[j] underestimates the true survived
        value of target j, a lazy cut is added that is active only for exactly
        this integer assignment x*.

    Implementation detail:
        The model uses integer x[i,j] directly (without one-hot expansion).
        Lazy cuts are generated at MIPSOL points from the true nonlinear cost
        of the incumbent assignment.
    """
    if delta <= 0:
        raise ValueError("delta must be positive for Branch-and-Adjust")

    start = perf_counter()
    mu_values = _resolve_mu(instance, mu)
    warm_start_assignment = _normalize_integer_warm_start(
        instance, warm_start, mu_values
    )

    weapons = instance.weapons
    targets = instance.targets

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model("WTA_BranchAdjust_Integer", env=env) as model:
            model.Params.LazyConstraints = 1
            if time_limit_seconds is not None:
                model.Params.TimeLimit = time_limit_seconds

            # Decision variables x[i,j]
            x: dict[tuple[int, int], gp.Var] = {}
            x_keys: list[tuple[int, int]] = []
            x_vars: list[gp.Var] = []

            for i in range(weapons):
                for j in range(targets):
                    if mu_values[i] == 1:
                        var = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                    else:
                        var = model.addVar(
                            vtype=GRB.INTEGER,
                            lb=0.0,
                            ub=float(mu_values[i]),
                            name=f"x_{i}_{j}",
                        )
                    x[i, j] = var
                    x_keys.append((i, j))
                    x_vars.append(var)

            # Weapon availability constraints: sum_j x[i,j] <= mu_i
            for i in range(weapons):
                model.addConstr(
                    gp.quicksum(x[i, j] for j in range(targets)) <= mu_values[i],
                    name=f"weapon_availability_{i}",
                )

            # Delta-based breakpoints per target.
            # b[j,0] = log(prod_i (1-p_ij)^mu_i), final breakpoint = 0.
            B: list[list[float]] = []
            for j in range(targets):
                B.append(
                    _compute_breakpoints(
                        instance.destruction_probabilities, mu_values, j, delta
                    )
                )

            # Lambda and z variables for compact convex under-approximation.
            lbda: dict[tuple[int, int], gp.Var] = {}
            for j in range(targets):
                for t in range(len(B[j])):
                    lbda[j, t] = model.addVar(lb=0.0, name=f"lbda_{j}_{t}")

            z: dict[int, gp.Var] = {}
            z_vars: list[gp.Var] = []
            for j in range(targets):
                z[j] = model.addVar(lb=0.0, obj=1.0, name=f"z_{j}")
                z_vars.append(z[j])

                w_j = instance.target_values[j]
                last_t = len(B[j]) - 1

                # Coefficients from Andersen's compact under-approximation:
                # first: w_j exp(b_0), middle: w_j(exp(b_t)-delta), last: w_j.
                under_approx = gp.quicksum(
                    (
                        w_j * math.exp(B[j][t])
                        if t == 0
                        else w_j * (math.exp(B[j][t]) - delta) if t < last_t else w_j
                    )
                    * lbda[j, t]
                    for t in range(len(B[j]))
                )
                model.addConstr(z[j] >= under_approx, name=f"z_underapprox_link_{j}")

            # Lambda convexity and log-survival linking constraints.
            for j in range(targets):
                model.addConstr(
                    gp.quicksum(lbda[j, t] for t in range(len(B[j]))) == 1.0,
                    name=f"lambda_sum_{j}",
                )

                lhs = gp.quicksum(B[j][t] * lbda[j, t] for t in range(len(B[j])))
                rhs = gp.quicksum(
                    math.log(max(1.0 - instance.destruction_probabilities[i][j], _EPS))
                    * x[i, j]
                    for i in range(weapons)
                )
                model.addConstr(lhs == rhs, name=f"log_survival_link_{j}")

            model.ModelSense = GRB.MINIMIZE

            # Branching priorities: prefer variables with greater objective impact.
            for i in range(weapons):
                for j in range(targets):
                    coef = abs(
                        instance.target_values[j]
                        * math.log(
                            max(1.0 - instance.destruction_probabilities[i][j], _EPS)
                        )
                    )
                    x[i, j].BranchPriority = int(min(2_000_000_000, coef * 1000))

            # Warm start, preserving integer counts.
            if warm_start_assignment is not None:
                for i in range(weapons):
                    for j in range(targets):
                        x[i, j].Start = warm_start_assignment[i][j]

            # Callback state.
            model._x = x
            model._x_keys = x_keys
            model._x_vars = x_vars
            model._z = z
            model._z_vars = z_vars
            model._mu_values = mu_values
            model._instance = instance
            model._best_true_obj = float("inf")
            model._best_assignment = None
            model._lazy_cuts_added = 0

            def bna_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                x_values = cb_model.cbGetSolution(cb_model._x_vars)
                z_values = cb_model.cbGetSolution(cb_model._z_vars)

                current_assignment, _, true_log_survival, true_target_cost, true_obj = (
                    _compute_integer_solution(
                        cb_model._x_keys,
                        x_values,
                        cb_model._mu_values,
                        cb_model._instance,
                    )
                )

                if true_obj < cb_model._best_true_obj - 1e-9:
                    cb_model._best_true_obj = true_obj
                    cb_model._best_assignment = tuple(
                        tuple(row) for row in current_assignment
                    )

                cb_model._lazy_cuts_added += _add_tangent_cuts(
                    cb_model, z_values, true_target_cost, true_log_survival
                )

            model.optimize(bna_callback)

            runtime = perf_counter() - start
            status_str = _GRB_STATUS_MAP.get(model.Status, f"status_{model.Status}")
            final_assignment, final_obj = _finalize_solution(model, instance, x)

            return WTASolution(
                assignment=final_assignment,
                objective_value=final_obj,
                runtime_seconds=runtime,
                method="branch_and_adjust_gurobi_integer",
                status=status_str,
            )
