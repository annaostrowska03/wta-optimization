from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence

import pulp
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import broyden1

from .models import WTAInstance, WTASolution, objective_value

# ---------------------------------------------------------------------------
# Numeric constants
# ---------------------------------------------------------------------------
_EPS = 1e-10
_MIN_EXP = 1e-320


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_warm_start(
        instance: WTAInstance,
        warm_start: WTASolution | Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    """Normalizes the warm start to a tuple-of-tuples with binary values."""
    if warm_start is None:
        return None

    assignment = warm_start.assignment if isinstance(warm_start, WTASolution) else warm_start
    if len(assignment) != instance.weapons:
        raise ValueError("warm start assignment row count must match the number of weapons")

    normalized: list[tuple[int, ...]] = []
    for row in assignment:
        if len(row) != instance.targets:
            raise ValueError("each warm start row must match the number of targets")
        normalized.append(tuple(int(bool(v)) for v in row))

    return tuple(normalized)


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
        prod *= q ** mu[i]          # ← ** mu[i] required by eq (3.14)
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


# ---------------------------------------------------------------------------
# Solver 1: WTA_A (upper approximation, section 3) — uses PuLP/CBC
# ---------------------------------------------------------------------------

def solve_exact(
        instance: WTAInstance,
        num_piecewise_segments: int = 20,
        warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
        time_limit_seconds: float | None = None,
) -> WTASolution:
    """
    Approximate MILP (WTA_A from Section 3 of Andersen) solved via PuLP/CBC.

    Uses a convex overapproximation of exp(y) by tangents — gives an upper bound
    on the objective value (does not solve the problem exactly).
    """
    start = perf_counter()
    warm_start_assignment = _normalize_warm_start(instance, warm_start)

    prob = pulp.LpProblem("WTA_A", pulp.LpMinimize)

    x = [
        [pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary) for j in range(instance.targets)]
        for i in range(instance.weapons)
    ]
    z = [pulp.LpVariable(f"z_{j}", lowBound=0.0) for j in range(instance.targets)]
    y = [pulp.LpVariable(f"y_{j}") for j in range(instance.targets)]

    if warm_start_assignment is not None:
        for i in range(instance.weapons):
            for j in range(instance.targets):
                x[i][j].setInitialValue(warm_start_assignment[i][j])

    prob += pulp.lpSum(instance.target_values[j] * z[j] for j in range(instance.targets))

    for i in range(instance.weapons):
        prob += pulp.lpSum(x[i]) <= 1

    for j in range(instance.targets):
        ln_q = [
            math.log(max(1.0 - instance.destruction_probabilities[i][j], _EPS))
            for i in range(instance.weapons)
        ]
        prob += y[j] == pulp.lpSum(x[i][j] * ln_q[i] for i in range(instance.weapons))

        min_y_j = sum(v for v in ln_q if v < 0)
        if min_y_j < 0:
            step = abs(min_y_j) / max(1, num_piecewise_segments - 1)
            for k in range(num_piecewise_segments):
                pk = min_y_j + k * step
                exp_pk = math.exp(pk)
                prob += z[j] >= exp_pk + exp_pk * (y[j] - pk)
        else:
            prob += z[j] >= 1.0

    solver = pulp.PULP_CBC_CMD(
        msg=0,
        warmStart=warm_start_assignment is not None,
        keepFiles=warm_start_assignment is not None,
        timeLimit=time_limit_seconds,
    )
    prob.solve(solver)
    solver_status = pulp.LpStatus.get(prob.status, f"status_{prob.status}")

    assignment = [[0] * instance.targets for _ in range(instance.weapons)]
    for i in range(instance.weapons):
        for j in range(instance.targets):
            if pulp.value(x[i][j]) is not None and pulp.value(x[i][j]) > 0.5:
                assignment[i][j] = 1

    frozen = tuple(tuple(row) for row in assignment)
    runtime = perf_counter() - start

    return WTASolution(
        assignment=frozen,
        objective_value=objective_value(instance, frozen),
        runtime_seconds=runtime,
        method="exact_mip_pulp_linearized_warm_start" if warm_start_assignment is not None
        else "exact_mip_pulp_linearized",
        status=solver_status,
    )


# ---------------------------------------------------------------------------
# Solver 2: Branch-and-Adjust (WTA_LA, Section 6 of Andersen) — uses Gurobi
# ---------------------------------------------------------------------------

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
        raise ValueError("mu must have one value for each weapon type / probability row")
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

    assignment = warm_start.assignment if isinstance(warm_start, WTASolution) else warm_start
    if len(assignment) != instance.weapons:
        raise ValueError("warm start assignment row count must match the number of weapon types")

    normalized: list[tuple[int, ...]] = []
    for i, row in enumerate(assignment):
        if len(row) != instance.targets:
            raise ValueError("each warm start row must match the number of targets")

        normalized_row: list[int] = []
        row_sum = 0
        for value in row:
            rounded = int(round(float(value)))
            if abs(float(value) - rounded) > 1e-7:
                raise ValueError("warm start values for Branch-and-Adjust must be integers")
            if rounded < 0 or rounded > mu[i]:
                raise ValueError("warm start value outside [0, mu_i]")
            normalized_row.append(rounded)
            row_sum += rounded

        if row_sum > mu[i]:
            raise ValueError("warm start uses more weapons of a type than allowed by mu_i")
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
                survived_value *= (1.0 - instance.destruction_probabilities[i][j]) ** count
        total += survived_value
    return total


def solve_branch_and_adjust(
        instance: WTAInstance,
        delta: float = 1e-4,
        warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
        time_limit_seconds: float = 5400.0,
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

    Important implementation detail:
        For x[i,j] with mu[i] > 1, one-hot value indicators are added:
            is_value[i,j,k] = 1 iff x[i,j] = k.
        These indicators make the lazy adjustment cut valid for integer-count
        assignments, not only for binary assignments.
    """
    if delta <= 0:
        raise ValueError("delta must be positive for Branch-and-Adjust")

    start = perf_counter()
    mu_values = _resolve_mu(instance, mu)
    warm_start_assignment = _normalize_integer_warm_start(instance, warm_start, mu_values)

    weapons = instance.weapons
    targets = instance.targets

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model("WTA_BranchAdjust_Integer", env=env) as model:
            model.Params.LazyConstraints = 1
            if time_limit_seconds is not None:
                model.Params.TimeLimit = time_limit_seconds

            # --------------------------------------------------------------
            # Decision variables x[i,j]
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # One-hot value indicators for integer variables with mu_i > 1.
            # Needed to build an exact-match lazy cut for integer assignments.
            # --------------------------------------------------------------
            value_is: dict[tuple[int, int, int], gp.Var] = {}
            for i in range(weapons):
                if mu_values[i] <= 1:
                    continue
                for j in range(targets):
                    indicators = []
                    for k in range(mu_values[i] + 1):
                        ind = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_is_{i}_{j}_{k}",
                        )
                        value_is[i, j, k] = ind
                        indicators.append(ind)

                    model.addConstr(
                        gp.quicksum(indicators) == 1,
                        name=f"x_value_onehot_{i}_{j}",
                    )
                    model.addConstr(
                        x[i, j] == gp.quicksum(k * value_is[i, j, k] for k in range(mu_values[i] + 1)),
                        name=f"x_value_link_{i}_{j}",
                    )

            # --------------------------------------------------------------
            # Delta-based breakpoints per target.
            # b[j,0] = log(prod_i (1-p_ij)^mu_i), final breakpoint = 0.
            # --------------------------------------------------------------
            B: list[list[float]] = []
            for j in range(targets):
                B.append(_compute_breakpoints(instance.destruction_probabilities, mu_values, j, delta))

            # --------------------------------------------------------------
            # Lambda and z variables for compact convex under-approximation.
            # --------------------------------------------------------------
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
                        w_j * math.exp(B[j][t]) if t == 0
                        else w_j * (math.exp(B[j][t]) - delta) if t < last_t
                        else w_j
                    ) * lbda[j, t]
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
                    math.log(max(1.0 - instance.destruction_probabilities[i][j], _EPS)) * x[i, j]
                    for i in range(weapons)
                )
                model.addConstr(lhs == rhs, name=f"log_survival_link_{j}")

            model.ModelSense = GRB.MINIMIZE

            # Branching priorities: prefer variables with greater objective impact.
            for i in range(weapons):
                for j in range(targets):
                    coef = abs(
                        instance.target_values[j]
                        * math.log(max(1.0 - instance.destruction_probabilities[i][j], _EPS))
                    )
                    x[i, j].BranchPriority = int(min(2_000_000_000, coef * 1000))

            # Warm start, preserving integer counts.
            if warm_start_assignment is not None:
                for i in range(weapons):
                    for j in range(targets):
                        start_value = warm_start_assignment[i][j]
                        x[i, j].Start = start_value
                        if mu_values[i] > 1:
                            for k in range(mu_values[i] + 1):
                                value_is[i, j, k].Start = 1.0 if k == start_value else 0.0

            # --------------------------------------------------------------
            # Callback state.
            # --------------------------------------------------------------
            model._x = x
            model._x_keys = x_keys
            model._x_vars = x_vars
            model._z = z
            model._z_vars = z_vars
            model._value_is = value_is
            model._mu_values = mu_values
            model._instance = instance
            model._total_x_positions = weapons * targets
            model._best_true_obj = float("inf")
            model._best_assignment = None
            model._lazy_cuts_added = 0

            def _match_indicator_expr(cb_model, i: int, j: int, value: int):
                """
                Linear expression equal to 1 iff x[i,j] equals `value`, for
                integer-feasible solutions. Uses x itself when mu_i = 1 and
                one-hot indicators when mu_i > 1.
                """
                mu_i = cb_model._mu_values[i]
                if mu_i == 0:
                    return 1.0
                if mu_i == 1:
                    return cb_model._x[i, j] if value == 1 else 1.0 - cb_model._x[i, j]
                return cb_model._value_is[i, j, value]

            def bna_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                inst = cb_model._instance
                mu_cb = cb_model._mu_values
                n_w = inst.weapons
                n_t = inst.targets

                x_values = cb_model.cbGetSolution(cb_model._x_vars)
                z_values = cb_model.cbGetSolution(cb_model._z_vars)

                # Build the integer assignment x* and compute the true nonlinear
                # WTA value: sum_j w_j prod_i (1-p_ij)^x_ij.
                current_assignment = [[0 for _ in range(n_t)] for _ in range(n_w)]
                true_survival = [1.0 for _ in range(n_t)]

                for idx, (i, j) in enumerate(cb_model._x_keys):
                    value = int(round(x_values[idx]))
                    if value < 0:
                        value = 0
                    elif value > mu_cb[i]:
                        value = mu_cb[i]

                    current_assignment[i][j] = value
                    if value > 0:
                        true_survival[j] *= (
                            1.0 - inst.destruction_probabilities[i][j]
                        ) ** value

                true_target_cost = [
                    inst.target_values[j] * true_survival[j]
                    for j in range(n_t)
                ]
                true_obj = sum(true_target_cost)

                if true_obj < cb_model._best_true_obj - 1e-9:
                    cb_model._best_true_obj = true_obj
                    cb_model._best_assignment = tuple(
                        tuple(row) for row in current_assignment
                    )

                # Exact-match distance for integer assignments:
                #   delta_x = 0  iff every x[i,j] equals its current integer value;
                #   delta_x >= 1 otherwise.
                # Then z[j] >= true_cost_j * (1 - delta_x) is active only at x*.
                needs_adjustment = [
                    j for j in range(n_t)
                    if z_values[j] < true_target_cost[j] - 1e-6
                ]
                if not needs_adjustment:
                    return

                match_count = gp.quicksum(
                    _match_indicator_expr(cb_model, i, j, current_assignment[i][j])
                    for i in range(n_w)
                    for j in range(n_t)
                )
                delta_x = cb_model._total_x_positions - match_count

                for j in needs_adjustment:
                    cb_model.cbLazy(
                        cb_model._z[j] >= true_target_cost[j] * (1.0 - delta_x)
                    )
                    cb_model._lazy_cuts_added += 1

            model.optimize(bna_callback)

            runtime = perf_counter() - start

            status_map = {
                GRB.OPTIMAL: "optimal",
                GRB.TIME_LIMIT: "time_limit",
                GRB.INFEASIBLE: "infeasible",
                GRB.INF_OR_UNBD: "inf_or_unbd",
                GRB.UNBOUNDED: "unbounded",
                GRB.INTERRUPTED: "interrupted",
                GRB.NODE_LIMIT: "node_limit",
                GRB.SOLUTION_LIMIT: "solution_limit",
                GRB.ITERATION_LIMIT: "iteration_limit",
            }
            status_str = status_map.get(model.Status, f"status_{model.Status}")

            if model._best_assignment is not None:
                final_assignment = model._best_assignment
                final_obj = model._best_true_obj
            elif model.SolCount > 0:
                # Fallback should rarely be needed, because every integer solution
                # should pass through MIPSOL. It is kept for robustness.
                final_assignment_list = [[0 for _ in range(targets)] for _ in range(weapons)]
                for i in range(weapons):
                    for j in range(targets):
                        final_assignment_list[i][j] = int(round(x[i, j].X))
                final_assignment = tuple(tuple(row) for row in final_assignment_list)
                final_obj = _integer_assignment_objective(instance, final_assignment)
            else:
                final_assignment = tuple(
                    tuple(0 for _ in range(targets)) for _ in range(weapons)
                )
                final_obj = float("inf")

            return WTASolution(
                assignment=final_assignment,
                objective_value=final_obj,
                runtime_seconds=runtime,
                method="branch_and_adjust_gurobi_integer",
                status=status_str,
            )
