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

def solve_branch_and_adjust(
        instance: WTAInstance,
        delta: float = 1e-4,
        warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
        time_limit_seconds: float = 5400.0,
) -> WTASolution:
    """
    Exact Branch-and-Adjust algorithm (Section 6 of Andersen et al. 2022).

    Uses a compact convex underapproximation of the WTA objective function (WTA_LA)
    as a lower bound in the B&B tree. Upon finding an integer solution,
    it calculates its TRUE nonlinear value and updates the upper bound.

    Key components:
    - Lambda variables with breakpoints from the delta-based procedure (eq. 5.2-5.4)
    - Lambda objective coefficients from eq. (5.5) / get_lbda_obj from original CPLEX
    - MIPSOL Callback: calculates true WTA, adds lazy cuts forcing
      the real z[j] value for the assignment found
    - BranchPriority: proxy for the branching strategy from Algorithm 1 (line 13)

    Parameters
    ----------
    instance            : WTA instance
    delta               : approximation precision parameter (smaller = more exact, slower)
    warm_start          : optional starting assignment
    time_limit_seconds  : time limit in seconds
    """
    start = perf_counter()
    warm_start_assignment = _normalize_warm_start(instance, warm_start)

    # mu[i] = 1 for all weapons (by default; extend WTAInstance if needed)
    mu: list[int] = getattr(instance, "mu", [1] * instance.weapons)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model("WTA_BranchAdjust", env=env) as model:
            model.Params.LazyConstraints = 1
            model.Params.TimeLimit = time_limit_seconds

            weapons = instance.weapons
            targets = instance.targets

            # ------------------------------------------------------------------
            # Decision variables x[i,j]
            # Binary for mu=1; change to INTEGER with ub=mu[i] when mu > 1
            # ------------------------------------------------------------------
            x: dict[tuple[int, int], gp.Var] = {}
            for i in range(weapons):
                for j in range(targets):
                    x[i, j] = model.addVar(
                        vtype=GRB.BINARY,   # GRB.INTEGER + ub=mu[i] when mu[i]>1
                        name=f"x_{i}_{j}",
                    )

            # Each type of weapon used at most mu[i] times in total (eq. 2.3)
            for i in range(weapons):
                model.addConstr(
                    gp.quicksum(x[i, j] for j in range(targets)) <= mu[i],
                    name=f"weapon_{i}",
                )

            # ------------------------------------------------------------------
            # Delta-based breakpoints (procedure from Section 5, eq. 3.14 + 5.2-5.4)
            # ------------------------------------------------------------------
            B: list[list[float]] = []
            for j in range(targets):
                b_j = _compute_breakpoints(instance.destruction_probabilities, mu, j, delta)
                B.append(b_j)

            # ------------------------------------------------------------------
            # Lambda variables lambda[j,t] — convex combination on breakpoints
            # Objective coefficients per get_lbda_obj (eq. 5.5 + Section 5):
            #   t == 0         : w_j * exp(b[j,0])
            #   0 < t < last   : w_j * (exp(b[j,t]) - delta)   ← correction -delta
            #   t == last      : w_j * 1.0                      ← exp(0) = 1
            # ------------------------------------------------------------------
            lbda: dict[tuple[int, int], gp.Var] = {}
            for j in range(targets):
                w_j = instance.target_values[j]
                n_t = len(B[j])
                for t, b_val in enumerate(B[j]):
                    if t == 0:
                        obj_coef = w_j * math.exp(b_val)
                    elif t < n_t - 1:
                        obj_coef = w_j * (math.exp(b_val) - delta)
                    else:
                        obj_coef = w_j * 1.0
                    lbda[j, t] = model.addVar(lb=0.0, obj=obj_coef, name=f"lbda_{j}_{t}")

            model.ModelSense = GRB.MINIMIZE

            # ------------------------------------------------------------------
            # Constraints for lambda variables
            # ------------------------------------------------------------------
            for j in range(targets):
                n_t = len(B[j])

                # (5.12): sum_t lambda[j,t] = 1
                model.addConstr(
                    gp.quicksum(lbda[j, t] for t in range(n_t)) == 1.0,
                    name=f"sc3_{j}",
                )

                # (5.11): sum_t lambda[j,t]*b[j,t] = sum_i ln(1-p[i,j]) * x[i,j]
                lhs = gp.quicksum(B[j][t] * lbda[j, t] for t in range(n_t))
                rhs = gp.quicksum(
                    math.log(max(1.0 - instance.destruction_probabilities[i][j], _EPS)) * x[i, j]
                    for i in range(weapons)
                )
                model.addConstr(lhs == rhs, name=f"sc2_{j}")

            # ------------------------------------------------------------------
            # Auxiliary variable z[j] = approximated weighted cost of target j
            # Needed in the callback to compare with true_cost_j = w_j * true_z_j
            # ------------------------------------------------------------------
            z: dict[int, gp.Var] = {}
            for j in range(targets):
                z[j] = model.addVar(lb=0.0, name=f"z_{j}")
                # z[j] >= sum of coefficients * lambda (lower link to approximation)
                under_approx = gp.quicksum(
                    (
                        instance.target_values[j] * math.exp(B[j][t]) if t == 0
                        else instance.target_values[j] * (math.exp(B[j][t]) - delta) if t < len(B[j]) - 1
                        else instance.target_values[j] * 1.0
                    ) * lbda[j, t]
                    for t in range(len(B[j]))
                )
                model.addConstr(z[j] >= under_approx, name=f"z_link_{j}")

            # ------------------------------------------------------------------
            # Branching priorities (proxy for Algorithm 1, line 13)
            # Variables with larger |w_j * ln(1-p_ij)| branched first
            # ------------------------------------------------------------------
            for i in range(weapons):
                for j in range(targets):
                    coef = instance.target_values[j] * math.log(
                        max(1.0 - instance.destruction_probabilities[i][j], _EPS)
                    )
                    x[i, j].BranchPriority = int(abs(coef) * 1000)

            # Warm start
            if warm_start_assignment is not None:
                for i in range(weapons):
                    for j in range(targets):
                        x[i, j].Start = warm_start_assignment[i][j]

            # Pass references to the callback via model attributes
            model._x = x
            model._z = z
            model._B = B
            model._instance = instance
            model._best_true_obj = float("inf")
            model._best_assignment = None

            # ------------------------------------------------------------------
            # CALLBACK: Branch-and-Adjust (equivalent to CPLEX IncumbentCallback)
            #
            # When Gurobi finds an integer solution x*:
            #   1. Calculate the TRUE nonlinear WTA(x*) value
            #   2. Update best_true_obj and best_assignment
            #   3. For each j where z[j] < w_j*true_z[j]:
            #      Add lazy cut: z[j] >= true_cost_j * (1 - delta_x)
            #      When delta_x = 0 (same x*): forces z[j] = true_cost_j
            #      When delta_x >= 1 (other x) : non-binding constraint (>=0)
            #   4. Gurobi re-evaluates the node with the new constraint and accepts
            #      the incumbent with the true cost — replaces CPLEX reject()+inject()
            # ------------------------------------------------------------------
            def bna_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                x_val: dict = cb_model.cbGetSolution(cb_model._x)
                z_val: dict = cb_model.cbGetSolution(cb_model._z)

                inst = cb_model._instance
                n_w = inst.weapons
                n_t = inst.targets

                # True survival value of each target: prod_i (1-p_ij)^x_ij
                true_z = [1.0] * n_t
                assigned: list[tuple[int, int]] = []    # (i,j) where x[i,j] = 1
                not_assigned: list[tuple[int, int]] = []

                for i in range(n_w):
                    for j in range(n_t):
                        val = x_val[i, j]
                        if val > 0.5:
                            # For binary variables val ≈ 1; for integer val = r
                            true_z[j] *= (1.0 - inst.destruction_probabilities[i][j]) ** round(val)
                            assigned.append((i, j))
                        else:
                            not_assigned.append((i, j))

                true_obj = sum(inst.target_values[j] * true_z[j] for j in range(n_t))

                # Update best solution
                if true_obj < cb_model._best_true_obj:
                    cb_model._best_true_obj = true_obj
                    cb_model._best_assignment = tuple(
                        tuple(
                            1 if (i, j) in set(assigned) else 0
                            for j in range(n_t)
                        )
                        for i in range(n_w)
                    )

                # Build Hamming distance expression:
                #   delta_x = sum_{(i,j) in S1} (1 - x[i,j])  +  sum_{(i,j) in S0} x[i,j]
                #   delta_x = 0  ↔  current x = x*
                #   delta_x >= 1  ↔  differs in at least one position
                delta_x_expr: gp.LinExpr | None = None

                for j in range(n_t):
                    true_cost_j = inst.target_values[j] * true_z[j]

                    # Add cut only when the model underestimated target j cost
                    if z_val[j] < true_cost_j - 1e-6:
                        if delta_x_expr is None:
                            # Create Hamming expression once, reuse for all j
                            delta_x_expr = gp.quicksum(
                                1 - cb_model._x[i, j2] for (i, j2) in assigned
                            ) + gp.quicksum(
                                cb_model._x[i, j2] for (i, j2) in not_assigned
                            )

                        # Lazy cut (eq. key Adjust step):
                        #   z[j] >= true_cost_j * (1 - delta_x)
                        # When delta_x = 0: z[j] >= true_cost_j  → solver will use true cost
                        # When delta_x >= 1: z[j] >= 0           → non-binding
                        cb_model.cbLazy(
                            cb_model._z[j] >= true_cost_j * (1.0 - delta_x_expr)
                        )

            model.optimize(bna_callback)

            # ------------------------------------------------------------------
            # Collect results
            # ------------------------------------------------------------------
            runtime = perf_counter() - start

            status_map = {
                GRB.OPTIMAL: "optimal",
                GRB.TIME_LIMIT: "time_limit",
                GRB.INFEASIBLE: "infeasible",
                GRB.INF_OR_UNBD: "inf_or_unbd",
            }
            status_str = status_map.get(model.Status, f"status_{model.Status}")

            if model._best_assignment is not None:
                final_assignment = model._best_assignment
                final_obj = model._best_true_obj
            else:
                # Fallback: no integer solution was found
                final_assignment = tuple(tuple(0 for _ in range(targets)) for _ in range(weapons))
                final_obj = float("inf")

            return WTASolution(
                assignment=final_assignment,
                objective_value=final_obj,
                runtime_seconds=runtime,
                method="branch_and_adjust_gurobi",
                status=status_str,
            )
