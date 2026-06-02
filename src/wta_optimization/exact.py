from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence

import gurobipy as gp
from gurobipy import GRB

from .bna_common import (_EPS, _GRB_STATUS_MAP, add_tangent_cuts,
                         compute_breakpoints, compute_integer_solution,
                         finalize_solution, normalize_integer_warm_start,
                         resolve_mu)
from .models import WTAInstance, WTASolution


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
    mu_values = resolve_mu(instance, mu)
    warm_start_assignment = normalize_integer_warm_start(
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
                    compute_breakpoints(
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
                    compute_integer_solution(
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

                cb_model._lazy_cuts_added += add_tangent_cuts(
                    cb_model, z_values, true_target_cost, true_log_survival
                )

            model.optimize(bna_callback)

            runtime = perf_counter() - start
            status_str = _GRB_STATUS_MAP.get(model.Status, f"status_{model.Status}")
            final_assignment, final_obj = finalize_solution(model, instance, x)

            return WTASolution(
                assignment=final_assignment,
                objective_value=final_obj,
                runtime_seconds=runtime,
                method="branch_and_adjust_gurobi_integer",
                status=status_str,
            )
