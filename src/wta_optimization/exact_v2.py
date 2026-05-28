"""
Branch-and-Adjust for WTA — closer to Andersen et al. (2022).

Key difference from exact.py:
  When a new best integer solution x* is found, inject a corrected incumbent
  via cbSetSolution with z*[j] = T*_j (true target cost) instead of the LP
  under-approximation L*_j.  Since z[j] >= under_approx is a *lower* bound,
  T*_j >= L*_j makes the injected solution feasible, so Gurobi immediately
  records T* as the incumbent objective.

  This mirrors Andersen's CPLEX mechanism:
    IncumbentCallback  → reject LP value L*,  store (x*, T*)
    HeuristicCallback  → inject (x*, T*) into the B&B tree

  Tangent cuts are still added to tighten the LP lower bound.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from time import perf_counter
from typing import Sequence

import gurobipy as gp
from gurobipy import GRB

from .models import WTAInstance, WTASolution
from .exact import (
    _EPS,
    _compute_breakpoints,
    _integer_assignment_objective,
    _normalize_integer_warm_start,
    _resolve_mu,
)


def _lbda_from_log_survival(B_j: list[float], y_star: float) -> dict[int, float]:
    """
    Interpolate breakpoint weights for target j given log-survival y_star.

    B_j is sorted ascending: [b_min, ..., 0].  Returns {t: weight} with at
    most two nonzero entries that satisfy:
        sum_t B_j[t] * weight[t] = y_star   (log-survival link)
        sum_t weight[t] = 1                  (simplex)
    """
    y = max(B_j[0], min(B_j[-1], y_star))  # clamp to valid range

    if y >= B_j[-1]:
        return {len(B_j) - 1: 1.0}

    # t0 = last index where B_j[t0] <= y
    t0 = bisect_right(B_j, y) - 1
    t0 = max(0, min(len(B_j) - 2, t0))

    denom = B_j[t0 + 1] - B_j[t0]
    if abs(denom) < 1e-15:
        return {t0: 1.0}

    alpha = (B_j[t0 + 1] - y) / denom   # weight for t0
    beta  = 1.0 - alpha                  # weight for t0+1
    result: dict[int, float] = {}
    if alpha > 1e-15:
        result[t0] = alpha
    if beta > 1e-15:
        result[t0 + 1] = beta
    return result


def solve_branch_and_adjust_v2(
        instance: WTAInstance,
        delta: float = 1e-5,
        warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
        time_limit_seconds: float | None = None,
        mu: Sequence[int] | None = None,
) -> WTASolution:
    """
    Branch-and-Adjust with cbSetSolution injection.

    Model structure is identical to solve_branch_and_adjust (exact.py):
        x[i,j] integer, lbda[j,t] for PWL under-approximation, z[j] in objective.

    Callback additions vs exact.py:
        1. When true_obj < best_true_obj:  build a feasible solution
           (x*, lbda*, z*[j]=T*_j) and inject it via cbSetSolution so that
           Gurobi's incumbent upper bound = T* immediately.
        2. Tangent lazy cuts still added to drive LP lower bound upward.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")

    start = perf_counter()
    mu_values = _resolve_mu(instance, mu)
    warm_start_assignment = _normalize_integer_warm_start(instance, warm_start, mu_values)

    weapons = instance.weapons
    targets = instance.targets

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model("WTA_BnA_v2", env=env) as model:
            model.Params.LazyConstraints = 1
            if time_limit_seconds is not None:
                model.Params.TimeLimit = time_limit_seconds

            # x[i,j] integer in {0, ..., mu_i}

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

            for i in range(weapons):
                model.addConstr(
                    gp.quicksum(x[i, j] for j in range(targets)) <= mu_values[i],
                    name=f"weapon_avail_{i}",
                )

            # Delta-based breakpoints per target
            B: list[list[float]] = [
                _compute_breakpoints(instance.destruction_probabilities, mu_values, j, delta)
                for j in range(targets)
            ]

            # lbda[j,t] — SOS-style weights for convex under-approximation
            lbda: dict[tuple[int, int], gp.Var] = {}
            lbda_vars: list[gp.Var] = []
            lbda_keys: list[tuple[int, int]] = []

            for j in range(targets):
                for t in range(len(B[j])):
                    var = model.addVar(lb=0.0, name=f"lbda_{j}_{t}")
                    lbda[j, t] = var
                    lbda_vars.append(var)
                    lbda_keys.append((j, t))

            # z[j] — objective variables (under-approximation of survival cost)
            z: dict[int, gp.Var] = {}
            z_vars: list[gp.Var] = []

            for j in range(targets):
                z[j] = model.addVar(lb=0.0, obj=1.0, name=f"z_{j}")
                z_vars.append(z[j])

                w_j = instance.target_values[j]
                last_t = len(B[j]) - 1

                # Under-approximation: Andersen et al. compact PWL form
                under_approx = gp.quicksum(
                    (
                        w_j * math.exp(B[j][t]) if t == 0
                        else w_j * (math.exp(B[j][t]) - delta) if t < last_t
                        else w_j
                    ) * lbda[j, t]
                    for t in range(len(B[j]))
                )
                model.addConstr(z[j] >= under_approx, name=f"z_underapprox_{j}")

                # Simplex constraint
                model.addConstr(
                    gp.quicksum(lbda[j, t] for t in range(len(B[j]))) == 1.0,
                    name=f"lbda_sum_{j}",
                )
                # Log-survival link
                model.addConstr(
                    gp.quicksum(B[j][t] * lbda[j, t] for t in range(len(B[j])))
                    == gp.quicksum(
                        math.log(max(1.0 - instance.destruction_probabilities[i][j], _EPS))
                        * x[i, j]
                        for i in range(weapons)
                    ),
                    name=f"log_survival_link_{j}",
                )

            model.ModelSense = GRB.MINIMIZE

            if warm_start_assignment is not None:
                for i in range(weapons):
                    for j in range(targets):
                        x[i, j].Start = warm_start_assignment[i][j]

            model._x = x
            model._x_keys = x_keys
            model._x_vars = x_vars
            model._lbda = lbda
            model._lbda_vars = lbda_vars
            model._lbda_keys = lbda_keys
            model._z = z
            model._z_vars = z_vars
            model._B = B
            model._mu_values = mu_values
            model._instance = instance
            model._best_true_obj = float("inf")
            model._best_assignment = None
            model._lazy_cuts_added = 0

            def bna_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                inst   = cb_model._instance
                mu_cb  = cb_model._mu_values
                n_w    = inst.weapons
                n_t    = inst.targets

                x_values = cb_model.cbGetSolution(cb_model._x_vars)
                z_values = cb_model.cbGetSolution(cb_model._z_vars)

                # Build x* and compute true nonlinear objective
                current_assignment = [[0] * n_t for _ in range(n_w)]
                true_survival     = [1.0] * n_t
                true_log_survival = [0.0] * n_t

                for idx, (i, j) in enumerate(cb_model._x_keys):
                    value = int(round(x_values[idx]))
                    value = max(0, min(mu_cb[i], value))
                    current_assignment[i][j] = value
                    if value > 0:
                        p_ij  = inst.destruction_probabilities[i][j]
                        log_q = math.log(max(1.0 - p_ij, _EPS))
                        true_survival[j]     *= (1.0 - p_ij) ** value
                        true_log_survival[j] += value * log_q

                true_target_cost = [
                    inst.target_values[j] * true_survival[j] for j in range(n_t)
                ]
                true_obj = sum(true_target_cost)

                # INJECTION: when true_obj improves, push (x*, lbda*, z*=T*)
                # into the B&B tree so Gurobi records T* as the incumbent.
                #
                # Feasibility of the injected solution:
                #   x* : satisfies integer & weapon constraints           
                #   lbda* : interpolated on breakpoints for y_j*          
                #   z*[j] = T*_j >= L*_j (= LP under-approx at y_j*)    
                #   Objective = sum_j T*_j = T*                           
                #
                # This mirrors Andersen's HeuristicCallback.setSolution(T*).

                if true_obj < cb_model._best_true_obj - 1e-9:
                    cb_model._best_true_obj  = true_obj
                    cb_model._best_assignment = tuple(
                        tuple(row) for row in current_assignment
                    )

                    # Compute lbda* by interpolating y_j* on the breakpoints
                    lbda_star_dict: dict[tuple[int, int], float] = {}
                    for j in range(n_t):
                        weights = _lbda_from_log_survival(
                            cb_model._B[j], true_log_survival[j]
                        )
                        for t_idx, w_val in weights.items():
                            lbda_star_dict[j, t_idx] = w_val

                    # Build flat solution arrays
                    inject_vars = (
                        cb_model._x_vars
                        + cb_model._lbda_vars
                        + cb_model._z_vars
                    )
                    inject_vals = (
                        [float(x_values[k]) for k in range(len(cb_model._x_vars))]
                        + [lbda_star_dict.get(key, 0.0)
                           for key in cb_model._lbda_keys]
                        + true_target_cost          # z*[j] = T*_j
                    )

                    cb_model.cbSetSolution(inject_vars, inject_vals)

                # TANGENT CUTS: reject LP under-estimates via globally valid cuts
                #   z[j] >= c_j * (1 - y_j*) + c_j * sum_i log(1-p_ij)*x[i,j]
                # where c_j = T*_j and y[j] = sum_i log(1-p_ij)*x[i,j].

                needs_adjustment = [
                    j for j in range(n_t)
                    if z_values[j] < true_target_cost[j] - 1e-6
                ]
                if not needs_adjustment:
                    return

                if len(needs_adjustment) > 50:
                    needs_adjustment = sorted(
                        needs_adjustment,
                        key=lambda jj: true_target_cost[jj] - z_values[jj],
                        reverse=True,
                    )[:50]

                for j in needs_adjustment:
                    c_j     = true_target_cost[j]
                    y_j_star = true_log_survival[j]
                    cb_model.cbLazy(
                        cb_model._z[j] >= c_j * (1.0 - y_j_star) + gp.quicksum(
                            c_j * math.log(
                                max(1.0 - inst.destruction_probabilities[i][j], _EPS)
                            ) * cb_model._x[i, j]
                            for i in range(n_w)
                        )
                    )
                    cb_model._lazy_cuts_added += 1
            model.optimize(bna_callback)

            runtime = perf_counter() - start

            status_map = {
                GRB.OPTIMAL:         "optimal",
                GRB.TIME_LIMIT:      "time_limit",
                GRB.INFEASIBLE:      "infeasible",
                GRB.INF_OR_UNBD:     "inf_or_unbd",
                GRB.UNBOUNDED:       "unbounded",
                GRB.INTERRUPTED:     "interrupted",
                GRB.NODE_LIMIT:      "node_limit",
                GRB.SOLUTION_LIMIT:  "solution_limit",
                GRB.ITERATION_LIMIT: "iteration_limit",
                GRB.MEM_LIMIT:       "mem_limit",
            }
            status_str = status_map.get(model.Status, f"status_{model.Status}")

            if model._best_assignment is not None:
                final_assignment = model._best_assignment
                final_obj        = model._best_true_obj
            elif model.SolCount > 0:
                final_assignment_list = [
                    [int(round(x[i, j].X)) for j in range(targets)]
                    for i in range(weapons)
                ]
                final_assignment = tuple(tuple(row) for row in final_assignment_list)
                final_obj        = _integer_assignment_objective(instance, final_assignment)
            else:
                final_assignment = tuple(
                    tuple(0 for _ in range(targets)) for _ in range(weapons)
                )
                final_obj = float("inf")

            return WTASolution(
                assignment=final_assignment,
                objective_value=final_obj,
                runtime_seconds=runtime,
                method="branch_and_adjust_v2_gurobi",
                status=status_str,
            )
