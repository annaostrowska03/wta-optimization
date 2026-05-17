from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence

import pulp
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, quicksum

from .models import WTAInstance, WTASolution, objective_value


def _normalize_warm_start(
    instance: WTAInstance,
    warm_start: WTASolution | Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if warm_start is None:
        return None

    assignment = warm_start.assignment if isinstance(warm_start, WTASolution) else warm_start
    if len(assignment) != instance.weapons:
        raise ValueError("warm start assignment row count must match the number of weapons")

    normalized_assignment: list[tuple[int, ...]] = []
    for row in assignment:
        if len(row) != instance.targets:
            raise ValueError("each warm start assignment row must match the number of targets")
        normalized_assignment.append(tuple(int(bool(value)) for value in row))

    return tuple(normalized_assignment)


def solve_exact(
    instance: WTAInstance,
    num_piecewise_segments: int = 20,
    warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
    time_limit_seconds: float | None = None,
) -> WTASolution:
    """
    Solve the Static WTA problem using an exact MILP formulation via pulp.
    Uses log transformation mapping and piecewise linear approximation for the
    exponential objective reduction.
    """
    start = perf_counter()
    warm_start_assignment = _normalize_warm_start(instance, warm_start)

    prob = pulp.LpProblem("Static_WTA_MIP", pulp.LpMinimize)

    # Decision variables
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
          for j in range(instance.targets)] for i in range(instance.weapons)]
          
    z = [pulp.LpVariable(f"z_{j}", lowBound=0.0) for j in range(instance.targets)]
    y = [pulp.LpVariable(f"y_{j}") for j in range(instance.targets)]

    if warm_start_assignment is not None:
        for i in range(instance.weapons):
            for j in range(instance.targets):
                x[i][j].setInitialValue(warm_start_assignment[i][j])

    # Objective: Minimize sum(V_j * z_j) where z_j ~ exp(y_j)
    prob += pulp.lpSum([instance.target_values[j] * z[j] for j in range(instance.targets)])

    # Each weapon assigned at most once
    for i in range(instance.weapons):
        prob += pulp.lpSum(x[i]) <= 1

    EPSILON = 1e-9  # to prevent log(0) if p=1.0

    for j in range(instance.targets):
        ln_q = []
        for i in range(instance.weapons):
            q_ij = 1.0 - instance.destruction_probabilities[i][j]
            q_ij = max(q_ij, EPSILON)
            ln_q.append(math.log(q_ij))
            
        # y_j = sum_i( x_{ij} * ln(q_ij) )
        prob += y[j] == pulp.lpSum([x[i][j] * ln_q[i] for i in range(instance.weapons)])
        
        # Piecewise linear envelope for exp(y_j) (since exp is convex, max of tangents is a lower bound)
        min_y_j = sum(val for val in ln_q if val < 0)
        
        if min_y_j < 0:
            step = abs(min_y_j) / max(1, num_piecewise_segments - 1)
            for k in range(num_piecewise_segments):
                pk = min_y_j + k * step
                exp_pk = math.exp(pk)
                # Tangent line: z >= exp(pk) + exp(pk) * (y - pk)
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

    assignment = [[0 for _ in range(instance.targets)] for _ in range(instance.weapons)]
    for i in range(instance.weapons):
        for j in range(instance.targets):
            if pulp.value(x[i][j]) is not None and pulp.value(x[i][j]) > 0.5:
                assignment[i][j] = 1

    frozen_assignment = tuple(tuple(row) for row in assignment)

    runtime = perf_counter() - start
    return WTASolution(
        assignment=frozen_assignment,
        objective_value=objective_value(instance, frozen_assignment),
        runtime_seconds=runtime,
        method="exact_mip_pulp_linearized_warm_start" if warm_start_assignment is not None else "exact_mip_pulp_linearized",
        status=solver_status,
    )


class TrueObjectiveTracker(Eventhdlr):
    """
    SCIP Event Handler that tracks the true nonlinear objective
    whenever a new integer-feasible solution is found in the tree.
    """

    def __init__(self, instance: WTAInstance, x_vars: list[list]):
        self.instance = instance
        self.x_vars = x_vars
        self.best_true_obj = float('inf')
        self.best_assignment = None

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        sol = self.model.getBestSol()
        if sol is None:
            return

        true_obj = 0.0
        targets = self.instance.targets
        weapons = self.instance.weapons
        assignment = [[0 for _ in range(targets)] for _ in range(weapons)]

        for j in range(targets):
            surv = self.instance.target_values[j]
            for i in range(weapons):
                val = self.model.getSolVal(sol, self.x_vars[i][j])
                if val > 0.5:
                    surv *= (1.0 - self.instance.destruction_probabilities[i][j])
                    assignment[i][j] = 1
            true_obj += surv

        if true_obj < self.best_true_obj:
            self.best_true_obj = true_obj
            self.best_assignment = tuple(tuple(row) for row in assignment)


def solve_branch_and_adjust(
        instance: WTAInstance,
        num_piecewise_segments: int = 20,
        warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
        time_limit_seconds: float = 5400.0,
) -> WTASolution:
    """
    Solve the Static WTA problem using the Branch-and-Adjust algorithm via SCIP.
    Utilizes PySCIPOpt, piecewise-linear convex under-approximation, and an Event
    Handler to track the true non-linear objective.
    """
    start = perf_counter()
    warm_start_assignment = _normalize_warm_start(instance, warm_start)

    model = Model("WTA_Branch_and_Adjust_SCIP")
    model.hideOutput(True)
    model.setRealParam("limits/time", time_limit_seconds)

    weapons = instance.weapons
    targets = instance.targets

    x = [[model.addVar(vtype="B", name=f"x_{i}_{j}") for j in range(targets)] for i in range(weapons)]
    y = [model.addVar(vtype="C", lb=None, name=f"y_{j}") for j in range(targets)]
    z = [model.addVar(vtype="C", lb=0.0, name=f"z_{j}") for j in range(targets)]

    model.setObjective(quicksum(instance.target_values[j] * z[j] for j in range(targets)), "minimize")

    for i in range(weapons):
        model.addCons(quicksum(x[i][j] for j in range(targets)) <= 1, name=f"w_{i}")

    EPSILON = 1e-9

    for j in range(targets):
        ln_q = []
        for i in range(weapons):
            q_ij = 1.0 - instance.destruction_probabilities[i][j]
            ln_q.append(math.log(max(q_ij, EPSILON)))

        model.addCons(y[j] == quicksum(x[i][j] * ln_q[i] for i in range(weapons)), name=f"y_def_{j}")

        min_y_j = sum(val for val in ln_q if val < 0)

        if min_y_j < 0:
            step = abs(min_y_j) / max(1, num_piecewise_segments - 1)
            for k in range(num_piecewise_segments):
                pk = min_y_j + k * step
                exp_pk = math.exp(pk)
                model.addCons(z[j] >= exp_pk + exp_pk * (y[j] - pk))
        else:
            model.addCons(z[j] >= 1.0)

    if warm_start_assignment is not None:
        sol = model.createSol()
        for i in range(weapons):
            for j in range(targets):
                model.setSolVal(sol, x[i][j], warm_start_assignment[i][j])
        model.addSol(sol)

    tracker = TrueObjectiveTracker(instance, x)
    model.includeEventhdlr(tracker, "TrueObjTracker", "Tracks true objective on new best solutions")

    model.optimize()

    runtime = perf_counter() - start

    scip_status = model.getStatus()
    status_map = {
        "optimal": "optimal",
        "timelimit": "time_limit",
    }
    status_str = status_map.get(scip_status, f"status_{scip_status}")

    if tracker.best_assignment is not None:
        final_assignment = tracker.best_assignment
        final_obj = tracker.best_true_obj
    else:
        final_assignment = tuple(tuple(0 for _ in range(targets)) for _ in range(weapons))
        final_obj = float('inf')

    return WTASolution(
        assignment=final_assignment,
        objective_value=final_obj,
        runtime_seconds=runtime,
        method="branch_and_adjust_scip",
        status=status_str,
    )


def solve_outer_approximation(
        instance: WTAInstance,
        warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
        time_limit_seconds: float = 5400.0,
        verbose: bool = False,
) -> WTASolution:
    """
    Solve the Static WTA problem using the iterative outer approximation
    (cutting plane / Kelley) method.

    Since the objective sum(V_j * exp(y_j)) is convex in y_j, tangent
    hyperplanes are globally valid cutting planes.  The algorithm works as
    follows:

      1. Initialise with two tangent cuts per target (at y=0 and at min_y_j).
      2. Solve the MIP with the current set of cuts.
      3. At the resulting integer solution x*, compute y*_j exactly and check
         whether the linearised z*_j >= exp(y*_j) is satisfied.
      4. If violated: add a new tangent cut at y*_j and go to step 2.
      5. If no violation: the LP objective = true objective => optimal.

    Based on: Andersen, Pavlikov & Toffolo, "Weapon-target assignment problem:
    Exact and approximate solution algorithms", Ann. Oper. Res. (2022).
    """
    start = perf_counter()
    warm_start_assignment = _normalize_warm_start(instance, warm_start)

    EPSILON = 1e-9
    # CONV_TOL controls when a tangent cut is considered violated (and therefore
    # added to the model).  A cut for target j is added only when the current
    # linearised value z*_j is strictly less than exp(y*_j) by more than this
    # threshold.  Larger values converge faster but may stop before the true
    # optimum; smaller values tighten the outer approximation at the cost of
    # more iterations.  1e-6 is well within floating-point precision and gives
    # a certified optimality gap of effectively zero.
    CONV_TOL = 1e-6

    weapons = instance.weapons
    targets = instance.targets

    # Pre-compute ln(q_ij) = ln(1 - p_ij)
    ln_q: list[list[float]] = [
        [math.log(max(1.0 - instance.destruction_probabilities[i][j], EPSILON))
         for j in range(targets)]
        for i in range(weapons)
    ]
    # Lower bound on y_j: all weapons assigned to target j
    min_y: list[float] = [
        sum(ln_q[i][j] for i in range(weapons)) for j in range(targets)
    ]

    # Initial tangent points: y=0 (exp-max) and min_y_j (exp-min)
    tangent_points: dict[int, list[float]] = {
        j: ([0.0, min_y[j]] if min_y[j] < -1e-9 else [0.0])
        for j in range(targets)
    }

    best_assignment: tuple[tuple[int, ...], ...] | None = warm_start_assignment
    converged = False

    if verbose:
        total_cuts = sum(len(v) for v in tangent_points.values())
        print(f"OA init: {instance.weapons}W x {instance.targets}T | initial cuts: {total_cuts}")
        print(f"{'Iter':>4} | {'Cuts':>5} | {'Lin.Obj':>12} | {'True Obj':>12} | {'New cuts':>8}")

    for _iteration in range(300):
        remaining = time_limit_seconds - (perf_counter() - start)
        if remaining <= 0:
            break

        # build SCIP model with current tangent cuts 
        model = Model("WTA_OA")
        model.hideOutput(True)
        model.setRealParam("limits/time", remaining)

        x = [
            [model.addVar(vtype="B", name=f"x_{i}_{j}") for j in range(targets)]
            for i in range(weapons)
        ]
        y = [
            model.addVar(vtype="C", lb=min_y[j], ub=0.0, name=f"y_{j}")
            for j in range(targets)
        ]
        z = [
            model.addVar(vtype="C", lb=0.0, name=f"z_{j}")
            for j in range(targets)
        ]

        model.setObjective(
            quicksum(instance.target_values[j] * z[j] for j in range(targets)),
            "minimize",
        )

        for i in range(weapons):
            model.addCons(quicksum(x[i][j] for j in range(targets)) <= 1, name=f"w_{i}")

        for j in range(targets):
            model.addCons(
                y[j] == quicksum(x[i][j] * ln_q[i][j] for i in range(weapons)),
                name=f"y_{j}",
            )
            for y_bar in tangent_points[j]:
                exp_ybar = math.exp(y_bar)
                model.addCons(z[j] >= exp_ybar + exp_ybar * (y[j] - y_bar))

        # Warm start from previous best assignment
        if best_assignment is not None:
            init_sol = model.createSol()
            for i in range(weapons):
                for j in range(targets):
                    model.setSolVal(init_sol, x[i][j], best_assignment[i][j])
            model.addSol(init_sol)

        model.optimize()

        best_scip_sol = model.getBestSol()
        if best_scip_sol is None:
            break

        # Extract integer assignment from SCIP solution
        assignment_list = [[0] * targets for _ in range(weapons)]
        for i in range(weapons):
            for j in range(targets):
                if model.getSolVal(best_scip_sol, x[i][j]) > 0.5:
                    assignment_list[i][j] = 1
        frozen = tuple(tuple(row) for row in assignment_list)

        # Update best by true objective
        true_obj = objective_value(instance, frozen)
        if best_assignment is None or true_obj < objective_value(instance, best_assignment):
            best_assignment = frozen

        # check for violated cuts at this integer solution 
        # y*_j is uniquely determined by the binary assignment
        any_violation = False
        new_cuts_this_iter = 0
        lin_obj = sum(
            instance.target_values[j] * model.getSolVal(best_scip_sol, z[j])
            for j in range(targets)
        )
        for j in range(targets):
            y_star = sum(frozen[i][j] * ln_q[i][j] for i in range(weapons))
            z_star = model.getSolVal(best_scip_sol, z[j])
            exp_y_star = math.exp(y_star)

            if z_star < exp_y_star - CONV_TOL:
                # Add tangent cut at y_star
                if all(abs(y_star - tp) > 1e-10 for tp in tangent_points[j]):
                    tangent_points[j].append(y_star)
                    any_violation = True
                    new_cuts_this_iter += 1

        if verbose:
            total_cuts = sum(len(v) for v in tangent_points.values())
            print(f"{_iteration+1:>4} | {total_cuts:>5} | {lin_obj:>12.6f} | {true_obj:>12.6f} | {new_cuts_this_iter:>8}")

        if not any_violation:
            converged = True
            if verbose:
                print(f"Converged after {_iteration+1} iteration(s). Status: optimal.")
            break

    runtime = perf_counter() - start

    if best_assignment is None:
        best_assignment = tuple(tuple(0 for _ in range(targets)) for _ in range(weapons))

    return WTASolution(
        assignment=best_assignment,
        objective_value=objective_value(instance, best_assignment),
        runtime_seconds=runtime,
        method="outer_approximation_scip",
        status="optimal" if converged else "time_limit",
    )