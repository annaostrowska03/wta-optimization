from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence

import pulp
import gurobipy as gp
from gurobipy import GRB

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


def solve_branch_and_adjust(
        instance: WTAInstance,
        num_piecewise_segments: int = 20,
        warm_start: WTASolution | Sequence[Sequence[int]] | None = None,
        time_limit_seconds: float = 5400.0,
) -> WTASolution:
    """
    Solve the Static WTA problem using the true Branch-and-Adjust algorithm via Gurobi.
    Uses continuous Piecewise Linear Under-Approximations and Lazy Constraints to
    dynamically "Adjust" the node objectives when integer solutions are found.
    """
    start = perf_counter()
    warm_start_assignment = _normalize_warm_start(instance, warm_start)

    # Używamy context managerów Gurobi (dobre praktyki zarządzania licencją)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)  # Ustaw na 1, jeśli chcesz logi Gurobi w konsoli
        env.start()

        with gp.Model("WTA_Branch_and_Adjust_Gurobi", env=env) as model:
            # Włączamy użycie Lazy Constraints w Gurobi! (KRYTYCZNE)
            model.Params.LazyConstraints = 1
            if time_limit_seconds is not None:
                model.Params.TimeLimit = time_limit_seconds

            weapons = instance.weapons
            targets = instance.targets

            # Tworzenie zmiennych (używamy słowników dla łatwego dostępu w Gurobi)
            x = {}
            for i in range(weapons):
                for j in range(targets):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

            y = {}
            z = {}
            for j in range(targets):
                y[j] = model.addVar(lb=-GRB.INFINITY, name=f"y_{j}")
                z[j] = model.addVar(lb=0.0, name=f"z_{j}")

            # Cel - minimalizacja przybliżonych kosztów Z
            model.setObjective(gp.quicksum(instance.target_values[j] * z[j] for j in range(targets)), GRB.MINIMIZE)

            # Każda broń użyta max raz
            for i in range(weapons):
                model.addConstr(gp.quicksum(x[i, j] for j in range(targets)) <= 1, name=f"w_{i}")

            EPSILON = 1e-9

            # Przybliżenie logarytmiczne i styczne pod krzywą (WTA_LA z artykułu)
            for j in range(targets):
                ln_q = []
                for i in range(weapons):
                    q_ij = 1.0 - instance.destruction_probabilities[i][j]
                    ln_q.append(math.log(max(q_ij, EPSILON)))

                model.addConstr(y[j] == gp.quicksum(x[i, j] * ln_q[i] for i in range(weapons)), name=f"y_def_{j}")

                min_y_j = sum(val for val in ln_q if val < 0)

                if min_y_j < 0:
                    step = abs(min_y_j) / max(1, num_piecewise_segments - 1)
                    for k in range(num_piecewise_segments):
                        pk = min_y_j + k * step
                        exp_pk = math.exp(pk)
                        model.addConstr(z[j] >= exp_pk + exp_pk * (y[j] - pk))
                else:
                    model.addConstr(z[j] >= 1.0)

            # Inicjacja rozwiązania (Warm Start)
            if warm_start_assignment is not None:
                for i in range(weapons):
                    for j in range(targets):
                        x[i, j].Start = warm_start_assignment[i][j]

            # Rejestrujemy obiekty w modelu, żeby Callback miał do nich dostęp
            model._x = x
            model._z = z
            model._instance = instance
            model._best_true_obj = float('inf')
            model._best_assignment = None

            # ---------------------------------------------------------
            # MAGIA BRANCH-AND-ADJUST: GUROBI LAZY CALLBACK
            # ---------------------------------------------------------
            def bna_callback(cb_model, where):
                # Gdy solver zatrzyma się na pełnym węźle całkowitoliczbowym
                if where == GRB.Callback.MIPSOL:
                    # Pobieramy to próbne przypisanie i jego przybliżone koszty
                    x_val = cb_model.cbGetSolution(cb_model._x)
                    z_val = cb_model.cbGetSolution(cb_model._z)

                    targets = cb_model._instance.targets
                    weapons = cb_model._instance.weapons

                    true_z = [1.0] * targets
                    S1_vars = []
                    S0_vars = []

                    # Liczymy prawdziwy nieliniowy wzór ułożenia
                    for i in range(weapons):
                        for j in range(targets):
                            if x_val[i, j] > 0.5:
                                true_z[j] *= (1.0 - cb_model._instance.destruction_probabilities[i][j])
                                S1_vars.append(cb_model._x[i, j])
                            else:
                                S0_vars.append(cb_model._x[i, j])

                    true_obj = sum(cb_model._instance.target_values[j] * true_z[j] for j in range(targets))

                    # Notujemy sobie ten układ w razie czego (gdyby limit czasu zablokował solver)
                    if true_obj < cb_model._best_true_obj:
                        cb_model._best_true_obj = true_obj
                        assignment = [[0 for _ in range(targets)] for _ in range(weapons)]
                        for i in range(weapons):
                            for j in range(targets):
                                if x_val[i, j] > 0.5:
                                    assignment[i][j] = 1
                        cb_model._best_assignment = tuple(tuple(row) for row in assignment)

                    # KROK ADJUST: Sprawdzamy czy przybliżony model zaniżył koszty.
                    # Zamiast 'reject()', w Gurobi wysyłamy poprawkę kosztu dla TEGO KONKRETNEGO ułożenia.
                    delta_x = None  # Wskaźnik, który wynosi 0 TYLKO wtedy, gdy ułożenie to dokładnie to obecne

                    for j in range(targets):
                        if z_val[j] < true_z[j] - 1e-6:  # Model skłamał (zaniżył) o więcej niż EPSILON
                            if delta_x is None:
                                # Konstrukcja odległości Hamminga dla macierzy binarnej w Gurobi
                                delta_x = len(S1_vars) - gp.quicksum(S1_vars) + gp.quicksum(S0_vars)

                            # Dodajemy tzw. cięcie no-good cut.
                            # Jeśli badane ułożenie jest dokładnie tym, które znaleźliśmy (delta_x == 0),
                            # to wymuszamy matematycznie: Z musi podskoczyć do Prawdziwego Z.
                            # Dla innych ułożeń (delta >= 1), równanie się kasuje do 0 i nie przeszkadza drzewu.
                            cb_model.cbLazy(cb_model._z[j] >= true_z[j] * (1.0 - delta_x))

            # Start optymalizacji wraz z "pilnującym" go callbackiem
            model.optimize(bna_callback)

            runtime = perf_counter() - start

            status = model.Status
            status_map = {
                GRB.OPTIMAL: "optimal",
                GRB.TIME_LIMIT: "time_limit",
                GRB.INFEASIBLE: "infeasible",
            }
            status_str = status_map.get(status, f"status_{status}")

            if model._best_assignment is not None:
                final_assignment = model._best_assignment
                final_obj = model._best_true_obj
            else:
                final_assignment = tuple(tuple(0 for _ in range(targets)) for _ in range(weapons))
                final_obj = float('inf')

            return WTASolution(
                assignment=final_assignment,
                objective_value=final_obj,
                runtime_seconds=runtime,
                method="branch_and_adjust_gurobi",
                status=status_str,
            )