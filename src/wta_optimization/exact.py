from __future__ import annotations

import math
from itertools import product
from time import perf_counter

import pulp

from .models import WTAInstance, WTASolution, objective_value


def solve_exact(instance: WTAInstance, num_piecewise_segments: int = 20) -> WTASolution:
    """
    Solve the Static WTA problem using an exact MILP formulation via pulp.
    Uses log transformation mapping and piecewise linear approximation for the 
    exponential objective reduction.
    """
    start = perf_counter()

    prob = pulp.LpProblem("Static_WTA_MIP", pulp.LpMinimize)

    # Decision variables
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary) 
          for j in range(instance.targets)] for i in range(instance.weapons)]
          
    z = [pulp.LpVariable(f"z_{j}", lowBound=0.0) for j in range(instance.targets)]
    y = [pulp.LpVariable(f"y_{j}") for j in range(instance.targets)]

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

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

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
        method="exact_mip_pulp_linearized",
    )
