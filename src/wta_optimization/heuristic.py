from __future__ import annotations

from time import perf_counter

from .models import WTAInstance, WTASolution, objective_value


def _greedy_initial_assignment(instance: WTAInstance) -> list[int]:
    """Helper to return a greedy starting point as a 1D list showing weapon -> target."""
    assignment_1d = [-1] * instance.weapons
    remaining_survival = list(instance.target_values)

    for weapon_index in range(instance.weapons):
        best_target = -1
        best_gain = -1.0
        for target_index in range(instance.targets):
            destruction = instance.destruction_probabilities[weapon_index][target_index]
            gain = remaining_survival[target_index] * destruction
            if gain > best_gain:
                best_gain = gain
                best_target = target_index

        if best_target != -1:
            assignment_1d[weapon_index] = best_target
            remaining_survival[best_target] *= 1.0 - instance.destruction_probabilities[weapon_index][best_target]
            
    return assignment_1d

def solve_greedy(instance: WTAInstance) -> WTASolution:
    start = perf_counter()
    assignment_1d = _greedy_initial_assignment(instance)
    
    assignment = [[0 for _ in range(instance.targets)] for _ in range(instance.weapons)]
    for w, t in enumerate(assignment_1d):
        if t != -1:
            assignment[w][t] = 1

    runtime = perf_counter() - start
    frozen_assignment = tuple(tuple(row) for row in assignment)
    return WTASolution(
        assignment=frozen_assignment,
        objective_value=objective_value(instance, frozen_assignment),
        runtime_seconds=runtime,
        method="greedy",
    )


def solve_local_search(instance: WTAInstance, max_iterations: int = 1000) -> WTASolution:
    """
    A strong local search heuristic (Hill Climbing).
    It starts with the greedy assignment and attempts two neighborhood operations:
    1. Shift/Move: Move weapon W to a different target.
    2. Swap: Exchange the targets of Weapon W1 and Weapon W2.
    Continues until a local optimum is reached or max_iterations is hit.
    """
    start = perf_counter()
    
    # Starting solution
    assignment_1d = _greedy_initial_assignment(instance)
    
    # We will evaluate objectives frequently, so we need a fast way
    def compute_obj(assign_1d):
        surv = list(instance.target_values)
        for w, t in enumerate(assign_1d):
            if t != -1:
                surv[t] *= (1.0 - instance.destruction_probabilities[w][t])
        return sum(surv)

    current_obj = compute_obj(assignment_1d)
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Neighborhood 1: 1-opt Shift Move
        for w in range(instance.weapons):
            t_old = assignment_1d[w]
            for t_new in range(instance.targets):
                if t_old == t_new:
                    continue
                
                # Test move
                assignment_1d[w] = t_new
                new_obj = compute_obj(assignment_1d)
                
                if new_obj < current_obj - 1e-9:
                    current_obj = new_obj
                    improved = True
                    break # immediate accept
                else:
                    # Revert move
                    assignment_1d[w] = t_old
                    
            if improved:
                break
                
        if improved:
            continue
            
        # Neighborhood 2: 2-opt Swap
        for w1 in range(instance.weapons):
            for w2 in range(w1 + 1, instance.weapons):
                t1 = assignment_1d[w1]
                t2 = assignment_1d[w2]
                if t1 == t2:
                    continue
                
                # Test swap
                assignment_1d[w1] = t2
                assignment_1d[w2] = t1
                new_obj = compute_obj(assignment_1d)
                
                if new_obj < current_obj - 1e-9:
                    current_obj = new_obj
                    improved = True
                    break
                else:
                    # Revert swap
                    assignment_1d[w1] = t1
                    assignment_1d[w2] = t2
                    
            if improved:
                break

    # Build the final 2D matrix
    assignment = [[0 for _ in range(instance.targets)] for _ in range(instance.weapons)]
    for w, t in enumerate(assignment_1d):
        if t != -1:
            assignment[w][t] = 1

    runtime = perf_counter() - start
    frozen_assignment = tuple(tuple(row) for row in assignment)
    return WTASolution(
        assignment=frozen_assignment,
        objective_value=current_obj,
        runtime_seconds=runtime,
        method="local_search_hill_climbing",
    )
