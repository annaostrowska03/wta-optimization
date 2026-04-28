from __future__ import annotations

import math
from random import Random
from time import perf_counter

from .models import WTAInstance, WTASolution, objective_value


def _compute_objective_1d(instance: WTAInstance, assignment_1d: list[int]) -> float:
    surv = list(instance.target_values)
    for weapon_index, target_index in enumerate(assignment_1d):
        if target_index != -1:
            surv[target_index] *= 1.0 - instance.destruction_probabilities[weapon_index][target_index]
    return sum(surv)


def _assignment_1d_to_matrix(instance: WTAInstance, assignment_1d: list[int]) -> tuple[tuple[int, ...], ...]:
    assignment = [[0 for _ in range(instance.targets)] for _ in range(instance.weapons)]
    for weapon_index, target_index in enumerate(assignment_1d):
        if target_index != -1:
            assignment[weapon_index][target_index] = 1
    return tuple(tuple(row) for row in assignment)


def _build_solution(
    instance: WTAInstance,
    assignment_1d: list[int],
    runtime_seconds: float,
    method: str,
    objective: float | None = None,
) -> WTASolution:
    frozen_assignment = _assignment_1d_to_matrix(instance, assignment_1d)
    return WTASolution(
        assignment=frozen_assignment,
        objective_value=objective if objective is not None else objective_value(instance, frozen_assignment),
        runtime_seconds=runtime_seconds,
        method=method,
    )


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

    runtime = perf_counter() - start
    return _build_solution(instance, assignment_1d, runtime, method="greedy")


def solve_local_search(
    instance: WTAInstance,
    max_iterations: int = 1000,
    initial_assignment: list[int] | None = None,
) -> WTASolution:
    """
    A strong local search heuristic (Hill Climbing).
    It starts with the greedy assignment and attempts two neighborhood operations:
    1. Shift/Move: Move weapon W to a different target.
    2. Swap: Exchange the targets of Weapon W1 and Weapon W2.
    Continues until a local optimum is reached or max_iterations is hit.
    """
    start = perf_counter()
    
    # Starting solution
    assignment_1d = list(initial_assignment) if initial_assignment is not None else _greedy_initial_assignment(instance)

    current_obj = _compute_objective_1d(instance, assignment_1d)
    
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
                new_obj = _compute_objective_1d(instance, assignment_1d)
                
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
                new_obj = _compute_objective_1d(instance, assignment_1d)
                
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

    runtime = perf_counter() - start
    return _build_solution(
        instance,
        assignment_1d,
        runtime,
        method="local_search_hill_climbing",
        objective=current_obj,
    )


def solve_simulated_annealing(
    instance: WTAInstance,
    max_iterations: int = 5000,
    initial_temperature: float | None = None,
    cooling_rate: float = 0.995,
    seed: int = 0,
) -> WTASolution:
    """
    Simulated annealing seeded with the greedy solution.
    It occasionally accepts worse moves early on, which helps escape local minima.
    """
    start = perf_counter()
    rng = Random(seed)

    current_assignment = _greedy_initial_assignment(instance)
    current_obj = _compute_objective_1d(instance, current_assignment)
    best_assignment = current_assignment.copy()
    best_obj = current_obj

    if initial_temperature is None:
        initial_temperature = max(instance.target_values) if instance.target_values else 1.0
    temperature = max(initial_temperature, 1e-6)

    for _ in range(max_iterations):
        candidate_assignment = current_assignment.copy()

        if instance.weapons > 1 and rng.random() < 0.5:
            w1, w2 = rng.sample(range(instance.weapons), 2)
            candidate_assignment[w1], candidate_assignment[w2] = candidate_assignment[w2], candidate_assignment[w1]
        else:
            weapon_index = rng.randrange(instance.weapons)
            current_target = candidate_assignment[weapon_index]
            available_targets = [target for target in range(instance.targets) if target != current_target]
            if available_targets:
                candidate_assignment[weapon_index] = rng.choice(available_targets)

        candidate_obj = _compute_objective_1d(instance, candidate_assignment)
        delta = candidate_obj - current_obj

        if delta < 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-9)):
            current_assignment = candidate_assignment
            current_obj = candidate_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_assignment = current_assignment.copy()

        temperature *= cooling_rate
        temperature = max(temperature, 1e-6)

    runtime = perf_counter() - start
    return _build_solution(
        instance,
        best_assignment,
        runtime,
        method="simulated_annealing",
        objective=best_obj,
    )
