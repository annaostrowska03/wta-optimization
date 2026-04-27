from __future__ import annotations

from time import perf_counter

from .models import WTAInstance, WTASolution, objective_value


def solve_greedy(instance: WTAInstance) -> WTASolution:
    start = perf_counter()
    assignment = [[0 for _ in range(instance.targets)] for _ in range(instance.weapons)]
    remaining_survival = list(instance.target_values)

    for weapon_index in range(instance.weapons):
        best_target = None
        best_gain = 0.0
        for target_index in range(instance.targets):
            destruction = instance.destruction_probabilities[weapon_index][target_index]
            gain = remaining_survival[target_index] * destruction
            if gain > best_gain:
                best_gain = gain
                best_target = target_index

        if best_target is not None:
            assignment[weapon_index][best_target] = 1
            remaining_survival[best_target] *= 1.0 - instance.destruction_probabilities[weapon_index][best_target]

    runtime = perf_counter() - start
    frozen_assignment = tuple(tuple(row) for row in assignment)
    return WTASolution(
        assignment=frozen_assignment,
        objective_value=objective_value(instance, frozen_assignment),
        runtime_seconds=runtime,
        method="greedy",
    )
