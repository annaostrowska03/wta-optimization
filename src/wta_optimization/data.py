from __future__ import annotations

from random import Random

from .models import WTAInstance


def generate_random_instance(
    weapons: int,
    targets: int,
    seed: int | None = None,
    target_value_range: tuple[float, float] = (1.0, 10.0),
    destruction_probability_range: tuple[float, float] = (0.1, 0.9),
) -> WTAInstance:
    rng = Random(seed)
    target_values = tuple(
        rng.uniform(*target_value_range) for _ in range(targets)
    )
    destruction_probabilities = tuple(
        tuple(rng.uniform(*destruction_probability_range) for _ in range(targets))
        for _ in range(weapons)
    )
    return WTAInstance(
        weapons=weapons,
        targets=targets,
        target_values=target_values,
        destruction_probabilities=destruction_probabilities,
    )
