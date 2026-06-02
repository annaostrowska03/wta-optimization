from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WTAInstance:
    """Immutable data class representing a WTA problem instance."""

    weapons: int
    targets: int
    target_values: tuple[float, ...]
    destruction_probabilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.weapons <= 0 or self.targets <= 0:
            raise ValueError("weapons and targets must be positive")
        if len(self.target_values) != self.targets:
            raise ValueError("target_values length must match targets")
        if len(self.destruction_probabilities) != self.weapons:
            raise ValueError("destruction_probabilities row count must match weapons")
        for row in self.destruction_probabilities:
            if len(row) != self.targets:
                raise ValueError("each probability row must match targets")
            for value in row:
                if not 0.0 <= value <= 1.0:
                    raise ValueError("destruction probabilities must be within [0, 1]")


@dataclass(frozen=True)
class WTASolution:
    """Immutable data class representing a WTA solution with solver metadata."""

    assignment: tuple[tuple[int, ...], ...]
    objective_value: float
    runtime_seconds: float = 0.0
    method: str = "unknown"
    status: str = "unknown"

    @property
    def assigned_pairs(self) -> list[tuple[int, int]]:
        """Return all (weapon, target) index pairs with a positive assignment."""
        return [
            (weapon_index, target_index)
            for weapon_index, row in enumerate(self.assignment)
            for target_index, value in enumerate(row)
            if value == 1
        ]

