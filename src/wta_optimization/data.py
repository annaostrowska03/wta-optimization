from __future__ import annotations

from random import Random
from pathlib import Path

from .models import WTAInstance


def generate_random_instance(
    weapons: int,
    targets: int,
    seed: int | None = None,
    target_value_range: tuple[float, float] = (1.0, 10.0),
    destruction_probability_range: tuple[float, float] = (0.1, 0.9),
) -> WTAInstance:
    """Generate a random WTA instance with uniform target values and destruction probabilities."""
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


def load_andersen_instance(filepath: str | Path) -> tuple[WTAInstance, int]:
    """Load a non-square WTA instance from Andersen et al. (2022) file format.

    File format:
        W T mu            ← header line (mu = weapon availability per weapon)
        v_1               ← T target values (integers), one per line
        ...
        v_T
        w_idx t_idx p     ← W×T destruction probabilities (one per line)
        ...

    Returns
    -------
    (instance, mu)  where mu is the integer weapon availability for all weapons.
    """
    path = Path(filepath)
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split()
    weapons, targets, mu = int(header[0]), int(header[1]), int(header[2])

    target_values = tuple(float(lines[i + 1]) for i in range(targets))

    # Each probability line: "w_idx t_idx prob_float"
    probs = [[0.0] * targets for _ in range(weapons)]
    for k in range(weapons * targets):
        parts = lines[targets + 1 + k].split()
        w_idx, t_idx, prob = int(parts[0]), int(parts[1]), float(parts[2])
        probs[w_idx][t_idx] = prob

    destruction_probabilities = tuple(tuple(row) for row in probs)

    return WTAInstance(
        weapons=weapons,
        targets=targets,
        target_values=target_values,
        destruction_probabilities=destruction_probabilities,
    ), mu


def load_instance_from_file(filepath: str | Path, is_survival_prob: bool = True) -> WTAInstance:
    """Helper to load WTA instances from a text file. The file format is expected to be:
N
V_1
...
V_N
q_11 q_12 ... q_1N
...
q_N1 q_N2 ... q_NN"""
    path = Path(filepath)
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    N = int(lines[0])
    
    target_values = tuple(float(x) for x in lines[1:N+1])
    
    probs_flat = [float(x) for x in lines[N+1:]]
    
    destruction_probabilities = []
    idx = 0
    for i in range(N):
        row = []
        for j in range(N):
            val = probs_flat[idx]
            idx += 1
            if is_survival_prob:
                row.append(1.0 - val)
            else:
                row.append(val)
        destruction_probabilities.append(tuple(row))
        
    return WTAInstance(
        weapons=N,
        targets=N,
        target_values=target_values,
        destruction_probabilities=tuple(destruction_probabilities),
    )
