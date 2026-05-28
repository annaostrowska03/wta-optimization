# Weapon Target Assignment (WTA) — Branch-and-Adjust

Implementation and empirical comparison of the **Branch-and-Adjust (BnA)** algorithm for the static Weapon-Target Assignment problem, based on Andersen et al. (2022).

## Problem

Static WTA is a nonlinear integer program: assign weapons to targets to minimise the total expected survival value of all targets.

$$\min \sum_{j} v_j \prod_{i} (1 - p_{ij})^{x_{ij}}$$

subject to $\sum_j x_{ij} \le \mu_i$ (weapon availability) and $x_{ij} \in \mathbb{Z}_{\ge 0}$.

## Branch-and-Adjust Algorithm

BnA linearises the nonlinear objective using a piecewise-linear lower approximation controlled by the parameter $\delta$. Whenever Gurobi finds an integer solution $x^*$, the MIPSOL callback:

1. Computes the true nonlinear cost $T^*_j$ for each target.
2. Adds **tangent lazy cuts** $z_j \ge c_j(1 + y_j - y_j^*)$ (globally valid lower bounds on $\exp$) that force the LP relaxation to converge to $T^*$.
3. *(v2 only)* Injects the corrected incumbent $(x^*, z^*_j = T^*_j)$ via `cbSetSolution` so Gurobi records $T^*$ immediately as the upper bound.

## Implementations

| File | Function | Mechanism | Speed |
|---|---|---|---|
| `exact.py` | `solve_branch_and_adjust` | tangent cuts only | baseline |
| `exact_v2.py` | `solve_branch_and_adjust_v2` | tangent cuts + `cbSetSolution` | ~2–3× faster |

`exact_v2.py` is recommended. Both produce correct results.

## Usage

```bash
# Run benchmark on all 30 Andersen instances (delta=1e-5, 7200 s limit)
python benchmark.py --method bna_v2

# Compare results against Andersen et al. (2022) Table 5
python compare_andersen.py
```

CLI options for `benchmark.py`:

| Flag | Default | Description |
|---|---|---|
| `--method` | `bna` | `bna` (exact.py) or `bna_v2` (exact_v2.py) |
| `--time-limit` | `7200` | Per-instance time limit in seconds |
| `--delta` | `1e-5` | Approximation parameter δ |
| `--data-dir` | `data/data_andersen` | Directory with instance files |

## Data

Instance files (`wta_{W}x{T}x{mu}.txt`) from the [Andersen et al. (2022) Mendeley dataset](https://data.mendeley.com/datasets/jt2ppwr62p/2). Sizes range from 50×100 to 500×1000 (weapons × targets), with μ ∈ {1, 2, 3}.

Andersen et al. reference results are in `data/results.csv`.

## Project structure

```
src/wta_optimization/
    models.py          — WTAInstance, WTASolution dataclasses
    data.py            — instance loaders (Andersen format + random)
    exact.py           — BnA v1 (tangent cuts)
    exact_v2.py        — BnA v2 (tangent cuts + cbSetSolution injection)
benchmark.py           — CLI benchmark runner (resume-capable)
compare_andersen.py    — comparison vs Andersen et al. Table 5
data/data_andersen/    — 30 benchmark instances
results/               — benchmark CSVs and comparison plots
```

## Team

* [Anna Ostrowska](https://github.com/annaostrowska03)
* [Gabriela Majstrak](https://github.com/GabrielaMajstrak)
* [Norbert Frydrysiak](https://github.com/fantasy2fry)

---
*Developed as a course project for Optimization in Data Analysis @ WUT.*
