# Weapon Target Allocation (WTA) Optimizer

This repository contains a comparative study of **Exact Mixed-Integer Programming (MIP)** formulations and **Fast Heuristic algorithms** for solving the Static Weapon Target Allocation (WTA) problem. 

The Static WTA problem is a classic NP-hard combinatorial optimization challenge. The core objective of this project is to analyze the fundamental trade-off between guaranteed mathematical optimality and computational feasibility, especially for large-scale, real-time systems where execution time is critical.


## Methodology

1. **Exact Approach:** Minimizes the expected survival value of targets using a strict mathematical formulation. To bypass the non-linear product terms (which cause standard solvers to fail), the model implements a logarithmic transformation trick.
2. **Heuristic Approach:** Prioritizes real-time execution by iteratively allocating the most effective weapons to the highest-value targets, optionally refined by local search to escape local minima.

## Team

* [Anna Ostrowska](https://github.com/annaostrowska03)
* [Gabriela Majstrak](https://github.com/GabrielaMajstrak)
* [Norbert Frydrysiak](https://github.com/fantasy2fry)

---
*Developed as a course project for Optimization in Data Analysis @ WUT.

## Branch-and-Adjust implementations

The repository contains two implementations of the Branch-and-Adjust (BnA) algorithm from Andersen et al. (2022), both in `src/wta_optimization/`:

### `exact.py` — `solve_branch_and_adjust`

Baseline implementation. Uses a MIP model with:
- `x[i,j]` integer decision variables
- `lbda[j,t]` weights for piecewise-linear under-approximation of the nonlinear objective
- `z[j]` variables in the objective, bounded below by the LP under-approximation

When Gurobi finds an integer solution `x*` with LP value `L*` (under-estimate of the true cost `T*`), the callback adds **tangent lazy cuts**:

$$z_j \geq c_j (1 + y_j - y_j^*)$$

where $c_j = w_j e^{y_j^*}$ is the true target cost at `x*` and $y_j = \sum_i \log(1-p_{ij}) x_{ij}$.  
These cuts are globally valid (tangent to exp is a lower bound everywhere) and force the LP to reject the current solution until `z[j]` converges to `T*`.

### `exact_v2.py` — `solve_branch_and_adjust_v2`

Improved implementation — closer to Andersen's original CPLEX mechanism. Same model as above, with one addition in the callback:

When a new best `x*` is found, a corrected incumbent is **injected via `cbSetSolution`** with `z*[j] = T*_j` (true nonlinear cost) instead of the LP under-approximation `L*_j`. Since `z[j] >= under_approx` is a lower bound and `T*_j >= L*_j`, the injected solution is feasible — Gurobi immediately records `T*` as the incumbent upper bound.

This mirrors Andersen's CPLEX approach:
- `IncumbentCallback.reject()` → prevents `L*` from becoming the incumbent
- `HeuristicCallback.setSolution(T*)` → injects the true value as the upper bound

| | `exact.py` | `exact_v2.py` |
|---|---|---|
| Gurobi learns T* via | tangent cuts (iterative) | `cbSetSolution` (immediate) |
| Upper bound available from | 2nd visit to x* | 1st visit to x* |
| Pruning efficiency | lower | higher |
| Speed (50×100, μ=2) | ~254s | ~78s (~3× faster) |

Both produce correct results. `exact_v2.py` is recommended for benchmarking.

## Data

The benchmark tests in this repository include experiments run using the dataset published on Mendeley Data: https://data.mendeley.com/datasets/jt2ppwr62p/2
