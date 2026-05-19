# Weapon Target Allocation (WTA) Optimizer

This repository contains a comparative study of exact and heuristic algorithms for solving the **Static Weapon-Target Assignment (WTA)** problem.

The Static WTA problem is a classic NP-hard combinatorial optimization challenge. Given a set of weapons and targets, each weapon must be assigned to exactly one target to minimise the total expected survival value. The core objective of this project is to analyse the trade-off between guaranteed mathematical optimality and computational feasibility.

## Solvers implemented

| Method | Category | Backend | Notes |
|--------|----------|---------|-------|
| **Exact MIP** | Exact | PuLP / CBC | Piecewise-linear approximation of the log-sum objective (Camm et al.) |
| **Branch & Adjust** | Exact | PySCIPOpt / SCIP | 20-segment piecewise linearisation; SCIP event handler tracks true objective |
| **Outer Approximation** | Exact | PySCIPOpt / SCIP | Iterative cutting-plane (Kelley) method; adds tangent cuts to `exp(y_j)` until convergence |
| **Greedy** | Heuristic | – | Iteratively allocates the most cost-effective weapon to the highest-value target |
| **Greedy + Local Search** | Heuristic | – | Greedy solution refined by 2-opt neighbourhood search |
| **Simulated Annealing** | Heuristic | – | SA on greedy initialisation |

The Outer Approximation solver is based on the cutting-plane framework described in:
> Andersen, Pavlikov & Toffolo (2022). *Weapon-target assignment problem: Exact and approximate solution algorithms.* Annals of Operations Research.

## Benchmark schemes

Results are reported on two families of instances:

* **Andersen instances** — real WTA benchmark files from Mendeley Data (N = 5 … 200, square N×N), sourced from Sonuç, Sen & Bayır (2017).
* **Bertsimas–Paskov instances** — synthetically generated following the two schemes from:
  > Bertsimas & Paskov (2025). *Solving Large-Scale Weapon-Target Assignment Problems in Seconds Using Branch-Price-And-Cut.* Naval Research Logistics.
  * Scheme 1 (hard): $p_{ij} \sim U(0,1)$, $v_j \sim U_{\text{int}}[1, 100]$
  * Scheme 2: $p_{ij} \sim U(0.6, 0.9)$, $v_j \sim U_{\text{int}}[25, 100]$

## Repository structure

```
src/wta_optimization/
    models.py       – WTAInstance / WTASolution dataclasses
    data.py         – instance generators and file loader
    exact.py        – solve_exact, solve_branch_and_adjust, solve_outer_approximation
    heuristic.py    – solve_greedy, solve_local_search, solve_simulated_annealing
benchmark.py        – CLI benchmark runner (see Usage below)
compare_bpc.py      – comparison chart: our OA/BnA vs Bertsimas BPC (Table 1)
demo_oa.py          – interactive demo showing OA cutting-plane iterations
data/WTA/           – Andersen benchmark instances (wta5.txt … wta200.txt)
results/            – saved CSVs and plots
```

## Usage

```bash
# Run on Andersen file-based instances (wta5–wta200)
python benchmark.py --mode files

# Run on randomly generated instances (N = 5–30)
python benchmark.py --mode random

# Run on Bertsimas Scheme 1/2 instances (N = 5–30)
python benchmark.py --mode bertsimas --exact-limit-seconds 120

# Live demo: watch OA add cutting planes iteration by iteration
python demo_oa.py --n 10
python demo_oa.py --n 15 --scheme 2

# Comparison chart vs Bertsimas BPC (Table 1 from paper)
python compare_bpc.py
```

## Team

* [Anna Ostrowska](https://github.com/annaostrowska03)
* [Gabriela Majstrak](https://github.com/GabrielaMajstrak)
* [Norbert Frydrysiak](https://github.com/fantasy2fry)

---
*Developed as a course project for Optimization in Data Analysis @ WUT.*

## Data

Andersen benchmark instances: Mendeley Data — https://data.mendeley.com/datasets/jt2ppwr62p/2

