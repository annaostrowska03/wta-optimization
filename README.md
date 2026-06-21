# Weapon–Target Assignment (WTA): Branch-and-Adjust

Implementation and empirical comparison of two **Branch-and-Adjust (BnA)** solver architectures for the static Weapon–Target Assignment problem, based on Andersen, Pavlikov, and Toffolo (2022).

The project implements a Gurobi adaptation of the Branch-and-Adjust framework and evaluates it on 30 benchmark instances.

## Problem

The static Weapon–Target Assignment problem assigns a limited inventory of heterogeneous weapons to targets in order to minimize the expected total surviving value of all targets:

$$
\min_x \sum_j v_j \prod_i (1-p_{ij})^{x_{ij}}
$$

subject to weapon-availability constraints:

$$
\sum_j x_{ij} \leq \mu_i,
\qquad
x_{ij} \in \mathbb{Z}_{\geq 0}.
$$

Here:

* $v_j$ is the value of target $j$,
* $p_{ij}$ is the destruction probability of weapon type $i$ against target $j$,
* $x_{ij}$ is the number of weapons of type $i$ assigned to target $j$,
* $\mu_i$ is the available inventory of weapon type $i$.

A feasible allocation provides an **upper bound** on the optimal objective value. The exact method must additionally construct valid lower bounds.

## Branch-and-Adjust Framework

The nonlinear survival product is transformed into log-survival space:

$$
y_j(x) = \sum_i \log(1-p_{ij})x_{ij},
$$

which gives:

$$
\prod_i (1-p_{ij})^{x_{ij}} = \exp(y_j(x)).
$$

The solver starts from a piecewise-linear lower approximation of the nonlinear objective. This produces a mixed-integer linear master problem with:

* integer allocation variables $x_{ij}$,
* convex-combination variables $\lambda_{j,t}$,
* auxiliary target-cost variables $z_j$.

Whenever Gurobi finds an integer candidate $x^*$, the `MIPSOL` callback:

1. evaluates the original nonlinear WTA objective;
2. checks whether the auxiliary values $z_j$ underestimate the true target costs;
3. adds tangent-based lazy cuts that strengthen the lower-bounding master problem.

For the convex target-cost function

$$
f_j(y) = v_j \exp(y),
$$

the tangent cut at $y_j^*$ is:

$$
z_j \geq
T_j^*(1-y_j^*)
+
\sum_i
T_j^* \log(1-p_{ij})x_{ij},
$$

where $T_j^*$ is the true nonlinear cost of target $j$ at the candidate allocation.

The cut does not exclude the allocation $x^*$ itself. It excludes only its underestimated representation in the master problem.

## Implementations

| File          | Function                     | Main mechanism                                                               |
| ------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| `exact.py`    | `solve_branch_and_adjust`    | Piecewise-linear master problem, nonlinear evaluation, and tangent lazy cuts |
| `exact_v2.py` | `solve_branch_and_adjust_v2` | All v1 mechanisms plus corrected-solution submission through `cbSetSolution` |

### BnA-v1

BnA-v1 evaluates each integer candidate with the original nonlinear objective and adds violated tangent lazy cuts. It also uses implementation-specific branching priorities.

### BnA-v2

BnA-v2 additionally reconstructs a feasible continuous representation of an improved candidate:

$$
(x^*, \lambda^*, z^*)
$$

where:

$$
z_j^* = T_j^*.
$$

The corrected tuple is submitted to Gurobi through `cbSetSolution`. If Gurobi accepts it, the candidate can improve the incumbent upper bound and enable additional pruning.

The two versions should be interpreted as **two complete solver architectures**, not as a perfectly isolated one-line ablation of `cbSetSolution`.

## Requirements

* Python 3.12+
* Gurobi Optimizer and a valid Gurobi license
* `gurobipy`
* NumPy
* Pandas
* Matplotlib

Install project dependencies according to the repository environment configuration.

## Environment Setup with uv

The repository includes both:

```text
pyproject.toml
uv.lock
```

The recommended way to create the environment is to use [uv](https://github.com/astral-sh/uv). The lock file ensures that the same package versions used for the experiments are installed.

From the repository root:

```bash
uv sync
```

This creates the local virtual environment and installs the locked dependencies.

To run commands inside the synchronized environment:

```bash
uv run python benchmark.py --method bna_v2
```

or:

```bash
uv run python results/make_report_figures.py
```


Using `uv sync` is preferred over manually installing packages with `pip`, because it reproduces the dependency versions specified in `uv.lock`.


## Benchmark Data

The benchmark instances originate from the reference implementation by Andersen et al.:

https://github.com/tuliotoffolo/wta

Before running experiments, copy the WTA benchmark instance files from the reference repository into:

```text
data/data_andersen/
```

The expected filenames follow this convention:

```text
wta_{W}x{T}x{mu}.txt
```

For example:

```text
wta_50x100x1.txt
wta_150x300x2.txt
wta_500x1000x3.txt
```

The full benchmark contains 30 instances:

* dimensions from `50 × 100` to `500 × 1000`,
* three availability levels: $\mu \in {1,2,3}$,
* ten instances for each availability level.

## Running Experiments

The recommended way to reproduce the full benchmark suite is:

```bash
bash run_all_benchmark.sh
```

Run this command from the repository root after placing all required Andersen benchmark files in:

```text
data/data_andersen/
```

The benchmark script runs the configured BnA-v1 and BnA-v2 experiments and saves result CSV files in the `results/` directory.

For an individual benchmark run, use:

```bash
python benchmark.py --method bna
```

or:

```bash
python benchmark.py --method bna_v2
```

Common options include:

| Flag           | Description                              |
| -------------- | ---------------------------------------- |
| `--method`     | `bna` for BnA-v1 or `bna_v2` for BnA-v2  |
| `--delta`      | Piecewise-linear approximation parameter |
| `--time-limit` | Optional per-instance time limit         |
| `--data-dir`   | Directory containing WTA benchmark files |
| `--files`      | Optional subset of benchmark files       |

## Results and Figures

Benchmark CSV files, generated figures, and report tables are stored in:

```text
results/
```

To generate the comparison figures used in the report, run from the repository root:

```bash
python results/make_report_figures.py
```

The script creates runtime comparisons, optimality-count plots, objective-quality plots, Andersen-reference comparisons, and LaTeX tables under:

```text
results/report_figures/
```

## Project Structure

```text
src/wta_optimization/
    models.py               # WTAInstance and WTASolution data structures
    data.py                 # Andersen-format and random-instance loaders
    bna_common.py           # Shared breakpoint, objective, and tangent-cut operations
    exact.py                # BnA-v1 implementation
    exact_v2.py             # BnA-v2 implementation

benchmark.py                # Benchmark runner
run_all_benchmark.sh        # Recommended full experimental pipeline

data/data_andersen/         # Andersen benchmark instances
results/                    # Benchmark CSV files, figures, and report tables
results/make_report_figures.py
                            # Script generating plots and LaTeX tables
```

## Reference

A. C. Andersen, K. Pavlikov, and T. A. M. Toffolo,
“Weapon-target assignment problem: exact and approximate solution algorithms,”
*Annals of Operations Research*, vol. 312, no. 2, pp. 581–606, 2022.
https://doi.org/10.1007/s10479-022-04525-6

## Team

* [Anna Ostrowska](https://github.com/annaostrowska03)
* [Gabriela Majstrak](https://github.com/GabrielaMajstrak)
* [Norbert Frydrysiak](https://github.com/fantasy2fry)

---

Developed as a course project for **Optimization and Data Analysis** at Warsaw University of Technology.
