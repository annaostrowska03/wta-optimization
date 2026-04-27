# Weapon Target Allocation (WTA) Optimizer

This repository contains a comparative study of **Exact Mixed-Integer Programming (MIP)** formulations and **Fast Heuristic algorithms** for solving the Static Weapon Target Allocation (WTA) problem. 

The Static WTA problem is a classic NP-hard combinatorial optimization challenge. The core objective of this project is to analyze the fundamental trade-off between guaranteed mathematical optimality and computational feasibility, especially for large-scale, real-time systems where execution time is critical.


## Methodology

1. **Exact Approach:** Minimizes the expected survival value of targets using a strict mathematical formulation. To bypass the non-linear product terms (which cause standard solvers to fail), the model implements a logarithmic transformation trick.
2. **Heuristic Approach:** Prioritizes real-time execution by iteratively allocating the most effective weapons to the highest-value targets, optionally refined by local search to escape local minima.

## Team

* Anna Ostrowska
* Gabriela Majstrak
* Norbert Frydrysiak

---
*Developed as a course project for Optimization in Data Analysis @ WUT.*
```
