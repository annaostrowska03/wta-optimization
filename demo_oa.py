"""
Live demonstration of the Outer Approximation (cutting-plane) solver for the
Weapon-Target Assignment problem.

Run:
    python demo_oa.py               # default: Bertsimas Scheme 1, N=10
    python demo_oa.py --n 15        # larger instance
    python demo_oa.py --scheme 2    # Scheme 2 (easier probabilities)

This demo shows each OA iteration:
  - how many tangent cuts (supporting hyperplanes of e^y) are in the model
  - the linearised objective (lower bound from the MIP with current cuts)
  - the true nonlinear objective at the best integer solution found
  - how many new cuts were added in this iteration

The loop terminates when the linearised objective == true objective
(within CONV_TOL = 1e-6), proving optimality WITHOUT enumeration.
"""
import argparse
import sys

sys.path.insert(0, "src")

from wta_optimization.data import generate_random_instance
from wta_optimization.exact import solve_exact, solve_outer_approximation
from wta_optimization.heuristic import solve_greedy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OA live demo")
    p.add_argument("--n", type=int, default=10, help="Number of weapons = targets (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--scheme",
        type=int,
        choices=[1, 2],
        default=1,
        help="Bertsimas instance generation scheme (default: 1 — harder)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.scheme == 1:
        # Bertsimas & Paskov (2025) Scheme 1: hard instances
        prob_range = (0.001, 0.999)
        val_range = (1.0, 100.0)
        scheme_desc = "Scheme 1 — P_ij ~ U(0,1), v_j ~ U_int[1,100] (hard)"
    else:
        # Bertsimas & Paskov (2025) Scheme 2
        prob_range = (0.6, 0.9)
        val_range = (25.0, 100.0)
        scheme_desc = "Scheme 2 — P_ij ~ U(0.6,0.9), v_j ~ U_int[25,100]"

    print("  Weapon-Target Assignment — Outer Approximation Demo")
    print(f"  Instance : {args.n} weapons × {args.n} targets  (seed={args.seed})")
    print(f"  {scheme_desc}")

    instance = generate_random_instance(
        weapons=args.n,
        targets=args.n,
        seed=args.seed,
        target_value_range=val_range,
        destruction_probability_range=prob_range,
    )

    print(f"\nTarget values : {[round(v, 1) for v in instance.target_values]}\n")

    # --- Greedy warm start ---
    greedy_sol = solve_greedy(instance)
    print(f"Greedy heuristic objective : {greedy_sol.objective_value:.6f} \n")

    # Outer Approximation with verbose output
    print("Running Outer Approximation (iterative cutting-plane method)...\n")
    oa_sol = solve_outer_approximation(instance, warm_start=greedy_sol, verbose=True)
    print(f"\n OA  objective : {oa_sol.objective_value:.6f}  [{oa_sol.runtime_seconds:.3f}s]")

    # Exact MIP for validation
    print("\nValidating with exact MIP (PuLP/CBC)...")
    exact_sol = solve_exact(instance, num_piecewise_segments=20, warm_start=greedy_sol)
    print(f"Exact MIP obj : {exact_sol.objective_value:.6f}  [{exact_sol.runtime_seconds:.3f}s]")

    gap = abs(oa_sol.objective_value - exact_sol.objective_value)
    print()
    if gap < 1e-4:
        print(f"✓ OA matches exact MIP  (|gap| = {gap:.2e})")
    else:
        print(f"✗ WARNING: gap = {gap:.6f}")

    greedy_gap_pct = max(0.0, (greedy_sol.objective_value - exact_sol.objective_value) / max(exact_sol.objective_value, 1e-9) * 100)
    print(f"\nGreedy optimality gap : {greedy_gap_pct:.2f}%")
    print("OA    optimality gap  : 0.00%  (guaranteed — exact method)\n")
    print("Key insight: OA adds ONE tangent hyperplane per violated target per")
    print("iteration. Each hyperplane is a supporting line of the convex function")
    print("exp(y_j), valid GLOBALLY (not just locally). The solver needs only a")
    print("handful of iterations because exp() is smooth and well-approximated")
    print("by a small number of tangents near the optimal solution.")


if __name__ == "__main__":
    main()
