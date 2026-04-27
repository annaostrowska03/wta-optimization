import csv
from pathlib import Path
from time import perf_counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wta_optimization.data import generate_random_instance
from wta_optimization.exact import solve_exact
from wta_optimization.heuristic import solve_greedy


def run_benchmark():
    sizes = [5, 10, 15, 20, 25, 30]  # List of grid sizes (weapons & targets)
    seeds = [42, 43, 44]            # Multiple seeds to average out the results
    
    results = []
    
    print(f"{'Size':<10} | {'Seed':<6} | {'Exact Time':<12} | {'Greedy Time':<12} | {'Gap %':<10}")
    
    for size in sizes:
        for seed in seeds:
            instance = generate_random_instance(
                weapons=size, 
                targets=size, 
                seed=seed
            )
            
            # Run Greedy
            sol_greedy = solve_greedy(instance)
            
            # Run Exact (PuLP)
            sol_exact = solve_exact(instance, num_piecewise_segments=20)
            
            # Calculate gap: how much worse is greedy than exact?
            base_obj = sol_exact.objective_value if sol_exact.objective_value > 1e-9 else 1e-9          
            gap_pct = max(0.0, (sol_greedy.objective_value - sol_exact.objective_value) / base_obj * 100)
            
            print(f"{size:<10} | {seed:<6} | {sol_exact.runtime_seconds:<12.4f} | {sol_greedy.runtime_seconds:<12.4f} | {gap_pct:<10.2f}")
            
            results.append({
                "size": size,
                "seed": seed,
                "exact_time_s": sol_exact.runtime_seconds,
                "greedy_time_s": sol_greedy.runtime_seconds,
                "exact_obj": sol_exact.objective_value,
                "greedy_obj": sol_greedy.objective_value,
                "optimality_gap_pct": gap_pct
            })

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    return df

def plot_results(df):
    sns.set_theme(style="whitegrid")
    output_dir = Path("results")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="size", y="exact_time_s", marker="o", label="Exact (PuLP MILP)")
    sns.lineplot(data=df, x="size", y="greedy_time_s", marker="o", label="Heuristic (Greedy)")
    plt.yscale("log")
    plt.title("Execution Time Comparison (Log Scale)")
    plt.xlabel("Problem Size (Number of Weapons and Targets)")
    plt.ylabel("Time (seconds) - Logarythmic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="size", y="optimality_gap_pct", marker="o", color="red")
    plt.title("Optimality Gap Between Greedy and Exact")
    plt.xlabel("Problem Size (Number of Weapons and Targets)")
    plt.ylabel("Optimality Gap (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "optimality_gap.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    df_results = run_benchmark()
    plot_results(df_results)
