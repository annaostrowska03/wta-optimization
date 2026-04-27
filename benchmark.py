import csv
from pathlib import Path
from time import perf_counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wta_optimization.data import generate_random_instance
from wta_optimization.exact import solve_exact
from wta_optimization.heuristic import solve_greedy, solve_local_search


def run_benchmark():
    sizes = [5, 10, 15, 20, 25, 30]  # List of grid sizes (weapons & targets)
    seeds = [42, 43, 44]            # Multiple seeds to average out the results
    
    results = []
    
    print("Starting WTA Optimization Benchmark (Extended)...")
    print("-" * 80)
    print(f"{'Size':<5} | {'Seed':<4} | {'Exact Time':<10} | {'Greedy T':<10} | {'LS Time':<10} | {'Gr Gap%':<8} | {'LS Gap%':<8}")
    print("-" * 80)
    
    for size in sizes:
        for seed in seeds:
            instance = generate_random_instance(
                weapons=size, 
                targets=size, 
                seed=seed
            )
            
            sol_greedy = solve_greedy(instance)
            sol_ls = solve_local_search(instance)
            sol_exact = solve_exact(instance, num_piecewise_segments=20)
            
            base_obj = sol_exact.objective_value if sol_exact.objective_value > 1e-9 else 1e-9
            
            gap_pct_greedy = max(0.0, (sol_greedy.objective_value - sol_exact.objective_value) / base_obj * 100)
            gap_pct_ls = max(0.0, (sol_ls.objective_value - sol_exact.objective_value) / base_obj * 100)
            
            print(f"{size:<5} | {seed:<4} | {sol_exact.runtime_seconds:<10.4f} | {sol_greedy.runtime_seconds:<10.4f} | {sol_ls.runtime_seconds:<10.4f} | {gap_pct_greedy:<8.2f} | {gap_pct_ls:<8.2f}")
            
            results.append({
                "size": size,
                "seed": seed,
                "exact_time_s": sol_exact.runtime_seconds,
                "greedy_time_s": sol_greedy.runtime_seconds,
                "ls_time_s": sol_ls.runtime_seconds,
                "exact_obj": sol_exact.objective_value,
                "greedy_obj": sol_greedy.objective_value,
                "ls_obj": sol_ls.objective_value,
                "optimality_gap_pct_greedy": gap_pct_greedy,
                "optimality_gap_pct_ls": gap_pct_ls
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
    sns.lineplot(data=df, x="size", y="ls_time_s", marker="o", label="Heuristic (Local Search)")
    plt.yscale("log")
    plt.title("Execution Time Comparison (Log Scale)")
    plt.xlabel("Problem Size (Number of Weapons and Targets)")
    plt.ylabel("Time (seconds) - Logarythmic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="size", y="optimality_gap_pct_greedy", marker="o", color="red", label="Greedy Gap %")
    sns.lineplot(data=df, x="size", y="optimality_gap_pct_ls", marker="o", color="blue", label="Local Search Gap %")
    plt.title("Optimality Gap Compared To Exact Solution")
    plt.xlabel("Problem Size (Number of Weapons and Targets)")
    plt.ylabel("Optimality Gap (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "optimality_gap.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    df_results = run_benchmark()
    plot_results(df_results)
