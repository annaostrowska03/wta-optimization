import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from wta_optimization.data import generate_random_instance, load_instance_from_file, load_andersen_instance
from wta_optimization.exact import solve_exact, solve_branch_and_adjust
from wta_optimization.heuristic import (
    solve_greedy,
    solve_local_search,
    solve_simulated_annealing,
)
from wta_optimization.models import WTAInstance, WTASolution


METHOD_SPECS = [
    ("greedy", "Greedy", "tab:orange"),
    ("ls", "Greedy + Local Search", "tab:blue"),
    ("sa", "Simulated Annealing", "tab:green"),
    ("exact", "Exact MIP (PuLP)", "black"),
    ("bna", "Branch & Adjust (Gurobi)", "tab:red"),
]

EXACT_TIME_LIMIT_SECONDS = 5400.0


def _numeric_file_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    numeric_part = int(match.group(1)) if match else 10**9
    return numeric_part, path.name


def _evaluate_methods(
    instance: WTAInstance,
    exact_time_limit_seconds: float = EXACT_TIME_LIMIT_SECONDS,
    use_exact_warm_start: bool = True,
) -> dict[str, WTASolution]:
    sol_greedy = solve_greedy(instance)
    sol_ls = solve_local_search(instance)
    sol_sa = solve_simulated_annealing(instance)
    sol_exact = solve_exact(
        instance,
        num_piecewise_segments=20,
        warm_start=sol_greedy if use_exact_warm_start else None,
        time_limit_seconds=exact_time_limit_seconds,
    )

    sol_bna = solve_branch_and_adjust(
        instance,
        warm_start=sol_greedy if use_exact_warm_start else None,
        time_limit_seconds=exact_time_limit_seconds,
    )

    return {
        "greedy": sol_greedy,
        "ls": sol_ls,
        "sa": sol_sa,
        "exact": sol_exact,
        "bna": sol_bna,
    }


def _gap_pct(candidate_obj: float, reference_obj: float) -> float:
    base_obj = reference_obj if reference_obj > 1e-9 else 1e-9
    return max(0.0, (candidate_obj - reference_obj) / base_obj * 100)


def _build_result_row(prefix_data: dict[str, int | str], solutions: dict[str, WTASolution]) -> dict[str, int | str | float]:
    exact_obj = solutions["exact"].objective_value
    row: dict[str, int | str | float] = dict(prefix_data)
    for method_key, _, _ in METHOD_SPECS:
        solution = solutions[method_key]
        row[f"{method_key}_time_s"] = solution.runtime_seconds
        row[f"{method_key}_obj"] = solution.objective_value
        row[f"{method_key}_status"] = solution.status
        if method_key != "exact":
            row[f"optimality_gap_pct_{method_key}"] = _gap_pct(solution.objective_value, exact_obj)
    return row


def _aggregate_for_plot(df: pd.DataFrame, from_file: bool) -> pd.DataFrame:
    if from_file:
        return df.copy()

    numeric_columns = [
        column
        for column in df.columns
        if column != "seed" and pd.api.types.is_numeric_dtype(df[column])
    ]
    aggregation = {
        column: "mean"
        for column in numeric_columns
        if column != "size"
    }
    return df.groupby("size", as_index=False).agg(aggregation)


def _to_tradeoff_frame(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method_key, label, color in METHOD_SPECS:
        records.append(
            {
                "method": label,
                "runtime_s": df[f"{method_key}_time_s"].mean(),
                "objective_value": df[f"{method_key}_obj"].mean(),
                "color": color,
            }
        )
    return pd.DataFrame(records)


def _target_loads(solution: WTASolution) -> list[int]:
    return [sum(row[target_index] for row in solution.assignment) for target_index in range(len(solution.assignment[0]))]


def _solve_and_record_sensitivity(
    scenario: str,
    instance: WTAInstance,
    primary_focus_target: int,
    primary_focus_label: str,
    secondary_focus_target: int | None = None,
    secondary_focus_label: str | None = None,
    exact_time_limit_seconds: float = EXACT_TIME_LIMIT_SECONDS,
    use_exact_warm_start: bool = True,
) -> list[dict[str, str | float | int]]:
    solutions = _evaluate_methods(
        instance,
        exact_time_limit_seconds=exact_time_limit_seconds,
        use_exact_warm_start=use_exact_warm_start,
    )
    exact_obj = solutions["exact"].objective_value
    rows = []

    for method_key, label, _ in METHOD_SPECS:
        solution = solutions[method_key]
        target_loads = _target_loads(solution)
        rows.append(
            {
                "scenario": scenario,
                "method": label,
                "runtime_s": solution.runtime_seconds,
                "objective_value": solution.objective_value,
                "optimality_gap_pct": 0.0 if method_key == "exact" else _gap_pct(solution.objective_value, exact_obj),
                "primary_focus_label": primary_focus_label,
                "primary_focus_allocations": target_loads[primary_focus_target],
                "secondary_focus_label": secondary_focus_label or "",
                "secondary_focus_allocations": target_loads[secondary_focus_target] if secondary_focus_target is not None else 0,
                "allocation_profile": ",".join(str(value) for value in target_loads),
            }
        )

    return rows


def run_benchmark(
    from_file: bool = False,
    dir_path: str | Path = "data/WTA",
    sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    exact_time_limit_seconds: float = EXACT_TIME_LIMIT_SECONDS,
    use_exact_warm_start: bool = True,
) -> pd.DataFrame:
    results = []

    if from_file:
        dir_path = Path(dir_path)
        files = sorted(dir_path.glob("*.txt"), key=_numeric_file_sort_key)

        print("Starting WTA Optimization Benchmark (From Files)...")
        print(f"Exact MIP time limit per instance: {exact_time_limit_seconds / 3600:.1f} h")
        print(f"Exact MIP warm start from greedy: {'yes' if use_exact_warm_start else 'no'}")
        print(
            f"{'File':<20} | {'Exact Time':<10} | {'Greedy T':<10} | {'LS Time':<10} | "
            f"{'SA Time':<10} | {'Gr Gap%':<8} | {'LS Gap%':<8} | {'SA Gap%':<8} | {'Exact Status':<12}"
        )

        for index, file in enumerate(files, start=1):
            print(f"Processing [{index}/{len(files)}]: {file.name}")
            instance = load_instance_from_file(file)
            solutions = _evaluate_methods(
                instance,
                exact_time_limit_seconds=exact_time_limit_seconds,
                use_exact_warm_start=use_exact_warm_start,
            )
            row = _build_result_row({"file": file.name}, solutions)
            results.append(row)

            print(
                f"{file.name:<20} | {row['exact_time_s']:<10.4f} | {row['greedy_time_s']:<10.4f} | "
                f"{row['ls_time_s']:<10.4f} | {row['sa_time_s']:<10.4f} | "
                f"{row['optimality_gap_pct_greedy']:<8.2f} | {row['optimality_gap_pct_ls']:<8.2f} | "
                f"{row['optimality_gap_pct_sa']:<8.2f} | {row['exact_status']:<12}"
            )

        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        df = pd.DataFrame(results)
        csv_path = output_dir / "benchmark_results_from_files.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        return df

    sizes = sizes or [5, 10, 15, 20, 25, 30]
    seeds = seeds or [42, 43, 44]

    print("Starting WTA Optimization Benchmark (Extended)...")
    print(f"Exact MIP time limit per instance: {exact_time_limit_seconds / 3600:.1f} h")
    print(f"Exact MIP warm start from greedy: {'yes' if use_exact_warm_start else 'no'}")
    print(
        f"{'Size':<5} | {'Seed':<4} | {'Exact Time':<10} | {'Greedy T':<10} | {'LS Time':<10} | "
        f"{'SA Time':<10} | {'Gr Gap%':<8} | {'LS Gap%':<8} | {'SA Gap%':<8} | {'Exact Status':<12}"
    )

    for size in sizes:
        for seed in seeds:
            instance = generate_random_instance(weapons=size, targets=size, seed=seed)
            solutions = _evaluate_methods(
                instance,
                exact_time_limit_seconds=exact_time_limit_seconds,
                use_exact_warm_start=use_exact_warm_start,
            )
            row = _build_result_row({"size": size, "seed": seed}, solutions)
            results.append(row)

            print(
                f"{size:<5} | {seed:<4} | {row['exact_time_s']:<10.4f} | {row['greedy_time_s']:<10.4f} | "
                f"{row['ls_time_s']:<10.4f} | {row['sa_time_s']:<10.4f} | "
                f"{row['optimality_gap_pct_greedy']:<8.2f} | {row['optimality_gap_pct_ls']:<8.2f} | "
                f"{row['optimality_gap_pct_sa']:<8.2f} | {row['exact_status']:<12}"
            )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return df


def plot_results(df: pd.DataFrame, from_file: bool = False) -> None:
    sns.set_theme(style="whitegrid")
    output_dir = Path("results")
    plot_df = _aggregate_for_plot(df, from_file=from_file)
    x_col = "file" if from_file else "size"
    x_label = "Input File" if from_file else "Problem Size (Number of Weapons and Targets)"
    time_title = "Execution Time Comparison" if from_file else "Execution Time Comparison (Log Scale)"

    plt.figure(figsize=(11, 6))
    for method_key, label, _ in METHOD_SPECS:
        plt.plot(plot_df[x_col], plot_df[f"{method_key}_time_s"], marker="o", label=label)
    if not from_file:
        plt.yscale("log")
    plt.title(time_title)
    plt.xlabel(x_label)
    plt.ylabel("Time (seconds)")
    plt.legend()
    if from_file:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / ("time_comparison_from_files.png" if from_file else "time_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for method_key, label, color in METHOD_SPECS:
        if method_key == "exact":
            continue
        plt.plot(plot_df[x_col], plot_df[f"optimality_gap_pct_{method_key}"], marker="o", color=color, label=f"{label} Gap %")
    plt.title("Optimality Gap Compared To Exact Solution")
    plt.xlabel(x_label)
    plt.ylabel("Optimality Gap (%)")
    plt.legend()
    if from_file:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / ("optimality_gap_from_files.png" if from_file else "optimality_gap.png"), dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def plot_tradeoff(df: pd.DataFrame, from_file: bool = False) -> None:
    sns.set_theme(style="whitegrid")
    output_dir = Path("results")
    tradeoff_df = _to_tradeoff_frame(df)

    plt.figure(figsize=(9, 6))
    for row in tradeoff_df.itertuples():
        plt.scatter(row.runtime_s, row.objective_value, s=120, color=row.color)
        plt.annotate(row.method, (row.runtime_s, row.objective_value), textcoords="offset points", xytext=(8, 6))
    plt.title("Time vs Objective Trade-off")
    plt.xlabel("Average Runtime (seconds)")
    plt.ylabel("Average Objective Value")
    plt.tight_layout()
    plt.savefig(output_dir / ("tradeoff_curve_from_files.png" if from_file else "tradeoff_curve.png"), dpi=300)
    plt.close()

    csv_path = output_dir / ("tradeoff_curve_from_files.csv" if from_file else "tradeoff_curve.csv")
    tradeoff_df.drop(columns=["color"]).to_csv(csv_path, index=False)
    print(f"Trade-off summary saved to {csv_path}")


def run_warm_start_study(
    sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    exact_time_limit_seconds: float = EXACT_TIME_LIMIT_SECONDS,
) -> pd.DataFrame:
    sizes = sizes or [10, 15, 20]
    seeds = seeds or [42, 43, 44]
    results = []

    print("\nStarting Warm Start Study...")
    print(f"{'Size':<5} | {'Seed':<4} | {'Cold MIP':<10} | {'Warm MIP':<10} | {'Speedup':<8}")

    for size in sizes:
        for seed in seeds:
            instance = generate_random_instance(weapons=size, targets=size, seed=seed)
            greedy_solution = solve_greedy(instance)
            cold_solution = solve_exact(
                instance,
                num_piecewise_segments=20,
                time_limit_seconds=exact_time_limit_seconds,
            )
            warm_solution = solve_exact(
                instance,
                num_piecewise_segments=20,
                warm_start=greedy_solution,
                time_limit_seconds=exact_time_limit_seconds,
            )

            speedup = cold_solution.runtime_seconds / max(warm_solution.runtime_seconds, 1e-9)
            results.append(
                {
                    "size": size,
                    "seed": seed,
                    "greedy_obj": greedy_solution.objective_value,
                    "exact_cold_time_s": cold_solution.runtime_seconds,
                    "exact_warm_time_s": warm_solution.runtime_seconds,
                    "exact_cold_obj": cold_solution.objective_value,
                    "exact_warm_obj": warm_solution.objective_value,
                    "exact_cold_status": cold_solution.status,
                    "exact_warm_status": warm_solution.status,
                    "speedup_ratio": speedup,
                    "warm_start_gap_pct": _gap_pct(greedy_solution.objective_value, cold_solution.objective_value),
                }
            )

            print(
                f"{size:<5} | {seed:<4} | {cold_solution.runtime_seconds:<10.4f} | "
                f"{warm_solution.runtime_seconds:<10.4f} | {speedup:<8.2f}"
            )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = output_dir / "warm_start_study.csv"
    df.to_csv(csv_path, index=False)
    print(f"Warm start results saved to: {csv_path}")
    return df


def plot_warm_start_study(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    output_dir = Path("results")
    plot_df = df.groupby("size", as_index=False).agg(
        exact_cold_time_s=("exact_cold_time_s", "mean"),
        exact_warm_time_s=("exact_warm_time_s", "mean"),
        speedup_ratio=("speedup_ratio", "mean"),
    )

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["size"], plot_df["exact_cold_time_s"], marker="o", label="Exact MIP (cold start)")
    plt.plot(plot_df["size"], plot_df["exact_warm_time_s"], marker="o", label="Exact MIP (warm start)")
    plt.title("Warm Start Impact on Solver Runtime")
    plt.xlabel("Problem Size")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "warm_start_time_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="size", y="speedup_ratio", color="tab:purple")
    plt.title("Warm Start Speedup")
    plt.xlabel("Problem Size")
    plt.ylabel("Cold / Warm Runtime Ratio")
    plt.tight_layout()
    plt.savefig(output_dir / "warm_start_speedup.png", dpi=300)
    plt.close()


def run_sensitivity_analysis(
    exact_time_limit_seconds: float = EXACT_TIME_LIMIT_SECONDS,
    use_exact_warm_start: bool = True,
) -> pd.DataFrame:
    high_value_base = generate_random_instance(weapons=12, targets=12, seed=2026)
    high_value_targets = list(high_value_base.target_values)
    high_value_targets[0] *= 100.0
    high_value_instance = WTAInstance(
        weapons=high_value_base.weapons,
        targets=high_value_base.targets,
        target_values=tuple(high_value_targets),
        destruction_probabilities=high_value_base.destruction_probabilities,
    )

    reliability_vs_quality_instance = WTAInstance(
        weapons=8,
        targets=3,
        target_values=(180.0, 45.0, 20.0),
        destruction_probabilities=(
            (0.45, 0.99, 0.15),
            (0.50, 0.98, 0.20),
            (0.55, 0.97, 0.15),
            (0.60, 0.96, 0.25),
            (0.72, 0.30, 0.40),
            (0.68, 0.25, 0.55),
            (0.75, 0.20, 0.35),
            (0.70, 0.25, 0.45),
        ),
    )

    rows = []
    rows.extend(
        _solve_and_record_sensitivity(
            scenario="High Value Target",
            instance=high_value_instance,
            primary_focus_target=0,
            primary_focus_label="Weapons on 100x target",
            exact_time_limit_seconds=exact_time_limit_seconds,
            use_exact_warm_start=use_exact_warm_start,
        )
    )
    rows.extend(
        _solve_and_record_sensitivity(
            scenario="Reliability vs Quality",
            instance=reliability_vs_quality_instance,
            primary_focus_target=0,
            primary_focus_label="Weapons on highest-value target",
            secondary_focus_target=1,
            secondary_focus_label="Weapons on most reliable target",
            exact_time_limit_seconds=exact_time_limit_seconds,
            use_exact_warm_start=use_exact_warm_start,
        )
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "sensitivity_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Sensitivity analysis saved to: {csv_path}")
    return df


def plot_sensitivity_analysis(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    output_dir = Path("results")

    plt.figure(figsize=(11, 6))
    sns.barplot(data=df, x="scenario", y="objective_value", hue="method")
    plt.title("Sensitivity Analysis: Objective Value by Scenario")
    plt.xlabel("")
    plt.ylabel("Objective Value")
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_objective.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(data=df, x="scenario", y="primary_focus_allocations", hue="method")
    plt.title("Sensitivity Analysis: Allocation to Primary Focus Target")
    plt.xlabel("")
    plt.ylabel("Assigned Weapons")
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_primary_focus.png", dpi=300)
    plt.close()

    secondary_df = df[df["secondary_focus_label"] != ""]
    if not secondary_df.empty:
        plt.figure(figsize=(11, 6))
        sns.barplot(data=secondary_df, x="scenario", y="secondary_focus_allocations", hue="method")
        plt.title("Sensitivity Analysis: Allocation to Secondary Focus Target")
        plt.xlabel("")
        plt.ylabel("Assigned Weapons")
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_secondary_focus.png", dpi=300)
        plt.close()


def run_andersen_benchmark(
    data_dir: str | Path = "data/data_andersen",
    exact_time_limit_seconds: float = EXACT_TIME_LIMIT_SECONDS,
    use_exact_warm_start: bool = True,
    methods: str = "both",  # "exact", "bna", or "both"
    bna_delta: float = 1e-5,  # Andersen Table 5 uses delta=0.00001
) -> pd.DataFrame:
    """Run solve_exact and/or solve_branch_and_adjust on all Andersen instance files.

    Files are expected in the format  wta_{W}x{T}x{seed}.txt  inside data_dir.
    Results are saved to results/benchmark_andersen.csv after every file.
    """
    data_dir = Path(data_dir)
    files = sorted(
        data_dir.glob("wta_*.txt"),
        key=lambda p: [int(x) for x in re.findall(r"\d+", p.stem)],
    )
    if not files:
        raise FileNotFoundError(f"No wta_*.txt files found in {data_dir}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "benchmark_andersen.csv"

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        results = existing_df.to_dict("records")
        done = {r["file"] for r in results if "error" not in r or pd.isna(r.get("error"))}
        print(f"Resuming — {len(done)} existing rows loaded from {csv_path}")
    else:
        results = []
        done: set[str] = set()

    print(f"Andersen benchmark — {len(files)} files, time limit {exact_time_limit_seconds:.0f}s")
    print(f"{'File':<22} | {'W':>4} | {'T':>4} | {'mu':>3} | {'Exact T':>9} | {'BnA T':>9} | {'Exact Obj':>11} | {'BnA Obj':>11}")

    for filepath in files:
        fname = filepath.name
        if fname in done:
            print(f"{fname:<22} | (skipped — already done)")
            continue

        # Parse W, T, mu from filename  wta_{W}x{T}x{mu}.txt
        nums = [int(x) for x in re.findall(r"\d+", filepath.stem)]
        weapons, targets, mu = nums[0], nums[1], nums[2]

        try:
            instance, mu = load_andersen_instance(filepath)
            mu_list = [mu] * instance.weapons

            sol_greedy = solve_greedy(instance) if use_exact_warm_start else None
            warm = sol_greedy if use_exact_warm_start else None

            run_exact = methods in ("exact", "both")
            run_bna = methods in ("bna", "both")

            sol_exact = (
                solve_exact(
                    instance,
                    num_piecewise_segments=20,
                    warm_start=warm,
                    time_limit_seconds=exact_time_limit_seconds,
                )
                if run_exact
                else None
            )
            sol_bna = (
                solve_branch_and_adjust(
                    instance,
                    delta=bna_delta,
                    warm_start=warm,
                    time_limit_seconds=exact_time_limit_seconds,
                    mu=mu_list,
                )
                if run_bna
                else None
            )

            row = {
                "file": fname,
                "weapons": weapons,
                "targets": targets,
                "mu": mu,
                "exact_time_s": sol_exact.runtime_seconds if sol_exact else float("nan"),
                "bna_time_s": sol_bna.runtime_seconds if sol_bna else float("nan"),
                "exact_obj": sol_exact.objective_value if sol_exact else float("nan"),
                "bna_obj": sol_bna.objective_value if sol_bna else float("nan"),
                "exact_status": sol_exact.status if sol_exact else "skipped",
                "bna_status": sol_bna.status if sol_bna else "skipped",
            }
            results.append(row)

            exact_t = f"{sol_exact.runtime_seconds:>9.2f}" if sol_exact else f"{'---':>9}"
            bna_t = f"{sol_bna.runtime_seconds:>9.2f}" if sol_bna else f"{'---':>9}"
            exact_o = f"{sol_exact.objective_value:>11.4f}" if sol_exact else f"{'---':>11}"
            bna_o = f"{sol_bna.objective_value:>11.4f}" if sol_bna else f"{'---':>11}"
            print(f"{fname:<22} | {weapons:>4} | {targets:>4} | {mu:>3} | {exact_t} | {bna_t} | {exact_o} | {bna_o}")
        except Exception as exc:
            print(f"{fname:<22} | ERROR: {exc}")
            results.append({"file": fname, "weapons": weapons, "targets": targets, "mu": mu, "error": str(exc)})

        pd.DataFrame(results).to_csv(csv_path, index=False)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WTA optimization benchmarks and analyses.")
    parser.add_argument(
        "--mode",
        choices=["files", "random", "warm", "sensitivity", "andersen", "all"],
        default="files",
        help="Select which analysis to run. Default: files.",
    )
    parser.add_argument(
        "--exact-limit-seconds",
        type=float,
        default=EXACT_TIME_LIMIT_SECONDS,
        help="Time limit for each exact MIP solve in seconds.",
    )
    parser.add_argument(
        "--no-exact-warm-start",
        action="store_true",
        help="Disable warm start from the greedy heuristic for exact benchmark solves.",
    )
    parser.add_argument(
        "--andersen-methods",
        choices=["exact", "bna", "both"],
        default="both",
        help="Which methods to run in andersen mode. Default: both.",
    )
    parser.add_argument(
        "--bna-delta",
        type=float,
        default=1e-5,
        help="Approximation parameter delta for Branch-and-Adjust. Andersen Table 5 uses 0.00001 (default).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    use_exact_warm_start = not args.no_exact_warm_start

    if args.mode in {"files", "all"}:
        df_results_from_files = run_benchmark(
            from_file=True,
            exact_time_limit_seconds=args.exact_limit_seconds,
            use_exact_warm_start=use_exact_warm_start,
        )
        plot_results(df_results_from_files, from_file=True)
        plot_tradeoff(df_results_from_files, from_file=True)

    if args.mode in {"random", "all"}:
        df_results = run_benchmark(
            exact_time_limit_seconds=args.exact_limit_seconds,
            use_exact_warm_start=use_exact_warm_start,
        )
        plot_results(df_results)
        plot_tradeoff(df_results)

    if args.mode in {"warm", "all"}:
        warm_start_df = run_warm_start_study(exact_time_limit_seconds=args.exact_limit_seconds)
        plot_warm_start_study(warm_start_df)

    if args.mode in {"sensitivity", "all"}:
        sensitivity_df = run_sensitivity_analysis(
            exact_time_limit_seconds=args.exact_limit_seconds,
            use_exact_warm_start=use_exact_warm_start,
        )
        plot_sensitivity_analysis(sensitivity_df)

    if args.mode == "andersen":
        run_andersen_benchmark(
            exact_time_limit_seconds=args.exact_limit_seconds,
            use_exact_warm_start=use_exact_warm_start,
            methods=args.andersen_methods,
            bna_delta=args.bna_delta,
        )


if __name__ == "__main__":
    main()
