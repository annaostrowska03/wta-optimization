import argparse
import re
from pathlib import Path

import pandas as pd

from wta_optimization.data import load_andersen_instance
from wta_optimization.exact import solve_branch_and_adjust
from wta_optimization.exact_v2 import solve_branch_and_adjust_v2

DEFAULT_TIME_LIMIT = 7200.0


def run_andersen_benchmark(
    data_dir: str | Path = "data/data_andersen",
    time_limit_seconds: float = DEFAULT_TIME_LIMIT,
    method: str = "bna",  # "bna" or "bna_v2"
    bna_delta: float = 1e-5,
) -> pd.DataFrame:
    """Run BnA (exact.py) or BnA-v2 (exact_v2.py) on all 30 Andersen instance files.

    Files are expected in the format  wta_{W}x{T}x{mu}.txt  inside data_dir.
    Results are saved to results/benchmark_andersen.csv or benchmark_andersen_v2.csv
    after every file (resume-capable).
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
    csv_name = "benchmark_andersen_v2.csv" if method == "bna_v2" else "benchmark_andersen.csv"
    csv_path = output_dir / csv_name

    solver_fn = solve_branch_and_adjust_v2 if method == "bna_v2" else solve_branch_and_adjust

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        results = existing_df.to_dict("records")
        done = {r["file"] for r in results if "error" not in r or pd.isna(r.get("error"))}
        print(f"Resuming — {len(done)} existing rows loaded from {csv_path}")
    else:
        results = []
        done: set[str] = set()

    print(f"Andersen benchmark ({method}) — {len(files)} files, time limit {time_limit_seconds:.0f}s")
    print(f"{'File':<22} | {'W':>4} | {'T':>4} | {'mu':>3} | {'Time [s]':>10} | {'Objective':>12} | Status")

    for filepath in files:
        fname = filepath.name
        if fname in done:
            print(f"{fname:<22} | (skipped — already done)")
            continue

        nums = [int(x) for x in re.findall(r"\d+", filepath.stem)]
        weapons, targets, mu_val = nums[0], nums[1], nums[2]

        try:
            instance, mu = load_andersen_instance(filepath)
            mu_list = [mu] * instance.weapons

            sol = solver_fn(
                instance,
                delta=bna_delta,
                warm_start=None,
                time_limit_seconds=time_limit_seconds,
                mu=mu_list,
            )

            row = {
                "file": fname,
                "weapons": weapons,
                "targets": targets,
                "mu": mu,
                "bna_time_s": sol.runtime_seconds,
                "bna_obj": sol.objective_value,
                "bna_status": sol.status,
            }
            results.append(row)
            print(f"{fname:<22} | {weapons:>4} | {targets:>4} | {mu:>3} | {sol.runtime_seconds:>10.2f} | {sol.objective_value:>12.4f} | {sol.status}")
        except Exception as exc:
            print(f"{fname:<22} | ERROR: {exc}")
            results.append({"file": fname, "weapons": weapons, "targets": targets, "mu": mu_val, "error": str(exc)})

        pd.DataFrame(results).to_csv(csv_path, index=False)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    return df


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Andersen benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark Branch-and-Adjust on Andersen et al. (2022) instances."
    )
    parser.add_argument(
        "--method",
        choices=["bna", "bna_v2"],
        default="bna",
        help="Solver: bna (exact.py) or bna_v2 (exact_v2.py, faster with cbSetSolution). Default: bna.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=f"Time limit per instance in seconds (default: {DEFAULT_TIME_LIMIT:.0f}).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Approximation parameter delta (default: 0.00001, matching Andersen Table 5).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/data_andersen",
        help="Directory containing wta_WxTxmu.txt instance files.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — parse arguments and run the benchmark."""
    args = _parse_args()
    run_andersen_benchmark(
        data_dir=args.data_dir,
        time_limit_seconds=args.time_limit,
        method=args.method,
        bna_delta=args.delta,
    )


if __name__ == "__main__":
    main()
