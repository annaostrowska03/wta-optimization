import argparse
from multiprocessing import Process, Queue
from queue import Empty
import re
from pathlib import Path

import pandas as pd

from wta_optimization.data import load_andersen_instance
from wta_optimization.exact import solve_branch_and_adjust
from wta_optimization.exact_v2 import solve_branch_and_adjust_v2

DEFAULT_TIME_LIMIT: float | None = None


def _solve_single_instance_worker(
    filepath_str: str,
    method: str,
    bna_delta: float,
    time_limit_seconds: float | None,
    out_queue: Queue,
) -> None:
    """Solve one instance in a child process so parent survives worker crashes."""
    try:
        filepath = Path(filepath_str)
        nums = [int(x) for x in re.findall(r"\d+", filepath.stem)]
        weapons, targets, _ = nums[0], nums[1], nums[2]

        instance, mu = load_andersen_instance(filepath)
        mu_list = [mu] * instance.weapons
        solver_fn = (
            solve_branch_and_adjust_v2 if method == "bna_v2" else solve_branch_and_adjust
        )
        sol = solver_fn(
            instance,
            delta=bna_delta,
            warm_start=None,
            time_limit_seconds=time_limit_seconds,
            mu=mu_list,
        )
        out_queue.put(
            {
                "ok": True,
                "file": filepath.name,
                "weapons": weapons,
                "targets": targets,
                "mu": mu,
                "bna_time_s": sol.runtime_seconds,
                "bna_obj": sol.objective_value,
                "bna_status": sol.status,
            }
        )
    except Exception as exc:
        out_queue.put({"ok": False, "error": str(exc)})


def run_andersen_benchmark(
    data_dir: str | Path = "data/data_andersen",
    time_limit_seconds: float | None = DEFAULT_TIME_LIMIT,
    method: str = "bna",  # "bna" or "bna_v2"
    bna_delta: float = 1e-5,
    files: list[str] | None = None,
    results_file: str | None = None,
) -> pd.DataFrame:
    """Run BnA (exact.py) or BnA-v2 (exact_v2.py) on all 30 Andersen instance files.

    Files are expected in the format  wta_{W}x{T}x{mu}.txt  inside data_dir.
    Results are saved to results/benchmark_andersen.csv or benchmark_andersen_v2.csv
    after every file (resume-capable).
    """
    data_dir = Path(data_dir)

    all_files = sorted(
        data_dir.glob("wta_*.txt"),
        key=lambda p: [int(x) for x in re.findall(r"\d+", p.stem)],
    )
    if files is not None:
        files_set = set(files)
        files = [p for p in all_files if p.name in files_set]
        if not files:
            raise FileNotFoundError(
                f"No matching files found in {data_dir} for --files {files}"
            )
    else:
        files = all_files
    if not files:
        raise FileNotFoundError(f"No wta_*.txt files found in {data_dir}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    if results_file is not None:
        csv_path = output_dir / results_file
    else:
        csv_name = (
            "benchmark_andersen_v2.csv"
            if method == "bna_v2"
            else "benchmark_andersen.csv"
        )
        csv_path = output_dir / csv_name

    # If using a custom results file, always rerun all requested files
    if results_file is not None:
        results = []
        done: set[str] = set()
    elif csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        results = existing_df.to_dict("records")
        done = {
            r["file"] for r in results if "error" not in r or pd.isna(r.get("error"))
        }
        print(f"Resuming — {len(done)} existing rows loaded from {csv_path}")
    else:
        results = []
        done: set[str] = set()

    time_limit_label = (
        "none" if time_limit_seconds is None else f"{time_limit_seconds:.0f}s"
    )
    print(
        f"Andersen benchmark ({method}) — {len(files)} files, time limit {time_limit_label}"
    )
    print(
        f"{'File':<22} | {'W':>4} | {'T':>4} | {'mu':>3} | {'Time [s]':>10} | {'Objective':>12} | Status"
    )

    for filepath in files:
        fname = filepath.name
        if fname in done:
            print(f"{fname:<22} | (skipped — already done)")
            continue

        nums = [int(x) for x in re.findall(r"\d+", filepath.stem)]
        weapons, targets, mu_val = nums[0], nums[1], nums[2]

        q: Queue = Queue(maxsize=1)
        proc = Process(
            target=_solve_single_instance_worker,
            args=(str(filepath), method, bna_delta, time_limit_seconds, q),
        )
        proc.start()
        proc.join()

        payload: dict
        if proc.exitcode == 0:
            try:
                payload = q.get_nowait()
            except Empty:
                payload = {
                    "ok": False,
                    "error": "worker_finished_without_result",
                }
        else:
            payload = {
                "ok": False,
                "error": f"worker_crashed_exit_code_{proc.exitcode}",
            }

        q.close()
        q.join_thread()

        if payload.get("ok"):
            results.append(payload)
            print(
                f"{fname:<22} | {payload['weapons']:>4} | {payload['targets']:>4} | {payload['mu']:>3} | {payload['bna_time_s']:>10.2f} | {payload['bna_obj']:>12.4f} | {payload['bna_status']}"
            )
        else:
            error_text = str(payload.get("error", "unknown_error"))
            print(f"{fname:<22} | ERROR: {error_text}")
            results.append(
                {
                    "file": fname,
                    "weapons": weapons,
                    "targets": targets,
                    "mu": mu_val,
                    "error": error_text,
                }
            )

        pd.DataFrame(results).to_csv(csv_path, index=False)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    return df


def _parse_args() -> argparse.Namespace:
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
        help="Time limit per instance in seconds (default: none / unlimited).",
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
    parser.add_argument(
        "--files",
        nargs="+",
        help="List of specific instance filenames to run (from data-dir). If omitted, runs all.",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Custom CSV file for results (in results/). If set, always reruns all requested files.",
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
        files=args.files,
        results_file=args.results_file,
    )


if __name__ == "__main__":
    main()
