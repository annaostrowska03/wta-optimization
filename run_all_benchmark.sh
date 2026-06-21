#!/usr/bin/env bash
set +e

FILES=(
  wta_50x100x1.txt
  wta_50x100x2.txt
  wta_50x100x3.txt
  wta_100x200x1.txt
  wta_100x200x2.txt
  wta_100x200x3.txt
  wta_150x300x1.txt
  wta_150x300x2.txt
  wta_150x300x3.txt
  wta_200x400x1.txt
  wta_200x400x2.txt
  wta_200x400x3.txt
  wta_250x500x1.txt
  wta_250x500x2.txt
  wta_250x500x3.txt
  wta_300x600x1.txt
  wta_300x600x2.txt
  wta_300x600x3.txt
  wta_350x700x1.txt
  wta_350x700x2.txt
  wta_350x700x3.txt
  wta_400x800x1.txt
  wta_400x800x2.txt
  wta_400x800x3.txt
  wta_450x900x1.txt
  wta_450x900x2.txt
  wta_450x900x3.txt
  wta_500x1000x1.txt
  wta_500x1000x2.txt
  wta_500x1000x3.txt
)

mkdir -p results/final_logs

run_benchmark() {
  local method="$1"
  local results_file="$2"
  local log_file="$3"

  echo "=== START method=${method} ==="

  uv run python -u benchmark.py \
    --method "${method}" \
    --files "${FILES[@]}" \
    --results-file "${results_file}" \
    2>&1 | tee "${log_file}"

  local code=${PIPESTATUS[0]}

  if [ "$code" -eq 0 ]; then
    echo "OK: method=${method}"
  else
    echo "FAILED/KILLED: method=${method}, exit code ${code}"
    echo "${method},${code}" >> results/killed_or_failed_methods.txt
  fi

  echo "=== END method=${method} ==="
  echo

  return "$code"
}

run_benchmark \
  bna \
  final_benchmark_andersen_rerun.csv \
  results/final_logs/benchmark_andersen_rerun.log

code_bna=$?

run_benchmark \
  bna_v2 \
  final_benchmark_andersen_v2_rerun.csv \
  results/final_logs/benchmark_andersen_v2_rerun.log

code_bna_v2=$?

if [ "$code_bna" -ne 0 ] || [ "$code_bna_v2" -ne 0 ]; then
  exit 1
fi

exit 0
