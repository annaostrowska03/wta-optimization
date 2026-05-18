"""
Comparison: Our OA / BnA methods vs Bertsimas & Paskov (2025) BPC.

BPC timing data is taken directly from Table 1 of:
  Bertsimas, D. & Paskov, I. (2025). "Solving Large-Scale Weapon-Target
  Assignment Problems in Seconds Using Branch-Price-And-Cut."
  Naval Research Logistics (NRL).

For N×N square instances (targets = weapons) from the Andersen benchmark:
  N=200 → BPC 0.058 s     N=300 → BPC 0.124 s
  N=350 → BPC 0.167 s     N=400 → BPC 0.232 s
  N=450 → BPC 0.290 s

Data source priority:
  1. benchmark_results_from_files.csv  — Andersen's actual wta*.txt instances
     (SAME instances as used by BPC/BA in Table 1 of the paper)
  2. benchmark_bertsimas.csv  — random Scheme-2 instances (fallback)

Usage:
    python compare_bpc.py
Output:
    results/comparison_bpc_vs_oa.png
    results/comparison_bpc_vs_oa.csv
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# 1.  Load our benchmark results — prefer file-based (same instances as paper)
# --------------------------------------------------------------------------
files_csv = RESULTS_DIR / "benchmark_results_from_files.csv"
bertsimas_csv = RESULTS_DIR / "benchmark_bertsimas.csv"

use_file_instances = False
if files_csv.exists():
    fdf = pd.read_csv(files_csv)
    if "oa_time_s" in fdf.columns and "bna_time_s" in fdf.columns:
        # Extract numeric N from filename (wta50.txt → 50)
        fdf["size"] = fdf["file"].str.extract(r"(\d+)").astype(int)
        # Keep only rows without errors
        fdf = fdf[fdf["error"].isna()] if "error" in fdf.columns else fdf
        if not fdf.empty:
            ours = fdf[["size", "oa_time_s", "bna_time_s", "exact_time_s",
                        "oa_status", "bna_status", "exact_status"]].copy()
            use_file_instances = True
            print("Using Andersen wta*.txt file instances (same data as paper Table 1).")

if not use_file_instances:
    df = pd.read_csv(bertsimas_csv)
    df2 = df[df["scheme"] == "scheme2"].copy()
    agg_cols = {col: "mean" for col in df2.columns
                if col not in ("scheme", "size", "seed")
                and pd.api.types.is_numeric_dtype(df2[col])}
    ours = df2.groupby("size", as_index=False).agg(agg_cols)
    print("Using random Bertsimas Scheme-2 instances (file benchmark not yet run).")

# Identify rows that reached the time limit for each method
for method in ("oa", "bna", "exact"):
    status_col = f"{method}_status"
    if status_col in ours.columns:
        ours[f"{method}_timed_out"] = ~ours[status_col].str.lower().str.contains("optimal", na=False)
    else:
        ours[f"{method}_timed_out"] = False

def split_optimal_timeout(df, method):
    to_col = f"{method}_timed_out"
    opt = df[~df[to_col].fillna(False)]
    tmo = df[df[to_col].fillna(False)]
    return opt, tmo

oa_opt,  oa_tmo  = split_optimal_timeout(ours, "oa")
bna_opt, bna_tmo = split_optimal_timeout(ours, "bna")

# --------------------------------------------------------------------------
# 2.  Bertsimas BPC Table 1 — N×N square instances (Andersen benchmark)
# --------------------------------------------------------------------------
bpc_data = pd.DataFrame({
    "size": [200, 250, 300, 350, 400, 450],
    "bpc_time": [0.058, 0.082, 0.124, 0.167, 0.232, 0.290],
})

# Branch-and-Adjust (Andersen 2022) on same instances
ba_andersen = pd.DataFrame({
    "size": [200, 250, 300, 350, 400],
    "ba_time": [419.9, 541.1, 1859.5, 2414.4, 1844.9],
})

# --------------------------------------------------------------------------
# 3.  Export CSV
# --------------------------------------------------------------------------
export = pd.merge(bpc_data, ba_andersen, on="size", how="outer")
combined = pd.concat(
    [ours[["size", "oa_time_s", "bna_time_s", "exact_time_s"]].assign(bpc_time=np.nan, ba_time=np.nan),
     export.assign(oa_time_s=np.nan, bna_time_s=np.nan, exact_time_s=np.nan)],
    ignore_index=True,
).sort_values("size")
combined.to_csv(RESULTS_DIR / "comparison_bpc_vs_oa.csv", index=False)
print(f"Saved → {RESULTS_DIR / 'comparison_bpc_vs_oa.csv'}")

label = "Andersen wta files" if use_file_instances else "Scheme-2 random (mean)"
print(f"\nOur methods ({label}):")
print(ours[["size", "oa_time_s", "bna_time_s", "exact_time_s"]].to_string(index=False))

print(f"\nBPC (Bertsimas) vs BA (Andersen) — large instances:")
print(pd.merge(bpc_data, ba_andersen, on="size", how="outer").to_string(index=False))

# --------------------------------------------------------------------------
# 4.  Plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

# — OA: optimal (solid) and timeout (open marker)
if not oa_opt.empty:
    ax.plot(oa_opt["size"], oa_opt["oa_time_s"],
            "o-", color="tab:purple", linewidth=2, markersize=7,
            label="Our OA — optimal")
if not oa_tmo.empty:
    ax.plot(oa_tmo["size"], oa_tmo["oa_time_s"],
            "o--", color="tab:purple", linewidth=1.5, markersize=7,
            markerfacecolor="none", markeredgewidth=2,
            label="Our OA — time limit reached")

# — BnA: optimal (solid) and timeout (open marker)
if not bna_opt.empty:
    ax.plot(bna_opt["size"], bna_opt["bna_time_s"],
            "s-", color="tab:orange", linewidth=2, markersize=7,
            label="Our BnA — optimal")
if not bna_tmo.empty:
    ax.plot(bna_tmo["size"], bna_tmo["bna_time_s"],
            "s--", color="tab:orange", linewidth=1.5, markersize=7,
            markerfacecolor="none", markeredgewidth=2,
            label="Our BnA — time limit reached")

# — BPC (Bertsimas 2025)
ax.plot(bpc_data["size"], bpc_data["bpc_time"],
        "D-", color="tab:green", linewidth=2.5, markersize=9,
        label="BPC — Bertsimas & Paskov (2025)")

# — BA Andersen 2022
ax.plot(ba_andersen["size"], ba_andersen["ba_time"],
        "x--", color="tab:red", linewidth=1.5, markersize=9, markeredgewidth=2,
        label="Branch-and-Adjust (Andersen 2022)")

# Annotate BPC points
for _, row in bpc_data.iterrows():
    ax.annotate(f"{row['bpc_time']:.3f}s",
                xy=(row["size"], row["bpc_time"]),
                xytext=(8, -14), textcoords="offset points",
                color="tab:green", fontsize=8)

# Time limit reference line (if any timeouts in our data)
if ours["oa_timed_out"].any() or ours["bna_timed_out"].any():
    tl = ours[["oa_time_s", "bna_time_s"]].max().max()
    ax.axhline(y=tl, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(ours["size"].min() + 1, tl * 1.05, f"our time limit ≈ {tl:.0f}s",
            color="gray", fontsize=8, alpha=0.8)

ax.set_yscale("log")
ax.set_xlabel("Problem size  N  (weapons = targets)", fontsize=12)
ax.set_ylabel("Solve time (seconds, log scale)", fontsize=12)

data_label = "Andersen wta*.txt — same instances as paper" if use_file_instances else "random Scheme-2"
n_our = f"{ours['size'].min()}–{ours['size'].max()}"
ax.set_title(
    f"Solve Time: Our Methods (N={n_our}, {data_label})\nvs Bertsimas & Paskov (2025) BPC (N=200–450)",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, which="both", alpha=0.3)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.text(0.98, 0.04,
        "Note: BPC uses Gurobi LP internally.\nOur methods use open-source SCIP / PuLP/CBC.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

fig.tight_layout()
out_path = RESULTS_DIR / "comparison_bpc_vs_oa.png"
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
