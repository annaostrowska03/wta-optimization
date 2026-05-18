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

Our benchmark ran on the Bertsimas Scheme 2 instances across a range of N.
Instances where our methods hit the time limit are marked with open markers.

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

# 1.  Our benchmark results from Bertsimas instances 
df = pd.read_csv(RESULTS_DIR / "benchmark_bertsimas.csv")
df2 = df[df["scheme"] == "scheme2"].copy()

# Identify rows that reached the time limit for each method
for method in ("oa", "bna", "exact"):
    status_col = f"{method}_status"
    if status_col in df2.columns:
        df2[f"{method}_timed_out"] = ~df2[status_col].str.lower().str.contains("optimal", na=False)

# Aggregate per size: mean time
agg_cols = {col: "mean" for col in df2.columns
            if col not in ("scheme", "size", "seed")
            and pd.api.types.is_numeric_dtype(df2[col])}
ours = df2.groupby("size", as_index=False).agg(agg_cols)

# Add "any timed out" flag per size for each method
for method in ("oa", "bna", "exact"):
    timed_col = f"{method}_timed_out"
    if timed_col in df2.columns:
        timeout_by_size = df2.groupby("size")[timed_col].any()
        ours[f"{method}_any_timeout"] = ours["size"].map(timeout_by_size)

def split_optimal_timeout(ours_df, method):
    time_col = f"{method}_time_s"
    to_col = f"{method}_any_timeout"
    if to_col not in ours_df.columns:
        return ours_df, pd.DataFrame()
    opt = ours_df[~ours_df[to_col].fillna(False)]
    tmo = ours_df[ours_df[to_col].fillna(False)]
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
    [ours[["size", "oa_time_s", "bna_time_s", "exact_time_s"]].rename(columns={"size": "n"}),
     export.assign(oa_time_s=np.nan, bna_time_s=np.nan, exact_time_s=np.nan).rename(columns={"size": "n"})],
    ignore_index=True,
)
combined.to_csv(RESULTS_DIR / "comparison_bpc_vs_oa.csv", index=False)
print(f"Saved → {RESULTS_DIR / 'comparison_bpc_vs_oa.csv'}")
print("\nOur methods (Scheme 2, mean over seeds):")
print(ours[["size", "oa_time_s", "bna_time_s", "exact_time_s"]].to_string(index=False))

# --------------------------------------------------------------------------
# 4.  Plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

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
                xytext=(10, -14), textcoords="offset points",
                color="tab:green", fontsize=8)

# Time limit reference line (if any timeouts in our data)
if "oa_any_timeout" in ours.columns and ours["oa_any_timeout"].any():
    tl = df2["oa_time_s"].max()
    ax.axhline(y=tl, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(ours["size"].min() + 1, tl * 1.05, f"our time limit ≈ {tl:.0f}s",
            color="gray", fontsize=8, alpha=0.8)

ax.set_yscale("log")
ax.set_xlabel("Problem size  N  (weapons = targets)", fontsize=12)
ax.set_ylabel("Solve time (seconds, log scale)", fontsize=12)

n_our = f"{ours['size'].min()}–{ours['size'].max()}"
ax.set_title(
    f"Solve Time: Our Methods (N={n_our}) vs Bertsimas & Paskov (2025) BPC (N=200–450)\n"
    "Open markers = time limit reached (solution may not be optimal)",
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
plt.show()
