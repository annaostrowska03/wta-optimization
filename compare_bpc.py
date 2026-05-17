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

Our benchmark ran on the Bertsimas Scheme 1/2 instances for N=5-30.
This script overlays both datasets on a single log-scale chart.

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

# 1.  Our benchmark results from Bertsimas instances (N=5-30)
df = pd.read_csv(RESULTS_DIR / "benchmark_bertsimas.csv")

# Keep only Scheme 2 (closer to Andersen/BPC benchmark distribution)
df2 = df[df["scheme"] == "scheme2"].copy()

# Average across seeds per size
ours = (
    df2.groupby("size")
    .agg(
        oa_time=("oa_time_s", "mean"),
        bna_time=("bna_time_s", "mean"),
        exact_time=("exact_time_s", "mean"),
    )
    .reset_index()
)

# 2.  Bertsimas BPC Table 1 data (N×N square instances, Andersen benchmark)
bpc_data = pd.DataFrame({
    "size": [200, 250, 300, 350, 400, 450],
    "bpc_time": [0.058, 0.082, 0.124, 0.167, 0.232, 0.290],
})

# BnA in paper uses BA = Branch-and-Adjust; for Andersen instances:
# BA 200: 419.9 s, 250: 541.1 s, 300: 1859.5 s, 350: 2414.4 s, 400: 1844.9 s
ba_andersen = pd.DataFrame({
    "size": [200, 250, 300, 350, 400],
    "ba_time": [419.9, 541.1, 1859.5, 2414.4, 1844.9],
})

export = pd.merge(bpc_data, ba_andersen, on="size", how="outer")
print("BPC (Bertsimas) vs BA (Andersen) — large instances:")
print(export.to_string(index=False))
print()
print("Our methods — small instances (Scheme 2, mean over 3 seeds):")
print(ours.to_string(index=False))

combined = pd.concat(
    [
        ours.rename(columns={"size": "n"}),
        export.assign(
            oa_time=np.nan,
            bna_time=np.nan,
            exact_time=np.nan,
        ).rename(columns={"size": "n"}),
    ],
    ignore_index=True,
)
combined.to_csv(RESULTS_DIR / "comparison_bpc_vs_oa.csv", index=False)
print(f"\nSaved → {RESULTS_DIR / 'comparison_bpc_vs_oa.csv'}")

fig, ax = plt.subplots(figsize=(10, 6))

# Our methods (N = 5–30, left cluster)
ax.plot(
    ours["size"], ours["oa_time"],
    "o-", color="tab:purple", linewidth=2, markersize=7,
    label="Our OA — Outer Approximation (N=5–30)",
)
ax.plot(
    ours["size"], ours["bna_time"],
    "s--", color="tab:orange", linewidth=2, markersize=7,
    label="Our BnA — Branch & Adjust (N=5–30)",
)
ax.plot(
    ours["size"], ours["exact_time"],
    "^:", color="tab:blue", linewidth=1.5, markersize=6,
    label="Our Exact MIP — PuLP/CBC (N=5–30)",
)

# BPC from Bertsimas 2025 (N = 200–450)
ax.plot(
    bpc_data["size"], bpc_data["bpc_time"],
    "D-", color="tab:green", linewidth=2.5, markersize=9,
    label="BPC — Bertsimas & Paskov (2025), N=200–450",
)

# BA from Andersen 2022 (N = 200–400, timeout = 7200 s marked separately)
ba_no_timeout = ba_andersen[ba_andersen["ba_time"] < 7200]
ax.plot(
    ba_no_timeout["size"], ba_no_timeout["ba_time"],
    "x--", color="tab:red", linewidth=1.5, markersize=9, markeredgewidth=2,
    label="Branch-and-Adjust (Andersen 2022) — before timeout",
)

# Annotate BPC points with exact times
for _, row in bpc_data.iterrows():
    ax.annotate(
        f"{row['bpc_time']:.3f}s",
        xy=(row["size"], row["bpc_time"]),
        xytext=(12, -14),
        textcoords="offset points",
        color="tab:green",
        fontsize=8,
    )

# Timeout annotation for BA
ax.axhline(y=7200, color="tab:red", linestyle=":", alpha=0.4, linewidth=1)
ax.text(210, 7200 * 1.05, "BA timeout (2 h)", color="tab:red", fontsize=8, alpha=0.7)

ax.set_yscale("log")
ax.set_xlabel("Problem size  N  (weapons = targets)", fontsize=12)
ax.set_ylabel("Solve time (seconds, log scale)", fontsize=12)
ax.set_title(
    "Solve Time Comparison: Outer Approximation / BnA / BPC\n"
    "Our methods (N=5–30) vs Bertsimas & Paskov (2025) BPC (N=200–450)",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, which="both", alpha=0.3)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

# Second x-axis note
ax.text(
    0.98, 0.05,
    "Note: BPC uses Gurobi LP solver internally.\nOur methods use SCIP / PuLP/CBC (open-source).",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=8, color="gray",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
)

fig.tight_layout()
out_path = RESULTS_DIR / "comparison_bpc_vs_oa.png"
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
