"""
Comparison of Branch-and-Adjust results (Gurobi, our implementation)
against Andersen et al. (2022), Table 5, δ=0.00001.

Usage:
    python compare_andersen.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).parent
ANDERSEN_R = ROOT / "data" / "results.csv"
OUR_CSV = ROOT / "results" / "benchmark_andersen.csv"
OUT_DIR = ROOT / "results"


# Load Andersen reference data
def load_andersen_ref(path: Path) -> pd.DataFrame:
    """Parse Andersen et al. (2022) reference CSV for the BnA δ=0.00001 section."""
    rows = []
    in_section = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "Approach: branch-and-adjust_delta_0.00001":
                in_section = True
                continue
            if in_section and line.startswith("Approach:"):
                break
            if not in_section:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            try:
                W = int(parts[1])
                T = int(parts[2])
                mu = int(parts[3])
                lb = float(parts[5])
                ub = float(parts[6])
                gap = parts[7].strip()
                t = float(parts[8])
            except (ValueError, IndexError):
                continue
            rows.append(
                dict(
                    weapons=W,
                    targets=T,
                    mu=mu,
                    and_lb=lb,
                    and_ub=ub,
                    and_gap=gap,
                    and_time=t,
                )
            )
    return pd.DataFrame(rows)


and_df = load_andersen_ref(ANDERSEN_R)
our_df = pd.read_csv(OUR_CSV)

df = pd.merge(
    our_df[["weapons", "targets", "mu", "bna_time_s", "bna_obj", "bna_status"]],
    and_df,
    on=["weapons", "targets", "mu"],
)

df["instance"] = df.apply(lambda r: f"{r.weapons}×{r.targets}", axis=1)
df["ub_diff_pct"] = (df["bna_obj"] - df["and_ub"]) / df["and_ub"] * 100
df["time_ratio"] = df["bna_time_s"] / df["and_time"]

status_map = {"optimal": "opt✓", "time_limit": "TL", "mem_limit": "MEM"}
df["our_status_short"] = df["bna_status"].map(status_map).fillna(df["bna_status"])
df["and_status_short"] = df["and_gap"].apply(lambda g: "opt✓" if g == "opt" else g)

table = df[
    [
        "instance",
        "mu",
        "and_lb",
        "and_ub",
        "and_status_short",
        "and_time",
        "bna_obj",
        "our_status_short",
        "bna_time_s",
        "ub_diff_pct",
        "time_ratio",
    ]
].copy()

table.columns = [
    "Instance",
    "μ",
    "And. LB",
    "And. UB",
    "And. status",
    "And. time [s]",
    "Our UB",
    "Our status",
    "Our time [s]",
    "Δ UB [%]",
    "Time ratio",
]

# display formatting
pd.set_option("display.float_format", "{:.4g}".format)
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 160)

print("\n" + "=" * 90)
print("  Branch-and-Adjust comparison: our implementation vs Andersen et al. (2022)")
print("  δ = 0.00001,  time limit = 7200 s")
print("=" * 90)
print(table.to_string(index=False))
print("=" * 90)
print("  Δ UB [%] = (Our UB − And. UB) / And. UB × 100%  (negative = ours better)")
print("  Time ratio = Our time / And. time  (>1 = slower)")
print()

# Save comparison CSV
out_table = OUT_DIR / "comparison_andersen.csv"
table.to_csv(out_table, index=False)
print(f"  Table saved → {out_table.relative_to(ROOT)}")

# plots
MU_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF5722"}
MU_MARKERS = {1: "o", 2: "s", 3: "^"}
TITLE_SUFFIX = (
    "BnA Gurobi (our impl.) vs Andersen et al. (2022)\nδ = 0.00001, time limit = 7200 s"
)

# runtime comparison (log-log scatter)
fig1, ax1 = plt.subplots(figsize=(7, 6))
fig1.suptitle(TITLE_SUFFIX, fontsize=12, fontweight="bold")

for mu_val, grp in df.groupby("mu"):
    ax1.scatter(
        grp["and_time"],
        grp["bna_time_s"],
        color=MU_COLORS[mu_val],
        marker=MU_MARKERS[mu_val],
        s=70,
        label=f"μ = {mu_val}",
        zorder=3,
    )

lim = [1, 20000]
ax1.plot(lim, lim, "k--", lw=1, alpha=0.5, label="equal time")
ax1.axhline(7200, color="gray", lw=1, ls=":", alpha=0.7, label="7200 s limit")
ax1.axvline(7200, color="gray", lw=1, ls=":", alpha=0.7)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(lim)
ax1.set_ylim(lim)
ax1.set_xlabel("Andersen time [s]", fontsize=11)
ax1.set_ylabel("Our time [s]", fontsize=11)
ax1.set_title("Runtime comparison", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)

fig1.tight_layout()
out_fig1 = OUT_DIR / "comparison_andersen_time.png"
fig1.savefig(out_fig1, dpi=150, bbox_inches="tight")
print(f"  Plot 1 saved → {out_fig1.relative_to(ROOT)}")

# UB relative difference [%]
fig2, ax2 = plt.subplots(figsize=(14, 5))
fig2.suptitle(TITLE_SUFFIX, fontsize=12, fontweight="bold")

x_pos = np.arange(len(df))
bar_colors = [MU_COLORS[m] for m in df["mu"]]

ax2.bar(x_pos, df["ub_diff_pct"], color=bar_colors, edgecolor="white", linewidth=0.4)

# Hatch bars where we hit a time or memory limit
for i, (_, row) in enumerate(df.iterrows()):
    if row["our_status_short"] in ("TL", "MEM"):
        ax2.bar(
            x_pos[i],
            row["ub_diff_pct"],
            color=bar_colors[i],
            edgecolor="black",
            linewidth=1.2,
            hatch="//",
        )

ax2.axhline(0, color="black", lw=1)

# x-axis: single-line label per bar, wider figure so they don't overlap
labels = [f"{r.weapons}×{r.targets} μ={r.mu}" for _, r in df.iterrows()]
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=8, rotation=55, ha="right")
ax2.set_ylabel("Δ UB [%]  = (ours − And.) / And. × 100", fontsize=10)
ax2.set_title("Relative UB difference (negative = ours better)", fontsize=11)

legend_elements = [
    Patch(facecolor=MU_COLORS[1], label="μ = 1"),
    Patch(facecolor=MU_COLORS[2], label="μ = 2"),
    Patch(facecolor=MU_COLORS[3], label="μ = 3"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="time/mem limit"),
]
ax2.legend(handles=legend_elements, fontsize=9, loc="upper left")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
ax2.grid(axis="y", alpha=0.3)

fig2.tight_layout()
out_fig2 = OUT_DIR / "comparison_andersen_ub.png"
fig2.savefig(out_fig2, dpi=150, bbox_inches="tight")
print(f"  Plot 2 saved → {out_fig2.relative_to(ROOT)}")

plt.show()

# Summary statistics
print()
print("  Summary:")
n = len(df)
n_opt_ours = (df["bna_status"] == "optimal").sum()
n_opt_and = (df["and_gap"] == "opt").sum()
n_we_better = (df["ub_diff_pct"] < -0.0001).sum()
n_we_worse = (df["ub_diff_pct"] > 0.0001).sum()
n_equal = n - n_we_better - n_we_worse

print(f"Instances: {n}")
print(f"Andersen opt: {n_opt_and}/{n}  |  Ours opt: {n_opt_ours}/{n}")
print(
    f"Better UB than Andersen: {n_we_better}  |  Equal: {n_equal}  |  Worse: {n_we_worse}"
)
print(f"Median Δ UB:       {df['ub_diff_pct'].median():.4f}%")
print(f"Median time ratio: {df['time_ratio'].median():.2f}×")
print(
    f"Min/Max time ratio: {df['time_ratio'].min():.2f}× / {df['time_ratio'].max():.2f}×"
)
