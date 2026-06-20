from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

RESULTS_DIR = Path("results")

BNA_FILE = RESULTS_DIR / "final_benchmark_andersen_rerun.csv"
BNA_V2_FILE = RESULTS_DIR / "final_benchmark_andersen_v2_rerun.csv"
ANDERSEN_FILE = RESULTS_DIR / "comparison_andersen.csv"

OUTPUT_DIR = RESULTS_DIR / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def parse_filename(filename: str) -> tuple[int, int, int]:
    """
    Parse:
        wta_50x100x1.txt -> (50, 100, 1)
    """
    match = re.search(r"wta_(\d+)x(\d+)x(\d+)\.txt", str(filename))
    if match is None:
        raise ValueError(f"Cannot parse instance filename: {filename}")

    weapons, targets, mu = map(int, match.groups())
    return weapons, targets, mu


def make_andersen_filename(instance: str, mu: int) -> str:
    """
    Convert:
        '50×100', mu=1 -> 'wta_50x100x1.txt'
    """
    instance = str(instance).replace("×", "x").replace(" ", "")
    return f"wta_{instance}x{int(mu)}.txt"


def status_is_optimal(status: object) -> bool:
    return str(status).strip().lower() == "optimal"


def andersen_status_is_optimal(status: object) -> bool:
    return "opt" in str(status).strip().lower()


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a standard figure with a consistent resolution and layout."""
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def has_nonempty_error(value: object) -> bool:
    """True if a CSV error field contains a meaningful error message."""
    if pd.isna(value):
        return False

    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "null"}


def method_failed(row: pd.Series, status_column: str, error_column: str) -> bool:
    """
    Treat an entry as failed if it has an explicit error message or a clearly
    failed/error-like status. Timeout/non-optimal statuses are not automatically
    treated as errors because they may still have a valid incumbent objective.
    """
    if has_nonempty_error(row.get(error_column, np.nan)):
        return True

    status = str(row.get(status_column, "")).strip().lower()
    failure_keywords = ("error", "failed", "exception", "infeasible", "unbounded")
    return any(keyword in status for keyword in failure_keywords)


def latex_escape(text: object) -> str:
    """Escape basic LaTeX special characters, mainly underscores in filenames."""
    text = str(text)

    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def format_number(value: object) -> str:
    """Format objective values compactly and consistently in the LaTeX table."""
    if pd.isna(value):
        return "--"

    value = float(value)

    if np.isinf(value):
        return r"$\infty$"

    if abs(value) >= 1000:
        return f"{value:.3f}"

    if abs(value) >= 100:
        return f"{value:.4f}"

    return f"{value:.6f}"


def objective_cell(
    row: pd.Series,
    objective_column: str,
    status_column: str | None = None,
    error_column: str | None = None,
) -> str:
    """
    Return a formatted objective. If the solver failed, return `error`.
    If the result is simply missing, return `--`.
    """
    if status_column is not None and error_column is not None:
        if method_failed(row, status_column, error_column):
            return r"\texttt{error}"

    return format_number(row.get(objective_column, np.nan))


def reference_gap_cell(
    row: pd.Series,
    objective_column: str,
    lower_bound_column: str,
    status_column: str | None = None,
    error_column: str | None = None,
) -> str:
    r"""
    Calculate an optimality/reference gap relative to Andersen's lower bound:

        100 * (UB - LB_Andersen) / |UB|

    This is a reference gap, not necessarily Gurobi's internally reported MIP gap.
    Negative values caused by rounding or a better incumbent than the reported
    Andersen bound are displayed as 0.000.
    """
    if status_column is not None and error_column is not None:
        if method_failed(row, status_column, error_column):
            return r"\texttt{error}"

    upper_bound = row.get(objective_column, np.nan)
    lower_bound = row.get(lower_bound_column, np.nan)

    if pd.isna(upper_bound) or pd.isna(lower_bound):
        return "--"

    upper_bound = float(upper_bound)
    lower_bound = float(lower_bound)

    if np.isinf(upper_bound) or abs(upper_bound) < 1e-12:
        return "--"

    gap = 100.0 * (upper_bound - lower_bound) / abs(upper_bound)
    gap = max(0.0, gap)

    return f"{gap:.3f}"


# =============================================================================
# Load and merge data
# =============================================================================

for file_path in [BNA_FILE, BNA_V2_FILE, ANDERSEN_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")

bna = pd.read_csv(BNA_FILE)
bna_v2 = pd.read_csv(BNA_V2_FILE)
andersen = pd.read_csv(ANDERSEN_FILE)

for df in [bna, bna_v2]:
    if "weapons" not in df.columns or "targets" not in df.columns:
        parsed = df["file"].apply(parse_filename)
        df["weapons"] = parsed.apply(lambda values: values[0])
        df["targets"] = parsed.apply(lambda values: values[1])

    if "mu" not in df.columns:
        df["mu"] = df["file"].apply(lambda name: parse_filename(name)[2])

    if "error" not in df.columns:
        df["error"] = np.nan

bna = bna.rename(
    columns={
        "bna_time_s": "BnA_time_s",
        "bna_obj": "BnA_obj",
        "bna_status": "BnA_status",
        "error": "BnA_error",
    }
)

bna_v2 = bna_v2.rename(
    columns={
        "bna_time_s": "BnA_v2_time_s",
        "bna_obj": "BnA_v2_obj",
        "bna_status": "BnA_v2_status",
        "error": "BnA_v2_error",
    }
)

bna_columns = [
    "file",
    "weapons",
    "targets",
    "mu",
    "BnA_time_s",
    "BnA_obj",
    "BnA_status",
    "BnA_error",
]

bna_v2_columns = [
    "file",
    "weapons",
    "targets",
    "mu",
    "BnA_v2_time_s",
    "BnA_v2_obj",
    "BnA_v2_status",
    "BnA_v2_error",
]

bna = bna[[column for column in bna_columns if column in bna.columns]]
bna_v2 = bna_v2[[column for column in bna_v2_columns if column in bna_v2.columns]]

comparison = bna.merge(
    bna_v2,
    on=["file", "weapons", "targets", "mu"],
    how="outer",
    validate="one_to_one",
)

andersen = andersen.copy()
andersen["mu"] = pd.to_numeric(andersen["μ"], errors="coerce").astype("Int64")
andersen["file"] = andersen.apply(
    lambda row: make_andersen_filename(row["Instance"], row["mu"]),
    axis=1,
)

andersen = andersen.rename(
    columns={
        "And. time [s]": "Andersen_time_s",
        "And. LB": "Andersen_LB",
        "And. UB": "Andersen_UB",
        "And. status": "Andersen_status",
    }
)

andersen_columns = [
    "file",
    "Andersen_time_s",
    "Andersen_LB",
    "Andersen_UB",
    "Andersen_status",
]

andersen = andersen[
    [column for column in andersen_columns if column in andersen.columns]
]

comparison = comparison.merge(
    andersen,
    on="file",
    how="left",
    validate="one_to_one",
)

expected_columns = [
    "BnA_time_s",
    "BnA_obj",
    "BnA_status",
    "BnA_error",
    "BnA_v2_time_s",
    "BnA_v2_obj",
    "BnA_v2_status",
    "BnA_v2_error",
    "Andersen_time_s",
    "Andersen_LB",
    "Andersen_UB",
    "Andersen_status",
]

for column in expected_columns:
    if column not in comparison.columns:
        comparison[column] = np.nan

comparison["size"] = comparison["weapons"] * comparison["targets"]
comparison["instance_label"] = (
    comparison["weapons"].astype(int).astype(str)
    + "×"
    + comparison["targets"].astype(int).astype(str)
)

comparison = comparison.sort_values(
    ["mu", "weapons", "targets"]
).reset_index(drop=True)

comparison.to_csv(
    OUTPUT_DIR / "comparison_merged.csv",
    index=False,
)

print(f"Loaded {len(comparison)} instances.")


# =============================================================================
# Figure 1: Runtime by instance size, separated by mu
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))

for mu in sorted(comparison["mu"].dropna().unique()):
    part = comparison[comparison["mu"] == mu].sort_values("size")

    ax.plot(
        part["size"],
        part["BnA_time_s"],
        marker="o",
        label=f"BnA, μ={mu}",
    )

    ax.plot(
        part["size"],
        part["BnA_v2_time_s"],
        marker="s",
        linestyle="--",
        label=f"BnA-v2, μ={mu}",
    )

ax.set_yscale("log")
ax.set_xlabel("Instance size: weapons × targets")
ax.set_ylabel("Runtime [s], logarithmic scale")
ax.set_title("Runtime comparison: BnA vs BnA-v2")
ax.grid(alpha=0.3)
ax.legend(ncol=2)

save_figure(fig, "01_runtime_bna_vs_bnav2_by_mu.png")


# =============================================================================
# Figure 2: Direct runtime scatter
# =============================================================================

paired_runtime = comparison.dropna(
    subset=["BnA_time_s", "BnA_v2_time_s"]
).copy()

fig, ax = plt.subplots(figsize=(7.5, 6.5))

markers = {1: "o", 2: "s", 3: "^"}

for mu in sorted(paired_runtime["mu"].dropna().unique()):
    part = paired_runtime[paired_runtime["mu"] == mu]

    ax.scatter(
        part["BnA_time_s"],
        part["BnA_v2_time_s"],
        marker=markers.get(int(mu), "o"),
        label=f"μ={mu}",
    )

minimum = min(
    paired_runtime["BnA_time_s"].min(),
    paired_runtime["BnA_v2_time_s"].min(),
)

maximum = max(
    paired_runtime["BnA_time_s"].max(),
    paired_runtime["BnA_v2_time_s"].max(),
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="same runtime",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("BnA runtime [s]")
ax.set_ylabel("BnA-v2 runtime [s]")
ax.set_title("Direct runtime comparison")
ax.grid(alpha=0.3)
ax.legend()

save_figure(fig, "02_runtime_scatter_bnav2_vs_bna.png")


# =============================================================================
# Figure 3: Speed ratio
# =============================================================================

paired_runtime["speed_ratio"] = (
    paired_runtime["BnA_v2_time_s"] / paired_runtime["BnA_time_s"]
)

fig, ax = plt.subplots(figsize=(11, 5.5))

for mu in sorted(paired_runtime["mu"].dropna().unique()):
    part = paired_runtime[paired_runtime["mu"] == mu].sort_values("size")

    ax.scatter(
        part["size"],
        part["speed_ratio"],
        marker=markers.get(int(mu), "o"),
        label=f"μ={mu}",
    )

ax.axhline(
    1.0,
    linestyle="--",
    label="same runtime",
)

ax.set_yscale("log")
ax.set_xlabel("Instance size: weapons × targets")
ax.set_ylabel("Runtime ratio: BnA-v2 / BnA")
ax.set_title("Relative speed of BnA-v2")
ax.grid(alpha=0.3)
ax.legend()

save_figure(fig, "03_speed_ratio_bnav2_over_bna.png")


# =============================================================================
# Figure 4: Objective difference from best available upper bound
# =============================================================================

quality_rows = []

for _, row in comparison.iterrows():
    candidates = {
        "BnA": row.get("BnA_obj", np.nan),
        "BnA-v2": row.get("BnA_v2_obj", np.nan),
        "Andersen UB": row.get("Andersen_UB", np.nan),
    }

    available = {
        method: value
        for method, value in candidates.items()
        if pd.notna(value)
    }

    if not available:
        continue

    best_objective = min(available.values())

    for method, objective in available.items():
        relative_gap = 100.0 * (objective - best_objective) / best_objective

        quality_rows.append(
            {
                "file": row["file"],
                "mu": row["mu"],
                "method": method,
                "objective": objective,
                "best_available_objective": best_objective,
                "relative_gap_pct": relative_gap,
            }
        )

quality = pd.DataFrame(quality_rows)

quality.to_csv(
    OUTPUT_DIR / "objective_quality_comparison.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(10, 6))

method_positions = {
    "BnA": 0,
    "BnA-v2": 1,
    "Andersen UB": 2,
}

offsets = {
    1: -0.12,
    2: 0.0,
    3: 0.12,
}

for mu in sorted(quality["mu"].dropna().unique()):
    part = quality[quality["mu"] == mu].copy()

    x = part["method"].map(method_positions).to_numpy(dtype=float)
    x = x + offsets.get(int(mu), 0.0)

    ax.scatter(
        x,
        part["relative_gap_pct"],
        marker=markers.get(int(mu), "o"),
        label=f"μ={mu}",
    )

ax.axhline(0.0, linestyle="--")

ax.set_xticks(list(method_positions.values()))
ax.set_xticklabels(list(method_positions.keys()))
ax.set_yscale("symlog", linthresh=0.01)
ax.set_ylabel("Objective excess over best available upper bound [%]")
ax.set_title("Quality of obtained solutions")
ax.grid(alpha=0.3, axis="y")
ax.legend(title="Weapon availability")

save_figure(fig, "04_objective_excess_to_best_known.png")


# =============================================================================
# Figure 5: Proven optimality count
# =============================================================================

comparison["BnA_optimal"] = comparison["BnA_status"].apply(status_is_optimal)
comparison["BnA_v2_optimal"] = comparison["BnA_v2_status"].apply(status_is_optimal)

if "Andersen_status" in comparison.columns:
    comparison["Andersen_optimal"] = comparison["Andersen_status"].apply(
        andersen_status_is_optimal
    )
else:
    comparison["Andersen_optimal"] = False

mu_values = sorted(comparison["mu"].dropna().unique())
x = np.arange(len(mu_values))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5.5))

methods = [
    ("BnA", "BnA_optimal"),
    ("BnA-v2", "BnA_v2_optimal"),
    ("Andersen", "Andersen_optimal"),
]

for index, (method_name, column_name) in enumerate(methods):
    values = []

    for mu in mu_values:
        count = comparison.loc[
            comparison["mu"] == mu,
            column_name,
        ].sum()

        values.append(count)

    bars = ax.bar(
        x + (index - 1) * width,
        values,
        width,
        label=method_name,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(int(value)),
            ha="center",
            va="bottom",
        )

ax.set_xticks(x)
ax.set_xticklabels([f"μ={int(mu)}" for mu in mu_values])
ax.set_xlabel("Weapon availability")
ax.set_ylabel("Instances solved to proven optimality")
ax.set_title("Number of proven-optimal solutions")
ax.grid(alpha=0.3, axis="y")
ax.legend()

save_figure(fig, "05_proven_optimality_counts.png")


# =============================================================================
# Figure 6: BnA v1 / BnA v2 objective compared with Andersen UB
# =============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6.2),
    sharey=True,
)

comparisons_to_andersen = [
    ("BnA v1", "BnA_obj"),
    ("BnA v2", "BnA_v2_obj"),
]

for ax, (method_name, objective_column) in zip(axes, comparisons_to_andersen):
    part = comparison.dropna(
        subset=[objective_column, "Andersen_UB"]
    ).copy()

    part["relative_difference_pct"] = (
        100.0
        * (part[objective_column] - part["Andersen_UB"])
        / part["Andersen_UB"]
    )

    for mu in sorted(part["mu"].dropna().unique()):
        mu_part = part[part["mu"] == mu].sort_values("size")

        ax.scatter(
            mu_part["size"],
            mu_part["relative_difference_pct"],
            marker=markers.get(int(mu), "o"),
            s=65,
            label=f"μ={int(mu)}",
        )

        ax.plot(
            mu_part["size"],
            mu_part["relative_difference_pct"],
            alpha=0.45,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.3,
        color="black",
    )

    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("Instance size: weapons × targets")
    ax.set_title(f"{method_name} vs Andersen UB")
    ax.grid(alpha=0.3)

axes[0].set_ylabel(
    "Relative objective difference vs Andersen UB [%]\n"
    "(negative = lower objective, positive = higher objective)"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.suptitle(
    "Objective comparison against Andersen reference upper bounds",
    y=0.98,
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=3,
    frameon=True,
)

fig.tight_layout(rect=[0, 0, 1, 0.80])

fig.savefig(
    OUTPUT_DIR / "06_bna_and_bnav2_vs_andersen_objective_difference.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "Saved:",
    OUTPUT_DIR / "06_bna_and_bnav2_vs_andersen_objective_difference.png",
)


# =============================================================================
# Section 7: LaTeX table with objectives and reference optimality gaps
# =============================================================================
#
# Gap definition:
#
#     100 * (UB - Andersen_LB) / |UB|
#
# For BnA v1 and BnA v2, it is a reference gap based on Andersen's lower bound.
# It is not necessarily the original MIP gap reported internally by Gurobi.
# If a solver produced an explicit error, both its objective and gap are printed
# as \texttt{error}.
# =============================================================================

latex_table_lines = [
    "% Requires in the LaTeX preamble:",
    "% \\usepackage{booktabs}",
    "% \\usepackage{longtable}",
    "%",
    "% If the table is too wide for the page, place \\small before \\input{...}.",
    "",
    "\\begin{longtable}{@{}lrrrrrrr@{}}",
    "\\caption{Comparison of objectives and reference optimality gaps. "
    "The gap is calculated as "
    "$100\\cdot(UB-LB_{\\mathrm{Andersen}})/|UB|$. "
    "Entries marked \\texttt{error} correspond to solver failures.}"
    "\\label{tab:wta-objective-gap-comparison} \\\\",
    "\\toprule",
    "File & $\\mu$ & BnA v1 obj. & BnA v2 obj. & Andersen UB & "
    "BnA v1 gap [\\%] & BnA v2 gap [\\%] & Andersen gap [\\%] \\\\",
    "\\midrule",
    "\\endfirsthead",
    "",
    "\\multicolumn{8}{c}{\\tablename\\ \\thetable\\ -- continued from previous page} \\\\",
    "\\toprule",
    "File & $\\mu$ & BnA v1 obj. & BnA v2 obj. & Andersen UB & "
    "BnA v1 gap [\\%] & BnA v2 gap [\\%] & Andersen gap [\\%] \\\\",
    "\\midrule",
    "\\endhead",
    "",
    "\\midrule",
    "\\multicolumn{8}{r}{Continued on next page} \\\\",
    "\\endfoot",
    "",
    "\\bottomrule",
    "\\endlastfoot",
]

table_data = comparison.sort_values(
    ["mu", "weapons", "targets", "file"]
).reset_index(drop=True)

latex_table_rows = []

for _, row in table_data.iterrows():
    file_name = latex_escape(row["file"])
    mu_value = int(row["mu"])

    bna_objective = objective_cell(
        row,
        objective_column="BnA_obj",
        status_column="BnA_status",
        error_column="BnA_error",
    )

    bna_v2_objective = objective_cell(
        row,
        objective_column="BnA_v2_obj",
        status_column="BnA_v2_status",
        error_column="BnA_v2_error",
    )

    andersen_objective = format_number(row.get("Andersen_UB", np.nan))

    bna_gap = reference_gap_cell(
        row,
        objective_column="BnA_obj",
        lower_bound_column="Andersen_LB",
        status_column="BnA_status",
        error_column="BnA_error",
    )

    bna_v2_gap = reference_gap_cell(
        row,
        objective_column="BnA_v2_obj",
        lower_bound_column="Andersen_LB",
        status_column="BnA_v2_status",
        error_column="BnA_v2_error",
    )

    andersen_gap = reference_gap_cell(
        row,
        objective_column="Andersen_UB",
        lower_bound_column="Andersen_LB",
    )

    latex_table_rows.append(
        {
            "file": row["file"],
            "mu": mu_value,
            "bna_v1_objective": bna_objective,
            "bna_v2_objective": bna_v2_objective,
            "andersen_ub": andersen_objective,
            "bna_v1_reference_gap_pct": bna_gap,
            "bna_v2_reference_gap_pct": bna_v2_gap,
            "andersen_reference_gap_pct": andersen_gap,
        }
    )

    latex_table_lines.append(
        f"\\texttt{{{file_name}}} & "
        f"{mu_value} & "
        f"{bna_objective} & "
        f"{bna_v2_objective} & "
        f"{andersen_objective} & "
        f"{bna_gap} & "
        f"{bna_v2_gap} & "
        f"{andersen_gap} \\\\"
    )

latex_table_lines.append("\\end{longtable}")

latex_table_path = OUTPUT_DIR / "07_objective_gap_comparison_table.tex"

latex_table_path.write_text(
    "\n".join(latex_table_lines),
    encoding="utf-8",
)

pd.DataFrame(latex_table_rows).to_csv(
    OUTPUT_DIR / "07_objective_gap_comparison_table.csv",
    index=False,
)

print(f"Saved LaTeX table: {latex_table_path}")
print(f"Saved table data: {OUTPUT_DIR / '07_objective_gap_comparison_table.csv'}")


# =============================================================================
# Text summary
# =============================================================================

summary_lines = []

summary_lines.append("WTA benchmark summary")
summary_lines.append("=" * 60)
summary_lines.append(f"Number of compared instances: {len(comparison)}")
summary_lines.append("")

for method, column in [
    ("BnA", "BnA_optimal"),
    ("BnA-v2", "BnA_v2_optimal"),
    ("Andersen", "Andersen_optimal"),
]:
    count = int(comparison[column].sum())
    summary_lines.append(
        f"{method}: proven optimal on {count}/{len(comparison)} instances"
    )

summary_lines.append("")
summary_lines.append(
    "Runtime note: Andersen runtimes come from another experimental environment."
)
summary_lines.append(
    "Use Andersen results as an external reference, not as a direct CPU-second comparison."
)
summary_lines.append(
    "The controlled runtime comparison is BnA versus BnA-v2."
)
summary_lines.append(
    "Table 07 gaps are reference gaps against Andersen LB, not Gurobi MIP gaps."
)

summary_path = OUTPUT_DIR / "summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print()
print("\n".join(summary_lines))
print()
print(f"All figures and tables saved in: {OUTPUT_DIR.resolve()}")
