import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _parse_andersen_instance(instance_label: str, mu_value: int) -> str:
	"""Convert label like '50×100' and mu=2 to Andersen filename."""
	wx_t = instance_label.replace("×", "x")
	return f"wta_{wx_t}x{int(mu_value)}.txt"

# Wczytaj wyniki
bna = pd.read_csv("results/benchmark_andersen.csv")
bna_v2 = pd.read_csv("results/benchmark_andersen_v2.csv")
andersen = pd.read_csv("results/comparison_andersen.csv")

# Zostaw tylko wspólne instancje (po "file")
common_files = set(bna["file"]).intersection(bna_v2["file"])
bna = bna[bna["file"].isin(common_files)].reset_index(drop=True)
bna_v2 = bna_v2[bna_v2["file"].isin(common_files)].reset_index(drop=True)

# Przygotuj tabelę porównawczą
summary = bna[["file", "weapons", "targets", "bna_time_s", "bna_obj", "bna_status"]].copy()
summary = summary.rename(columns={"bna_time_s": "BnA_time", "bna_obj": "BnA_obj", "bna_status": "BnA_status"})
summary["BnA_v2_time"] = bna_v2["bna_time_s"]
summary["BnA_v2_obj"] = bna_v2["bna_obj"]
summary["BnA_v2_status"] = bna_v2["bna_status"]

summary.to_csv("results/bna_vs_bnav2_summary.csv", index=False)

# Wykres: czas vs rozmiar instancji (BnA, BnA-v2, Andersen)
sizes = bna["weapons"] * bna["targets"]

andersen = andersen.copy()
andersen["mu"] = pd.to_numeric(andersen["μ"], errors="coerce")
andersen["file"] = andersen.apply(
	lambda r: _parse_andersen_instance(str(r["Instance"]), r["mu"]),
	axis=1,
)
andersen_runtime = andersen[["file", "And. time [s]"]].rename(columns={"And. time [s]": "Andersen_time"})
andersen_obj = andersen[["file", "And. UB"]].rename(columns={"And. UB": "Andersen_obj"})

runtime_cmp = summary.merge(andersen_runtime, on="file", how="inner")
runtime_cmp = runtime_cmp.sort_values(["weapons", "targets", "file"]).reset_index(drop=True)
x = runtime_cmp["weapons"] * runtime_cmp["targets"]

plt.figure(figsize=(10, 6))
plt.plot(x, runtime_cmp["BnA_time"], "o-", label="BnA")
plt.plot(x, runtime_cmp["BnA_v2_time"], "s-", label="BnA-v2")
plt.plot(x, runtime_cmp["Andersen_time"], "^-", label="Andersen")
plt.xlabel("Instance size (weapons × targets)")
plt.ylabel("Time [s]")
plt.yscale("log")
plt.title("Runtime comparison: BnA vs BnA-v2 vs Andersen")
plt.legend()
plt.tight_layout()
plt.savefig("results/bna_bnav2_andersen_time.png", dpi=200)

# Wykres: wartość celu / wynik (podział po mu + oznaczenie limitów u nas)
obj_cmp = summary.merge(andersen_obj, on="file", how="inner")
obj_cmp["mu"] = obj_cmp["file"].str.extract(r"x(\d+)\.txt$").astype(int)
obj_cmp["size"] = obj_cmp["weapons"] * obj_cmp["targets"]
obj_cmp["size_label"] = obj_cmp["weapons"].astype(str) + "x" + obj_cmp["targets"].astype(str)
obj_cmp = obj_cmp.sort_values(["mu", "weapons", "targets", "file"]).reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, mu_value in zip(axes, [1, 2, 3]):
	part = obj_cmp[obj_cmp["mu"] == mu_value].copy().reset_index(drop=True)
	if part.empty:
		ax.set_title(f"mu={mu_value} (no data)")
		continue

	xs = part["size_label"].tolist()
	x_idx = pd.Series(range(len(xs)), dtype=float)
	offset = 0.18
	x_bna = x_idx - offset
	x_bna_v2 = x_idx
	x_and = x_idx + offset

	ax.plot(
		x_bna,
		part["BnA_obj"],
		"o-",
		label="BnA",
		color="#1f77b4",
		alpha=0.75,
		linewidth=2.0,
		markersize=6,
		zorder=3,
	)
	ax.plot(
		x_bna_v2,
		part["BnA_v2_obj"],
		"s-",
		label="BnA-v2",
		color="#ff7f0e",
		alpha=0.75,
		linewidth=2.0,
		markersize=6,
		zorder=2,
	)
	ax.plot(
		x_and,
		part["Andersen_obj"],
		"^--",
		label="Andersen UB",
		color="#2ca02c",
		alpha=0.75,
		linewidth=1.8,
		markersize=6,
		zorder=1,
	)

	# Zaznacz punkty, gdzie solver nie zakończył się statusem optimal.
	bad_bna = part["BnA_status"] != "optimal"
	bad_bna_v2 = part["BnA_v2_status"] != "optimal"
	if bad_bna.any():
		ax.scatter(x_bna[bad_bna], part.loc[bad_bna, "BnA_obj"], marker="x", s=75, color="#1f77b4", zorder=4)
	if bad_bna_v2.any():
		ax.scatter(x_bna_v2[bad_bna_v2], part.loc[bad_bna_v2, "BnA_v2_obj"], marker="x", s=75, color="#ff7f0e", zorder=4)

	ax.set_title(f"mu = {mu_value}")
	ax.set_xticks(x_idx)
	ax.set_xticklabels(xs, rotation=45, ha="right")
	ax.set_xlabel("Instance size (W x T)")
	ax.grid(alpha=0.25)

axes[0].set_ylabel("Objective value (lower is better)")
for ax in axes:
	ax.set_yscale("log")

legend_items = [
	Line2D([0], [0], color="#1f77b4", marker="o", linestyle="-", label="BnA"),
	Line2D([0], [0], color="#ff7f0e", marker="s", linestyle="-", label="BnA-v2"),
	Line2D([0], [0], color="#2ca02c", marker="^", linestyle="-", label="Andersen UB"),
	Line2D([0], [0], color="black", marker="x", linestyle="None", label="non-optimal (time/mem/interrupted)"),
]
fig.legend(handles=legend_items, loc="upper center", ncol=2)
fig.suptitle("Objective comparison by mu: BnA vs BnA-v2 vs Andersen")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("results/bna_bnav2_andersen_objective_by_mu.png", dpi=220)

# Wykres 3D: weapons, targets, objective (osobno dla mu)
fig3d = plt.figure(figsize=(16, 5.2))
for idx, mu_value in enumerate([1, 2, 3], start=1):
	ax3d = fig3d.add_subplot(1, 3, idx, projection="3d")
	part = obj_cmp[obj_cmp["mu"] == mu_value].copy()
	if part.empty:
		ax3d.set_title(f"mu = {mu_value} (no data)")
		continue

	# Lekkie przesunięcie XY, żeby serie nie nakładały się dokładnie na siebie.
	w_bna = part["weapons"] - 1.8
	t_bna = part["targets"] - 3.6
	w_bna_v2 = part["weapons"]
	t_bna_v2 = part["targets"]
	w_and = part["weapons"] + 1.8
	t_and = part["targets"] + 3.6

	ax3d.scatter(w_bna, t_bna, part["BnA_obj"], c="#1f77b4", marker="o", s=36, alpha=0.8)
	ax3d.scatter(w_bna_v2, t_bna_v2, part["BnA_v2_obj"], c="#ff7f0e", marker="s", s=36, alpha=0.8)
	ax3d.scatter(w_and, t_and, part["Andersen_obj"], c="#2ca02c", marker="^", s=40, alpha=0.8)

	bad_bna = part["BnA_status"] != "optimal"
	bad_bna_v2 = part["BnA_v2_status"] != "optimal"
	if bad_bna.any():
		ax3d.scatter(w_bna[bad_bna], t_bna[bad_bna], part.loc[bad_bna, "BnA_obj"], c="black", marker="x", s=50)
	if bad_bna_v2.any():
		ax3d.scatter(
			w_bna_v2[bad_bna_v2],
			t_bna_v2[bad_bna_v2],
			part.loc[bad_bna_v2, "BnA_v2_obj"],
			c="black",
			marker="x",
			s=50,
		)

	ax3d.set_title(f"mu = {mu_value}")
	ax3d.set_xlabel("Weapons")
	ax3d.set_ylabel("Targets")
	ax3d.set_zlabel("Objective")
	ax3d.view_init(elev=24, azim=-58)

legend_items_3d = [
	Line2D([0], [0], color="#1f77b4", marker="o", linestyle="None", label="BnA"),
	Line2D([0], [0], color="#ff7f0e", marker="s", linestyle="None", label="BnA-v2"),
	Line2D([0], [0], color="#2ca02c", marker="^", linestyle="None", label="Andersen UB"),
	Line2D([0], [0], color="black", marker="x", linestyle="None", label="non-optimal (time/mem/interrupted)"),
]
fig3d.legend(handles=legend_items_3d, loc="upper center", ncol=4)
fig3d.suptitle("3D objective comparison by mu: Weapons vs Targets vs Objective")
fig3d.tight_layout(rect=[0, 0, 1, 0.9])
fig3d.savefig("results/bna_bnav2_andersen_objective_3d_by_mu.png", dpi=220)

# Wykres: statusy
plt.figure(figsize=(8, 2))
plt.scatter(sizes, bna["bna_status"] == "optimal", label="BnA optimal", marker="o")
plt.scatter(sizes, bna_v2["bna_status"] == "optimal", label="BnA-v2 optimal", marker="s")
plt.xlabel("Instance size")
plt.yticks([0, 1], ["not optimal", "optimal"])
plt.title("Optimality by instance size")
plt.legend()
plt.tight_layout()
plt.savefig("results/bna_vs_bnav2_status.png", dpi=200)
