import matplotlib.pyplot as plt
import numpy as np

architectures = ["VGG16", "VGG11", "ResNet-18", "ResNet-34"]

# internal keys (simple, stable)
settings = [
    "czr_high",
    "czr_low",
    "ds_set2_laplace",
    "ds_set2_rand",
    "gauss_set2_laplace",
    "laplace_set2_laplace",
]

# what you want to SEE in the legend
label_map = {
    "czr_high":           "CZR-high + Laplace for set-3",
    "czr_low":            "CZR-low + Laplace for set-3",
    "ds_set2_laplace":    "DeepSteal-set-2 + Laplace for set-3",
    "ds_set2_rand":       "DeepSteal-set-2 + rand",
    "gauss_set2_laplace": "Gauss-set-2 + Laplace for set-3",
    "laplace_set2_laplace":"Laplace-set-2 + Laplace for set-3",
}

# ---- accuracy data (match internal keys) ----
acc_data = {
    "czr_high":           [91.10, 89.77, 93.02, 92.52],
    "czr_low":            [88.48, 87.15, 92.69, 92.80],
    "ds_set2_laplace":    [90.70, 89.39, 93.57, 93.48],
    "ds_set2_rand":       [87.88, 88.82, 93.56, 93.55],
    "gauss_set2_laplace": [90.94, 87.28, 93.23, 93.58],
    "laplace_set2_laplace":[90.96, 89.92, 93.58, 93.59],
}

# ---- fidelity data ----
fid_data = {
    "czr_high":           [91.56, 93.27, 95.40, 94.97],
    "czr_low":            [87.90, 88.75, 94.71, 94.26],
    "ds_set2_laplace":    [91.42, 92.51, 96.16, 95.81],
    "ds_set2_rand":       [87.65, 91.67, 95.87, 95.96],
    "gauss_set2_laplace": [91.03, 89.28, 95.91, 95.35],
    "laplace_set2_laplace":[90.94, 92.42, 95.61, 95.34],
}

# ---- explicit color palette ----
colors = {
    "czr_high":           "#1f77b4",
    "czr_low":            "#ff7f0e",
    "ds_set2_laplace":    "#2ca02c",
    "ds_set2_rand":       "#d62728",
    "gauss_set2_laplace": "#9467bd",
    "laplace_set2_laplace":"#8c564b",
}

plt.rcParams.update({"font.size": 11})

x = np.arange(len(architectures))
bar_width = 0.15
offsets = (np.arange(len(settings)) - (len(settings) - 1) / 2) * bar_width

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

def plot_panel(ax, data_dict, ylabel, title):
    for i, key in enumerate(settings):
        heights = data_dict[key]
        bars = ax.bar(
            x + offsets[i],
            heights,
            width=bar_width,
            label=label_map[key],     
            color=colors[key],
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=6, rotation=60)

    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylabel(ylabel)
    ax.set_ylim(70, 100)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

# Plot both panels
plot_panel(axes[0], acc_data, "Accuracy (%)", "Post-training Accuracy")
plot_panel(axes[1], fid_data, "Fidelity (%)", "Post-training Fidelity")

# ------- legend using handles+labels from one axis -------
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=True,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.00),
)

plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave room for legend
plt.savefig("results_plots.png", dpi=300)
# or plt.show()