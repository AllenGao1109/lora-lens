"""
LoRA-Lens: Visualization
=========================
Publication-quality figures for spectral analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional

from src.spectral_analysis import ModelSpectralReport, SpectralProfile

# Style config
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "figure.facecolor": "white",
})


def plot_singular_value_distribution(
    reports: list[ModelSpectralReport],
    layer_idx: int = 0,
    figsize: tuple = (10, 5),
):
    """
    Compare singular value distributions across configs for a given layer.

    This is the LoRA equivalent of plotting the KL divergence per latent
    dimension in a VAE — shows which directions are "active."
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for i, report in enumerate(reports):
        if layer_idx >= len(report.layer_profiles):
            continue
        profile = report.layer_profiles[layer_idx]
        sv = profile.singular_values
        color = COLORS[i % len(COLORS)]

        # Left: Raw singular values
        ax1.plot(range(len(sv)), sv, '-o', markersize=3,
                 label=report.config_name, color=color)

        # Right: Normalized (probability distribution)
        sv_norm = sv / sv.sum() if sv.sum() > 0 else sv
        ax2.bar(range(len(sv_norm)), sv_norm, alpha=0.6,
                label=report.config_name, color=color)

    layer_name = reports[0].layer_profiles[layer_idx].layer_name.split(".")[-2:]
    layer_label = ".".join(layer_name)

    ax1.set_xlabel("Singular Value Index")
    ax1.set_ylabel("σ_i")
    ax1.set_title(f"Singular Values ({layer_label})")
    ax1.legend(fontsize=9)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Singular Value Index")
    ax2.set_ylabel("σ̃_i (normalized)")
    ax2.set_title(f"Spectral Distribution ({layer_label})")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_effective_rank_vs_nominal(
    reports: list[ModelSpectralReport],
    eval_scores: Optional[dict] = None,
    figsize: tuple = (12, 5),
):
    """
    Key Figure 1: Rank vs Effective Rank vs OOD Score.

    Hypothesized U-curve: too low rank → underfitting, too high → overfitting
    with "junk" dimensions. Analogous to β-VAE sweet spot.
    """
    fig, axes = plt.subplots(1, 2 if eval_scores else 1, figsize=figsize)
    if not eval_scores:
        axes = [axes]

    nominal_ranks = []
    eff_ranks = []
    utilizations = []
    config_names = []

    for report in reports:
        nominal_ranks.append(report.layer_profiles[0].nominal_rank)
        eff_ranks.append(report.mean_effective_rank)
        utilizations.append(report.mean_capacity_utilization)
        config_names.append(report.config_name)

    # Left: Effective rank vs nominal rank
    ax = axes[0]
    ax.plot(nominal_ranks, eff_ranks, 'o-', color=COLORS[0],
            markersize=8, linewidth=2, label="Effective Rank")
    ax.plot(nominal_ranks, nominal_ranks, '--', color="gray",
            alpha=0.5, label="Perfect utilization")
    ax.fill_between(nominal_ranks, eff_ranks, nominal_ranks,
                     alpha=0.15, color=COLORS[0])

    for x, y, name in zip(nominal_ranks, eff_ranks, config_names):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    ax.set_xlabel("Nominal Rank (r)")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Capacity Utilization: Effective vs Nominal Rank")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: If eval scores provided, show rank vs generalization
    if eval_scores and len(axes) > 1:
        ax2 = axes[1]
        scores = [eval_scores.get(name, {}) for name in config_names]
        id_scores = [s.get("id", 0) for s in scores]
        ood_scores = [s.get("ood", 0) for s in scores]

        ax2.plot(nominal_ranks, id_scores, 's-', color=COLORS[1],
                 markersize=8, linewidth=2, label="In-Domain")
        ax2.plot(nominal_ranks, ood_scores, 'D-', color=COLORS[2],
                 markersize=8, linewidth=2, label="Out-of-Domain")

        ax2.set_xlabel("Nominal Rank (r)")
        ax2.set_ylabel("Eval Score")
        ax2.set_title("Generalization vs Rank (U-curve?)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_layer_heatmap(
    report: ModelSpectralReport,
    figsize: tuple = (14, 6),
):
    """
    Key Figure 2: Layer-wise spectral heatmap.

    Shows how spectral properties vary from shallow to deep layers.
    Expected: shallow layers = higher cosine sim to pretrained (syntax),
    deep layers = more intruders (semantics).
    """
    layers = report.layer_profiles
    n_layers = len(layers)

    # Extract short layer names
    short_names = []
    for p in layers:
        parts = p.layer_name.split(".")
        # Find layer number and module type
        layer_num = ""
        module_type = parts[-1] if parts else ""
        for part in parts:
            if part.isdigit():
                layer_num = part
                break
        short_names.append(f"L{layer_num}.{module_type}")

    # Build metric matrix
    metrics = {
        "Effective Rank": [p.effective_rank for p in layers],
        "Cap. Utilization": [p.capacity_utilization for p in layers],
        "Spectral Entropy": [p.spectral_entropy for p in layers],
        "SVR Top-1": [p.svr_top1 for p in layers],
        "SVR Top-5": [p.svr_top5 for p in layers],
        "Intruder Count": [p.intruder_count for p in layers],
        "Frobenius Norm": [p.frobenius_norm for p in layers],
    }

    matrix = np.array(list(metrics.values()))
    # Normalize each row to [0, 1] for heatmap
    matrix_norm = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        rmin, rmax = row.min(), row.max()
        if rmax > rmin:
            matrix_norm[i] = (row - rmin) / (rmax - rmin)
        else:
            matrix_norm[i] = 0.5

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix_norm, cmap="YlOrRd", aspect="auto")

    # Annotate with actual values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            fmt = f"{val:.2f}" if val < 100 else f"{val:.0f}"
            color = "white" if matrix_norm[i, j] > 0.6 else "black"
            ax.text(j, i, fmt, ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(list(metrics.keys()), fontsize=9)
    ax.set_title(f"Layer-wise Spectral Profile: {report.config_name}")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized Value")
    plt.tight_layout()
    plt.show()


def plot_capacity_utilization_curve(
    reports: list[ModelSpectralReport],
    x_key: str = "nominal_rank",  # or "train_samples"
    x_values: Optional[list] = None,
    figsize: tuple = (8, 5),
):
    """
    Key Figure 3: Capacity utilization vs controlling variable.

    Directly analogous to plotting "active latent dimensions / total dimensions"
    vs β in β-VAE, or vs training data size.

    When utilization drops below 1.0, the adapter has "wasted" capacity —
    the LoRA equivalent of posterior collapse.
    """
    fig, ax = plt.subplots(figsize=figsize)

    utilizations = [r.mean_capacity_utilization for r in reports]
    eff_ranks = [r.mean_effective_rank for r in reports]
    names = [r.config_name for r in reports]

    if x_values is None:
        x_values = list(range(len(reports)))

    ax.plot(x_values, utilizations, 'o-', color=COLORS[0],
            markersize=8, linewidth=2)

    # Shade the "collapse zone"
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
               label="Full utilization")
    ax.axhline(y=0.5, color=COLORS[2], linestyle=":", alpha=0.5,
               label="50% utilization")
    ax.fill_between(x_values, 0, utilizations, alpha=0.1, color=COLORS[0])

    for x, y, name in zip(x_values, utilizations, names):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(5, 8), fontsize=9)

    ax.set_xlabel("Nominal Rank" if x_key == "nominal_rank" else "Training Samples")
    ax.set_ylabel("Capacity Utilization (erank / rank)")
    ax.set_title("Adapter Capacity Utilization\n(< 1.0 = LoRA equivalent of posterior collapse)")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_spectral_predictor(
    reports: list[ModelSpectralReport],
    eval_scores: dict[str, float],
    figsize: tuple = (8, 6),
):
    """
    Key Figure 5: Can spectral metrics predict generalization?

    Scatter plot: spectral metric vs eval score, with correlation.
    If R² is high, we can skip expensive eval and just look at the spectrum.
    """
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))

    metrics_to_plot = [
        ("Spectral Entropy", [r.mean_spectral_entropy for r in reports]),
        ("Effective Rank", [r.mean_effective_rank for r in reports]),
        ("Cap. Utilization", [r.mean_capacity_utilization for r in reports]),
    ]

    scores = [eval_scores.get(r.config_name, 0) for r in reports]

    for ax, (metric_name, metric_values) in zip(axes, metrics_to_plot):
        ax.scatter(metric_values, scores, s=80, color=COLORS[0],
                   edgecolors="black", linewidth=0.5, zorder=5)

        # Annotate each point
        for x, y, r in zip(metric_values, scores, reports):
            ax.annotate(r.config_name, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        # Correlation line
        if len(metric_values) > 2:
            from scipy import stats
            slope, intercept, r_val, p_val, _ = stats.linregress(metric_values, scores)
            x_line = np.linspace(min(metric_values), max(metric_values), 100)
            ax.plot(x_line, slope * x_line + intercept, '--', color=COLORS[1], alpha=0.7)
            ax.text(0.05, 0.95, f"R²={r_val**2:.3f}\np={p_val:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel(metric_name)
        ax.set_ylabel("Eval Score (OOD)")
        ax.set_title(f"{metric_name} → Generalization")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Can Spectral Metrics Predict Generalization?", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_intruder_analysis(
    report: ModelSpectralReport,
    figsize: tuple = (12, 5),
):
    """
    Key Figure 4: Intruder dimension analysis across layers.
    """
    layers = report.layer_profiles
    n_layers = len(layers)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Intruder count per layer
    short_names = []
    intruder_counts = []
    for p in layers:
        parts = p.layer_name.split(".")
        layer_num = ""
        module_type = parts[-1] if parts else ""
        for part in parts:
            if part.isdigit():
                layer_num = part
                break
        short_names.append(f"L{layer_num}.{module_type}")
        intruder_counts.append(p.intruder_count)

    ax1.barh(range(n_layers), intruder_counts, color=COLORS[3], alpha=0.7)
    ax1.set_yticks(range(n_layers))
    ax1.set_yticklabels(short_names, fontsize=8)
    ax1.set_xlabel("Intruder Dimension Count")
    ax1.set_title(f"Intruder Dimensions per Layer ({report.config_name})")
    ax1.invert_yaxis()

    # Right: Intruder score distribution (all layers combined)
    all_ids = np.concatenate([p.intruder_scores for p in layers if len(p.intruder_scores) > 0])
    if len(all_ids) > 0:
        ax2.hist(all_ids, bins=30, color=COLORS[0], alpha=0.7, edgecolor="black")
        ax2.axvline(x=0.9, color=COLORS[3], linestyle="--",
                    label="Intruder threshold (0.9)")
        ax2.set_xlabel("Intruder Dimension Score (IDS)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Intruder Scores")
        ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_full_report(
    reports: list[ModelSpectralReport],
    eval_scores: Optional[dict] = None,
    save_path: Optional[str] = None,
):
    """Generate the complete 4-panel report figure."""
    print("Generating full spectral report...")
    plot_effective_rank_vs_nominal(reports, eval_scores)
    if len(reports) > 0:
        plot_layer_heatmap(reports[len(reports) // 2])  # Middle config
    plot_capacity_utilization_curve(
        reports,
        x_values=[r.layer_profiles[0].nominal_rank for r in reports],
    )
    if eval_scores:
        plot_spectral_predictor(reports, eval_scores)
    print("✓ Report complete")
