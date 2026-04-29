# %% [markdown]
# # LoRA-Lens: Spectral Anatomy of Low-Rank Adapters
# *A Representation Learning Perspective*
#
# This notebook runs all 4 core experiments:
# 1. **Rank Ablation** — How does rank affect spectral structure and generalization?
# 2. **Layer-wise Analysis** — Which layers matter most?
# 3. **Data Scaling** — How does data size affect capacity utilization?
# 4. **Target Module Selection** — Q+V vs all linear

# %% Cell 1: Setup (Colab)
# !pip install -q torch transformers peft datasets accelerate
# !pip install -q scipy seaborn matplotlib pandas numpy
# !git clone https://github.com/AllenGao1109/LoRA-Lens.git
# %cd LoRA-Lens

# %% Cell 2: Setup (Local — uncomment if running on M3 Max)
# %cd /path/to/LoRA-Lens

# %% Cell 3: Imports & Config
import os
import sys
import json
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.insert(0, ".")

from src.train_lora import (
    LoRATrainConfig, train_lora, load_trained_adapter,
    run_rank_ablation, run_data_scaling, run_target_module_ablation,
)
from src.spectral_analysis import (
    analyze_adapter, extract_lora_delta_weights,
    compute_effective_rank, compute_spectral_entropy, compute_svr,
    compute_subspace_overlap, ModelSpectralReport,
)
from src.visualization import (
    plot_singular_value_distribution,
    plot_effective_rank_vs_nominal,
    plot_layer_heatmap,
    plot_capacity_utilization_curve,
    plot_spectral_predictor,
    plot_intruder_analysis,
    plot_full_report,
)

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./results/adapters"
RESULTS_DIR = "./results/spectral"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## Experiment 1: Rank Ablation
# Train LoRA adapters with rank ∈ {4, 8, 16, 32, 64} and analyze spectral structure.

# %% Cell 4: Train adapters with different ranks
RANKS = [4, 8, 16, 32, 64]

rank_adapter_dirs = run_rank_ablation(
    ranks=RANKS,
    model_name=MODEL_NAME,
    max_train_samples=1000,
    output_dir=OUTPUT_DIR,
)
print(f"\nTrained {len(rank_adapter_dirs)} adapters: {rank_adapter_dirs}")

# %% Cell 5: Spectral analysis on each rank config
rank_reports = []

for r, adapter_dir in zip(RANKS, rank_adapter_dirs):
    print(f"\nAnalyzing R{r}...")
    model, tokenizer = load_trained_adapter(adapter_dir, MODEL_NAME)
    report = analyze_adapter(
        model,
        config_name=f"R{r}",
        model_name=MODEL_NAME,
        nominal_rank=r,
    )
    rank_reports.append(report)

    # Save report
    report_path = os.path.join(RESULTS_DIR, f"spectral_R{r}.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Free memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n✓ Spectral analysis complete for {len(rank_reports)} configs")

# %% Cell 6: Visualize rank ablation results
# Figure 1: Effective rank vs nominal rank
plot_effective_rank_vs_nominal(rank_reports)

# %% Cell 7: Singular value distributions across ranks
# Compare how energy is distributed across singular directions
# This is like comparing KL per latent dimension in β-VAE with different β
plot_singular_value_distribution(rank_reports, layer_idx=0)

# %% Cell 8: Capacity utilization curve
# The LoRA equivalent of "active latent dimensions / total dimensions" in VAE
plot_capacity_utilization_curve(
    rank_reports,
    x_values=RANKS,
)

# %% Cell 9: Summary table for rank ablation
rows = []
for report in rank_reports:
    rows.append({
        "Config": report.config_name,
        "Nominal Rank": report.layer_profiles[0].nominal_rank,
        "Effective Rank": round(report.mean_effective_rank, 2),
        "Capacity Util.": round(report.mean_capacity_utilization, 3),
        "Spectral Entropy": round(report.mean_spectral_entropy, 3),
        "Intruder Dims": report.total_intruder_count,
    })

df_rank = pd.DataFrame(rows)
print(df_rank.to_string(index=False))

# %% [markdown]
# ## Experiment 2: Layer-wise Spectral Analysis
# Using R16 as baseline, analyze spectral properties per layer.

# %% Cell 10: Layer-wise heatmap for R16
# Find R16 report
r16_report = [r for r in rank_reports if r.config_name == "R16"][0]
plot_layer_heatmap(r16_report)

# %% Cell 11: Layer-wise heatmaps for all ranks (comparison)
for report in rank_reports:
    print(f"\n--- {report.config_name} ---")
    plot_layer_heatmap(report)

# %% Cell 12: Intruder dimension analysis
for report in rank_reports:
    plot_intruder_analysis(report)

# %% Cell 13: Layer depth vs spectral metrics
# Extract layer depth (integer) and plot trends
def extract_layer_depth(layer_name: str) -> int:
    """Extract the transformer layer number from a parameter name."""
    parts = layer_name.split(".")
    for p in parts:
        if p.isdigit():
            return int(p)
    return -1

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, report in enumerate(rank_reports):
    depths = [extract_layer_depth(p.layer_name) for p in report.layer_profiles]
    eff_ranks = [p.effective_rank for p in report.layer_profiles]
    entropies = [p.spectral_entropy for p in report.layer_profiles]
    intruders = [p.intruder_count for p in report.layer_profiles]
    color = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'][i % 5]

    axes[0].scatter(depths, eff_ranks, alpha=0.6, color=color, label=report.config_name, s=20)
    axes[1].scatter(depths, entropies, alpha=0.6, color=color, s=20)
    axes[2].scatter(depths, intruders, alpha=0.6, color=color, s=20)

axes[0].set_xlabel("Layer Depth"); axes[0].set_ylabel("Effective Rank")
axes[0].set_title("Effective Rank vs Depth"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("Layer Depth"); axes[1].set_ylabel("Spectral Entropy")
axes[1].set_title("Spectral Entropy vs Depth")
axes[2].set_xlabel("Layer Depth"); axes[2].set_ylabel("Intruder Count")
axes[2].set_title("Intruder Dimensions vs Depth")

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.suptitle("Layer Depth Trends Across Rank Configs", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Experiment 3: Data Scaling
# Fixed rank=16, vary training samples: how does data size affect capacity utilization?

# %% Cell 14: Train adapters with different data sizes
SAMPLE_SIZES = [100, 500, 1000, 2000, 5000]

data_adapter_dirs = run_data_scaling(
    sample_sizes=SAMPLE_SIZES,
    model_name=MODEL_NAME,
    lora_r=16,
    output_dir=OUTPUT_DIR,
)

# %% Cell 15: Spectral analysis for data scaling
data_reports = []

for n, adapter_dir in zip(SAMPLE_SIZES, data_adapter_dirs):
    print(f"\nAnalyzing D{n}...")
    model, tokenizer = load_trained_adapter(adapter_dir, MODEL_NAME)
    report = analyze_adapter(
        model,
        config_name=f"D{n}",
        model_name=MODEL_NAME,
        nominal_rank=16,
    )
    data_reports.append(report)

    report_path = os.path.join(RESULTS_DIR, f"spectral_D{n}.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% Cell 16: Capacity utilization vs data size
# This is the "posterior collapse" analog:
# With less data, fewer singular directions get activated
plot_capacity_utilization_curve(
    data_reports,
    x_key="train_samples",
    x_values=SAMPLE_SIZES,
)

# %% Cell 17: Data scaling summary
rows = []
for report in data_reports:
    rows.append({
        "Config": report.config_name,
        "Train Samples": int(report.config_name[1:]),
        "Effective Rank": round(report.mean_effective_rank, 2),
        "Capacity Util.": round(report.mean_capacity_utilization, 3),
        "Spectral Entropy": round(report.mean_spectral_entropy, 3),
    })
df_data = pd.DataFrame(rows)
print(df_data.to_string(index=False))

# %% [markdown]
# ## Experiment 4: Target Module Selection
# Compare adapting Q+V only vs all attention vs all linear layers.

# %% Cell 18: Train adapters with different target modules
tm_adapter_dirs = run_target_module_ablation(
    model_name=MODEL_NAME,
    lora_r=16,
    max_train_samples=1000,
    output_dir=OUTPUT_DIR,
)

# %% Cell 19: Spectral analysis for target modules
tm_reports = []
tm_names = ["TM_QV", "TM_QKVO", "TM_ALL"]

for name, adapter_dir in zip(tm_names, tm_adapter_dirs):
    print(f"\nAnalyzing {name}...")
    model, tokenizer = load_trained_adapter(adapter_dir, MODEL_NAME)
    report = analyze_adapter(
        model, config_name=name, model_name=MODEL_NAME, nominal_rank=16,
    )
    tm_reports.append(report)
    del model, tokenizer
    gc.collect()

# %% Cell 20: Compare target module configs
rows = []
for report in tm_reports:
    rows.append({
        "Config": report.config_name,
        "Num Layers": len(report.layer_profiles),
        "Mean Eff. Rank": round(report.mean_effective_rank, 2),
        "Mean Cap. Util.": round(report.mean_capacity_utilization, 3),
        "Total Intruders": report.total_intruder_count,
    })
df_tm = pd.DataFrame(rows)
print(df_tm.to_string(index=False))

for report in tm_reports:
    plot_layer_heatmap(report)

# %% [markdown]
# ## Cross-Experiment Analysis: Spectral Signatures

# %% Cell 21: Aggregate all results
all_reports = rank_reports + data_reports + tm_reports

summary_rows = []
for report in all_reports:
    summary_rows.append({
        "Config": report.config_name,
        "Eff. Rank": round(report.mean_effective_rank, 2),
        "Cap. Util.": round(report.mean_capacity_utilization, 3),
        "Entropy": round(report.mean_spectral_entropy, 3),
        "Intruders": report.total_intruder_count,
    })

df_all = pd.DataFrame(summary_rows)
print("\n=== Full Results Summary ===")
print(df_all.to_string(index=False))

# %% Cell 22: Correlation matrix of spectral metrics
# Are different spectral metrics telling us the same thing?
metrics_df = pd.DataFrame([
    {
        "effective_rank": r.mean_effective_rank,
        "capacity_utilization": r.mean_capacity_utilization,
        "spectral_entropy": r.mean_spectral_entropy,
        "intruder_count": r.total_intruder_count,
    }
    for r in all_reports
])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(metrics_df.corr(), annot=True, cmap="coolwarm", center=0,
            fmt=".2f", square=True, ax=ax)
ax.set_title("Correlation Between Spectral Metrics")
plt.tight_layout()
plt.show()

# %% Cell 23: Save all results
results_summary = {
    "rank_ablation": [r.to_dict() for r in rank_reports],
    "data_scaling": [r.to_dict() for r in data_reports],
    "target_modules": [r.to_dict() for r in tm_reports],
}

with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
    json.dump(results_summary, f, indent=2)
print("✓ All results saved to results/spectral/all_results.json")

# %% [markdown]
# ## Key Findings Template
#
# Fill in after running:
#
# 1. **Rank sweet spot:** Effective rank peaks at R=___ with capacity utilization of ___.
#    Beyond R=___, additional dimensions are mostly "dead" (utilization drops to ___).
#
# 2. **Layer patterns:** Shallow layers (0-___) show [higher/lower] effective rank than
#    deep layers (___-N). Intruder dimensions concentrate in layers ___.
#
# 3. **Data-dependent collapse:** With only 100 samples, capacity utilization drops to ___,
#    suggesting ___% of LoRA dimensions undergo the equivalent of posterior collapse.
#
# 4. **Target module impact:** Adding FFN modules [increases/doesn't increase] total
#    effective rank proportionally. The "marginal return" of extra modules is ___.
#
# 5. **Spectral predictor:** Spectral entropy correlates with downstream score at R²=___.
#    This suggests spectral analysis can serve as a cheap proxy for full evaluation.

# %% [markdown]
# ## Next Steps
#
# 1. Add downstream evaluation (integrate llm_eval framework) to get generalization scores
# 2. Run Experiment 5 (adapter composability) if time permits
# 3. Produce final publication-quality figures for cover letter
# 4. Write 1-page research summary connecting findings to CLeAR-VAE analogy
