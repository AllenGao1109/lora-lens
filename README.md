# LoRA-Lens 🔍

**Spectral Anatomy of Low-Rank Adapters through a Representation Learning Lens**

A systematic study of LoRA adapter internals via SVD-based spectral analysis. We analyze how spectral properties (effective rank, intruder dimensions, singular value concentration) relate to downstream generalization — bridging insights from disentangled representation learning (VAEs) with parameter-efficient fine-tuning.

## Motivation

Most LoRA users treat adapters as black boxes: pick a rank, train, evaluate. But recent work from MIT (NeurIPS 2025) revealed that LoRA produces fundamentally different weight structures than full fine-tuning — including "intruder dimensions" that cause forgetting. This project goes deeper: we systematically map how adapter spectral structure changes across rank, layer, data regime, and target modules, and whether spectral signatures can **predict** generalization without running downstream evaluation.

**Key insight:** The tools developed for analyzing VAE latent spaces (effective dimensionality, disentanglement metrics, posterior collapse detection) transfer directly to analyzing LoRA adapters. Both are low-rank structures learned within a high-dimensional parameter space.

## Research Questions

- **RQ1:** How does spectral structure (effective rank, intruder count, spectral entropy) vary with nominal rank r?
- **RQ2:** Do shallow vs deep layers exhibit systematically different spectral patterns?
- **RQ3:** What is the quantitative relationship between adapter effective rank and OOD generalization?
- **RQ4:** Can spectral properties alone predict adapter quality (without downstream eval)?

## Project Structure

```
LoRA-Lens/
├── configs/
│   └── experiment_configs.yaml       # All hyperparameter configs
├── src/
│   ├── __init__.py
│   ├── train_lora.py                 # LoRA training with PEFT
│   ├── spectral_analysis.py          # SVD extraction, effective rank, intruder detection
│   └── visualization.py              # Publication-quality figures
├── notebooks/
│   └── 01_train_and_analyze.ipynb    # Main experiment notebook (Colab-ready)
├── results/                          # Auto-generated experiment outputs
│   ├── adapters/                     # Saved LoRA weights
│   └── spectral/                     # SVD dumps + JSON reports
├── requirements.txt
└── README.md
```

## Setup

### Option A: Google Colab (Recommended for GPU Training)

```python
# Cell 1: Install
!pip install -q torch transformers peft datasets accelerate
!pip install -q sentence-transformers matplotlib seaborn pandas numpy scipy

# Cell 2: Clone repo
!git clone https://github.com/AllenGao1109/LoRA-Lens.git
%cd LoRA-Lens

# Cell 3: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")  # Should show T4/A100/etc.
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

Colab free tier (T4 16GB) can handle:
- Qwen2.5-0.5B: all experiments, ~15 min per LoRA training run
- Qwen2.5-1.5B: rank ≤ 32, ~30 min per run

### Option B: Apple Silicon (M3 Max)

```bash
# Clone
git clone https://github.com/AllenGao1109/LoRA-Lens.git
cd LoRA-Lens

# Create environment
conda create -n lora-spectral python=3.12 -y
conda activate lora-spectral

# Install dependencies
pip install -r requirements.txt

# Note: Use CPU for training (MPS has compatibility issues with some model architectures)
# 0.5B model on CPU: ~60 min per training run, spectral analysis is instant
```

### Requirements

```
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
datasets>=2.19.0
accelerate>=0.30.0
sentence-transformers>=2.2.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

## Experiments Overview

### Experiment 1: Rank Ablation

How does nominal rank affect spectral structure and generalization?

| Config | Rank | Target Modules | Train Data | ~Time (Colab T4) |
|--------|------|----------------|------------|-------------------|
| R4     | 4    | q_proj, v_proj | Alpaca-1k  | 15 min            |
| R8     | 8    | q_proj, v_proj | Alpaca-1k  | 15 min            |
| R16    | 16   | q_proj, v_proj | Alpaca-1k  | 18 min            |
| R32    | 32   | q_proj, v_proj | Alpaca-1k  | 20 min            |
| R64    | 64   | q_proj, v_proj | Alpaca-1k  | 25 min            |

**Metrics per config:** eval scores (ID + OOD), effective rank per layer, intruder dimension count, spectral entropy, SVR concentration ratio.

### Experiment 2: Layer-wise Spectral Analysis

Using rank=16 baseline, extract per-layer spectral fingerprint and correlate with layer importance (ablation: zero out each layer's adapter, measure eval drop).

### Experiment 3: Data Scaling

Fixed rank=16, vary training samples: 100 / 500 / 1k / 2k / 5k. Measure how data availability affects capacity utilization (effective_rank / nominal_rank).

### Experiment 4: Target Module Selection

Compare q+v only vs all attention vs all linear layers. Does adding more adaptable parameters improve effective rank proportionally?

### Experiment 5: Adapter Composability (Optional)

Train separate adapters on instruction-following vs math reasoning. Merge with varying interpolation weights. Does subspace overlap predict merge quality?

## Spectral Analysis Toolkit

Core metrics implemented in `src/spectral_analysis.py`:

**From existing literature:**
- **Effective Rank** — exponential of spectral entropy; measures active dimensionality
- **Intruder Dimension Score** — cosine distance of each fine-tuned singular vector to nearest pretrained singular vector
- **Singular Value Concentration Ratio (SVR)** — energy fraction in top-k directions
- **Spectral Entropy** — Shannon entropy of normalized singular value distribution

**Novel (bridging from VAE research):**
- **Adapter Disentanglement Index (ADI)** — do different singular directions specialize in different eval dimensions?
- **Subspace Overlap Score (SOS)** — how much do two adapters' active subspaces overlap?
- **Capacity Utilization Ratio** — effective_rank / nominal_rank, analogous to active latent dimensions / total dimensions in VAE

## Expected Key Figures

1. **Rank vs Generalization U-Curve** — X: nominal rank, Y: OOD score, color: effective rank
2. **Layer-wise Spectral Heatmap** — layers × spectral metrics, showing shallow vs deep patterns
3. **Capacity Utilization Curve** — effective_rank / nominal_rank vs training data size (posterior collapse analog)
4. **Spectral Predictor Scatter** — spectral entropy vs OOD score (can we predict generalization from spectra?)
5. **Intruder Dimension Anatomy** — per-layer intruder count + intervention effect (scaling down intruders)

## Connection to Prior Work

| Paper | Key Finding | Our Extension |
|-------|-------------|---------------|
| LoRA vs Full FT (MIT, NeurIPS 2025) | Intruder dimensions exist in LoRA | How do intruders vary with rank? Do they accumulate in specific layers? |
| Weight Spectra (SJTU, 2025) | Top singular values amplified during FT | We track this per-layer and per-rank, adding effective rank metric |
| Flat-LoRA (ICML 2025) | Flat loss landscape → better generalization | We connect flatness to spectral structure: are spectrally efficient adapters also flatter? |
| Spectral Surgery (2026) | Many LoRA singular directions are wasted | We quantify waste ratio across rank/data/layer and propose capacity utilization metric |
| CLeAR-VAE (our prior work) | Disentanglement metrics for latent space analysis | We transfer DCI-style analysis to LoRA singular directions |

## Citation

If you use the spectral analysis toolkit or experimental design:

```bibtex
@misc{gao2026loralens,
  author = {Gao, Zhiyuan},
  title = {LoRA-Lens: Spectral Anatomy of Low-Rank Adapters through a Representation Learning Lens},
  year = {2026},
  url = {https://github.com/AllenGao1109/LoRA-Lens}
}
```

## License

MIT
