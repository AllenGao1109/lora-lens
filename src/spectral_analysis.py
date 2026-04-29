"""
LoRA-Lens: Spectral Analysis Toolkit
=====================================
SVD-based analysis of LoRA adapter weight structures.
Computes effective rank, intruder dimensions, spectral entropy,
and novel disentanglement-inspired metrics.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpectralProfile:
    """Spectral analysis results for a single weight matrix."""
    layer_name: str
    nominal_rank: int
    singular_values: np.ndarray          # σ_1, ..., σ_r
    effective_rank: float                 # exp(spectral entropy)
    spectral_entropy: float              # H(σ̃)
    capacity_utilization: float          # effective_rank / nominal_rank
    svr_top1: float                      # energy fraction in top-1 direction
    svr_top5: float                      # energy fraction in top-5 directions
    intruder_scores: np.ndarray          # per singular vector IDS
    intruder_count: int                  # count of IDS > threshold
    frobenius_norm: float                # ||ΔW||_F


@dataclass
class ModelSpectralReport:
    """Aggregated spectral report across all adapted layers."""
    model_name: str
    config_name: str
    layer_profiles: list[SpectralProfile] = field(default_factory=list)

    @property
    def mean_effective_rank(self) -> float:
        return np.mean([p.effective_rank for p in self.layer_profiles])

    @property
    def mean_capacity_utilization(self) -> float:
        return np.mean([p.capacity_utilization for p in self.layer_profiles])

    @property
    def total_intruder_count(self) -> int:
        return sum(p.intruder_count for p in self.layer_profiles)

    @property
    def mean_spectral_entropy(self) -> float:
        return np.mean([p.spectral_entropy for p in self.layer_profiles])

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "config_name": self.config_name,
            "mean_effective_rank": self.mean_effective_rank,
            "mean_capacity_utilization": self.mean_capacity_utilization,
            "mean_spectral_entropy": self.mean_spectral_entropy,
            "total_intruder_count": self.total_intruder_count,
            "num_layers": len(self.layer_profiles),
            "per_layer": [
                {
                    "layer": p.layer_name,
                    "effective_rank": round(p.effective_rank, 4),
                    "capacity_utilization": round(p.capacity_utilization, 4),
                    "spectral_entropy": round(p.spectral_entropy, 4),
                    "intruder_count": p.intruder_count,
                    "svr_top1": round(p.svr_top1, 4),
                    "svr_top5": round(p.svr_top5, 4),
                    "frobenius_norm": round(p.frobenius_norm, 6),
                    "singular_values": p.singular_values.tolist(),
                }
                for p in self.layer_profiles
            ],
        }


# =============================================================================
# CORE SPECTRAL METRICS
# =============================================================================

def compute_singular_values(delta_w: np.ndarray) -> np.ndarray:
    """
    Compute singular values of the weight update ΔW = BA.

    Args:
        delta_w: Weight update matrix (d_out × d_in)
    Returns:
        Singular values in descending order
    """
    U, S, Vt = np.linalg.svd(delta_w, full_matrices=False)
    return S


def compute_spectral_entropy(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """
    Spectral entropy: H(σ̃) = -Σ σ̃_i log(σ̃_i)
    where σ̃_i = σ_i / Σ_j σ_j (normalized singular values).

    Measures how "spread out" the energy is across singular directions.
    High entropy = energy uniformly distributed = many active directions.
    Low entropy = energy concentrated in few directions.

    Analogy to VAE: This is like measuring the effective dimensionality
    of the latent space — how many dimensions actually carry information.
    """
    sv = singular_values[singular_values > eps]
    if len(sv) == 0:
        return 0.0
    # Normalize to probability distribution
    sv_norm = sv / sv.sum()
    entropy = -np.sum(sv_norm * np.log(sv_norm + eps))
    return float(entropy)


def compute_effective_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """
    Effective rank = exp(spectral_entropy).

    Interpretation: The "equivalent number of equal singular values"
    that would produce the same entropy. Ranges from 1 (all energy
    in one direction) to r (perfectly uniform).

    Analogy to VAE: Like counting active latent dimensions.
    A VAE with posterior collapse has low effective dimensionality;
    a LoRA adapter with rank deficiency has low effective rank.
    """
    entropy = compute_spectral_entropy(singular_values, eps)
    return float(np.exp(entropy))


def compute_svr(singular_values: np.ndarray, top_k: int = 5) -> float:
    """
    Singular Value concentration Ratio: fraction of total energy
    in the top-k directions.

    SVR_k = (Σ_{i=1}^{k} σ_i²) / (Σ_{i=1}^{r} σ_i²)

    High SVR = most energy in few directions = concentrated/efficient.
    """
    sv_sq = singular_values ** 2
    total = sv_sq.sum()
    if total == 0:
        return 0.0
    top_k_energy = sv_sq[:min(top_k, len(sv_sq))].sum()
    return float(top_k_energy / total)


# =============================================================================
# INTRUDER DIMENSION DETECTION
# =============================================================================

def compute_intruder_scores(
    W_pretrained: np.ndarray,
    W_finetuned: np.ndarray,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """
    Intruder Dimension Score (IDS) from MIT "Illusion of Equivalence" paper.

    For each singular vector u_i of W_finetuned, compute:
        IDS(u_i) = 1 - max_j |cos(u_i, u_j^pretrained)|

    High IDS means the direction is "new" — not present in pretrained model.

    Args:
        W_pretrained: Pretrained weight matrix
        W_finetuned: Fine-tuned weight matrix
        top_k: Only analyze top-k singular vectors (default: all)
    Returns:
        Array of IDS values per singular vector
    """
    U_pre, _, _ = np.linalg.svd(W_pretrained, full_matrices=False)
    U_ft, _, _ = np.linalg.svd(W_finetuned, full_matrices=False)

    if top_k is not None:
        U_ft = U_ft[:, :top_k]

    # Cosine similarity matrix: (n_ft_vectors × n_pre_vectors)
    cos_sim = np.abs(U_ft.T @ U_pre)  # shape: (k, m)

    # IDS = 1 - max similarity to any pretrained direction
    max_sim = cos_sim.max(axis=1)
    ids = 1.0 - max_sim

    return ids


# =============================================================================
# ADAPTER EXTRACTION
# =============================================================================

def extract_lora_delta_weights(model) -> dict[str, np.ndarray]:
    """
    Extract ΔW = B @ A for each LoRA-adapted layer.

    Returns:
        {layer_name: delta_W_numpy} dict
    """
    delta_weights = {}
    lora_A_weights = {}
    lora_B_weights = {}

    for name, param in model.named_parameters():
        if "lora_A" in name:
            # Key: extract the base layer name
            base_name = name.split(".lora_A")[0]
            lora_A_weights[base_name] = param.detach().cpu().float().numpy()
        elif "lora_B" in name:
            base_name = name.split(".lora_B")[0]
            lora_B_weights[base_name] = param.detach().cpu().float().numpy()

    for base_name in lora_A_weights:
        if base_name in lora_B_weights:
            A = lora_A_weights[base_name]  # (r, d_in)
            B = lora_B_weights[base_name]  # (d_out, r)
            delta_W = B @ A                # (d_out, d_in)
            delta_weights[base_name] = delta_W

    return delta_weights


def extract_pretrained_weights(model) -> dict[str, np.ndarray]:
    """
    Extract the frozen pretrained weight matrices for layers that have LoRA adapters.

    Returns:
        {layer_name: W0_numpy} dict
    """
    pretrained_weights = {}

    # Find which layers have LoRA
    lora_layers = set()
    for name, _ in model.named_parameters():
        if "lora_A" in name:
            base_name = name.split(".lora_A")[0]
            lora_layers.add(base_name)

    # Extract base weights for those layers
    for name, param in model.named_parameters():
        if "lora" not in name and "weight" in name:
            # Check if this corresponds to a LoRA layer
            # Strip ".weight" to get module path
            module_path = name.replace(".weight", "")
            # Match against known LoRA layer paths
            for lora_name in lora_layers:
                # lora_name might be like "model.layers.0.self_attn.q_proj"
                # base weight name might be "base_model.model.model.layers.0.self_attn.q_proj.weight"
                if lora_name.replace("base_model.model.", "") in module_path or module_path in lora_name:
                    pretrained_weights[lora_name] = param.detach().cpu().float().numpy()
                    break

    return pretrained_weights


# =============================================================================
# FULL SPECTRAL ANALYSIS PIPELINE
# =============================================================================

def analyze_adapter(
    model,
    config_name: str = "",
    model_name: str = "",
    nominal_rank: int = 16,
    intruder_threshold: float = 0.9,
) -> ModelSpectralReport:
    """
    Run full spectral analysis on all LoRA-adapted layers.

    Args:
        model: HuggingFace model with PEFT LoRA adapters
        config_name: Name of this experiment config (e.g., "R16")
        model_name: Base model name
        nominal_rank: The rank r used for LoRA
        intruder_threshold: IDS threshold for counting intruder dimensions
    Returns:
        ModelSpectralReport with per-layer spectral profiles
    """
    print(f"Extracting LoRA delta weights...")
    delta_weights = extract_lora_delta_weights(model)
    print(f"Found {len(delta_weights)} adapted layers")

    # Try to get pretrained weights for intruder analysis
    print(f"Extracting pretrained weights for intruder analysis...")
    pretrained_weights = extract_pretrained_weights(model)
    print(f"Found {len(pretrained_weights)} matching pretrained weights")

    report = ModelSpectralReport(
        model_name=model_name,
        config_name=config_name,
    )

    for layer_name, delta_w in sorted(delta_weights.items()):
        # Singular values
        sv = compute_singular_values(delta_w)

        # Core metrics
        eff_rank = compute_effective_rank(sv)
        entropy = compute_spectral_entropy(sv)
        cap_util = eff_rank / max(nominal_rank, 1)
        svr1 = compute_svr(sv, top_k=1)
        svr5 = compute_svr(sv, top_k=5)
        frob = float(np.linalg.norm(delta_w, 'fro'))

        # Intruder dimensions
        if layer_name in pretrained_weights:
            W0 = pretrained_weights[layer_name]
            W_ft = W0 + delta_w
            ids = compute_intruder_scores(W0, W_ft, top_k=nominal_rank)
        else:
            # Fallback: can't compute intruders without pretrained weights
            ids = np.zeros(min(nominal_rank, len(sv)))

        intruder_count = int((ids > intruder_threshold).sum())

        profile = SpectralProfile(
            layer_name=layer_name,
            nominal_rank=nominal_rank,
            singular_values=sv,
            effective_rank=eff_rank,
            spectral_entropy=entropy,
            capacity_utilization=cap_util,
            svr_top1=svr1,
            svr_top5=svr5,
            intruder_scores=ids,
            intruder_count=intruder_count,
            frobenius_norm=frob,
        )
        report.layer_profiles.append(profile)

    print(f"\n{'='*60}")
    print(f"Spectral Report: {config_name}")
    print(f"{'='*60}")
    print(f"  Mean effective rank:        {report.mean_effective_rank:.3f} / {nominal_rank}")
    print(f"  Mean capacity utilization:  {report.mean_capacity_utilization:.3f}")
    print(f"  Mean spectral entropy:      {report.mean_spectral_entropy:.3f}")
    print(f"  Total intruder dimensions:  {report.total_intruder_count}")
    print(f"  Layers analyzed:            {len(report.layer_profiles)}")
    print(f"{'='*60}")

    return report


# =============================================================================
# NOVEL METRICS (FROM VAE RESEARCH)
# =============================================================================

def compute_capacity_utilization_curve(
    reports: list[ModelSpectralReport],
) -> list[dict]:
    """
    Capacity utilization across different configs.
    Analogous to plotting active latent dimensions vs β in β-VAE.

    Returns list of {config_name, nominal_rank, effective_rank, utilization}
    """
    curve = []
    for r in reports:
        for p in r.layer_profiles:
            curve.append({
                "config": r.config_name,
                "layer": p.layer_name,
                "nominal_rank": p.nominal_rank,
                "effective_rank": p.effective_rank,
                "utilization": p.capacity_utilization,
            })
    return curve


def compute_subspace_overlap(
    delta_w_a: np.ndarray,
    delta_w_b: np.ndarray,
    top_k: int = 8,
) -> float:
    """
    Subspace Overlap Score (SOS) between two adapters.

    Measures how much the active subspaces overlap.
    High overlap → similar directions learned → potential conflict in merging.
    Low overlap → orthogonal directions → safe to merge.

    Analogous to measuring overlap between z_c and z_s subspaces in CLeAR-VAE.

    Args:
        delta_w_a, delta_w_b: Two adapter weight updates
        top_k: Number of top singular directions to compare
    Returns:
        Overlap score in [0, 1]
    """
    U_a, _, _ = np.linalg.svd(delta_w_a, full_matrices=False)
    U_b, _, _ = np.linalg.svd(delta_w_b, full_matrices=False)

    U_a_top = U_a[:, :top_k]
    U_b_top = U_b[:, :top_k]

    # Grassmann distance proxy: Frobenius norm of cross-correlation
    overlap = np.linalg.norm(U_a_top.T @ U_b_top, 'fro') / top_k
    return float(np.clip(overlap, 0, 1))


def compute_layer_ablation_importance(
    model,
    eval_fn,
    delta_weights: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Measure each layer's importance by zeroing out its adapter and
    measuring eval score drop.

    Args:
        model: PEFT model
        eval_fn: Callable that returns a score (higher = better)
        delta_weights: Pre-extracted delta weights
    Returns:
        {layer_name: score_drop} dict
    """
    # Baseline score with all adapters active
    baseline_score = eval_fn(model)

    importance = {}
    for layer_name in delta_weights:
        # Zero out this layer's LoRA weights temporarily
        lora_params = {}
        for name, param in model.named_parameters():
            if layer_name in name and ("lora_A" in name or "lora_B" in name):
                lora_params[name] = param.data.clone()
                param.data.zero_()

        # Evaluate
        ablated_score = eval_fn(model)
        importance[layer_name] = baseline_score - ablated_score

        # Restore
        for name, param in model.named_parameters():
            if name in lora_params:
                param.data.copy_(lora_params[name])

    return importance
