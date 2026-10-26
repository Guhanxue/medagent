"""
Few-shot adaptation module for MedAdaptAgent.

When zero-shot TTA isn't sufficient, the agent recommends few-shot
fine-tuning with labeled target data. This module:

1. Selects the best fine-tuning strategy based on agent analysis
   (shift type, severity, contrast mechanism — same reasoning pipeline)
2. Fine-tunes the adapted model on N labeled target cases
3. Provides scaling law curves (Dice vs N annotations)

The agent's analysis drives strategy selection:
  - Cross-modality (CT→MR): FDA augmentation + source mixing + decoder fine-tuning
  - Cross-sequence (T1→T2): source mixing + partial fine-tuning
  - Cross-site (mild): BN+output only, conservative LR
  - The scope widens as more labeled data becomes available

Usage:
    from medadaptagent.adaptation.few_shot import FewShotAdapter

    adapter = FewShotAdapter(
        adapted_model, analysis, device, num_classes, class_names,
        source_images=src_imgs, source_labels=src_lbls,
    )
    scaling = adapter.run_scaling_law(
        target_images, target_labels,
        volume_ids=vol_ids,  # group slices by volume
        shot_counts=[1, 2, 5, 10],
    )
"""

import copy
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ================================================================
# Dataset
# ================================================================

class FewShotDataset(Dataset):
    """Dataset for few-shot fine-tuning with augmentation."""

    def __init__(self, images, labels, augment=True):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        lbl = self.labels[idx].copy()

        if self.augment:
            if np.random.random() > 0.5:
                img = img[:, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
            if np.random.random() > 0.5:
                img = img[::-1].copy()
                lbl = lbl[::-1].copy()
            k = np.random.randint(4)
            if k > 0:
                img = np.rot90(img, k, axes=(0, 1)).copy()
                lbl = np.rot90(lbl, k, axes=(0, 1)).copy()
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-0.1, 0.1)
                img = np.clip(img * alpha + beta, 0, 1).astype(np.float32)

        # HWC -> CHW
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1)

        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()


# ================================================================
# Strategy selection (agent reasoning)
# ================================================================

def select_strategy(analysis: Dict, n_shots: int,
                     val_dice: float = 0.0,
                     model_info: Dict = None) -> Dict:
    """Agent-driven few-shot method selection.

    Uses analysis signals + val_dice + model structure to decide the
    best approach. val_dice is the adapted model's Dice on the N labeled
    cases (available in practice — no full GT needed).

    Available method components:
    1. Supervised (CE+Dice) on labeled target — always on
    2. Source mixing — regularizes when few target labels
    3. FDA augmentation — style-transfers source to target domain
    4. Mean Teacher — semi-supervised on unlabeled target data
    5. Scope/LR selection — how aggressively to fine-tune

    Each component is activated based on agent observations:
    - val_dice: how well does the adapted model already work on target?
    - halluc_ratio: are zero-shot predictions trustworthy? (label-free)
    - contrast_mechanism: what kind of domain gap?
    - what_worked_zero_shot: which tools already proved useful?
    - severity: how aggressive should fine-tuning be?
    - model_info: model size/structure (from analyze_model_structure)
    """
    contrast_change = analysis.get("contrast_mechanism_change", "same")
    severity = analysis.get("verified_severity", 0.5)
    if not isinstance(severity, (int, float)):
        severity = 0.5
    severity = float(severity)
    intensity_mapping = analysis.get("intensity_mapping_type", "linear")
    halluc_ratio = analysis.get("halluc_ratio", 0.0)
    # Signals from zero-shot pipeline about what worked
    zero_shot_used_fda = analysis.get("used_fda", False)

    num_classes = analysis.get("num_classes", 5)
    has_gpu = analysis.get("has_gpu", True)

    # Model structure signals (from analyze_model_structure)
    # Defaults are for our 7.8M param BatchNorm UNet (backward compatible).
    if model_info is None:
        model_info = {}
    total_params = model_info.get("total_params", 7_800_000)
    norm_type = model_info.get("norm_type", "batchnorm")
    arch_family = model_info.get("arch_family", "cnn")
    has_batchnorm = model_info.get("has_batchnorm", True)
    scope_params = model_info.get("scope_params", {})

    strategy = {
        "mode": "train",  # "calibrate" or "train"
        "scope": "all",
        "lr": 1e-4,
        "epochs": 50,
        "use_source_mix": False,
        "source_mix_ratio": 0.0,
        "use_fda_augment": False,
        "use_semi_supervised": False,
        "denoising_method": "prototype",
        "use_feature_augment": False,
        "rationale": "",
    }

    rationale_parts = []

    # ============================================================
    # DECISION 0: What to do with N labeled volumes
    # ============================================================
    #
    # The agent reasons through 3 explicit factors:
    #
    #   Factor 1: DATA SUFFICIENCY — do we have enough labeled data
    #             to train without overfitting?
    #   Factor 2: ADAPTATION GAP — how much room for improvement
    #             does the model have?
    #   Factor 3: OVERFITTING RISK — given data/params ratio,
    #             what scope and regularization are safe?
    #
    # The decision is NOT binary (calibrate vs train). It's a
    # spectrum of aggressiveness:
    #
    #   N=0:     identity (no labeled data)
    #   N=1:     identity OR output-only (1 volume too noisy for deeper)
    #   N=2-4:   train with heavy regularization (source mix, low LR,
    #            decoder+output scope, few epochs)
    #   N=5-10:  train with moderate regularization
    #   N=10+:   train aggressively (all params, standard LR)

    avg_slices = analysis.get("avg_slices_per_vol", 50)
    samples_per_class = n_shots * avg_slices / max(num_classes - 1, 1)

    # --- Factor 1: Data sufficiency ---
    # total labeled pixels available for training
    total_labeled_samples = samples_per_class * max(num_classes - 1, 1)
    decoder_params = scope_params.get("decoder+output", total_params * 0.5)
    output_params = scope_params.get("output", total_params * 0.01)
    data_to_decoder_ratio = total_labeled_samples / max(decoder_params / 1e4, 1)
    # Rough scale: ratio > 1 means ~10K samples per 10K params (safe)
    #              ratio < 0.1 means dangerously few samples

    # --- Factor 2: Adaptation gap ---
    # How much could the model improve?
    adaptation_gap = 1.0 - val_dice if val_dice > 0 else 0.5
    model_needs_help = adaptation_gap > 0.2  # >20% gap means room to improve

    # --- Factor 3: Overfitting risk ---
    # Combine data sufficiency with model quality
    # If model already good AND data is sparse → don't risk it
    # If model needs help AND data is sparse → train carefully
    # If data is abundant → always train

    if not has_gpu:
        # No GPU: threshold calibration only
        strategy["mode"] = "calibrate"
        strategy["calibration_method"] = "threshold"
        rationale_parts.append(f"Calibrate: threshold (no GPU)")

    elif n_shots == 0:
        # No labeled data: identity
        strategy["mode"] = "calibrate"
        strategy["calibration_method"] = "identity"
        rationale_parts.append(f"Calibrate: identity (no labeled data)")

    elif n_shots == 1 and not model_needs_help:
        # 1 volume, model already works → identity (1 volume too noisy)
        strategy["mode"] = "calibrate"
        strategy["calibration_method"] = "identity"
        rationale_parts.append(
            f"Calibrate: identity (N=1 too noisy, model already at "
            f"{val_dice:.0%}, gap={adaptation_gap:.0%} < 20%)")

    elif n_shots == 1 and model_needs_help:
        # 1 volume, model needs help → output-only fine-tuning
        # Only touch the output layer (~130 params) to remap features
        # to target labels. Encoder features are preserved.
        strategy["mode"] = "calibrate"
        if halluc_ratio > 0.5 and has_batchnorm:
            strategy["calibration_method"] = "bn_output_ft"
            strategy["calibration_epochs"] = 25
            strategy["calibration_lr"] = 5e-4
            rationale_parts.append(
                f"Calibrate: bn+output FT (N=1, halluc {halluc_ratio:.0%}: "
                f"BN stats wrong + output mapping wrong)")
        else:
            strategy["calibration_method"] = "output_ft"
            strategy["calibration_epochs"] = 20
            strategy["calibration_lr"] = 1e-3
            rationale_parts.append(
                f"Calibrate: output FT (N=1, ~{int(total_labeled_samples)} samples, "
                f"gap={adaptation_gap:.0%}: remap output layer only)")

    else:
        # N >= 2: TRAIN — adjust scope/LR/regularization by data sufficiency
        strategy["mode"] = "train"
        rationale_parts.append(f"Train (~{int(samples_per_class)} samples/class)")

    rationale_parts.append(
        f"Train (~{samples_per_class:.0f} samples/class)")

    # ============================================================
    # DECISION 1: Semi-supervised learning (Mean Teacher)
    # ============================================================
    # Mean Teacher generates pseudo-labels from teacher (EMA of student)
    # on unlabeled target data. This ONLY helps when the adapted model's
    # predictions are mostly correct — otherwise the teacher propagates
    # wrong labels and hurts performance.
    #
    # Signal: halluc_ratio from DiagnosisAgent
    #   - Low hallucination (<30%): predictions trustworthy -> MT helps
    #   - High hallucination (>30%): teacher would reinforce errors -> skip
    #
    # This mirrors the zero-shot decision: when entropy minimization was
    # disabled due to hallucination, semi-supervised is also unsafe.
    if halluc_ratio < 0.3:
        strategy["use_semi_supervised"] = True
        # Clean predictions → prototype denoising sufficient
        strategy["denoising_method"] = "prototype"
        rationale_parts.append(
            f"MeanTeacher ON (halluc {halluc_ratio:.0%} < 30%: "
            f"predictions reliable, prototype denoising)")
    elif halluc_ratio < 0.5:
        # Moderate noise → still use MT but with VLM proxy denoising
        # (ProDe-inspired: CLIP gates class presence to filter hallucinations)
        strategy["use_semi_supervised"] = True
        strategy["denoising_method"] = "vlm_proxy"
        strategy["use_feature_augment"] = True  # SF(DA)²-inspired
        rationale_parts.append(
            f"MeanTeacher ON + VLM denoising (halluc {halluc_ratio:.0%}: "
            f"moderate noise, ProDe-style CLIP gating + "
            f"SF(DA)²-style feature augmentation)")
    else:
        strategy["denoising_method"] = "none"
        rationale_parts.append(
            f"MeanTeacher OFF (halluc {halluc_ratio:.0%}: "
            f"teacher would propagate errors)")

    # ============================================================
    # DECISION 2: Data augmentation strategy
    # ============================================================
    # The agent selects augmentation based on what the domain gap is
    # and what worked during zero-shot:
    #
    # Cross-modality (fundamentally_different):
    #   -> FDA worked in zero-shot to bridge style gap
    #   -> Source labels are valuable (different physics but same anatomy)
    #   -> Use FDA-augmented source + source mixing
    #
    # Cross-sequence (different contrast):
    #   -> Source data helps but intensities are reversed
    #   -> Source mixing with lower ratio (target labels more important)
    #
    # Same modality (mild/moderate shift):
    #   -> Source mixing for regularization, no FDA needed
    if contrast_change == "fundamentally_different":
        strategy["use_fda_augment"] = True  # proven in zero-shot
        # Source mixing rationale: BatchNorm accumulates running statistics
        # across source+target batches, so source data regularizes BN stats.
        # InstanceNorm normalizes per-sample, so source samples contribute
        # no transferable statistics — their gradient signal is wasted at
        # test time when only target data appears.
        if has_batchnorm:
            strategy["use_source_mix"] = True
            strategy["source_mix_ratio"] = max(0.3, 1.0 - n_shots / 20)
        else:
            # InstanceNorm/LayerNorm: minimal source mixing for label
            # diversity only, capped low to avoid diluting target signal
            strategy["use_source_mix"] = n_shots <= 2
            strategy["source_mix_ratio"] = 0.2 if n_shots <= 2 else 0.0
        rationale_parts.append(
            f"FDA augment (cross-modality, "
            f"{'proven in zero-shot' if zero_shot_used_fda else 'bridging physics gap'})"
            + (f" + source mix {strategy['source_mix_ratio']:.1f}"
               if strategy["use_source_mix"] else ""))

    elif contrast_change == "different" or intensity_mapping == "non_linear":
        strategy["use_source_mix"] = True
        strategy["source_mix_ratio"] = max(0.2, 0.8 - n_shots / 25)
        rationale_parts.append(
            f"Source mix {strategy['source_mix_ratio']:.1f} "
            f"(cross-sequence, contrast reversal)")

    elif severity > 0.3:
        strategy["use_source_mix"] = True
        strategy["source_mix_ratio"] = 0.3
        rationale_parts.append(
            f"Source mix 0.3 (moderate shift, regularization)")

    # ============================================================
    # DECISION 3: Scope — how many parameters to train
    # ============================================================
    #
    # Scope determines overfitting risk. The agent reasons:
    #   - params_per_sample ratio: how many params per labeled sample?
    #   - N (labeled volumes): more data → wider scope is safe
    #   - shift severity: harder shifts need deeper adaptation
    #
    # Spectrum:
    #   output_only → bn+output → decoder+output → all
    #   (safest)                                   (most powerful)
    #
    # The key: at N=2, we CAN train, but we must restrict scope
    # to avoid overfitting. At N=10+, we can open up to all params.

    all_params = scope_params.get("all", total_params)
    effective_samples = total_labeled_samples

    if n_shots <= 2:
        # Very few labels: restrict to decoder+output at most
        # This is enough to remap features → target classes
        # but doesn't risk destroying encoder representations
        if contrast_change == "fundamentally_different":
            strategy["scope"] = "decoder+output"
        else:
            strategy["scope"] = "decoder+output"
        rationale_parts.append(
            f"scope={strategy['scope']} (N={n_shots}: restrict scope, "
            f"~{int(decoder_params/1e6):.1f}M trainable params)")
    elif n_shots <= 5:
        # Moderate data: decoder+output or all depending on model size
        params_per_sample = all_params / max(effective_samples, 1)
        if params_per_sample < 300_000:
            strategy["scope"] = "all"
        else:
            strategy["scope"] = "decoder+output"
        rationale_parts.append(
            f"scope={strategy['scope']} (~{int(effective_samples)} samples, "
            f"{params_per_sample/1e3:.0f}K params/sample)")
    else:
        # N >= 10: enough data for full model
        strategy["scope"] = "all"
        params_per_sample = all_params / max(effective_samples, 1)
        rationale_parts.append(
            f"scope=all (~{int(effective_samples)} samples, "
            f"{params_per_sample/1e3:.0f}K params/sample)")

    # ============================================================
    # DECISION 4: Learning rate — scaled by data sufficiency
    # ============================================================
    #
    # Principle: LR should be inversely proportional to overfitting risk.
    # - Low N (2-4): use low LR (5e-5) to avoid overshooting
    # - Medium N (5-10): standard LR (5e-4 or 1e-4)
    # - High N (10+): can be more aggressive (5e-4)
    #
    # Shift severity also matters: cross-modality needs larger updates.

    if n_shots <= 2:
        strategy["lr"] = 5e-5
    elif n_shots <= 5:
        strategy["lr"] = 5e-5 if severity < 0.5 else 1e-4
    else:
        strategy["lr"] = 5e-4 if severity > 0.5 else 1e-4

    rationale_parts.append(
        f"lr={strategy['lr']:.0e}")

    # ============================================================
    # DECISION 5: Training duration + regularization
    # ============================================================
    #
    # More epochs at low N to extract maximum signal from few samples.
    # Source mixing is stronger at low N for regularization.

    if n_shots <= 2:
        strategy["epochs"] = 60
        # Heavy source mixing to regularize small target set
        if not strategy.get("use_source_mix"):
            strategy["use_source_mix"] = True
            strategy["source_mix_ratio"] = 0.9  # mostly source
            rationale_parts.append(
                f"source mix 0.9 (heavy regularization for N={n_shots})")
    elif n_shots <= 5:
        strategy["epochs"] = 50 if severity > 0.5 else 35
    else:
        strategy["epochs"] = 40 if n_shots <= 10 else 30

    rationale_parts.append(
        f"epochs={strategy['epochs']}, "
        f"model={total_params/1e6:.1f}M/{arch_family}/{norm_type}")

    # Log actual trainable params for the chosen scope
    # Map strategy scope names to scope_params keys
    scope_to_key = {
        "all": "all", "decoder+output": "decoder+output",
        "bn+output": "norm+output", "bn+decoder1+output": "norm+output",
        "output_only": "output_only",
    }
    sp_key = scope_to_key.get(strategy["scope"], "all")
    chosen_scope_params = scope_params.get(sp_key, total_params)
    rationale_parts.append(
        f"scope={strategy['scope']} (~{chosen_scope_params/1e6:.1f}M params), "
        f"lr={strategy['lr']:.0e}, epochs={strategy['epochs']}, "
        f"model={total_params/1e6:.1f}M/{arch_family}/{norm_type}")
    strategy["rationale"] = " | ".join(rationale_parts)
    strategy["model_info"] = model_info

    return strategy



# ================================================================
# Model structure analysis
# ================================================================

def analyze_model_structure(model: nn.Module) -> Dict:
    """Detect model architecture properties for agent reasoning.

    Inspects the model to determine:
    - Total parameter count (overfitting risk scales with this)
    - Normalization type (BN recalibration only works with BatchNorm)
    - Architecture family (CNN vs Transformer — different layer semantics)
    - Per-scope parameter counts (actual trainable params at each scope)

    Returns a dict that can be added to the analysis for select_strategy.
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Detect normalization type — dispatches via the norm_utils single source
    # of truth so new norm types appear everywhere at once.
    from medadaptagent.graph.norm_utils import count_norm_layers, detect_norm_type
    norm_counts = count_norm_layers(model)
    norm_type = detect_norm_type(model)

    # Detect architecture family from layer types
    has_attention = any(
        hasattr(m, "num_heads") or "attention" in type(m).__name__.lower()
        or "multihead" in type(m).__name__.lower()
        for m in model.modules()
    )
    has_conv = any(isinstance(m, (nn.Conv2d, nn.Conv3d)) for m in model.modules())

    if has_attention:
        arch_family = "transformer"
    else:
        arch_family = "cnn"

    # Compute per-scope parameter counts. Norm params are identified by
    # module type (via norm_utils.iter_norm_modules), not by name substring.
    from medadaptagent.graph.norm_utils import iter_norm_modules
    norm_param_ids = set()
    for _, m, _ in iter_norm_modules(model):
        for p in m.parameters():
            norm_param_ids.add(id(p))

    output_pats = ["out_conv", "final", "seg_head", "seg_output", "head", "classifier"]
    # MONAI SwinUNETR uses "model.out.conv" — detect ".out." as module boundary
    output_boundary_pats = [".out."]
    decoder_pats = ["dec", "up", "upsample", "decoder", "bottleneck", "bridge"]

    scope_params = {"output_only": 0, "norm+output": 0, "decoder+output": 0, "all": 0}
    for name, param in model.named_parameters():
        name_lower = name.lower()
        n = param.numel()
        scope_params["all"] += n
        is_output = (any(p in name_lower for p in output_pats) or
                     any(p in name_lower for p in output_boundary_pats))
        is_norm = (id(param) in norm_param_ids) or any(
            p in name_lower for p in ["bn", "norm", "layernorm"])
        is_decoder = any(p in name_lower for p in decoder_pats)
        if is_output:
            scope_params["output_only"] += n
        if is_output or is_norm:
            scope_params["norm+output"] += n
        if is_output or is_norm or is_decoder:
            scope_params["decoder+output"] += n

    return {
        "total_params": total_params,
        "norm_type": norm_type,
        "arch_family": arch_family,
        "scope_params": scope_params,
        "has_batchnorm": norm_counts["batchnorm"] > 0,
    }


def _classify_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Classify each parameter into a semantic group.

    Uses naming heuristics that work across architectures:
    - encoder: enc, down, patch_embed, layers.0, layers.1, swinViT
    - decoder: dec, up, upsample, layers.3, layers.4
    - bottleneck: bottleneck, bridge, layers.2 (middle)
    - output: out_conv, final, seg_head, head, classifier
    - norm: bn, norm, layernorm (when not inside encoder/decoder)
    """
    classified = []

    # Heuristic patterns for each group
    encoder_patterns = ["enc", "down", "patch_embed", "swinViT", "encoder"]
    decoder_patterns = ["dec", "up", "upsample", "decoder"]
    bottleneck_patterns = ["bottleneck", "bridge"]
    output_patterns = ["out_conv", "final", "seg_head", "seg_output", "head", "classifier"]
    norm_patterns = ["bn", "norm"]

    for name, param in model.named_parameters():
        name_lower = name.lower()

        # Check output first (highest priority — always "output" group)
        if any(p in name_lower for p in output_patterns):
            classified.append(("output", param))
        elif any(p in name_lower for p in bottleneck_patterns):
            classified.append(("bottleneck", param))
        elif any(p in name_lower for p in decoder_patterns):
            classified.append(("decoder", param))
        elif any(p in name_lower for p in encoder_patterns):
            classified.append(("encoder", param))
        elif any(p in name_lower for p in norm_patterns):
            classified.append(("norm", param))
        else:
            classified.append(("other", param))

    return classified


# ================================================================
# Trainable scope
# ================================================================

def set_trainable_scope(model: nn.Module, scope: str,
                        model_info: Dict = None) -> Tuple[int, int]:
    """Set which parameters are trainable based on scope.

    Uses name-based matching that works across architectures.
    Scope definitions:
    - "all": everything trainable
    - "decoder+output": decoder blocks + bottleneck + all norms + output head
    - "bn+output": all normalization layers + output head (anywhere in model)
    - "output_only": just the final output/segmentation head
    """
    for p in model.parameters():
        p.requires_grad = False

    # Detect norm params by module type (handles BN inside Sequential blocks
    # where param names don't contain "bn"/"norm")
    from medadaptagent.graph.norm_utils import iter_norm_modules as _inm
    norm_param_ids = set()
    for _, m, _ in _inm(model):
        for p in m.parameters():
            norm_param_ids.add(id(p))

    output_patterns = ["out_conv", "final", "seg_head", "seg_output", "head", "classifier"]
    output_boundary_pats = [".out."]  # MONAI SwinUNETR: "model.out.conv"
    decoder_patterns = ["dec", "up", "upsample", "decoder", "bottleneck", "bridge"]

    trainable = 0
    total = 0

    for name, param in model.named_parameters():
        total += param.numel()
        name_lower = name.lower()
        is_output = (any(p in name_lower for p in output_patterns) or
                     any(p in name_lower for p in output_boundary_pats))
        is_norm = (id(param) in norm_param_ids) or any(
            p in name_lower for p in ["bn", "norm", "layernorm"])
        is_decoder = any(p in name_lower for p in decoder_patterns)
        should_train = False

        if scope == "all":
            should_train = True
        elif scope == "decoder+output":
            should_train = is_decoder or is_output or is_norm
        elif scope == "bn+output":
            should_train = is_output or is_norm
        elif scope == "output_only":
            if is_output:
                should_train = True

        if should_train:
            param.requires_grad = True
            trainable += param.numel()

    return trainable, total


# ================================================================
# Few-shot fine-tuning
# ================================================================

def few_shot_fine_tune(
    model: nn.Module,
    train_images: List[np.ndarray],
    train_labels: List[np.ndarray],
    strategy: Dict,
    device: str,
    num_classes: int,
    source_images: List[np.ndarray] = None,
    source_labels: List[np.ndarray] = None,
    unlabeled_images: List[np.ndarray] = None,
    fda_images: List[np.ndarray] = None,
    fda_labels: List[np.ndarray] = None,
    class_descriptions: List[str] = None,
) -> nn.Module:
    """Semi-supervised few-shot adaptation using agent-selected strategy.

    Best practices from literature for few-shot medical domain adaptation:
    1. Supervised loss on labeled target (CE + Dice)
    2. Semi-supervised loss on unlabeled target (consistency regularization
       via Mean Teacher with augmentation: teacher = EMA of student)
    3. Source mixing for regularization (prevents forgetting)
    4. Agent selects scope, LR, and method based on shift analysis

    The method adapts based on available data:
    - labeled only: supervised fine-tuning with augmentation
    - labeled + unlabeled: Mean Teacher semi-supervised learning
    - labeled + source: source-regularized fine-tuning
    - all three: full semi-supervised with source regularization
    """
    student = copy.deepcopy(model)
    student.to(device)
    student.train()

    trainable, total = set_trainable_scope(
        student, strategy["scope"],
        model_info=strategy.get("model_info"))

    # Build labeled dataset (target + optional source mix)
    labeled_images = list(train_images)
    labeled_labels = list(train_labels)

    if strategy.get("use_source_mix") and source_images is not None:
        ratio = strategy["source_mix_ratio"]
        n_source_mix = int(len(train_images) * ratio / max(1 - ratio, 0.1))
        n_source_mix = min(n_source_mix, len(source_images))
        if n_source_mix > 0:
            idx = np.random.choice(len(source_images), n_source_mix,
                                   replace=False)
            labeled_images.extend([source_images[i] for i in idx])
            labeled_labels.extend([source_labels[i] for i in idx])

    if strategy.get("use_fda_augment") and fda_images is not None:
        n_fda = min(len(fda_images), len(train_images) * 3)
        if n_fda > 0:
            idx = np.random.choice(len(fda_images), n_fda, replace=False)
            labeled_images.extend([fda_images[i] for i in idx])
            labeled_labels.extend([fda_labels[i] for i in idx])

    labeled_ds = FewShotDataset(labeled_images, labeled_labels, augment=True)
    labeled_loader = DataLoader(
        labeled_ds, batch_size=min(8, len(labeled_ds)),
        shuffle=True, num_workers=0, drop_last=False)

    # Semi-supervised: Mean Teacher on unlabeled target data
    use_semi = (unlabeled_images is not None and
                len(unlabeled_images) > 0 and
                strategy.get("use_semi_supervised", True))

    teacher = None
    unlabeled_loader = None
    unlabeled_raw = None  # Keep raw images for VLM denoising
    if use_semi:
        teacher = copy.deepcopy(student)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Sample subset of unlabeled for efficiency
        n_unlabeled = min(len(unlabeled_images), max(500, len(train_images) * 20))
        ul_idx = np.random.choice(len(unlabeled_images), n_unlabeled,
                                  replace=False)
        ul_imgs = [unlabeled_images[i] for i in ul_idx]
        unlabeled_raw = ul_imgs  # raw numpy for VLM denoising
        # Dummy labels (not used for loss, just for DataLoader)
        ul_lbls = [np.zeros_like(ul_imgs[0][:, :, 0] if ul_imgs[0].ndim == 3
                                 else ul_imgs[0], dtype=np.int64)
                    for _ in ul_imgs]
        unlabeled_ds = FewShotDataset(ul_imgs, ul_lbls, augment=True)
        unlabeled_loader = DataLoader(
            unlabeled_ds, batch_size=min(16, len(unlabeled_ds)),
            shuffle=True, num_workers=0, drop_last=False)

    # Initialize denoising components based on agent strategy
    denoising_method = strategy.get("denoising_method", "prototype")
    use_feature_augment = strategy.get("use_feature_augment", False)

    prototype_denoiser = None
    vlm_denoiser = None
    feature_augmenter = None

    if use_semi:
        from medadaptagent.adaptation.advanced_tta import (
            PrototypeDenoiser, extract_features_unet)
        if denoising_method in ("prototype", "both"):
            # Feature dim from model bottleneck — detect dynamically
            feat_dim = 256  # default
            for name, mod in student.named_modules():
                if any(k in name.lower() for k in ["bottleneck", "enc4", "bridge"]):
                    for p in mod.parameters():
                        if p.dim() >= 2:
                            feat_dim = p.shape[0]
                            break
                    break
            prototype_denoiser = PrototypeDenoiser(
                feat_dim, num_classes, momentum=0.99, similarity_threshold=0.5)
            print(f"    Prototype denoiser: feat_dim={feat_dim}")

        if denoising_method in ("vlm_proxy", "both") and class_descriptions:
            from medadaptagent.adaptation.advanced_tta import VLMProxyDenoiser
            vlm_denoiser = VLMProxyDenoiser(
                class_descriptions, device=device)
            print(f"    VLM proxy denoiser: {len(class_descriptions)} classes")

        if use_feature_augment:
            from medadaptagent.adaptation.advanced_tta import FeatureAugmentRegularizer
            feature_augmenter = FeatureAugmentRegularizer(
                perturbation_strength=0.1, low_freq_ratio=0.1)
            print(f"    Feature augmentation regularizer: ON")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=strategy["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=strategy["epochs"])

    ema_decay = 0.99
    consistency_weight = 0.1  # ramp up over epochs

    for epoch in range(strategy["epochs"]):
        # Ramp up consistency weight (0 -> target over first 30% of training)
        ramp = min(1.0, epoch / max(1, strategy["epochs"] * 0.3))
        cur_consistency = consistency_weight * ramp

        # --- Supervised pass on labeled data ---
        for imgs, lbls in labeled_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = student(imgs)

            # CE + Dice loss
            ce_loss = F.cross_entropy(logits, lbls, ignore_index=255)
            probs = F.softmax(logits, dim=1)
            dice_loss = 0.0
            n_valid = 0
            for c in range(1, num_classes):
                pred_c = probs[:, c]
                gt_c = (lbls == c).float()
                if gt_c.sum() > 0:
                    dice = (2 * (pred_c * gt_c).sum() + 1) / \
                           (pred_c.sum() + gt_c.sum() + 1)
                    dice_loss += (1 - dice)
                    n_valid += 1
            if n_valid > 0:
                dice_loss = dice_loss / n_valid

            loss = ce_loss + dice_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA teacher
            if teacher is not None:
                with torch.no_grad():
                    for tp, sp in zip(teacher.parameters(),
                                      student.parameters()):
                        tp.data.mul_(ema_decay).add_(sp.data, alpha=1 - ema_decay)

        # --- Semi-supervised pass on unlabeled data ---
        if use_semi and unlabeled_loader is not None and cur_consistency > 0:
            student.train()
            batch_start = 0
            for ul_imgs, _ in unlabeled_loader:
                ul_imgs = ul_imgs.to(device)
                batch_size_ul = ul_imgs.shape[0]

                # Teacher pseudo-labels
                with torch.no_grad():
                    teacher_logits = teacher(ul_imgs)
                    teacher_probs = F.softmax(teacher_logits, dim=1)
                    confidence, pseudo = teacher_probs.max(dim=1)
                    # Baseline: high-confidence filtering
                    mask = confidence > 0.9

                # --- Prototype denoising ---
                if prototype_denoiser is not None:
                    from medadaptagent.adaptation.advanced_tta import extract_features_unet
                    with torch.no_grad():
                        features = extract_features_unet(
                            teacher, ul_imgs,
                            target_size=(ul_imgs.shape[2], ul_imgs.shape[3]))
                    # Update prototypes from confident predictions
                    prototype_denoiser.update_prototypes(
                        features, pseudo, confidence, conf_threshold=0.9)
                    # Denoise pseudo-labels
                    denoised, proto_mask = prototype_denoiser.denoise(
                        features, pseudo, confidence)
                    pseudo = denoised
                    mask = mask & proto_mask

                # --- VLM proxy denoising ---
                if vlm_denoiser is not None and unlabeled_raw is not None:
                    raw_batch = unlabeled_raw[
                        batch_start:batch_start + batch_size_ul]
                    if len(raw_batch) == batch_size_ul:
                        denoised, vlm_mask = vlm_denoiser.denoise(
                            raw_batch, pseudo, confidence)
                        pseudo = denoised
                        mask = mask & vlm_mask

                batch_start += batch_size_ul

                if mask.sum() == 0:
                    continue

                # Student prediction
                student_logits = student(ul_imgs)
                student_log_probs = F.log_softmax(student_logits, dim=1)

                # Consistency loss: KL(teacher || student) on reliable pixels
                teacher_dist = teacher_probs.detach()
                kl_loss = F.kl_div(
                    student_log_probs, teacher_dist, reduction='none'
                ).sum(dim=1)
                consistency_loss = (kl_loss * mask.float()).sum() / \
                                   max(mask.float().sum(), 1.0)

                # Pseudo-label CE on denoised labels (reliable pixels only)
                pl_loss = torch.tensor(0.0, device=device)
                valid_pseudo = pseudo.clamp(min=0)
                if mask.sum() > 0:
                    ce = F.cross_entropy(
                        student_logits, valid_pseudo, reduction='none')
                    pl_loss = (ce * mask.float()).sum() / mask.float().sum()

                # Feature augmentation regularization
                feat_aug_loss = torch.tensor(0.0, device=device)
                if feature_augmenter is not None:
                    feat_aug_loss = feature_augmenter(student, ul_imgs)

                loss = (cur_consistency * consistency_loss +
                        0.5 * cur_consistency * pl_loss +
                        0.1 * feat_aug_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update EMA teacher
                with torch.no_grad():
                    for tp, sp in zip(teacher.parameters(),
                                      student.parameters()):
                        tp.data.mul_(ema_decay).add_(sp.data, alpha=1 - ema_decay)

        scheduler.step()

    student.eval()
    return student


# ================================================================
# Calibration methods (agent-selected)
# ================================================================
#
# When the agent decides "calibrate" instead of "train", it selects
# from a spectrum of calibration methods based on analysis signals:
#
# 1. threshold   — shift per-class decision boundaries (0 params)
#    Best when: halluc < 30%, model predictions mostly right
#    Fails when: features fundamentally wrong (high halluc)
#
# 2. output_ft   — retrain only the output conv layer (~100-500 params)
#    Best when: halluc > 30%, features partially discriminative
#    Key insight: intermediate features may encode useful structure
#    even when the final class mapping is wrong. Relearning just
#    the mapping avoids overfitting while fixing the classification.
#
# 3. bn_output_ft — BN recalibration + output layer FT
#    Best when: cross-modality shift where both statistics AND
#    classification need updating, but not enough data for decoder FT
#
# Agent selection logic (in select_strategy):
#   halluc > 50% + GPU  → output_ft (features wrong, must remap)
#   halluc > 50% no GPU → threshold (best we can do)
#   halluc 30-50% + GPU → output_ft (moderate corruption)
#   halluc < 30%        → threshold (predictions mostly right)
#   cross-modality + GPU → bn_output_ft (stats + mapping both off)
# ================================================================


def threshold_calibration(
    model: nn.Module,
    train_images: List[np.ndarray],
    train_labels: List[np.ndarray],
    device: str,
    num_classes: int,
) -> nn.Module:
    """Per-class threshold calibration using labeled data.

    Finds the optimal probability threshold per class that maximizes
    Dice on the labeled cases, then shifts the model's decision
    boundaries accordingly. No parameters are updated.

    Works when: model predictions are approximately correct but
    miscalibrated (e.g., correct regions but wrong confidence).
    Fails when: features are fundamentally wrong (high hallucination).
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()

    # Collect predictions on all labeled slices
    all_probs = []
    all_gts = []
    with torch.no_grad():
        for img, lbl in zip(train_images, train_labels):
            if img.ndim == 3 and img.shape[2] == 3:
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            else:
                t = torch.from_numpy(img).unsqueeze(0)
            t = t.float().to(device)
            probs = F.softmax(model(t), dim=1)  # (1, C, H, W)
            all_probs.append(probs.cpu())
            all_gts.append(torch.from_numpy(lbl).long())

    all_probs = torch.cat(all_probs, dim=0)  # (N, C, H, W)
    all_gts = torch.stack(all_gts, dim=0)     # (N, H, W)

    # Find optimal threshold per class
    best_thresholds = {}
    for c in range(1, num_classes):
        gt_c = (all_gts == c).float()
        if gt_c.sum() == 0:
            best_thresholds[c] = 0.5
            continue

        prob_c = all_probs[:, c]  # (N, H, W)
        best_dice = -1
        best_t = 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            pred_c = (prob_c > t).float()
            if pred_c.sum() == 0:
                continue
            dice = float(
                2 * (pred_c * gt_c).sum() /
                (pred_c.sum() + gt_c.sum() + 1e-8))
            if dice > best_dice:
                best_dice = dice
                best_t = t
        best_thresholds[c] = best_t

    # Monkey-patch forward to apply calibrated thresholds
    original_forward = model.forward

    def calibrated_forward(x):
        logits = original_forward(x)
        probs = F.softmax(logits, dim=1)
        adjusted = torch.zeros_like(probs[:, 0:1])
        for c in range(1, num_classes):
            t = best_thresholds.get(c, 0.5)
            shift = 0.5 - t
            adjusted = torch.cat([adjusted, probs[:, c:c+1] + shift], dim=1)
        fg_sum = adjusted[:, 1:].clamp(0, 1).sum(dim=1, keepdim=True)
        adjusted[:, 0:1] = 1.0 - fg_sum
        return torch.log(adjusted.clamp(min=1e-7))

    model.forward = calibrated_forward
    return model


def output_layer_finetune(
    model: nn.Module,
    train_images: List[np.ndarray],
    train_labels: List[np.ndarray],
    device: str,
    num_classes: int,
    epochs: int = 30,
    lr: float = 1e-3,
) -> nn.Module:
    """Ultra-lightweight fine-tuning of only the output layer.

    When hallucination is high, the model's learned features are wrong
    for the target domain — no threshold shift can fix that. But the
    intermediate features may still be discriminative (encoding edges,
    textures, spatial structure). By retraining only the output conv
    (~100-500 parameters), we learn a new feature→class mapping
    without overfitting.

    This is the sweet spot for N=1-2 with high hallucination:
    - Full training risks overfitting on so few samples
    - Threshold calibration can't fix wrong features
    - Output-layer FT only updates the linear mapping
    """
    student = copy.deepcopy(model)
    student.to(device)
    student.train()

    # Freeze everything
    for p in student.parameters():
        p.requires_grad = False

    # Unfreeze only output conv layer
    trainable = 0
    for name, param in student.named_parameters():
        if (any(k in name for k in ["out_conv", "final", "seg_conv", "seg_output"])
                or ".out." in name):
            param.requires_grad = True
            trainable += param.numel()

    # Fallback: find last conv layer producing num_classes channels
    if trainable == 0:
        for name, param in reversed(list(student.named_parameters())):
            if "weight" in name and param.dim() >= 2:
                if param.shape[0] == num_classes:
                    param.requires_grad = True
                    trainable += param.numel()
                    # Also unfreeze matching bias
                    bias_name = name.replace("weight", "bias")
                    for n2, p2 in student.named_parameters():
                        if n2 == bias_name:
                            p2.requires_grad = True
                            trainable += p2.numel()
                    break

    if trainable == 0:
        print(f"    [output_ft] WARNING: no output layer found, "
              f"falling back to threshold calibration")
        student.eval()
        return threshold_calibration(
            model, train_images, train_labels, device, num_classes)

    print(f"    [output_ft] Training {trainable} params "
          f"(output layer only), {epochs} epochs, lr={lr}")

    # Save original output weights for L2 regularization.
    # This prevents the output layer from drifting too far from the
    # adapted model's mapping, which is important when zero-shot
    # adaptation already improved performance.
    orig_params = {n: p.clone().detach()
                   for n, p in student.named_parameters()
                   if p.requires_grad}

    ds = FewShotDataset(train_images, train_labels, augment=True)
    loader = DataLoader(
        ds, batch_size=min(8, len(ds)),
        shuffle=True, num_workers=0, drop_last=False)

    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    for epoch in range(epochs):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = student(imgs)

            ce_loss = F.cross_entropy(logits, lbls, ignore_index=255)
            probs = F.softmax(logits, dim=1)
            dice_loss = 0.0
            n_valid = 0
            for c in range(1, num_classes):
                pred_c = probs[:, c]
                gt_c = (lbls == c).float()
                if gt_c.sum() > 0:
                    dice = (2 * (pred_c * gt_c).sum() + 1) / \
                           (pred_c.sum() + gt_c.sum() + 1)
                    dice_loss += (1 - dice)
                    n_valid += 1
            if n_valid > 0:
                dice_loss = dice_loss / n_valid

            # L2 regularization toward adapted model's weights
            # Prevents destroying good adapted features
            l2_reg = sum(
                ((p - orig_params[n]) ** 2).sum()
                for n, p in student.named_parameters()
                if p.requires_grad and n in orig_params
            )

            loss = ce_loss + dice_loss + 0.1 * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    student.eval()
    return student


def bn_output_finetune(
    model: nn.Module,
    train_images: List[np.ndarray],
    train_labels: List[np.ndarray],
    device: str,
    num_classes: int,
    epochs: int = 20,
    lr: float = 5e-4,
) -> nn.Module:
    """BN recalibration + output layer fine-tuning.

    For cross-modality shifts, both the batch statistics AND the
    output classification are wrong. This method:
    1. Re-estimates BN running stats from labeled target data
    2. Fine-tunes only BN affine params + output conv

    More aggressive than output_ft alone, but still safe for N=1-2
    because BN has very few learnable params (2 per feature channel).
    """
    student = copy.deepcopy(model)
    student.to(device)

    # Step 1: Re-estimate norm running stats from labeled data.
    student.train()
    from medadaptagent.graph.norm_utils import reset_norm_running_stats
    reset_norm_running_stats(student)

    # Forward pass to collect stats
    with torch.no_grad():
        for img in train_images:
            if img.ndim == 3 and img.shape[2] == 3:
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            else:
                t = torch.from_numpy(img).unsqueeze(0)
            student(t.float().to(device))

    # Step 2: Freeze everything, unfreeze BN affine + output layer
    for p in student.parameters():
        p.requires_grad = False

    trainable = 0
    for name, param in student.named_parameters():
        if (any(k in name for k in [
            "bn", "norm", "out_conv", "final", "seg_conv", "seg_output"
        ]) or ".out." in name):
            param.requires_grad = True
            trainable += param.numel()

    # Fallback for output layer
    if not any("out_conv" in n or "final" in n or "seg_conv" in n or "seg_output" in n
               for n, _ in student.named_parameters() if _.requires_grad):
        for name, param in reversed(list(student.named_parameters())):
            if "weight" in name and param.dim() >= 2:
                if param.shape[0] == num_classes:
                    param.requires_grad = True
                    trainable += param.numel()
                    bias_name = name.replace("weight", "bias")
                    for n2, p2 in student.named_parameters():
                        if n2 == bias_name:
                            p2.requires_grad = True
                            trainable += p2.numel()
                    break

    print(f"    [bn_output_ft] Training {trainable} params "
          f"(BN + output), {epochs} epochs, lr={lr}")

    student.train()
    ds = FewShotDataset(train_images, train_labels, augment=True)
    loader = DataLoader(
        ds, batch_size=min(8, len(ds)),
        shuffle=True, num_workers=0, drop_last=False)

    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = student(imgs)

            ce_loss = F.cross_entropy(logits, lbls, ignore_index=255)
            probs = F.softmax(logits, dim=1)
            dice_loss = 0.0
            n_valid = 0
            for c in range(1, num_classes):
                pred_c = probs[:, c]
                gt_c = (lbls == c).float()
                if gt_c.sum() > 0:
                    dice = (2 * (pred_c * gt_c).sum() + 1) / \
                           (pred_c.sum() + gt_c.sum() + 1)
                    dice_loss += (1 - dice)
                    n_valid += 1
            if n_valid > 0:
                dice_loss = dice_loss / n_valid

            loss = ce_loss + dice_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student.eval()
    return student


def calibrate_few_shot(
    model: nn.Module,
    train_images: List[np.ndarray],
    train_labels: List[np.ndarray],
    strategy: Dict,
    device: str,
    num_classes: int,
) -> nn.Module:
    """Dispatcher: agent selects and runs the best calibration method.

    The calibration_method field in strategy was set by select_strategy
    based on halluc_ratio, shift type, GPU availability, etc.
    """
    method = strategy.get("calibration_method", "threshold")

    if method == "identity":
        # Return model unchanged — adapted model is already good,
        # any calibration on tiny samples risks degradation.
        return copy.deepcopy(model)
    elif method == "output_ft":
        return output_layer_finetune(
            model, train_images, train_labels, device, num_classes,
            epochs=strategy.get("calibration_epochs", 30),
            lr=strategy.get("calibration_lr", 1e-3),
        )
    elif method == "bn_output_ft":
        return bn_output_finetune(
            model, train_images, train_labels, device, num_classes,
            epochs=strategy.get("calibration_epochs", 20),
            lr=strategy.get("calibration_lr", 5e-4),
        )
    else:  # "threshold"
        return threshold_calibration(
            model, train_images, train_labels, device, num_classes)


def select_informative_case(
    model: nn.Module,
    target_images: List[np.ndarray],
    volume_groups: Dict[str, List[int]],
    device: str,
    num_classes: int,
) -> Tuple[str, float]:
    """Active selection: pick the most informative volume to annotate.

    Uses prediction entropy as informativeness measure. The volume where
    the model is most uncertain is the most valuable to annotate.

    Returns (volume_id, uncertainty_score)
    """
    model.eval()
    vol_uncertainty = {}

    with torch.no_grad():
        for vid, indices in volume_groups.items():
            uncertainties = []
            for idx in indices:
                img = target_images[idx]
                if img.ndim == 3 and img.shape[2] == 3:
                    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                else:
                    t = torch.from_numpy(img).unsqueeze(0)
                t = t.float().to(device)

                probs = F.softmax(model(t), dim=1)
                # Entropy per pixel, then average
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                uncertainties.append(entropy.mean().item())

            vol_uncertainty[vid] = float(np.mean(uncertainties))

    # Sort by uncertainty (highest = most informative)
    sorted_vols = sorted(vol_uncertainty.items(), key=lambda x: -x[1])
    return sorted_vols[0]


# ================================================================
# Evaluation
# ================================================================

def evaluate_model(
    model: nn.Module,
    images: List[np.ndarray],
    labels: List[np.ndarray],
    device: str,
    num_classes: int,
    class_names: Dict[int, str],
    batch_size: int = 32,
    compute_hd95: bool = False,
) -> Dict:
    """Generic multi-class segmentation evaluation (batched for speed).

    Args:
        compute_hd95: If True, also compute HD95 per class (slower).
    """
    model.eval()
    class_dices = defaultdict(list)
    class_hd95s = defaultdict(list)

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            batch_lbls = labels[i:i + batch_size]

            tensors = []
            for img in batch_imgs:
                if img.ndim == 3 and img.shape[2] == 3:
                    t = torch.from_numpy(img).permute(2, 0, 1)
                else:
                    t = torch.from_numpy(img)
                tensors.append(t.float())
            batch = torch.stack(tensors).to(device)
            preds = model(batch).argmax(dim=1).cpu().numpy()

            for j, lbl in enumerate(batch_lbls):
                pred = preds[j]
                for c in range(1, num_classes):
                    pred_c = (pred == c).astype(float)
                    gt_c = (lbl == c).astype(float)
                    if gt_c.sum() > 0:
                        dice = float(
                            2 * (pred_c * gt_c).sum() /
                            (pred_c.sum() + gt_c.sum() + 1e-8))
                        class_dices[c].append(dice)

                        if compute_hd95:
                            from medadaptagent.evaluation.eval_metrics import hausdorff_distance_95
                            hd = hausdorff_distance_95(pred_c.astype(bool), gt_c.astype(bool))
                            if np.isfinite(hd):
                                class_hd95s[c].append(hd)

    result = {}
    for c in range(1, num_classes):
        name = class_names.get(c, f"class_{c}")
        vals = class_dices[c]
        result[f"{name}_dice"] = float(np.mean(vals)) if vals else 0.0
        if compute_hd95:
            hd_vals = class_hd95s[c]
            result[f"{name}_hd95"] = float(np.mean(hd_vals)) if hd_vals else float('inf')

    dice_vals = [v for k, v in result.items() if k.endswith("_dice") and v > 0]
    result["mean_dice"] = float(np.mean(dice_vals)) if dice_vals else 0.0

    if compute_hd95:
        hd_vals = [v for k, v in result.items() if k.endswith("_hd95") and np.isfinite(v)]
        result["mean_hd95"] = float(np.mean(hd_vals)) if hd_vals else float('inf')

    return result


# ================================================================
# Volume grouping utilities
# ================================================================

def group_by_volume(filenames: List[str]) -> Dict[str, List[int]]:
    """Group slice indices by volume ID from filenames.

    Supports patterns like:
      mr_1001_0030.npz  -> volume 'mr_1001'
      BRATS_001_slice_0005.npz -> volume 'BRATS_001'
      ct_0001_0042.npz -> volume 'ct_0001'
    """
    volumes = defaultdict(list)
    for i, fn in enumerate(filenames):
        # Remove extension
        base = fn.replace(".npz", "")
        # Split and rejoin all but last part (slice index)
        parts = base.split("_")
        if "slice" in parts:
            # BRATS_001_slice_0005 -> volume = BRATS_001
            slice_idx = parts.index("slice")
            vol_id = "_".join(parts[:slice_idx])
        else:
            # mr_1001_0030 -> volume = mr_1001
            vol_id = "_".join(parts[:-1])
        volumes[vol_id].append(i)
    return dict(volumes)


def sample_volumes(
    volume_groups: Dict[str, List[int]],
    n_volumes: int,
    seed: int = 0,
) -> List[int]:
    """Sample n_volumes and return all their slice indices."""
    rng = np.random.RandomState(seed)
    vol_ids = sorted(volume_groups.keys())
    n_volumes = min(n_volumes, len(vol_ids))
    selected = rng.choice(vol_ids, n_volumes, replace=False)
    indices = []
    for vid in selected:
        indices.extend(volume_groups[vid])
    return sorted(indices)


# ================================================================
# FewShotAdapter: main interface
# ================================================================

class FewShotAdapter:
    """Agent-driven few-shot adaptation with scaling law assessment.

    After zero-shot adaptation, this module:
    1. Assesses whether zero-shot performance is sufficient
    2. If not, runs scaling law experiments (Dice vs N annotated volumes)
    3. Fits a log curve to predict performance at arbitrary N
    4. Recommends how many annotations are needed for a target performance
    """

    def __init__(
        self,
        adapted_model: nn.Module,
        analysis: Dict,
        device: str,
        num_classes: int,
        class_names: Dict[int, str],
        source_images: List[np.ndarray] = None,
        source_labels: List[np.ndarray] = None,
        fda_images: List[np.ndarray] = None,
        fda_labels: List[np.ndarray] = None,
        class_descriptions: List[str] = None,
    ):
        self.adapted_model = adapted_model
        self.analysis = analysis
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.source_images = source_images
        self.source_labels = source_labels
        self.fda_images = fda_images
        self.fda_labels = fda_labels
        self.class_descriptions = class_descriptions

        # Analyze model structure once at init — used for all strategy decisions
        self.model_info = analyze_model_structure(adapted_model)
        mi = self.model_info
        print(f"  Model: {mi['total_params']/1e6:.1f}M params, "
              f"{mi['arch_family']}, {mi['norm_type']}, "
              f"BN={mi['has_batchnorm']}")
        for scope_name, count in mi['scope_params'].items():
            print(f"    scope '{scope_name}': {count/1e6:.2f}M params")

    def _run_one_strategy(self, strategy, train_imgs, train_lbls,
                          indices, target_images):
        """Execute a single strategy (calibration or training) and return model."""
        if strategy["mode"] == "calibrate":
            return calibrate_few_shot(
                self.adapted_model,
                train_imgs, train_lbls,
                strategy,
                self.device, self.num_classes,
            )
        else:
            labeled_set = set(indices)
            unlabeled_imgs = [
                target_images[i]
                for i in range(len(target_images))
                if i not in labeled_set
            ]
            return few_shot_fine_tune(
                self.adapted_model, train_imgs, train_lbls,
                strategy, self.device, self.num_classes,
                source_images=self.source_images,
                source_labels=self.source_labels,
                unlabeled_images=unlabeled_imgs,
                fda_images=self.fda_images,
                fda_labels=self.fda_labels,
                class_descriptions=self.class_descriptions,
            )

    def run_scaling_law(
        self,
        target_images: List[np.ndarray],
        target_labels: List[np.ndarray],
        volume_groups: Dict[str, List[int]],
        shot_counts: List[int] = None,
        n_seeds: int = 3,
        zero_shot_dice: float = None,
        source_val_dice: float = None,
    ) -> Dict:
        """Run scaling law: Dice vs N annotated volumes.

        Args:
            target_images: All target images
            target_labels: All target labels (GT)
            volume_groups: {volume_id: [slice_indices]}
            shot_counts: List of N values to test (default: auto from n_volumes)
            n_seeds: Random seeds for averaging
            zero_shot_dice: Zero-shot adapted Dice (for reference)
            source_val_dice: Source domain val Dice (performance ceiling)

        Returns:
            Dict with scaling law results, fitted curve, recommendations
        """
        n_volumes = len(volume_groups)

        if shot_counts is None:
            # Auto-select shot counts based on available volumes
            candidates = [1, 2, 3, 5, 8, 10, 15, 20]
            shot_counts = [n for n in candidates if n <= n_volumes // 2]
            if not shot_counts:
                shot_counts = [1]

        print(f"\n  === Few-Shot Scaling Law ===")
        print(f"  Available: {n_volumes} volumes, "
              f"testing N = {shot_counts}")
        print(f"  Seeds: {n_seeds}, strategies: agent-selected per N")

        results = {}

        # Active selection: which volume is most informative to annotate?
        best_vol, uncertainty = select_informative_case(
            self.adapted_model, target_images, volume_groups,
            self.device, self.num_classes)
        print(f"  Active selection: most informative volume = {best_vol} "
              f"(uncertainty={uncertainty:.4f})")

        for n_shots in shot_counts:
            # Compute val_dice: evaluate adapted model on the labeled
            # cases BEFORE choosing strategy. This is available in
            # practice (we have these labels) and tells the agent how
            # well the current model works on target data.
            probe_indices = sample_volumes(volume_groups, n_shots,
                                           seed=0 * 100 + n_shots)
            probe_imgs = [target_images[i] for i in probe_indices]
            probe_lbls = [target_labels[i] for i in probe_indices]
            probe_eval = evaluate_model(
                self.adapted_model, probe_imgs, probe_lbls,
                self.device, self.num_classes, self.class_names)
            val_dice = probe_eval["mean_dice"]

            strategy = select_strategy(self.analysis, n_shots,
                                       val_dice=val_dice,
                                       model_info=self.model_info)
            print(f"\n  N={n_shots} volumes | val_dice={val_dice:.3f} | "
                  f"{strategy['rationale']}")

            seed_results = []
            for seed in range(n_seeds):
                # Sample N volumes
                indices = sample_volumes(volume_groups, n_shots,
                                         seed=seed * 100 + n_shots)
                train_imgs = [target_images[i] for i in indices]
                train_lbls = [target_labels[i] for i in indices]

                model = self._run_one_strategy(
                    strategy, train_imgs, train_lbls,
                    indices, target_images)

                # Evaluate on ALL target images
                eval_result = evaluate_model(
                    model, target_images, target_labels,
                    self.device, self.num_classes, self.class_names)
                seed_results.append(eval_result)

                print(f"    seed={seed}: mean_dice={eval_result['mean_dice']:.4f}")

                del model
                torch.cuda.empty_cache()

            # Average across seeds
            avg = {}
            for key in seed_results[0]:
                vals = [r[key] for r in seed_results]
                avg[key] = float(np.mean(vals))
                avg[f"{key}_std"] = float(np.std(vals))

            results[n_shots] = {
                "n_slices": len(indices),
                "strategy": strategy,
                "avg": avg,
                "per_seed": seed_results,
            }
            print(f"    avg: {avg['mean_dice']:.4f} "
                  f"(+/-{avg['mean_dice_std']:.4f})")

        # Fit scaling law curve: dice(n) = a + b * log(n + 1)
        curve = self._fit_scaling_curve(results, zero_shot_dice)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            curve, zero_shot_dice, source_val_dice, n_volumes)

        return {
            "shot_counts": shot_counts,
            "results": results,
            "curve": curve,
            "recommendations": recommendations,
            "zero_shot_dice": zero_shot_dice,
            "source_val_dice": source_val_dice,
        }

    def _fit_scaling_curve(
        self, results: Dict, zero_shot_dice: float = None
    ) -> Dict:
        """Fit log curve: dice(n) = a + b * log(n + 1)."""
        ns = []
        dices = []

        # Include zero-shot as n=0
        if zero_shot_dice is not None:
            ns.append(0)
            dices.append(zero_shot_dice)

        for n_shots, data in sorted(results.items()):
            ns.append(n_shots)
            dices.append(data["avg"]["mean_dice"])

        ns = np.array(ns, dtype=float)
        dices = np.array(dices, dtype=float)

        # Fit: dice = a + b * log(n + 1)
        log_n = np.log(ns + 1)
        if len(ns) >= 2:
            A = np.vstack([np.ones_like(log_n), log_n]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, dices, rcond=None)
            a, b = float(coeffs[0]), float(coeffs[1])
        else:
            a = float(dices[0]) if len(dices) > 0 else 0.0
            b = 0.0

        return {
            "a": a,
            "b": b,
            "formula": f"dice(n) = {a:.4f} + {b:.4f} * log(n+1)",
            "data_points": list(zip(ns.tolist(), dices.tolist())),
        }

    def _generate_recommendations(
        self,
        curve: Dict,
        zero_shot_dice: float,
        source_val_dice: float,
        n_volumes: int,
    ) -> Dict:
        """Generate practical recommendations from scaling law."""
        a, b = curve["a"], curve["b"]

        def predict_dice(n):
            return a + b * np.log(n + 1)

        def needed_for_target(target_dice):
            if b <= 0:
                return None
            n = np.exp((target_dice - a) / b) - 1
            return max(1, int(np.ceil(n))) if n > 0 else None

        recs = {
            "zero_shot": zero_shot_dice,
            "predicted_at": {},
            "needed_for": {},
        }

        # Predict at common N values
        for n in [1, 2, 5, 10, 20, 50]:
            if n <= n_volumes:
                recs["predicted_at"][n] = float(predict_dice(n))

        # How many needed for performance targets
        if source_val_dice:
            targets = [0.5, 0.6, 0.7, 0.8, 0.9]
            for t in targets:
                if t > zero_shot_dice:
                    n_needed = needed_for_target(t)
                    if n_needed is not None and n_needed <= n_volumes * 2:
                        recs["needed_for"][f"{t:.0%}"] = n_needed

        # Summary recommendation
        if zero_shot_dice and source_val_dice:
            gap = source_val_dice - zero_shot_dice
            if gap < 0.05:
                recs["summary"] = (
                    f"Zero-shot adaptation achieves {zero_shot_dice:.1%}, "
                    f"close to source performance ({source_val_dice:.1%}). "
                    f"Few-shot annotation is unlikely to help significantly.")
            elif gap < 0.2:
                n5 = predict_dice(5)
                recs["summary"] = (
                    f"Zero-shot: {zero_shot_dice:.1%}. "
                    f"With 5 annotated volumes: ~{n5:.1%}. "
                    f"Moderate annotation effort recommended.")
            else:
                n5 = predict_dice(5)
                n10 = predict_dice(10)
                recs["summary"] = (
                    f"Zero-shot: {zero_shot_dice:.1%} "
                    f"(large gap from source {source_val_dice:.1%}). "
                    f"5 volumes: ~{n5:.1%}, 10 volumes: ~{n10:.1%}. "
                    f"Annotation strongly recommended for this shift type.")

        return recs

    def print_report(self, scaling_results: Dict):
        """Print human-readable scaling law report."""
        print("\n" + "=" * 60)
        print("AGENT 7: Adaptation Sufficiency Assessment")
        print("=" * 60)

        zs = scaling_results.get("zero_shot_dice")
        sv = scaling_results.get("source_val_dice")
        if zs is not None:
            print(f"  Zero-shot adapted Dice: {zs:.1%}")
        if sv is not None:
            print(f"  Source domain val Dice: {sv:.1%}")
            if zs is not None:
                print(f"  Performance gap: {sv - zs:.1%}")

        print(f"\n  Scaling Law: {scaling_results['curve']['formula']}")
        print(f"\n  Few-shot results (Dice vs N annotated volumes):")
        print(f"  {'N':>4s}  {'Slices':>6s}  {'Mean Dice':>10s}  "
              f"{'Std':>6s}  Strategy")
        print(f"  {'----':>4s}  {'------':>6s}  {'----------':>10s}  "
              f"{'------':>6s}  --------")

        if zs is not None:
            print(f"  {'0':>4s}  {'--':>6s}  {zs:>10.4f}  "
                  f"{'--':>6s}  zero-shot TTA")

        for n_shots in scaling_results["shot_counts"]:
            data = scaling_results["results"][n_shots]
            avg = data["avg"]
            strat = data["strategy"]["rationale"].split(" | ")[0]
            print(f"  {n_shots:>4d}  {data['n_slices']:>6d}  "
                  f"{avg['mean_dice']:>10.4f}  "
                  f"{avg['mean_dice_std']:>6.4f}  {strat}")

        recs = scaling_results["recommendations"]
        if recs.get("needed_for"):
            print(f"\n  Estimated volumes needed:")
            for target, n in recs["needed_for"].items():
                print(f"    {target} Dice -> ~{n} volumes")

        if recs.get("summary"):
            print(f"\n  Recommendation: {recs['summary']}")
