"""
Information-theoretic measures for CGFD framework.

Three novel measures for principled domain adaptation diagnosis:
1. CCPED: Class-Conditional Predictive Entropy Decomposition
2. LDDP: Layer-wise Domain Discrepancy Profile
3. ACS: Anatomical Consistency Score

All measures are architecture-agnostic and task-agnostic — they operate on
model predictions and feature maps, parameterized by TaskConfig.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from typing import Dict, List, Optional, Tuple

from medadaptagent.diagnosis.hypothesis import CCPEDResult, LDDPResult, ACSResult


# ================================================================
# Measure 1: Class-Conditional Predictive Entropy Decomposition
# ================================================================

def compute_ccped(
    logits: torch.Tensor,
    class_names: Dict[str, int],
    anatomy_priors: Dict[str, Dict] = None,
    boundary_width: int = 5,
    image_shape: Tuple[int, int] = None,
) -> Dict[str, CCPEDResult]:
    """Compute Class-Conditional Predictive Entropy Decomposition.

    Decomposes per-pixel entropy into interior, boundary, and absent-region
    components for each class. This reveals the *type* of failure per class.

    Args:
        logits: (B, C, H, W) or (C, H, W) raw model outputs.
        class_names: Dict mapping class_name -> label_index (index 0 = background).
        anatomy_priors: Per-class priors with 'area_range' for expected regions.
            Used to define "absent regions" where class is expected but not predicted.
        boundary_width: Pixel width for boundary region (default 5).
        image_shape: Original image shape (H, W) if logits were computed at
            different resolution. If None, uses logits spatial dims.

    Returns:
        Dict mapping class_name -> CCPEDResult.
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    if logits.ndim != 4:
        raise ValueError(f"Expected 3D or 4D logits, got {logits.ndim}D")

    # Average over batch
    probs = F.softmax(logits, dim=1).mean(0)  # (C, H, W)
    log_probs = F.log_softmax(logits, dim=1).mean(0)
    entropy_map = -(probs * log_probs).sum(dim=0)  # (H, W)
    pred_map = probs.argmax(dim=0)  # (H, W)

    entropy_np = entropy_map.cpu().numpy()
    pred_np = pred_map.cpu().numpy()
    h, w = entropy_np.shape

    results = {}

    for class_name, class_idx in class_names.items():
        if class_idx == 0:  # skip background
            continue

        # Binary mask for this class
        class_mask = (pred_np == class_idx)

        # Boundary region: dilate - erode
        if class_mask.any():
            dilated = ndimage.binary_dilation(
                class_mask, iterations=boundary_width)
            eroded = ndimage.binary_erosion(
                class_mask, iterations=boundary_width)
            boundary_mask = dilated & ~eroded
            interior_mask = eroded
        else:
            boundary_mask = np.zeros_like(class_mask)
            interior_mask = np.zeros_like(class_mask)

        # Absent region: where class is expected but not predicted
        # Use anatomy priors to estimate expected region
        absent_mask = np.zeros_like(class_mask)
        if anatomy_priors and class_name in anatomy_priors:
            prior = anatomy_priors[class_name]
            area_range = prior.get("area_range", (0.0, 1.0))
            expected_min_area = area_range[0] * h * w

            # If predicted area is much smaller than expected minimum,
            # the expected region is approximately the center/full image
            predicted_area = class_mask.sum()
            if predicted_area < expected_min_area * 0.5:
                # Use high-entropy regions as "absent" candidates
                # (where model is uncertain but class should be present)
                entropy_threshold = np.percentile(entropy_np, 70)
                absent_mask = (entropy_np > entropy_threshold) & ~class_mask
        elif not class_mask.any():
            # Class completely absent — entire image is "absent region"
            absent_mask = np.ones_like(class_mask)

        # Compute mean entropy in each region
        h_interior = (float(entropy_np[interior_mask].mean())
                      if interior_mask.any() else 0.0)
        h_boundary = (float(entropy_np[boundary_mask].mean())
                      if boundary_mask.any() else 0.0)
        h_absent = (float(entropy_np[absent_mask].mean())
                    if absent_mask.any() else 0.0)

        # Classify failure type
        failure_type = _classify_entropy_pattern(
            h_interior, h_boundary, h_absent,
            n_interior=int(interior_mask.sum()),
            n_boundary=int(boundary_mask.sum()),
            n_class=int(class_mask.sum()),
            total_pixels=h * w,
            anatomy_priors=anatomy_priors.get(class_name, {}) if anatomy_priors else {},
        )

        results[class_name] = CCPEDResult(
            class_name=class_name,
            h_interior=h_interior,
            h_boundary=h_boundary,
            h_absent=h_absent,
            n_interior=int(interior_mask.sum()),
            n_boundary=int(boundary_mask.sum()),
            n_absent=int(absent_mask.sum()),
            failure_type=failure_type,
        )

    return results


def _classify_entropy_pattern(
    h_interior: float, h_boundary: float, h_absent: float,
    n_interior: int, n_boundary: int, n_class: int, total_pixels: int,
    anatomy_priors: Dict = None,
) -> str:
    """Classify failure type from entropy decomposition pattern.

    Returns one of: "miss", "boundary", "confusion", "partial_miss", "healthy".
    """
    area_fraction = n_class / max(total_pixels, 1)
    expected_min = 0.0
    if anatomy_priors:
        area_range = anatomy_priors.get("area_range", (0.0, 1.0))
        expected_min = area_range[0]

    # Complete miss: class not predicted at all (or negligible area)
    if area_fraction < max(expected_min * 0.1, 0.001):
        return "miss"

    # Very small prediction vs expectation -> partial miss
    if expected_min > 0 and area_fraction < expected_min * 0.3:
        return "partial_miss"

    # High entropy thresholds (log(C) normalized)
    HIGH_ENTROPY = 0.5  # relative threshold

    # Both interior and boundary high → confusion (wrong class assigned)
    if h_interior > HIGH_ENTROPY and h_boundary > HIGH_ENTROPY:
        return "confusion"

    # Interior low but boundary high → boundary failure
    if h_interior < HIGH_ENTROPY and h_boundary > HIGH_ENTROPY:
        return "boundary"

    # Interior high but boundary low → interior confusion
    if h_interior > HIGH_ENTROPY and h_boundary < HIGH_ENTROPY:
        return "confusion"

    # All low entropy → healthy prediction
    return "healthy"


def compute_ccped_batch(
    model: nn.Module,
    images: List[np.ndarray],
    class_names: Dict[str, int],
    anatomy_priors: Dict[str, Dict] = None,
    image_size: int = 256,
    boundary_width: int = 5,
    max_images: int = 20,
    preprocess_fn=None,
) -> Dict[str, CCPEDResult]:
    """Compute CCPED averaged over multiple images.

    Args:
        model: Segmentation model (eval mode).
        images: List of (H, W, 3) float32 [0,1] images.
        class_names: Dict class_name -> label_index.
        anatomy_priors: Per-class anatomy priors.
        image_size: Model input size.
        boundary_width: Boundary region width in pixels.
        max_images: Max number of images to process.
        preprocess_fn: Optional function(image) -> tensor (B, C, H, W).

    Returns:
        Aggregated CCPEDResult per class.
    """
    from PIL import Image as PILImage

    model.eval()
    all_results = {name: [] for name in class_names if class_names[name] > 0}

    subset = images[:max_images]
    device = next(model.parameters()).device

    with torch.no_grad():
        for img in subset:
            if preprocess_fn is not None:
                tensor = preprocess_fn(img).to(device)
            else:
                pil = PILImage.fromarray((img * 255).astype(np.uint8))
                pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
                tensor = torch.from_numpy(
                    np.array(pil).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)

            logits = model(tensor)
            per_class = compute_ccped(
                logits.cpu(), class_names, anatomy_priors, boundary_width)

            for name, result in per_class.items():
                all_results[name].append(result)

    # Average across images
    aggregated = {}
    for name, results_list in all_results.items():
        if not results_list:
            continue
        aggregated[name] = CCPEDResult(
            class_name=name,
            h_interior=float(np.mean([r.h_interior for r in results_list])),
            h_boundary=float(np.mean([r.h_boundary for r in results_list])),
            h_absent=float(np.mean([r.h_absent for r in results_list])),
            n_interior=int(np.mean([r.n_interior for r in results_list])),
            n_boundary=int(np.mean([r.n_boundary for r in results_list])),
            n_absent=int(np.mean([r.n_absent for r in results_list])),
            # Most common failure type
            failure_type=max(
                set(r.failure_type for r in results_list),
                key=lambda t: sum(1 for r in results_list if r.failure_type == t),
            ),
        )

    return aggregated


# ================================================================
# Measure 2: Layer-wise Domain Discrepancy Profile (LDDP)
# ================================================================

def compute_lddp(
    model: nn.Module,
    source_images: List[np.ndarray],
    target_images: List[np.ndarray],
    image_size: int = 256,
    max_images: int = 20,
    n_spatial_samples: int = 256,
    preprocess_fn=None,
) -> LDDPResult:
    """Compute Layer-wise Domain Discrepancy Profile.

    Registers hooks at each major stage of the network, passes source and
    target images, computes MMD² between feature distributions at each layer.

    Args:
        model: Segmentation model.
        source_images: List of source domain images.
        target_images: List of target domain images.
        image_size: Model input size.
        max_images: Max images per domain.
        n_spatial_samples: Number of spatial locations to sample per feature map.
        preprocess_fn: Optional function(image) -> tensor.

    Returns:
        LDDPResult with per-layer MMD values and profile interpretation.
    """
    from PIL import Image as PILImage

    model.eval()
    device = next(model.parameters()).device

    # Identify hookable layers
    hook_layers = _find_hook_layers(model)
    if not hook_layers:
        return LDDPResult(
            layer_names=["no_hooks"],
            mmd_values=[0.0],
            profile_type="unknown",
        )

    # Collect features
    source_features = _collect_features(
        model, source_images[:max_images], hook_layers, device,
        image_size, preprocess_fn)
    target_features = _collect_features(
        model, target_images[:max_images], hook_layers, device,
        image_size, preprocess_fn)

    # Compute MMD at each layer
    # When using __encoder_features__, features have their own keys (e1, e2, etc.)
    # rather than matching hook_layers names
    feature_keys = sorted(
        set(source_features.keys()) & set(target_features.keys()),
        key=lambda k: list(source_features.keys()).index(k)
    )
    layer_names = []
    mmd_values = []

    for layer_name in feature_keys:
        src_feat = source_features.get(layer_name)
        tgt_feat = target_features.get(layer_name)
        if src_feat is None or tgt_feat is None:
            continue

        # Spatial sampling: flatten spatial dims, sample random locations
        src_flat = _flatten_and_sample(src_feat, n_spatial_samples)
        tgt_flat = _flatten_and_sample(tgt_feat, n_spatial_samples)

        mmd = _compute_mmd(src_flat, tgt_flat)
        layer_names.append(layer_name)
        mmd_values.append(float(mmd))

    if not mmd_values:
        return LDDPResult(
            layer_names=["empty"],
            mmd_values=[0.0],
            profile_type="unknown",
        )

    # Normalize to [0, 1]
    max_mmd = max(mmd_values) if max(mmd_values) > 0 else 1.0
    mmd_normalized = [v / max_mmd for v in mmd_values]

    # Interpret profile
    peak_idx = int(np.argmax(mmd_values))
    peak_layer = layer_names[peak_idx]
    profile_type = _interpret_profile(mmd_normalized)
    adaptation_depth = _recommend_depth(mmd_normalized, layer_names)

    return LDDPResult(
        layer_names=layer_names,
        mmd_values=mmd_values,
        peak_layer=peak_layer,
        profile_type=profile_type,
        recommended_adaptation_depth=adaptation_depth,
    )


def _find_hook_layers(model: nn.Module) -> List[str]:
    """Find major stages/blocks to hook for feature extraction.

    Architecture-agnostic: works with UNet, ViT, Swin UNETR, nnUNet.
    """
    hook_layers = []

    # Strategy 1: Look for get_encoder_features method
    if hasattr(model, 'get_encoder_features'):
        # Model provides its own feature extraction
        return ["__encoder_features__"]

    # Strategy 2: Named children at top level (encoder blocks, etc.)
    named_children = dict(model.named_children())

    # UNet-style: enc1, enc2, enc3, enc4, bottleneck
    for name in ["enc1", "enc2", "enc3", "enc4", "bottleneck"]:
        if name in named_children:
            hook_layers.append(name)

    # nnUNet / STU-Net style: encoder_blocks
    if "model" in named_children:
        inner = named_children["model"]
        inner_children = dict(inner.named_children())
        if "encoder_blocks" in inner_children:
            enc_blocks = inner_children["encoder_blocks"]
            for i, _ in enumerate(enc_blocks):
                hook_layers.append(f"model.encoder_blocks.{i}")

    # Swin UNETR: swinViT.layers
    if hasattr(model, 'swinViT'):
        for i, _ in enumerate(model.swinViT.layers):
            hook_layers.append(f"swinViT.layers.{i}")

    # Fallback: use sequential children
    if not hook_layers:
        for name, child in model.named_children():
            if any(kw in name.lower() for kw in
                   ["enc", "down", "layer", "block", "stage", "conv"]):
                hook_layers.append(name)

    # Limit to reasonable number
    return hook_layers[:8]


def _collect_features(
    model: nn.Module,
    images: List[np.ndarray],
    hook_layers: List[str],
    device: torch.device,
    image_size: int = 256,
    preprocess_fn=None,
) -> Dict[str, torch.Tensor]:
    """Collect features from specified layers using forward hooks."""
    from PIL import Image as PILImage

    # Special case: model provides get_encoder_features
    if hook_layers == ["__encoder_features__"]:
        return _collect_via_method(model, images, device, image_size, preprocess_fn)

    # Register hooks
    feature_store = {}
    handles = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                feature_store[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                feature_store[name] = output[0].detach().cpu()
        return hook_fn

    for layer_name in hook_layers:
        # Navigate to the module
        module = model
        try:
            for part in layer_name.split("."):
                if part.isdigit():
                    module = list(module.children())[int(part)]
                else:
                    module = getattr(module, part)
            handle = module.register_forward_hook(make_hook(layer_name))
            handles.append(handle)
        except (AttributeError, IndexError):
            continue

    # Forward pass
    all_features = {name: [] for name in hook_layers}

    with torch.no_grad():
        for img in images:
            if preprocess_fn is not None:
                tensor = preprocess_fn(img).to(device)
            else:
                pil = PILImage.fromarray((img * 255).astype(np.uint8))
                pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
                tensor = torch.from_numpy(
                    np.array(pil).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)

            feature_store.clear()
            _ = model(tensor)

            for name in hook_layers:
                if name in feature_store:
                    all_features[name].append(feature_store[name])

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Stack features per layer
    result = {}
    for name, feats in all_features.items():
        if feats:
            result[name] = torch.cat(feats, dim=0)  # (N, C, H, W)

    return result


def _collect_via_method(
    model: nn.Module,
    images: List[np.ndarray],
    device: torch.device,
    image_size: int = 256,
    preprocess_fn=None,
) -> Dict[str, torch.Tensor]:
    """Collect features via model.get_encoder_features()."""
    from PIL import Image as PILImage

    all_features = {}

    with torch.no_grad():
        for img in images:
            if preprocess_fn is not None:
                tensor = preprocess_fn(img).to(device)
            else:
                pil = PILImage.fromarray((img * 255).astype(np.uint8))
                pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
                tensor = torch.from_numpy(
                    np.array(pil).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)

            features = model.get_encoder_features(tensor)
            for name, feat in features.items():
                if name not in all_features:
                    all_features[name] = []
                all_features[name].append(feat.detach().cpu())

    return {name: torch.cat(feats, dim=0) for name, feats in all_features.items()}


def _flatten_and_sample(features: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Flatten spatial dims and randomly sample feature vectors.

    Args:
        features: (N, C, ...) feature tensor with spatial dims.
        n_samples: Number of feature vectors to sample.

    Returns:
        (n_samples, C) sampled feature vectors.
    """
    n, c = features.shape[:2]
    spatial = features.reshape(n, c, -1)  # (N, C, S)
    spatial = spatial.permute(0, 2, 1).reshape(-1, c)  # (N*S, C)

    if spatial.shape[0] > n_samples:
        indices = torch.randperm(spatial.shape[0])[:n_samples]
        spatial = spatial[indices]

    return spatial


def _compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = None) -> float:
    """Compute Maximum Mean Discrepancy with Gaussian RBF kernel.

    MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

    Args:
        x: (N, D) samples from distribution P.
        y: (M, D) samples from distribution Q.
        sigma: RBF bandwidth. If None, uses median heuristic.

    Returns:
        MMD² value (non-negative float).
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0

    # Normalize features for numerical stability
    x = F.normalize(x.float(), dim=1)
    y = F.normalize(y.float(), dim=1)

    # Median heuristic for sigma
    if sigma is None:
        with torch.no_grad():
            # Sample subset for bandwidth estimation
            n_sub = min(200, x.shape[0], y.shape[0])
            x_sub = x[:n_sub]
            y_sub = y[:n_sub]
            dists = torch.cdist(x_sub, y_sub)
            sigma = float(dists.median()) + 1e-8

    def rbf_kernel(a, b):
        dists = torch.cdist(a, b)
        return torch.exp(-dists ** 2 / (2 * sigma ** 2))

    # Subsample for efficiency
    max_n = 500
    if x.shape[0] > max_n:
        x = x[torch.randperm(x.shape[0])[:max_n]]
    if y.shape[0] > max_n:
        y = y[torch.randperm(y.shape[0])[:max_n]]

    kxx = rbf_kernel(x, x)
    kyy = rbf_kernel(y, y)
    kxy = rbf_kernel(x, y)

    # Exclude diagonal (unbiased estimator)
    n = kxx.shape[0]
    m = kyy.shape[0]
    kxx.fill_diagonal_(0)
    kyy.fill_diagonal_(0)

    mmd2 = (kxx.sum() / max(n * (n - 1), 1)
            + kyy.sum() / max(m * (m - 1), 1)
            - 2 * kxy.mean())

    return max(float(mmd2), 0.0)


def _interpret_profile(mmd_normalized: List[float]) -> str:
    """Interpret the LDDP profile shape.

    Returns: "early_shift", "deep_shift", "uniform_shift", or "minimal_shift".
    """
    n = len(mmd_normalized)
    if n <= 1:
        return "minimal_shift"

    mid = n // 2
    early_mean = np.mean(mmd_normalized[:mid])
    deep_mean = np.mean(mmd_normalized[mid:])
    overall_mean = np.mean(mmd_normalized)

    if overall_mean < 0.2:
        return "minimal_shift"
    if early_mean > deep_mean * 1.5:
        return "early_shift"
    if deep_mean > early_mean * 1.5:
        return "deep_shift"
    return "uniform_shift"


def _recommend_depth(mmd_normalized: List[float], layer_names: List[str]) -> int:
    """Recommend how deep to adapt based on LDDP profile.

    Returns layer index up to which adaptation is recommended.
    """
    # Find first layer where MMD drops below threshold
    threshold = 0.3
    for i in range(len(mmd_normalized) - 1, -1, -1):
        if mmd_normalized[i] > threshold:
            return i
    return 0


# ================================================================
# Measure 3: Anatomical Consistency Score (ACS)
# ================================================================

def compute_acs(
    predictions: List[np.ndarray],
    class_names: Dict[str, int],
    containment_rules: List[Tuple[str, str, str]],
    anatomy_priors: Dict[str, Dict] = None,
) -> ACSResult:
    """Compute Anatomical Consistency Score.

    Evaluates prediction quality WITHOUT ground truth by checking whether
    predicted structures satisfy known anatomical constraints.

    Args:
        predictions: List of (H, W) predicted label maps.
        class_names: Dict class_name -> label_index.
        containment_rules: List of (A, relation, B) anatomical rules.
            Relations: "contains" (A should contain B), "adjacent", "disjoint".
        anatomy_priors: Per-class shape/size priors.

    Returns:
        ACSResult with total score, per-rule scores, and violations.
    """
    if not predictions:
        return ACSResult(total_score=0.0)

    rule_scores = {}
    class_scores = {}
    violations = []

    # Evaluate containment rules
    for outer, relation, inner in containment_rules:
        outer_idx = class_names.get(outer, -1)
        inner_idx = class_names.get(inner, -1)
        if outer_idx < 0 or inner_idx < 0:
            continue

        rule_name = f"{outer}_{relation}_{inner}"
        scores_per_image = []

        for pred in predictions:
            outer_mask = (pred == outer_idx)
            inner_mask = (pred == inner_idx)

            if relation == "contains":
                score = _containment_score(outer_mask, inner_mask)
            elif relation == "adjacent":
                score = _adjacency_score(outer_mask, inner_mask)
            elif relation == "disjoint":
                score = _disjointness_score(outer_mask, inner_mask)
            else:
                score = 1.0

            scores_per_image.append(score)

        mean_score = float(np.mean(scores_per_image))
        rule_scores[rule_name] = mean_score

        if mean_score < 0.5:
            violations.append(
                f"Rule '{outer} {relation} {inner}' violated "
                f"(score={mean_score:.3f})"
            )

    # Evaluate per-class shape/size consistency
    if anatomy_priors:
        for class_name, prior in anatomy_priors.items():
            class_idx = class_names.get(class_name, -1)
            if class_idx <= 0:
                continue

            class_score = _class_consistency_score(
                predictions, class_idx, prior)
            class_scores[class_name] = class_score

            if class_score < 0.3:
                violations.append(
                    f"Class '{class_name}' shape/size inconsistent "
                    f"(score={class_score:.3f})"
                )

    # Composite score
    all_scores = list(rule_scores.values()) + list(class_scores.values())
    total_score = float(np.mean(all_scores)) if all_scores else 0.5

    return ACSResult(
        total_score=total_score,
        per_rule_scores=rule_scores,
        per_class_scores=class_scores,
        violations=violations,
    )


def _containment_score(outer_mask: np.ndarray, inner_mask: np.ndarray) -> float:
    """Score how well outer contains inner (1.0 = perfect containment).

    Components:
    - Containment ratio: fraction of inner pixels inside outer
    - Size ratio: inner should be smaller than outer
    - Centroid alignment: inner centroid should be near outer centroid
    """
    if not inner_mask.any():
        # Inner class absent — can't evaluate containment
        # If outer is also absent, neutral; if outer present, partial penalty
        return 0.5 if outer_mask.any() else 0.5

    if not outer_mask.any():
        # Outer absent but inner present — strong violation
        return 0.0

    # 1. Containment ratio: what fraction of inner pixels are spatially
    # within the outer region. In label maps, inner replaces outer pixels,
    # so we check if inner is adjacent to / enclosed by the outer mask.
    inner_pixels = inner_mask.sum()
    effective_outer = outer_mask | inner_mask  # the full combined region

    # Check spatial containment: dilate outer by 1px and check inner is inside
    from scipy import ndimage as _ndi
    outer_dilated = _ndi.binary_dilation(outer_mask, iterations=2)
    # Inner pixels should be near or within the outer mask region
    inner_near_outer = (inner_mask & outer_dilated).sum()
    containment_ratio = inner_near_outer / max(inner_pixels, 1)

    # 2. Size ratio: inner should be meaningfully smaller
    outer_pixels = effective_outer.sum()
    size_ratio = inner_pixels / max(outer_pixels, 1)
    # Score: 1.0 if ratio is reasonable (0.1-0.8), penalty for extremes
    if size_ratio > 0.95:
        size_score = 0.3  # inner almost same size as outer
    elif size_ratio < 0.01:
        size_score = 0.5  # inner negligibly small
    else:
        size_score = 1.0

    # 3. Centroid alignment
    inner_coords = np.argwhere(inner_mask)
    outer_coords = np.argwhere(effective_outer)
    inner_centroid = inner_coords.mean(axis=0)
    outer_centroid = outer_coords.mean(axis=0)

    # Normalize distance by outer region size
    h, w = outer_mask.shape
    max_dist = np.sqrt(h**2 + w**2) / 2
    dist = np.linalg.norm(inner_centroid - outer_centroid)
    centroid_score = max(0, 1 - dist / max(max_dist * 0.3, 1))

    # Weighted combination
    return 0.5 * containment_ratio + 0.25 * size_score + 0.25 * centroid_score


def _adjacency_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Score whether two structures are adjacent (touching)."""
    if not mask_a.any() or not mask_b.any():
        return 0.5  # can't evaluate

    # Dilate each mask by 1 pixel and check overlap
    a_dilated = ndimage.binary_dilation(mask_a, iterations=2)
    b_dilated = ndimage.binary_dilation(mask_b, iterations=2)

    touching = (a_dilated & mask_b).any() or (mask_a & b_dilated).any()
    return 1.0 if touching else 0.0


def _disjointness_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Score whether two structures are disjoint (non-overlapping)."""
    if not mask_a.any() or not mask_b.any():
        return 1.0

    overlap = (mask_a & mask_b).sum()
    total = min(mask_a.sum(), mask_b.sum())
    overlap_ratio = overlap / max(total, 1)
    return max(0, 1 - overlap_ratio)


def _class_consistency_score(
    predictions: List[np.ndarray],
    class_idx: int,
    prior: Dict,
) -> float:
    """Score per-class shape/size consistency against anatomy priors.

    Checks:
    1. Presence: Is the class predicted in most images?
    2. Area: Is the area within expected range?
    3. Compactness: Is the shape reasonable (not fragmented)?
    4. Connectivity: Is it a single connected component?
    """
    scores = []
    area_range = prior.get("area_range", (0.0, 1.0))

    for pred in predictions:
        class_mask = (pred == class_idx)
        h, w = pred.shape
        area_fraction = class_mask.sum() / (h * w)

        # Presence score
        if not class_mask.any():
            scores.append(0.0)
            continue

        score_components = []

        # Area score: 1.0 if within range, decreasing outside
        if area_range[0] <= area_fraction <= area_range[1]:
            score_components.append(1.0)
        elif area_fraction < area_range[0]:
            ratio = area_fraction / max(area_range[0], 1e-6)
            score_components.append(min(ratio, 1.0))
        else:
            ratio = area_range[1] / max(area_fraction, 1e-6)
            score_components.append(min(ratio, 1.0))

        # Connectivity: prefer single connected component
        labeled, n_components = ndimage.label(class_mask)
        if n_components <= 1:
            score_components.append(1.0)
        elif n_components <= 3:
            score_components.append(0.7)
        else:
            score_components.append(0.3)

        # Compactness: 4π·area / perimeter² (circle = 1.0)
        perimeter = _compute_perimeter(class_mask)
        area = class_mask.sum()
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            compactness = min(compactness, 1.0)
            score_components.append(compactness)

        scores.append(float(np.mean(score_components)))

    return float(np.mean(scores)) if scores else 0.0


def _compute_perimeter(mask: np.ndarray) -> float:
    """Compute perimeter of a binary mask."""
    # Perimeter = sum of boundary pixels
    eroded = ndimage.binary_erosion(mask)
    boundary = mask & ~eroded
    return float(boundary.sum())
