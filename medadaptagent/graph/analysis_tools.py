"""
Analysis tools for the LLM agent to call selectively.

Instead of computing ALL statistics, the KnowledgeAgent recommends
which analyses are relevant, and only those are computed.

Available tools:
  - intensity_analysis: KL divergence, per-channel histograms
  - frequency_analysis: FFT low-freq (style) and high-freq (texture) energy
  - scale_analysis: foreground area ratio comparison
  - confidence_analysis: model confidence/entropy on target predictions
  - hallucination_detection: TTA-based hallucination ratio
  - bn_statistics_analysis: BN running mean/var shift magnitude
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from scipy import stats as scipy_stats


def intensity_analysis(source_images: list, target_images: list,
                       max_samples: int = 200) -> Dict:
    """Compute intensity distribution statistics between domains."""
    src = source_images[:max_samples]
    tgt = target_images[:max_samples]

    src_flat = np.concatenate([img.flatten() for img in src])
    tgt_flat = np.concatenate([img.flatten() for img in tgt])

    # Overall KL divergence
    bins = np.linspace(0, 1, 51)
    src_hist, _ = np.histogram(src_flat, bins=bins, density=True)
    tgt_hist, _ = np.histogram(tgt_flat, bins=bins, density=True)
    src_hist = src_hist / (src_hist.sum() + 1e-10)
    tgt_hist = tgt_hist / (tgt_hist.sum() + 1e-10)
    kl = float(np.sum(tgt_hist * np.log((tgt_hist + 1e-10) / (src_hist + 1e-10))))

    # Per-channel stats
    channel_shifts = {}
    for c in range(min(3, src[0].shape[-1] if src[0].ndim == 3 else 1)):
        src_c = np.array([img[..., c].mean() if img.ndim == 3 else img.mean()
                          for img in src])
        tgt_c = np.array([img[..., c].mean() if img.ndim == 3 else img.mean()
                          for img in tgt])
        channel_shifts[f"channel_{c}"] = float(tgt_c.mean() - src_c.mean())

    return {
        "tool": "intensity_analysis",
        "kl_divergence": kl,
        "source_mean": float(src_flat.mean()),
        "target_mean": float(tgt_flat.mean()),
        "mean_shift": float(tgt_flat.mean() - src_flat.mean()),
        "channel_shifts": channel_shifts,
    }


def frequency_analysis(source_images: list, target_images: list,
                        max_samples: int = 100) -> Dict:
    """Compute frequency spectrum shift (style vs texture)."""
    def fft_energy(images, low_ratio=0.25):
        energies_low, energies_high = [], []
        for img in images[:max_samples]:
            if img.ndim == 3:
                gray = img.mean(axis=-1)
            else:
                gray = img
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            h, w = gray.shape
            ch, cw = h // 2, w // 2
            r = int(min(h, w) * low_ratio / 2)
            mask_low = np.zeros((h, w), dtype=bool)
            Y, X = np.ogrid[:h, :w]
            mask_low[(Y - ch)**2 + (X - cw)**2 <= r**2] = True
            mag = np.abs(fft_shift)
            energies_low.append(mag[mask_low].mean())
            energies_high.append(mag[~mask_low].mean())
        return np.mean(energies_low), np.mean(energies_high)

    src_low, src_high = fft_energy(source_images)
    tgt_low, tgt_high = fft_energy(target_images)

    style_shift = abs(tgt_low - src_low) / (src_low + 1e-10)
    texture_shift = abs(tgt_high - src_high) / (src_high + 1e-10)

    return {
        "tool": "frequency_analysis",
        "style_shift": float(style_shift),
        "texture_shift": float(texture_shift),
        "interpretation": (
            "high style_shift → low-frequency appearance differs (FDA can help). "
            "high texture_shift → fine detail differs (harder to fix)."
        ),
    }


def scale_analysis(source_images: list, target_images: list,
                    model: nn.Module, device: str,
                    max_samples: int = 100) -> Dict:
    """Compare foreground scale between source and target predictions."""
    model.eval()
    src_fg, tgt_fg = [], []

    with torch.no_grad():
        for imgs, ratios in [(source_images, src_fg), (target_images, tgt_fg)]:
            for img in imgs[:max_samples]:
                if img.ndim == 3 and img.shape[2] in (1, 3):
                    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                elif img.ndim == 2:
                    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
                else:
                    t = torch.from_numpy(img.astype(np.float32))
                pred = model(t.unsqueeze(0).to(device)).argmax(dim=1).cpu().numpy()[0]
                ratios.append((pred > 0).mean())

    src_mean = np.mean(src_fg) if src_fg else 0
    tgt_mean = np.mean(tgt_fg) if tgt_fg else 0

    return {
        "tool": "scale_analysis",
        "source_fg_ratio": float(src_mean),
        "target_fg_ratio": float(tgt_mean),
        "scale_ratio": float(tgt_mean / (src_mean + 1e-10)),
    }


def confidence_analysis(target_images: list, model: nn.Module,
                         device: str, max_samples: int = 200,
                         task=None) -> Dict:
    """Analyze model confidence and entropy on target data. If a task
    adapter is provided, defer tensor prep + forward to it (handles 3D,
    sliding-window inference for seg_3d, etc.); else fall back to the
    legacy 2D channels-last image path."""
    model.eval()
    confidences, entropies, fg_ratios = [], [], []

    with torch.no_grad():
        for img in target_images[:max_samples]:
            if task is not None:
                t = task.to_tensor(img, device)
                logits, _ = task.forward_pred(model, t)
                if logits.ndim > 2:
                    # seg_3d logits arrive as [B, C, D, H, W]; fg = fg-vox %
                    probs = F.softmax(logits, dim=1)
                else:
                    # cls_2d / cls_3d logits are [B, C]
                    probs = F.softmax(logits, dim=1)
            else:
                if img.ndim == 3 and img.shape[2] in (1, 3):
                    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                elif img.ndim == 2:
                    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
                else:
                    t = torch.from_numpy(img.astype(np.float32))
                logits = model(t.unsqueeze(0).to(device))
                probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1)[0].mean().item()
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
            fg = (logits.argmax(dim=1) > 0).float().mean().item()
            confidences.append(conf)
            entropies.append(ent)
            fg_ratios.append(fg)

    return {
        "tool": "confidence_analysis",
        "mean_confidence": float(np.mean(confidences)),
        "mean_entropy": float(np.mean(entropies)),
        "fg_ratio": float(np.mean(fg_ratios)),
        "interpretation": (
            f"conf={np.mean(confidences):.3f}: "
            f"{'model is confident (may be confidently wrong)' if np.mean(confidences) > 0.95 else 'model is uncertain (predictions unreliable)'}. "
            f"ent={np.mean(entropies):.3f}: "
            f"{'low entropy — sharp predictions' if np.mean(entropies) < 0.05 else 'high entropy — model confused'}. "
            f"fg={np.mean(fg_ratios):.1%}: predicted foreground fraction."
        ),
    }


def hallucination_detection(target_images: list, model: nn.Module,
                             device: str, max_samples: int = 100,
                             task=None) -> Dict:
    """Detect hallucination using TTA consistency.

    Compare predictions on original vs spatially-flipped input. Stable
    predictions + non-trivial foreground = potential hallucination
    (model is confidently predicting structures that aren't there).
    Inconsistent predictions = ordinary uncertainty.

    Task-aware: when a task adapter is supplied, defers tensor prep +
    forward to it (handles 3D volumes / sliding-window inference); else
    falls back to the legacy 2D channels-last image path."""
    model.eval()
    flip_consistencies = []
    fg_ratios = []

    with torch.no_grad():
        for img in target_images[:max_samples]:
            try:
                if task is not None:
                    tensor = task.to_tensor(img, device)
                    logits1, _ = task.forward_pred(model, tensor)
                    pred1 = logits1.argmax(dim=1)
                    flipped_input = torch.flip(tensor, [-1])
                    logits2, _ = task.forward_pred(model, flipped_input)
                    pred2 = logits2.argmax(dim=1)
                    if pred2.shape == pred1.shape:
                        pred2 = torch.flip(pred2, [-1])
                else:
                    if img.ndim == 3 and img.shape[2] in (1, 3):
                        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                    elif img.ndim == 2:
                        t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
                    else:
                        t = torch.from_numpy(img.astype(np.float32))
                    tensor = t.unsqueeze(0).to(device)
                    pred1 = model(tensor).argmax(dim=1)
                    flipped_input = torch.flip(tensor, [-1])
                    pred2 = model(flipped_input).argmax(dim=1)
                    pred2 = torch.flip(pred2, [-1])
                agreement = (pred1 == pred2).float().mean().item()
                fg = (pred1 > 0).float().mean().item()
                flip_consistencies.append(agreement)
                fg_ratios.append(fg)
            except Exception:
                continue  # skip problematic images

    mean_consistency = np.mean(flip_consistencies)
    mean_fg = np.mean(fg_ratios)

    # Hallucination = high confidence + high consistency + unexpected foreground
    if mean_consistency > 0.95 and mean_fg > 0.1:
        halluc_ratio = min(1.0, mean_fg * 2)  # heuristic
    else:
        halluc_ratio = max(0.0, 1.0 - mean_consistency)

    return {
        "tool": "hallucination_detection",
        "flip_consistency": float(mean_consistency),
        "mean_fg_ratio": float(mean_fg),
        "halluc_ratio": float(halluc_ratio),
        "interpretation": (
            f"consistency={mean_consistency:.2f}, fg={mean_fg:.1%}: "
            f"{'HIGH hallucination risk — model confidently predicts wrong structures' if halluc_ratio > 0.5 else 'low hallucination risk — predictions are structurally reasonable'}."
        ),
    }





def bn_statistics_analysis(source_images: list, target_images: list,
                           model: nn.Module, device: str,
                           max_samples: int = 200, task=None) -> Dict:
    """Analyze BatchNorm running statistics shift between source and target.

    Re-runs target images in train mode (with reset BN stats) to see how
    much the running mean/var would drift — indicates per-layer distribution
    shift severity.

    Task-aware: when a task adapter is supplied, defers tensor prep to it
    so 3D volumes work. Else uses the legacy 2D channels-last path."""
    model.eval()

    # Collect current BN stats (trained on source)
    source_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                source_stats[name] = {
                    "mean": m.running_mean.clone(),
                    "var": m.running_var.clone(),
                }

    if not source_stats:
        return {
            "tool": "bn_statistics_analysis",
            "has_batchnorm": False,
            "interpretation": "Model has no BatchNorm layers — BN-based adaptation (Tent/SAR) won't work.",
        }

    # Re-estimate BN stats on target (without updating model)
    import copy
    model_copy = copy.deepcopy(model)
    model_copy.train()
    for m in model_copy.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average

    with torch.no_grad():
        for img in target_images[:max_samples]:
            try:
                if task is not None:
                    tensor = task.to_tensor(img, device)
                else:
                    if img.ndim == 3 and img.shape[2] in (1, 3):
                        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                    elif img.ndim == 2:
                        t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
                    else:
                        t = torch.from_numpy(img.astype(np.float32))
                    tensor = t.unsqueeze(0).to(device)
                model_copy(tensor)
            except Exception:
                continue  # skip incompatible samples

    # Compare source vs target BN stats
    shifts = []
    per_layer = {}
    for name, m in model_copy.named_modules():
        if name in source_stats and isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            src_mean = source_stats[name]["mean"]
            tgt_mean = m.running_mean
            src_var = source_stats[name]["var"]
            tgt_var = m.running_var

            mean_shift = (tgt_mean - src_mean).abs().mean().item()
            var_shift = (tgt_var - src_var).abs().mean().item()
            shifts.append(mean_shift + var_shift)
            per_layer[name] = {"mean_shift": mean_shift, "var_shift": var_shift}

    del model_copy

    avg_shift = float(np.mean(shifts)) if shifts else 0
    max_shift = float(np.max(shifts)) if shifts else 0

    return {
        "tool": "bn_statistics_analysis",
        "has_batchnorm": True,
        "n_bn_layers": len(source_stats),
        "avg_bn_shift": avg_shift,
        "max_bn_shift": max_shift,
        "per_layer_top5": dict(sorted(per_layer.items(),
                                       key=lambda x: x[1]["mean_shift"] + x[1]["var_shift"],
                                       reverse=True)[:5]),
        "interpretation": (
            f"BN shift={avg_shift:.4f} (max={max_shift:.4f}): "
            f"{'severe BN mismatch — BN reset is critical' if avg_shift > 0.5 else 'moderate BN mismatch — BN reset should help' if avg_shift > 0.1 else 'mild BN mismatch — BN reset may help slightly'}."
        ),
    }


# Registry of available tools
ANALYSIS_TOOLS = {
    "intensity_analysis": {
        "fn": intensity_analysis,
        "needs_model": False,
        "description": "Compute intensity distribution KL divergence and per-channel shifts",
    },
    "frequency_analysis": {
        "fn": frequency_analysis,
        "needs_model": False,
        "description": "Compute FFT-based style (low-freq) and texture (high-freq) shifts",
    },
    "scale_analysis": {
        "fn": scale_analysis,
        "needs_model": True,
        "description": "Compare predicted foreground area between source and target",
    },
    "confidence_analysis": {
        "fn": confidence_analysis,
        "needs_model": True,
        "description": "Measure model confidence and entropy on target predictions",
    },
    "hallucination_detection": {
        "fn": hallucination_detection,
        "needs_model": True,
        "description": "Detect if model is confidently predicting wrong structures",
    },
    "bn_statistics_analysis": {
        "fn": bn_statistics_analysis,
        "needs_model": True,
        "description": "Measure BatchNorm running statistics shift between source and target domains",
    },
}
