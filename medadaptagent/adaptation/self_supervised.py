"""
Self-supervised adaptation objectives for annotation-free training.

Four objectives:
(a) Entropy minimization on high-confidence regions
(b) Consistency regularization under augmentations
(c) Pseudo-label self-training with reliability filtering
(d) Text-guided contrastive alignment (with BiomedCLIP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class EntropyMinimization(nn.Module):
    """Minimize entropy on high-confidence predictions.

    Only applies to pixels where the model is already reasonably confident,
    avoiding reinforcement of uncertain/wrong predictions.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output.
        Returns:
            Scalar entropy loss.
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)  # (B, H, W)

        # Only minimize entropy where model is confident
        max_probs = probs.max(dim=1)[0]
        confident_mask = max_probs > self.confidence_threshold

        if confident_mask.sum() == 0:
            return entropy.mean() * 0.1  # Small loss when nothing is confident

        return entropy[confident_mask].mean()


class ConsistencyRegularization(nn.Module):
    """Enforce prediction consistency under random augmentations.

    Encourages augmentation-invariant predictions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits_orig: torch.Tensor,
                logits_aug: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_orig: (B, C, H, W) predictions on original images.
            logits_aug: (B, C, H, W) predictions on augmented images
                        (already un-augmented to match spatial alignment).
        Returns:
            Scalar consistency loss.
        """
        probs_orig = F.softmax(logits_orig, dim=1)
        probs_aug = F.softmax(logits_aug, dim=1)

        # Symmetric KL divergence
        kl_forward = F.kl_div(
            probs_aug.log(), probs_orig, reduction="batchmean"
        )
        kl_backward = F.kl_div(
            probs_orig.log(), probs_aug, reduction="batchmean"
        )
        return (kl_forward + kl_backward) / 2


class PseudoLabelSelfTraining(nn.Module):
    """Self-training with reliable pseudo-labels.

    Filters pseudo-labels by confidence threshold and optional
    anatomical plausibility checks.
    """

    def __init__(self, confidence_threshold: float = 0.8):
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(self, logits: torch.Tensor,
                plausibility_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output.
            plausibility_mask: Optional (B, H, W) bool mask of anatomically plausible regions.
        Returns:
            Scalar pseudo-label loss.
        """
        probs = F.softmax(logits, dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)

        # Reliable mask: high confidence
        reliable_mask = max_probs > self.confidence_threshold

        # Combine with anatomical plausibility
        if plausibility_mask is not None:
            reliable_mask = reliable_mask & plausibility_mask

        if reliable_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, pseudo_labels, reduction="none")
        return (loss * reliable_mask.float()).sum() / reliable_mask.sum()


class TextGuidedAlignment(nn.Module):
    """Contrastive alignment between visual features and BiomedCLIP text embeddings.

    Encourages visual features for each class to be similar to the
    corresponding text description embedding.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, visual_features: torch.Tensor,
                text_features: torch.Tensor,
                pseudo_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, D, H, W) features from model.
            text_features: (num_classes, D) from BiomedCLIP.
            pseudo_labels: (B, H, W) pseudo-label map.
        Returns:
            Scalar alignment loss.
        """
        B, D, H, W = visual_features.shape
        num_classes = text_features.shape[0]

        # Normalize features
        vis = F.normalize(visual_features, dim=1)  # (B, D, H, W)
        txt = F.normalize(text_features, dim=1)  # (num_classes, D)

        total_loss = torch.tensor(0.0, device=visual_features.device)
        count = 0

        for c in range(num_classes):
            mask = pseudo_labels == (c + 1)  # classes are 1-indexed
            if mask.sum() < 10:
                continue

            # Pool visual features for this class
            mask_expanded = mask.unsqueeze(1).float()  # (B, 1, H, W)
            class_features = (vis * mask_expanded).sum(dim=[2, 3]) / mask_expanded.sum(dim=[2, 3]).clamp(min=1)
            class_features = F.normalize(class_features, dim=1)  # (B, D)

            # Contrastive: class features should be similar to corresponding text
            # and dissimilar to other class texts
            sim = class_features @ txt.T / self.temperature  # (B, num_classes)
            labels = torch.full((B,), c, device=sim.device, dtype=torch.long)
            loss = F.cross_entropy(sim, labels)
            total_loss = total_loss + loss
            count += 1

        return total_loss / max(count, 1)


class SelfSupervisedObjectives(nn.Module):
    """Combined self-supervised training objectives.

    Manages all four objectives with configurable weights.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}

        self.entropy_min = EntropyMinimization(
            confidence_threshold=config.get("entropy_confidence_threshold", 0.6)
        )
        self.consistency = ConsistencyRegularization()
        self.pseudo_label = PseudoLabelSelfTraining(
            confidence_threshold=config.get("pseudo_label_threshold", 0.8)
        )
        self.text_alignment = TextGuidedAlignment(
            temperature=config.get("temperature", 0.07)
        )

        self.w_ent = config.get("entropy_weight", 1.0)
        self.w_con = config.get("consistency_weight", 1.0)
        self.w_pse = config.get("pseudo_label_weight", 1.0)
        self.w_txt = config.get("text_alignment_weight", 0.5)

    def forward(self, logits: torch.Tensor,
                logits_aug: torch.Tensor = None,
                visual_features: torch.Tensor = None,
                text_features: torch.Tensor = None,
                plausibility_mask: torch.Tensor = None,
                active_objectives: list = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined self-supervised loss.

        Args:
            logits: (B, C, H, W) predictions on original images.
            logits_aug: (B, C, H, W) predictions on augmented images. Optional.
            visual_features: (B, D, H, W) for text alignment. Optional.
            text_features: (num_classes, D) BiomedCLIP embeddings. Optional.
            plausibility_mask: (B, H, W) anatomical plausibility. Optional.
            active_objectives: List of active objective names. If None, use all available.

        Returns:
            total_loss: Scalar loss.
            loss_dict: Per-objective losses for logging.
        """
        if active_objectives is None:
            active_objectives = ["entropy_min", "consistency", "pseudo_label", "text_alignment"]

        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        loss_dict = {}

        if "entropy_min" in active_objectives:
            l_ent = self.entropy_min(logits)
            total_loss = total_loss + self.w_ent * l_ent
            loss_dict["entropy"] = l_ent.item()

        if "consistency" in active_objectives and logits_aug is not None:
            l_con = self.consistency(logits, logits_aug)
            total_loss = total_loss + self.w_con * l_con
            loss_dict["consistency"] = l_con.item()

        if "pseudo_label" in active_objectives:
            l_pse = self.pseudo_label(logits, plausibility_mask)
            total_loss = total_loss + self.w_pse * l_pse
            loss_dict["pseudo_label"] = l_pse.item()

        if "text_alignment" in active_objectives and visual_features is not None and text_features is not None:
            pseudo_labels = logits.argmax(dim=1)
            l_txt = self.text_alignment(visual_features, text_features, pseudo_labels)
            total_loss = total_loss + self.w_txt * l_txt
            loss_dict["text_alignment"] = l_txt.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
