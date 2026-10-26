"""
Composite Quality Score Q(x) for annotation-free self-evaluation.

Components:
1. TTA Consistency (r ~ 0.6-0.8 with Dice)
2. Ensemble Agreement (r ~ 0.7-0.9 with Dice)
3. Calibrated Confidence (r ~ 0.5-0.7 with Dice)
4. Anatomical Plausibility (binary filter, 0.9+ precision)
5. Shape Prior Consistency (r ~ 0.4-0.6 with Dice)

Weights learned on source domain to maximize Kendall's tau with Dice.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import label as scipy_label
from scipy.stats import kendalltau


class QualityScore:
    """Composite quality metric for annotation-free evaluation."""

    def __init__(self, weights: Dict[str, float] = None,
                 anatomy_config: dict = None):
        """
        Args:
            weights: Component weights. Default: w1=0.3, w2=0.35, w3=0.15, w4=0.15, w5=0.05.
            anatomy_config: Expected anatomical properties (volumes, component counts, etc.).
        """
        self.weights = weights or {
            "tta_consistency": 0.30,
            "ensemble_agreement": 0.35,
            "calibrated_confidence": 0.15,
            "anatomical_plausibility": 0.15,
            "shape_prior": 0.05,
        }
        self.anatomy_config = anatomy_config or {}
        self.temperature = 1.0  # Calibration temperature

    def compute(self, predictions: List[np.ndarray],
                logits_list: List[torch.Tensor] = None,
                tta_predictions: List[List[np.ndarray]] = None,
                ensemble_predictions: List[List[np.ndarray]] = None) -> Tuple[float, Dict[str, float]]:
        """Compute composite quality score.

        Args:
            predictions: List of prediction masks.
            logits_list: List of raw logit tensors.
            tta_predictions: Per-image list of TTA augmented predictions.
            ensemble_predictions: Per-image list of ensemble member predictions.

        Returns:
            Q: Composite score (0-1).
            components: Per-component scores.
        """
        components = {}

        # 1. TTA Consistency
        if tta_predictions is not None:
            components["tta_consistency"] = self._tta_consistency(tta_predictions)
        else:
            components["tta_consistency"] = 0.5  # Default when not available

        # 2. Ensemble Agreement
        if ensemble_predictions is not None:
            components["ensemble_agreement"] = self._ensemble_agreement(ensemble_predictions)
        else:
            components["ensemble_agreement"] = 0.5

        # 3. Calibrated Confidence
        if logits_list is not None:
            components["calibrated_confidence"] = self._calibrated_confidence(logits_list)
        else:
            components["calibrated_confidence"] = 0.5

        # 4. Anatomical Plausibility
        components["anatomical_plausibility"] = self._anatomical_plausibility(predictions)

        # 5. Shape Prior
        components["shape_prior"] = self._shape_prior(predictions)

        # Composite score
        Q = sum(self.weights[k] * components[k] for k in self.weights if k in components)

        return float(Q), components

    def _tta_consistency(self, tta_predictions: List[List[np.ndarray]]) -> float:
        """Mean pairwise IoU across TTA augmented predictions."""
        all_ious = []
        for image_preds in tta_predictions:
            if len(image_preds) < 2:
                continue
            ious = []
            for i in range(len(image_preds)):
                for j in range(i + 1, len(image_preds)):
                    iou = self._iou(image_preds[i], image_preds[j])
                    ious.append(iou)
            if ious:
                all_ious.append(np.mean(ious))
        return float(np.mean(all_ious)) if all_ious else 0.5

    def _ensemble_agreement(self, ensemble_predictions: List[List[np.ndarray]]) -> float:
        """Mean pairwise IoU across ensemble member predictions."""
        all_ious = []
        for image_preds in ensemble_predictions:
            if len(image_preds) < 2:
                continue
            ious = []
            for i in range(len(image_preds)):
                for j in range(i + 1, len(image_preds)):
                    iou = self._iou(image_preds[i], image_preds[j])
                    ious.append(iou)
            if ious:
                all_ious.append(np.mean(ious))
        return float(np.mean(all_ious)) if all_ious else 0.5

    def _calibrated_confidence(self, logits_list: List[torch.Tensor]) -> float:
        """Mean calibrated confidence over predictions."""
        confidences = []
        for logits in logits_list:
            if logits.ndim == 4:
                logits = logits[0]
            probs = F.softmax(logits / self.temperature, dim=0)
            max_prob = probs.max(dim=0)[0]
            confidences.append(float(max_prob.mean()))
        return float(np.mean(confidences)) if confidences else 0.5

    def _anatomical_plausibility(self, predictions: List[np.ndarray]) -> float:
        """Check anatomical constraints: volume range, connectivity, topology."""
        if not self.anatomy_config:
            return 0.8  # Default when no constraints

        passing = 0
        total = len(predictions)

        for pred in predictions:
            checks_passed = 0
            checks_total = 0

            # Volume check
            volume_range = self.anatomy_config.get("volume_range")
            if volume_range is not None:
                pred_volume = (pred > 0).sum()
                if volume_range[0] <= pred_volume <= volume_range[1]:
                    checks_passed += 1
                checks_total += 1

            # Connectivity check
            max_components = self.anatomy_config.get("max_components", 10)
            _, n_components = scipy_label(pred > 0)
            if n_components <= max_components:
                checks_passed += 1
            checks_total += 1

            # Non-empty check
            if (pred > 0).sum() > 0:
                checks_passed += 1
            checks_total += 1

            if checks_total > 0:
                if checks_passed / checks_total >= 0.5:
                    passing += 1

        return passing / total if total > 0 else 0.5

    def _shape_prior(self, predictions: List[np.ndarray]) -> float:
        """Basic shape prior: compactness ratio."""
        compactness_scores = []
        for pred in predictions:
            binary = (pred > 0).astype(np.uint8)
            if binary.sum() == 0:
                continue
            # Compactness: 4*pi*area / perimeter^2 (1 for circle)
            area = binary.sum()
            from scipy.ndimage import binary_erosion
            perimeter = (binary ^ binary_erosion(binary)).sum()
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
                compactness_scores.append(min(compactness, 1.0))
        return float(np.mean(compactness_scores)) if compactness_scores else 0.5

    @staticmethod
    def _iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        m1 = mask1.astype(bool)
        m2 = mask2.astype(bool)
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return float(intersection / union) if union > 0 else 1.0

    def calibrate_temperature(self, logits_list: List[torch.Tensor],
                               dice_scores: List[float]) -> float:
        """Calibrate temperature on source validation set."""
        best_tau = -1
        best_temp = 1.0
        for temp in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            self.temperature = temp
            confidences = []
            for logits in logits_list:
                if logits.ndim == 4:
                    logits = logits[0]
                probs = F.softmax(logits / temp, dim=0)
                confidences.append(float(probs.max(dim=0)[0].mean()))
            tau, _ = kendalltau(confidences, dice_scores)
            if tau > best_tau:
                best_tau = tau
                best_temp = temp
        self.temperature = best_temp
        return best_temp

    def calibrate_weights(self, component_scores_list: List[Dict[str, float]],
                           dice_scores: List[float]) -> Dict[str, float]:
        """Calibrate component weights on source validation set.

        Finds weights that maximize Kendall's tau between Q and Dice.
        Uses simple grid search.
        """
        from itertools import product

        best_tau = -1
        best_weights = self.weights.copy()
        weight_options = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

        keys = list(self.weights.keys())
        # Simplified: just search over main 3 components
        main_keys = ["tta_consistency", "ensemble_agreement", "calibrated_confidence"]

        for w1, w2, w3 in product(weight_options, repeat=3):
            if abs(w1 + w2 + w3 - 0.8) > 0.01:
                continue
            trial_weights = {
                "tta_consistency": w1,
                "ensemble_agreement": w2,
                "calibrated_confidence": w3,
                "anatomical_plausibility": 0.15,
                "shape_prior": 0.05,
            }
            q_scores = []
            for comp in component_scores_list:
                q = sum(trial_weights.get(k, 0) * comp.get(k, 0.5) for k in trial_weights)
                q_scores.append(q)
            tau, _ = kendalltau(q_scores, dice_scores)
            if tau > best_tau:
                best_tau = tau
                best_weights = trial_weights

        self.weights = best_weights
        return best_weights
