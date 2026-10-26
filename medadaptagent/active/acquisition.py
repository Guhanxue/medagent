"""Information-theoretic active labeling for MedAdaptAgent.

When a label budget of K target samples is available, the agentic system
should spend it wisely: rather than drawing K samples at random (or by a
foreground heuristic), it *acquisition-scores* every unlabeled target
sample with the source model's own predictive signal and requests labels
for the most informative ones. Agent-selected labels carry more signal
per label than passively drawn ones, which the active-vs-stratified-vs-
random ablation makes auditable.

The scorer is fully task-agnostic. It reads only objects that are already
in scope for any task in the sandbox:

  * the source ``nn.Module`` and the unlabeled target images, and
  * the ``TaskAdapter`` (drives ``to_tensor`` / ``forward_pred`` so the
    same calibrated preprocessing + inference path is reused for seg 2D,
    seg 3D, cls 2D, and cls 3D), and
  * Agent-1's foreground prior (``expected_area_fraction``), when present.

No dataset names, structure names, or modality strings ever enter the
scoring math; everything is derived from the model's predictive
distribution and the Agent-1 prior. The same code path therefore applies
to any task the registry can describe.

Three complementary acquisition signals are combined:

  (a) *Predictive entropy* of the softmax (high = the model is unsure;
      labeling these resolves the most ambiguity).
  (b) *Prior deviation* — distance between the predicted foreground
      fraction and Agent-1's expected area fraction. Samples whose
      coverage sits near the boundary of the expected range are the
      ones whose labels most discipline the adaptation. For tasks
      without a spatial foreground notion (classification), this signal
      is simply absent and contributes nothing.
  (c) *Flip disagreement* — an epistemic proxy from a horizontal-flip
      test-time-augmentation pass: a confident, well-specified model is
      flip-equivariant, so high disagreement flags samples on which the
      source model is unstable under shift.

Each signal is rank-normalized to [0, 1] across the candidate pool
(robust to differing raw scales) and averaged into a single acquisition
score. The top-K scoring samples are selected.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------
# Agent-1 prior parsing (mirrors execution_engineer's reader so the
# active selector and the final-check gate agree on the expected range)
# --------------------------------------------------------------
def _expected_area_fraction(knowledge: Optional[Dict]
                              ) -> Optional[Tuple[float, float]]:
    """Pull (min, max) expected foreground area fraction from Agent-1's
    knowledge dict. Returns None when the prior is absent (e.g., cls
    tasks, or a stubbed Agent-1)."""
    fg_priors = (knowledge or {}).get("foreground_priors", {}) or {}
    ea = fg_priors.get("expected_area_fraction")
    if isinstance(ea, (list, tuple)) and len(ea) >= 2:
        try:
            lo, hi = float(ea[0]), float(ea[1])
            if hi >= lo:
                return lo, hi
        except (TypeError, ValueError):
            return None
    return None


# --------------------------------------------------------------
# Per-sample predictive signals via the TaskAdapter forward path
# --------------------------------------------------------------
def _softmax_logits(task, model, image: np.ndarray, device: str
                      ) -> Optional[torch.Tensor]:
    """Run the calibrated forward path and return softmax probabilities
    over the class dimension as a CPU tensor [C, ...spatial], or [C] for
    classification. Returns None on any per-sample failure (kept robust
    so one bad volume never aborts acquisition scoring)."""
    try:
        tensor = task.to_tensor(image, device)
        logits, _ = task.forward_pred(model, tensor)
        if not torch.is_tensor(logits):
            return None
        logits = logits.detach().float()
        # Collapse leading batch dim if present: [B, C, ...] -> [C, ...].
        if logits.ndim >= 2 and logits.shape[0] == 1:
            logits = logits[0]
        probs = torch.softmax(logits, dim=0)
        return probs.cpu()
    except Exception:
        return None


def _entropy_of(probs: torch.Tensor) -> float:
    """Mean predictive entropy (nats), normalized to [0, 1] by log(C).
    probs is [C, ...spatial] or [C]. Spatial entropy is averaged over
    all spatial positions, giving a single scalar per sample that is
    comparable across seg (dense) and cls (single position)."""
    c = probs.shape[0]
    ent = -(probs * torch.clamp(probs, min=1e-8).log()).sum(dim=0)
    mean_ent = float(ent.mean().item())
    denom = float(np.log(max(c, 2)))
    return float(min(max(mean_ent / denom, 0.0), 1.0))


def _fg_fraction_of(probs: torch.Tensor) -> Optional[float]:
    """Predicted foreground fraction for dense (segmentation) outputs:
    the spatial mean of P(class != background). Returns None for
    classification (no spatial axis), where the prior-deviation signal
    does not apply."""
    if probs.ndim < 2:
        return None  # classification: [C] only, no spatial extent
    fg_prob = 1.0 - probs[0]                 # P(not background), per voxel
    return float(fg_prob.mean().item())


def _flip_disagreement(task, model, image: np.ndarray, device: str,
                          base_probs: Optional[torch.Tensor]) -> Optional[float]:
    """Epistemic proxy: predict on a horizontally-flipped copy of the
    input, flip the prediction back, and measure disagreement with the
    un-augmented prediction. The flip is applied to the LAST spatial
    axis of the numpy image, which is the width axis for both 2D
    (H, W[, C]) and 3D (D, H, W) inputs, so the operation is 3D-safe.

    For dense outputs disagreement is the fraction of spatial positions
    whose argmax flips; for classification it is the total-variation
    distance between the two class distributions. Returns None if the
    flipped forward cannot be computed."""
    arr = np.asarray(image)
    # Width is the last spatial axis. For channel-last 2D images
    # (H, W, C) flip axis -2; otherwise flip the last axis.
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        flip_axis = -2
    else:
        flip_axis = -1
    flipped = np.ascontiguousarray(np.flip(arr, axis=flip_axis))
    flip_probs = _softmax_logits(task, model, flipped, device)
    if flip_probs is None or base_probs is None:
        return None
    if flip_probs.ndim < 2:
        # Classification: compare class distributions directly (a flip
        # should not change the label of a confident classifier).
        if flip_probs.shape != base_probs.shape:
            return None
        tv = 0.5 * float((flip_probs - base_probs).abs().sum().item())
        return float(min(max(tv, 0.0), 1.0))
    # Dense: un-flip the spatial prediction, compare argmax maps.
    base_pred = base_probs.argmax(dim=0)
    flip_pred = flip_probs.argmax(dim=0)
    # Flip back along the spatial width axis of the prediction map.
    # base_pred is [...spatial]; width is its last axis.
    flip_pred_back = torch.flip(flip_pred, dims=[flip_pred.ndim - 1])
    if flip_pred_back.shape != base_pred.shape:
        return None
    disagree = float((base_pred != flip_pred_back).float().mean().item())
    return float(min(max(disagree, 0.0), 1.0))


# --------------------------------------------------------------
# Rank normalization (robust to differing raw scales across signals)
# --------------------------------------------------------------
def _rank_normalize(values: List[Optional[float]]) -> np.ndarray:
    """Map a list of scores to [0, 1] by rank. None entries (signal not
    available for that sample) get the neutral midpoint 0.5 so they
    neither boost nor penalize the combined score. A signal that is
    None for *every* sample contributes a flat 0.5 (i.e., it drops out
    of the ranking), which is exactly the desired behavior when a whole
    signal does not apply to the task."""
    arr = np.array([np.nan if v is None else float(v) for v in values],
                     dtype=np.float64)
    out = np.full(arr.shape, 0.5, dtype=np.float64)
    valid = ~np.isnan(arr)
    n = int(valid.sum())
    if n <= 1:
        return out
    order = np.argsort(np.argsort(arr[valid]))      # 0..n-1 ranks
    out[valid] = order.astype(np.float64) / float(n - 1)
    return out


# --------------------------------------------------------------
# Public API
# --------------------------------------------------------------
def score_target_samples(
    task,
    model,
    target_images: Sequence[np.ndarray],
    candidate_indices: List[int],
    knowledge: Optional[Dict],
    device: str,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[int, Dict[str, Optional[float]]]:
    """Acquisition-score each candidate target sample.

    Parameters
    ----------
    task : TaskAdapter
        Drives ``to_tensor`` + ``forward_pred`` so the calibrated
        preprocessing + inference path is reused for any task type.
    model : nn.Module
        The source model (its predictive signal drives acquisition).
    target_images : sequence of np.ndarray
        The full target pool (2D arrays or 3D volumes).
    candidate_indices : list of int
        Indices into ``target_images`` that are eligible for labeling
        (e.g., those with loadable ground truth at request time).
    knowledge : dict or None
        Agent-1 domain analysis; supplies ``expected_area_fraction``.
    device : str
        Torch device string.
    weights : (entropy, prior_deviation, flip) tuple
        Relative weights for the three rank-normalized signals.

    Returns
    -------
    dict mapping global index -> per-signal record with keys
    ``{entropy, prior_dev, flip, score}`` (raw signals plus the final
    combined acquisition score in [0, 1]). Indices the model could not
    be run on at all are returned with score 0.0.
    """
    was_training = model.training
    model.eval()

    ea_range = _expected_area_fraction(knowledge)

    raw_entropy: List[Optional[float]] = []
    raw_prior_dev: List[Optional[float]] = []
    raw_flip: List[Optional[float]] = []
    runnable: List[bool] = []

    for idx in candidate_indices:
        img = target_images[idx]
        probs = _softmax_logits(task, model, img, device)
        if probs is None:
            raw_entropy.append(None)
            raw_prior_dev.append(None)
            raw_flip.append(None)
            runnable.append(False)
            continue
        runnable.append(True)
        # (a) predictive entropy
        raw_entropy.append(_entropy_of(probs))
        # (b) deviation of predicted fg-fraction from the Agent-1 prior
        fg = _fg_fraction_of(probs)
        if fg is None or ea_range is None:
            raw_prior_dev.append(None)
        else:
            lo, hi = ea_range
            # Distance OUTSIDE the expected band; samples landing in the
            # ambiguous boundary / out-of-prior region score higher.
            if fg < lo:
                raw_prior_dev.append(lo - fg)
            elif fg > hi:
                raw_prior_dev.append(fg - hi)
            else:
                # Inside the band: NEGATED distance to the nearer edge so
                # near-boundary samples (min~0 -> ~0) outrank dead-center
                # ones (min large -> very negative). Matches the spec that
                # boundary cases are more informative than prior-consistent
                # dead-center ones; rank-normalized downstream.
                raw_prior_dev.append(-min(fg - lo, hi - fg))
        # (c) flip-TTA disagreement
        raw_flip.append(_flip_disagreement(task, model, img, device, probs))

    if was_training:
        model.train()

    n_ent = _rank_normalize(raw_entropy)
    n_dev = _rank_normalize(raw_prior_dev)
    n_flip = _rank_normalize(raw_flip)

    w_e, w_d, w_f = weights
    w_sum = (w_e + w_d + w_f) or 1.0

    out: Dict[int, Dict[str, Optional[float]]] = {}
    for j, idx in enumerate(candidate_indices):
        if not runnable[j]:
            out[idx] = {"entropy": None, "prior_dev": None,
                          "flip": None, "score": 0.0}
            continue
        score = (w_e * n_ent[j] + w_d * n_dev[j] + w_f * n_flip[j]) / w_sum
        out[idx] = {
            "entropy": raw_entropy[j],
            "prior_dev": raw_prior_dev[j],
            "flip": raw_flip[j],
            "score": float(score),
        }
    return out


def select_active_indices(
    task,
    model,
    target_images: Sequence[np.ndarray],
    candidate_indices: List[int],
    k: int,
    knowledge: Optional[Dict],
    device: str,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[List[int], Dict[int, Dict[str, Optional[float]]]]:
    """Pick the top-K most informative candidate indices.

    Returns ``(chosen_sorted, scores)`` where ``chosen_sorted`` is the
    selected global indices in ascending order (so downstream label
    gathering is order-stable) and ``scores`` is the full per-candidate
    record from :func:`score_target_samples` (for auditing the
    ablation). Ties are broken by lower index for determinism.
    """
    scores = score_target_samples(
        task, model, target_images, candidate_indices, knowledge,
        device, weights=weights)
    k = min(k, len(candidate_indices))
    ranked = sorted(
        candidate_indices,
        key=lambda i: (-(scores[i]["score"] or 0.0), i),
    )
    chosen = sorted(ranked[:k])
    return chosen, scores
