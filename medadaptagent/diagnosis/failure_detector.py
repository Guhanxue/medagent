"""
Failure Detector: Compute uncertainty and failure indicators from model predictions.
Generates entropy maps, confidence maps, and TTA variance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FailureMap:
    """Collection of uncertainty/failure indicators for a single image."""
    entropy_map: np.ndarray       # Pixel-wise entropy
    confidence_map: np.ndarray    # Pixel-wise max probability
    tta_variance: np.ndarray      # Pixel-wise variance across augmentations
    mean_entropy: float
    mean_confidence: float
    mean_tta_variance: float
    prediction: np.ndarray        # Original prediction mask


def compute_entropy_map(logits: torch.Tensor) -> np.ndarray:
    """Compute pixel-wise entropy from logits.

    Args:
        logits: (C, H, W) or (1, C, H, W) raw model output.
    Returns:
        entropy: (H, W) entropy map.
    """
    if logits.ndim == 4:
        logits = logits[0]
    probs = F.softmax(logits, dim=0)
    log_probs = F.log_softmax(logits, dim=0)
    entropy = -(probs * log_probs).sum(dim=0)
    return entropy.cpu().numpy()


def compute_confidence_map(logits: torch.Tensor) -> np.ndarray:
    """Compute pixel-wise confidence (max probability).

    Args:
        logits: (C, H, W) or (1, C, H, W).
    Returns:
        confidence: (H, W) confidence map.
    """
    if logits.ndim == 4:
        logits = logits[0]
    probs = F.softmax(logits, dim=0)
    confidence = probs.max(dim=0)[0]
    return confidence.cpu().numpy()


def compute_tta_variance(model, image: np.ndarray, n_augmentations: int = 8,
                         predict_fn=None,
                         image_size: int = 256,
                         output_size: Tuple[int, int] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute test-time augmentation variance.

    Args:
        model: SAM2 wrapper, UNet, or any model.
        image: Input image (H,W,3) float32 [0,1].
        n_augmentations: Number of augmented versions.
        predict_fn: Function(model, aug_image) -> mask. If None, auto-detects.
        image_size: Resize target for UNet-style models.
        output_size: (H, W) target size for output. If None, uses image_size for
            UNet or original size for SAM2.

    Returns:
        variance_map: (H, W) pixel-wise variance.
        augmented_preds: List of augmented predictions.
    """
    from scipy.ndimage import zoom as scipy_zoom
    original_h, original_w = image.shape[:2]

    # Determine target output size
    if output_size is not None:
        target_h, target_w = output_size
    elif not hasattr(model, 'predict_grid') and predict_fn is None:
        # UNet-style: output at model resolution
        target_h, target_w = image_size, image_size
    else:
        target_h, target_w = original_h, original_w

    augmented_preds = []

    for k in range(n_augmentations):
        aug_image = _random_augment(image, seed=k)
        if predict_fn is not None:
            pred = predict_fn(model, aug_image)
        elif hasattr(model, 'predict_grid'):
            # SAM2 interface
            masks, scores, _ = model.predict_grid(aug_image, grid_size=64)
            if len(masks) > 0:
                best_idx = np.argmax(scores)
                pred = masks[best_idx].astype(np.float32)
            else:
                pred = np.zeros(aug_image.shape[:2], dtype=np.float32)
        else:
            # UNet-style: forward pass with preprocessing
            from PIL import Image as PILImage
            pil = PILImage.fromarray((aug_image * 255).astype(np.uint8))
            pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
            tensor = torch.from_numpy(
                np.array(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            device = next(model.parameters()).device
            with torch.no_grad():
                logits = model(tensor.to(device))
                pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.float32)

        # Undo augmentation on prediction
        pred = _inverse_augment(pred, seed=k)

        # Resize to target output size using scipy (handles float and multi-class)
        if pred.shape != (target_h, target_w):
            zoom_h = target_h / pred.shape[0]
            zoom_w = target_w / pred.shape[1]
            pred = scipy_zoom(pred, (zoom_h, zoom_w), order=0)  # order=0 = nearest

        augmented_preds.append(pred)

    preds_stack = np.stack(augmented_preds, axis=0)
    variance_map = preds_stack.var(axis=0)
    return variance_map, augmented_preds


def _random_augment(image: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply random augmentation for TTA."""
    rng = np.random.RandomState(seed)
    aug = image.copy()

    # Random horizontal flip
    if rng.random() > 0.5:
        aug = np.flip(aug, axis=1).copy()

    # Random rotation (90 degree increments)
    k = rng.randint(0, 4)
    if k > 0:
        aug = np.rot90(aug, k=k, axes=(0, 1)).copy()

    # Random intensity jitter
    if aug.dtype == np.float32:
        scale = rng.uniform(0.9, 1.1)
        shift = rng.uniform(-0.05, 0.05)
        aug = np.clip(aug * scale + shift, 0, 1)

    return aug


def _inverse_augment(pred: np.ndarray, seed: int = 0) -> np.ndarray:
    """Invert the augmentation applied with the given seed."""
    rng = np.random.RandomState(seed)
    result = pred.copy()

    flip = rng.random() > 0.5
    k = rng.randint(0, 4)
    _ = rng.uniform(0.9, 1.1)  # consume same random state
    _ = rng.uniform(-0.05, 0.05)

    # Inverse rotation
    if k > 0:
        result = np.rot90(result, k=4 - k, axes=(0, 1)).copy()

    # Inverse flip
    if flip:
        result = np.flip(result, axis=1).copy()

    return result


def detect_failures(model, images: List[np.ndarray],
                    predict_fn=None,
                    logits_fn=None,
                    n_tta: int = 8,
                    image_size: int = 256) -> List[FailureMap]:
    """Run failure detection on a batch of images.

    Args:
        model: Model (SAM2 wrapper, UNet, or any model).
        images: List of input images (H,W,3) float32 [0,1].
        predict_fn: Function(model, image) -> prediction mask (H,W).
        logits_fn: Function(model, image) -> logits tensor (C,H,W).
            If None, tries model.predict_grid() for SAM2 or model() for UNet.
        n_tta: Number of TTA augmentations.
        image_size: Resize target for UNet-style models.

    Returns:
        List of FailureMap objects.
    """
    failure_maps = []

    for image in images:
        if logits_fn is not None:
            # Use provided logits function (works with any model)
            logits = logits_fn(model, image)
            if logits.ndim == 4:
                logits = logits[0]  # remove batch dim
            prediction = logits.argmax(dim=0).cpu().numpy()
        elif hasattr(model, 'predict_grid'):
            # SAM2 interface
            masks, scores, logits_np = model.predict_grid(image, grid_size=64)
            if len(masks) > 0:
                best_idx = np.argmax(scores)
                prediction = masks[best_idx]
                logits = torch.from_numpy(logits_np[best_idx:best_idx + 1]).float()
            else:
                prediction = np.zeros(image.shape[:2], dtype=np.uint8)
                logits = torch.zeros(1, image.shape[0], image.shape[1])
        else:
            # UNet-style: forward pass with preprocessing
            from PIL import Image as PILImage
            pil = PILImage.fromarray((image * 255).astype(np.uint8))
            pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
            tensor = torch.from_numpy(
                np.array(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            device = next(model.parameters()).device
            with torch.no_grad():
                logits = model(tensor.to(device))[0].cpu()  # (C, H, W)
            prediction = logits.argmax(dim=0).numpy()

        # Ensure logits have correct shape
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        # For binary: create 2-channel logits
        if logits.shape[0] == 1:
            logits = torch.cat([-logits, logits], dim=0)

        entropy = compute_entropy_map(logits)
        confidence = compute_confidence_map(logits)

        # Resize entropy/confidence to match prediction shape if needed
        from scipy.ndimage import zoom as scipy_zoom
        pred_h, pred_w = prediction.shape[:2]
        if entropy.shape != (pred_h, pred_w):
            zoom_h = pred_h / entropy.shape[0]
            zoom_w = pred_w / entropy.shape[1]
            entropy = scipy_zoom(entropy, (zoom_h, zoom_w), order=1)
            confidence = scipy_zoom(confidence, (zoom_h, zoom_w), order=1)

        # TTA variance — request output at prediction resolution directly
        tta_var, _ = compute_tta_variance(model, image, n_augmentations=n_tta,
                                          predict_fn=predict_fn,
                                          image_size=image_size,
                                          output_size=(pred_h, pred_w))

        # Final safety check
        if tta_var.shape != (pred_h, pred_w):
            zoom_h = pred_h / tta_var.shape[0]
            zoom_w = pred_w / tta_var.shape[1]
            tta_var = scipy_zoom(tta_var, (zoom_h, zoom_w), order=1)

        failure_maps.append(FailureMap(
            entropy_map=entropy,
            confidence_map=confidence,
            tta_variance=tta_var,
            mean_entropy=float(entropy.mean()),
            mean_confidence=float(confidence.mean()),
            mean_tta_variance=float(tta_var.mean()),
            prediction=prediction,
        ))

    return failure_maps
