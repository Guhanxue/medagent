"""
Vision utilities for multimodal LLM agents.

Converts numpy arrays to base64 images for GPT-4/5 multimodal input.
Creates visualization overlays (predictions on images) for assessment.
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def array_to_b64(arr: np.ndarray, size: int = 256) -> str:
    """Convert numpy array to base64 PNG string for LLM input."""
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1]:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # Normalize to 0-255
    if arr.max() <= 1.0:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    if img.size != (size, size):
        img = img.resize((size, size))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def prediction_overlay(image: np.ndarray, prediction: np.ndarray,
                       alpha: float = 0.4, size: int = 256) -> str:
    """Create image with colored prediction overlay, return as base64.

    Colors: class 1 = red, class 2 = green, class 3 = blue, class 4 = yellow
    """
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.max() <= 1.0:
        base = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        base = image.clip(0, 255).astype(np.uint8)

    # Create colored overlay
    colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [255, 255, 0]}
    overlay = base.copy().astype(float)
    for cls_id, color in colors.items():
        mask = prediction == cls_id
        if mask.any():
            for c in range(3):
                overlay[:, :, c][mask] = (
                    (1 - alpha) * base[:, :, c][mask] + alpha * color[c])

    result = overlay.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(result)
    if img.size != (size, size):
        img = img.resize((size, size))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def get_model_predictions(model: nn.Module, images: List[np.ndarray],
                          device: str, max_samples: int = 5
                          ) -> List[np.ndarray]:
    """Run model on images and return prediction maps."""
    model.eval()
    preds = []
    with torch.no_grad():
        for img in images[:max_samples]:
            if img.ndim == 3 and img.shape[2] in (1, 3):
                t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            elif img.ndim == 2:
                t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
            else:
                t = torch.from_numpy(img.astype(np.float32))
            pred = model(t.unsqueeze(0).to(device)).argmax(dim=1).cpu().numpy()[0]
            preds.append(pred)
    return preds


def prepare_sample_images(images: List[np.ndarray], n: int = 3) -> List[str]:
    """Select n representative images and convert to base64."""
    # Pick evenly spaced samples
    indices = np.linspace(0, len(images) - 1, n, dtype=int)
    return [array_to_b64(images[i]) for i in indices]


def prepare_prediction_samples(images: List[np.ndarray],
                               model: nn.Module, device: str,
                               n: int = 3, task=None) -> List[Tuple[str, str]]:
    """Prepare n (image, prediction_overlay) pairs for visual assessment.

    If a `task` adapter is provided AND it implements
    `render_prediction_panel(model, image, device)`, that method is called
    per sample to render task-appropriate panels (e.g., 3D mid-slice +
    class caption for cls_3d, 3-view montage with mask overlay for
    seg_3d). Falls back to the legacy 2D image+overlay path otherwise."""
    indices = np.linspace(0, len(images) - 1, min(n, len(images)), dtype=int)
    samples = [images[i] for i in indices]

    if task is not None and hasattr(task, "render_prediction_panel"):
        pairs = []
        for img in samples:
            try:
                raw_b64, overlay_b64 = task.render_prediction_panel(
                    model, img, device)
                pairs.append((raw_b64, overlay_b64))
            except Exception as _e:
                # Soft-fall back to legacy if task method errors
                preds = get_model_predictions(model, [img], device,
                                                 max_samples=1)
                pairs.append((array_to_b64(img),
                                prediction_overlay(img, preds[0])))
        return pairs

    # Legacy 2D path
    preds = get_model_predictions(model, samples, device, max_samples=n)
    pairs = []
    for img, pred in zip(samples, preds):
        img_b64 = array_to_b64(img)
        overlay_b64 = prediction_overlay(img, pred)
        pairs.append((img_b64, overlay_b64))
    return pairs


def build_multimodal_message(text: str,
                             images: List[str] = None,
                             image_labels: List[str] = None) -> list:
    """Build a multimodal message for the LLM.

    Args:
        text: The text prompt
        images: List of base64 image strings
        image_labels: Optional labels like "Source Image 1", "Target with Prediction"

    Returns:
        Content list for OpenAI/Anthropic multimodal API
    """
    content = [{"type": "text", "text": text}]

    if images:
        for i, img_b64 in enumerate(images):
            if image_labels and i < len(image_labels):
                content.append({"type": "text", "text": f"\n**{image_labels[i]}:**"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

    return content
