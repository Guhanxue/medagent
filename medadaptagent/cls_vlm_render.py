"""GradCAM-based VLM rendering for classification tasks.

Classification models don't produce spatial output, so we can't show
the VLM a mask overlay (the seg path). Instead, we use GradCAM to
visualize WHERE the model is looking when it produces its prediction.
The VLM then judges:

  1. is the attention on clinically-relevant anatomy, or on artifacts
     (border, label text, background)?
  2. is the predicted class plausible given the visible content?
  3. did adaptation move attention TOWARD or AWAY from meaningful
     regions compared to source?

This makes cls VLM judgment structurally analogous to seg's:
  seg          | cls
  raw image    | raw image
  source mask  | source GradCAM heatmap + predicted class caption
  current mask | current GradCAM heatmap + predicted class caption
"""
from __future__ import annotations
import base64
import io
from typing import List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False


def _pick_gradcam_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """Heuristic target-layer picker. Works for ResNet (use layer4)
    and timm ViTs (use last block's norm1 layer)."""
    # ResNet-style
    if hasattr(model, "layer4"):
        return model.layer4
    # timm ViT-style
    if hasattr(model, "blocks") and len(getattr(model, "blocks", [])) > 0:
        last_block = model.blocks[-1]
        if hasattr(last_block, "norm1"):
            return last_block.norm1
        return last_block
    # Fall back to last conv2d in the model
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def _reshape_transform_vit(tensor, height: int = 14, width: int = 14):
    """Reshape ViT's [B, N+1, C] tokens to [B, C, H, W] for GradCAM."""
    # Drop the cls token, reshape patch tokens to (H, W, C), permute to (C, H, W)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(-1))
    return result.permute(0, 3, 1, 2)


def _make_cam(model: nn.Module, target_layer: nn.Module,
                is_vit: bool = False) -> Optional[Any]:
    """Construct a GradCAM object. Returns None if unsupported."""
    if not _GRADCAM_AVAILABLE:
        return None
    try:
        if is_vit:
            cam = GradCAM(model=model, target_layers=[target_layer],
                            reshape_transform=_reshape_transform_vit)
        else:
            cam = GradCAM(model=model, target_layers=[target_layer])
        return cam
    except Exception:
        return None


def _array_to_b64(arr: np.ndarray, size: int = 256) -> str:
    """HWC uint8 image -> base64 PNG."""
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil = PILImage.fromarray(arr)
    if pil.size != (size, size):
        pil = pil.resize((size, size), PILImage.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_cls_attention(model: nn.Module,
                            image: np.ndarray,
                            input_tensor: torch.Tensor,
                            pred_class: int,
                            target_layer: Optional[nn.Module] = None,
                            is_vit: bool = False,
                            size: int = 224) -> Tuple[Optional[str], Optional[str]]:
    """Compute GradCAM heatmap for `pred_class` and overlay on raw image.

    Returns (overlay_b64, raw_b64) — both 224×224 PNG strings ready for
    a VLM prompt. If GradCAM isn't available (library missing) or model
    is incompatible (no Conv / no blocks), returns (None, raw_b64) so the
    caller can still render the raw + caption path.
    """
    # Raw always succeeds
    if image.dtype != np.uint8:
        if image.max() <= 1.0 + 1e-6:
            raw_arr = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            raw_arr = np.clip(image, 0, 255).astype(np.uint8)
    else:
        raw_arr = image
    if raw_arr.ndim == 2:
        raw_arr = np.stack([raw_arr] * 3, axis=-1)
    raw_pil = PILImage.fromarray(raw_arr).resize((size, size), PILImage.BILINEAR)
    raw_resized = np.array(raw_pil)
    raw_b64 = _array_to_b64(raw_resized, size=size)

    # Try GradCAM
    if target_layer is None:
        target_layer = _pick_gradcam_target_layer(model)
    cam_obj = _make_cam(model, target_layer, is_vit=is_vit) if target_layer is not None else None
    if cam_obj is None:
        return None, raw_b64
    try:
        targets = [ClassifierOutputTarget(int(pred_class))]
        # input_tensor expected to require grad-tape. GradCAM internally
        # handles backward.
        gray = cam_obj(input_tensor=input_tensor, targets=targets)[0]
        # show_cam_on_image expects raw in [0,1] float
        raw_float = raw_resized.astype(np.float32) / 255.0
        overlay = show_cam_on_image(raw_float, gray, use_rgb=True)
        return _array_to_b64(overlay, size=size), raw_b64
    except Exception as e:
        # Fall back to raw-only
        return None, raw_b64


# ----------------------------------------------------------------
# Per-iter VLM prompt for cls — analogous to seg's PER_ITER_VLM_PROMPT
# ----------------------------------------------------------------

CLS_VLM_PROMPT_V2 = """Compare the SOURCE model's classification with the CURRENT \
adapted model's classification on {k} target samples.

## Diagnosed shift mechanism (condition your vote on this)
{mechanism_hint}

## Task

Each sample has THREE inputs:

  1. RAW IMAGE — the actual target image. Use it to locate what is
     visible (anatomy, lesion, tissue type, etc.).

  2. SOURCE attention map + predicted class caption — where the SOURCE
     model is looking, and what it predicted. The attention map is a
     GradCAM heatmap overlaid on the raw image (red/hot = focus areas).

  3. CURRENT attention map + predicted class caption — same for the
     adapted model.

If an attention map says "(no attention map)", judge by class label
alone using your medical knowledge of the modality.

## Two per-sample judgments

1. **Relative** (CURRENT vs SOURCE): better | similar | worse.
   Vote based on:
   - Does CURRENT's predicted class better match the visible content?
   - Did CURRENT's attention move TOWARD clinically relevant regions
     (target anatomy, lesion) vs SOURCE's focus on artifacts (corners,
     labels, background)?
   - If both predictions are visually plausible and attention is
     comparable, vote "similar".
   - If CURRENT predicts something obviously wrong for the visible
     content while SOURCE was right, vote "worse".

2. **Absolute quality of CURRENT alone**: usable | broken.
   - "broken" — predicted class is inconsistent with what you see,
     OR attention is entirely on irrelevant regions (background,
     image border, frame artifacts), OR confidence is very low and
     the prediction is contradicted by visible content.
   - "usable" — predicted class is plausible and attention is on
     clinically relevant anatomy (even if the boundary is imperfect).

Return ONLY this JSON (no extra text):
```json
{{"per_sample":          ["better|similar|worse", ...],
  "per_sample_absolute": ["usable|broken", ...],
  "note": "<one-sentence summary>"}}
```"""


__all__ = ["render_cls_attention", "CLS_VLM_PROMPT_V2",
            "_pick_gradcam_target_layer"]
