"""Spatial / geometric tools: crop, resize, flip detection.

Install preprocess_fn on the ToolContext. Do not modify the model.
"""
from __future__ import annotations
import numpy as np
from PIL import Image as PILImage
from .base import Tool, ToolContext, register


def _fn_crop_center(ctx: ToolContext) -> ToolContext:
    """Center-crop each target image to a fraction of its volume.
    Works on 2D (H, W), 2D-color (H, W, C), and 3D (D, H, W). Crops
    symmetrically along each spatial axis; for 2D-color, the last
    axis is left untouched (channels).

    Useful when target has wider FOV than source training — e.g.,
    full-body 3D CT vs abdomen-only training, or fundus scans with
    excess black background."""
    notes = ctx.notes.get("crop_center", {})
    frac = float(notes.get("fraction", 0.75))
    frac = float(np.clip(frac, 0.25, 1.0))
    prev_pre = ctx.preprocess_fn

    def new_pre(img):
        x = prev_pre(img)
        # 2D-color image: skip last axis (channels)
        if x.ndim == 3 and x.shape[-1] <= 4:
            spatial = x.shape[:2]
            slices = tuple(slice((d - int(d * frac)) // 2,
                                   (d - int(d * frac)) // 2 + int(d * frac))
                            for d in spatial) + (slice(None),)
        else:
            slices = tuple(slice((d - int(d * frac)) // 2,
                                   (d - int(d * frac)) // 2 + int(d * frac))
                            for d in x.shape)
        return x[slices]

    ctx.preprocess_fn = new_pre
    return ctx


def _fn_resize_match(ctx: ToolContext) -> ToolContext:
    """Resize each target image to a target H×W before inference.
    Defaults to 256×256 (common source training size)."""
    notes = ctx.notes.get("resize_match", {})
    size = int(notes.get("size", 256))
    prev_pre = ctx.preprocess_fn
    def new_pre(img):
        x = prev_pre(img)
        is_float = x.dtype.kind == "f" and x.max() <= 1.0 + 1e-6
        arr = (x * 255).astype(np.uint8) if is_float else x.astype(np.uint8)
        pil = PILImage.fromarray(arr)
        pil = pil.resize((size, size), PILImage.BILINEAR)
        out = np.array(pil)
        return out.astype(np.float32) / 255.0 if is_float else out
    ctx.preprocess_fn = new_pre
    return ctx


def _fn_flip_detect(ctx: ToolContext) -> ToolContext:
    """Detect horizontal flip by predicting confidence on a sample of
    target images both as-is and flipped; keep the flip that maximizes
    mean max-probability (i.e., confidence). If flipping increases
    confidence meaningfully (>5%), install a preprocess that flips
    every image. Uses the current model state at call time."""
    import torch, torch.nn.functional as F
    from PIL import Image as PILImage
    prev_pre = ctx.preprocess_fn

    def _to_tensor(img):
        x = prev_pre(img)
        pil = PILImage.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)
                                  if x.max() <= 1.0 else x.astype(np.uint8))
        pil = pil.resize((256, 256), PILImage.BILINEAR)
        arr = np.array(pil).astype(np.float32) / 255.0
        if arr.ndim == 2:
            t = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
        else:
            t = torch.from_numpy(arr).permute(2, 0, 1)
        return t.unsqueeze(0).to(ctx.device)

    model = ctx.current_model
    model.eval()
    conf_raw, conf_flip = [], []
    with torch.no_grad():
        for img in ctx.target_images[:min(10, len(ctx.target_images))]:
            t = _to_tensor(img)
            p = F.softmax(model(t), dim=1)
            conf_raw.append(float(p.max(dim=1).values.mean()))
            t_flip = torch.flip(t, dims=[3])
            p_flip = F.softmax(model(t_flip), dim=1)
            conf_flip.append(float(p_flip.max(dim=1).values.mean()))
    c_raw = float(np.mean(conf_raw))
    c_flip = float(np.mean(conf_flip))
    ctx.notes.setdefault("flip_detect", {}).update(
        {"conf_raw": c_raw, "conf_flip": c_flip})
    if c_flip > c_raw + 0.05:
        ctx.notes["flip_detect"]["applied"] = "horizontal_flip"
        ctx.preprocess_fn = lambda img: np.ascontiguousarray(
            prev_pre(img)[:, ::-1] if prev_pre(img).ndim == 2
            else prev_pre(img)[:, ::-1, :])
    else:
        ctx.notes["flip_detect"]["applied"] = "none"
    return ctx


register(Tool(
    name="crop_center", fn=_fn_crop_center,
    category="spatial", side_effects=["preprocess_fn"],
    profile=(
        "MECHANISM: removes peripheral image content via center crop — corrects FOV mismatch on the INPUT. Center-crop each target image to 75% of area (tunable). Use "
        "when target's FOV is wider than source (more background/neighbouring "
        "tissue visible). Near-zero risk; hurts only if target structure "
        "sits at the periphery."),
))
register(Tool(
    name="resize_match", fn=_fn_resize_match,
    category="spatial", side_effects=["preprocess_fn"],
    supports_3d=False,
    profile=(
        "MECHANISM: rescales target to the source training resolution — corrects pixel-spacing mismatch on the INPUT. Resize target to 256x256 (or supplied size). Corrects pixel-spacing "
        "/ resolution mismatch. Usually redundant with default resize; "
        "worth applying when target is very high- or low-resolution."),
))
register(Tool(
    name="flip_detect", fn=_fn_flip_detect,
    category="spatial", side_effects=["preprocess_fn"],
    supports_3d=False,
    profile=(
        "MECHANISM: evaluates source confidence on raw vs flipped target; if flip raises confidence, installs a flip preprocess — corrects orientation mismatch on the INPUT. Runs the source model on first 10 target images both as-is and "
        "horizontally-flipped; keeps the orientation with higher mean "
        "confidence. Use when target may be mirrored relative to source "
        "(e.g., different DICOM axis convention)."),
))
