"""Test-time augmentation tools: predict on multiple views, aggregate.

Install an `inference_fn` on the ToolContext that replaces the default
single forward pass. Composable: if `inference_fn` is already set by
a previous tool, we wrap it.
"""
from __future__ import annotations
from typing import Callable
import torch
import torch.nn.functional as F
from .base import Tool, ToolContext, register


def _compose_inference(prev: Callable, new: Callable) -> Callable:
    """Default composition: `new` replaces `prev` entirely. Caller can
    write a wrapper if they want to chain multi-view augmentation."""
    return new


def _fn_flip_tta(ctx: ToolContext) -> ToolContext:
    """Horizontal-flip TTA. Predicts on the image and its horizontal
    flip, averages softmax probabilities, returns the averaged logits.
    Handles arbitrary preprocess_fn via ctx (it's run upstream)."""
    prev_inf = ctx.inference_fn

    def inf(model, x):
        base = prev_inf(model, x) if prev_inf is not None else model(x)
        x_flip = torch.flip(x, dims=[3])
        flip = prev_inf(model, x_flip) if prev_inf is not None else model(x_flip)
        # average probabilities, un-flip the flipped branch
        p_base = F.softmax(base, dim=1)
        p_flip = F.softmax(torch.flip(flip, dims=[3]), dim=1)
        p_avg = (p_base + p_flip) / 2
        return torch.log(p_avg + 1e-10)  # log-probs; argmax equivalent
    ctx.inference_fn = inf
    return ctx


def _fn_multi_scale_tta(ctx: ToolContext) -> ToolContext:
    """Multi-scale TTA: predict at 0.8x, 1.0x, 1.2x, upsample back, avg.
    Useful for scale invariance when pixel spacing differs or lesion
    size distribution shifts. Cost: 3x inference time. Dispatches
    bilinear (4D) vs trilinear (5D)."""
    prev_inf = ctx.inference_fn
    notes = ctx.notes.get("multi_scale_tta", {})
    scales = notes.get("scales", [0.8, 1.0, 1.2])

    def inf(model, x):
        if x.ndim == 5:
            d, h, w = x.shape[-3:]
            mode = "trilinear"
            spatial = lambda s: (int(d * s), int(h * s), int(w * s))
            orig = (d, h, w)
        else:
            h, w = x.shape[-2:]
            mode = "bilinear"
            spatial = lambda s: (int(h * s), int(w * s))
            orig = (h, w)
        preds = []
        for s in scales:
            xs = F.interpolate(x, size=spatial(s),
                                mode=mode, align_corners=False)
            out = prev_inf(model, xs) if prev_inf is not None else model(xs)
            out = F.interpolate(out, size=orig,
                                 mode=mode, align_corners=False)
            preds.append(F.softmax(out, dim=1))
        p_avg = torch.stack(preds, dim=0).mean(dim=0)
        return torch.log(p_avg + 1e-10)
    ctx.inference_fn = inf
    return ctx


register(Tool(
    name="flip_tta", fn=_fn_flip_tta,
    category="tta_aug", side_effects=["inference_fn"],
    profile=(
        "MECHANISM: averages predictions across input augmentations — stabilizes INFERENCE, does not change weights. Predict on original + horizontal-flip, average softmax. "
        "Improves symmetric-anatomy segmentation (heart, prostate, "
        "lungs). Doubles inference cost. Cannot hurt if model is "
        "well-behaved, but wastes compute on strongly-asymmetric "
        "structures."),
))
register(Tool(
    name="multi_scale_tta", fn=_fn_multi_scale_tta,
    category="tta_aug", side_effects=["inference_fn"],
    profile=(
        "MECHANISM: averages predictions at multiple input scales — stabilizes INFERENCE under scale uncertainty. Predict at 0.8x, 1.0x, 1.2x scales; upsample to original; "
        "average softmax. Helps when target pixel spacing / structure "
        "size differs from training. ~3x inference time."),
))
