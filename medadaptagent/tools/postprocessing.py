"""Post-processing tools: mask cleanup. Install postprocess_fn.

Cheap and often give free Dice on already-correct-ish masks. Can hurt
when the "right" mask legitimately has small holes or disconnected
components (bilateral structures, multi-focal lesions).
"""
from __future__ import annotations
import numpy as np
from .base import Tool, ToolContext, register


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component per non-background class.
    No-op on binary or ternary masks if the component count < 2."""
    try:
        from scipy.ndimage import label
    except ImportError:
        return mask
    out = np.zeros_like(mask)
    classes = [c for c in np.unique(mask) if c != 0]
    for c in classes:
        binary = (mask == c).astype(np.int32)
        lbl, n = label(binary)
        if n <= 1:
            out[binary > 0] = c
            continue
        sizes = [(lbl == i).sum() for i in range(1, n + 1)]
        keep = int(np.argmax(sizes)) + 1
        out[lbl == keep] = c
    return out


def _hole_fill(mask: np.ndarray) -> np.ndarray:
    """Fill small interior holes in each non-background class."""
    try:
        from scipy.ndimage import binary_fill_holes
    except ImportError:
        return mask
    out = mask.copy()
    for c in [c for c in np.unique(mask) if c != 0]:
        filled = binary_fill_holes(mask == c)
        out[filled & (mask == 0)] = c
    return out


def _small_comp_filter(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    try:
        from scipy.ndimage import label
    except ImportError:
        return mask
    out = mask.copy()
    for c in [c for c in np.unique(mask) if c != 0]:
        binary = (mask == c).astype(np.int32)
        lbl, n = label(binary)
        for i in range(1, n + 1):
            if (lbl == i).sum() < min_size:
                out[lbl == i] = 0
    return out


def _morph_open_close(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    try:
        from scipy.ndimage import binary_opening, binary_closing
    except ImportError:
        return mask
    out = np.zeros_like(mask)
    # Match structuring-element dimensionality to mask. 2D mask -> 2D
    # square; 3D mask -> 3D cube. Otherwise scipy raises on mismatch.
    ndim = mask.ndim
    struct = np.ones((radius * 2 + 1,) * ndim, dtype=bool)
    for c in [c for c in np.unique(mask) if c != 0]:
        m = mask == c
        m = binary_opening(m, structure=struct)
        m = binary_closing(m, structure=struct)
        out[m] = c
    return out


def _fn_largest_cc(ctx: ToolContext) -> ToolContext:
    prev = ctx.postprocess_fn
    ctx.postprocess_fn = lambda m: _largest_cc(prev(m))
    return ctx


def _fn_hole_fill(ctx: ToolContext) -> ToolContext:
    prev = ctx.postprocess_fn
    ctx.postprocess_fn = lambda m: _hole_fill(prev(m))
    return ctx


def _fn_small_comp_filter(ctx: ToolContext) -> ToolContext:
    notes = ctx.notes.get("small_comp_filter", {})
    ms = int(notes.get("min_size", 50))
    prev = ctx.postprocess_fn
    ctx.postprocess_fn = lambda m: _small_comp_filter(prev(m), ms)
    return ctx


def _fn_morph_open_close(ctx: ToolContext) -> ToolContext:
    notes = ctx.notes.get("morph_open_close", {})
    r = int(notes.get("radius", 2))
    prev = ctx.postprocess_fn
    ctx.postprocess_fn = lambda m: _morph_open_close(prev(m), r)
    return ctx


register(Tool(
    name="largest_connected_component", fn=_fn_largest_cc,
    category="postprocessing", requires=["pixel_predictions"],
    side_effects=["postprocess_fn"],
    profile=(
        "MECHANISM: connected-components labeling, retain the biggest blob per non-background class — fixes SHAPE artifacts (specks, fragments). Acts on output mask. Keep only the largest connected component per class. Great for "
        "single-organ segmentation (prostate, liver). Hurts on "
        "multi-focal anatomy (both lungs, both kidneys, multi-lesion)."),
))
register(Tool(
    name="hole_fill", fn=_fn_hole_fill,
    category="postprocessing", requires=["pixel_predictions"],
    side_effects=["postprocess_fn"],
    profile=(
        "MECHANISM: binary fill-holes per class on the output mask — fixes HOLE artifacts in solid structures. Fill small interior holes in each non-BG class. Cheap, usually "
        "free Dice on solid structures. Hurts only when the structure "
        "legitimately contains voids (ventricles, bowel lumen)."),
))
register(Tool(
    name="small_component_filter", fn=_fn_small_comp_filter,
    category="postprocessing", requires=["pixel_predictions"],
    side_effects=["postprocess_fn"],
    profile=(
        "MECHANISM: connected-components labeling, drop blobs below a size threshold — fixes SPECK artifacts. Drop components smaller than min_size pixels (default 50). "
        "Cleans small specks / false-positive splatters. Hurts if the "
        "target anatomy is genuinely small (sub-mm lesions)."),
))
register(Tool(
    name="morph_open_close", fn=_fn_morph_open_close,
    category="postprocessing", requires=["pixel_predictions"],
    side_effects=["postprocess_fn"],
    profile=(
        "MECHANISM: per-class binary opening (erode-then-dilate) + closing (dilate-then-erode) — smooths BOUNDARY artifacts on the output. Morphological open+close per class (radius=2). Smooths mask "
        "boundaries, removes single-pixel noise. Slight boundary "
        "softening; usually neutral-to-positive."),
))
