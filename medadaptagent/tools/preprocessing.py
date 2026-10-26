"""Preprocessing tools: intensity / histogram / bias transforms on
target images. These install a `preprocess_fn` on the ToolContext that
is applied per-image at inference time. They do NOT modify the model.

Philosophy: preprocessing tools are cheap, well-understood, hard to
make worse — they're a safe first line of defense before any gradient
adaptation. Agent 3 should prefer these when Agent 1 reports a clean
intensity/contrast/bias gap with no geometric or semantic shift.
"""
from __future__ import annotations
import numpy as np
from .base import Tool, ToolContext, register


# --------------------------------------------------------------
# Internal helpers (match semantics to medadaptagent/adaptation/data_adaptation.py
# where possible; re-implement minimal versions here to avoid coupling)
# --------------------------------------------------------------

def _histogram_match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match histogram of `src` to the CDF of `ref`. Grayscale arrays."""
    s = src.ravel()
    r = ref.ravel()
    s_vals, s_idx, s_counts = np.unique(s, return_inverse=True,
                                         return_counts=True)
    r_vals, r_counts = np.unique(r, return_counts=True)
    s_cdf = np.cumsum(s_counts).astype(np.float64) / s.size
    r_cdf = np.cumsum(r_counts).astype(np.float64) / r.size
    interp = np.interp(s_cdf, r_cdf, r_vals)
    return interp[s_idx].reshape(src.shape).astype(src.dtype)


def _clahe(img: np.ndarray, clip_limit: float = 2.0,
             tile_grid: int = 8) -> np.ndarray:
    """CLAHE via OpenCV; fail explicitly when the dependency is unavailable."""
    try:
        import cv2
        if img.dtype != np.uint8:
            lo, hi = np.percentile(img, [1, 99])
            if hi - lo < 1e-6:
                return img
            img_u8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        else:
            img_u8 = img
        cl = cv2.createCLAHE(clipLimit=clip_limit,
                              tileGridSize=(tile_grid, tile_grid))
        if img_u8.ndim == 3:
            out = np.stack([cl.apply(img_u8[..., c])
                            for c in range(img_u8.shape[2])], axis=-1)
        else:
            out = cl.apply(img_u8)
        return out.astype(img.dtype)
    except ImportError:
        raise RuntimeError("CLAHE requires opencv-python-headless; install package dependencies") from None


# --------------------------------------------------------------
# Tool functions
# --------------------------------------------------------------

def _fn_histogram_match(ctx: ToolContext) -> ToolContext:
    """Match each target image's histogram to a reference computed from
    the current target set (robust self-reference). If a `ref_image`
    override is in ctx.notes, use that instead.

    3D handling: for a (D, H, W) volume, match the whole volume against
    the whole reference (single CDF over all voxels). For 2D color
    images (H, W, 3), match per-channel. Disambiguated by checking the
    last-axis size: small (<= 4) = color channels."""
    notes = ctx.notes.get("histogram_match", {})
    ref = notes.get("ref_image")
    if ref is None:
        # Use median-intensity image from the batch as reference
        means = [float(np.mean(x)) for x in ctx.target_images]
        ref_idx = int(np.argsort(means)[len(means) // 2])
        ref = ctx.target_images[ref_idx]
    prev_pre = ctx.preprocess_fn
    def new_pre(img):
        out = prev_pre(img)
        is_2d_color = (out.ndim == 3 and ref.ndim == 3
                          and out.shape[-1] <= 4 and ref.shape[-1] <= 4)
        if is_2d_color:
            chans = [_histogram_match_channel(out[..., c], ref[..., c])
                     for c in range(out.shape[2])]
            return np.stack(chans, axis=-1)
        return _histogram_match_channel(out, ref)
    ctx.preprocess_fn = new_pre
    return ctx


def _fn_clahe(ctx: ToolContext) -> ToolContext:
    """Apply CLAHE (contrast-limited adaptive histogram equalization)."""
    notes = ctx.notes.get("clahe", {})
    clip = float(notes.get("clip_limit", 2.0))
    tile = int(notes.get("tile_grid", 8))
    prev_pre = ctx.preprocess_fn
    ctx.preprocess_fn = lambda img: _clahe(prev_pre(img), clip, tile)
    return ctx


def _fn_percentile_clip(ctx: ToolContext) -> ToolContext:
    """Clip each image to [lo, hi] percentile, then linearly remap to
    [0, 1]. Robust to outliers (saturated pixels, hot voxels)."""
    notes = ctx.notes.get("percentile_clip", {})
    lo_p = float(notes.get("lo", 1.0))
    hi_p = float(notes.get("hi", 99.0))
    prev_pre = ctx.preprocess_fn
    def new_pre(img):
        x = prev_pre(img).astype(np.float32)
        lo, hi = np.percentile(x, [lo_p, hi_p])
        if hi - lo < 1e-6:
            return x
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    ctx.preprocess_fn = new_pre
    return ctx


def _fn_gamma_correct(ctx: ToolContext) -> ToolContext:
    """Gamma correction. If `gamma` is given, apply it; otherwise
    estimate gamma so the batch mean matches the provided `target_mean`
    (default 0.5 after min-max normalization)."""
    notes = ctx.notes.get("gamma_correct", {})
    gamma = notes.get("gamma")
    if gamma is None:
        target_mean = float(notes.get("target_mean", 0.5))
        # Estimate from batch
        all_vals = np.concatenate([x.astype(np.float32).ravel()
                                     for x in ctx.target_images[:20]])
        lo, hi = np.percentile(all_vals, [1, 99])
        if hi - lo < 1e-6:
            gamma = 1.0
        else:
            normed = np.clip((all_vals - lo) / (hi - lo), 0, 1)
            cur_mean = float(np.mean(normed))
            gamma = float(np.log(target_mean + 1e-6) /
                           np.log(cur_mean + 1e-6)) if cur_mean > 0 else 1.0
            gamma = float(np.clip(gamma, 0.25, 4.0))
    ctx.notes.setdefault("gamma_correct", {})["gamma_used"] = gamma
    prev_pre = ctx.preprocess_fn
    def new_pre(img):
        x = prev_pre(img).astype(np.float32)
        lo, hi = np.percentile(x, [1, 99])
        if hi - lo < 1e-6:
            return x
        normed = np.clip((x - lo) / (hi - lo), 0, 1)
        return np.power(normed, gamma)
    ctx.preprocess_fn = new_pre
    return ctx


def _fn_zscore_norm(ctx: ToolContext) -> ToolContext:
    """Per-image z-score: (x - mean) / std. Clip final to [-3, 3]."""
    prev_pre = ctx.preprocess_fn
    def new_pre(img):
        x = prev_pre(img).astype(np.float32)
        m, s = float(np.mean(x)), float(np.std(x))
        if s < 1e-6:
            return np.zeros_like(x)
        return np.clip((x - m) / s, -3.0, 3.0)
    ctx.preprocess_fn = new_pre
    return ctx


def _fn_n4_bias(ctx: ToolContext) -> ToolContext:
    """N4 bias field correction via SimpleITK. MRI-specific; no-op if
    SimpleITK isn't installed. Installs as a preprocess_fn."""
    try:
        import SimpleITK as sitk
    except ImportError:
        ctx.notes.setdefault("n4_bias", {})["skipped"] = "SimpleITK_missing"
        return ctx
    prev_pre = ctx.preprocess_fn
    def new_pre(img):
        x = prev_pre(img).astype(np.float32)
        # SimpleITK expects the image to have positive intensities
        lo = float(np.min(x))
        if lo <= 0:
            x = x - lo + 1e-3
        itk_img = sitk.GetImageFromArray(x)
        itk_img = sitk.Cast(itk_img, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(itk_img, 0, 1, 200)
        try:
            corr = sitk.N4BiasFieldCorrectionImageFilter()
            out = corr.Execute(itk_img, mask)
            return sitk.GetArrayFromImage(out).astype(np.float32)
        except Exception:
            return x
    ctx.preprocess_fn = new_pre
    return ctx


# --------------------------------------------------------------
# Registration
# --------------------------------------------------------------

register(Tool(
    name="histogram_match", fn=_fn_histogram_match,
    category="preprocessing", side_effects=["preprocess_fn"],
    profile=(
        "MECHANISM: per-image, remap each target image's intensity CDF to match a source-distribution reference image — changes the INPUT, not the model. Matches each target image's intensity histogram to a reference "
        "(defaults to the target batch's median-intensity image). Helps "
        "when source vs target differ mainly in intensity distribution "
        "(scanner vendor, contrast agent, gain). Safe; rarely hurts."),
))
register(Tool(
    name="clahe", fn=_fn_clahe,
    category="preprocessing", side_effects=["preprocess_fn"],
    supports_3d=False,
    profile=(
        "MECHANISM: locally rescales intensity histograms in image tiles to expand contrast — changes the INPUT, not the model. Contrast-limited adaptive histogram equalization. Helps on "
        "low-contrast targets (dark/washed-out scans). Can over-sharpen "
        "noise on already-clean images."),
))
register(Tool(
    name="percentile_clip", fn=_fn_percentile_clip,
    category="preprocessing", side_effects=["preprocess_fn"],
    profile=(
        "MECHANISM: drops top/bottom 1% intensity outliers and rescales the rest to [0,1] — changes the INPUT distribution, robust to saturated pixels. Clip intensities to [1%, 99%] percentile and remap to [0,1]. "
        "Robust normalization; mandatory when outliers / saturated "
        "pixels distort mean/std normalization. Near-zero risk."),
))
register(Tool(
    name="gamma_correct", fn=_fn_gamma_correct,
    category="preprocessing", side_effects=["preprocess_fn"],
    profile=(
        "MECHANISM: applies pixel-wise power transform x^gamma so the target batch mean matches an assumed source mean — changes the INPUT brightness curve. Auto-gamma to match target batch mean to source training mean. "
        "Helps on global brightness shifts. Small risk of over-compressing "
        "highlights on already-balanced targets."),
))
register(Tool(
    name="zscore_norm", fn=_fn_zscore_norm,
    category="preprocessing", side_effects=["preprocess_fn"],
    profile=(
        "MECHANISM: rescales each image to mean=0, std=1 — changes the INPUT scale; useful when source was z-score-trained. Per-image z-score normalization, clipped to ±3 sigma. Good "
        "baseline normalization when source was trained with z-score. "
        "Sensitive to large BG regions — may normalize to noise."),
))
register(Tool(
    name="n4_bias_correction", fn=_fn_n4_bias,
    category="preprocessing", side_effects=["preprocess_fn"],
    profile=(
        "MECHANISM: estimates and removes a smooth low-frequency multiplicative bias field from MRI images — corrects scanner-coil shading on the INPUT. N4 bias-field correction via SimpleITK. Classical MRI fix for "
        "intensity non-uniformity (coil shading). MRI-specific; no-op "
        "on CT or fundus. Slow (~1s/image) but usually pure win on MRI."),
    requires=[],
))
