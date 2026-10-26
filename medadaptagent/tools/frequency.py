"""Frequency-domain tools: FDA (Fourier Domain Adaptation).

Wraps the existing fda_source_to_target implementation in
medadaptagent/adaptation/advanced_tta.py. FDA swaps the low-frequency
magnitude band between a source reference image and each target image
— style transfer without a network, cheap and often pure-win on
scanner-vendor / stain / contrast-agent gaps.
"""
from __future__ import annotations
import numpy as np
from .base import Tool, ToolContext, register


def _fda_core(source_ref: np.ndarray, target: np.ndarray,
                beta: float = 0.05) -> np.ndarray:
    """Low-frequency magnitude swap from source_ref into target. Returns
    a target-style image with source's low-freq magnitude. Operates on
    2D grayscale (H,W), 2D-color (H,W,C — per-channel), and 3D volumes
    (D,H,W — single fftn over the whole volume)."""
    def _one(s, t):
        """N-dim FFT swap. Replaces the low-frequency hyper-rectangle
        (size = beta * min(shape) / 2 along each axis) of T's magnitude
        with S's, preserves T's phase."""
        S = np.fft.fftn(s); T = np.fft.fftn(t)
        S_mag, T_phase = np.abs(S), np.angle(T)
        T_mag = np.abs(T)
        # Build a centered low-freq slice for each axis
        b = max(1, int(min(S.shape) * beta / 2))
        T_mag_shift = np.fft.fftshift(T_mag)
        S_mag_shift = np.fft.fftshift(S_mag)
        slices = tuple(slice(d // 2 - b, d // 2 + b) for d in S.shape)
        T_mag_shift[slices] = S_mag_shift[slices]
        T_mag_new = np.fft.ifftshift(T_mag_shift)
        out_complex = T_mag_new * np.exp(1j * T_phase)
        out = np.real(np.fft.ifftn(out_complex))
        lo, hi = float(out.min()), float(out.max())
        return (out - lo) / (hi - lo + 1e-8)

    # 2D-color: per-channel (last axis small, e.g. RGB)
    if source_ref.ndim == 3 and source_ref.shape[-1] <= 4:
        return np.stack([_one(source_ref[..., c], target[..., c])
                         for c in range(source_ref.shape[2])], axis=-1)
    return _one(source_ref, target)


def _fn_fda_warmup(ctx: ToolContext) -> ToolContext:
    """Install an FDA preprocess: each target image is reshaped to adopt
    the low-freq magnitude of a source reference. Reference is taken
    from `ctx.notes['fda_warmup']['source_ref']` if available; otherwise
    falls back to the first target image (degenerate no-op — warn).

    Handles 2D images and 3D volumes. Resizes source_ref to match target
    shape via scipy.ndimage.zoom (N-D) instead of PIL."""
    notes = ctx.notes.get("fda_warmup", {})
    source_ref = notes.get("source_ref")
    beta = float(notes.get("beta", 0.05))
    if source_ref is None:
        ctx.notes.setdefault("fda_warmup", {})["warning"] = \
            "no_source_ref_provided; FDA is a no-op"
        return ctx
    prev_pre = ctx.preprocess_fn

    def _resize_ref_to(sref: np.ndarray, target_shape: tuple) -> np.ndarray:
        """N-D resize of source_ref to target_shape. Uses scipy.ndimage.zoom
        which handles 2D and 3D natively. Returns float32."""
        sref = sref.astype(np.float32)
        if sref.max() > 1.5:
            sref = sref / 255.0
        if sref.shape == target_shape:
            return sref
        from scipy.ndimage import zoom
        # Per-axis zoom factor. For 2D-color (last dim small), don't
        # scale the channel axis.
        if (sref.ndim == 3 and sref.shape[-1] <= 4
                and len(target_shape) == 3 and target_shape[-1] == sref.shape[-1]):
            factors = tuple(t / s for s, t in zip(sref.shape[:2], target_shape[:2])) + (1.0,)
        else:
            factors = tuple(t / s for s, t in zip(sref.shape, target_shape))
        return zoom(sref, factors, order=1).astype(np.float32)

    def new_pre(img):
        x = prev_pre(img).astype(np.float32)
        xf = x / 255.0 if x.max() > 1.5 else x
        sref = _resize_ref_to(source_ref, xf.shape)
        return _fda_core(sref, xf, beta)

    ctx.preprocess_fn = new_pre
    return ctx


register(Tool(
    name="fda_warmup", fn=_fn_fda_warmup,
    category="frequency", side_effects=["preprocess_fn"],
    profile=(
        "MECHANISM: replaces the low-frequency magnitude band of each target image's FFT with that of a source reference, preserving target phase — transfers SOURCE STYLE onto target structure. Fourier Domain Adaptation: swap low-frequency magnitude band "
        "(beta=0.05) from a source reference into each target image. "
        "Preserves target structure while adopting source style. Good "
        "for scanner-vendor / contrast-agent / stain gaps. Requires a "
        "source_ref image (caller-supplied) — no-op otherwise."),
))
