"""Label-free (K=0) adaptation tools for the 2D CLASSIFICATION adapter
(harness/cls_adapter.py): INPUT-SPACE color/intensity tools PLUS two
TRANSFORMER-NATIVE norm/classifier-space levers (t3a, norm_stat_tent_safe) that
close the "no label-free norm lever for a ViT" gap (bn_reset/tent are no-ops on
LayerNorm; plain norm_stat_tent collapses the ViT).

WHY THIS MODULE EXISTS
----------------------
The cls adapter previously exposed only `bn_reset` / `tent` as label-free moves
(plus the finetune ladder, which needs labels). But the cls shifts in the suite
are COLOR / ILLUMINATION shifts:
  * ham2isic       : cross-source dermoscopy (HAM10000 -> ISIC2019) — different
                     scanners / color calibration / vignetting.
  * aptos2eyepacs  : cross-camera diabetic-retinopathy fundus (APTOS -> EyePACS)
                     — different fundus cameras, illumination, color cast.
  * eyepacs2aptos  : the reverse cross-camera DR shift.
bn_reset/tent only touch BatchNorm statistics; they do NOT fix a color cast or an
illumination / histogram gap. The on-target lever is INPUT-SPACE color/histogram/
FDA matching of the target images toward the SOURCE domain. The 2D primitives for
this already exist in medadaptagent (data_adaptation.py / tools/preprocessing.py /
adaptation/advanced_tta.py) but were never wired as SOURCE-REFERENCED label-free
candidates in the cls toolbox. (The registry-bridged `histogram_match`/`clahe`/
`gamma` use a TARGET self-reference, not the source domain — so they cannot do
real cross-domain color matching.)

DESIGN — wrapper-model candidates (mirrors harness/labelfree_3d_tools.py)
------------------------------------------------------------------------
Every input-space tool produces a CANDIDATE scored by the EXISTING cls machinery
(`_conf_stats`, `_metric`, `_predict_probs`, safety_check) WITHOUT touching the
scoring path. We wrap the source model in a thin `nn.Module` whose `forward(t)`
first transforms the **ImageNet-normalized [B,3,224,224] tensor** that
`_cls_tensor` feeds, then calls the source model. The transform de-normalizes the
tensor back to RGB [0,1], applies the color/intensity op, re-normalizes, and runs
the base model. Because the transform lives inside forward(), the whole
`_cls_tensor -> model -> softmax -> AUROC/conf` chain runs unchanged and
apples-to-apples with the source reference (same scoring code, same AUROC fn).

Operating in the model's own input space is deliberate: the SOURCE reference
(histogram CDF / mean / std / amplitude spectrum) is built by pushing a handful of
unlabeled source images through the SAME `_cls_tensor` and de-normalizing — so
"make the target look source-like" is a like-for-like match in the exact space the
first conv consumes.

THE TOOLS (all RGB, 3-channel; per-channel where the op is color-sensitive)
---------------------------------------------------------------------------
 1 histogram_match_src   : per-channel CDF transport of each target image to the
                           SOURCE per-channel reference CDF (color + contrast).
 2 match_intensity_stats : per-channel mean/std affine so target channel stats
                           match the SOURCE channel mean/std (linear color cast fix).
 3 fda_src               : Fourier Domain Adaptation — replace the target's
                           low-frequency amplitude band with the SOURCE reference
                           amplitude (per channel), keep target phase (content).
 4 gamma_src             : auto-gamma x^g per image so the luminance mean matches
                           the SOURCE luminance mean (global brightness curve).
 5 clahe                 : CLAHE local contrast equalization on luminance (no ref;
                           contrast-only, source-agnostic).
 6 percentile_normalize  : per-image robust [1,99]-percentile clip+rescale (no ref).
 7 zscore_to_src         : per-channel z-score then rescale to SOURCE channel
                           mean/std (a normalization-style intensity match).

Reference-needing tools (histogram_match_src / match_intensity_stats / fda_src /
gamma_src / zscore_to_src) take a precomputed `ref` from `build_source_ref_2d`.
The two ref-free tools (clahe / percentile_normalize) take no ref.

Reuses the data_adaptation.py / preprocessing.py / advanced_tta.py math (CDF
transport, mean/std affine, FDA low-freq amplitude swap, auto-gamma) — not
reinvented; ported to operate per-channel on the de-normalized RGB tensor.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet stats — must match medadaptagent.tasks.classification._IMAGENET_*.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MEAN_T = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)


# =====================================================================
# Source reference (label-free): a few unlabeled source-domain images,
# pushed through the SAME _cls_tensor and de-normalized to RGB [0,1],
# reduced to the per-channel stats the color tools need.
# =====================================================================
# FOREGROUND mask: fundus / dermoscopy images carry a large near-black border that
# DOMINATES a whole-image histogram. Matching the whole image to the source then pulls
# everything toward the source background level and crushes lesion contrast (Phase-A
# baseline: full transport -> AUROC collapse to ~0.56 on ham2isic). So ref stats AND the
# per-image transport are computed on FOREGROUND pixels (luminance > _BG_EPS) only —
# mirrors the 3D module's background-exclusion fix.
_BG_EPS = 0.06


def _fg_mask(rgb):
    """Luminance > _BG_EPS foreground mask for a [3,H,W] RGB array."""
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    m = lum > _BG_EPS
    return m if m.sum() > 0.02 * m.size else np.ones_like(m)


def build_source_ref_2d(to_tensor, source_imgs, device, n=24, bins=256, fda_beta=0.01):
    """Build a SOURCE reference from up to `n` unlabeled source images, in the
    model's de-normalized RGB [0,1] input space (uses the adapter's own
    `to_tensor` = _cls_tensor, then de-normalizes). FOREGROUND-only stats (the
    near-black border is excluded). Returns a dict with:
      ref_cdf[c]/ref_vals : per-channel FOREGROUND [0,1] CDF for CDF transport
      ref_mean[c]/ref_std[c] : per-channel FOREGROUND mean/std (affine + zscore)
      ref_lum_mean        : FOREGROUND luminance mean (for auto-gamma)
      ref_amp             : per-channel mean centered amplitude spectrum (FDA)
    All in [0,1] RGB so it is comparable to the de-normalized target tensor the
    candidate transforms see at inference."""
    if not source_imgs:
        return None
    idx = np.linspace(0, len(source_imgs) - 1, min(n, len(source_imgs)), dtype=int)
    per_c = [[] for _ in range(3)]      # foreground pixel pools per channel for CDF
    means = np.zeros(3, np.float64); stds = np.zeros(3, np.float64); cnt = 0
    lab_means = np.zeros(3, np.float64); lab_stds = np.zeros(3, np.float64)  # LAB ref (Reinhard)
    amps = []
    lum_means = []
    for i in idx:
        t = to_tensor(source_imgs[int(i)], device)       # [1,3,224,224] ImageNet-norm
        rgb = _denorm(t)[0].detach().cpu().numpy().astype(np.float32)  # [3,H,W] in [0,1]
        fg = _fg_mask(rgb)
        for c in range(3):
            ch = rgb[c][fg].ravel()                      # FOREGROUND pixels only
            if ch.size == 0:
                ch = rgb[c].ravel()
            if ch.size > 5000:                           # subsample so the CDF stays light
                ch = ch[np.linspace(0, ch.size - 1, 5000, dtype=int)]
            per_c[c].append(ch)
        for c in range(3):
            fgc = rgb[c][fg]
            means[c] += float(fgc.mean()); stds[c] += float(fgc.std())
        lab = _rgb_to_lab(np.clip(rgb, 0, 1))            # FOREGROUND LAB stats (Reinhard)
        for c in range(3):
            fgl = lab[c][fg]
            lab_means[c] += float(fgl.mean()); lab_stds[c] += float(fgl.std())
        cnt += 1
        amps.append(_amp_spectrum(rgb))                  # [3,H,W]
        lum_means.append(float((0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])[fg].mean()))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ref_vals = 0.5 * (edges[:-1] + edges[1:])
    ref_cdf = []
    for c in range(3):
        allc = np.concatenate(per_c[c])
        h, _ = np.histogram(allc, bins=bins, range=(0.0, 1.0))
        ref_cdf.append(np.cumsum(h) / (h.sum() + 1e-8))
    return {"ref_vals": ref_vals, "ref_cdf": ref_cdf, "bins": bins,
            "ref_mean": (means / max(cnt, 1)).astype(np.float32),
            "ref_std": (stds / max(cnt, 1)).astype(np.float32),
            "lab_mean": (lab_means / max(cnt, 1)).astype(np.float32),  # Reinhard LAB ref
            "lab_std": (lab_stds / max(cnt, 1)).astype(np.float32),
            "ref_lum_mean": float(np.mean(lum_means)),
            "ref_amp": np.mean(amps, axis=0).astype(np.float32),
            "fda_beta": fda_beta}


def _denorm(t):
    """ImageNet-normalized [B,3,H,W] tensor -> RGB [0,1] tensor."""
    return (t * _STD_T.to(t.device) + _MEAN_T.to(t.device)).clamp(0, 1)


def _renorm(rgb):
    """RGB [0,1] [B,3,H,W] tensor -> ImageNet-normalized tensor."""
    return (rgb - _MEAN_T.to(rgb.device)) / _STD_T.to(rgb.device)


def _amp_spectrum(rgb):
    """Per-channel centered amplitude spectrum of a [3,H,W] RGB array (for FDA)."""
    out = np.empty_like(rgb)
    for c in range(rgb.shape[0]):
        out[c] = np.abs(np.fft.fftshift(np.fft.fft2(rgb[c]))).astype(np.float32)
    return out


# =====================================================================
# Per-channel RGB [0,1] transforms (numpy), ported from data_adaptation.py /
# preprocessing.py / advanced_tta.py. Each takes a [3,H,W] array -> [3,H,W].
# =====================================================================
def _hist_match_np(rgb, ref, blend=0.5):
    """Per-channel CDF-transport histogram match of a [3,H,W] RGB image to the SOURCE
    per-channel FOREGROUND ref CDF (data_adaptation.histogram_match_channel math), but
    (a) the per-image CDF is built on FOREGROUND pixels only and (b) the result is
    BLENDED `blend` with the original instead of a full swap. Both fixes mirror the 3D
    module: a full whole-image transport is dominated by the near-black border and
    collapses lesion contrast (Phase-A baseline)."""
    bins = ref["bins"]; ref_vals = ref["ref_vals"]
    fg = _fg_mask(rgb)
    out = np.array(rgb, dtype=np.float32, copy=True)
    for c in range(3):
        fgv = np.clip(rgb[c][fg].ravel(), 0, 1)
        if fgv.size == 0:
            continue
        h, edges = np.histogram(fgv, bins=bins, range=(0.0, 1.0))
        src_cdf = np.cumsum(h) / (h.sum() + 1e-8)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mapped_centers = np.interp(src_cdf, ref["ref_cdf"][c], ref_vals)
        matched = np.interp(rgb[c], centers, mapped_centers).reshape(rgb[c].shape)
        out[c][fg] = (1 - blend) * rgb[c][fg] + blend * matched[fg]   # FG only, blended
    return np.clip(out, 0, 1).astype(np.float32)


def _match_stats_np(rgb, ref, blend=0.6):
    """Per-channel mean/std affine so each channel's FOREGROUND stats match the SOURCE
    FOREGROUND channel mean/std (data_adaptation.match_intensity_stats math), BLENDED
    `blend` with the original and applied on FOREGROUND pixels only (the border stays
    put). Linear color-cast / illumination correction."""
    fg = _fg_mask(rgb)
    out = np.array(rgb, dtype=np.float32, copy=True)
    for c in range(3):
        fgc = rgb[c][fg]
        m = float(fgc.mean()); s = float(fgc.std())
        s = s if s > 1e-6 else 1.0
        rescaled = (rgb[c][fg] - m) / s * float(ref["ref_std"][c]) + float(ref["ref_mean"][c])
        out[c][fg] = (1 - blend) * rgb[c][fg] + blend * rescaled
    return np.clip(out, 0, 1).astype(np.float32)


def _zscore_to_src_np(rgb, ref):
    """Per-channel FOREGROUND z-score then rescale toward SOURCE channel mean/std, with
    a stronger pull than match_intensity_stats (standardize-first family,
    preprocessing.zscore)."""
    return _match_stats_np(rgb, ref, blend=0.8)


def _fda_np(rgb, ref, beta, blend=0.5):
    """Per-channel FDA: BLEND the target's low-frequency amplitude band toward the SOURCE
    reference amplitude (keep target phase = content). A blend, not a hard swap — a hard
    full swap over a wide band destroys the image (mirrors the 3D module). target<-source
    direction (make the TARGET look source-like)."""
    ra = ref["ref_amp"]
    out = np.empty_like(rgb)
    for c in range(3):
        v = rgb[c].astype(np.float32)
        rac = ra[c] if ra[c].shape == v.shape else _match_shape(ra[c], v.shape)
        fft = np.fft.fftshift(np.fft.fft2(v))
        amp = np.abs(fft); phase = np.angle(fft)
        h, w = v.shape; cy, cx = h // 2, w // 2
        b = max(1, int(min(h, w) * beta))
        mixed = amp.copy()
        sl = (slice(cy - b, cy + b), slice(cx - b, cx + b))
        mixed[sl] = (1 - blend) * amp[sl] + blend * rac[sl]   # blend low-freq amplitude
        rec = np.real(np.fft.ifft2(np.fft.ifftshift(mixed * np.exp(1j * phase))))
        out[c] = rec
    return np.clip(out, 0, 1).astype(np.float32)


def _match_shape(a, shape):
    out = np.zeros(shape, dtype=a.dtype)
    sl_o, sl_a = [], []
    for ao, so in zip(a.shape, shape):
        n = min(ao, so); oa = (ao - n) // 2; oo = (so - n) // 2
        sl_a.append(slice(oa, oa + n)); sl_o.append(slice(oo, oo + n))
    out[tuple(sl_o)] = a[tuple(sl_a)]
    return out


def _gamma_src_np(rgb, ref_lum_mean):
    """Auto-gamma x^g so the luminance mean matches the SOURCE luminance mean
    (preprocessing._fn_gamma_correct math). Applied to all channels equally."""
    lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
    m = float(np.clip(lum.mean(), 1e-4, 1 - 1e-4))
    rm = float(np.clip(ref_lum_mean, 1e-4, 1 - 1e-4))
    g = float(np.clip(np.log(rm + 1e-6) / np.log(m + 1e-6), 0.25, 4.0))
    return np.clip(rgb, 0, 1) ** g


def _clahe_np(rgb, clip_limit=0.01, kernel_frac=0.125):
    """CLAHE on luminance via skimage.exposure.equalize_adapthist (cv2 may be
    absent), re-applied as a per-pixel gain to all channels so color is preserved.
    Source-agnostic local-contrast equalization (preprocessing._clahe family)."""
    from skimage.exposure import equalize_adapthist
    lum = np.clip(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2], 0, 1).astype(np.float64)
    ks = max(8, int(min(lum.shape) * kernel_frac))
    try:
        eq = equalize_adapthist(lum, kernel_size=ks, clip_limit=clip_limit, nbins=256)
    except Exception:
        return rgb
    gain = (eq + 1e-4) / (lum + 1e-4)
    out = np.clip(rgb * gain[None, :, :], 0, 1).astype(np.float32)
    return out


def _percentile_np(rgb, lo=1.0, hi=99.0):
    """Per-image robust [lo,hi]-percentile clip + rescale to [0,1], per channel
    (data_adaptation.percentile_normalize / preprocessing._fn_percentile_clip)."""
    out = np.empty_like(rgb)
    for c in range(3):
        l, h = np.percentile(rgb[c], [lo, hi])
        if h - l < 1e-6:
            out[c] = rgb[c]
        else:
            out[c] = np.clip((rgb[c] - l) / (h - l), 0, 1)
    return out.astype(np.float32)


# =====================================================================
# STAIN NORMALIZATION (histopathology) — the canonical color-shift case.
# Camelyon17 cross-hospital shift is a H&E STAIN shift (different scanner /
# stain protocol per hospital). Reinhard (Reinhard et al. 2001) matches the
# TARGET image's per-channel mean/std to a SOURCE reference IN LAB COLOR SPACE
# (perceptually-uniform; the standard space for stain transfer), then converts
# back to RGB. Source-referenced (like histogram_match_src): the LAB ref stats
# come from build_source_ref_2d's source-domain images. No extra deps (RGB<->LAB
# via a fixed sRGB->XYZ->LAB matrix). Foreground-only + blended, mirroring the
# other source-referenced tools (the near-white slide background must not drag
# the match).
# =====================================================================
# sRGB <-> XYZ (D65) and XYZ <-> CIE-LAB constants (no skimage dependency).
_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]], dtype=np.float64)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_XYZ_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)  # D65 white point


def _srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)


def _f_lab(t):
    d = 6.0 / 29.0
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)


def _f_lab_inv(t):
    d = 6.0 / 29.0
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))


def _rgb_to_lab(rgb):
    """[3,H,W] RGB [0,1] -> [3,H,W] LAB (L in [0,100], a/b roughly [-128,127])."""
    lin = _srgb_to_linear(rgb)                              # [3,H,W]
    flat = lin.reshape(3, -1)                               # [3,N]
    xyz = (_RGB2XYZ @ flat)                                 # [3,N]
    xyz = xyz / _XYZ_WHITE[:, None]
    fx, fy, fz = _f_lab(xyz[0]), _f_lab(xyz[1]), _f_lab(xyz[2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b]).reshape(rgb.shape).astype(np.float64)


def _lab_to_rgb(lab):
    """[3,H,W] LAB -> [3,H,W] RGB [0,1]."""
    L, a, b = lab[0].ravel(), lab[1].ravel(), lab[2].ravel()
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz = np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)]) * _XYZ_WHITE[:, None]
    lin = (_XYZ2RGB @ xyz)
    rgb = _linear_to_srgb(lin)
    return np.clip(rgb, 0, 1).reshape(lab.shape).astype(np.float32)


def _reinhard_np(rgb, ref, blend=1.0):
    """Reinhard stain normalization: shift+scale each LAB channel of the TARGET so
    its FOREGROUND mean/std match the SOURCE LAB reference (ref['lab_mean'] /
    ref['lab_std']). Operates on foreground (luminance>_BG_EPS) so the near-white
    slide background does not dominate; blended with the original. The canonical
    H&E stain-transfer color match (Reinhard et al. 2001)."""
    if ref is None or "lab_mean" not in ref:
        return rgb
    fg = _fg_mask(rgb)
    lab = _rgb_to_lab(np.clip(rgb, 0, 1))
    out = np.array(lab, dtype=np.float64, copy=True)
    rm, rs = ref["lab_mean"], ref["lab_std"]
    for c in range(3):
        fgc = lab[c][fg]
        if fgc.size == 0:
            continue
        m = float(fgc.mean()); s = float(fgc.std())
        s = s if s > 1e-6 else 1.0
        rescaled = (lab[c][fg] - m) / s * float(rs[c]) + float(rm[c])
        out[c][fg] = (1 - blend) * lab[c][fg] + blend * rescaled
    return _lab_to_rgb(out)


# =====================================================================
# Candidate wrapper — transform happens inside forward(), in the model's
# de-normalized RGB [0,1] input space, so the existing scoring path runs
# unchanged (apples-to-apples with the source reference).
# =====================================================================
class _InputTransformCls(nn.Module):
    """Wrap a source model so each forward(t) (t = ImageNet-normalized
    [B,3,224,224]) is de-normalized to RGB [0,1], passed through `fn` (a per-image
    [3,H,W]->[3,H,W] numpy transform), re-normalized, then run through the base
    model. Lets the existing cls scoring score an INPUT-SPACE color adaptation as
    an ordinary 'candidate model' (no scoring-path changes)."""
    def __init__(self, base, fn):
        super().__init__()
        self.base = base
        self.fn = fn

    def forward(self, t):
        rgb = _denorm(t)                                  # [B,3,H,W] in [0,1]
        arr = rgb.detach().cpu().numpy()
        out = np.empty_like(arr)
        for b in range(arr.shape[0]):
            out[b] = self.fn(arr[b])
        rgb2 = torch.from_numpy(out).to(t.device)
        return self.base(_renorm(rgb2))


# =====================================================================
# Tool builders — each returns (candidate_model, info).
# =====================================================================
def make_histogram_match_src(base, ref):
    if ref is None:
        return copy.deepcopy(base), {"tool": "histogram_match_src", "note": "no source ref"}
    m = _InputTransformCls(base, lambda x: _hist_match_np(x, ref)).eval()
    return m, {"tool": "histogram_match_src", "ref_mean": [round(float(v), 3) for v in ref["ref_mean"]]}


def make_match_intensity_stats(base, ref):
    if ref is None:
        return copy.deepcopy(base), {"tool": "match_intensity_stats", "note": "no source ref"}
    m = _InputTransformCls(base, lambda x: _match_stats_np(x, ref)).eval()
    return m, {"tool": "match_intensity_stats",
               "ref_mean": [round(float(v), 3) for v in ref["ref_mean"]],
               "ref_std": [round(float(v), 3) for v in ref["ref_std"]]}


def make_zscore_to_src(base, ref):
    if ref is None:
        return copy.deepcopy(base), {"tool": "zscore_to_src", "note": "no source ref"}
    m = _InputTransformCls(base, lambda x: _zscore_to_src_np(x, ref)).eval()
    return m, {"tool": "zscore_to_src"}


def make_fda_src(base, ref):
    if ref is None:
        return copy.deepcopy(base), {"tool": "fda_src", "note": "no source ref"}
    beta = ref["fda_beta"]
    m = _InputTransformCls(base, lambda x: _fda_np(x, ref, beta)).eval()
    return m, {"tool": "fda_src", "beta": beta}


def make_gamma_src(base, ref):
    rm = (ref or {}).get("ref_lum_mean", 0.45)
    m = _InputTransformCls(base, lambda x: _gamma_src_np(x, rm)).eval()
    return m, {"tool": "gamma_src", "ref_lum_mean": round(float(rm), 4)}


def make_clahe(base, clip_limit=0.01):
    m = _InputTransformCls(base, lambda x: _clahe_np(x, clip_limit=clip_limit)).eval()
    return m, {"tool": "clahe", "clip_limit": clip_limit}


def make_percentile_normalize(base, lo=1.0, hi=99.0):
    m = _InputTransformCls(base, lambda x: _percentile_np(x, lo=lo, hi=hi)).eval()
    return m, {"tool": "percentile_normalize", "lo": lo, "hi": hi}


def make_reinhard_stain(base, ref):
    """Reinhard stain normalization candidate — LAB mean/std match toward the SOURCE
    reference. The canonical histopathology H&E stain-transfer color match."""
    if ref is None or "lab_mean" not in ref:
        return copy.deepcopy(base), {"tool": "reinhard_stain", "note": "no source LAB ref"}
    m = _InputTransformCls(base, lambda x: _reinhard_np(x, ref)).eval()
    return m, {"tool": "reinhard_stain",
               "lab_mean": [round(float(v), 2) for v in ref["lab_mean"]],
               "lab_std": [round(float(v), 2) for v in ref["lab_std"]]}


# ---- Macenko stain normalization (optional; needs `torchstain`) -------------
class _MacenkoTransformCls(nn.Module):
    """Wrap the source model so each forward() Macenko-normalizes the de-normalized
    RGB target tensor toward a torchstain normalizer FIT on a source reference image,
    then re-normalizes and runs the base model. Macenko et al. 2009 stain
    deconvolution (SVD of optical-density H&E vectors). Per-image; falls through to
    the unchanged image on any failure (e.g. background-only patch)."""
    def __init__(self, base, normalizer):
        super().__init__()
        self.base = base
        self.norm = normalizer

    def forward(self, t):
        rgb = _denorm(t)                                  # [B,3,H,W] in [0,1]
        arr = (rgb.detach().cpu().numpy() * 255.0).astype(np.float32)
        out = np.empty_like(arr)
        for b in range(arr.shape[0]):
            try:
                ten = torch.from_numpy(arr[b])            # [3,H,W] in [0,255]
                normed, _, _ = self.norm.normalize(I=ten, stains=False)
                # torchstain returns [H,W,3] uint8-range float
                n = normed.cpu().numpy()
                if n.ndim == 3 and n.shape[-1] == 3:
                    n = np.transpose(n, (2, 0, 1))
                out[b] = np.clip(n / 255.0, 0, 1)
            except Exception:
                out[b] = arr[b] / 255.0                   # fall through unchanged
        rgb2 = torch.from_numpy(out.astype(np.float32)).to(t.device)
        return self.base(_renorm(rgb2))


def make_macenko_stain(base, to_tensor, source_imgs, device):
    """Build a Macenko-normalization candidate. Returns (deepcopy(base), note) if
    torchstain is unavailable or no source image fits — so the toolbox degrades
    gracefully (skip, don't crash)."""
    try:
        import torchstain
    except Exception:
        return copy.deepcopy(base), {"tool": "macenko_stain", "note": "torchstain not installed"}
    if not source_imgs:
        return copy.deepcopy(base), {"tool": "macenko_stain", "note": "no source image"}
    try:
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        # fit on a mid-list source image (a representative source-stain patch)
        si = source_imgs[len(source_imgs) // 2]
        t = to_tensor(si, device)                         # ImageNet-norm [1,3,224,224]
        ref_rgb = (_denorm(t)[0].detach().cpu().numpy() * 255.0).astype(np.float32)  # [3,H,W]
        normalizer.fit(torch.from_numpy(ref_rgb))
    except Exception as e:
        return copy.deepcopy(base), {"tool": "macenko_stain", "note": f"fit failed: {type(e).__name__}"}
    m = _MacenkoTransformCls(base, normalizer).eval()
    return m, {"tool": "macenko_stain", "fit": "source-ref Macenko (torchstain)"}


# =====================================================================
# norm_stat_tent (2D/cls) — entropy-min on the affine params of EVERY
# norm layer: LayerNorm (transformers!) / GroupNorm / InstanceNorm /
# BatchNorm. This is the NORM-AGNOSTIC lever. On a ViT, bn_reset/tent are
# structural no-ops (LayerNorm has no running stats); this tool adapts the
# LayerNorm weight+bias by entropy minimization on the target. Mirrors the
# 3D version in labelfree_3d_tools.make_norm_stat_tent. Returns (model, info)
# with n_trainable so a no-op is visible in the logs.
# =====================================================================
_ALL_NORM_TYPES_2D = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                      nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                      nn.GroupNorm, nn.LayerNorm)


def make_norm_stat_tent(base, to_tensor, imgs, device, steps=4, lr=1e-3, n_imgs=64):
    """Entropy-minimization TTA on the affine params (weight+bias) of EVERY norm
    layer — BatchNorm AND InstanceNorm/GroupNorm/LayerNorm. This is the lever the
    BN-only tent/bn_reset cannot give on a LayerNorm transformer (ViT) or a
    GroupNorm/InstanceNorm CNN. For BatchNorm we additionally put it in train() so
    batch stats are used during the entropy steps (classic Tent); IN/GN/LN have no
    running stats, only affines, which is exactly what we optimize. Scores nothing —
    just returns the adapted (model, info). `imgs` is the UNLABELED target list (the
    images we adapt to); `to_tensor(img, device)` -> normalized [1,3,224,224]."""
    m = copy.deepcopy(base).to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    params, n_norm = [], 0
    n_ln = n_gn = n_inn = n_bn = 0
    for mod in m.modules():
        if isinstance(mod, _ALL_NORM_TYPES_2D):
            n_norm += 1
            if isinstance(mod, nn.LayerNorm):
                n_ln += 1
            elif isinstance(mod, nn.GroupNorm):
                n_gn += 1
            elif isinstance(mod, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                n_inn += 1
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                n_bn += 1
                mod.train()                          # use batch stats (Tent)
            if getattr(mod, "weight", None) is not None:
                mod.weight.requires_grad_(True); params.append(mod.weight)
            if getattr(mod, "bias", None) is not None:
                mod.bias.requires_grad_(True); params.append(mod.bias)
    n_trainable = sum(p.numel() for p in params)
    if not params:
        m.eval()
        return m, {"tool": "norm_stat_tent", "n_norm_layers": n_norm, "n_trainable": 0,
                   "note": "NO-OP: no norm affine params"}
    opt = torch.optim.Adam(params, lr=lr)
    idx = np.linspace(0, len(imgs) - 1, min(n_imgs, len(imgs)), dtype=int)
    ent_first = ent_last = None
    for s in range(steps):
        for i in idx:
            x = to_tensor(imgs[int(i)], device)      # [1,3,224,224] normalized
            logits = m(x)
            p = F.softmax(logits, 1)
            ent = -(p * torch.clamp(p, 1e-6, 1).log()).sum(1).mean()
            opt.zero_grad(); ent.backward(); opt.step()
            ev = float(ent.detach().cpu())
            if ent_first is None:
                ent_first = ev
            ent_last = ev
    m.eval()
    return m, {"tool": "norm_stat_tent", "n_norm_layers": n_norm,
               "n_trainable": int(n_trainable), "steps": steps, "lr": lr,
               "n_layernorm": n_ln, "n_groupnorm": n_gn,
               "n_instancenorm": n_inn, "n_batchnorm": n_bn,
               "entropy_first": (None if ent_first is None else round(ent_first, 4)),
               "entropy_last": (None if ent_last is None else round(ent_last, 4))}


# =====================================================================
# t3a (2D/cls) — Test-Time Template Adjustment (Iwasawa & Matsuo, NeurIPS 2021).
# TRANSFORMER-NATIVE, training-free, label-free, collapse-free (no gradient steps,
# no weight updates). Ported from scripts/modern_tta_baselines.baseline_t3a, but
# specialized to the GLOBAL classifier head (cls) instead of the per-pixel seg
# variant: collect the PENULTIMATE feature (the input the final Linear head
# consumes = ViT [CLS] token after norm) + a pseudo-label for each TARGET image,
# build per-class prototypes from the HIGHEST-CONFIDENCE support samples (seeded
# from the source classifier rows = T3A "support set init"), then classify by
# cosine-to-prototype. The head weight is REPLACED in place by the (unit, temp-
# scaled) prototypes so model(x) -> logits is unchanged and the whole cls scoring
# path (softmax/AUROC/conf) runs apples-to-apples. No BN/LN dependence at all —
# works identically on ViT/ResNet/etc. Cannot collapse: it never optimizes, only
# re-centers the decision boundary onto target-domain feature clusters.
# =====================================================================
def _find_cls_head(model, num_classes):
    """Return the final nn.Linear classifier head (out_features == num_classes),
    searching common timm/torchvision attribute names first, then the last Linear
    in module order. None if not found (T3A no-ops -> source)."""
    for attr in ("head", "fc", "classifier", "last_linear"):
        h = getattr(model, attr, None)
        if isinstance(h, nn.Linear) and h.out_features == num_classes:
            return h
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear) and mod.out_features == num_classes:
            last = mod
    return last


class _PrototypeHeadCls(nn.Module):
    """Cosine-to-prototype classifier head: logits = T * cos(feat, proto_c). Drop-in
    replacement for a Linear head (same (in->num_classes) signature). Normalizing
    BOTH feature and prototype makes this a pure direction match (T3A); the
    temperature T keeps the softmax sharpness comparable to the linear head."""
    def __init__(self, protos, temperature=10.0):
        super().__init__()
        self.register_buffer("protos", F.normalize(protos, dim=1))   # [C, D]
        self.temperature = float(temperature)

    def forward(self, x):
        return self.temperature * (F.normalize(x, dim=1) @ self.protos.t())


def make_t3a(base, to_tensor, imgs, device, n_support=200, conf_filter_frac=0.5,
             temperature=10.0):
    """Build a T3A candidate (training-free, label-free). `imgs` = UNLABELED TARGET
    list (T3A builds its support set + pseudo-labels FROM the target). A forward
    pre-hook on the final Linear head captures the penultimate feature (head input);
    per target image we take the softmax pseudo-label + confidence, keep only the
    top `conf_filter_frac` most-confident samples (T3A reliability filter), and form
    each class prototype as the confidence-weighted mean of its penultimate vectors,
    SEEDED from the L2-normalized source head rows (so a class with no confident
    target support falls back to the source template). The Linear head is then
    swapped for a cosine-to-prototype head. Returns (model, info)."""
    m = copy.deepcopy(base).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    num_classes_seen = _infer_num_classes(m, to_tensor, imgs, device)
    head = _find_cls_head(m, num_classes=num_classes_seen)
    if head is None:
        return m, {"tool": "t3a", "note": "NO-OP: no final Linear head found"}
    num_classes = head.out_features
    feat_dim = head.in_features
    seed = F.normalize(head.weight.detach().reshape(num_classes, feat_dim), dim=1)
    holder = {}

    def _pre(module, inp):
        holder["x"] = inp[0].detach()

    handle = head.register_forward_pre_hook(_pre)
    psum = torch.zeros(num_classes, feat_dim, device=device)
    pcnt = torch.zeros(num_classes, device=device)
    idx = np.linspace(0, len(imgs) - 1, min(n_support, len(imgs)), dtype=int)
    confs, preds, feats = [], [], []
    with torch.no_grad():
        for i in idx:
            logits = m(to_tensor(imgs[int(i)], device))      # [B,C]
            f = holder["x"]                                   # [B,D] penultimate
            if f.ndim > 2:                                    # safety: flatten any spatial dims
                f = f.reshape(f.shape[0], f.shape[1], -1).mean(-1)
            prob = F.softmax(logits, dim=1)
            c, p = prob.max(dim=1)
            confs.append(c.reshape(-1)); preds.append(p.reshape(-1))
            feats.append(F.normalize(f.reshape(-1, feat_dim), dim=1))
    handle.remove()
    confs = torch.cat(confs); preds = torch.cat(preds); feats = torch.cat(feats, 0)
    n_seen = int(confs.numel())
    if conf_filter_frac < 1.0 and n_seen > num_classes:      # keep most-confident support
        k = max(num_classes, int(n_seen * conf_filter_frac))
        keep = torch.topk(confs, k).indices
        confs, preds, feats = confs[keep], preds[keep], feats[keep]
    n_support_kept = int(confs.numel())
    per_class = {}
    for c in range(num_classes):
        msk = (preds == c)
        per_class[c] = int(msk.sum().item())
        if msk.any():
            psum[c] = (feats[msk] * confs[msk].unsqueeze(1)).sum(0)
            pcnt[c] = confs[msk].sum()
    protos = seed.clone()
    n_from_target = 0
    for c in range(num_classes):
        if pcnt[c] > 0:
            protos[c] = F.normalize(psum[c] / pcnt[c], dim=0)
            n_from_target += 1
    new_head = _PrototypeHeadCls(protos, temperature=temperature).to(device)
    _swap_head(m, head, new_head)
    m.eval()
    return m, {"tool": "t3a", "num_classes": int(num_classes), "feat_dim": int(feat_dim),
               "n_support_seen": n_seen, "n_support_kept": n_support_kept,
               "conf_filter_frac": conf_filter_frac,
               "support_per_class": per_class,
               "protos_from_target": int(n_from_target),
               "protos_from_source_seed": int(num_classes - n_from_target),
               "temperature": temperature}


def _infer_num_classes(model, to_tensor, imgs, device):
    """Run one image to read the logit width (num_classes)."""
    model.eval()
    with torch.no_grad():
        out = model(to_tensor(imgs[0], device))
    return int(out.shape[-1])


def _swap_head(model, old_head, new_head):
    """Replace `old_head` (a child module) with `new_head` in `model`, by attribute
    or by walking named_modules and rebinding the parent's attribute."""
    for attr in ("head", "fc", "classifier", "last_linear"):
        if getattr(model, attr, None) is old_head:
            setattr(model, attr, new_head)
            return True
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if child is old_head:
                setattr(parent, name, new_head)
                return True
    return False


# =====================================================================
# norm_stat_tent_safe (2D/cls) — a COLLAPSE-SAFE LayerNorm-TTA. The plain
# norm_stat_tent collapses the ViT (naive entropy-min on the 19,200 LayerNorm
# affines drives entropy 0.44 -> 0.00 and AUROC -0.30). This variant adds the
# SAR/EATA-family safeguards so LayerNorm adaptation either HELPS or, if it would
# collapse, REVERTS to source (no harm):
#   (a) reliable-sample selection: per step, only the low-entropy fraction
#       (rel_frac) of the batch contributes the entropy loss (SAR §3) — the
#       high-entropy tail that drives degenerate collapse is dropped.
#   (b) last-N LayerNorms only: adapt just the final `last_n` LayerNorm affines
#       (a small, stable lever) instead of all 25 (which is what collapses).
#   (c) gentler LR + fewer steps than the plain tool.
#   (d) collapse-guard + REVERT: a label-free proxy (mean prediction confidence
#       and the fraction of samples predicting the majority class) is tracked on a
#       fixed monitor batch each step; if confidence cratered OR predictions
#       collapsed onto one class beyond a margin over source, restore the
#       best (highest-confidence, non-collapsed) snapshot — falling back to the
#       UNMODIFIED source weights if NO step ever improved. So the worst case is a
#       no-op revert to source, never the -0.30 collapse.
# =====================================================================
def _ln_collapse_proxy(model, to_tensor, mon_imgs, device):
    """Label-free collapse proxy on a fixed monitor batch: returns (mean_conf,
    majority_frac, mean_entropy). High majority_frac or near-zero entropy = the
    degenerate everything-one-class collapse the plain tool falls into."""
    model.eval()
    confs, preds, ents = [], [], []
    with torch.no_grad():
        for img in mon_imgs:
            logit = model(to_tensor(img, device))
            prob = F.softmax(logit, 1)
            c, p = prob.max(1)
            ent = -(prob * torch.clamp(prob, 1e-6, 1).log()).sum(1)
            confs.append(float(c.mean())); preds.append(int(p[0]))
            ents.append(float(ent.mean()))
    from collections import Counter
    maj = max(Counter(preds).values()) / max(len(preds), 1)
    return float(np.mean(confs)), float(maj), float(np.mean(ents))


def make_norm_stat_tent_safe(base, to_tensor, imgs, device, steps=3, lr=2.5e-4,
                             n_imgs=64, last_n=6, rel_frac=0.5, n_monitor=24,
                             ent_floor_frac=0.30, maj_margin=0.20):
    """Collapse-safe LayerNorm entropy-min TTA (SAR-style). Adapts ONLY the last
    `last_n` LayerNorm affines, with reliable-sample selection (lowest-entropy
    `rel_frac` of each batch), a gentler lr / fewer steps, and a label-free
    collapse-guard that REVERTS to the best non-collapsed snapshot (or source) so
    it never does net harm. `imgs` = UNLABELED TARGET list. Returns (model, info)."""
    m = copy.deepcopy(base).to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    lns = [mod for mod in m.modules() if isinstance(mod, nn.LayerNorm)]
    if not lns:
        m.eval()
        return m, {"tool": "norm_stat_tent_safe", "n_layernorm": 0, "n_trainable": 0,
                   "note": "NO-OP: no LayerNorm layers"}
    sel = lns[-last_n:] if last_n and last_n < len(lns) else lns
    params = []
    for mod in sel:
        if getattr(mod, "weight", None) is not None:
            mod.weight.requires_grad_(True); params.append(mod.weight)
        if getattr(mod, "bias", None) is not None:
            mod.bias.requires_grad_(True); params.append(mod.bias)
    n_trainable = sum(p.numel() for p in params)
    if not params:
        m.eval()
        return m, {"tool": "norm_stat_tent_safe", "n_layernorm": len(lns), "n_trainable": 0,
                   "note": "NO-OP: selected LayerNorms have no affine params"}
    opt = torch.optim.Adam(params, lr=lr)
    idx = np.linspace(0, len(imgs) - 1, min(n_imgs, len(imgs)), dtype=int)
    mon_idx = np.linspace(0, len(imgs) - 1, min(n_monitor, len(imgs)), dtype=int)
    mon_imgs = [imgs[int(i)] for i in mon_idx]

    src_conf, src_maj, src_ent = _ln_collapse_proxy(base, to_tensor, mon_imgs, device)
    # best snapshot tracking: a higher-confidence, non-collapsed state. Seed with the
    # (pre-adaptation == source) weights, so a revert with no improving step is a no-op.
    best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    best_conf = src_conf
    best_step = 0
    reverted = False
    ent_first = ent_last = None
    proxy_trace = [{"step": 0, "conf": round(src_conf, 4), "maj": round(src_maj, 4),
                    "ent": round(src_ent, 4)}]

    def _img_entropy(model, ii):
        model.eval()
        with torch.no_grad():
            prob = F.softmax(model(to_tensor(imgs[int(ii)], device)), 1)
            return float(-(prob * torch.clamp(prob, 1e-6, 1).log()).sum(1).mean())

    for s in range(1, steps + 1):
        # (a) reliable-sample selection (SAR §3): the cls path feeds one image at a
        # time (batch==1), so select reliable IMAGES across the support set — keep
        # the lowest-entropy rel_frac of `idx` for THIS step; the high-entropy tail
        # that drives degenerate collapse is dropped from the loss.
        if rel_frac < 1.0 and len(idx) > 2:
            ents_pre = np.array([_img_entropy(m, i) for i in idx])
            k = max(2, int(len(idx) * rel_frac))
            step_idx = idx[np.argsort(ents_pre)[:k]]
        else:
            step_idx = idx
        m.train()
        for mod in sel:                       # LayerNorm.train() == eval(): no running stats
            mod.train()
        step_ents = []
        for i in step_idx:
            x = to_tensor(imgs[int(i)], device)
            logit = m(x)
            prob = F.softmax(logit, 1)
            ent = -(prob * torch.clamp(prob, 1e-6, 1).log()).sum(1)   # [1]
            loss = ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ev = float(loss.detach().cpu())
            step_ents.append(ev)
            if ent_first is None:
                ent_first = ev
            ent_last = ev
        # (d) collapse-guard on a fixed monitor batch
        conf, maj, ment = _ln_collapse_proxy(m, to_tensor, mon_imgs, device)
        proxy_trace.append({"step": s, "conf": round(conf, 4), "maj": round(maj, 4),
                            "ent": round(ment, 4)})
        collapsed = (ment < ent_floor_frac * max(src_ent, 1e-6)) or \
                    (maj > min(1.0, src_maj + maj_margin))
        if collapsed:
            reverted = True
            break                             # stop; revert to best snapshot below
        # KEEP-vs-REVERT (conservative): only adopt an adapted state if it is more
        # confident AND not more class-imbalanced than source. Entropy-min trivially
        # raises confidence, so the no-extra-imbalance condition guards against a
        # silent partial-collapse that doesn't trip the hard threshold. If no step
        # qualifies, best_state stays = source (a clean no-op revert).
        if conf > best_conf and maj <= src_maj + 0.5 * maj_margin:
            best_conf = conf; best_step = s
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # restore the best non-collapsed snapshot (source if none improved)
    m.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    m.eval()
    return m, {"tool": "norm_stat_tent_safe", "n_layernorm": len(lns),
               "n_adapted_layernorm": len(sel), "n_trainable": int(n_trainable),
               "steps": steps, "lr": lr, "last_n": last_n, "rel_frac": rel_frac,
               "reverted_on_collapse": bool(reverted), "kept_step": int(best_step),
               "source_conf": round(src_conf, 4), "best_conf": round(best_conf, 4),
               "source_entropy": round(src_ent, 4),
               "entropy_first": (None if ent_first is None else round(ent_first, 4)),
               "entropy_last": (None if ent_last is None else round(ent_last, 4)),
               "proxy_trace": proxy_trace,
               "note": ("reverted to source/best (collapse guard fired)" if reverted
                        else "kept adapted state" if best_step > 0
                        else "no improving step; kept source")}


def _torchstain_available():
    try:
        import torchstain  # noqa: F401
        return True
    except Exception:
        return False


# Registry name -> needs-source-ref? (used by the cls adapter toolbox).
# reinhard_stain (LAB mean/std match) is the canonical histopathology stain tool;
# macenko_stain (stain deconvolution) is appended only if torchstain is installed.
LABELFREE_2D_TOOL_NAMES = [
    "histogram_match_src", "match_intensity_stats", "zscore_to_src",
    "fda_src", "gamma_src", "reinhard_stain", "clahe", "percentile_normalize",
    "norm_stat_tent", "norm_stat_tent_safe", "t3a",
] + (["macenko_stain"] if _torchstain_available() else [])
_NEEDS_REF_2D = {"histogram_match_src", "match_intensity_stats", "zscore_to_src",
                 "fda_src", "gamma_src", "reinhard_stain"}
# Tools that need source IMAGES + to_tensor (not just the reduced ref dict).
_NEEDS_SRC_IMGS_2D = {"macenko_stain"}
# Tools that need to_tensor + the UNLABELED TARGET images (transformer-native,
# weight/classifier-space levers that adapt TO the target — no source ref dict):
#   norm_stat_tent      : entropy-min on ALL norm affines (collapses the ViT).
#   norm_stat_tent_safe : collapse-safe last-N LayerNorm entropy-min (SAR-style).
#   t3a                 : training-free prototype (template) classifier head.
_NEEDS_TARGET_IMGS_2D = {"norm_stat_tent", "norm_stat_tent_safe", "t3a"}


def build_labelfree_2d_candidate(name, base, ref, to_tensor=None, source_imgs=None, device=None):
    """Dispatch: build the (candidate_model, info) for `name`. ref may be None for
    the ref-free tools (clahe / percentile_normalize). macenko_stain additionally
    needs (to_tensor, source_imgs, device) to fit the torchstain normalizer."""
    if name == "histogram_match_src":
        return make_histogram_match_src(base, ref)
    if name == "match_intensity_stats":
        return make_match_intensity_stats(base, ref)
    if name == "zscore_to_src":
        return make_zscore_to_src(base, ref)
    if name == "fda_src":
        return make_fda_src(base, ref)
    if name == "gamma_src":
        return make_gamma_src(base, ref)
    if name == "reinhard_stain":
        return make_reinhard_stain(base, ref)
    if name == "macenko_stain":
        return make_macenko_stain(base, to_tensor, source_imgs, device)
    if name == "norm_stat_tent":
        # `source_imgs` here carries the UNLABELED TARGET images (the adapter passes
        # the target list for this tool). Falls back to source_imgs if no target.
        return make_norm_stat_tent(base, to_tensor, source_imgs, device)
    if name == "norm_stat_tent_safe":
        # `source_imgs` carries the UNLABELED TARGET images (see _NEEDS_TARGET_IMGS_2D).
        return make_norm_stat_tent_safe(base, to_tensor, source_imgs, device)
    if name == "t3a":
        # `source_imgs` carries the UNLABELED TARGET images (T3A builds its support
        # set + pseudo-labels FROM the target).
        return make_t3a(base, to_tensor, source_imgs, device)
    if name == "clahe":
        return make_clahe(base)
    if name == "percentile_normalize":
        return make_percentile_normalize(base)
    raise ValueError(f"unknown labelfree 2D tool {name}")
