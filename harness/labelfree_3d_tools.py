"""Label-free (K=0) INPUT-SPACE + prediction-time + norm-stat adaptation tools
for the 3D deployment adapters (harness/seg3d_adapter.py and its subclasses).

WHY THIS MODULE EXISTS
----------------------
The 3D adapters previously exposed only `bn_reset` / `tent` as label-free moves.
Both are STRUCTURAL NO-OPS on the modern 3D source models:
  * liver-tumor  : MONAI SegResNet -> GroupNorm (25 layers), NO BatchNorm
  * spleen       : MONAI UNet      -> norm="batch" (BatchNorm present, but the
                   contrast shift is an INTENSITY/appearance gap that BN-reset
                   alone barely touches).
The 2D input-space tools (histogram_match / clahe / gamma / fda) lived in
medadaptagent (data_adaptation.py / tools/preprocessing.py / advanced_tta.py)
but were 2D-only and never bridged to the 3D toolbox. So at K=0 on a 3D
InstanceNorm/GroupNorm model the agent had NO real lever -> it escalated / kept
source on every 3D K=0 cell.

DESIGN — wrapper-model candidates
---------------------------------
Every input-space / prediction-time tool produces a CANDIDATE that is scored by
the EXISTING adapter machinery (`_predict_stats_3d`, `_dice_of_3d`,
`forward_pred` sliding-window, the VLM pairwise signal) WITHOUT touching the
scoring path. We do this by wrapping the source model in a thin `nn.Module`
whose `forward(x)` first transforms the **windowed [0,1] patch tensor** that the
sliding-window inferer feeds, then calls the source model. Because the transform
lives inside forward(), the whole sliding-window / Dice / stats / VLM chain runs
unchanged and apples-to-apples with the 0.4712 source reference.

Operating in the model's WINDOWED [0,1] space (not raw HU) is deliberate and
correct: the source-reference histogram/spectrum is computed in the SAME [0,1]
space, so "make the target look source-like" is a like-for-like match, and the
transform sits exactly where the model consumes intensities (after the adapter's
HU window/normalize, before the first conv) — which is what "BEFORE the model's
window/normalize so the model sees a source-like distribution" means once the
window is the model's own fixed ScaleIntensityRange.

THE 7 TOOLS
-----------
 1 histogram_match : per-volume 3D histogram match of the target patch to a
                     SOURCE reference histogram (built from a few unlabeled
                     source-domain volumes, windowed [0,1]). CDF transport.
 2 clahe           : 3D CLAHE (skimage.exposure.equalize_adapthist) — local
                     contrast equalization; cv2 is unavailable so we use skimage.
 3 gamma           : auto-gamma power transform x^g so the target patch mean
                     matches the SOURCE reference mean (input brightness curve).
 4 fda             : Fourier Domain Adaptation — swap the low-frequency amplitude
                     band of the target with the SOURCE reference amplitude
                     (3D FFT, per-volume), preserve target phase (content).
 5 tta_aug         : prediction-time TTA — average the source model's softmax over
                     flips + 90 deg in-plane rotations. PURE inference, NO training,
                     NORM-AGNOSTIC -> the cheap universal lever bn_reset/tent
                     cannot give on GroupNorm/InstanceNorm.
 6 denoise         : input denoise (Gaussian + median; optional NL-means) of the
                     target patch — directly targets the low-dose NOISE component.
                     Also exposed as a denoise->histogram_match combo.
 7 norm_stat_tent  : the real IN/GN lever — entropy-minimization TTA on the affine
                     params of ALL norm layers (GroupNorm / InstanceNorm /
                     LayerNorm / BatchNorm), NOT just BatchNorm. Non-no-op on
                     SegResNet's 25 GroupNorm layers (verified by #trainable>0).

Each builder returns (candidate_model, info_dict). The reference-needing tools
(histogram_match / fda / gamma) take a precomputed `ref` from `build_source_ref`.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Source reference (label-free): a few unlabeled source-domain volumes,
# reduced to the windowed [0,1] intensity statistics the input-space
# tools need (histogram CDF, mean, mean low-freq amplitude band).
# =====================================================================
_BG_EPS = 0.02   # voxels with windowed intensity <= this are treated as background
                 # (air/empty); EXCLUDED from histogram/mean refs so the foreground
                 # soft-tissue distribution drives matching (Phase-A smoke proved that
                 # whole-patch CDF transport is dominated by the background spike and
                 # destroys tumor contrast -> 0 Dice).


def build_source_ref(to_tensor, source_vols, device, n=4, bins=256,
                     fda_band=0.01, patch=(96, 96, 96)):
    """Build a SOURCE reference from up to `n` unlabeled source volumes, in the
    model's windowed [0,1] space (uses the adapter's own `to_tensor`). Returns a
    dict with:
      ref_cdf/ref_vals : FOREGROUND-only [0,1] CDF (bg excluded) for CDF-transport
      ref_mean         : FOREGROUND mean windowed intensity (for auto-gamma)
      ref_amp          : mean centered amplitude spectrum (patch-shaped, for FDA)
    All computed on a CENTER patch per source volume so it is comparable to the
    sliding-window patch the candidate transforms see at inference."""
    if not source_vols:
        return None
    idx = np.linspace(0, len(source_vols) - 1, min(n, len(source_vols)), dtype=int)
    fg_pixels = []
    means = []
    amps = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in idx:
        x = to_tensor(source_vols[int(i)], device)          # [1,1,D,H,W] windowed [0,1]
        xp = _center_patch(x, patch)[0, 0].detach().cpu().numpy().astype(np.float32)
        fg = xp[xp > _BG_EPS]                                # drop background spike
        if fg.size == 0:
            fg = xp.ravel()
        fg_pixels.append(fg)
        means.append(float(fg.mean()))
        amps.append(_lowfreq_amp(xp, fda_band))
    allfg = np.concatenate(fg_pixels)
    ref_hist, _ = np.histogram(allfg, bins=bins, range=(0.0, 1.0))
    ref_vals = 0.5 * (edges[:-1] + edges[1:])
    return {"ref_vals": ref_vals,
            "ref_cdf": np.cumsum(ref_hist) / (ref_hist.sum() + 1e-8),
            "ref_mean": float(np.mean(means)),
            "ref_amp": np.mean(amps, axis=0),
            "bins": bins, "fda_band": fda_band, "patch": tuple(patch)}


def _center_patch(x, patch=(96, 96, 96)):
    """Center-crop+pad a [B,C,D,H,W] tensor to `patch` (matches the adapter's
    _center_patch so reference stats are patch-comparable)."""
    D, H, W = x.shape[-3:]
    sl = []
    for dim, p in zip((D, H, W), patch):
        start = max(0, (dim - p) // 2)
        sl.append(slice(start, start + min(p, dim)))
    xc = x[:, :, sl[0], sl[1], sl[2]]
    pads = []
    for i in range(3):
        pads += [0, max(0, patch[2 - i] - xc.shape[4 - i])]
    if any(pads):
        xc = F.pad(xc, pads, mode="constant", value=0)
    return xc


def _lowfreq_amp(vol, band):
    """Mean low-frequency amplitude band of a [D,H,W] volume (3D FFT, centered).
    Returns the full centered amplitude spectrum (so FDA can read its center)."""
    fft = np.fft.fftshift(np.fft.fftn(vol))
    return np.abs(fft).astype(np.float32)


# =====================================================================
# Tensor-space transforms (operate on a windowed [0,1] patch tensor
# [B,1,d,h,w], the exact thing sliding_window_inference feeds forward()).
# =====================================================================
def _hist_match_tensor(x, ref, blend=0.6):
    """FOREGROUND-only 3D CDF-transport histogram match of each [0,1] patch to the
    source ref CDF, blended `blend` with the original. Background voxels (<= _BG_EPS)
    are left UNCHANGED so the air/empty spike does not dominate the transport (the
    Phase-A smoke showed whole-patch transport collapses tumor Dice to ~0). Only the
    foreground soft-tissue distribution is matched to the source foreground CDF."""
    bins = ref["bins"]; ref_cdf = ref["ref_cdf"]; ref_vals = ref["ref_vals"]
    out = torch.empty_like(x)
    arr = x.detach().cpu().numpy()
    for b in range(arr.shape[0]):
        v = arr[b, 0].astype(np.float32)
        flat = np.clip(v.ravel(), 0.0, 1.0)
        fg_mask = flat > _BG_EPS
        matched = flat.copy()
        if fg_mask.sum() > 0:
            fg = flat[fg_mask]
            src_hist, edges = np.histogram(fg, bins=bins, range=(0.0, 1.0))
            src_cdf = np.cumsum(src_hist) / (src_hist.sum() + 1e-8)
            centers = 0.5 * (edges[:-1] + edges[1:])
            mapped_centers = np.interp(src_cdf, ref_cdf, ref_vals)   # fg-only transport
            new_fg = np.interp(fg, centers, mapped_centers)
            matched[fg_mask] = (1 - blend) * fg + blend * new_fg     # blend, not full swap
        out[b, 0] = torch.from_numpy(
            np.clip(matched, 0, 1).reshape(v.shape).astype(np.float32)).to(x.device)
    return out.clamp(0, 1)


def _clahe_tensor(x, clip_limit=0.005, kernel_frac=0.5, blend=0.4, max_dim=48):
    """3D CLAHE via skimage.exposure.equalize_adapthist on each [0,1] patch, BLENDED
    `blend` with the original. equalize_adapthist globally remaps intensities (full
    histogram equalization per tile) which breaks the model's fixed training window
    if applied at full strength (Phase-A smoke: full CLAHE -> 0.08 Dice). A large
    kernel + small clip + 40% blend recovers local contrast while keeping the global
    intensity scale the model expects.

    SPEED: equalize_adapthist on a full 96^3 patch is prohibitively slow per
    sliding-window patch (Phase-A: >10 min for one tool over 46 vols). We run it on a
    DOWNSAMPLED copy (longest dim <= max_dim) then upsample the equalized result back
    and blend — the local-contrast correction is low-frequency so the downsample is
    near-lossless, and the cost drops ~(96/48)^3 = 8x+."""
    from skimage.exposure import equalize_adapthist
    from scipy import ndimage
    out = torch.empty_like(x)
    arr = x.detach().cpu().numpy()
    for b in range(arr.shape[0]):
        v = np.clip(arr[b, 0].astype(np.float64), 0.0, 1.0)
        try:
            f = max_dim / max(v.shape)
            vs = ndimage.zoom(v, f, order=1) if f < 1.0 else v
            ks = max(8, int(min(vs.shape) * kernel_frac))
            eqs = equalize_adapthist(vs, kernel_size=ks, clip_limit=clip_limit, nbins=256)
            eq = ndimage.zoom(eqs, np.array(v.shape) / np.array(eqs.shape), order=1) \
                if eqs.shape != v.shape else eqs
            mixed = (1 - blend) * v + blend * eq
        except Exception:
            mixed = v
        out[b, 0] = torch.from_numpy(mixed.astype(np.float32)).to(x.device)
    return out.clamp(0, 1)


def _gamma_tensor(x, ref_mean, gamma=None):
    """Auto-gamma: choose g so the patch mean matches ref_mean (x^g). If gamma is
    given, use it directly. Brightness-curve correction in [0,1]."""
    if gamma is None:
        m = float(x.mean().clamp(1e-4, 1 - 1e-4))
        rm = float(np.clip(ref_mean, 1e-4, 1 - 1e-4))
        # solve mean(x^g) ~ rm via the log-mean heuristic (same as tools/preprocessing)
        gamma = float(np.clip(np.log(rm + 1e-6) / np.log(m + 1e-6), 0.25, 4.0))
    return x.clamp(0, 1).pow(gamma), gamma


def _fda_tensor(x, ref_amp, band, blend=0.5):
    """3D Fourier Domain Adaptation: BLEND the low-freq amplitude band of each [0,1]
    patch toward the source ref amplitude (keep target phase = content). A small band
    (~0.01, the FDA-paper default) + a 50% blend is subtle style transfer; a hard full
    swap over a wide band destroys the volume (Phase-A smoke: band=0.06 full-swap ->
    0.03 Dice). ref_amp is the centered amplitude spectrum of a source PATCH."""
    out = torch.empty_like(x)
    arr = x.detach().cpu().numpy()
    ra = ref_amp
    for b in range(arr.shape[0]):
        v = arr[b, 0].astype(np.float32)
        ra_use = ra if ra.shape == v.shape else _match_shape(ra, v.shape)
        fft = np.fft.fftshift(np.fft.fftn(v))
        amp = np.abs(fft); phase = np.angle(fft)
        D, H, W = v.shape
        cz, cy, cx = D // 2, H // 2, W // 2
        bz = max(1, int(D * band)); by = max(1, int(H * band)); bx = max(1, int(W * band))
        mixed = amp.copy()
        sl = (slice(cz - bz, cz + bz), slice(cy - by, cy + by), slice(cx - bx, cx + bx))
        mixed[sl] = (1 - blend) * amp[sl] + blend * ra_use[sl]   # blend low-freq amplitude
        rec = np.real(np.fft.ifftn(np.fft.ifftshift(mixed * np.exp(1j * phase))))
        out[b, 0] = torch.from_numpy(
            np.clip(rec, 0, 1).astype(np.float32)).to(x.device)
    return out


def _match_shape(a, shape):
    """Center-crop or zero-pad array `a` to `shape`."""
    out = np.zeros(shape, dtype=a.dtype)
    sl_o, sl_a = [], []
    for ao, so in zip(a.shape, shape):
        n = min(ao, so)
        oa = (ao - n) // 2; oo = (so - n) // 2
        sl_a.append(slice(oa, oa + n)); sl_o.append(slice(oo, oo + n))
    out[tuple(sl_o)] = a[tuple(sl_a)]
    return out


def _denoise_tensor(x, sigma=0.6, median_size=3, nlm=False):
    """Input denoise of each [0,1] patch: Gaussian + median (+ optional NL-means).
    Targets the low-dose additive-noise component."""
    from scipy import ndimage
    out = torch.empty_like(x)
    arr = x.detach().cpu().numpy()
    for b in range(arr.shape[0]):
        v = arr[b, 0].astype(np.float32)
        d = ndimage.gaussian_filter(v, sigma=sigma)
        if median_size and median_size > 1:
            d = ndimage.median_filter(d, size=median_size)
        if nlm:
            try:
                from skimage.restoration import denoise_nl_means, estimate_sigma
                s = float(np.mean(estimate_sigma(d)))
                d = denoise_nl_means(d, h=0.8 * s, sigma=s, fast_mode=True,
                                     patch_size=3, patch_distance=3)
            except Exception:
                pass
        out[b, 0] = torch.from_numpy(np.clip(d, 0, 1).astype(np.float32)).to(x.device)
    return out


# =====================================================================
# Candidate wrappers
# =====================================================================
class _InputTransformModel(nn.Module):
    """Wrap a source model so each forward(x) first applies `fn` to the windowed
    [0,1] patch tensor x, then runs the source model. Lets the existing 3D
    sliding-window / Dice / stats / VLM path score an INPUT-SPACE adaptation as
    an ordinary 'candidate model' (no scoring-path changes)."""
    def __init__(self, base, fn):
        super().__init__()
        self.base = base
        self.fn = fn

    def forward(self, x):
        return self.base(self.fn(x))


class _TTAAugModel(nn.Module):
    """Prediction-time TTA: average the source softmax over flips + 90 deg in-plane
    rotations. Norm-agnostic (no params change). Each augmentation is applied to the
    input patch, the model run, and the prediction inverse-transformed before
    averaging. Returns AVERAGED LOGITS (log of mean softmax) so downstream argmax /
    entropy behave normally."""
    def __init__(self, base, flips=True, rots=True):
        super().__init__()
        self.base = base
        self.flips = flips
        self.rots = rots

    def _augs(self):
        # (name, fwd, inv) on a [B,C,D,H,W] tensor; in-plane axes are H,W = dims 3,4
        augs = [("id", lambda t: t, lambda t: t)]
        if self.flips:
            augs += [
                ("flipH", lambda t: torch.flip(t, dims=[3]), lambda t: torch.flip(t, dims=[3])),
                ("flipW", lambda t: torch.flip(t, dims=[4]), lambda t: torch.flip(t, dims=[4])),
            ]
        if self.rots:
            augs += [
                ("rot90", lambda t: torch.rot90(t, 1, dims=[3, 4]),
                 lambda t: torch.rot90(t, -1, dims=[3, 4])),
                ("rot270", lambda t: torch.rot90(t, 3, dims=[3, 4]),
                 lambda t: torch.rot90(t, -3, dims=[3, 4])),
            ]
        return augs

    def forward(self, x):
        probs = None
        n = 0
        for _, fwd, inv in self._augs():
            xa = fwd(x)
            logits = self.base(xa)
            p = inv(F.softmax(logits, dim=1))
            probs = p if probs is None else probs + p
            n += 1
        mean_p = (probs / max(n, 1)).clamp(1e-6, 1.0)
        return torch.log(mean_p)               # log-prob -> argmax/entropy unchanged


# =====================================================================
# Tool builders — each returns (candidate_model, info)
# =====================================================================
def make_histogram_match(base, ref):
    if ref is None:
        return copy.deepcopy(base), {"tool": "histogram_match", "note": "no source ref"}
    m = _InputTransformModel(base, lambda x: _hist_match_tensor(x, ref)).eval()
    return m, {"tool": "histogram_match", "ref_mean": round(ref["ref_mean"], 4)}


def make_clahe(base, clip_limit=0.01):
    m = _InputTransformModel(base, lambda x: _clahe_tensor(x, clip_limit=clip_limit)).eval()
    return m, {"tool": "clahe", "clip_limit": clip_limit}


def make_gamma(base, ref):
    rm = (ref or {}).get("ref_mean", 0.2)
    m = _InputTransformModel(base, lambda x: _gamma_tensor(x, rm)[0]).eval()
    return m, {"tool": "gamma", "ref_mean": round(float(rm), 4)}


def make_fda(base, ref):
    if ref is None:
        return copy.deepcopy(base), {"tool": "fda", "note": "no source ref"}
    band = ref["fda_band"]; amp = ref["ref_amp"]
    m = _InputTransformModel(base, lambda x: _fda_tensor(x, amp, band)).eval()
    return m, {"tool": "fda", "band": band}


def make_tta_aug(base):
    m = _TTAAugModel(base).eval()
    return m, {"tool": "tta_aug", "augs": "id+flipH+flipW+rot90+rot270"}


def make_denoise(base, sigma=0.6, median_size=3, nlm=False):
    m = _InputTransformModel(
        base, lambda x: _denoise_tensor(x, sigma=sigma, median_size=median_size, nlm=nlm)).eval()
    return m, {"tool": "denoise", "sigma": sigma, "median": median_size, "nlm": nlm}


def make_denoise_histmatch(base, ref, sigma=0.6, median_size=3):
    if ref is None:
        return make_denoise(base, sigma=sigma, median_size=median_size)
    def _fn(x):
        return _hist_match_tensor(_denoise_tensor(x, sigma=sigma, median_size=median_size), ref)
    m = _InputTransformModel(base, _fn).eval()
    return m, {"tool": "denoise+histogram_match", "sigma": sigma, "median": median_size}


# ---------------------------------------------------------------------
# 7. norm_stat_tent — entropy-min on ALL norm-layer affine params
#    (GroupNorm / InstanceNorm / LayerNorm / BatchNorm). The real IN/GN
#    lever. Returns (model, info) with n_norm_params so a no-op is visible.
# ---------------------------------------------------------------------
_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
               nn.GroupNorm, nn.LayerNorm)


def make_norm_stat_tent(base, vols, to_tensor, device, steps=4, lr=1e-3,
                        n_vols=4, patch=(96, 96, 96)):
    """Entropy-minimization TTA on the affine params (weight+bias) of EVERY norm
    layer — BatchNorm AND InstanceNorm/GroupNorm/LayerNorm. This is the lever the
    BN-only tent/bn_reset cannot give on a GroupNorm SegResNet. For BatchNorm we
    additionally put it in train() so batch stats are used during the entropy
    steps (classic Tent); IN/GN/LN have no running stats, only affines, which is
    exactly what we optimize. Returns (model, info)."""
    m = copy.deepcopy(base).to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    params = []
    n_norm = 0
    for mod in m.modules():
        if isinstance(mod, _NORM_TYPES):
            n_norm += 1
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.train()                          # use batch stats (Tent)
            if getattr(mod, "weight", None) is not None:
                mod.weight.requires_grad_(True); params.append(mod.weight)
            if getattr(mod, "bias", None) is not None:
                mod.bias.requires_grad_(True); params.append(mod.bias)
    n_trainable = sum(p.numel() for p in params)
    if not params:
        m.eval()
        return m, {"tool": "norm_stat_tent", "n_norm_layers": n_norm,
                   "n_trainable": 0, "note": "NO-OP: no norm affine params"}
    opt = torch.optim.Adam(params, lr=lr)
    idx = np.linspace(0, len(vols) - 1, min(n_vols, len(vols)), dtype=int)
    ent_first = ent_last = None
    for s in range(steps):
        for i in idx:
            x = to_tensor(vols[int(i)], device)
            xc = _center_patch(x, patch)
            logits = m(xc)
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
               "entropy_first": (None if ent_first is None else round(ent_first, 4)),
               "entropy_last": (None if ent_last is None else round(ent_last, 4))}


# Convenience registry name -> builder signature (used by the adapter toolbox).
LABELFREE_3D_TOOL_NAMES = [
    "histogram_match", "clahe", "gamma", "fda",
    "tta_aug", "denoise", "denoise_histmatch", "norm_stat_tent",
]
