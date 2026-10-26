"""Foundation-model agreement critic (Nat-Comm redesign Step 0).

The source-INDEPENDENT label-free selection signal that replaces the
rubber-stamp VLM gate. For a candidate's adapted model M', it measures
how well M's predicted mask agrees with a frozen medical foundation
model's (MedSAM) anatomy mask prompted FROM M's own prediction:

    agreement = mean over anchors of
        [ w_iou * softIoU(M_mask, MedSAM_mask)
        + w_conf * MedSAM_iou_confidence
        + w_stab * prompt_stability(jittered boxes) ]

Rationale (see docs/REDESIGN_NATCOMM_V2.md): target prediction entropy
is computed by the SAME drifted model that made the error, so on
confident-wrong masks it is self-confirming (the mechanism behind the
baseline collapse + the mnms catastrophe). MedSAM is an external,
source-independent second opinion grounded in millions of medical
masks, so a confidently-wrong mask DISAGREES with it and scores low.

Dataset-agnostic: uses only the model + a frozen FM. No dataset names.
Offline: loads MedSAM from the shared HF cache; no runtime internet.
Degradable: if MedSAM cannot load, fm_available() returns False and the
caller falls back to the existing quality_score signals.
"""
from __future__ import annotations
import os
import numpy as np

# Point HF at the pre-fetched shared cache so compute nodes never need net.
_DEF_CACHE = os.path.expanduser("~/.cache/medadaptagent/huggingface")
os.environ.setdefault("HF_HOME", _DEF_CACHE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F

_MEDSAM = {"model": None, "processor": None, "device": None}
MEDSAM_ID = "flaviagiammarino/medsam-vit-base"


def load_medsam(device="cuda:0"):
    """Lazy-load the frozen MedSAM (transformers SamModel) once."""
    if _MEDSAM["model"] is not None and _MEDSAM["device"] == device:
        return _MEDSAM["model"], _MEDSAM["processor"]
    from transformers import SamModel, SamProcessor
    m = SamModel.from_pretrained(MEDSAM_ID).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    proc = SamProcessor.from_pretrained(MEDSAM_ID)
    _MEDSAM.update(model=m, processor=proc, device=device)
    return m, proc


def fm_available():
    try:
        import transformers  # noqa: F401
        return os.path.isdir(_DEF_CACHE)
    except Exception:
        return False


def _to_rgb_uint8(img):
    """Host image (HxW or HxWx{1,3}, float[0,1] or uint8) -> HxWx3 uint8."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.0 \
            else a.astype(np.uint8)
    return a


def _mask_to_boxes(pred_mask, num_classes, min_pixels=20):
    """Per-foreground-class bounding boxes [x0,y0,x1,y1] from an argmax mask.
    Binary-collapses if num_classes==2. Returns list of boxes (may be empty)."""
    boxes = []
    classes = ([1] if num_classes == 2
               else list(range(1, num_classes)))
    for c in classes:
        ys, xs = np.where(pred_mask == c if num_classes != 2
                          else pred_mask >= 1)
        if len(xs) < min_pixels:
            continue
        boxes.append([int(xs.min()), int(ys.min()),
                      int(xs.max()), int(ys.max())])
        if num_classes == 2:
            break
    return boxes


@torch.no_grad()
def medsam_masks(image_rgb, boxes, device="cuda:0"):
    """Run MedSAM with box prompts; return (union_mask[H,W] bool, iou_conf)."""
    m, proc = load_medsam(device)
    H, W = image_rgb.shape[:2]
    if not boxes:
        return np.zeros((H, W), bool), 0.0
    inp = proc(image_rgb, input_boxes=[boxes], return_tensors="pt").to(device)
    out = m(**inp, multimask_output=False)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(),
        inp["reshaped_input_sizes"].cpu())[0]  # [n_boxes,1,H,W] bool
    union = masks.squeeze(1).any(dim=0).numpy().astype(bool)
    conf = float(out.iou_scores.mean().item())
    return union, conf


def _soft_iou(a_bool, b_bool):
    a = a_bool.astype(np.float32); b = b_bool.astype(np.float32)
    inter = (a * b).sum(); union = a.sum() + b.sum() - inter
    return float(inter / (union + 1e-6)) if union > 0 else (
        1.0 if a.sum() == 0 and b.sum() == 0 else 0.0)


@torch.no_grad()
def fm_agreement(seg_model, images, to_tensor, num_classes, device="cuda:0",
                  n_anchors=16, w_iou=0.7, w_conf=0.2, w_stab=0.1,
                  jitter=8):
    """Mean FM-agreement score in [0,1] for seg_model over anchor images.

    to_tensor(img, device) -> model input tensor [1,C,H,W] matching how the
    pipeline feeds the model (passed in so the eval grid matches _avg_dice).
    Returns (score, per_anchor list of dicts) for auditability/credit-assign.
    """
    seg_model.eval()
    if not fm_available():
        return None, []
    idx = np.linspace(0, len(images) - 1,
                       min(n_anchors, len(images)), dtype=int)
    per = []
    for i in idx:
        img = images[int(i)]
        rgb = _to_rgb_uint8(img)
        H, W = rgb.shape[:2]
        logits = seg_model(to_tensor(img, device))
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        # resize pred to rgb grid if model ran at a different size
        if pred.shape != (H, W):
            pt = torch.from_numpy(pred.astype(np.int64))[None, None].float()
            pred = F.interpolate(pt, size=(H, W), mode="nearest")[0, 0].numpy().astype(np.int64)
        boxes = _mask_to_boxes(pred, num_classes)
        model_fg = (pred >= 1)
        fm_mask, conf = medsam_masks(rgb, boxes, device)
        iou = _soft_iou(model_fg, fm_mask)
        # prompt stability: jitter boxes, re-run, IoU of the two FM masks
        stab = 1.0
        if boxes:
            jb = [[max(0, b[0] - jitter), max(0, b[1] - jitter),
                   min(W - 1, b[2] + jitter), min(H - 1, b[3] + jitter)]
                  for b in boxes]
            fm2, _ = medsam_masks(rgb, jb, device)
            stab = _soft_iou(fm_mask, fm2)
        s = w_iou * iou + w_conf * conf + w_stab * stab
        per.append({"i": int(i), "iou": iou, "conf": conf, "stab": stab,
                     "score": s, "model_fg_frac": float(model_fg.mean()),
                     "fm_fg_frac": float(fm_mask.mean())})
    score = float(np.mean([p["score"] for p in per])) if per else 0.0
    return score, per


# ============================================================
# MIFR — Model-Independent FM Reference (redesign Step-5)
# ============================================================
# The fm_agreement above prompts MedSAM FROM the model's own mask, which
# is self-confirming on clean-but-wrong blobs (prostate spectral: Dice
# 0.066 yet FM 0.610). MIFR instead builds ONE fixed anatomy reference
# per image by GRID-prompting MedSAM (model-independent), then scores
# every candidate against that SAME reference. A clean-but-mislocated
# mask has LOW IoU with a fixed outline that does not move toward it, so
# it can no longer self-confirm. Verified on the 616445 Dice table to
# lift FM-select 0.4478 -> 0.5367 and (with a tent-anchored switch rule)
# to 0.5577 (+0.036 over fixed-tent).

@torch.no_grad()
def medsam_grid_reference(image_rgb, device="cuda:0", p=3, debug=False):
    """One model-INDEPENDENT anatomy reference R(x) via a fixed BOX grid.
    MedSAM is BOX-prompt-trained: point prompts give tiny (area<0.01)
    masks and its iou_scores run LOW (~0.4-0.75 on medical images), so an
    iou threshold rejects everything (the deg=1.0 bug). Instead prompt pxp
    tile boxes + one organ-scale center box, take the dominant structure in
    each (multimask_output=False), keep organ-scale masks (area in
    [0.01,0.55]) with soft-IoU>0.7 NMS, and union them.
    Returns (R[H,W] bool, degenerate bool)."""
    m, proc = load_medsam(device)
    H, W = image_rgb.shape[:2]
    boxes = []
    for gy in range(p):
        for gx in range(p):
            boxes.append([int(W * gx / p), int(H * gy / p),
                          int(W * (gx + 1) / p), int(H * (gy + 1) / p)])
    boxes.append([int(W * 0.2), int(H * 0.2), int(W * 0.8), int(H * 0.8)])
    inp = proc(image_rgb, input_boxes=[boxes], return_tensors="pt").to(device)
    out = m(**inp, multimask_output=False)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(),
        inp["reshaped_input_sizes"].cpu())[0]          # [n_boxes,1,H,W]
    io = out.iou_scores.cpu().numpy().reshape(-1)
    props = masks.squeeze(1).numpy().astype(bool)
    keep = []
    for j in np.argsort(-io):
        a = props[j]; fr = a.mean()
        if fr < 0.01 or fr > 0.55:
            continue
        if any(_soft_iou(a, props[k]) > 0.7 for k in keep):
            continue
        keep.append(j)
    if debug:
        print(f"    [boxref] {len(props)} props -> kept {len(keep)} "
              f"areas={[round(float(props[k].mean()), 3) for k in keep]}")
    R = np.zeros((H, W), bool)
    for k in keep:
        R |= props[k]
    return R, (R.mean() < 0.01)


@torch.no_grad()
def mifr_score(seg_model, images, to_tensor, num_classes, device="cuda:0",
                n_anchors=10, input_transform=None, ref_cache=None,
                score_classes=None, return_probe=False):
    """Per-candidate MIFR score in [0,1] vs the fixed grid reference:
    0.6*softIoU + 0.25*boundaryF + 0.15*coverage. ref_cache is a dict
    keyed by anchor index, shared across candidates with the SAME input
    space (raw refs reused for source/bn_reset/tent/proto_align; spectral
    passes its own input_transform).

    score_classes: iterable of foreground class ids to score against the
      anatomy reference. None (default) = all foreground (pred>=1), the
      legacy whole-foreground behaviour. For MULTI-STRUCTURE tasks the
      router passes only the MIFR-applicable (salient organ/region) classes
      so a focal-lesion channel is NOT judged by the anatomy reference.
    return_probe: also return the applicability probe dict
      {model_frac, ref_frac, cov_ratio} where cov_ratio = ref_frac/model_frac.
      A large cov_ratio (reference covers >> the scored mask) means the
      reference is a much larger structure than the target -> MIFR is
      measuring the wrong thing (the lesion-vs-anatomy mismatch). The router
      uses this to cross-check the user's declaration label-free.

    Returns (score, degenerate_rate) or, if return_probe, that plus the
    probe dict as a third element."""
    from scipy.ndimage import binary_dilation, binary_erosion
    seg_model.eval()
    idx = np.linspace(0, len(images) - 1,
                       min(n_anchors, len(images)), dtype=int)
    sc_set = None if score_classes is None else set(int(c) for c in score_classes)
    scores = []; deg = 0; mfrac = []; rfrac = []
    for i in idx:
        ii = int(i); img = images[ii]
        ref_img = input_transform(img) if input_transform else img
        rgb = _to_rgb_uint8(ref_img)
        if ref_cache is not None and ii in ref_cache:
            R, d = ref_cache[ii]
        else:
            R, d = medsam_grid_reference(rgb, device)
            if ref_cache is not None:
                ref_cache[ii] = (R, d)
        if d:
            deg += 1; continue
        pred = seg_model(to_tensor(img, device)).argmax(dim=1)[0].cpu().numpy()
        if pred.shape != R.shape:
            pt = torch.from_numpy(pred.astype(np.int64))[None, None].float()
            pred = F.interpolate(pt, size=R.shape, mode="nearest")[0, 0].numpy().astype(np.int64)
        M = np.isin(pred, list(sc_set)) if sc_set is not None else (pred >= 1)
        mfrac.append(float(M.mean())); rfrac.append(float(R.mean()))
        iou = _soft_iou(M, R)
        eM = M ^ binary_erosion(M); eR = R ^ binary_erosion(R)
        dR = binary_dilation(eR, iterations=3); dM = binary_dilation(eM, iterations=3)
        prec = (eM & dR).sum() / (eM.sum() + 1e-6)
        rec = (eR & dM).sum() / (eR.sum() + 1e-6)
        bf = 2 * prec * rec / (prec + rec + 1e-6)
        cov = 1 - abs(M.sum() - R.sum()) / (M.sum() + R.sum() + 1e-6)
        scores.append(0.6 * iou + 0.25 * bf + 0.15 * cov)
    score = float(np.mean(scores)) if scores else 0.0
    deg_rate = deg / max(len(idx), 1)
    if return_probe:
        mf = float(np.mean(mfrac)) if mfrac else 0.0
        rf = float(np.mean(rfrac)) if rfrac else 0.0
        probe = {"model_frac": mf, "ref_frac": rf,
                 "cov_ratio": (rf / mf) if mf > 1e-6 else float("inf")}
        return score, deg_rate, probe
    return score, deg_rate
