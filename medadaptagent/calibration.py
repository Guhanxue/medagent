"""Source-model preprocessing calibration probe.

The "auto-fix for new sites" promise requires the pipeline to figure
out what input preprocessing the source model expects — without that,
even a well-architected adaptation will silently produce garbage when
the inference-time preprocessing mismatches what the model was trained
with.

This module provides `probe_source_preprocessing()`: given a source
model + a small sample of source-domain images (with or without
labels), try each candidate preprocessing convention, score it by
accuracy/Dice (labeled mode) or entropy sharpness (unlabeled mode),
and return the best.

The detected preprocessing is then installed on the TaskAdapter via
`task.set_source_preprocessing(...)` and used by `task.to_tensor()`
throughout the pipeline. Tools (Tent / SAR / EATA / etc.) get correctly-
preprocessed inputs without per-tool changes.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_chw_01(img: np.ndarray, size: int) -> np.ndarray:
    """Resize + scale to [0,1] CHW float32. Handles uint8 and float
    inputs, grayscale auto-expanded to 3 channels."""
    if img.max() <= 1.0 + 1e-6 and img.dtype.kind == "f":
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        u8 = img.astype(np.uint8)
    pil = PILImage.fromarray(u8)
    pil = pil.resize((size, size), PILImage.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr.transpose(2, 0, 1)  # CHW


# --------------------------------------------------------------
# Candidate preprocessings. Each is `(img HWC, size, device) ->
# tensor [1,3,H,W]`. To add a new candidate, append to the dict.
# --------------------------------------------------------------

def _pp_imagenet(img: np.ndarray, size: int, device: str) -> torch.Tensor:
    chw = _resize_chw_01(img, size)
    chw = (chw - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)
    return torch.from_numpy(chw).unsqueeze(0).to(device)


def _pp_raw01(img: np.ndarray, size: int, device: str) -> torch.Tensor:
    chw = _resize_chw_01(img, size)
    return torch.from_numpy(chw).unsqueeze(0).to(device)


def _pp_zscore(img: np.ndarray, size: int, device: str) -> torch.Tensor:
    chw = _resize_chw_01(img, size)
    mean = float(chw.mean()); std = float(chw.std()) + 1e-6
    chw = (chw - mean) / std
    return torch.from_numpy(chw.astype(np.float32)).unsqueeze(0).to(device)


def _pp_percentile_clip(img: np.ndarray, size: int, device: str) -> torch.Tensor:
    chw = _resize_chw_01(img, size)
    lo, hi = np.percentile(chw, [1, 99])
    chw = np.clip((chw - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    return torch.from_numpy(chw).unsqueeze(0).to(device)


def _pp_no_normalize_uint8(img: np.ndarray, size: int, device: str) -> torch.Tensor:
    """Pass uint8 [0,255] without scaling — for models that consumed
    raw uint8 / HU values."""
    if img.max() <= 1.0 + 1e-6 and img.dtype.kind == "f":
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        u8 = img.astype(np.uint8)
    pil = PILImage.fromarray(u8)
    pil = pil.resize((size, size), PILImage.BILINEAR)
    arr = np.array(pil).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _pp_ct_window_01(img: np.ndarray, size: int, device: str) -> torch.Tensor:
    """CT-windowed scaling: clip HU to [-1024, 1024], scale to [0, 1].
    Soft-tissue window default; can be reconfigured by per-task tools.
    Input may be a 3D volume (D, H, W) or 2D image (H, W). 3D returns
    [1, 1, D, H, W]; 2D returns [1, 3, H, W] (broadcast)."""
    arr = img.astype(np.float32)
    # CT HU window (covers most diagnostic ranges); narrower windows can
    # be set per-task via preprocessing.json overrides.
    arr = np.clip(arr, -1024.0, 1024.0)
    arr = (arr + 1024.0) / 2048.0  # to [0, 1]
    if arr.ndim == 3:
        # 3D volume (D, H, W). Return [1, 1, D, H, W] — MONAI 3D UNets
        # take single-channel CT.
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
        return t
    # 2D fallback — broadcast to 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _pp_ct_window_soft_tissue(img: np.ndarray, size: int, device: str) -> torch.Tensor:
    """Narrow soft-tissue CT window [-57, 164] HU (MONAI spleen bundle's
    ScaleIntensityRanged convention). Optimal for spleen / liver / kidney
    segmentation where these organs have HU around 30-70."""
    arr = img.astype(np.float32)
    arr = np.clip(arr, -57, 164)
    arr = (arr + 57) / (164 + 57)  # to [0, 1]
    if arr.ndim == 3:
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


PREPROCESSING_CANDIDATES: Dict[str, Callable] = {
    "imagenet_norm":      _pp_imagenet,
    "raw_01":             _pp_raw01,
    "zscore_per_image":   _pp_zscore,
    "percentile_clip_01": _pp_percentile_clip,
    "no_normalize_uint8": _pp_no_normalize_uint8,
    "ct_window_01":       _pp_ct_window_01,
    "ct_window_soft_tissue": _pp_ct_window_soft_tissue,
}


# --------------------------------------------------------------
# MONAI chain preprocessing — for 3D bundles whose published
# preprocessing is a Compose of (Orientation, Spacing, ScaleIntensity).
# The sidecar declares the chain; we build a Compose at resolve time
# and return a callable matching the standard preprocessing signature.
# Volumes flowing through this fn need affine metadata, supplied via
# the .affine attribute on the input array (Volume wrapper).
# --------------------------------------------------------------

class Volume(np.ndarray):
    """ndarray + .affine attribute. Use Volume.from_array(arr, affine)
    to construct. Compatible with all the simple intensity-only
    preprocessing fns (they treat it as a plain ndarray)."""

    def __new__(cls, input_array, affine=None):
        obj = np.asarray(input_array).view(cls)
        obj.affine = affine
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.affine = getattr(obj, "affine", None)

    @classmethod
    def from_array(cls, arr, affine):
        return cls(arr, affine=affine)


def _build_monai_compose(chain_cfg: List[Dict[str, Any]]):
    """Construct a MONAI Compose from a list of transform specs."""
    from monai import transforms as MT
    transforms = []
    for spec in chain_cfg:
        name = spec["transform"]
        kwargs = {k: v for k, v in spec.items() if k != "transform"}
        kwargs.setdefault("keys", "image")
        cls = getattr(MT, name, None)
        if cls is None:
            raise ValueError(f"Unknown MONAI transform '{name}'")
        transforms.append(cls(**kwargs))
    return MT.Compose(transforms)


def _build_monai_chain_fn(chain_cfg: List[Dict[str, Any]]) -> Callable:
    """Build a preprocessing callable from a list of MONAI transform
    specs. Returns a fn matching (img, size, device) -> torch.Tensor
    with shape [1, C, D, H, W]. Requires `img` to be a Volume (carries
    .affine) — falls back to identity affine if missing."""
    from monai.data import MetaTensor
    compose = _build_monai_compose(chain_cfg)

    def _fn(img: np.ndarray, size: int, device: str) -> torch.Tensor:
        arr = np.asarray(img).astype(np.float32)
        affine = getattr(img, "affine", None)
        if affine is None:
            affine = np.eye(4, dtype=np.float32)
        mt = MetaTensor(torch.from_numpy(arr.copy()),
                          affine=torch.from_numpy(np.asarray(affine).astype(np.float32)))
        out = compose({"image": mt})["image"]
        return out.unsqueeze(0).float().to(device)

    _fn.__monai_chain__ = chain_cfg
    return _fn


def build_label_chain_fn(chain_cfg: List[Dict[str, Any]]) -> Callable:
    """Companion fn for labels — same chain but with nearest interpolation
    expected on Spacingd. Returns (label_arr, affine=None) -> np.ndarray
    shape (D, H, W)."""
    from monai.data import MetaTensor
    compose = _build_monai_compose(chain_cfg)

    def _fn(label: np.ndarray, affine: np.ndarray = None) -> np.ndarray:
        arr = np.asarray(label).astype(np.int64)
        aff = affine if affine is not None else getattr(label, "affine", None)
        aff = aff if aff is not None else np.eye(4, dtype=np.float32)
        mt = MetaTensor(torch.from_numpy(arr.copy()),
                          affine=torch.from_numpy(np.asarray(aff).astype(np.float32)))
        out = compose({"image": mt})["image"]
        return out.squeeze(0).cpu().numpy().astype(np.int64)
    return _fn


# --------------------------------------------------------------
# Scoring helpers — accuracy for cls, mean-Dice for seg
# --------------------------------------------------------------

def _score_cls(model: nn.Module, images, labels, fn: Callable,
                  size: int, device: str, max_samples: int = 30) -> float:
    model.eval()
    correct, n = 0, 0
    for img, lbl in zip(images[:max_samples], labels[:max_samples]):
        if lbl is None:
            continue
        try:
            t = fn(img, size, device)
            with torch.no_grad():
                logits = model(t)
            pred = int(logits.argmax(dim=1).item())
            try:
                true = (int(lbl) if not hasattr(lbl, "shape")
                          else int(np.asarray(lbl).flat[0]))
            except (TypeError, ValueError):
                continue
            correct += int(pred == true); n += 1
        except Exception:
            continue
    return correct / max(n, 1)


def _score_seg(model: nn.Module, images, labels, num_classes: int,
                  fn: Callable, size: int, device: str,
                  max_samples: int = 20) -> float:
    model.eval()
    dices_all = []
    for img, lbl in zip(images[:max_samples], labels[:max_samples]):
        if lbl is None:
            continue
        try:
            t = fn(img, size, device)
            with torch.no_grad():
                logits = model(t)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            # Resize lbl to pred shape
            lbl_pil = PILImage.fromarray(lbl.astype(np.int32).astype(np.uint8))
            lbl_r = np.array(lbl_pil.resize(pred.shape[::-1],
                                              PILImage.NEAREST))
            dices = []
            for c in range(1, num_classes):
                p = (pred == c).astype(float); g = (lbl_r == c).astype(float)
                if g.sum() > 0:
                    d = float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))
                    dices.append(d)
            if dices:
                dices_all.append(float(np.mean(dices)))
        except Exception:
            continue
    return float(np.mean(dices_all)) if dices_all else 0.0


def _score_unlabeled_entropy(model: nn.Module, images,
                                  fn: Callable, size: int, device: str,
                                  max_samples: int = 20) -> float:
    """Entropy-based fallback when no labels available. Correctly-
    preprocessed input → sharper logits → lower entropy.

    Limitation: a confidently-wrong model can give low entropy on bad
    inputs. Use only as fallback when labels aren't supplied.
    Returns NEGATIVE mean entropy so higher = better."""
    model.eval()
    ents = []
    for img in images[:max_samples]:
        try:
            t = fn(img, size, device)
            with torch.no_grad():
                logits = model(t)
                if logits.ndim > 2:
                    probs = F.softmax(logits, dim=1).mean(dim=[2, 3])
                else:
                    probs = F.softmax(logits, dim=1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
                ents.append(ent)
        except Exception:
            continue
    return -float(np.mean(ents)) if ents else -10.0


# --------------------------------------------------------------
# Public probe entry point
# --------------------------------------------------------------

def probe_source_preprocessing(
    model: nn.Module,
    source_images: List[np.ndarray],
    source_labels: Optional[List[Any]],
    task,
    num_classes: int,
    device: str,
    candidate_sizes: Tuple[int, ...] = (224, 256),
    candidates: Optional[Dict[str, Callable]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Find the preprocessing convention that best fits the source model.

    Args:
        source_images : list of HWC images (uint8 or float [0,1]) from
                         the SOURCE training distribution. ~30 is enough.
        source_labels : list of GT labels (int for cls, ndarray mask
                         for seg) for the same images. If None or empty,
                         falls back to entropy-based probe.
        task           : TaskAdapter instance (we only read task.name).
        candidate_sizes: input resolutions to try (224 for ImageNet-
                         style, 256 for seg-style by default).

    Returns:
        dict with `best_name`, `best_size`, `best_fn`, `score`,
        `all_scores`, `mode` (labeled_probe | entropy_probe).
    """
    if candidates is None:
        candidates = PREPROCESSING_CANDIDATES

    use_labels = source_labels is not None and any(
        l is not None for l in source_labels)

    all_scores: Dict[Tuple[int, str], float] = {}
    for sz in candidate_sizes:
        for name, fn in candidates.items():
            if use_labels:
                if task.name == "classification":
                    s = _score_cls(model, source_images, source_labels,
                                      fn, sz, device)
                elif task.name == "segmentation":
                    s = _score_seg(model, source_images, source_labels,
                                      num_classes, fn, sz, device)
                else:
                    s = 0.0
            else:
                s = _score_unlabeled_entropy(model, source_images,
                                                fn, sz, device)
            all_scores[(sz, name)] = s
            if verbose:
                print(f"    probe[{sz},{name:22s}] = {s:+.4f}")
    best_key = max(all_scores, key=all_scores.get)
    return {
        "best_size": best_key[0],
        "best_name": best_key[1],
        "best_fn": candidates[best_key[1]],
        "score": all_scores[best_key],
        "all_scores": {f"{sz}_{name}": v for (sz, name), v in all_scores.items()},
        "mode": "labeled_probe" if use_labels else "entropy_probe",
    }


__all__ = [
    "PREPROCESSING_CANDIDATES",
    "probe_source_preprocessing",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]


# --------------------------------------------------------------
# Tier 1: documented metadata. Look for explicit preprocessing
# config next to the checkpoint or inside the state_dict.
# --------------------------------------------------------------
import os, json


PREPROCESSING_DOC_FILES = (
    "preprocessing.json",   # explicit user-supplied
    "model_config.json",
    "config.json",          # HuggingFace convention
)


def _normalize_preprocessing_name(name: str) -> Optional[str]:
    """Map common doc strings to candidate names."""
    if name is None:
        return None
    n = str(name).strip().lower()
    aliases = {
        "imagenet": "imagenet_norm", "imagenet_norm": "imagenet_norm",
        "imagenet_normalize": "imagenet_norm",
        "ilsvrc_norm": "imagenet_norm",
        "0_1": "raw_01", "raw01": "raw_01", "raw_01": "raw_01",
        "01": "raw_01", "scale_01": "raw_01", "minmax": "raw_01",
        "zscore": "zscore_per_image", "z_score": "zscore_per_image",
        "zscore_per_image": "zscore_per_image",
        "percentile_clip": "percentile_clip_01",
        "percentile_clip_01": "percentile_clip_01",
        "robust_01": "percentile_clip_01",
        "none": "no_normalize_uint8", "raw": "no_normalize_uint8",
        "uint8": "no_normalize_uint8",
        "no_normalize": "no_normalize_uint8",
        "ct_window": "ct_window_01",
        "ct_window_01": "ct_window_01",
        "ct_window_soft_tissue": "ct_window_soft_tissue",
        "ct_soft_tissue": "ct_window_soft_tissue",
        "monai_spleen": "ct_window_soft_tissue",
        "monai_chain": "monai_chain",
    }
    return aliases.get(n)


def find_preprocessing_metadata(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """Tier 1: look for explicit preprocessing documentation alongside
    the checkpoint. Returns a dict with `name` (canonical candidate
    name) + `size` if found, else None.

    Checks (in order):
      a) `<ckpt_dir>/preprocessing.json`
      b) `<ckpt_dir>/<basename>.meta.json`  (our training-time meta)
      c) `<ckpt_dir>/meta.json`             (sibling)
      d) state_dict's `_meta` / `meta` / `config` key
      e) HuggingFace `config.json` next to ckpt
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    ckpt_dir = os.path.dirname(ckpt_path)
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    # (a)-(e): try each
    for fname in (f"{base}.meta.json", f"{base}.preprocessing.json",
                    "preprocessing.json", "meta.json", "config.json",
                    "model_config.json"):
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        # Look for known fields
        for key in ("preprocessing", "input_preprocessing",
                      "training_preprocessing", "input_format"):
            if key in doc:
                name = _normalize_preprocessing_name(doc[key])
                if name:
                    return {"name": name, "size": doc.get("input_size", 224),
                              "source_file": path,
                              "raw_doc": doc,
                              "chain": doc.get("preprocessing_chain"),
                              "label_chain": doc.get("label_chain")}
        # HF-style: image_processor or feature_extractor key indicates ImageNet
        if "image_processor" in doc or "feature_extractor" in doc:
            return {"name": "imagenet_norm",
                      "size": doc.get("image_size", 224),
                      "source_file": path,
                      "raw_doc": doc}
    # (d): try state_dict metadata
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(ckpt, dict):
        for key in ("_meta", "meta", "config", "preprocessing"):
            if key in ckpt:
                val = ckpt[key]
                if isinstance(val, dict):
                    for k2 in ("preprocessing", "input_format"):
                        if k2 in val:
                            name = _normalize_preprocessing_name(val[k2])
                            if name:
                                return {"name": name,
                                          "size": val.get("input_size", 224),
                                          "source_file": "state_dict",
                                          "raw_doc": val}
    return None


# --------------------------------------------------------------
# Tier 2: architecture convention
# --------------------------------------------------------------

ARCH_CONVENTIONS = {
    # torchvision / timm classifiers — always ImageNet normalization
    "resnet18":  {"name": "imagenet_norm", "size": 224},
    "resnet50":  {"name": "imagenet_norm", "size": 224},
    "vit_small": {"name": "imagenet_norm", "size": 224},
    # In-house seg models — trained with [0,1] raw scaling
    "unet":      {"name": "raw_01", "size": 256},
    "stunet":    {"name": "raw_01", "size": 256},
    "nnunet":    {"name": "raw_01", "size": 256},
    # MONAI 3D UNet — trained with CT-windowed [0,1] intensity range
    # (typical: [-1024 HU, 1024 HU] -> clip to [-200, 200] -> scale [0,1]).
    # Inference uses 96^3 patches with sliding-window overlap.
    "monai_unet3d":  {"name": "ct_window_01", "size": 96},
}


def infer_preprocessing_from_arch(arch: str) -> Optional[Dict[str, Any]]:
    """Tier 2: return the known convention for a registered arch, if any."""
    return ARCH_CONVENTIONS.get(arch)


# --------------------------------------------------------------
# Orchestrator: three-tier resolution
# --------------------------------------------------------------

def resolve_source_preprocessing(
    ckpt_path: Optional[str],
    arch: Optional[str],
    model: nn.Module,
    source_images: Optional[List[np.ndarray]],
    source_labels: Optional[List[Any]],
    task,
    num_classes: int,
    device: str,
    candidate_sizes: Tuple[int, ...] = (224, 256),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Three-tier resolution. Returns dict with `name`, `size`, `fn`,
    `tier` (documented | arch_convention | probed), plus debugging."""
    # Tier 1
    if ckpt_path:
        doc = find_preprocessing_metadata(ckpt_path)
        if doc:
            name = doc["name"]
            chain = doc.get("chain")
            label_chain = doc.get("label_chain")
            if chain:
                # Chain overrides simple candidate — build MONAI Compose
                fn = _build_monai_chain_fn(chain)
                if verbose:
                    print(f"  preprocessing (tier 1 - documented chain, "
                           f"{len(chain)} transforms): {[t['transform'] for t in chain]} "
                           f"from {doc.get('source_file')}")
                out = {
                    "tier": "documented", "name": name,
                    "size": int(doc.get("size", 96)),
                    "fn": fn,
                    "chain": chain,
                    "details": doc,
                }
                if label_chain:
                    out["label_fn"] = build_label_chain_fn(label_chain)
                    out["label_chain"] = label_chain
                return out
            if verbose:
                print(f"  preprocessing (tier 1 — documented): {name} "
                       f"from {doc.get('source_file')}")
            return {
                "tier": "documented", "name": name,
                "size": int(doc.get("size", 224)),
                "fn": PREPROCESSING_CANDIDATES[name],
                "details": doc,
            }
    # Tier 2
    if arch:
        conv = infer_preprocessing_from_arch(arch)
        if conv:
            if verbose:
                print(f"  preprocessing (tier 2 — arch convention "
                       f"'{arch}'): {conv['name']}")
            return {
                "tier": "arch_convention", "name": conv["name"],
                "size": int(conv["size"]),
                "fn": PREPROCESSING_CANDIDATES[conv["name"]],
                "details": {"arch": arch},
            }
    # Tier 3
    if not source_images:
        # Last-resort default: ImageNet for cls, raw_01 for seg
        default_name = ("imagenet_norm" if task.name == "classification"
                          else "raw_01")
        if verbose:
            print(f"  preprocessing (tier 3 fallback — no probe possible): "
                   f"defaulting to {default_name}")
        return {
            "tier": "default_fallback", "name": default_name,
            "size": 224 if default_name == "imagenet_norm" else 256,
            "fn": PREPROCESSING_CANDIDATES[default_name],
            "details": {"reason": "no source images for probing"},
        }
    if verbose:
        print(f"  preprocessing (tier 3 — probing {len(source_images)} samples):")
    res = probe_source_preprocessing(
        model, source_images, source_labels, task, num_classes,
        device, candidate_sizes=candidate_sizes, verbose=verbose)
    if verbose:
        print(f"  best probe: {res['best_name']} @ size {res['best_size']} "
               f"(score={res['score']:+.4f}, mode={res['mode']})")
    return {
        "tier": "probed", "name": res["best_name"],
        "size": res["best_size"], "fn": res["best_fn"],
        "details": res,
    }
