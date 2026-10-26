"""External model loader — turn the pipeline into a drop-in adapter
for any pre-trained model, not just our in-house seg/cls archs.

Supports:
  (a) Local checkpoint .pt files. Auto-detects the architecture from
      state_dict keys when possible (resnet18, resnet50, vit_small,
      unet, stunet, nnunet); falls back to user-supplied `arch` hint.

  (b) HuggingFace model identifiers like
      "microsoft/swin-tiny-patch4-window7-224". Uses transformers.
      AutoModelForImageClassification or AutoModelForSemanticSegmentation.

  (c) timm model names like "resnet50", "vit_small_patch16_224" loaded
      with weights via `timm.create_model(..., pretrained=True)` or from
      a sibling .pt file with `pretrained=False`.

  (d) Existing in-house archs (unet, stunet, nnunet, resnet18/50,
      vit_small) — falls through to scripts/_source_loader.py.

Output contract:
  load_any_model(spec) -> (model, info)
  where info dict includes:
      arch             — best-guess architecture name
      detected_format  — local_pt | huggingface | timm | builtin
      num_classes      — inferred class count where determinable
      input_size       — suggested input resolution
      preprocessing_hint  — best guess (imagenet_norm | raw_01 | ...)
                            handed off to the calibration probe
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


KNOWN_INHOUSE_ARCHS = {"unet", "stunet", "nnunet",
                          "resnet18", "resnet50", "vit_small"}


# --------------------------------------------------------------
# State-dict shape heuristics — recognize common architectures
# from key patterns alone. Useful when the user dumps a .pt with
# no metadata.
# --------------------------------------------------------------

# Fingerprints: keys that MUST be present (positive) and keys that
# MUST be absent (negative). Order matters — first match wins, so the
# more-specific arch (resnet50 has conv3 layers; resnet18 does not)
# is checked first.
_ARCH_FINGERPRINTS = {
    "resnet50":  {
        "present": ["layer1.0.conv3.weight", "layer4.2.conv3.weight", "fc.weight"],
        "absent":  [],
    },
    "resnet18":  {
        "present": ["layer1.0.conv1.weight", "layer4.1.conv2.weight", "fc.weight"],
        "absent":  ["layer1.0.conv3.weight"],   # resnet18 has no Bottleneck conv3
    },
    "vit_small": {
        "present": ["blocks.0.attn.qkv.weight", "blocks.11.norm1.weight"],
        "absent":  [],
    },
    "swin":      {"present": ["layers.0.blocks.0.attn.qkv.weight"], "absent": []},
    "convnext":  {"present": ["stages.0.blocks.0.dwconv.weight"], "absent": []},
    "unet":      {"present": ["encoder.0.0.weight", "decoder.0.up.weight"], "absent": []},
    "nnunet":    {"present": ["encoder.0.conv1.weight"], "absent": []},
}


def fingerprint_state_dict(state: Dict[str, torch.Tensor]) -> Optional[str]:
    """Best-guess architecture from state_dict keys. Each fingerprint
    has present-keys + absent-keys; both conditions must hold. Returns
    None if no fingerprint matches (conservative — no answer over wrong)."""
    if not isinstance(state, dict):
        return None
    keys = set(state.keys())
    for arch, sigs in _ARCH_FINGERPRINTS.items():
        all_present = all(any(sig in k for k in keys) for sig in sigs.get("present", []))
        none_absent = all(not any(sig in k for k in keys) for sig in sigs.get("absent", []))
        if all_present and none_absent:
            return arch
    return None


# --------------------------------------------------------------
# Path (a): local .pt with auto-detection
# --------------------------------------------------------------

def _load_local_pt(path: str, num_classes: int, device: str,
                     arch_hint: Optional[str] = None,
                     project_root: Optional[str] = None
                     ) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a local checkpoint. Uses arch_hint if provided; otherwise
    fingerprints state_dict to detect arch automatically."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = (ckpt.get("model_state_dict", ckpt)
              if isinstance(ckpt, dict) else ckpt)
    # Determine arch
    arch = arch_hint
    if arch is None:
        arch = fingerprint_state_dict(state)
    if arch is None:
        raise ValueError(
            f"Could not auto-detect architecture for {path}. "
            f"Provide `arch_hint` explicitly.")
    if arch not in KNOWN_INHOUSE_ARCHS and arch in ("swin", "convnext"):
        # timm fallback for fingerprinted-but-not-builtin archs
        return _load_timm_model(arch, num_classes, device,
                                   state_dict=state)
    # Builtin path
    from scripts._source_loader import load_source_model
    model = load_source_model(path, num_classes, device, arch=arch,
                                project_root=project_root)
    info = {"arch": arch, "detected_format": "local_pt",
              "num_classes": num_classes, "input_size": 224 if
              arch in ("resnet18", "resnet50", "vit_small") else 256,
              "preprocessing_hint": ("imagenet_norm" if arch in
                                       ("resnet18","resnet50","vit_small")
                                       else "raw_01")}
    return model, info


# --------------------------------------------------------------
# Path (b): HuggingFace model
# --------------------------------------------------------------

def _load_huggingface_model(name: str, num_classes: int, device: str,
                                task_hint: str = "classification"
                                ) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a HuggingFace model. Examples:
        microsoft/swin-tiny-patch4-window7-224
        nvidia/mit-b0
    Requires `transformers` package."""
    try:
        from transformers import (
            AutoModelForImageClassification, AutoModelForSemanticSegmentation,
            AutoConfig, AutoImageProcessor,
        )
    except ImportError as e:
        raise ImportError(
            "HuggingFace loader requires `pip install transformers`") from e

    if task_hint == "segmentation":
        model = AutoModelForSemanticSegmentation.from_pretrained(
            name, num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForImageClassification.from_pretrained(
            name, num_labels=num_classes, ignore_mismatched_sizes=True)
    model = model.to(device).eval()
    # Try to pull input size from the model's image processor
    try:
        proc = AutoImageProcessor.from_pretrained(name)
        size = (proc.size.get("shortest_edge") or
                 proc.size.get("height") or 224)
    except Exception:
        size = 224
    info = {
        "arch": name, "detected_format": "huggingface",
        "num_classes": num_classes, "input_size": int(size),
        "preprocessing_hint": "imagenet_norm",
    }
    return model, info


# --------------------------------------------------------------
# Path (c): timm model
# --------------------------------------------------------------

def _load_timm_model(name: str, num_classes: int, device: str,
                        pretrained: bool = False,
                        state_dict: Optional[Dict] = None
                        ) -> Tuple[nn.Module, Dict[str, Any]]:
    import timm
    model = timm.create_model(name, pretrained=pretrained,
                                num_classes=num_classes).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    info = {
        "arch": name, "detected_format": "timm",
        "num_classes": num_classes, "input_size": 224,
        "preprocessing_hint": "imagenet_norm",
    }
    return model, info


# --------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------

@dataclass
class ModelSpec:
    """How the user identifies the model to load.

    One of:
      - path: local .pt path
      - hf: HuggingFace identifier
      - timm: timm model name + optional weights path
    """
    path: Optional[str] = None
    hf: Optional[str] = None
    timm: Optional[str] = None
    arch_hint: Optional[str] = None
    task_hint: str = "classification"  # for HF segmentation vs cls dispatch


def load_any_model(spec: Union[ModelSpec, Dict, str],
                     num_classes: int, device: str = "cuda:0",
                     project_root: Optional[str] = None
                     ) -> Tuple[nn.Module, Dict[str, Any]]:
    """Unified entry point. `spec` may be:
        - ModelSpec dataclass
        - dict with keys path|hf|timm
        - str: tries path first; if file doesn't exist, tries HF;
          otherwise raises.
    Returns (model_on_device, info_dict)."""
    # Normalize
    if isinstance(spec, str):
        if os.path.exists(spec):
            spec = ModelSpec(path=spec)
        elif "/" in spec:
            spec = ModelSpec(hf=spec)
        else:
            spec = ModelSpec(timm=spec)
    elif isinstance(spec, dict):
        spec = ModelSpec(**spec)
    # Dispatch
    if spec.path:
        return _load_local_pt(spec.path, num_classes, device,
                                 arch_hint=spec.arch_hint,
                                 project_root=project_root)
    if spec.hf:
        return _load_huggingface_model(spec.hf, num_classes, device,
                                            task_hint=spec.task_hint)
    if spec.timm:
        return _load_timm_model(spec.timm, num_classes, device)
    raise ValueError(f"Empty ModelSpec — provide path / hf / timm")


__all__ = ["ModelSpec", "load_any_model", "fingerprint_state_dict",
            "_load_huggingface_model", "_load_timm_model"]
