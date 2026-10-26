"""Classification TaskAdapter — image-level classification.

Implements the same TaskAdapter interface as SegmentationTask, but
operates on image-level scalar labels and class predictions.

Logits shape: [B, C] (no spatial dims)
Pred shape:   scalar class index per image
Metric:       accuracy (other options: AUROC, F1; we keep accuracy for
              the gate, log others alongside)
CE:           image-level cross-entropy on K labeled samples
Coverage:     mean predicted-class confidence (proxy for the seg
              fg-ratio sanity signal)
VLM render:   raw image + caption "source: class X (p=...), current:
              class Y (p=...)"  — no mask overlays
"""
from __future__ import annotations
import io
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

from .base import TaskAdapter, TaskRegistry


# ImageNet stats — the standard normalization for torchvision /
# timm classifiers. Cls source models trained with this MUST
# receive normalized inputs at inference time.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _img_to_tensor(img: np.ndarray, device: str, size: int = 224) -> torch.Tensor:
    """Classification: resize, [0,1] scale, ImageNet-normalize. Required
    to match the source classifier's training-time preprocessing."""
    pil = PILImage.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)
                              if img.max() <= 1.0 else img.astype(np.uint8))
    pil = pil.resize((size, size), PILImage.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    # ImageNet normalize
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t.unsqueeze(0).to(device)


def _array_to_b64(arr: np.ndarray, size: int = 256) -> str:
    """Image -> base64 PNG string. Matches vision_utils.array_to_b64
    contract."""
    import base64
    if arr.max() <= 1.0:
        arr8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    else:
        arr8 = arr.astype(np.uint8)
    pil = PILImage.fromarray(arr8)
    pil = pil.resize((size, size), PILImage.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


CLS_VLM_PROMPT = """Compare the SOURCE model's classification with the CURRENT \
adapted model's classification on {k} target samples.

## Task

Each sample has TWO predicted class labels (and confidences). Your job:
look at the RAW IMAGE and judge which prediction (SOURCE vs CURRENT) is
more plausible for the visible content.

For each sample you see ONE image plus a text caption listing the two
predictions:

  Sample i RAW IMAGE
  Sample i CAPTION: "SOURCE: class A (conf=0.85). CURRENT: class B (conf=0.72)."

## Two per-sample judgments

1. **Relative** (CURRENT vs SOURCE): better | similar | worse.
   Which prediction better matches what you see in the raw image?
   - "better" if CURRENT's class label is more anatomically/clinically
     plausible than SOURCE's, given the visible content.
   - "worse" if CURRENT's class shifted to something less plausible.
   - "similar" if both are about equally plausible OR the same class.

2. **Absolute quality of CURRENT alone**: usable | broken.
   Looking at the RAW image, is CURRENT's predicted class plausible?
     - "broken" — the predicted class doesn't match what's visible
       (e.g., visible benign tissue but predicted aggressive tumor).
     - "usable" — predicted class is anatomically/clinically consistent
       with the visible content.

Return ONLY this JSON (no extra text):
```json
{{"per_sample":          ["better|similar|worse", ...],
  "per_sample_absolute": ["usable|broken", ...],
  "note": "<one-sentence summary>"}}
```"""


class ClassificationTask(TaskAdapter):

    @property
    def name(self) -> str:
        return "classification"

    @property
    def metric_name(self) -> str:
        return "accuracy"

    # ----------------------------------------------------------------
    # Forward + prediction
    # ----------------------------------------------------------------

    def forward_pred(self, model: nn.Module, image_tensor: torch.Tensor):
        with torch.no_grad():
            logits = model(image_tensor)           # [B, C]
            if logits.ndim != 2:
                # Some classifiers produce extra dims; collapse safely
                logits = logits.flatten(1)
            pred = logits.argmax(dim=1)            # [B]
        return logits, pred[0].cpu().numpy()        # scalar 0-d array

    # ----------------------------------------------------------------
    # Per-sample correctness (1.0 if predicted class matches GT, else 0.0)
    # ----------------------------------------------------------------

    def sample_metric(self, model: nn.Module, image: np.ndarray,
                        label: Any, num_classes: int, device: str,
                        size: int = 256) -> Optional[float]:
        if label is None:
            return None
        t = _img_to_tensor(image, device, size)
        _, pred = self.forward_pred(model, t)
        try:
            true = int(label) if not hasattr(label, "shape") else int(np.asarray(label).flat[0])
        except (TypeError, ValueError):
            return None
        return 1.0 if int(pred) == true else 0.0

    def avg_metric(self, model: nn.Module, images: List[np.ndarray],
                     labels: List[Any], num_classes: int, device: str,
                     size: int = 256, metric: Optional[str] = None) -> float:
        """Cls metric dispatcher. metric in {None|'accuracy'|'auroc'}.
        None defaults to self.metric_name (accuracy). AUROC is computed
        on the whole batch using softmax probabilities — appropriate
        for imbalanced binary cls where accuracy is misleading."""
        metric = (metric or self.metric_name).lower()
        if metric == "auroc":
            return self._auroc(model, images, labels, num_classes,
                                  device, size)
        # Default: per-sample accuracy via base implementation
        return super().avg_metric(model, images, labels, num_classes,
                                       device, size)

    def _auroc(self, model: nn.Module, images: List[np.ndarray],
                  labels: List[Any], num_classes: int, device: str,
                  size: int = 256) -> float:
        """Binary or multi-class macro-OvR AUROC on a paired set. Returns
        0.5 (chance) if only one class present in labels."""
        import torch.nn.functional as F
        probs, trues = [], []
        for img, lbl in zip(images, labels):
            if lbl is None:
                continue
            try:
                t = int(lbl) if not hasattr(lbl, "shape") else int(np.asarray(lbl).flat[0])
            except (TypeError, ValueError):
                continue
            tensor = _img_to_tensor(img, device, size)
            with torch.no_grad():
                logits, _ = self.forward_pred(model, tensor)
                p = F.softmax(logits, dim=1).cpu().numpy()[0]
            probs.append(p)
            trues.append(t)
        if not probs or len(set(trues)) < 2:
            return 0.5
        probs = np.stack(probs, axis=0)
        trues = np.array(trues)
        try:
            from sklearn.metrics import roc_auc_score
            if num_classes == 2 or probs.shape[1] == 2:
                return float(roc_auc_score(trues, probs[:, 1]))
            return float(roc_auc_score(trues, probs, multi_class="ovr",
                                              average="macro"))
        except Exception:
            return 0.5

    # ----------------------------------------------------------------
    # Image-level CE on K labeled samples for *_ss tools
    # ----------------------------------------------------------------

    def ce_loss_on_labels(self, model, images, labels, indices,
                              preprocess_fn, num_classes, device,
                              size: int = 256):
        if not indices or not labels:
            return torch.tensor(0.0, device=device)
        losses = []
        for i, lbl in zip(indices, labels):
            if lbl is None:
                continue
            img = preprocess_fn(images[i])
            t = _img_to_tensor(img, device, size)
            try:
                cls_idx = int(lbl) if not hasattr(lbl, "shape") else int(np.asarray(lbl).flat[0])
            except (TypeError, ValueError):
                continue
            cls_idx = max(0, min(num_classes - 1, cls_idx))
            target = torch.tensor([cls_idx], device=device, dtype=torch.long)
            logits = model(t)                       # [1, C]
            if logits.ndim != 2:
                logits = logits.flatten(1)
            ce = F.cross_entropy(logits, target, reduction="mean")
            losses.append(ce)
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    # ----------------------------------------------------------------
    # Coverage signal — for cls we use mean max-confidence as a proxy.
    # Catastrophic collapse for cls would be "all predictions on one
    # class" or "all predictions zero confidence" — return mean conf.
    # ----------------------------------------------------------------

    def coverage_signal(self, model: nn.Module,
                            images: List[np.ndarray],
                            device: str,
                            max_samples: int = 200) -> Optional[float]:
        confs = []
        n = min(max_samples, len(images))
        for img in images[:n]:
            t = _img_to_tensor(img, device)
            with torch.no_grad():
                logits = model(t)
                if logits.ndim != 2:
                    logits = logits.flatten(1)
                p = F.softmax(logits, dim=1)
                confs.append(float(p.max(dim=1).values.mean()))
        return float(np.mean(confs)) if confs else None

    # ----------------------------------------------------------------
    # VLM rendering for cls: raw image + caption with class predictions
    # ----------------------------------------------------------------

    def render_pair_for_vlm(self, model_source: nn.Module,
                                model_current: nn.Module,
                                anchor_imgs: List[np.ndarray],
                                num_classes: int,
                                device: str,
                                size: int = 224) -> Tuple[List[str], List[str]]:
        """For each anchor, render THREE images: raw + source GradCAM
        overlay + current GradCAM overlay. If GradCAM is unavailable
        for the model (e.g., unusual architecture), fall back to raw-
        plus-caption so the VLM still has something to judge."""
        from medadaptagent.cls_vlm_render import render_cls_attention
        images, labels = [], []
        # Detect ViT-style models for GradCAM reshape transform
        is_vit_src = hasattr(model_source, "blocks")
        is_vit_cur = hasattr(model_current, "blocks")
        for i, img in enumerate(anchor_imgs):
            t = self.to_tensor(img, device, size).requires_grad_(True)
            with torch.no_grad():
                src_logits, src_pred = self.forward_pred(model_source, t)
                cur_logits, cur_pred = self.forward_pred(model_current, t)
                src_p = F.softmax(src_logits, dim=1).max().item()
                cur_p = F.softmax(cur_logits, dim=1).max().item()
            # Generate GradCAM overlays (re-derive grad-enabled tensors
            # since the no_grad forward above blocked the tape).
            t_src = self.to_tensor(img, device, size)
            t_cur = self.to_tensor(img, device, size)
            src_overlay, raw_b64 = render_cls_attention(
                model_source, img, t_src, int(src_pred),
                is_vit=is_vit_src, size=size)
            cur_overlay, _ = render_cls_attention(
                model_current, img, t_cur, int(cur_pred),
                is_vit=is_vit_cur, size=size)
            # Triplet per sample
            images.append(raw_b64)
            labels.append(f"Sample {i+1} RAW IMAGE")
            if src_overlay is not None:
                images.append(src_overlay)
                labels.append(
                    f"Sample {i+1} SOURCE attention map. "
                    f"SOURCE predicted class {int(src_pred)} "
                    f"(conf={src_p:.2f})")
            else:
                # Fall back to raw image as placeholder + caption
                images.append(raw_b64)
                labels.append(
                    f"Sample {i+1} SOURCE (no attention map). "
                    f"SOURCE predicted class {int(src_pred)} "
                    f"(conf={src_p:.2f})")
            if cur_overlay is not None:
                images.append(cur_overlay)
                labels.append(
                    f"Sample {i+1} CURRENT attention map. "
                    f"CURRENT predicted class {int(cur_pred)} "
                    f"(conf={cur_p:.2f})")
            else:
                images.append(raw_b64)
                labels.append(
                    f"Sample {i+1} CURRENT (no attention map). "
                    f"CURRENT predicted class {int(cur_pred)} "
                    f"(conf={cur_p:.2f})")
        return images, labels

    def vlm_prompt_template(self) -> str:
        from medadaptagent.cls_vlm_render import CLS_VLM_PROMPT_V2
        return CLS_VLM_PROMPT_V2

    def task_capabilities(self) -> List[str]:
        return ["scalar_predictions"]

    def to_tensor(self, image: np.ndarray, device: str,
                    size: int = 224) -> torch.Tensor:
        """Cls routes through the configured preprocessing if set by
        the calibration probe; otherwise falls back to legacy ImageNet
        norm at 224."""
        if self._source_pp_fn is not None:
            return self._source_pp_fn(image,
                                        self._source_pp_size or size,
                                        device)
        return _img_to_tensor(image, device, size)


TaskRegistry.register("classification", ClassificationTask)
