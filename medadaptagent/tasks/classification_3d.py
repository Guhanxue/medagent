"""Classification3D TaskAdapter — 3D image-level binary/multiclass cls.

Mirrors ClassificationTask but for 3D inputs:
  Input shape: (D, H, W) np.ndarray
  Tensor shape: (1, 1, D, H, W) — single-channel volumetric
  Logits: [B, C] (no spatial dims)
  Pred: scalar class index per volume
  Metric: accuracy (gate); AUROC computable downstream
  CE: volume-level cross-entropy on K labeled samples
  Coverage: mean predicted-class confidence
  VLM render: middle-slice triplet (axial / coronal / sagittal) +
    caption "source: class X (p=...), current: class Y (p=...)"
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


def _vol_to_tensor(vol: np.ndarray, device: str) -> torch.Tensor:
    """3D cls input: (D, H, W) -> (1, 1, D, H, W). Assumes the volume
    is already z-score normalized by the data prep step (per-case z)."""
    arr = vol.astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]  # drop spurious leading channel
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _mid_slice_b64(vol: np.ndarray, axis: int = 0, size: int = 224) -> str:
    """Render middle slice along `axis` as base64 PNG for VLM."""
    import base64
    if axis == 0:
        slc = vol[vol.shape[0] // 2]
    elif axis == 1:
        slc = vol[:, vol.shape[1] // 2, :]
    else:
        slc = vol[:, :, vol.shape[2] // 2]
    lo, hi = float(slc.min()), float(slc.max())
    if hi - lo < 1e-6:
        slc8 = np.zeros_like(slc, dtype=np.uint8)
    else:
        slc8 = ((slc - lo) / (hi - lo) * 255).astype(np.uint8)
    pil = PILImage.fromarray(slc8)
    pil = pil.resize((size, size), PILImage.BILINEAR)
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


CLS3D_VLM_PROMPT = """Compare SOURCE vs CURRENT 3D classifier predictions \
on {k} target volumes.

## Task

Each volume has TWO predicted class labels (and confidences) from a
SOURCE and a CURRENT (adapted) model. Your job: look at the volume's
mid-slice views and judge which prediction is more plausible.

For each sample you see THREE views (axial, coronal, sagittal middle
slices) plus a caption with the two predictions:

  Sample i AXIAL / CORONAL / SAGITTAL mid-slice
  Sample i CAPTION: "SOURCE: class A (conf=0.85). CURRENT: class B (conf=0.72)."

## Two per-sample judgments

1. **Relative** (CURRENT vs SOURCE): better | similar | worse.
2. **Absolute quality of CURRENT alone**: usable | broken.

Return ONLY this JSON:
```json
{{"per_sample":          ["better|similar|worse", ...],
  "per_sample_absolute": ["usable|broken", ...],
  "note": "<one-sentence summary>"}}
```"""


class Classification3DTask(TaskAdapter):

    @property
    def name(self) -> str:
        return "classification_3d"

    @property
    def metric_name(self) -> str:
        return "accuracy"

    # ----------------------------------------------------------------
    # Forward + prediction (same as 2D cls: logits [B,C] -> argmax)
    # ----------------------------------------------------------------

    def forward_pred(self, model: nn.Module, image_tensor: torch.Tensor):
        with torch.no_grad():
            logits = model(image_tensor)
            if logits.ndim > 2:
                logits = logits.flatten(1)
            pred = logits.argmax(dim=1)
        return logits, pred[0].cpu().numpy()

    # ----------------------------------------------------------------
    # Per-sample correctness (1.0 if predicted class matches GT, else 0.0)
    # ----------------------------------------------------------------

    def sample_metric(self, model: nn.Module, image: np.ndarray,
                        label: Any, num_classes: int, device: str,
                        size: int = 96) -> Optional[float]:
        if label is None:
            return None
        t = self.to_tensor(image, device, size)
        _, pred = self.forward_pred(model, t)
        try:
            true = int(label) if not hasattr(label, "shape") \
                    else int(np.asarray(label).flat[0])
        except (TypeError, ValueError):
            return None
        return 1.0 if int(pred) == true else 0.0

    def avg_metric(self, model: nn.Module, images: List[np.ndarray],
                     labels: List[Any], num_classes: int, device: str,
                     size: int = 96, metric: Optional[str] = None) -> float:
        """3D cls metric dispatcher. metric in {None|'accuracy'|'auroc'}.
        AUROC computed on the batch from softmax probabilities — the
        correct metric for imbalanced binary cls (e.g., PI-CAI csPCa
        prevalence ~30%, where predicting all-negative gives ~70% acc)."""
        metric = (metric or self.metric_name).lower()
        if metric == "auroc":
            return self._auroc(model, images, labels, num_classes,
                                  device, size)
        return super().avg_metric(model, images, labels, num_classes,
                                       device, size)

    def _auroc(self, model: nn.Module, images: List[np.ndarray],
                  labels: List[Any], num_classes: int, device: str,
                  size: int = 96) -> float:
        """Binary or multi-class macro-OvR AUROC on a paired set of 3D
        volumes. Returns 0.5 (chance) if only one class present."""
        import torch.nn.functional as F
        probs, trues = [], []
        for img, lbl in zip(images, labels):
            if lbl is None:
                continue
            try:
                t = int(lbl) if not hasattr(lbl, "shape") \
                        else int(np.asarray(lbl).flat[0])
            except (TypeError, ValueError):
                continue
            tensor = self.to_tensor(img, device, size)
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
    # CE loss on K labeled VOLUMES for *_ss / finetune tools
    # ----------------------------------------------------------------

    def ce_loss_on_labels(self, model, images, labels, indices,
                              preprocess_fn, num_classes, device,
                              size: int = 96):
        if not indices or not labels:
            return torch.tensor(0.0, device=device)
        losses = []
        for i, lbl in zip(indices, labels):
            if lbl is None:
                continue
            img = preprocess_fn(images[i])
            t = self.to_tensor(img, device, size)
            try:
                cls_idx = (int(lbl) if not hasattr(lbl, "shape")
                              else int(np.asarray(lbl).flat[0]))
            except (TypeError, ValueError):
                continue
            cls_idx = max(0, min(num_classes - 1, cls_idx))
            target = torch.tensor([cls_idx], device=device,
                                     dtype=torch.long)
            logits = model(t)
            if logits.ndim > 2:
                logits = logits.flatten(1)
            ce = F.cross_entropy(logits, target, reduction="mean")
            losses.append(ce)
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    # ----------------------------------------------------------------
    # Coverage signal — mean max-prob (proxy for per-class collapse)
    # ----------------------------------------------------------------

    def coverage_signal(self, model: nn.Module,
                            images: List[np.ndarray],
                            device: str,
                            max_samples: int = 50) -> Optional[float]:
        confs = []
        n = min(max_samples, len(images))
        for vol in images[:n]:
            t = self.to_tensor(vol, device)
            with torch.no_grad():
                logits = model(t)
                if logits.ndim > 2:
                    logits = logits.flatten(1)
                p = F.softmax(logits, dim=1)
                confs.append(float(p.max(dim=1).values.mean()))
        return float(np.mean(confs)) if confs else None

    # ----------------------------------------------------------------
    # VLM rendering: 3-view mid-slice montage per sample + caption
    # ----------------------------------------------------------------

    def render_pair_for_vlm(self, model_source: nn.Module,
                                model_current: nn.Module,
                                anchor_imgs: List[np.ndarray],
                                num_classes: int,
                                device: str,
                                size: int = 224) -> Tuple[List[str], List[str]]:
        images, labels = [], []
        for i, vol in enumerate(anchor_imgs):
            t = self.to_tensor(vol, device, size)
            src_logits, src_pred = self.forward_pred(model_source, t)
            cur_logits, cur_pred = self.forward_pred(model_current, t)
            src_p = F.softmax(src_logits, dim=1).max().item()
            cur_p = F.softmax(cur_logits, dim=1).max().item()
            arr = np.asarray(vol)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            ax_b64 = _mid_slice_b64(arr, axis=0, size=size)
            co_b64 = _mid_slice_b64(arr, axis=1, size=size)
            sa_b64 = _mid_slice_b64(arr, axis=2, size=size)
            cap = (f"Sample {i+1}. SOURCE: class {int(src_pred)} "
                    f"(conf={src_p:.2f}). CURRENT: class {int(cur_pred)} "
                    f"(conf={cur_p:.2f}).")
            images.extend([ax_b64, co_b64, sa_b64])
            labels.extend([f"Sample {i+1} AXIAL mid-slice. {cap}",
                              f"Sample {i+1} CORONAL mid-slice. {cap}",
                              f"Sample {i+1} SAGITTAL mid-slice. {cap}"])
        return images, labels

    def vlm_prompt_template(self) -> str:
        return CLS3D_VLM_PROMPT

    def task_capabilities(self) -> List[str]:
        return ["scalar_predictions", "volumetric_predictions"]

    def to_tensor(self, image: np.ndarray, device: str,
                    size: int = 96) -> torch.Tensor:
        """3D cls input. Prefer _vol_to_tensor for any 3D input; only
        delegate to _source_pp_fn for non-3D inputs (it's typically a
        2D-image-style preprocessing that crashes on volumetric arrays).
        Caller is responsible for ensuring the volume is already
        normalized at prep time (per-case z-score is the convention)."""
        arr = np.asarray(image)
        if arr.ndim >= 3 or self._source_pp_fn is None:
            return _vol_to_tensor(image, device)
        return self._source_pp_fn(image,
                                    self._source_pp_size or size,
                                    device)

    def render_prediction_panel(self, model: nn.Module, image: np.ndarray,
                                  device: str, size: int = 224
                                  ) -> Tuple[str, str]:
        """Render a single (raw_view, prediction_panel) pair for Agent 2's
        visual inspection. For 3D cls: both panels are the axial mid-slice;
        the 'overlay' carries the predicted class + confidence as the
        caption surfaced through the multimodal message builder. Returns
        (raw_b64_png, panel_b64_png)."""
        t = self.to_tensor(image, device, size)
        with torch.no_grad():
            logits, pred = self.forward_pred(model, t)
            prob = F.softmax(logits, dim=1).max().item()
        arr = np.asarray(image)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        ax_b64 = _mid_slice_b64(arr, axis=0, size=size)
        # For cls there is no spatial mask to overlay — re-use the raw
        # mid-slice as the "prediction panel" and rely on the caption /
        # build_multimodal_message to surface the class + confidence.
        # Agent 2's LLM call receives the caption alongside the image.
        return ax_b64, ax_b64


TaskRegistry.register("classification_3d", Classification3DTask)
