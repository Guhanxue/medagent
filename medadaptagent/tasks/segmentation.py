"""Segmentation TaskAdapter — wraps the existing seg-specific helpers
(Dice metric, mask overlay rendering, pixel-CE loss) behind the
TaskAdapter interface.

This is a refactor: behavior matches the pre-refactor pipeline. The
agent layer now dispatches through `task.eval_metric(...)` etc.
instead of importing seg-specific helpers directly.
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


def _img_to_tensor(img: np.ndarray, device: str, size: int = 256) -> torch.Tensor:
    pil = PILImage.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)
                              if img.max() <= 1.0 else img.astype(np.uint8))
    pil = pil.resize((size, size), PILImage.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        t = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
    else:
        t = torch.from_numpy(arr).permute(2, 0, 1)
    return t.unsqueeze(0).to(device)


class SegmentationTask(TaskAdapter):

    @property
    def name(self) -> str:
        return "segmentation"

    @property
    def metric_name(self) -> str:
        return "dice"

    # ----------------------------------------------------------------
    # Forward + prediction
    # ----------------------------------------------------------------

    def forward_pred(self, model: nn.Module, image_tensor: torch.Tensor):
        with torch.no_grad():
            logits = model(image_tensor)            # [B, C, H, W]
            pred = logits.argmax(dim=1)             # [B, H, W]
        return logits, pred[0].cpu().numpy()        # 2D mask

    # ----------------------------------------------------------------
    # Per-sample Dice. Mirrors the legacy `_avg_dice` per-sample inner
    # loop. Skips classes with zero ground-truth fg.
    # ----------------------------------------------------------------

    def sample_metric(self, model: nn.Module, image: np.ndarray,
                        label: Any, num_classes: int, device: str,
                        size: int = 256) -> Optional[float]:
        if label is None:
            return None
        t = _img_to_tensor(image, device, size)
        _, pred = self.forward_pred(model, t)
        lbl_pil = PILImage.fromarray(label.astype(np.int32).astype(np.uint8))
        lbl_r = np.array(lbl_pil.resize((size, size), PILImage.NEAREST))
        dices = []
        for c in range(1, num_classes):
            if num_classes == 3 and c == 1:
                p = (pred >= 1).astype(float)
                g = (lbl_r >= 1).astype(float)
            else:
                p = (pred == c).astype(float)
                g = (lbl_r == c).astype(float)
            if g.sum() > 0:
                d = float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))
                dices.append(d)
        if not dices:
            return None
        return float(np.mean(dices))

    # ----------------------------------------------------------------
    # Pixel-level CE loss for *_ss tools. Resizes label to model
    # resolution, clamps class indices, returns mean across K samples.
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
            lbl_pil = PILImage.fromarray(lbl.astype(np.int32).astype(np.uint8))
            lbl_r = np.array(lbl_pil.resize((size, size), PILImage.NEAREST))
            lbl_r = np.clip(lbl_r, 0, num_classes - 1)
            lbl_t = torch.from_numpy(lbl_r).long().unsqueeze(0).to(device)
            logits = model(t)               # [1, C, H, W]
            ce = F.cross_entropy(logits, lbl_t, reduction="mean")
            losses.append(ce)
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    # ----------------------------------------------------------------
    # Foreground ratio averaged over the target set (capped). Used by
    # catastrophic-collapse gate.
    # ----------------------------------------------------------------

    def coverage_signal(self, model: nn.Module,
                            images: List[np.ndarray],
                            device: str,
                            max_samples: int = 200) -> Optional[float]:
        ratios = []
        n = min(max_samples, len(images))
        for img in images[:n]:
            t = _img_to_tensor(img, device)
            _, pred = self.forward_pred(model, t)
            ratios.append(float((pred > 0).mean()))
        return float(np.mean(ratios)) if ratios else 0.0

    # ----------------------------------------------------------------
    # VLM rendering: triplets of (raw, source overlay, current overlay)
    # for K anchors. Mirrors the existing prepare_prediction_samples
    # path.
    # ----------------------------------------------------------------

    def render_pair_for_vlm(self, model_source: nn.Module,
                                model_current: nn.Module,
                                anchor_imgs: List[np.ndarray],
                                num_classes: int,
                                device: str,
                                size: int = 256) -> Tuple[List[str], List[str]]:
        from medadaptagent.graph.vision_utils import (
            prepare_prediction_samples,
        )
        k = len(anchor_imgs)
        src_pairs = prepare_prediction_samples(
            anchor_imgs, model_source, device, n=k)
        cur_pairs = prepare_prediction_samples(
            anchor_imgs, model_current, device, n=k)
        images, labels = [], []
        for i in range(k):
            images.append(src_pairs[i][0])
            labels.append(f"Sample {i+1} RAW IMAGE (no mask — use this to "
                            f"locate the actual structure)")
            images.append(src_pairs[i][1])
            labels.append(f"Sample {i+1} SOURCE mask overlay")
            images.append(cur_pairs[i][1])
            labels.append(f"Sample {i+1} CURRENT mask overlay")
        return images, labels

    def vlm_prompt_template(self) -> str:
        # Use the existing PER_ITER_VLM_PROMPT (kept unchanged, lives
        # in execution_engineer.py). We return a marker that the caller
        # interprets as "use the legacy seg prompt".
        return "__SEG_LEGACY_PROMPT__"

    def task_capabilities(self) -> List[str]:
        return ["pixel_predictions"]

    def to_tensor(self, image: np.ndarray, device: str,
                    size: int = 256) -> torch.Tensor:
        """Seg models route through the configured preprocessing
        if set by the calibration probe; otherwise fall back to legacy
        [0,1] scaling that matches the UNet/STU-Net/nnU-Net convention."""
        if self._source_pp_fn is not None:
            return self._source_pp_fn(image,
                                        self._source_pp_size or size,
                                        device)
        return _img_to_tensor(image, device, size)


TaskRegistry.register("segmentation", SegmentationTask)
