"""Segmentation-3D TaskAdapter.

3D segmentation works through the same TaskAdapter interface as 2D
seg / cls — same metric / loss / VLM-rendering / capability methods —
but operates on volume tensors (D, H, W) instead of images (H, W).

Key differences from 2D seg:

  - Inputs are 3D volumes; forward uses MONAI's sliding-window
    inference so we don't load the whole volume at once. Default
    patch size (96, 96, 96) matches the MONAI spleen bundle.

  - Dice is computed in 3D (volumetric overlap), not averaged over
    2D slices.

  - VLM rendering shows multi-planar reconstruction: middle slice of
    the volume in axial, coronal, sagittal views. Mask overlays
    drawn per view.

  - to_tensor handles 3D HU volumes via the calibration probe's
    ct_window_01 candidate by default.

Registered as 'segmentation_3d' in the TaskRegistry.
"""
from __future__ import annotations
import io
import base64
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

from .base import TaskAdapter, TaskRegistry


_DEFAULT_PATCH = (96, 96, 96)


def _vol_to_tensor(vol: np.ndarray, device: str,
                      window_lo: float = -1024, window_hi: float = 1024,
                      ) -> torch.Tensor:
    """Default 3D tensor prep: CT-windowed [0, 1] over the full volume.
    Returns [1, 1, D, H, W]."""
    arr = vol.astype(np.float32)
    arr = np.clip(arr, window_lo, window_hi)
    arr = (arr - window_lo) / (window_hi - window_lo + 1e-6)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _sliding_window_predict(model: nn.Module, x: torch.Tensor,
                                patch: Tuple[int, int, int] = _DEFAULT_PATCH,
                                batch_size: int = 2, overlap: float = 0.5
                                ) -> torch.Tensor:
    """Run sliding-window inference while preserving the input spatial grid.

    MONAI pads short axes internally and crops logits back to the input shape.
    Manual padding here would leave an expanded prediction grid that callers
    could incorrectly rescale instead of cropping.
    """
    from monai.inferers import sliding_window_inference
    with torch.no_grad():
        out = sliding_window_inference(
            inputs=x, roi_size=patch, sw_batch_size=batch_size,
            predictor=model, overlap=overlap, mode="gaussian",
        )
    if out.shape[2:] != x.shape[2:]:
        raise RuntimeError(f"Sliding-window output grid {out.shape[2:]} != input {x.shape[2:]}")
    return out


def _b64_png(arr: np.ndarray, size: int = 256) -> str:
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil = PILImage.fromarray(arr).resize((size, size), PILImage.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _overlay_mask_on_slice(slice_2d: np.ndarray, mask_2d: np.ndarray,
                              num_classes: int) -> np.ndarray:
    """Composite a 2D class mask onto a grayscale CT slice. Returns
    HWC uint8 RGB."""
    if slice_2d.max() <= 1.0 + 1e-6:
        gray = (np.clip(slice_2d, 0, 1) * 255).astype(np.uint8)
    else:
        gray = slice_2d.astype(np.uint8)
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0],
                          [0, 0, 255], [255, 255, 0], [255, 0, 255]])
    for c in range(1, min(num_classes, 6)):
        m = mask_2d == c
        base[m] = 0.5 * base[m] + 0.5 * colors[c]
    return np.clip(base, 0, 255).astype(np.uint8)


SEG3D_VLM_PROMPT = """Compare 3D segmentation predictions from the SOURCE model \
vs the CURRENT adapted model on {k} target volumes, viewed via three \
mid-planar reconstructions per volume (axial, coronal, sagittal).

## Task
Target structure: {structure_name}
Failure type: {failure_type}

## Diagnosed shift mechanism (condition your vote on this)
{mechanism_hint}

For each volume you see NINE images in order:
  1. Sample i RAW axial mid-slice (no overlay)
  2. Sample i SOURCE axial mid-slice with mask overlay
  3. Sample i CURRENT axial mid-slice with mask overlay
  4-6. Same triplet for coronal mid-slice
  7-9. Same triplet for sagittal mid-slice

The axial / coronal / sagittal views together let you assess 3D
correctness — a mask that looks right axially may be wrong in
coronal / sagittal direction.

## Two per-volume judgments

1. **Relative** (CURRENT vs SOURCE): better | similar | worse.
   Across all 3 views, which mask better covers the actual
   {structure_name}?

2. **Absolute quality of CURRENT alone**: usable | broken.
   - "broken" — mask is largely on non-target anatomy, OR mask is
     near-empty when the structure is clearly visible, OR mask
     covers far more than the actual structure.
   - "usable" — mask covers the real {structure_name} in all 3
     views; boundaries may be imperfect.

Return ONLY this JSON:
```json
{{"per_sample":          ["better|similar|worse", ...],
  "per_sample_absolute": ["usable|broken", ...],
  "note": "<one-sentence summary>"}}
```"""


class SegmentationTask3D(TaskAdapter):

    @property
    def name(self) -> str:
        return "segmentation_3d"

    @property
    def metric_name(self) -> str:
        return "dice_3d"

    # ----------------------------------------------------------------
    # Forward + prediction (patch-based sliding window)
    # ----------------------------------------------------------------

    def forward_pred(self, model: nn.Module, image_tensor: torch.Tensor):
        """For 3D, image_tensor is [1, 1, D, H, W]. Returns
        (logits [1, C, D, H, W], pred_mask [D, H, W])."""
        logits = _sliding_window_predict(model, image_tensor)
        pred = logits.argmax(dim=1)[0].cpu().numpy()  # [D, H, W]
        return logits, pred

    def to_tensor(self, image: np.ndarray, device: str,
                    size: int = 96) -> torch.Tensor:
        """Default 3D preprocessing: CT-windowed [0, 1]. If the
        TaskAdapter has been configured with a calibration-probe
        result, use that instead."""
        if self._source_pp_fn is not None:
            return self._source_pp_fn(image, self._source_pp_size or size, device)
        return _vol_to_tensor(image, device)

    # ----------------------------------------------------------------
    # 3D volumetric Dice
    # ----------------------------------------------------------------

    def sample_metric(self, model: nn.Module, image: np.ndarray,
                        label: Any, num_classes: int, device: str,
                        size: int = 96) -> Optional[float]:
        if label is None:
            return None
        t = self.to_tensor(image, device, size)
        _, pred = self.forward_pred(model, t)
        # label is 3D mask [D, H, W]; pred is same shape
        if pred.shape != label.shape:
            # Pad pred back to label shape (sliding-window may have
            # padded). Truncate.
            d, h, w = label.shape
            pred = pred[:d, :h, :w]
        dices = []
        for c in range(1, num_classes):
            p = (pred == c).astype(float)
            g = (label == c).astype(float)
            if g.sum() > 0:
                d = float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))
                dices.append(d)
        return float(np.mean(dices)) if dices else None

    # ----------------------------------------------------------------
    # CE loss on K labeled volumes for *_ss tools. Pixel-level CE on
    # the full volume — caller may need to memory-budget K small.
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
            # For supervised CE we need gradients — disable sliding-
            # window (it uses no_grad). Run forward in train mode on a
            # single patch crop centered on the labeled region.
            with torch.enable_grad():
                logits = model(t)
            # Resize label to match logits if necessary
            lbl_t = torch.from_numpy(
                np.clip(lbl, 0, num_classes - 1).astype(np.int64)
                ).unsqueeze(0).to(device)
            if logits.shape[-3:] != lbl_t.shape[-3:]:
                # crop or skip if mismatched
                continue
            ce = F.cross_entropy(logits, lbl_t, reduction="mean")
            losses.append(ce)
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    # ----------------------------------------------------------------
    # Coverage / fg-ratio for the catastrophic-collapse gate
    # ----------------------------------------------------------------

    def coverage_signal(self, model: nn.Module,
                            images: List[np.ndarray],
                            device: str,
                            max_samples: int = 4) -> Optional[float]:
        ratios = []
        for vol in images[:max_samples]:
            t = self.to_tensor(vol, device)
            _, pred = self.forward_pred(model, t)
            ratios.append(float((pred > 0).mean()))
        return float(np.mean(ratios)) if ratios else 0.0

    # ----------------------------------------------------------------
    # Multi-planar mid-slice VLM rendering
    # ----------------------------------------------------------------

    def render_prediction_panel(self, model: nn.Module, image: np.ndarray,
                                  device: str, size: int = 256,
                                  num_classes: int = 2):
        """Single-model prediction panel for Agent 2's visual inspection.
        Returns (raw_b64, overlay_b64) — both axial mid-slice of the
        RESAMPLED volume (the grid the model actually saw), so the
        predicted mask aligns by construction. Falls back gracefully
        if any step errors out."""
        try:
            t = self.to_tensor(image, device, size)  # 5D [1, C, D, H, W]
            with torch.no_grad():
                _, pred = self.forward_pred(model, t)  # pred: [D, H, W]
            vol_resampled = t[0, 0].cpu().numpy()
            D, H, W = vol_resampled.shape[-3:]
            ax_idx = D // 2
            slc = vol_resampled[ax_idx]
            lo, hi = float(slc.min()), float(slc.max())
            gray = (((slc - lo) / (hi - lo + 1e-6)) * 255).astype(np.uint8) \
                if hi > lo else np.zeros_like(slc, dtype=np.uint8)
            raw_b64 = _b64_png(gray, size=size)
            if pred.ndim == 3:
                if pred.shape[0] != D:
                    pred = pred[:D]
                mask = pred[ax_idx]
            else:
                mask = pred
            overlay_rgb = _overlay_mask_on_slice(gray, mask, num_classes)
            overlay_b64 = _b64_png(overlay_rgb, size=size)
            return raw_b64, overlay_b64
        except Exception:
            # Soft fallback: just render the raw axial slice for both
            # panels so the VLM at least sees the image even if mask
            # overlay can't be built.
            arr = np.asarray(image)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            ax_idx = arr.shape[0] // 2 if arr.ndim >= 3 else 0
            slc = arr[ax_idx] if arr.ndim >= 3 else arr
            lo, hi = float(slc.min()), float(slc.max())
            gray = (((slc - lo) / (hi - lo + 1e-6)) * 255).astype(np.uint8) \
                if hi > lo else np.zeros_like(slc, dtype=np.uint8)
            b = _b64_png(gray, size=size)
            return b, b

    def render_pair_for_vlm(self, model_source: nn.Module,
                                model_current: nn.Module,
                                anchor_imgs: List[np.ndarray],
                                num_classes: int,
                                device: str,
                                size: int = 256) -> Tuple[List[str], List[str]]:
        """Render mid-slice axial / coronal / sagittal triplets for each
        volume. Output: 9 images per volume (raw / src / cur × 3 views)."""
        images, labels = [], []
        for i, vol in enumerate(anchor_imgs):
            t = self.to_tensor(vol, device)
            with torch.no_grad():
                _, src_pred = self.forward_pred(model_source, t)
                _, cur_pred = self.forward_pred(model_current, t)
            # vol is [D, H, W]; use min-clipped windowed slice for display
            vol_disp = np.clip((vol.astype(np.float32) + 1024) / 2048, 0, 1)
            if vol_disp.shape != src_pred.shape:
                d, h, w = vol_disp.shape
                src_pred = src_pred[:d, :h, :w]
                cur_pred = cur_pred[:d, :h, :w]
            D, H, W = vol_disp.shape
            views = [("axial",    D // 2, lambda a: (vol_disp[a], src_pred[a], cur_pred[a])),
                       ("coronal",  H // 2, lambda a: (vol_disp[:, a, :], src_pred[:, a, :], cur_pred[:, a, :])),
                       ("sagittal", W // 2, lambda a: (vol_disp[:, :, a], src_pred[:, :, a], cur_pred[:, :, a]))]
            for view_name, idx, getter in views:
                raw_slice, src_slice, cur_slice = getter(idx)
                images.append(_b64_png(
                    (raw_slice * 255).astype(np.uint8), size=size))
                labels.append(f"Sample {i+1} RAW {view_name} mid-slice")
                images.append(_b64_png(_overlay_mask_on_slice(
                    raw_slice, src_slice, num_classes), size=size))
                labels.append(f"Sample {i+1} SOURCE {view_name} mask overlay")
                images.append(_b64_png(_overlay_mask_on_slice(
                    raw_slice, cur_slice, num_classes), size=size))
                labels.append(f"Sample {i+1} CURRENT {view_name} mask overlay")
        return images, labels

    def vlm_prompt_template(self) -> str:
        return SEG3D_VLM_PROMPT

    def task_capabilities(self) -> List[str]:
        return ["pixel_predictions", "volumetric_predictions"]


TaskRegistry.register("segmentation_3d", SegmentationTask3D)
