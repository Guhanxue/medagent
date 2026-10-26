"""Agent 6: Delivery — bundles a deployable artifact from the Manager's
output.

Turns the system from "produces .pt + JSONs" to "ships a deployment
folder". The bundle has everything a user needs to deploy the adapted
model on the new site:

  <bundle_dir>/
    adapted.pt                — adapted model state_dict
    preprocessing.json        — the input preprocessing convention
                                  (from calibration probe)
    deployment_report.md      — human-readable summary
    config.json               — machine-readable: arch, num_classes,
                                  task, source/target metadata
    sanity_samples/
      sample_<i>_input.png    — raw target sample
      sample_<i>_source_pred  — source prediction
      sample_<i>_adapted_pred — adapted prediction
    history.json              — full Manager iteration history

Inference snippet (printed in report) shows how to load + run the model
in deployment.
"""
from __future__ import annotations
import io
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage


def _save_sanity_overlay(image: np.ndarray, pred: np.ndarray,
                            out_path: str, size: int = 256) -> None:
    """For a seg sample: save raw + colored prediction side by side."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0 + 1e-6:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    raw_pil = PILImage.fromarray(image).resize((size, size), PILImage.BILINEAR)
    # Colored overlay
    overlay = np.array(raw_pil).astype(np.float32)
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0],
                          [0, 0, 255], [255, 255, 0], [255, 0, 255]])
    pred_resized = np.array(PILImage.fromarray(
        pred.astype(np.uint8)).resize((size, size), PILImage.NEAREST))
    for c in range(1, min(6, pred_resized.max() + 1)):
        mask = pred_resized == c
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * colors[c]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    # Side-by-side
    canvas = np.concatenate([np.array(raw_pil), overlay], axis=1)
    PILImage.fromarray(canvas).save(out_path)


def _save_sanity_3d(volume: np.ndarray, pred: np.ndarray,
                       out_path: str, size: int = 256) -> None:
    """3D-seg sanity sample: render axial/coronal/sagittal mid-planar
    overlays as a single side-by-side PNG. `volume` is (D, H, W) — we
    strip a leading channel if present and slice through middle of each
    axis."""
    vol = np.asarray(volume).astype(np.float32)
    pr = np.asarray(pred)
    if vol.ndim == 4:
        vol = vol[0]
    if pr.ndim == 4:
        pr = pr[0]
    if vol.shape != pr.shape:
        d, h, w = pr.shape
        vol = vol[:d, :h, :w]
    if vol.max() > 1.5:
        # Raw HU (or other un-normalized range). Robust percentile
        # window for display so this isn't tied to a particular CT
        # convention. Chain-resampled inputs skip this branch — vol is
        # already in [0, 1].
        lo, hi = np.percentile(vol, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        v = np.clip((vol - lo) / (hi - lo), 0, 1)
    else:
        v = np.clip(vol, 0, 1)
    D, H, W = v.shape
    # Slice through the prediction centroid when there's foreground, so
    # the displayed slice actually contains the structure of interest.
    # Full-body / thorax-abdomen-pelvis scans otherwise show the pelvis at
    # the volume midpoint and miss the abdominal target. Fall back to
    # volume midpoint when the prediction is empty.
    fg = pr > 0
    if fg.any():
        idxs = np.argwhere(fg)
        cz, cy, cx = idxs.mean(axis=0).astype(int)
        cz = int(np.clip(cz, 0, D - 1))
        cy = int(np.clip(cy, 0, H - 1))
        cx = int(np.clip(cx, 0, W - 1))
    else:
        cz, cy, cx = D // 2, H // 2, W // 2
    views = [v[cz], v[:, cy], v[:, :, cx]]
    pred_views = [pr[cz], pr[:, cy], pr[:, :, cx]]
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0],
                          [0, 0, 255], [255, 255, 0], [255, 0, 255]])
    panels = []
    for s2d, p2d in zip(views, pred_views):
        gray = (s2d * 255).astype(np.uint8)
        rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
        for c in range(1, min(6, int(p2d.max()) + 1)):
            m = p2d == c
            rgb[m] = 0.5 * rgb[m] + 0.5 * colors[c]
        pil = PILImage.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        pil = pil.resize((size, size), PILImage.BILINEAR)
        panels.append(np.array(pil))
    canvas = np.concatenate(panels, axis=1)
    PILImage.fromarray(canvas).save(out_path)


def _save_sanity_cls(image: np.ndarray, src_pred: int, ada_pred: int,
                       src_p: float, ada_p: float, out_path: str,
                       label_names: Optional[List[str]] = None,
                       size: int = 256) -> None:
    """For a cls sample: save raw image with overlaid caption. Handles
    both 2D (H,W or H,W,C) and 3D (D,H,W or 1,1,D,H,W) inputs — for 3D
    volumes, renders a horizontal axial/coronal/sagittal mid-slice
    montage. Caption goes in the per-sample JSON sidecar."""
    arr = np.asarray(image)
    if arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1:
        arr = arr[0, 0]
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # 3D volume → 3-view mid-slice montage
    if arr.ndim == 3 and not (arr.shape[-1] <= 4):
        # Treat as (D, H, W). Render axial/coronal/sagittal mid-slices.
        def _slice_to_u8(s):
            lo, hi = float(s.min()), float(s.max())
            if hi - lo < 1e-6:
                return np.zeros_like(s, dtype=np.uint8)
            return ((s - lo) / (hi - lo) * 255).astype(np.uint8)
        ax = _slice_to_u8(arr[arr.shape[0] // 2])
        co = _slice_to_u8(arr[:, arr.shape[1] // 2, :])
        sa = _slice_to_u8(arr[:, :, arr.shape[2] // 2])
        panels = []
        for s in (ax, co, sa):
            p = PILImage.fromarray(s).resize((size, size), PILImage.BILINEAR)
            panels.append(np.array(p))
        canvas = np.concatenate(panels, axis=1)  # side-by-side
        # Promote grayscale to RGB so the saved PNG is consistent
        if canvas.ndim == 2:
            canvas = np.stack([canvas] * 3, axis=-1)
        PILImage.fromarray(canvas).save(out_path)
    else:
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 + 1e-6:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pil = PILImage.fromarray(arr).resize((size, size), PILImage.BILINEAR)
        pil.save(out_path)
    # Caption is recorded in the per-sample JSON sidecar; keep image clean.
    side = {
        "source_pred": int(src_pred), "source_conf": float(src_p),
        "adapted_pred": int(ada_pred), "adapted_conf": float(ada_p),
        "label_names": label_names or [],
    }
    with open(out_path.replace(".png", ".json"), "w") as f:
        json.dump(side, f, indent=2)


def _build_report_md(config: Dict, history: List[Dict],
                       summary: Dict) -> str:
    """Human-readable deployment report."""
    lines = []
    lines.append(f"# Deployment Report — {config.get('dataset_name', '<dataset>')}")
    lines.append("")
    lines.append(f"- **Task:** {config.get('task')}")
    lines.append(f"- **Source arch:** {config.get('arch')}")
    lines.append(f"- **Num classes:** {config.get('num_classes')}")
    lines.append(f"- **Input preprocessing:** {config.get('preprocessing', {}).get('name')}")
    lines.append(f"- **Input size:** {config.get('preprocessing', {}).get('size')}")
    lines.append(f"- **Iterations:** {summary.get('n_iterations')}")
    lines.append(f"- **Best iteration:** {summary.get('best_iteration')}")
    lines.append(f"- **Verdict:** {summary.get('verdict')}")
    # Task-aware metric label. Prefer the metric_name reported by
    # Agent 5 (reflects --metric flag, e.g., "auroc"); fall back to
    # task-default ("dice" for seg, "accuracy" for cls).
    _task = config.get("task", "")
    _mraw = summary.get("metric_name") or (
        "accuracy" if "classification" in _task else "dice")
    _metric_label = {"auroc": "AUROC", "accuracy": "Accuracy",
                       "dice": "Dice"}.get(_mraw.lower(), _mraw.title())
    lines.append(f"- **Delivered {_metric_label}:** {summary.get('delivered_metric')}")
    lines.append(f"- **Source {_metric_label} on K labels:** {summary.get('source_metric')}")
    lines.append(f"- **Delta (delivered - source):** {summary.get('delta')}")
    lines.append(f"- **Recommendation:** **{summary.get('recommendation')}**")
    lines.append("")
    lines.append("## What the pipeline tried")
    lines.append("")
    for h in history:
        delta = h.get("delta")
        delta_s = f"{delta:+.4f}" if delta is not None else "n/a"
        lines.append(f"- iter {h.get('iteration')}: "
                       f"tools={h.get('tools')} → verdict={h.get('verdict')}, "
                       f"delta={delta_s}")
        if h.get("reason"):
            lines.append(f"  - reason: {h.get('reason')[:200]}")
    lines.append("")
    lines.append("## Inference snippet")
    lines.append("")
    lines.append("```python")
    arch = config.get("arch")
    if arch in ("resnet18", "resnet50"):
        lines.append(f"from torchvision.models import {arch}")
        lines.append(f"import torch")
        lines.append(f"import torch.nn as nn")
        lines.append(f"model = {arch}(weights=None)")
        lines.append(f"model.fc = nn.Linear(model.fc.in_features, "
                       f"{config.get('num_classes')})")
    elif arch == "vit_small":
        lines.append("import timm, torch")
        lines.append(f"model = timm.create_model('vit_small_patch16_224', "
                       f"pretrained=False, num_classes={config.get('num_classes')})")
    elif arch == "monai_unet3d":
        arch_meta = config.get("arch_meta", {}) or {}
        chans = arch_meta.get("channels", "<channels from sidecar>")
        strides = arch_meta.get("strides", "<strides from sidecar>")
        num_res_units = arch_meta.get("num_res_units", 2)
        norm = arch_meta.get("norm", "batch")
        lines.append("from monai.networks.nets import UNet")
        lines.append(f"model = UNet(spatial_dims=3, in_channels=1, "
                       f"out_channels={config.get('num_classes')}, "
                       f"channels={chans}, strides={strides}, "
                       f"num_res_units={num_res_units}, norm={norm!r})")
    elif arch == "monai_bundle":
        # Bundle-agnostic snippet — works for UNet, SegResNet, DiNTS, etc.
        # Reader points the parser at the bundle's own inference config.
        lines.append("import os, torch")
        lines.append("from monai.bundle import ConfigParser")
        lines.append(f"bundle_dir = {config.get('bundle_dir', '<bundle_dir>')!r}")
        lines.append("parser = ConfigParser()")
        lines.append("cfg = next(p for p in ("
                       "'configs/inference.json', 'configs/inference.yaml') "
                       "if os.path.exists(os.path.join(bundle_dir, p)))")
        lines.append("parser.read_config(os.path.join(bundle_dir, cfg))")
        lines.append("parser['bundle_root'] = bundle_dir")
        lines.append("parser['device'] = 'cuda:0'")
        lines.append("if 'load_pretrain' in parser.config:")
        lines.append("    parser['load_pretrain'] = False")
        lines.append("model = parser.get_parsed_content('network_def').cuda()")
        ce = config.get("class_extract")
        if ce:
            lines.append(f"# class_extract = {ce} — wrap model output to "
                           "expose only these channels as foreground.")
    else:
        lines.append(f"# Architecture: {arch}")
        lines.append("# (See your in-house model class definition.)")
    lines.append("model.load_state_dict(torch.load('adapted.pt', "
                   "map_location='cuda:0'))")
    lines.append("model.eval()")
    lines.append("")
    pp_cfg = config.get("preprocessing", {})
    lines.append("# Apply documented preprocessing:")
    lines.append(f"# {pp_cfg.get('name')} @ size {pp_cfg.get('size')}")
    if pp_cfg.get("chain"):
        lines.append("# MONAI Compose (build via monai.transforms):")
        for t in pp_cfg["chain"]:
            tname = t.get("transform")
            kw = ", ".join(f"{k}={v!r}" for k, v in t.items() if k != "transform")
            lines.append(f"#   {tname}({kw})")
        roi = pp_cfg.get("size", 96)
        lines.append(f"# Then sliding_window_inference(roi_size=({roi},{roi},{roi}), "
                       "overlap=0.5, mode='gaussian')")
    lines.append("```")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    rec = summary.get("recommendation", "")
    if rec == "ship":
        lines.append("- Pipeline accepted the adapted model. K-label "
                       "evaluation showed clear improvement.")
    elif rec == "ship_source_already_strong":
        lines.append("- Source model already saturated on K target "
                       "labels. Adapted model = source. Safe to deploy.")
    elif rec == "ship_with_caveat":
        lines.append("- Gate was conservative; K-label evidence was "
                       "positive. Validate on a wider held-out set before "
                       "production.")
    elif "label_recommended" in rec:
        lines.append("- Pipeline could not beat source given K labels. "
                       "Consider providing more labels for semi-supervised "
                       "mode, or audit the source model's fit to this site.")
    else:
        lines.append(f"- Recommendation: {rec}")
    lines.append("")
    return "\n".join(lines)


def build_delivery_bundle(
    out_dir: str,
    source_model: nn.Module,
    delivered_model: nn.Module,
    target_images: List[np.ndarray],
    target_labels: Optional[List[Any]],
    config: Dict[str, Any],
    summary: Dict[str, Any],
    history: List[Dict[str, Any]],
    task,
    device: str,
    sanity_n: int = 5,
) -> Dict[str, Any]:
    """Produce a deployable folder with the adapted model + everything
    a user needs to deploy it.

    Args:
        out_dir         : target directory; will be created
        delivered_model : the model to ship (may be source on revert)
        config          : dict with `dataset_name`, `arch`, `task`,
                            `num_classes`, `preprocessing` (calibration
                            probe result)
        summary         : Manager's return dict (verdict, delta, ...)
        history         : Manager's per-iteration history
        task            : TaskAdapter (used for predictions)
    Returns dict with bundle paths."""
    os.makedirs(out_dir, exist_ok=True)
    sanity_dir = os.path.join(out_dir, "sanity_samples")
    os.makedirs(sanity_dir, exist_ok=True)

    # 1. adapted.pt
    ck_path = os.path.join(out_dir, "adapted.pt")
    torch.save(delivered_model.state_dict(), ck_path)

    # 2. preprocessing.json
    pp_path = os.path.join(out_dir, "preprocessing.json")
    with open(pp_path, "w") as f:
        json.dump(config.get("preprocessing", {}), f, indent=2)

    # 3. config.json
    config_path = os.path.join(out_dir, "config.json")
    safe_config = {k: v for k, v in config.items()
                     if k not in ("model_obj",) and (
                         isinstance(v, (str, int, float, bool, list, dict))
                         or v is None)}
    with open(config_path, "w") as f:
        json.dump(safe_config, f, indent=2, default=str)

    # 4. history.json
    hist_path = os.path.join(out_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # 5. sanity samples — predictions on N target images
    delivered_model.eval()
    source_model.eval()
    sample_indices = np.linspace(0, len(target_images) - 1,
                                    min(sanity_n, len(target_images)),
                                    dtype=int).tolist()
    for k, idx in enumerate(sample_indices):
        img = target_images[idx]
        t = task.to_tensor(img, device)
        with torch.no_grad():
            src_logits, src_pred = task.forward_pred(source_model, t)
            ada_logits, ada_pred = task.forward_pred(delivered_model, t)
        out_file = os.path.join(sanity_dir, f"sample_{k}.png")
        if task.name == "segmentation":
            _save_sanity_overlay(img, ada_pred, out_file)
        elif task.name == "segmentation_3d":
            # Use the chain-resampled input tensor as display volume so
            # spatial shape matches the prediction. Strip batch + channel.
            disp_vol = t[0, 0].detach().cpu().numpy().astype(np.float32)
            _save_sanity_3d(disp_vol, ada_pred, out_file)
        else:
            import torch.nn.functional as F
            sp = float(F.softmax(src_logits, dim=1).max())
            ap = float(F.softmax(ada_logits, dim=1).max())
            _save_sanity_cls(img, int(src_pred), int(ada_pred),
                                sp, ap, out_file,
                                label_names=config.get("label_names"))

    # 6. deployment_report.md
    report = _build_report_md(config, history, summary)
    report_path = os.path.join(out_dir, "deployment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "out_dir": out_dir,
        "ckpt": ck_path,
        "preprocessing": pp_path,
        "config": config_path,
        "history": hist_path,
        "sanity_dir": sanity_dir,
        "report": report_path,
        "n_sanity_samples": len(sample_indices),
    }


__all__ = ["build_delivery_bundle"]
