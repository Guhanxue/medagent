"""Model-adaptation tools: BN-reset, Tent, SAR, EATA, SFDA.

These modify `ctx.current_model` in place (gradient updates on norm
affine parameters, or running-stat re-estimation). They do NOT touch
target_images or preprocess_fn. All implementations mirror the
published formulations and reuse the baseline code in
scripts/run_baseline_tta.py where possible.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from .base import Tool, ToolContext, register
from medadaptagent.graph.norm_utils import (
    iter_norm_affine_params, reset_norm_running_stats,
    disable_running_stats,
)


# --------------------------------------------------------------
# Shared helpers — use ctx.preprocess_fn so preprocessing tools
# applied before adaptation also shape the adaptation signal.
# --------------------------------------------------------------

def _to_tensor(ctx: ToolContext, img: np.ndarray, size: int = 256) -> torch.Tensor:
    """Dispatch tensor-prep via ctx.task when available so cls inputs
    get ImageNet-normalized and seg inputs stay raw [0,1]. Falls back
    to legacy seg-style (no normalization) when ctx.task is None."""
    img = ctx.preprocess_fn(img)
    if ctx.task is not None:
        # Each task picks its own default size — cls=224, seg=256
        return ctx.task.to_tensor(img, ctx.device,
                                     size=size if size != 256 else (
                                         224 if ctx.task.name == "classification"
                                         else 256))
    pil = PILImage.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)
                              if img.max() <= 1.0 else img.astype(np.uint8))
    pil = pil.resize((size, size), PILImage.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        t = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
    else:
        t = torch.from_numpy(arr).permute(2, 0, 1)
    return t.unsqueeze(0).to(ctx.device)


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()


def _train_forward(ctx: ToolContext, img, label=None, patch: int = 96):
    """3D-safe training forward. For 5D tensors (3D volumes), random-
    crops a `patch`-cube before the forward pass — avoids MONAI UNet's
    skip-connection size mismatch when full-volume dims aren't multiples
    of the stride product. For 2D (4D tensors), passes through.

    Args:
      img:   image to preprocess + forward (np.ndarray)
      label: optional ground-truth label aligned with the chain-resampled
             image grid. If provided AND 3D, cropped to the same patch
             coords as `img` and returned as a torch.Tensor [1, D, H, W]
             ready for F.cross_entropy.
      patch: cube side length for 3D random crop (default 96 = MONAI bundle ROI)

    Returns: (logits, label_tensor_or_None)
    """
    t = _to_tensor(ctx, img)
    lbl_t = None
    if t.ndim == 5:
        D, H, W = t.shape[-3:]
        if D >= patch and H >= patch and W >= patch:
            d = int(np.random.randint(0, D - patch + 1))
            h = int(np.random.randint(0, H - patch + 1))
            w = int(np.random.randint(0, W - patch + 1))
            t = t[..., d:d + patch, h:h + patch, w:w + patch]
            if label is not None:
                lbl_arr = np.asarray(label)
                if lbl_arr.shape == (D, H, W):
                    lbl_t = torch.from_numpy(
                        lbl_arr[d:d + patch, h:h + patch,
                                w:w + patch].astype(np.int64)
                    ).unsqueeze(0).to(ctx.device)
    elif label is not None:
        # 2D — no cropping; just convert label
        lbl_arr = np.asarray(label).astype(np.int64)
        lbl_t = torch.from_numpy(lbl_arr).unsqueeze(0).to(ctx.device)
    logits = ctx.current_model(t)
    return logits, lbl_t


# --------------------------------------------------------------
# Tool functions — each uses kwargs from ctx.notes[<name>]
# --------------------------------------------------------------

def _fn_bn_reset(ctx: ToolContext) -> ToolContext:
    """Reset BN running stats and re-estimate them from target images
    via forward passes in train() mode. No gradient updates."""
    notes = ctx.notes.get("bn_reset", {})
    max_fwd = int(notes.get("max_forward", 200))
    model = ctx.current_model
    model.train()
    reset_norm_running_stats(model)
    with torch.no_grad():
        for img in ctx.target_images[:max_fwd]:
            try:
                model(_to_tensor(ctx, img))
            except Exception:
                pass
    model.eval()
    return ctx


def _collect_affine(model):
    params = list(iter_norm_affine_params(model))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    return params


def _fn_tent(ctx: ToolContext) -> ToolContext:
    """Tent (Wang et al., ICLR 2021): entropy minimization on norm
    affine params. Batch-stat mode (no running stats) during adapt."""
    notes = ctx.notes.get("tent", {})
    lr = float(notes.get("lr", 1e-3)); epochs = int(notes.get("epochs", 1))
    disable_running_stats(ctx.current_model)
    params = _collect_affine(ctx.current_model)
    if not params:
        return ctx
    opt = torch.optim.Adam(params, lr=lr)
    ctx.current_model.train()
    for _ in range(epochs):
        for img in ctx.target_images:
            opt.zero_grad()
            logits, _ = _train_forward(ctx, img)
            probs = F.softmax(logits, dim=1)
            ent = _entropy(probs)
            ent.backward(); opt.step()
    ctx.current_model.eval()
    return ctx


def _fn_sar(ctx: ToolContext) -> ToolContext:
    """SAR (Niu et al., ICLR 2023): Tent + sharpness-aware step +
    entropy-margin reliability filter."""
    notes = ctx.notes.get("sar", {})
    lr = float(notes.get("lr", 5e-3)); rho = float(notes.get("rho", 0.05))
    ent_margin = float(notes.get("ent_margin", 0.4))
    epochs = int(notes.get("epochs", 1))
    disable_running_stats(ctx.current_model)
    params = _collect_affine(ctx.current_model)
    if not params:
        return ctx
    ctx.current_model.train()
    with torch.no_grad():
        logits0, _ = _train_forward(ctx, ctx.target_images[0])
        num_classes = logits0.shape[1]
    threshold = ent_margin * float(np.log(max(num_classes, 2)))
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    for _ in range(epochs):
        for img in ctx.target_images:
            opt.zero_grad()
            logits1, _ = _train_forward(ctx, img)
            ent1 = _entropy(F.softmax(logits1, dim=1))
            if ent1.item() > threshold:
                continue
            ent1.backward()
            with torch.no_grad():
                grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params
                                             if p.grad is not None) + 1e-12)
                scale = rho / (grad_norm + 1e-12)
                for p in params:
                    if p.grad is not None:
                        e_w = p.grad * scale
                        p.add_(e_w); p._sar_perturb = e_w
            opt.zero_grad()
            logits2, _ = _train_forward(ctx, img)
            _entropy(F.softmax(logits2, dim=1)).backward()
            with torch.no_grad():
                for p in params:
                    if hasattr(p, "_sar_perturb"):
                        p.sub_(p._sar_perturb); del p._sar_perturb
            opt.step()
    ctx.current_model.eval()
    return ctx


def _fn_eata(ctx: ToolContext) -> ToolContext:
    """EATA (Niu et al., ICML 2022): Tent + reliable-sample filter +
    cosine redundancy filter + source-anchor regularization."""
    notes = ctx.notes.get("eata", {})
    lr = float(notes.get("lr", 1e-3))
    ent_margin = float(notes.get("ent_margin", 0.4))
    anchor_beta = float(notes.get("anchor_beta", 2000.0))
    cos_thr = float(notes.get("cosine_thresh", 0.05))
    epochs = int(notes.get("epochs", 1))
    disable_running_stats(ctx.current_model)
    params = _collect_affine(ctx.current_model)
    if not params:
        return ctx
    anchor = [p.detach().clone() for p in params]
    ctx.current_model.train()
    with torch.no_grad():
        logits0, _ = _train_forward(ctx, ctx.target_images[0])
        num_classes = logits0.shape[1]
    threshold = ent_margin * float(np.log(max(num_classes, 2)))
    opt = torch.optim.Adam(params, lr=lr)
    running_mean_prob = None
    for _ in range(epochs):
        for img in ctx.target_images:
            opt.zero_grad()
            logits, _ = _train_forward(ctx, img)
            probs = F.softmax(logits, dim=1)
            ent = _entropy(probs)
            if ent.item() > threshold:
                continue
            # Mean over spatial dims (H,W for 4D logits; D,H,W for 5D)
            mean_prob = probs.mean(dim=list(range(2, probs.ndim))).flatten().detach()
            if running_mean_prob is not None:
                cos = F.cosine_similarity(running_mean_prob, mean_prob,
                                            dim=0).item()
                if cos > 1.0 - cos_thr:
                    continue
                running_mean_prob = 0.9 * running_mean_prob + 0.1 * mean_prob
            else:
                running_mean_prob = mean_prob
            anchor_loss = sum(((p - a) ** 2).sum()
                                for p, a in zip(params, anchor))
            (ent + (1.0 / anchor_beta) * anchor_loss).backward()
            opt.step()
    ctx.current_model.eval()
    return ctx


# --------------------------------------------------------------

register(Tool(
    name="bn_reset", fn=_fn_bn_reset,
    category="stats_adapt", side_effects=["model_weights"],
    requires=["bn_running_stats"],
    profile=(
        "MECHANISM: zeros BN's running mean/var, then re-estimates them by forwarding target images in train mode — corrects CHANNEL-STATISTICS shift without touching learned weights. No gradients. Reset BN running stats and re-estimate from target via forward "
        "passes. No gradient updates — trivially safe, very fast. Great "
        "first try on pure channel-stats shift (scanner-vendor change). "
        "Requires BN architecture (won't help on IN/LN)."),
))
register(Tool(
    name="tent", fn=_fn_tent,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["tent_affine"],
    profile=(
        "MECHANISM: gradient-descent on norm-layer affine (gamma/beta) params to minimize per-pixel softmax entropy on target — corrects DECISION-BOUNDARY drift by sharpening confident predictions. Tent: entropy minimization on norm-affine params, Adam lr=1e-3, "
        "1 epoch. Workhorse TTA; works broadly when source is moderately "
        "miscalibrated on target. Risk: low-confidence starts can drift "
        "further. Pair with `percentile_clip` or `histogram_match` for "
        "bigger shifts."),
))
register(Tool(
    name="sar", fn=_fn_sar,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["tent_affine"],
    profile=(
        "MECHANISM: Tent's entropy minimization plus a sharpness-aware perturbation step + reliable-sample filter that skips high-entropy (broken) samples — corrects DECISION-BOUNDARY drift more robustly to noisy target sets. SAR: Tent + sharpness-aware minimizer + entropy-margin filter. "
        "Better than Tent on noisy target sets (some clearly-broken "
        "samples). Skips high-entropy samples; SGD+SAM step. Small "
        "overhead vs Tent."),
))
register(Tool(
    name="eata", fn=_fn_eata,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["tent_affine"],
    profile=(
        "MECHANISM: Tent's entropy minimization plus reliable-sample filter, redundancy filter, and an L2 anchor pulling weights back toward source — corrects DECISION-BOUNDARY drift while staying close to the source model. EATA: Tent + reliable-sample filter + redundancy filter + "
        "source-anchor regularization. Better than Tent when you want "
        "to stay close to source (shift is mild) and filter out "
        "broken/duplicate samples. Most conservative gradient TTA."),
))

# --------------------------------------------------------------
# Semi-supervised variants (b-5): when K target labels are
# available, add a CE term on the labeled samples to the
# entropy / SAR / EATA losses. The CE pulls the model toward
# correct predictions on K real samples while the entropy term
# still operates on the full target set.
# --------------------------------------------------------------

def _ce_loss_on_labels(ctx: ToolContext, num_classes: int) -> torch.Tensor:
    """Mean cross-entropy across K labeled samples. Dispatches via
    ctx.task so seg uses pixel-CE and cls uses image-CE. Returns a
    scalar tensor on ctx.device. Returns 0 if no task or no labels."""
    if not ctx.fewshot_indices or not ctx.fewshot_labels:
        return torch.tensor(0.0, device=ctx.device)
    if ctx.task is None:
        # Default to segmentation task for backward compat
        from medadaptagent.tasks import get_task
        task = get_task("segmentation")
    else:
        task = ctx.task
    return task.ce_loss_on_labels(
        model=ctx.current_model,
        images=ctx.target_images,
        labels=ctx.fewshot_labels,
        indices=ctx.fewshot_indices,
        preprocess_fn=ctx.preprocess_fn,
        num_classes=num_classes,
        device=ctx.device,
    )


def _fn_tent_ss(ctx: ToolContext) -> ToolContext:
    """Semi-supervised Tent: entropy on full target + lambda * CE on K labels.

    Equivalent to Tent when no labels available (falls back). When labels
    are present, the supervised term anchors the model toward correct
    predictions on K samples while entropy still calibrates over the rest.

    kwargs: lr (default 1e-3), epochs (default 1), lambda_ce (default 0.5)
    """
    if not ctx.fewshot_indices or not ctx.fewshot_labels:
        return _fn_tent(ctx)
    notes = ctx.notes.get("tent_ss", {})
    lr = float(notes.get("lr", 1e-3)); epochs = int(notes.get("epochs", 1))
    lam = float(notes.get("lambda_ce", 0.5))
    disable_running_stats(ctx.current_model)
    params = _collect_affine(ctx.current_model)
    if not params:
        return ctx
    opt = torch.optim.Adam(params, lr=lr)
    ctx.current_model.train()
    fewshot_lookup = dict(zip(ctx.fewshot_indices, ctx.fewshot_labels))
    for _ in range(epochs):
        for i, img in enumerate(ctx.target_images):
            opt.zero_grad()
            label = fewshot_lookup.get(i)
            logits, lbl_t = _train_forward(ctx, img, label=label)
            ent = _entropy(F.softmax(logits, dim=1))
            loss = ent
            if lbl_t is not None and logits.shape[-3:] == lbl_t.shape[-3:]:
                ce = F.cross_entropy(logits, lbl_t)
                loss = ent + lam * ce
            loss.backward(); opt.step()
    ctx.current_model.eval()
    return ctx


def _fn_sar_ss(ctx: ToolContext) -> ToolContext:
    """Semi-supervised SAR: SAR's sharpness-aware step + lambda * CE on K labels.
    Falls back to pure SAR when no labels available."""
    if not ctx.fewshot_indices or not ctx.fewshot_labels:
        return _fn_sar(ctx)
    notes = ctx.notes.get("sar_ss", {})
    lr = float(notes.get("lr", 5e-3)); rho = float(notes.get("rho", 0.05))
    ent_margin = float(notes.get("ent_margin", 0.4))
    epochs = int(notes.get("epochs", 1))
    lam = float(notes.get("lambda_ce", 0.5))
    disable_running_stats(ctx.current_model)
    params = _collect_affine(ctx.current_model)
    if not params:
        return ctx
    ctx.current_model.train()
    with torch.no_grad():
        logits0, _ = _train_forward(ctx, ctx.target_images[0])
        num_classes = logits0.shape[1]
    threshold = ent_margin * float(np.log(max(num_classes, 2)))
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    fewshot_lookup = dict(zip(ctx.fewshot_indices, ctx.fewshot_labels))
    for _ in range(epochs):
        for i, img in enumerate(ctx.target_images):
            opt.zero_grad()
            label = fewshot_lookup.get(i)
            # NOTE: SAR's two-forward SAM step needs the SAME patch both
            # times. _train_forward random-samples a new patch each call,
            # so we sample once and reuse via patch_size=None on the 2nd
            # call — instead, run forward, snapshot the input patch, and
            # reuse it for the post-perturb forward.
            logits1, lbl_t = _train_forward(ctx, img, label=label)
            ent1 = _entropy(F.softmax(logits1, dim=1))
            loss1 = ent1
            if lbl_t is not None and logits1.shape[-3:] == lbl_t.shape[-3:]:
                ce = F.cross_entropy(logits1, lbl_t)
                loss1 = ent1 + lam * ce
            if ent1.item() > threshold:
                # high-entropy filter: skip SAM step entirely
                continue
            loss1.backward()
            with torch.no_grad():
                grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params
                                             if p.grad is not None) + 1e-12)
                scale = rho / (grad_norm + 1e-12)
                for p in params:
                    if p.grad is not None:
                        e_w = p.grad * scale
                        p.add_(e_w); p._sar_perturb = e_w
            opt.zero_grad()
            # Second forward: a fresh random patch is fine — SAR's
            # sharpness step is over weight space, not input space.
            logits2, lbl_t2 = _train_forward(ctx, img, label=label)
            ent2 = _entropy(F.softmax(logits2, dim=1))
            loss2 = ent2
            if lbl_t2 is not None and logits2.shape[-3:] == lbl_t2.shape[-3:]:
                loss2 = ent2 + lam * F.cross_entropy(logits2, lbl_t2)
            loss2.backward()
            with torch.no_grad():
                for p in params:
                    if hasattr(p, "_sar_perturb"):
                        p.sub_(p._sar_perturb); del p._sar_perturb
            opt.step()
    ctx.current_model.eval()
    return ctx


def _fn_eata_ss(ctx: ToolContext) -> ToolContext:
    """Semi-supervised EATA: EATA + lambda * CE on K labels.
    Falls back to pure EATA when no labels available."""
    if not ctx.fewshot_indices or not ctx.fewshot_labels:
        return _fn_eata(ctx)
    notes = ctx.notes.get("eata_ss", {})
    lr = float(notes.get("lr", 1e-3))
    ent_margin = float(notes.get("ent_margin", 0.4))
    anchor_beta = float(notes.get("anchor_beta", 2000.0))
    cos_thr = float(notes.get("cosine_thresh", 0.05))
    epochs = int(notes.get("epochs", 1))
    lam = float(notes.get("lambda_ce", 0.5))
    disable_running_stats(ctx.current_model)
    params = _collect_affine(ctx.current_model)
    if not params:
        return ctx
    anchor = [p.detach().clone() for p in params]
    ctx.current_model.train()
    with torch.no_grad():
        logits0, _ = _train_forward(ctx, ctx.target_images[0])
        num_classes = logits0.shape[1]
    threshold = ent_margin * float(np.log(max(num_classes, 2)))
    opt = torch.optim.Adam(params, lr=lr)
    running_mean_prob = None
    fewshot_lookup = dict(zip(ctx.fewshot_indices, ctx.fewshot_labels))
    for _ in range(epochs):
        for i, img in enumerate(ctx.target_images):
            opt.zero_grad()
            label = fewshot_lookup.get(i)
            logits, lbl_t = _train_forward(ctx, img, label=label)
            probs = F.softmax(logits, dim=1)
            ent = _entropy(probs)
            ce_present = lbl_t is not None and logits.shape[-3:] == lbl_t.shape[-3:]
            if ent.item() > threshold:
                # high-entropy unreliable — but still apply CE if labeled
                if ce_present:
                    (lam * F.cross_entropy(logits, lbl_t)).backward()
                    opt.step()
                continue
            # Mean over spatial dims (H,W for 4D logits; D,H,W for 5D)
            mean_prob = probs.mean(dim=list(range(2, probs.ndim))).flatten().detach()
            if running_mean_prob is not None:
                cos = F.cosine_similarity(running_mean_prob, mean_prob,
                                            dim=0).item()
                if cos > 1.0 - cos_thr:
                    continue
                running_mean_prob = 0.9 * running_mean_prob + 0.1 * mean_prob
            else:
                running_mean_prob = mean_prob
            anchor_loss = sum(((p - a) ** 2).sum()
                                for p, a in zip(params, anchor))
            total = ent + (1.0 / anchor_beta) * anchor_loss
            if ce_present:
                total = total + lam * F.cross_entropy(logits, lbl_t)
            total.backward()
            opt.step()
    ctx.current_model.eval()
    return ctx


def _find_output_head(model):
    """Heuristic: return the last Conv2d/Conv3d/Linear module in the
    model. Most segmentation heads are a 1x1 (or 1x1x1) Conv that maps
    feature channels to num_classes."""
    last = None
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d,
                            torch.nn.Linear)):
            last = m
    return last


def _fn_finetune_head(ctx: ToolContext) -> ToolContext:
    """Few-shot fine-tune the OUTPUT HEAD only (last Conv/Linear). Most
    conservative finetune — recalibrates the classifier mapping without
    touching encoder features. Falls back to no-op when no labels."""
    if not ctx.fewshot_indices or not ctx.fewshot_labels:
        return ctx
    notes = ctx.notes.get("finetune_head", {})
    lr = float(notes.get("lr", 1e-3)); steps = int(notes.get("steps", 50))
    head = _find_output_head(ctx.current_model)
    if head is None:
        return ctx
    for p in ctx.current_model.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(True)
    params = [p for p in head.parameters() if p.requires_grad]
    if not params:
        return ctx
    opt = torch.optim.Adam(params, lr=lr)
    ctx.current_model.train()
    fewshot_lookup = dict(zip(ctx.fewshot_indices, ctx.fewshot_labels))
    labeled_indices = list(fewshot_lookup.keys())
    if not labeled_indices:
        return ctx
    for step in range(steps):
        i = labeled_indices[step % len(labeled_indices)]
        img = ctx.target_images[i]
        label = fewshot_lookup[i]
        opt.zero_grad()
        logits, lbl_t = _train_forward(ctx, img, label=label)
        if lbl_t is None or logits.shape[-3:] != lbl_t.shape[-3:]:
            continue
        F.cross_entropy(logits, lbl_t).backward(); opt.step()
    ctx.current_model.eval()
    return ctx


def _fn_finetune_partial(ctx: ToolContext) -> ToolContext:
    """Few-shot fine-tune the LAST 30% of parameters by depth order
    (typically the decoder / upsample path). Middle ground between
    norm-only (tent_ss) and full fine-tune. Adds an entropy term on
    every target image plus CE on K labeled patches.

    kwargs: lr (default 1e-5, kept tiny for stability), steps (default
    50), tail_frac (default 0.3), lambda_ce (default 1.0)."""
    if not ctx.fewshot_indices or not ctx.fewshot_labels:
        return ctx
    notes = ctx.notes.get("finetune_partial", {})
    lr = float(notes.get("lr", 1e-5)); steps = int(notes.get("steps", 50))
    tail_frac = float(notes.get("tail_frac", 0.3))
    lam = float(notes.get("lambda_ce", 1.0))
    named = list(ctx.current_model.named_parameters())
    if not named:
        return ctx
    tail_n = max(1, int(len(named) * tail_frac))
    tail = named[-tail_n:]
    for _, p in named:
        p.requires_grad_(False)
    for _, p in tail:
        p.requires_grad_(True)
    params = [p for _, p in tail if p.requires_grad]
    if not params:
        return ctx
    opt = torch.optim.Adam(params, lr=lr)
    ctx.current_model.train()
    fewshot_lookup = dict(zip(ctx.fewshot_indices, ctx.fewshot_labels))
    n_img = len(ctx.target_images)
    for step in range(steps):
        i = step % n_img
        img = ctx.target_images[i]
        label = fewshot_lookup.get(i)
        opt.zero_grad()
        logits, lbl_t = _train_forward(ctx, img, label=label)
        ent = _entropy(F.softmax(logits, dim=1))
        loss = ent
        if lbl_t is not None and logits.shape[-3:] == lbl_t.shape[-3:]:
            loss = ent + lam * F.cross_entropy(logits, lbl_t)
        loss.backward(); opt.step()
    ctx.current_model.eval()
    return ctx


class _LoRAConv(torch.nn.Module):
    """LoRA adapter wrapping a Conv2d or Conv3d. Original conv stays
    frozen; trainable low-rank update Δy = scale * B(A(x)) is added to
    the base output. At init Δy = 0 (B initialized to zero) so the
    wrapped layer is functionally identical to the base."""

    def __init__(self, base, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        is_3d = isinstance(base, torch.nn.Conv3d)
        Conv = torch.nn.Conv3d if is_3d else torch.nn.Conv2d
        in_c, out_c = base.in_channels, base.out_channels
        ksz = base.kernel_size
        stride = base.stride
        padding = base.padding
        dilation = base.dilation
        # A: in -> r with same receptive field
        self.lora_A = Conv(in_c, rank, kernel_size=ksz, stride=stride,
                            padding=padding, dilation=dilation, bias=False)
        # B: r -> out, 1x1(x1) — pointwise mixing
        ones = (1,) * (3 if is_3d else 2)
        self.lora_B = Conv(rank, out_c, kernel_size=ones, bias=False)
        torch.nn.init.kaiming_normal_(self.lora_A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.scale = float(alpha) / float(rank)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def _wrap_convs_with_lora(model, rank=4, alpha=1.0):
    """Walk the module tree and replace every Conv2d/Conv3d (whose
    kernel volume > 1 — skip pure 1x1 projection convs) with _LoRAConv.
    Returns a list of newly-installed lora module references for the
    optimizer. Modifies model in place."""
    new_lora = []
    for name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, (torch.nn.Conv2d, torch.nn.Conv3d)):
                k = child.kernel_size
                kv = 1
                for d in k:
                    kv *= d
                if kv <= 1:
                    continue  # skip pointwise; not useful for LoRA
                wrapped = _LoRAConv(child, rank=rank, alpha=alpha).to(
                    next(child.parameters()).device)
                setattr(parent, child_name, wrapped)
                new_lora.append(wrapped)
    return new_lora


def _fn_finetune_lora(ctx: ToolContext) -> ToolContext:
    """Few-shot LoRA finetune. Wraps every multi-kernel Conv with a
    low-rank adapter (rank=4 by default), freezes all base weights,
    trains only the LoRA matrices on K labeled patches + entropy on
    unlabeled. Parameter-efficient — typically <2% trainable params.

    kwargs: lr (default 1e-3), steps (default 50), rank (default 4),
    alpha (default 1.0 — LoRA scaling factor), lambda_ce (default 1.0)."""
    if not ctx.fewshot_indices or not ctx.fewshot_labels:
        return ctx
    notes = ctx.notes.get("finetune_lora", {})
    lr = float(notes.get("lr", 1e-3)); steps = int(notes.get("steps", 50))
    rank = int(notes.get("rank", 4))
    alpha = float(notes.get("alpha", 1.0))
    lam = float(notes.get("lambda_ce", 1.0))

    # Freeze everything, then wrap convs (which freezes base + adds adapters)
    for p in ctx.current_model.parameters():
        p.requires_grad_(False)
    lora_mods = _wrap_convs_with_lora(ctx.current_model, rank=rank,
                                          alpha=alpha)
    if not lora_mods:
        return ctx
    params = [p for m in lora_mods for p in m.parameters()
                if p.requires_grad]
    if not params:
        return ctx
    opt = torch.optim.Adam(params, lr=lr)
    ctx.current_model.train()
    fewshot_lookup = dict(zip(ctx.fewshot_indices, ctx.fewshot_labels))
    n_img = len(ctx.target_images)
    for step in range(steps):
        i = step % n_img
        img = ctx.target_images[i]
        label = fewshot_lookup.get(i)
        opt.zero_grad()
        logits, lbl_t = _train_forward(ctx, img, label=label)
        ent = _entropy(F.softmax(logits, dim=1))
        loss = ent
        if lbl_t is not None and logits.shape[-3:] == lbl_t.shape[-3:]:
            loss = ent + lam * F.cross_entropy(logits, lbl_t)
        loss.backward(); opt.step()
    ctx.current_model.eval()
    return ctx


register(Tool(
    name="tent_ss", fn=_fn_tent_ss,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["tent_affine", "fewshot_labels"],
    profile=(
        "MECHANISM: Tent's entropy minimization on norm-affine params PLUS "
        "a cross-entropy term (lambda=0.5 by default) on K labeled target "
        "samples — true semi-supervised TTA. Requires K target labels. "
        "When the dataset is hard for pure TTA (mnms-class regression "
        "patterns), tent_ss often closes the gap that pure Tent leaves."),
))
register(Tool(
    name="sar_ss", fn=_fn_sar_ss,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["tent_affine", "fewshot_labels"],
    profile=(
        "MECHANISM: SAR (sharpness-aware + entropy-margin filter) PLUS CE "
        "on K labeled samples. Robust semi-supervised variant when target "
        "samples are noisy/heterogeneous. Requires K target labels."),
))
register(Tool(
    name="eata_ss", fn=_fn_eata_ss,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["tent_affine", "fewshot_labels"],
    profile=(
        "MECHANISM: EATA (Tent + reliable-sample filter + redundancy "
        "filter + source-anchor regularization) PLUS CE on K labeled "
        "samples. Most conservative semi-supervised variant. Requires "
        "K target labels."),
))
register(Tool(
    name="finetune_head", fn=_fn_finetune_head,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["fewshot_labels"],
    profile=(
        "MECHANISM: brief supervised fine-tune of the OUTPUT HEAD only "
        "(last Conv/Linear that maps features to num_classes) on K "
        "labeled target patches — recalibrates the classifier mapping "
        "without touching encoder features. Most conservative finetune; "
        "low overfit risk; useful when only the label-space mapping "
        "needs to be re-fit. Requires K target labels."),
))
register(Tool(
    name="finetune_partial", fn=_fn_finetune_partial,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["fewshot_labels"],
    profile=(
        "MECHANISM: brief supervised fine-tune of the LAST 30% of "
        "parameters (decoder / upsample path on most U-shaped nets) on "
        "K labeled target patches PLUS entropy minimization on the "
        "remaining unlabeled targets. Middle ground between norm-only "
        "(tent_ss) and full fine-tune. Lr=1e-5 tiny for stability. "
        "Requires K target labels."),
))
register(Tool(
    name="finetune_lora", fn=_fn_finetune_lora,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["fewshot_labels"],
    profile=(
        "MECHANISM: wraps every multi-kernel Conv2d/Conv3d in the model "
        "with a low-rank LoRA adapter (rank=4 by default). Base conv "
        "weights stay frozen; only the small A and B adapter matrices "
        "train on K labeled patches plus entropy on unlabeled. "
        "Parameter-efficient finetune — typically under 2% trainable "
        "params. Lower overfit risk than finetune_partial, more "
        "expressive than finetune_head. Requires K target labels."),
))


# --------------------------------------------------------------
# Label-free LoRA test-time adaptation (P1). Unlike finetune_lora
# (supervised, needs K labels), this minimizes prediction ENTROPY on
# the UNLABELED target, updating ONLY the injected LoRA adapter
# matrices. Available at K=0. Reuses the same _LoRAConv /
# _wrap_convs_with_lora helpers used by finetune_lora.
# --------------------------------------------------------------

def _fn_lora_tta(ctx: ToolContext) -> ToolContext:
    """Label-free LoRA test-time adaptation. Injects rank-`rank` low-rank
    adapters into every multi-kernel Conv2d/Conv3d, freezes ALL base
    weights, and minimizes prediction ENTROPY on the unlabeled target —
    updating ONLY the LoRA matrices. No target labels are needed.

    Versus `tent`: trains low-rank conv adapters instead of norm-affine
    params, so the adaptable subspace is richer than gamma/beta yet still
    a tiny fraction of the backbone (typically ~100x fewer trainable
    params than full fine-tune), which lets a more aggressive learning
    rate stay stable. Versus `finetune_lora`: that one is SUPERVISED
    (needs K labels + CE); this one is fully unsupervised (entropy only),
    so it is available at K=0.

    A SAR-style entropy-margin reliability filter skips high-entropy
    (likely-broken) samples so noisy target slices do not pull the
    adapter off course.

    3D-safe: every forward routes through `_train_forward` (patch-crops
    5D volumes) and `_entropy` reads only the class dim, so 4D
    [N,C,H,W] and 5D [N,C,D,H,W] logits both work. Norm-agnostic: the
    adapted params are LoRA conv weights, independent of BN/IN/LN/GN.

    kwargs:
      rank        (default 4)    : LoRA rank for the injected adapters
      alpha       (default 1.0)  : LoRA scaling factor (Delta = alpha/rank)
      lr          (default 5e-3) : Adam lr; higher than Tent is safe here
                                    because only the low-rank params move
      epochs      (default 1)    : passes over the target set
      ent_margin  (default 0.4)  : reliability filter; samples whose
                                    entropy exceeds ent_margin*log(C) are
                                    skipped (0 disables the filter)
    """
    notes = ctx.notes.get("lora_tta", {})
    rank = int(notes.get("rank", 4))
    alpha = float(notes.get("alpha", 1.0))
    lr = float(notes.get("lr", 5e-3))
    epochs = int(notes.get("epochs", 1))
    ent_margin = float(notes.get("ent_margin", 0.4))

    if not ctx.target_images:
        return ctx

    # Freeze everything; wrap convs (each _LoRAConv re-freezes its base
    # and adds zero-initialized adapters so the model is identical at
    # step 0).
    for p in ctx.current_model.parameters():
        p.requires_grad_(False)
    lora_mods = _wrap_convs_with_lora(ctx.current_model, rank=rank,
                                         alpha=alpha)
    if not lora_mods:
        # No multi-kernel convs to adapt — leave the model untouched.
        return ctx
    params = [p for m in lora_mods for p in m.parameters()
                if p.requires_grad]
    if not params:
        return ctx

    opt = torch.optim.Adam(params, lr=lr)
    ctx.current_model.train()

    # Per-sample entropy threshold from class count (3D-safe: logits may
    # be 4D or 5D; we only read dim 1).
    threshold = None
    if ent_margin > 0:
        with torch.no_grad():
            logits0, _ = _train_forward(ctx, ctx.target_images[0])
            num_classes = logits0.shape[1]
        threshold = ent_margin * float(np.log(max(num_classes, 2)))

    for _ in range(epochs):
        for img in ctx.target_images:
            opt.zero_grad()
            logits, _ = _train_forward(ctx, img)
            ent = _entropy(F.softmax(logits, dim=1))
            if threshold is not None and ent.item() > threshold:
                continue  # reliability filter — skip likely-broken sample
            ent.backward()
            opt.step()
    ctx.current_model.eval()
    return ctx


register(Tool(
    name="lora_tta", fn=_fn_lora_tta,
    category="gradient_adapt", side_effects=["model_weights"],
    requires=["tent_affine"],
    profile=(
        "MECHANISM: injects rank-r low-rank LoRA adapters into every "
        "multi-kernel Conv2d/Conv3d, freezes all base weights, and "
        "minimizes per-prediction softmax ENTROPY on the UNLABELED target "
        "(updating only the LoRA matrices) — corrects DECISION-BOUNDARY "
        "drift through a richer subspace than norm-affine (Tent) while "
        "keeping ~100x fewer trainable params than full fine-tune, so a "
        "more aggressive lr (5e-3) stays stable. Label-free: works at K=0. "
        "Distinct from finetune_lora, which is supervised. A SAR-style "
        "entropy-margin filter skips high-entropy samples. Try when Tent "
        "underfits the shift but full/partial fine-tune overfits or has "
        "no labels."),
))
