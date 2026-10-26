"""Bridge the harness adapters to the EXISTING medadaptagent/tools/ registry, so the
agent's per-task toolbox is the full ~27-tool arsenal (preprocessing, spatial, test-time
aug, BN/gradient TTA, frequency, postprocessing, meta) — capability-filtered per model —
instead of the 3 hand-listed tools the rebuild kept.

A registry tool is `fn(ToolContext)->ToolContext`; it may adapt the model weights OR wrap
inference (preprocess→model→postprocess). So a bridged candidate is a ToolContext (not a
bare nn.Module), and scoring runs through the registry's `run_inference`.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from medadaptagent.tools import (TOOL_REGISTRY, ToolContext, apply_tool,
                                  available_tools, run_inference)
from medadaptagent.graph.norm_utils import model_capabilities

# Harness owns holdout-aware finetune_* + these aren't useful as agent moves.
_EXCLUDE_NAMES = {"finetune_head", "finetune_partial", "finetune_lora", "noop", "chain"}
# Registry is seg-oriented; for classification keep only model/input transforms.
_CLS_KEEP_CATS = {"stats_adapt", "gradient_adapt", "preprocessing", "frequency"}


def make_ctx(model, imgs, nc, device, task=None, fewshot_idx=None, fewshot_lbls=None):
    return ToolContext(source_model=model, current_model=model, target_images=imgs,
                       device=device, num_classes=nc, task=task,
                       fewshot_indices=fewshot_idx, fewshot_labels=fewshot_lbls)


def apply_registry_tool(name, model, imgs, nc, device, task=None,
                        fewshot_idx=None, fewshot_lbls=None):
    """Apply ONE registry tool to a FRESH context built from a deep copy of `model`.
    Returns the resulting ToolContext (current_model possibly adapted; preprocess/
    postprocess/inference possibly wrapped)."""
    ctx = make_ctx(copy.deepcopy(model).to(device), imgs, nc, device, task,
                   fewshot_idx, fewshot_lbls)
    return apply_tool(name, ctx)


def bridged_specs(task_type, model):
    """The registry Tools applicable to (task_type, model). Returns (list[Tool], caps)."""
    caps = list(model_capabilities(model))                 # norm caps (bn_running_stats/tent_affine)
    is_cls = str(task_type).startswith("cls") or str(task_type) == "classification"
    if not is_cls:
        caps.append("pixel_predictions")                   # seg masks -> enable postprocessing tools
    out = []
    for t in available_tools(caps):
        if t.name in _EXCLUDE_NAMES:
            continue
        if is_cls and t.category not in _CLS_KEEP_CATS:
            continue
        out.append(t)
    return out, caps


# ---- ctx-aware scoring (seg) -------------------------------------------------
def ctx_masks(ctx, imgs, to_tensor):
    return [run_inference(ctx, img, to_tensor) for img in imgs]


def ctx_dice(ctx, imgs, lbls, nc, to_tensor):
    """Mean per-image foreground Dice via the full preprocess→model→postprocess chain."""
    dices = []
    for img, lbl in zip(imgs, lbls):
        pred = run_inference(ctx, img, to_tensor)
        lbl = np.asarray(lbl)
        if pred.shape != lbl.shape:
            continue
        ds = []
        for c in range(1, nc):
            p = (pred == c); g = (lbl == c)
            if g.sum() == 0 and p.sum() == 0:
                continue
            ds.append(2.0 * (p & g).sum() / (p.sum() + g.sum() + 1e-6))
        if ds:
            dices.append(float(np.mean(ds)))
    return float(np.mean(dices)) if dices else 0.0


def ctx_stats_seg(ctx, imgs, to_tensor, n=24):
    """Label-free quality signals from the ctx: fg fraction + fragmentation (from final
    masks) and softmax entropy (from logits, pre-postproc). Mirrors _predict_stats keys."""
    from scipy import ndimage
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs)), dtype=int)
    fg, frag, ent = [], [], []
    inf = ctx.inference_fn or (lambda m, t: m(t))
    with torch.no_grad():
        for i in idx:
            img = imgs[int(i)]
            pre = ctx.preprocess_fn(img)
            t = to_tensor(pre, ctx.device)
            logit = inf(ctx.current_model, t)
            p = F.softmax(logit, 1)
            e = float((-(p * (p + 1e-9).log()).sum(1)).mean().cpu())
            mask = ctx.postprocess_fn(logit.argmax(1)[0].cpu().numpy())
            fg.append(float((mask >= 1).mean()))
            _, ncomp = ndimage.label(mask >= 1)
            frag.append(ncomp)
            ent.append(e)
    mean_components = float(np.mean(frag))
    return {"fg_frac": float(np.mean(fg)), "entropy": float(np.mean(ent)),
            "n_comp": mean_components, "frag": mean_components}


# ---- candidate wrapper so a ctx flows through the harness like a model -------
class CtxCandidate(torch.nn.Module):
    """Wraps a ToolContext so it can be a harness candidate (shipped, safety-checked,
    audited) like a bare model. Scoring branches on isinstance(CtxCandidate) and uses the
    full preprocess→model→postprocess chain; `forward` is best-effort logits (no numpy
    preprocess) for any caller that still tensorises first."""
    def __init__(self, ctx, to_tensor):
        super().__init__()
        self._ctx = ctx
        self._tt = to_tensor

    @property
    def ctx(self):
        return self._ctx

    def predict_mask(self, img):
        return run_inference(self._ctx, img, self._tt)

    def forward(self, t):
        inf = self._ctx.inference_fn or (lambda m, x: m(x))
        return inf(self._ctx.current_model, t)


def seg_dice_of(cand, imgs, lbls, nc, device, to_tensor=None):
    """Held-out / oracle Dice that honors a ctx candidate's full inference chain."""
    if isinstance(cand, CtxCandidate):
        return ctx_dice(cand.ctx, imgs, lbls, nc, to_tensor or cand._tt)
    from scripts.run_baseline_tta import _avg_dice
    return float(_avg_dice(cand, imgs, lbls, nc, device))


def seg_stats_of(cand, imgs, device, to_tensor=None, n=24, predict_stats=None):
    """Label-free stats that honor a ctx candidate's chain; bare models use the adapter's
    `predict_stats` (passed in to avoid a circular import)."""
    if isinstance(cand, CtxCandidate):
        return ctx_stats_seg(cand.ctx, imgs, to_tensor or cand._tt, n)
    return predict_stats(cand, imgs, device, n)


# ---- ctx-aware scoring (classification) -------------------------------------
def _ctx_probs(ctx, imgs, to_tensor):
    """Per-image class probabilities through preprocess→model (no postproc/argmax)."""
    inf = ctx.inference_fn or (lambda m, t: m(t))
    out = []
    with torch.no_grad():
        for img in imgs:
            pre = ctx.preprocess_fn(img)
            logit = inf(ctx.current_model, to_tensor(pre, ctx.device))
            if logit.dim() > 2:
                logit = logit.flatten(1)
            out.append(F.softmax(logit, 1)[0].cpu().numpy())
    return np.stack(out)                              # [N, nc]


def ctx_auroc(ctx, imgs, labs, nc, to_tensor):
    probs = _ctx_probs(ctx, imgs, to_tensor)
    y = np.asarray([int(v) for v in labs])
    from sklearn.metrics import roc_auc_score
    score = float(roc_auc_score(y, probs[:, 1] if nc == 2 else probs, multi_class="ovr"))
    if not np.isfinite(score):
        raise ValueError("AUROC is undefined for these labels; accuracy was not substituted")
    return score


def ctx_stats_cls(ctx, imgs, to_tensor, n=64):
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs)), dtype=int)
    probs = _ctx_probs(ctx, [imgs[int(i)] for i in idx], to_tensor)
    conf = probs.max(1)
    ent = -(probs * np.log(probs + 1e-9)).sum(1)
    pred = probs.argmax(1)
    frac = np.bincount(pred, minlength=probs.shape[1]).max() / len(pred)
    return {"confidence": float(conf.mean()), "entropy": float(ent.mean()),
            "max_class_frac": float(frac)}


def cls_metric_of(cand, imgs, labs, nc, device, metric_kind="auroc", to_tensor=None, metric_fn=None):
    if isinstance(cand, CtxCandidate):
        return ctx_auroc(cand.ctx, imgs, labs, nc, to_tensor or cand._tt)
    return float(metric_fn(cand, imgs, labs, device, nc, metric_kind))


def cls_stats_of(cand, imgs, device, to_tensor=None, n=64, conf_stats=None):
    if isinstance(cand, CtxCandidate):
        return ctx_stats_cls(cand.ctx, imgs, to_tensor or cand._tt, n)
    return conf_stats(cand, imgs, device)


def bridged_specs_cls(model):
    """Registry tools applicable to a CLASSIFIER. The cls adapter already provides its own
    bn_reset/tent; exclude those + finetune/noop/chain. Keeps preprocessing + frequency +
    the remaining gradient TTA (sar/eata/lora_tta)."""
    specs, caps = bridged_specs("classification", model)
    return [t for t in specs if t.name not in ("bn_reset", "tent")], caps
