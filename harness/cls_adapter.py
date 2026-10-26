"""Classification TaskAdapter for the deployment harness.

Mirrors seg_adapter but for CLASSIFICATION: metric = AUROC (binary) / accuracy,
label-free signal = prediction confidence + class-collapse, adaptations = BN-stat
recalibration + entropy-minimization (tent); no mask/spectral. Reuses the verbatim
cls model builder (train_source_cls.build_model) + cls tensor transform
(tasks.classification._img_to_tensor). K-shot labels are the trustworthy signal,
same as seg. No VLM here (label-free cls quality is confidence-based); recommend_
alternative + ship-by-name + the deterministic labeled-decision logic are shared.
"""
from __future__ import annotations
import os
import copy
import numpy as np
from harness.core import Tool, SafetyVerdict, Budget

# cls bridge skips: hand-written bn_reset/tent; n4_bias_correction (MRI bias field — slow +
# irrelevant for 2D cls); multi_scale_tta (pre-existing bug).
_BRIDGE_SKIP_CLS = {"multi_scale_tta", "bn_reset", "tent", "n4_bias_correction"}


# ---- MEDICAL FOUNDATION TRANSFORMER source (RAD-DINO) ----------------------
# microsoft/rad-dino is a DINOv2 ViT-B (HF Dinov2Model, hidden_size=768, 25
# LayerNorm, ZERO BatchNorm) self-supervised on chest radiographs — a genuine
# medical foundation transformer (vs the ImageNet timm ViT/Swin branches). We
# wrap the frozen HF backbone + a Linear(768->nc) head. forward() returns
# (B, nc) logits directly so it is a drop-in for the harness, exactly like the
# resnet/vit/swin branches. The backbone is loaded from the LOCAL HF snapshot
# (HF_HUB_OFFLINE-safe) so the module structure — and hence the state_dict keys
# — are IDENTICAL between training (build_model) and harness (build_cls_model),
# giving a clean round-trip. We save the FULL state_dict (backbone + head) so
# the ckpt is self-contained and reproducible regardless of HF cache state.
_RAD_DINO_REPO = "microsoft/rad-dino"


class RadDinoClassifier:  # placeholder; real nn.Module defined in _make_rad_dino
    pass


def _make_rad_dino(nc):
    """Build the RAD-DINO backbone + linear head wrapper as an nn.Module.

    Uses pooler_output (LayerNorm(CLS token), 768-d) -> Linear(nc). Backbone is
    AutoModel.from_pretrained(microsoft/rad-dino); weights come from the local HF
    snapshot when offline. The trainer FREEZES the backbone and trains only the
    head (the foundation-ViT low-lr lesson: a frozen-backbone linear probe is
    stable; full-finetune would need lr<=5e-5)."""
    import torch.nn as nn
    from transformers import AutoModel

    class _RadDino(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(_RAD_DINO_REPO)
            hid = self.backbone.config.hidden_size  # 768
            self.head = nn.Linear(hid, num_classes)

        def forward(self, x):
            out = self.backbone(pixel_values=x)
            feat = out.pooler_output  # (B, 768) = LayerNorm(CLS token)
            return self.head(feat)

    return _RadDino(nc)


# ---- verbatim from scripts/train_source_cls.py::build_model -----------------
def build_cls_model(arch, nc):
    import torch.nn as nn
    if arch == "rad_dino":
        # MEDICAL FOUNDATION TRANSFORMER. Pure LayerNorm (25 LN, ZERO BatchNorm),
        # so bn_reset/tent are STRUCTURAL no-ops (early-return in _bn_reset/_tent);
        # the norm-agnostic transformer levers (norm_stat_tent_safe / t3a /
        # norm_stat_tent) are the adaptation path, same as the vit_small/swin_tiny
        # branches. forward returns (B, nc) directly. Backbone weights load from the
        # local HF snapshot; the harness then OVERWRITES them with the trained ckpt.
        return _make_rad_dino(nc)
    if arch == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, nc); return m
    if arch == "resnet50":
        from torchvision.models import resnet50
        m = resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, nc); return m
    if arch == "vit_small":
        import timm
        return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=nc)
    if arch == "swin_tiny":
        # timm swin_tiny_patch4_window7_224 — a hierarchical-window TRANSFORMER.
        # Pure LayerNorm (29 LayerNorms, ZERO BatchNorm), so bn_reset/tent are
        # STRUCTURAL no-ops (early-return below); the norm-agnostic levers
        # (norm_stat_tent_safe / t3a / norm_stat_tent) are the transformer-native
        # adaptation path. timm ClassifierHead pools internally -> forward returns
        # (B, nc) directly, NO reshape needed. pretrained=False here (the harness
        # loads the TRAINED ckpt state_dict); the trainer uses pretrained=True.
        import timm
        return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=nc)
    raise ValueError(f"unknown cls arch {arch}")


def load_cls_model(ckpt, arch, nc, device):
    import torch
    m = build_cls_model(arch, nc)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model_state_dict", sd.get("model", sd.get("state_dict", sd)))
    m.load_state_dict(sd)
    return m.to(device).eval()


def load_cls_npz(path):
    """Return (images list, labels list-or-None). Handles common key names."""
    z = np.load(path, allow_pickle=False)
    keys = list(z.keys())
    ik = next((k for k in ["images", "x", "X", "data", "arr_0"] if k in keys), keys[0])
    lk = next((k for k in ["labels", "y", "Y", "targets", "arr_1"] if k in keys), None)
    imgs = list(z[ik])
    labs = list(z[lk]) if lk is not None else None
    return imgs, labs


def _cls_tensor(img, device, size=224):
    from medadaptagent.tasks.classification import _img_to_tensor
    return _img_to_tensor(img, device, size)


def _predict_probs(model, imgs, device, n=None):
    import torch, torch.nn.functional as F
    model.eval()
    idx = (range(len(imgs)) if n is None else
           np.linspace(0, len(imgs) - 1, min(n, len(imgs)), dtype=int))
    out = []
    with torch.no_grad():
        for i in idx:
            logit = model(_cls_tensor(imgs[int(i)], device))
            out.append(F.softmax(logit, 1)[0].cpu().numpy())
    return np.array(out)


def _metric(model, imgs, labs, device, nc, kind="auroc"):
    p = _predict_probs(model, imgs, device)
    y = np.array([int(l) for l in labs])
    if kind == "auroc":
        from sklearn.metrics import roc_auc_score
        score = float(roc_auc_score(y, p[:, 1] if nc == 2 else p, multi_class="ovr"))
        if not np.isfinite(score):
            raise ValueError("AUROC is undefined for these labels; accuracy was not substituted")
        return score
    if kind == "accuracy":
        return float((p.argmax(1) == y).mean())
    raise ValueError("Unsupported classification metric")


def _conf_stats(model, imgs, device, n=40):
    p = _predict_probs(model, imgs, device, n=n)
    pred = p.argmax(1)
    from collections import Counter
    dist = Counter(pred.tolist())
    return {"confidence": float(p.max(1).mean()),
            "entropy": float((-(p * np.log(p + 1e-9)).sum(1)).mean()),
            "max_class_frac": float(max(dist.values()) / max(len(pred), 1))}


def _quality(s):
    # confident AND not collapsed to a single class
    collapse = 1.0 - max(0.0, s["max_class_frac"] - 0.9) / 0.1
    return float(0.7 * s["confidence"] + 0.3 * max(0.0, min(1.0, collapse)))


def _diagnose_cls(s, src):
    if s["max_class_frac"] > 0.92 and s["max_class_frac"] > src["max_class_frac"] + 0.1:
        return "class_collapse (predicting one class for almost everything)"
    if s["confidence"] < 0.75 * max(src["confidence"], 1e-6):
        return "low_confidence (calibration shift, predictions uncertain)"
    if s["entropy"] > 1.3 * max(src["entropy"], 1e-6):
        return "high_uncertainty (entropy up vs source)"
    return "mild_shift (confidence largely intact)"


def _bn_reset(model, imgs, device):
    import torch
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return                                # vit: no BN
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    with torch.no_grad():
        for i in np.linspace(0, len(imgs) - 1, min(64, len(imgs)), dtype=int):
            model(_cls_tensor(imgs[int(i)], device))
    model.eval()


def _tent(model, imgs, device, epochs=2, lr=1e-3):
    import torch, torch.nn.functional as F
    params = [p for m in model.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
              for p in (m.weight, m.bias) if p is not None]
    if not params:
        return                                # vit: no BN affine to tune
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(epochs):
        for i in np.linspace(0, len(imgs) - 1, min(64, len(imgs)), dtype=int):
            opt.zero_grad()
            logit = model(_cls_tensor(imgs[int(i)], device))
            p = F.softmax(logit, 1)
            (-(p * torch.log(p + 1e-9)).sum(1).mean()).backward()
            opt.step()
    model.eval()


class ClsTask:
    dims = "2D"

    def __init__(self, source_model, simgs, timgs, nc, device, tlabs=None,
                 metric="auroc", task_desc="", explore_eval_n=None):
        self.type = "classification"
        self.source_model = source_model
        self.simgs = simgs
        # SPEED / EXPLORATION-EVAL-SUBSET FIX (2026-06, task #cls-cam17): every in-loop label-free
        # candidate is scored on self.timgs (per-image color/stain transforms inside forward()).
        # On a large OOD eval set that makes the LLM loop SLOW and risks a no-progress stall while
        # the agent builds the ~11-tool pool. So during EXPLORATION the adapter scores candidates
        # on a SMALL, class-stratified subset of the target (EXPLORE_EVAL_N, default 4000); the
        # FINAL shipped candidate is re-scored on the FULL eval set by the runner's audit (which
        # holds its own full timgs/tlabs). Keep the full set available for any caller that wants it.
        self._timgs_full, self._tlabs_full = timgs, tlabs
        ev_n = explore_eval_n if explore_eval_n is not None else \
            int(__import__("os").environ.get("EXPLORE_EVAL_N", "4000") or 0)
        timgs, tlabs = self._subsample_explore(timgs, tlabs, ev_n)
        self.timgs = timgs
        self.tlabs = tlabs
        self.nc, self.device = nc, device
        self.metric_name = metric.upper()
        self.metric_kind = metric
        self.task_desc = task_desc
        self._src_stats = None
        self.built = []
        self.llm = None                       # for recommend_alternative
        self.USABLE_BAR = 0.60                # AUROC below which a cls deploy is unusable
        self.labeled_k = 0
        self._labeled_dice = None             # {name: HELD-OUT metric} (key reused by core/policy)
        # K labels split TRAIN (finetune) / HOLDOUT (rank+safety) — same as SegTask.
        self._labels_revealed = False
        self._train = None
        self._hold = None

    def set_llm(self, llm):
        self.llm = llm

    @staticmethod
    def _subsample_explore(timgs, tlabs, n):
        """Class-STRATIFIED subsample of the target set to <= n images for FAST in-loop scoring.
        n<=0 or n>=len -> the full set unchanged. Stratified so AUROC stays defined (>=1 of each
        class) and the label-free quality signal is representative. Deterministic (linspace)."""
        if timgs is None or n is None or n <= 0 or len(timgs) <= n:
            return timgs, tlabs
        if tlabs is None:                       # no labels: plain even subsample
            idx = np.linspace(0, len(timgs) - 1, n, dtype=int)
            return [timgs[int(i)] for i in idx], None
        y = np.array([int(l) for l in tlabs])
        classes = sorted(set(y.tolist()))
        per = max(1, n // max(len(classes), 1))
        idx = []
        for c in classes:
            ci = np.where(y == c)[0]
            sel = ci[np.linspace(0, len(ci) - 1, min(per, len(ci)), dtype=int)]
            idx.extend(int(i) for i in sel)
        idx = sorted(set(idx))
        return [timgs[i] for i in idx], [tlabs[i] for i in idx]

    def toolbox(self):
        B1 = Budget(compute_minutes=1); Bt = Budget(compute_minutes=2, n_trials=1)
        # STRUCTURE-AWARE menu: HARD-EXCLUDE structurally-unsupported tools. bn_reset/tent act on
        # BatchNorm; a model with NONE (LayerNorm/transformer) makes them PROVABLE no-ops (Delta=0.0
        # measured) -> REMOVE them from the menu entirely rather than describe-and-hope-LLM-skips.
        # Soft analysis still drives the choice among APPLICABLE tools; only provably-inert tools are
        # hard-filtered. (mirrors the 3D adapter; the transformer-native LN levers stay in the pool.)
        import torch.nn as _nn
        _bn = sum(1 for _m in self.source_model.modules()
                  if isinstance(_m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)))
        if _bn > 0:
            _bn_tools = [
                Tool("bn_reset", "recalibrate BatchNorm running stats to target (no labels); this "
                     "model has %d BatchNorm layer(s) so it IS a live lever." % _bn, self._bn, Bt),
                Tool("tent", "entropy-minimization TTA on BN affine params (live: %d BN layer(s))."
                     % _bn, self._tent, Bt)]
        else:
            _bn_tools = []   # LayerNorm/transformer (no BatchNorm) -> BN-TTA structurally EXCLUDED
        return [
            Tool("load_and_deploy", "run the classifier on target data; report confidence",
                 self._deploy, B1),
            Tool("analyze_domain_shift", "MEASURE source-vs-target image stats + characterize the "
                 "shift (intensity / contrast / modality / resolution / FOV, severity)",
                 self._analyze_shift, Budget(compute_minutes=1, llm_usd=0.02)),
            Tool("inspect_model", "introspect the model (architecture, normalization, in/out, "
                 "likely training domain) — tells you which adaptations apply",
                 self._inspect_model, Budget(llm_usd=0.02)),
            Tool("diagnose_cls", "cls failure analysis: confidence drop, class-collapse, calibration",
                 self._diagnose, B1),
            Tool("plan", "REASON FIRST: returns the capability inventory (which repairs this model "
                 "admits, what each COSTS in labels, the budget, the data profile). Call once with "
                 "no args to see it, then AGAIN with steps=[...] and rationale='...' to commit to "
                 "an ordered plan chosen for THIS model/budget/objective. Revise by calling again "
                 "when an observation contradicts the plan.",
                 self._plan, Budget(llm_usd=0.02)),
            Tool("diagnose_failure_mode", "WHY it fails -> the MECHANISM, and therefore which repair "
                 "class applies: head_misaligned (features separate but the head misreads them -> "
                 "probe_head, cheap in labels) vs representation (features degraded -> input repair "
                 "then fine-tune) vs threshold_prior (ranking intact, operating point wrong -> "
                 "recalibrate) vs collapsed_output. Label-free (uses no labels); refines itself if "
                 "labels already exist. RUN THIS BEFORE CHOOSING A REPAIR.",
                 self._diagnose_failure_mode, B1),
            *_bn_tools,
            Tool("evaluate", "label-free confidence of the current best candidate", self._eval, B1),
        ] + self._labelfree_2d_tools() + [
            Tool("request_labels", "ask the clinician for labels (default = your FULL authorized budget; request fewer only deliberately); split TRAIN+HOLDOUT, then "
                 "rank ALL candidates by HELD-OUT AUROC",
                 self._request, lambda a: Budget(k_labels=int(a.get("K", int(os.environ.get("KLAB", "6") or 6))))),
            Tool("probe_head", "REPAIR A MISALIGNED HEAD: refit ONLY the final linear layer on "
                 "FROZEN features (logistic probe). Use when the representation still separates "
                 "the classes but the deployed head maps it wrongly -- far cheaper in labels than "
                 "fine-tuning (works from ~4-12) and cannot damage the backbone. NOT the fix for a "
                 "threshold/prior shift, nor for a genuinely broken representation. args: K=",
                 self._probe_head, Budget(compute_minutes=2)),
            Tool("finetune_head", "FINE-TUNE the classifier head (lowest capacity, safest at tiny K). "
                 "DESIGN THE RECIPE: optional args lr= and steps= override the per-budget default. "
                 "Read the returned loss/val/collapse_guard/overfit diagnostics and RE-CALL with a "
                 "revised recipe (lower lr / fewer steps / smaller capacity) if it overfit or collapsed.",
                 lambda st, **k: self._finetune("head", **k), Budget(compute_minutes=3)),
            Tool("finetune_early", "SURGICAL fine-tune: FIRST conv stage + head only (Lee et al.) -- "
                 "the right capacity for an INPUT/appearance-level shift (stain, brightness, camera), "
                 "where adapting early layers beats head-tuning. DESIGN the recipe (lr=, steps=).",
                 lambda st, **k: self._finetune("early", **k), Budget(compute_minutes=3)),
            Tool("finetune_partial", "fine-tune last block(s)+head (medium capacity). DESIGN the recipe "
                 "(args lr=, steps=); revise from the returned diagnostics.",
                 lambda st, **k: self._finetune("partial", **k), Budget(compute_minutes=4)),
            Tool("finetune_lora", "parameter-efficient LoRA adapters (low-param, collapse-resistant). "
                 "DESIGN the recipe (args lr=, steps=); revise from diagnostics.",
                 lambda st, **k: self._finetune("lora", **k), Budget(compute_minutes=4)),
            Tool("finetune_full", "fine-tune ALL weights (highest capacity; overfits at small K). DESIGN "
                 "the recipe (args lr=, steps=) — prefer a LOW lr at small K and revise DOWN if the "
                 "diagnostics show overfit/collapse.",
                 lambda st, **k: self._finetune("full", **k), Budget(compute_minutes=6)),
            Tool("recommend_alternative", "when adaptation can't reach usable AUROC: recommend "
                 "modality-appropriate classifiers + a fine-tuning label estimate",
                 self._recommend, Budget(llm_usd=0.05)),
        ] + self._scaling_tools() + self._bridged_tools()

    def _scaling_tools(self):
        # Component B — env-gated tool registration (SCALING_EVAL=0 default), design §3.6.
        import os
        if os.environ.get("SCALING_EVAL", "0") != "1":
            return []
        return [Tool("scaling_eval", "few-shot SCALING-LAW: k-fold CV fine-tune across label "
                     "budgets -> honest delivered-model estimate + an 'annotate N -> expect X' "
                     "plan. Prefer over the manual finetune ladder once labels are revealed.",
                     lambda st, **k: self._scaling_eval(**k), Budget(compute_minutes=8))]

    # -- the label-free arsenal from the registry (cls subset: preprocessing + frequency +
    #    remaining gradient TTA sar/eata/lora_tta), ctx-aware scored. --
    def _bridged_tools(self):
        from harness.registry_bridge import bridged_specs_cls
        try:
            specs, caps = bridged_specs_cls(self.source_model)
        except Exception:
            return []
        out = []
        for t in specs:
            if t.name in _BRIDGE_SKIP_CLS:
                continue
            prof = (t.profile or "").split(".")[0][:140]
            out.append(Tool(t.name, f"[{t.category}] {prof}",
                            (lambda st, _n=t.name, **k: self._apply_bridged(_n)),
                            Budget(compute_minutes=2, n_trials=1)))
        return out

    def _cls_task_obj(self):
        # Registry tools (sar/eata/...) call task.to_tensor(); the cls path must pass a real
        # ClassificationTask (mirrors seg's _seg_task_obj). Without it the bridge falls back to a
        # task STRING -> 'str' object has no attribute 'to_tensor' -> sar/eata silently empty on cls
        # (the transformer-native TTA) + agent step-thrashes on the broken tool.
        if getattr(self, "_clstask", None) is None:
            from medadaptagent.tasks import get_task
            self._clstask = get_task("classification")
        return self._clstask

    def _apply_bridged(self, name):
        from harness.registry_bridge import apply_registry_tool, CtxCandidate
        try:
            ctx = apply_registry_tool(name, self.source_model, self.timgs, self.nc, self.device,
                                      task=self._cls_task_obj())
        except Exception as e:
            return {"evidence": {"error": f"{name} failed: {type(e).__name__}: {str(e)[:120]}"}}
        cand = CtxCandidate(ctx, _cls_tensor)
        s = self._stats_of(cand); q = _quality(s)
        self.built.append({"name": name, "model": cand, "quality": q, "stats": s})
        if self._labels_revealed:
            self._rank_on_holdout()
        return {"candidate": cand, "evidence": {"cand_stats": s, "quality": q, "last_method": name,
                "tools_applied": list(ctx.tools_applied)}, "metric_estimate": q}

    # -- INPUT-SPACE color/intensity label-free tools (cls cross-camera/cross-source
    #    color & illumination shifts; bn_reset/tent only touch BN stats). The
    #    source-referenced tools build their reference from a handful of UNLABELED
    #    source-domain images (the model owner has these = self.simgs). Mirrors
    #    harness/labelfree_3d_tools.py but 2D (RGB, no 3D handling). --
    def _labelfree_2d_tools(self):
        try:
            from harness.labelfree_2d_tools import LABELFREE_2D_TOOL_NAMES
        except Exception:
            return []
        Bc = Budget(compute_minutes=2, n_trials=1)
        desc = {
            "histogram_match_src": "[input-space color] match each TARGET image's per-channel "
                "intensity CDF to the SOURCE domain (fixes color cast / contrast across cameras "
                "& sources) — no labels; source images only",
            "match_intensity_stats": "[input-space color] per-channel mean/std affine so target "
                "channel statistics match the SOURCE domain (linear color-cast / illumination fix) "
                "— no labels",
            "fda_src": "[input-space color] Fourier Domain Adaptation: swap the target's "
                "low-frequency amplitude for the SOURCE amplitude (appearance/style match, keep "
                "content) — no labels",
            "gamma_src": "[input-space color] auto-gamma so target brightness matches the SOURCE "
                "brightness (global illumination fix) — no labels",
            "zscore_to_src": "[input-space color] per-channel z-score then rescale to SOURCE "
                "channel mean/std (intensity normalization toward source) — no labels",
            "reinhard_stain": "[stain normalization] Reinhard LAB-space mean/std match of each "
                "TARGET patch toward the SOURCE domain — the canonical H&E STAIN-transfer color "
                "match for histopathology cross-scanner / cross-hospital stain shift — no labels",
            "macenko_stain": "[stain normalization] Macenko stain-deconvolution normalization "
                "(SVD of H&E optical-density vectors) toward a SOURCE reference patch — stronger "
                "histopathology stain transfer than Reinhard — no labels",
            "clahe": "[input-space contrast] CLAHE local contrast equalization on luminance "
                "(low-contrast / washed-out targets) — no labels, source-agnostic",
            "percentile_normalize": "[input-space norm] robust [1,99]-percentile clip+rescale per "
                "channel (saturated-pixel / exposure outliers) — no labels, source-agnostic",
            "norm_stat_tent": "[norm-layer TTA] entropy-minimization on the affine weight+bias of "
                "EVERY norm layer (LayerNorm / GroupNorm / InstanceNorm / BatchNorm) — the "
                "NORM-AGNOSTIC lever. Works on a transformer (ViT) where bn_reset/tent are no-ops, "
                "BUT naive entropy-min on all LayerNorm affines can COLLAPSE a ViT (predictions go "
                "to one class). Prefer norm_stat_tent_safe on transformers — no labels",
            "norm_stat_tent_safe": "[norm-layer TTA, collapse-safe] TRANSFORMER-NATIVE LayerNorm "
                "entropy-min that does NOT collapse the ViT: adapts only the LAST few LayerNorm "
                "affines with SAR-style reliable-sample selection, gentle lr/few steps, and a "
                "label-free collapse-guard that REVERTS to source if predictions would collapse. "
                "The safe weight-space lever for ViT/LayerNorm models — no labels",
            "t3a": "[classifier-space, training-free] T3A test-time template adjustment "
                "(Iwasawa 2021): collect ViT [CLS] penultimate features + pseudo-labels over the "
                "target, build per-class PROTOTYPES from the most-confident support samples (seeded "
                "from the source head), and classify by cosine-to-prototype. ARCH-AGNOSTIC, "
                "no gradient steps, CANNOT collapse — works on a transformer where bn_reset/tent "
                "are no-ops — no labels",
        }
        return [Tool(n, desc[n], (lambda st, _n=n, **k: self._apply_labelfree_2d(_n)), Bc)
                for n in LABELFREE_2D_TOOL_NAMES]

    def _source_ref_2d(self):
        """Lazily build the SOURCE reference (per-channel CDF / mean / std / amplitude
        / luminance mean) from a handful of UNLABELED source-domain images, in the
        model's de-normalized RGB [0,1] input space. Cached. None if no source imgs."""
        if getattr(self, "_src_ref_2d", "MISSING") != "MISSING":
            return self._src_ref_2d
        ref = None
        try:
            from harness.labelfree_2d_tools import build_source_ref_2d
            src = self.simgs if self.simgs is not None else self.timgs
            ref = build_source_ref_2d(_cls_tensor, src, self.device, n=24)
        except Exception:
            ref = None
        self._src_ref_2d = ref
        return ref

    def _apply_labelfree_2d(self, name):
        """Build an input-space color/intensity candidate and score it label-free
        (exactly like bn_reset/tent via _score) so it ranks apples-to-apples."""
        from harness.labelfree_2d_tools import (build_labelfree_2d_candidate, _NEEDS_REF_2D,
                                                 _NEEDS_SRC_IMGS_2D, _NEEDS_TARGET_IMGS_2D)
        ref = self._source_ref_2d() if name in _NEEDS_REF_2D else None
        if name in _NEEDS_REF_2D and ref is None:
            return {"evidence": {"error": f"{name} needs a source reference but no source "
                                          f"images are available"}}
        src = self.simgs if self.simgs is not None else self.timgs
        if name in _NEEDS_TARGET_IMGS_2D:
            # norm_stat_tent ADAPTS norm-layer affines TO the target via entropy-min,
            # so it must receive the TARGET images (passed through the source_imgs slot).
            kw_extra = dict(to_tensor=_cls_tensor, source_imgs=self.timgs, device=self.device)
        elif name in _NEEDS_SRC_IMGS_2D:
            kw_extra = dict(to_tensor=_cls_tensor, source_imgs=src, device=self.device)
        else:
            kw_extra = {}
        try:
            cand, info = build_labelfree_2d_candidate(name, self.source_model, ref, **kw_extra)
        except Exception as e:
            return {"evidence": {"error": f"{name} failed: {type(e).__name__}: {str(e)[:120]}"}}
        res = self._score(name, cand)
        res.setdefault("evidence", {}).update({"tool_info": info, "last_method": name})
        if self._labels_revealed:                 # if labels already out, rank it on holdout too
            self._rank_on_holdout()
        return res

    def _metric_of(self, cand, imgs, labs):
        from harness.registry_bridge import cls_metric_of
        return cls_metric_of(cand, imgs, labs, self.nc, self.device,
                             metric_kind=self.metric_kind, metric_fn=_metric)

    def _stats_of(self, cand):
        from harness.registry_bridge import cls_stats_of
        return cls_stats_of(cand, self.timgs, self.device, conf_stats=_conf_stats)

    # -- tools --------------------------------------------------------------
    def _deploy(self, st, **kw):
        s = _conf_stats(self.source_model, self.timgs, self.device)
        self._src_stats = s; q = _quality(s)
        self.built.append({"name": "source", "model": self.source_model, "quality": q, "stats": s})
        return {"deployment": "source@target", "candidate": self.source_model,
                "evidence": {"source_stats": s, "quality": q, "last_method": "source"},
                "metric_estimate": q}

    def _diagnose(self, st, **kw):
        cand = st.best_candidate or self.source_model
        s = _conf_stats(cand, self.timgs, self.device)
        return {"evidence": {"failure_mode": _diagnose_cls(s, self._src_stats or s), "stats": s}}

    def _analyze_shift(self, st, **kw):
        if getattr(self, "_shift_ev", None) is not None:
            return {"evidence": {"domain_shift": self._shift_ev,
                    "directive": "shift ALREADY analyzed — do NOT call again; ACT now."}}
        from harness.analysis import analyze_domain_shift
        self._shift_ev = analyze_domain_shift(self.simgs, self.timgs, self.llm, self.task_desc, model=self.source_model)
        return {"evidence": {"domain_shift": self._shift_ev, **self._retrieve_lessons(st)}}

    def _inspect_model(self, st, **kw):
        if getattr(self, "_model_ev", None) is not None:
            return {"evidence": {"model_info": self._model_ev,
                    "directive": "model ALREADY inspected — do NOT call again; ACT now."}}
        from harness.analysis import inspect_model
        self._model_ev = inspect_model(self.source_model, self.llm, self.task_desc)
        return {"evidence": {"model_info": self._model_ev, **self._retrieve_lessons(st)}}


    def _retrieve_lessons(self, st):
        """Component E (gated LESSONS): once BOTH diagnostics exist, surface >=moderate
        categorical lessons as ADVISORY evidence. {} when off/empty/single-diagnostic."""
        if (getattr(st, "evidence", None) or {}).get("retrieved_lessons"):
            return {}
        try:
            from harness.lessons import maybe_retrieve
        except ImportError:
            from lessons import maybe_retrieve
        return maybe_retrieve(getattr(self, "_shift_ev", None),
                              getattr(self, "_model_ev", None), "classification")
    def _score(self, name, cand):
        s = _conf_stats(cand, self.timgs, self.device); q = _quality(s)
        self.built.append({"name": name, "model": cand, "quality": q, "stats": s})
        return {"candidate": cand, "evidence": {"cand_stats": s, "quality": q, "last_method": name},
                "metric_estimate": q}

    def _bn(self, st, **kw):
        cand = copy.deepcopy(self.source_model); _bn_reset(cand, self.timgs, self.device)
        return self._score("bn_reset", cand)

    def _tent(self, st, **kw):
        cand = copy.deepcopy(self.source_model); _tent(cand, self.timgs, self.device)
        return self._score("tent", cand)

    def _eval(self, st, **kw):
        cand = st.best_candidate or self.source_model
        s = _conf_stats(cand, self.timgs, self.device)
        return {"evidence": {"eval_stats": s, "quality": _quality(s)}, "metric_estimate": _quality(s)}

    def _ensure_labels(self, K=6):
        """Reveal K labels ONCE, CLASS-BALANCED, split TRAIN/HOLDOUT per class so HOLDOUT
        carries ≥1 of each class (AUROC is defined). TRAIN feeds finetune_*, HOLDOUT
        ranks/safety-checks. Idempotent."""
        if self._labels_revealed:
            return self._train
        K = int(K)
        y = np.array([int(l) for l in self.tlabs])
        classes = sorted(set(y.tolist()))
        take = max(2, K // max(len(classes), 1))
        tr_idx, ho_idx = [], []
        # LABEL_DRAW_SEED (opt-in): the support set is normally chosen DETERMINISTICALLY
        # (linspace), which makes runs reproducible but hides the real operational question --
        # how sensitive is the decision to WHICH cases the clinician happens to annotate?
        # Setting LABEL_DRAW_SEED=<n> draws a different class-balanced random subset so decision
        # stability across plausible annotation draws can be measured. Unset => unchanged behavior.
        _draw = os.environ.get("LABEL_DRAW_SEED", "")
        _rng = np.random.RandomState(int(_draw)) if _draw != "" else None
        for c in classes:
            ci = np.where(y == c)[0]
            if _rng is not None:
                sel = _rng.choice(ci, size=min(take, len(ci)), replace=False)
                sel = np.sort(sel)
            else:
                sel = ci[np.linspace(0, len(ci) - 1, min(take, len(ci)), dtype=int)]
            # spread holdout (every 3rd) across the class, not the tail
            m = len(sel)
            hp = set(range(0, m, 3)) if m >= 2 else set()
            ho_idx += [sel[p] for p in range(m) if p in hp]
            tr_idx += [sel[p] for p in range(m) if p not in hp]
        self._train = ([self.timgs[i] for i in tr_idx], [self.tlabs[i] for i in tr_idx])
        self._hold = ([self.timgs[i] for i in ho_idx], [self.tlabs[i] for i in ho_idx])
        self.labeled_k += len(tr_idx) + len(ho_idx)
        self._labels_revealed = True
        return self._train

    def _rank_on_holdout(self):
        # label-free candidates + source: rank on ALL K (never trained on labels, reliable);
        # finetune_* candidates: rank on HOLDOUT only (trained on TRAIN, bias-free).
        # CV-GATE (SCALING_EVAL=1, cls-path): the tiny HOLDOUT AUROC saturates to a fake 1.0 at
        # K=6 (2-sample holdout). This is the SINGLE chokepoint every path funnels through, so
        # override EVERY finetune candidate's gate with a POOLED out-of-fold CV AUROC (one AUROC
        # over ALL K), and use source's pooled-OOF CV on the SAME folds. Inert when SCALING_EVAL!=1.
        import os
        cv_gate = os.environ.get("SCALING_EVAL", "0") == "1"
        h_imgs, h_labs = self._hold
        a_imgs = list(self._train[0]) + list(h_imgs)
        a_labs = list(self._train[1]) + list(h_labs)
        ld = {}
        cv_src = None
        for b in self.built:
            name = str(b["name"])
            if cv_gate and name.startswith("finetune"):
                mode = name.split("finetune_", 1)[-1] or "head"
                try:
                    cv_cand, cv_s = self._pooled_cv_gate(mode, len(a_imgs))
                    ld[name] = round(float(cv_cand), 4)
                    cv_src = cv_s if cv_src is None else cv_src
                    continue
                except Exception:
                    pass   # fall through to single-holdout if CV fails
            if name == "probe_head":
                # probe trains on ALL K -> holdout ranking is train-contaminated (reads a fake 1.0).
                # Rank by the UNBIASED CV-OOF AUROC over all K instead.
                oof = self._probe_oof_probs()
                if oof is not None:
                    _imgs, _lbls, _pp = oof
                    _y = np.asarray([int(v) for v in _lbls])
                    try:
                        from sklearn.metrics import roc_auc_score
                        ld[name] = round(float(roc_auc_score(_y, _pp[:, 1]) if self.nc == 2
                                               else roc_auc_score(_y, _pp, multi_class="ovr")), 4)
                        continue
                    except Exception:
                        pass
            ii, ll = (h_imgs, h_labs) if (name.startswith("finetune") or name.startswith("wise_") or name == "probe_head") else (a_imgs, a_labs)
            ld[name] = round(float(self._metric_of(b["model"], ii, ll)), 4)
        if cv_gate and cv_src is not None:
            ld["source"] = round(float(cv_src), 4)   # gate source on the SAME pooled-OOF folds
        # SELECTION-REGIME AUROC-SATURATION FIX (2026-06, task #cls-cam17): at the tiny K-label
        # SELECTION budget (NO_FINETUNE, no CV gate), AUROC over K saturates to 1.0 for source AND
        # the strong stain candidates (6 labels cannot separate AUROC 0.90 from 0.98) — so the
        # margin gate ties everything at the ceiling and the agent ships SOURCE, MISSING the proven
        # +0.08 stain recovery. ACCURACY over the same K labels is per-sample and DOES separate them
        # (Camelyon17 source acc≈0.83 vs stain acc≈0.95, a +0.12 gap that clears the 0.05 margin).
        # So when AUROC saturates (source AUROC>=0.999) substitute K-label ACCURACY as the gate for
        # the LABEL-FREE candidates (which never train on labels, so all-K accuracy is unbiased).
        # Scoped to NO_FINETUNE + auroc + saturation; the CV-gate finetune path (SCALING_EVAL=1) is
        # untouched. Inert whenever AUROC is already discriminative.
        if (os.environ.get("NO_FINETUNE", "0") != "0" and not cv_gate
                and self.metric_kind == "auroc" and float(ld.get("source", 0.0)) >= 0.999):
            acc = {}
            for b in self.built:
                name = str(b["name"])
                if name.startswith("finetune"):
                    continue                       # finetune ranks on its own holdout, leave it
                a = float(_metric(b["model"], a_imgs, a_labs, self.device, self.nc, "acc"))
                acc[name] = round(a, 4)
            # only override if accuracy actually separates (else keep AUROC; no spurious flip)
            if acc and (max(acc.values()) - acc.get("source", 0.0)) > 1e-6:
                ld.update(acc)                     # gate on the separable accuracy signal
                self._gate_metric_note = "accuracy@K (AUROC saturated at this label budget)"
        self._labeled_dice = ld
        return ld

    def _estimate_performance(self):
        """Task-grounded expected/usable metric from the literature (via the LLM), not a
        hardcoded bar. e.g. referable-DR fundus AUROC ~0.95+, many cls tasks ~0.85-0.95."""
        if getattr(self, "_perf_est", None) is not None:
            return self._perf_est
        fallback = {"sota": 0.90, "usable": 0.75, "rationale": "generic cls default (no LLM)"}
        if self.llm is None:
            self._perf_est = fallback; return fallback
        import json, re
        prompt = (f"Medical-image CLASSIFICATION task: {self.task_desc}. Metric: {self.metric_name}.\n"
                  f"From the published literature for THIS task, give the {self.metric_name} a "
                  f"COMPETENT / near-SOTA model typically reaches, and the MINIMUM clinically-usable "
                  f"value. Respond ONLY JSON: {{\"sota\": <0..1>, \"usable\": <0..1>, "
                  f"\"rationale\": \"<=15 words citing the typical benchmark\"}}.")
        try:
            d = json.loads(re.search(r"\{.*\}", self.llm.chat(prompt), re.S).group(0))
            d = {"sota": float(d.get("sota", 0.90)), "usable": float(d.get("usable", 0.75)),
                 "rationale": str(d.get("rationale", ""))[:160]}
        except Exception:
            d = fallback
        self._perf_est = d
        return d

    def _request(self, st, K=None, **kw):
        # DEFAULT = the FULL authorized budget. The budget is the clinician's consent, not a
        # target to undercut: requesting 6 of 12 authorized labels starves the probe/finetune
        # AND the holdout (2 samples -> every candidate saturates to 1.0, the paired verifier
        # cannot run), observed to turn a +0.20 probe recovery into keep_source. Fewer than the
        # budget is allowed only as an explicit choice.
        if K is None:
            K = int(os.environ.get("KLAB", "6") or 6)
        if self.tlabs is None:
            return {"k_labels_used": 0, "evidence": {"error": "no labels available"}}
        self._ensure_labels(K)
        # STRUCTURAL: the cheapest labeled repair (refit the final linear layer on frozen
        # features) is built AUTOMATICALLY the moment labels are revealed -- measured to fix
        # head-misalignment at K=4-12 where fine-tuning overfits. It reuses these same labels,
        # costs seconds, and _rank_on_holdout + the ship gate then judge it like any candidate.
        try:
            pr = self._probe_head(st, K=K)
            _perr = (pr.get("evidence", {}) or {}).get("error")
            if _perr: print("[auto-probe] NOT BUILT:", _perr, flush=True)
            # register for ship-by-name: core only registers candidates carried by the CURRENT
            # tool result (request_labels'), so the auto-built probe must self-register --
            # otherwise ship{candidate: probe_head} SILENTLY falls back to best_candidate.
            if _perr is None and pr.get("candidate") is not None and st is not None:
                try:
                    st.named_candidates["probe_head"] = pr["candidate"]
                    st.candidates_log.append({"method": "probe_head",
                                              "metric_estimate": pr.get("metric_estimate")})
                except Exception:
                    print("[auto-probe] state registration failed", flush=True)
        except Exception as _e:
            import traceback; traceback.print_exc()
            print("[auto-probe] EXCEPTION:", repr(_e)[:200], flush=True)
        ld = self._rank_on_holdout()
        best = max(ld, key=ld.get) if ld else None
        est = self._estimate_performance()        # task-grounded, not hardcoded
        gate_note = getattr(self, "_gate_metric_note", None)   # set if AUROC saturated -> acc@K gate
        metric_lbl = "accuracy@K" if gate_note else self.metric_name
        return {"k_labels_used": self.labeled_k,
                "evidence": {"labeled_dice": ld, "best_by_labels": best, "best_verified": bool(getattr(self, "_best_verified", False)),
                             "n_train": len(self._train[0]), "n_holdout": len(self._hold[0]),
                             "metric": metric_lbl, "gate_metric": gate_note,
                             "expected_metric": est["sota"], "usable_floor": est["usable"],
                             "safety_margin": self.safety_margin(ld.get("source", 0.0)),
                             "perf_rationale": est["rationale"],
                             "note": f"HELD-OUT ranking on {metric_lbl} ({self.labeled_k} labels)"
                                     + (f" [{gate_note}]" if gate_note else "")
                                     + f". Ship '{best}' if it beats source (margin "
                                     f"{self.safety_margin(ld.get('source', 0.0)):.3f}); a verified "
                                     f"gain over source is a real, safe improvement to ship."}}

    def _finetune(self, mode, K=6, base="source", **kw):
        """Few-shot FINE-TUNE on TRAIN labels, then re-rank on HOLDOUT. The LLM DESIGNS the recipe:
        optional kw lr= and steps= override the per-budget default, so the agent can react to the
        diagnostics returned here (loss trajectory, val src->best, collapse_guard, overfit) and
        RE-CALL with a revised recipe. Deterministic safety floor stays in finetune() (best-ckpt +
        never-worse-than-source guard)."""
        from harness.finetune import finetune
        if self.tlabs is None:
            return {"evidence": {"error": "no labels available — cannot finetune"}}
        tr_imgs, tr_labs = self._ensure_labels(K)
        if len(tr_imgs) == 0:
            return {"evidence": {"error": "no train labels after split (K too small)"}}
        from harness.registry_bridge import CtxCandidate
        start = self.source_model
        if base == "best":
            models = [b for b in self.built if not isinstance(b["model"], CtxCandidate)]
            if models:
                start = max(models, key=lambda b: b.get("quality", 0))["model"]
        lr = kw.get("lr"); steps = kw.get("steps")
        try: lr = float(lr) if lr is not None else None
        except Exception: lr = None
        try: steps = int(steps) if steps is not None else None
        except Exception: steps = None
        model, info = finetune(start, tr_imgs, tr_labs, task="cls", nc=self.nc, device=self.device,
                               to_tensor=_cls_tensor, mode=mode, lr=lr, steps=steps)
        name = f"finetune_{mode}"
        self.built = [b for b in self.built if b.get("name") != name]   # clean-replace on re-call
        if self._labeled_dice: self._labeled_dice.pop(name, None)
        res = self._score(name, model)
        # WiSE-FT (Wortsman et al. 2022): the 50/50 weight interpolation of source and fine-tuned
        # models is routinely BETTER out-of-distribution than the fine-tuned endpoint and softens
        # any collapse. Build it as a sibling candidate; it is label-trained (contains fine-tuned
        # weights) so it ranks/verifies holdout-only like its parent.
        try:
            import copy as _cp, torch as _t
            wname = f"wise_{mode}"
            wm = _cp.deepcopy(self.source_model).to(self.device)
            sd_s = self.source_model.state_dict(); sd_f = model.state_dict()
            wm.load_state_dict({k: (0.5*sd_s[k].float() + 0.5*sd_f[k].float()).to(sd_s[k].dtype)
                                if sd_s[k].dtype.is_floating_point else sd_f[k]
                                for k in sd_s})
            wm.eval()
            self.built = [b for b in self.built if b.get("name") != wname]
            wst = self._stats_of(wm)
            self.built.append({"name": wname, "model": wm, "quality": _quality(wst), "stats": wst})
        except Exception as _e:
            print("[wise] blend failed:", repr(_e)[:100], flush=True)
        import os
        self._rank_on_holdout()
        hd = (self._labeled_dice or {}).get(name)
        gate_note = (" [pooled-OOF CV gate]" if os.environ.get("SCALING_EVAL", "0") == "1" else "")
        lf, ll = info.get("loss_first"), info.get("loss_last")
        sv, bv = info.get("src_val_loss"), info.get("best_val_loss")
        diag = []
        if info.get("collapse_guard"):
            diag.append("COLLAPSE-GUARD fired (no val gain -> returned source; try lower lr / smaller capacity)")
        if (lf is not None and ll is not None and lf and ll < 0.2 * lf
                and sv is not None and bv is not None and bv >= sv - 1e-6):
            diag.append("OVERFIT (train loss crashed but val did not improve -> lower lr / fewer steps)")
        if info.get("used_val") is False:
            diag.append("small-K: trained on all labels, no internal val (more labels would enable best-ckpt selection)")
        res.setdefault("evidence", {})
        best_by = self._best_by_labels_verified() if self._labeled_dice else None
        res["evidence"].update({"finetune": info, "labeled_dice": self._labeled_dice,
                                "best_by_labels": best_by, "best_verified": bool(getattr(self, "_best_verified", False)), "usable_bar": self.USABLE_BAR,
                                f"holdout_{self.metric_kind}": hd, "trained_on": len(tr_imgs),
                                "recipe_used": info.get("recipe"), "ft_diagnostics": diag,
                                "note": (f"{name}: labeled {self.metric_name} {hd}{gate_note} | "
                                         f"recipe {info.get('recipe')} | loss {lf}->{ll}, "
                                         f"val {sv}->{bv} (pick@{info.get('picked_step')}) | "
                                         f"{'; '.join(diag) if diag else 'clean fit'} | "
                                         f"to REVISE re-call a finetune_* with args lr=/steps=")})
        return res

    # -- pooled out-of-fold CV gate (shared by _finetune + _scaling_eval) ----
    def _pooled_score_fn(self, m, ev_imgs):
        """Per-sample held-out predictions for pooling. Binary AUROC -> positive-class prob;
        multiclass/accuracy -> argmax."""
        p = _predict_probs(m, ev_imgs, self.device)
        if self.metric_kind == "auroc" and self.nc == 2:
            return p[:, 1]
        return p.argmax(1).astype(float)

    def _pooled_metric_fn(self, scores, labels):
        """ONE metric over the POOLED out-of-fold predictions (all K held-out samples)."""
        y = np.asarray([int(round(v)) for v in labels])
        if self.metric_kind == "auroc" and self.nc == 2:
            try:
                from sklearn.metrics import roc_auc_score
                if len(set(y.tolist())) < 2:          # one class pooled -> AUROC undefined
                    return 0.5
                return float(roc_auc_score(y, np.asarray(scores, float)))
            except Exception:
                pass
        return float((np.asarray(scores, float).round().astype(int) == y).mean())

    def _pooled_cv_gate(self, mode, K):
        """Compute (candidate_pooled_cv, source_pooled_cv) AUROC for `mode` over the full K
        labeled pool via run_scaling_eval's pooled-OOF GATE. Single mode -> the GATE value
        IS this mode's pooled CV (run_scaling_eval picks best of `modes`, here just one)."""
        from harness.scaling import run_scaling_eval
        from harness.finetune import finetune
        import torch
        self._ensure_labels(K)
        imgs = list(self._train[0]) + list(self._hold[0])
        lbls = list(self._train[1]) + list(self._hold[1])

        def fit_metric(m, ei, el):
            return float(_metric(m, ei, el, self.device, self.nc, self.metric_kind))

        def finetune_fn(_mode, ti, tl):
            torch.manual_seed(0); np.random.seed(0)
            m, _ = finetune(self.source_model, ti, tl, task="cls", nc=self.nc,
                            device=self.device, to_tensor=_cls_tensor, mode=_mode)
            return m

        rep = run_scaling_eval(source_model=self.source_model, imgs=imgs, lbls=lbls,
                               fit_metric=fit_metric, finetune_fn=finetune_fn,
                               safety_margin_fn=self.safety_margin, modes=(mode,),
                               metric_name=self.metric_name, seed=0,
                               pooled_score_fn=self._pooled_score_fn,
                               pooled_metric_fn=self._pooled_metric_fn)
        return rep.shipped_eval_cv_mean, rep.source_metric

    def _scaling_eval(self, K=12, modes=("head", "partial"), **kw):
        """Component B (cls): k-fold CV fine-tune across label budgets; GATE = full-K POOLED
        out-of-fold AUROC (one AUROC over ALL K held-out preds, NOT a 2-sample per-fold
        holdout that reads a fake 1.0). Mirrors the seg CV-gate (scaling.py); SCALING_EVAL-gated."""
        from harness.scaling import run_scaling_eval
        from harness.finetune import finetune
        import torch
        if self.tlabs is None:
            return {"evidence": {"error": "no labels available — cannot scale"}}
        self._ensure_labels(K)
        imgs = list(self._train[0]) + list(self._hold[0])
        lbls = list(self._train[1]) + list(self._hold[1])

        def fit_metric(m, ei, el):
            return float(_metric(m, ei, el, self.device, self.nc, self.metric_kind))

        def finetune_fn(mode, ti, tl):
            torch.manual_seed(0); np.random.seed(0)
            m, _ = finetune(self.source_model, ti, tl, task="cls", nc=self.nc,
                            device=self.device, to_tensor=_cls_tensor, mode=mode)
            return m

        est = self._estimate_performance()
        # POOLED out-of-fold GATE (shared fns): one AUROC over all K held-out preds, not a
        # 2-sample per-fold holdout (which reads a fake 1.0). This is the cls verification fix.
        rep = run_scaling_eval(source_model=self.source_model, imgs=imgs, lbls=lbls,
                               fit_metric=fit_metric, finetune_fn=finetune_fn,
                               safety_margin_fn=self.safety_margin, modes=tuple(modes),
                               metric_name=self.metric_name, usable_floor=est.get("usable"), seed=0,
                               pooled_score_fn=self._pooled_score_fn,
                               pooled_metric_fn=self._pooled_metric_fn)
        ship_model = finetune_fn(rep.best_mode, imgs, lbls)
        res = self._score(rep.shipped_candidate, ship_model)
        self._labeled_dice = {"source": round(rep.source_metric, 4),
                              rep.shipped_candidate: round(rep.shipped_eval_cv_mean, 4)}
        best_by = self._best_by_labels_verified()
        res.setdefault("evidence", {})
        res["evidence"].update({"labeled_dice": self._labeled_dice, "best_by_labels": best_by, "best_verified": bool(getattr(self, "_best_verified", False)),
                                "scaling_law": rep.to_evidence(), "scaling_plan": rep.plan,
                                "safety_margin": rep.margin, "expected_metric": est["sota"],
                                "usable_floor": est["usable"], "perf_rationale": est["rationale"],
                                "metric": self.metric_name, "note": rep.plan})
        return res

    def _recommend(self, st, **kw):
        """Grounded smart escalation: search the HuggingFace Hub for REAL modality-appropriate
        classifiers (reading their model + training-dataset cards), and estimate the labels to
        fine-tune the existing model to usable AUROC — not a hallucinated repo id."""
        from harness.model_search import recommend_alternative_models
        best = max((self._labeled_dice or {}).values(), default=None)
        rec = recommend_alternative_models(
            self.llm, task_desc=self.task_desc, task_type="classification",
            metric_name=self.metric_name, best_metric=best, usable_target=0.75,
            domain_shift=st.evidence.get("domain_shift"),
            model_info=st.evidence.get("model_info"))
        return {"evidence": {"recommendation": rec}}

    # -- CAPABILITY INVENTORY + PLANNING -------------------------------------
    def _capability_inventory(self):
        """WHAT IS AVAILABLE — the structural facts a plan must be reasoned from. All of this is
        knowable with ZERO labels, which is what makes planning tractable where label-free
        mechanism inference was not: we cannot know whether the class signal is still decodable,
        but we CAN know which repairs are applicable, what each costs, and what budget exists."""
        import torch
        m = self.source_model
        lin = [x for x in m.modules() if isinstance(x, torch.nn.Linear)]
        bn = sum(1 for x in m.modules()
                 if isinstance(x, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)))
        ln = sum(1 for x in m.modules() if isinstance(x, torch.nn.LayerNorm))
        y = np.asarray([int(v) for v in self.tlabs]) if self.tlabs is not None else None
        bal = None
        if y is not None and len(y):
            _, cnt = np.unique(y, return_counts=True)
            bal = round(float(cnt.min() / cnt.sum()), 4)
        k_left = int((self.budget_remaining().get("k_labels", 0)) if hasattr(self, "budget_remaining") else 0)
        inv = {
            "model": {"has_linear_head": bool(lin), "feature_dim":
                          int(lin[-1].in_features) if lin else None,
                      "batchnorm_layers": bn, "layernorm_layers": ln,
                      "bn_tta_applicable": bn > 0,
                      "head_repair_applicable": bool(lin)},
            "data": {"n_target": len(self.timgs), "n_classes": self.nc,
                     "minority_fraction": bal},
            "budget": {"labels_available": k_left, "labels_revealed": bool(self._labels_revealed)},
            "repairs": [
                {"name": "input-space repair (histogram/stain/gamma/fda)", "label_cost": 0,
                 "applicable": True,
                 "fixes": "appearance shift; can indirectly re-align a head the shift displaced"},
                {"name": "bn_reset / tent", "label_cost": 0, "applicable": bn > 0,
                 "fixes": "normalization-statistic shift; STRUCTURALLY INERT without BatchNorm"},
                {"name": "probe_head", "label_cost": "reuses the SAME labels already requested (no extra)",
                 "applicable": bool(lin),
                 "fixes": "a head that misreads intact features; cheap (one linear layer), works from ~4-12 labels"},
                {"name": "finetune_{head,partial,lora}", "label_cost": "the requested K",
                 "applicable": True,
                 "fixes": "genuinely degraded representation; most capacity, most overfit risk at small K"},
            ],
        }
        self._inventory = inv
        return inv

    def _plan(self, st, **kw):
        """Produce/record an explicit PLAN before acting.

        The point of this step is that the harness commits to a *reasoned* course of action derived
        from (a) what is available -- which repairs this architecture admits and what each costs --
        and (b) what the user asked for, rather than sweeping every tool and taking the maximum.
        The plan is recorded in the trace so the delivered decision can be explained and audited,
        and it is revised when observations contradict it."""
        inv = self._capability_inventory()
        steps = kw.get("steps"); rationale = kw.get("rationale", "")
        if steps is None:
            return {"evidence": {"inventory": inv,
                                 "directive": "Now call plan again with args steps=[...] and "
                                              "rationale='...' committing to an ORDERED plan that "
                                              "respects the label budget and the objective."}}
        if isinstance(steps, str):
            steps = [x.strip() for x in steps.split(",") if x.strip()]
        steps = [str(x) for x in steps]
        # STRUCTURAL VALIDATION: a plan referencing tools that do not exist would silently no-op
        # at execution (observed: 'input_space_repair_stain_or_histogram'). Reject unknown names
        # at commit time and show the valid vocabulary, so the plan is executable by construction.
        valid = {t.name for t in self.toolbox()} | {"ship", "keep_source", "escalate",
                                                    "request_labels", "load_and_deploy"}
        unknown = [x for x in steps if x not in valid]
        steps = [x for x in steps if x in valid]
        # AUTO-DROP unknown step names rather than rejecting the plan: a full rejection left the
        # model with repairs still locked and nothing valid to do (observed: forced escalate).
        # The executable subset commits; the dropped names are reported so the model can add the
        # real tool via a plan revision if it meant one (e.g. 'input_repair_stain_normalize' ->
        # histogram_match_src / macenko_stain).
        if not steps:
            return {"evidence": {"plan_invalid": {"unknown_steps": unknown,
                                                  "valid_tools": sorted(valid)},
                                 "inventory": inv}}
        self._planned = {"steps": list(steps), "rationale": str(rationale),
                         "dropped_steps": unknown,
                         "objective": getattr(self, "objective", "maximize target performance "
                                                                 "without shipping a worse model")}
        return {"evidence": {"plan": self._planned, "inventory": inv,
                             "note": "follow this plan; call plan again to REVISE it if an "
                                     "observation contradicts it (say what changed and why)"}}

    # -- MECHANISM DIAGNOSIS -------------------------------------------------
    def _diagnose_failure_mode(self, st, **kw):
        """WHY is this model failing here? Returns a MECHANISM, not just 'accuracy dropped'.

        Different failures need different repairs, and applying the wrong class wastes labels and
        can make things worse. The key distinction this makes is between a broken REPRESENTATION
        (features no longer separate the classes -> only fine-tuning helps) and a misaligned HEAD
        (features still separate them, but the final layer maps them wrongly -> a linear probe
        fixes it from a handful of labels, and TTA structurally cannot).

        LABEL-FREE evidence (always available):
          * feature_separability : silhouette of k-means (k=nc) on frozen penultimate features --
            'is the class structure still present in the representation?'
          * head_cluster_agreement : adjusted Rand index between those clusters and the head's
            predicted labels -- 'is the head actually reading that structure?'
            HIGH separability + LOW agreement  ==>  HEAD MISALIGNED (probe it)
          * pred_entropy / max_class_frac : degenerate or collapsed output
        LABELED evidence (only if labels are already revealed -- never requests them):
          * ranking metric vs accuracy at the default operating point -> threshold/prior shift
        """
        import numpy as _np
        ev = {}
        imgs = self._subsample(min(256, len(self.timgs)))[0] if hasattr(self, "_subsample") else self.timgs[:256]
        F = self._feats_of(self.source_model, imgs)
        P = None
        try:
            P = self._probs_on(self.source_model, imgs)
        except Exception:
            pass
        sep = agree = None
        if F is not None and len(F) >= max(8, 2 * self.nc) and P is not None:
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score, adjusted_rand_score
                Fz = (F - F.mean(0)) / (F.std(0) + 1e-6)
                km = KMeans(n_clusters=self.nc, n_init=10, random_state=0).fit(Fz)
                if len(set(km.labels_.tolist())) > 1:
                    sep = float(silhouette_score(Fz, km.labels_))
                agree = float(adjusted_rand_score(km.labels_, P.argmax(1)))
            except Exception:
                pass
        stt = self._src_stats or self._stats_of(self.source_model)
        ent = float(stt.get("mean_entropy", _np.nan)) if isinstance(stt, dict) else float("nan")
        coll = float(stt.get("max_class_frac", _np.nan)) if isinstance(stt, dict) else float("nan")
        ev.update(feature_separability=None if sep is None else round(sep, 4),
                  head_cluster_agreement=None if agree is None else round(agree, 4),
                  pred_entropy=None if ent != ent else round(ent, 4),
                  max_class_frac=None if coll != coll else round(coll, 4))
        # labeled refinement (uses ONLY labels already revealed)
        thresh_gap = None
        if self._labels_revealed and self._hold is not None:
            try:
                ii = list(self._train[0]) + list(self._hold[0])
                ll = _np.asarray([int(v) for v in (list(self._train[1]) + list(self._hold[1]))])
                pp = self._probs_on(self.source_model, ii)
                acc = float((pp.argmax(1) == ll).mean())
                if self.nc == 2 and len(set(ll.tolist())) > 1:
                    ths = _np.unique(pp[:, 1])
                    best = max(float(((pp[:, 1] >= t).astype(int) == ll).mean()) for t in ths)
                    thresh_gap = round(best - acc, 4)
                    ev.update(acc_default=round(acc, 4), acc_best_threshold=round(best, 4),
                              threshold_gap=thresh_gap)
            except Exception:
                pass
        # ---- adjudicate ----
        mech, why, repair = "undetermined", "insufficient signal", "diagnose further"
        if coll == coll and coll is not None and coll > 0.97:
            mech, why, repair = ("collapsed_output",
                "the head assigns nearly every case to one class",
                "input-space repair at K=0; a linear probe (probe_head) once a few labels exist")
        elif thresh_gap is not None and thresh_gap > 0.08:
            mech, why, repair = ("threshold_prior",
                f"accuracy rises {thresh_gap:+.3f} by moving the decision threshold alone -- the "
                "ranking is intact, the operating point is wrong",
                "recalibrate the threshold (~2 labels). A probe or fine-tune is the WRONG repair here")
        elif sep is not None and agree is not None and sep >= 0.15 and agree < 0.30:
            mech, why, repair = ("head_misaligned",
                f"frozen features still form separated clusters (silhouette {sep:.2f}) but the "
                f"head's predictions barely agree with that structure (ARI {agree:.2f}) -- the "
                "representation survived the shift, the final layer no longer reads it",
                "probe_head: refit ONLY the last linear layer on frozen features (works from ~4-12 "
                "labels). BN/entropy TTA cannot fix a head")
        elif sep is not None and sep < 0.05:
            mech, why, repair = ("representation",
                f"frozen features no longer form separated clusters (silhouette {sep:.2f}) -- the "
                "representation itself degraded",
                "input-space repair first (free); if that fails, fine-tune -- a probe cannot help")
        elif sep is not None and agree is not None and agree >= 0.30:
            mech, why, repair = ("head_consistent",
                f"the head agrees with the feature structure (ARI {agree:.2f}); no head fault detected",
                "if performance is still poor the shift is likely appearance-driven: input-space repair")
        ev.update(mechanism=mech, why=why, recommended_repair=repair)
        self._failure_mode = ev
        return {"evidence": {"failure_mode": ev}}

    # -- HEAD REPAIR: linear probe on frozen features ------------------------
    def _feats_of(self, model, imgs, bs=32):
        """Penultimate features (input to the final Linear) — the representation the head reads."""
        import torch
        F=[]; h={}
        lin=[m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        if not lin:
            return None
        hd = lin[-1].register_forward_hook(lambda m,i,o: h.__setitem__("f", i[0].detach()))
        model.eval()
        try:
            with torch.no_grad():
                for i in range(0, len(imgs), bs):
                    x = torch.cat([_cls_tensor(im, self.device) for im in imgs[i:i+bs]], 0)
                    model(x)
                    if "f" in h:
                        F.append(h["f"].float().cpu().numpy().reshape(len(x), -1))
        finally:
            hd.remove()
        return np.concatenate(F) if F else None

    def _probe_oof_probs(self, n_folds=4):
        """Out-of-fold probe predictions over ALL K labels: fit the logistic probe on K-1 folds,
        predict the held-out fold, rotate. Every label yields an UNBIASED prediction from a model
        that never saw it. Returns (imgs, lbls, probs) or None."""
        if self._hold is None:
            return None
        imgs = list(self._train[0]) + list(self._hold[0])
        lbls = list(self._train[1]) + list(self._hold[1])
        y = np.asarray([int(v) for v in lbls])
        if len(y) < 6 or len(set(y.tolist())) < 2:
            return None
        F = self._feats_of(self.source_model, imgs)
        if F is None:
            return None
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        probs = np.zeros((len(y), self.nc), dtype=np.float64)
        try:
            skf = StratifiedKFold(n_splits=min(n_folds, int(np.bincount(y).min())), shuffle=True,
                                  random_state=0)
            for tr, te in skf.split(F, y):
                if len(set(y[tr].tolist())) < 2:
                    return None
                lr = LogisticRegression(max_iter=3000).fit(F[tr], y[tr])
                pp = lr.predict_proba(F[te])
                for j, c in enumerate(lr.classes_):
                    probs[te, int(c)] = pp[:, j]
        except Exception:
            return None
        return imgs, lbls, probs

    def _probe_head(self, st, **kw):
        """REPAIR for a MISALIGNED HEAD: refit ONLY the final linear layer on frozen features.

        Measured motivation: on this suite the dominant failure is not a broken representation but
        a head that no longer maps it correctly (a logistic probe on frozen features reaches
        0.89-0.99 where the deployed head reads 0.70-0.80). Because it fits a handful of parameters
        on top of an unchanged backbone, it needs FAR fewer labels than fine-tuning -- it already
        beats the source at K=4-12 -- and it cannot damage the representation. Wrong repair for a
        THRESHOLD/PRIOR failure (recalibration) or a genuine REPRESENTATION shift (fine-tune)."""
        import torch, copy
        K = int(kw.get("K", 12))
        self._ensure_labels(K)
        tr_i, tr_l = self._train
        ho_i, ho_l = self._hold
        imgs = list(tr_i) + list(ho_i); lbls = list(tr_l) + list(ho_l)
        y = np.asarray([int(v) for v in lbls])
        if len(set(y.tolist())) < 2:
            return {"evidence": {"error": "probe_head needs >=2 classes in the labeled set"}}
        F = self._feats_of(self.source_model, imgs)
        if F is None:
            return {"evidence": {"error": "no Linear head found; probe_head not applicable"}}
        from sklearn.linear_model import LogisticRegression
        try:
            lr = LogisticRegression(max_iter=3000, C=float(kw.get("C", 1.0))).fit(F, y)
        except Exception as e:
            return {"evidence": {"error": f"probe fit failed: {type(e).__name__}"}}
        m = copy.deepcopy(self.source_model).to(self.device)
        lin = [mm for mm in m.modules() if isinstance(mm, torch.nn.Linear)]
        head = lin[-1]
        W = np.zeros((self.nc, F.shape[1]), dtype=np.float32)
        b = np.zeros((self.nc,), dtype=np.float32)
        classes = list(lr.classes_)
        if len(classes) == 2:                      # sklearn stores one row for binary
            W[classes[1]] = lr.coef_[0];  b[classes[1]] = lr.intercept_[0]
            W[classes[0]] = -lr.coef_[0]; b[classes[0]] = -lr.intercept_[0]
        else:
            for i, c in enumerate(classes):
                W[int(c)] = lr.coef_[i]; b[int(c)] = lr.intercept_[i]
        with torch.no_grad():
            head.weight.copy_(torch.tensor(W, device=self.device))
            if head.bias is not None:
                head.bias.copy_(torch.tensor(b, device=self.device))
        m.eval()
        self.built = [x for x in self.built if x.get("name") != "probe_head"]
        stt = self._stats_of(m); q = _quality(stt)
        self.built.append({"name": "probe_head", "model": m, "quality": q, "stats": stt})
        if self._labels_revealed:
            self._rank_on_holdout()
        hd = (self._labeled_dice or {}).get("probe_head")
        return {"candidate": m, "metric_estimate": q,
                "evidence": {"last_method": "probe_head", "quality": q,
                             "probe_head": {"n_labels": int(len(y)), "feat_dim": int(F.shape[1]),
                                            "train_acc": float(lr.score(F, y))},
                             f"holdout_{self.metric_kind}": hd,
                             "labeled_dice": self._labeled_dice,
                             "note": "refit ONLY the final linear layer on frozen features"}}

    # -- paired-bootstrap verifier (VERIFIER=paired) -------------------------
    def _probs_on(self, model, imgs):
        """Per-sample positive-class (or full) probabilities — computed ONCE so the bootstrap
        resamples arrays, not forward passes."""
        import torch
        out = []
        model.eval()
        with torch.no_grad():
            for im in imgs:
                lg = model(_cls_tensor(im, self.device))
                lg = lg.logits if hasattr(lg, "logits") else lg
                out.append(torch.softmax(lg.float(), dim=1)[0].cpu().numpy())
        return np.asarray(out)

    def _best_by_labels_verified(self):
        """The candidate the directive should RECOMMEND: best labeled score among candidates that
        ALSO pass their verification gate. Recommending the raw labeled-max caused a dead end --
        the top pick (often a labeled-1.0 tie) gets REFUSED by the paired gate at ship time and
        the model concludes keep_source, even though the next-best candidate passes (observed:
        breakhis tent p10 +0.145 never recommended because probe_head sat first in the tie).
        Falls back to the raw max when the paired verifier is off."""
        ld = self._labeled_dice or {}
        cands = [n for n in ld if n != "source"]
        if not cands:
            return None
        # TIE-BREAK BY PAIRED MEAN (ledger 13): labeled scores saturate/tie at tiny K (many 1.0s)
        # and build-order tie-breaking shipped a +0.115 candidate over a +0.197 one. When scores
        # tie, the paired-gate MEAN (already-computed evidence) orders by actual estimated gain.
        def _pm(n):
            try:
                pg = self._paired_gate(n)
                return pg[2] if pg else 0.0
            except Exception:
                return 0.0
        ranked = sorted(cands, key=lambda n: ((ld.get(n) or 0), _pm(n)), reverse=True)
        if os.environ.get("VERIFIER", "") != "paired":
            return ranked[0]
        for n in ranked:
            try:
                pg = self._paired_gate(n)
            except Exception:
                pg = None
            if pg is None or pg[0]:          # gate unavailable -> let the ship-time check decide
                self._best_verified = bool(pg and pg[0])
                return n
        return "source"                       # nothing passes verification

    def _paired_gate(self, name, B=2000, lo_pct=None):
        # OBJECTIVE-CONDITIONED confidence bound (from the user's intake contract):
        #   never_worse (default) -> ship only if the 10th pct of the paired gain > 0 (~90% conf)
        #   max_performance       -> 25th pct (~75% conf): the user asked to prioritize recovery
        # This is the safety/sensitivity dial made explicit and USER-OWNED -- the threshold follows
        # the stated objective, not benchmark tuning. min_annotation uses the strict bound too.
        if lo_pct is None:
            obj = getattr(self, "objective", None) or os.environ.get("OBJECTIVE", "never_worse")
            lo_pct = 25.0 if str(obj) == "max_performance" else 10.0
        """PAIRED PER-SAMPLE verification (replaces bootstrapping AUROC).

        Two problems killed the AUROC version at deployment-realistic budgets: fine-tuned
        candidates are only verifiable on the ~4-sample holdout, and AUROC over <=12 samples is
        so coarse that most bootstrap resamples tie at exactly 0. Both vanish if we compare the
        models WHERE THE SIGNAL IS -- per case. For each labeled case i we take
              d_i = p_candidate(true class_i) - p_source(true class_i),
        a paired per-sample quantity (n real values, not one rank statistic). Ship only if the
        lower percentile of the bootstrapped mean of d is still > 0, i.e. the per-case improvement
        survives resampling the annotation set. Label-free candidates never train on labels, so
        they are scored on ALL K; fine-tunes are scored on the held-out split only.
        Returns (ok, lo, mean_delta, n, frac_improved) or None."""
        rec = next((b for b in self.built if b["name"] == name), None)
        if rec is None or self._hold is None:
            return None
        if str(name) == "probe_head":
            # CV-OOF PAIRING: the probe is cheap to refit, so instead of judging it on the 4-sample
            # holdout (too coarse -- blocked a true 0.898 on breakhis), refit it per-fold on the K
            # labels and pair on OUT-OF-FOLD predictions: every one of the K labels contributes an
            # unbiased paired sample. Only possible because the probe is a one-second logistic fit;
            # finetunes stay on the holdout (per-fold refits would cost minutes each).
            oof = self._probe_oof_probs()
            if oof is not None:
                imgs, lbls, pc = oof
                y = np.asarray([int(v) for v in lbls])
                try:
                    ps = self._probs_on(self.source_model, imgs)
                except Exception:
                    return None
                idx = np.arange(len(y))
                d = pc[idx, y] - ps[idx, y]
                n = len(d)
                rng = np.random.RandomState(0)
                means = np.array([d[rng.randint(0, n, n)].mean() for _ in range(B)])
                lo = float(np.percentile(means, lo_pct))
                return (bool(lo > 0.0), lo, float(d.mean()), n, float((d > 0).mean()))
            imgs, lbls = self._hold
        elif str(name).startswith("finetune") or str(name).startswith("wise_"):
            imgs, lbls = self._hold
        else:
            imgs = list(self._train[0]) + list(self._hold[0])
            lbls = list(self._train[1]) + list(self._hold[1])
        y = np.asarray([int(v) for v in lbls])
        if len(y) < 3:
            return None
        try:
            pc = self._probs_on(rec["model"], imgs)
            ps = self._probs_on(self.source_model, imgs)
        except Exception:
            return None
        idx = np.arange(len(y))
        d = pc[idx, y] - ps[idx, y]          # per-case gain in true-class probability
        n = len(d)
        # POWER GUARD (ledger 10): a bootstrap over <6 samples is not a test -- a fine-tune passed
        # it on noise and shipped a -0.107 collapse the flat gate would have refused. Below n=6 the
        # paired gate ABSTAINS (returns None) so the margin gate stays authoritative.
        if (str(name).startswith("finetune") or str(name).startswith("wise_")) and n < 6:
            return None
        rng = np.random.RandomState(0)
        means = np.array([d[rng.randint(0, n, n)].mean() for _ in range(B)])
        lo = float(np.percentile(means, lo_pct))
        # practical-significance floor: a +-0.001 tie is not a deployable improvement
        ok = bool(lo > 0.0) and float(d.mean()) >= 0.005
        return (ok, lo, float(d.mean()), n, float((d > 0).mean()))

    # -- I1 safety ----------------------------------------------------------
    def safety_margin(self, source_metric) -> float:
        """The ONE numeric decision gate (design §A.1) — identical formula to SegTask."""
        return 0.05  # flat NOISE floor (2026-06-08); was max(0.05,0.10*source) — the 10%-relative term over-rejected real verified gains at high baselines

    def safety_check(self, candidate, source, evidence):
        if self._src_stats is None:
            return SafetyVerdict(False, "not deployed yet")
        if candidate is source:
            return SafetyVerdict(True, "source baseline")
        rec = next((b for b in self.built if b["model"] is candidate), None)
        name = rec["name"] if rec else None
        # (1) labels authoritative (with margin — few labels are noisy)
        if self._labeled_dice is not None and name in self._labeled_dice:
            if os.environ.get("VERIFIER", "") == "paired":
                pg = self._paired_gate(name)
                if pg is not None:
                    ok, lo, mean_d, npair, frac = pg
                    self._gate_metric_note = (f"paired per-case n={npair} mean{mean_d:+.3f} "
                                              f"p10 {lo:+.3f} improved {frac:.0%}")
                    return SafetyVerdict(ok, f"paired per-case gain mean {mean_d:+.4f}, p10 "
                                             f"{lo:+.4f}, {frac:.0%} of {npair} cases improved")
            cd = self._labeled_dice[name]; sd = self._labeled_dice.get("source", 0.5)
            margin = self.safety_margin(sd)
            if cd < sd + margin:
                return SafetyVerdict(False, f"labeled {self.metric_name} {cd} not clearly > source {sd}")
            return SafetyVerdict(True, f"labeled {self.metric_name} {cd} > source {sd}")
        # (2) no labels: don't ship a class-collapsed model; and (confidence is an unreliable
        # proxy, like seg's quality) require the candidate to CLEARLY beat source confidence,
        # not merely tie — a marginal label-free win must be label-verified instead.
        s = self._stats_of(candidate)
        if s["max_class_frac"] > 0.97:
            return SafetyVerdict(False, "candidate collapsed to one class (degenerate)")
        if _quality(s) < _quality(self._src_stats) + 0.03:
            return SafetyVerdict(False, "label-free confidence not clearly above source (verify with labels)")
        return SafetyVerdict(True, "passes label-free safety vs source")
