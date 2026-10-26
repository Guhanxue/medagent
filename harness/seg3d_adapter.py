"""Segmentation-3D TaskAdapter for the deployment harness.

The 3D sibling of harness/seg_adapter.py::SegTask. It exposes the SAME interface
(class attrs, __init__ fields, toolbox/safety_check/_request/_finetune/... method
shapes + result-dict keys) so harness/core.py and harness/llm_policy.py run it
UNCHANGED. The only differences are volumetric: inference is MONAI sliding-window,
Dice is 3D volumetric overlap, and the label-free signals are computed over whole
volumes (entropy / fg-fraction / 3D connected-components).

REUSE (do NOT reimplement): all 3D inference / Dice / VLM-rendering comes from
medadaptagent/tasks/segmentation_3d.py::SegmentationTask3D (via get_task), the
volume loader from scripts.legacy.run_agent_sandbox._load_3d_volumes, the model
loader from scripts._source_loader.load_source_model, and the pairwise-VLM signal
from harness.seg_adapter._vlm_prefer_fraction.

Three annotation-budget regimes (mirrors the 2D path exactly):
  K=0          label-free: deploy + bn_reset/tent, ship only if VLM prefers it
               over source and the output is not degenerate.
  K>0 no-FT    labels-for-selection: request_labels reveals K labels, ranks ALL
               candidates by HELD-OUT 3D Dice, ships the label-verified best.
  K>0 few-shot finetune_{head,partial,lora}: trains on the K TRAIN volumes,
               ranks on the HOLDOUT split.

NOTE — registry bridge deferred for 3D: the 2D _bridged_tools / bridged_specs
ctx-scoring path is 2D-oriented (it replays a numpy 2D preprocess chain) and is
risky in 3D, so the 3D toolbox is the hand-written set only.

NOTE — 3D fine-tuning: harness/finetune.py's _seg_loss + param-selection are
2D-only (4D one_hot/permute, Conv2d-only head detection); calling it on a 3D
(5D-logit, Conv3d) model is a silent no-op/shape error. Rather than MODIFY that
shared file, this adapter does its few-shot 3D fine-tune locally in _finetune,
reusing the 3D task's patch-crop CE pattern (ce_loss_on_labels) + 3D Dice and a
Conv3d-aware param selector. Same capacity tiers (head/partial/lora) and the same
result-dict shape as the 2D _finetune.
"""
from __future__ import annotations
import os
import copy
import numpy as np
from harness.core import Tool, SafetyVerdict, Budget

# Reuse the 2D label-free quality fn, pairwise-VLM signal, and morphology-diagnosis
# verbatim (they are dimension-agnostic — they consume a stats dict, not images).
from harness.seg_adapter import (
    _quality_score, _vlm_prefer_fraction, _diagnose_from_stats,
)
# Label-free (K=0) INPUT-SPACE / prediction-time / norm-stat tools (the real K=0
# lever on InstanceNorm/GroupNorm 3D models — bn_reset/tent are no-ops there).
from harness import labelfree_3d_tools as _lf3d


# --------------------------------------------------------------- label-free signals (3D)
def _predict_stats_3d(model, vols, t3d, device, n=4):
    """Mean softmax entropy, mean fg fraction (voxels pred>=1 over the whole volume),
    mean 3D connected-components over up to `n` evenly-spaced target volumes.
    Sliding-window inference comes from the 3D task (t3d.forward_pred)."""
    import torch
    import torch.nn.functional as F
    from scipy import ndimage
    model.eval()
    if len(vols) == 0:
        return {"entropy": 0.0, "fg_frac": 0.0, "n_comp": 0.0}
    idx = np.linspace(0, len(vols) - 1, min(n, len(vols)), dtype=int)
    ents, fgs, comps = [], [], []
    with torch.no_grad():
        for i in idx:
            x = t3d.to_tensor(vols[int(i)], device)        # [1,1,D,H,W], CT-windowed
            logits, pred = t3d.forward_pred(model, x)       # logits [1,C,D,H,W], pred [D,H,W]
            p = F.softmax(logits, 1)
            ents.append(float(-(p * torch.clamp(p, 1e-6, 1).log()).sum(1).mean()))
            fg = (pred >= 1)
            fgs.append(float(fg.mean()))
            comps.append(int(ndimage.label(fg)[1]) if fg.any() else 0)
    return {"entropy": float(np.mean(ents)), "fg_frac": float(np.mean(fgs)),
            "n_comp": float(np.mean(comps))}


# --------------------------------------------------------------- per-dataset preprocessing
def _make_window_to_tensor(a_min, a_max, pixdim=None):
    """Build a per-dataset `to_tensor(img, size, device) -> [1,1,D,H,W]` closure that
    reproduces the source MONAI bundle's intensity (and optional spacing) preprocessing.

    THE BUG THIS FIXES: the generic default windowed CT to [-1024,1024]->[0,1], ~20x too
    wide for the spleen bundle (which trains on ScaleIntensityRange a_min=-57 a_max=164),
    feeding the model garbage intensities -> 0 Dice even on its own training data. The
    window is read per-dataset from the bundle's configs/inference.json, never a constant.

    pixdim (optional): if given, also resample the volume to the bundle's voxel spacing via
    the calibration.Volume's .affine (MONAI Spacingd equivalent). Intensity-only by default."""
    a_min = float(a_min); a_max = float(a_max); rng = (a_max - a_min) + 1e-6

    def _to_tensor(img, size, device):
        import torch
        arr = np.asarray(img).astype(np.float32)
        affine = getattr(img, "affine", None)
        if pixdim is not None and affine is not None:
            arr = _resample_to_spacing(arr, affine, pixdim, order=1)
        arr = np.clip(arr, a_min, a_max)
        arr = (arr - a_min) / rng
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    _to_tensor._window = (a_min, a_max)
    _to_tensor._pixdim = pixdim
    return _to_tensor


def _spacing_from_affine(affine):
    """Voxel size (sx,sy,sz) in mm from a 4x4 affine (column norms)."""
    a = np.asarray(affine, dtype=np.float64)
    return np.sqrt((a[:3, :3] ** 2).sum(axis=0))


def _resample_to_spacing(arr, affine, pixdim, order=1):
    """Resample a [D,H,W] array from its native voxel spacing (from affine) to `pixdim`
    (MONAI Spacingd equivalent, no reorientation — bundle's Orientationd is a no-op for our
    already-RAS-ish loaders). order=1 bilinear for images, 0 nearest for labels."""
    from scipy import ndimage
    cur = _spacing_from_affine(affine)
    if cur.shape[0] != 3 or np.any(cur <= 0):
        return arr
    pix = np.asarray(pixdim, dtype=np.float64)
    zoom = cur / pix
    if np.allclose(zoom, 1.0, atol=1e-3):
        return arr
    return ndimage.zoom(arr, zoom, order=order)


class SegTask3D:
    type, dims, metric_name = "segmentation_3d", "3D", "Dice"

    def __init__(self, source_model, simgs, timgs, nc, device, fg_prior=0.01, tlbls=None,
                 task_desc="", a_min=-57.0, a_max=164.0, pixdim=None):
        self.source_model = source_model
        self.simgs, self.timgs = simgs, timgs
        self.tlbls = tlbls         # target GT volumes — used ONLY by request_labels/finetune
        self.nc, self.device = nc, device
        self.fg_prior = fg_prior   # expected fg VOXEL fraction (spleen-in-abdomen is tiny)
        # PER-DATASET intensity window (read from the source bundle's inference.json
        # ScaleIntensityRanged) + optional voxel spacing — NOT a magic constant. Threaded
        # through EVERY tensor conversion (deploy/dice/bn_reset/tent/finetune) via the
        # cached 3D task's source-preprocessing hook in _t3d().
        self.a_min, self.a_max, self.pixdim = float(a_min), float(a_max), pixdim
        self._to_tensor = _make_window_to_tensor(self.a_min, self.a_max, self.pixdim)
        self.task_desc = task_desc
        self.USABLE_BAR = 0.30
        self._src_stats = None
        self.built = []
        self.vlm = None
        self.VLM_PREFER_THRESH = 0.66
        self.labeled_k = 0
        self._labeled_dice = None
        self._labels_revealed = False
        self._train = None
        self._hold = None
        self._segtask = None       # cached SegmentationTask3D
        self._shift_ev = None
        self._model_ev = None
        self._perf_est = None
        self._source_ref = None    # cached label-free source-intensity reference (lazy)
        self._source_ref_built = False

    # -- 3D task object (the source of all inference/Dice/VLM rendering) -----------
    def _t3d(self):
        if self._segtask is None:
            from medadaptagent.tasks import get_task
            self._segtask = get_task("segmentation_3d")
            # Install the per-dataset window so EVERY to_tensor (deploy/dice/bn_reset/tent/
            # finetune/sample_metric/forward_pred) uses the bundle's intensity range, not the
            # generic [-1024,1024] default. Single point => deploy, dice, finetune are consistent.
            tag = f"bundle_window[{self.a_min},{self.a_max}]" + (
                f"+spacing{self.pixdim}" if self.pixdim is not None else "")
            self._segtask.set_source_preprocessing(tag, self._to_tensor, 96)
        return self._segtask

    def set_vlm(self, llm, task_ctx, task, anchors):
        """Enable the VLM pairwise-preference signal (needs Azure key + tunnel).
        `task` is the SegmentationTask3D obj — its render_pair_for_vlm handles 3D."""
        self.vlm = {"llm": llm, "ctx": task_ctx, "task": task, "anchors": anchors}

    def _vlm_of(self, cand):
        if not self.vlm:
            return None
        v = self.vlm
        return _vlm_prefer_fraction(cand, self.source_model, v["anchors"], v["llm"],
                                    v["ctx"], v["task"], self.device)

    def _llm(self):
        return self.vlm["llm"] if self.vlm else None

    # -- task-specific toolbox (mirror 2D; registry bridge deferred for 3D) --------
    def toolbox(self):
        B1 = Budget(compute_minutes=1)
        Bt = Budget(compute_minutes=2, n_trials=1)
        # Dynamic norm-type detection: bn_reset/tent are LIVE iff the model actually has BatchNorm.
        # (MONAI spleen-UNet is BatchNorm3d; SegResNet/SwinUNETR are InstanceNorm/GroupNorm.) The
        # descriptions used to hard-code "no-op on InstanceNorm/GroupNorm", which wrongly told the
        # agent to skip BN-TTA on a BatchNorm model (missed synapse tent +0.215). Reflect reality.
        import torch.nn as _nn
        _bn = sum(1 for _m in self.source_model.modules()
                  if isinstance(_m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)))
        if _bn > 0:
            _bn_tools = [
                Tool("bn_reset", "recalibrate BatchNorm stats to target volumes (no labels); this "
                     "model has %d BatchNorm layer(s) so it IS live -- high-variance, VERIFY before "
                     "shipping (the safety gate keeps you >= source)." % _bn, self._bn_reset, Bt),
                Tool("tent", "entropy-min TTA on BN affine params (live: %d BN layer(s); high-variance"
                     ", verify)." % _bn, self._tent, Bt)]
        else:
            _bn_tools = []   # InstanceNorm/GroupNorm/LayerNorm (no BatchNorm) -> BN-TTA hard-EXCLUDED
        return [
            Tool("load_and_deploy", "run the model on target volumes; report fg/entropy",
                 self._deploy, B1),
            Tool("analyze_domain_shift", "MEASURE source-vs-target image stats + characterize "
                 "the shift (intensity / contrast / modality / resolution / FOV, severity)",
                 self._analyze_shift, Budget(compute_minutes=1, llm_usd=0.02)),
            Tool("inspect_model", "introspect the model (architecture, normalization, in/out, "
                 "likely training domain) — tells you which adaptations apply",
                 self._inspect_model, Budget(llm_usd=0.02)),
            Tool("diagnose_masks", "seg failure analysis: under/over-seg, fragmentation, confidence",
                 self._diagnose, B1),
            *_bn_tools,
            # -------- LABEL-FREE (K=0) input-space / prediction-time / norm-stat tools --------
            # The real K=0 lever on InstanceNorm/GroupNorm 3D models. Each produces a
            # label-free candidate scored exactly like bn_reset (no labels needed).
            Tool("histogram_match", "LABEL-FREE: match the target intensity histogram to a few "
                 "UNLABELED source-domain volumes (CDF transport) so the model sees a source-like "
                 "distribution — fixes an intensity/contrast appearance gap with no labels",
                 self._lf_histogram_match, Bt),
            Tool("clahe", "LABEL-FREE: 3D CLAHE local-contrast equalization of the target volume "
                 "(no source ref, no labels) — recovers contrast washed out by the shift",
                 self._lf_clahe, Bt),
            Tool("gamma", "LABEL-FREE: auto-gamma brightness-curve correction so the target mean "
                 "matches the source mean (no labels) — fixes a global brightness gap",
                 self._lf_gamma, Bt),
            Tool("fda", "LABEL-FREE: Fourier Domain Adaptation — swap the low-frequency amplitude "
                 "band of the target with a few UNLABELED source volumes (preserves content) to "
                 "bridge a style/appearance gap with no labels",
                 self._lf_fda, Bt),
            Tool("tta_aug", "LABEL-FREE: prediction-time test-time augmentation — average the "
                 "source softmax over flips + 90-deg rotations. PURE inference, NO training, "
                 "NORM-AGNOSTIC (works on InstanceNorm/GroupNorm where bn_reset/tent are no-ops)",
                 self._lf_tta_aug, Bt),
            Tool("denoise", "LABEL-FREE: input denoise (Gaussian + median) of the target volume — "
                 "directly targets a low-dose / acquisition NOISE component, no labels",
                 self._lf_denoise, Bt),
            Tool("denoise_histmatch", "LABEL-FREE: denoise THEN histogram-match to source — "
                 "combo for a shift that is both noisy AND intensity-shifted, no labels",
                 self._lf_denoise_histmatch, Bt),
            Tool("norm_stat_tent", "LABEL-FREE: entropy-minimization TTA on the affine params of "
                 "ALL norm layers (GroupNorm/InstanceNorm/LayerNorm/BatchNorm, not just BN) — the "
                 "REAL norm lever on InstanceNorm/GroupNorm models where bn_reset/tent are no-ops",
                 self._lf_norm_stat_tent, Bt),
            Tool("evaluate", "label-free quality of the current best candidate",
                 self._evaluate, B1),
            Tool("request_labels", "ask the clinician for K (3-12) labeled volumes; split into "
                 "TRAIN+HOLDOUT, then rank ALL candidates by HELD-OUT 3D Dice — use when "
                 "label-free signals can't decide or the best is still poor",
                 self._request, lambda a: Budget(k_labels=int(a.get("K", int(os.environ.get("KLAB", "6") or 6))))),
            Tool("finetune_head", "FINE-TUNE only the output head on the K train volumes — "
                 "cheapest, safest at tiny K; use first when labels are present but TTA stalls",
                 lambda st, **k: self._finetune("head", **k), Budget(compute_minutes=3)),
            Tool("finetune_partial", "fine-tune the decoder + head on the K train volumes — "
                 "more capacity to bridge a real shift; needs a few more labels",
                 lambda st, **k: self._finetune("partial", **k), Budget(compute_minutes=4)),
            Tool("finetune_lora", "parameter-efficient LoRA adapters on the K train volumes — "
                 "best capacity-per-label; good when head/partial underfit",
                 lambda st, **k: self._finetune("lora", **k), Budget(compute_minutes=4)),
            Tool("recommend_alternative", "when training-free adaptation CANNOT reach usable "
                 "Dice (a fundamental shift): get modality-appropriate pretrained models to "
                 "use INSTEAD + a fine-tuning label-budget estimate",
                 self._recommend_alternative, Budget(llm_usd=0.05)),
        ] + self._scaling_tools()

    def _scaling_tools(self):
        # Env-gated TOOL REGISTRATION (SCALING_EVAL=0 default) — true inertness when off.
        if os.environ.get("SCALING_EVAL", "0") != "1":
            return []
        return [Tool("scaling_eval", "few-shot SCALING-LAW: fine-tune across label budgets -> "
                     "the honest delivered-model estimate + an 'annotate N -> expect X' plan.",
                     lambda st, **k: self._scaling_eval(**k), Budget(compute_minutes=8))]

    # -- stats / dice helpers (3D, via the 3D task) -------------------------------
    def _stats_of(self, cand, n=4):
        return _predict_stats_3d(cand, self.timgs, self._t3d(), self.device, n=n)

    def _dice_of_3d(self, model, vols, lbls):
        """Mean 3D foreground Dice over (vol,lbl) pairs using the PER-DATASET window.

        Reuses the 3D task's windowed to_tensor + forward_pred (same window as deploy/
        finetune), then the same foreground-Dice formula as SegmentationTask3D.sample_metric.
        If pixdim resampling is active the prediction grid differs from the label grid, so we
        resample the prediction back to label space (nearest) before scoring. With pixdim=None
        (intensity-only, the default) the pred grid == label grid and this matches sample_metric
        exactly."""
        import torch
        t3d = self._t3d()
        model.eval()
        ds = []
        with torch.no_grad():
            for vol, lbl in zip(vols, lbls):
                lbl = np.asarray(lbl)
                x = t3d.to_tensor(vol, self.device)            # [1,1,D,H,W], windowed
                _, pred = t3d.forward_pred(model, x)            # [D,H,W] on the (resampled) grid
                pred = np.asarray(pred)
                if pred.shape != lbl.shape:
                    pred = self._resize_pred_to(pred, lbl.shape)
                d = self._fg_dice(pred, lbl)
                if d is not None:
                    ds.append(float(d))
        return round(float(np.mean(ds)), 4) if ds else 0.0

    def _resize_pred_to(self, pred, shape):
        """Nearest-resample an integer prediction mask to `shape` (label grid)."""
        if pred.shape == tuple(shape):
            return pred
        from scipy import ndimage
        zoom = np.asarray(shape, dtype=np.float64) / np.asarray(pred.shape, dtype=np.float64)
        return ndimage.zoom(pred.astype(np.float32), zoom, order=0).astype(pred.dtype)

    def _fg_dice(self, pred, label):
        """Foreground (classes 1..nc-1) mean Dice — same formula as sample_metric."""
        dices = []
        for c in range(1, self.nc):
            p = (pred == c).astype(float)
            g = (label == c).astype(float)
            if g.sum() > 0:
                dices.append(float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8)))
        return float(np.mean(dices)) if dices else None

    # -- tools (each returns the SAME dict shape as 2D) ---------------------------
    def _deploy(self, st, **kw):
        s = _predict_stats_3d(self.source_model, self.timgs, self._t3d(), self.device)
        self._src_stats = s
        q = _quality_score(s, self.fg_prior)
        self.built.append({"name": "source", "model": self.source_model,
                           "quality": q, "vlm_prefer": None, "stats": s})
        return {"deployment": "source@target", "candidate": self.source_model,
                "evidence": {"source_stats": s, "quality": q, "last_method": "source"},
                "metric_estimate": q}

    def _diagnose(self, st, **kw):
        cand = st.best_candidate or self.source_model
        s = self._stats_of(cand)
        src = self._src_stats or s
        mode = _diagnose_from_stats(s, src)
        return {"evidence": {"failure_mode": mode, "stats": s}}

    def _analyze_shift(self, st, **kw):
        if self._shift_ev is not None:
            return {"evidence": {"domain_shift": self._shift_ev,
                    "directive": "shift ALREADY analyzed — do NOT call again; ACT: try an "
                                 "adaptation tool now."}}
        from harness.analysis import analyze_domain_shift
        # analyze_domain_shift reduces volumes to intensity stats; it is dimension-agnostic.
        self._shift_ev = analyze_domain_shift(self.simgs, self.timgs, self._llm(),
                                              self.task_desc, model=self.source_model)
        return {"evidence": {"domain_shift": self._shift_ev, **self._retrieve_lessons(st)}}

    def _inspect_model(self, st, **kw):
        if self._model_ev is not None:
            return {"evidence": {"model_info": self._model_ev,
                    "directive": "model ALREADY inspected — do NOT call again; ACT: try an "
                                 "adaptation tool now."}}
        from harness.analysis import inspect_model
        self._model_ev = inspect_model(self.source_model, self._llm(), self.task_desc)
        return {"evidence": {"model_info": self._model_ev, **self._retrieve_lessons(st)}}

    def _retrieve_lessons(self, st):
        if (getattr(st, "evidence", None) or {}).get("retrieved_lessons"):
            return {}
        try:
            from harness.lessons import maybe_retrieve
        except ImportError:
            from lessons import maybe_retrieve
        return maybe_retrieve(self._shift_ev, self._model_ev, "segmentation_3d")

    def _score_candidate(self, name, cand):
        s = _predict_stats_3d(cand, self.timgs, self._t3d(), self.device)
        q = _quality_score(s, self.fg_prior)
        vlm = self._vlm_of(cand)
        self.built.append({"name": name, "model": cand, "quality": q,
                           "vlm_prefer": vlm, "stats": s})
        ev = {"cand_stats": s, "quality": q, "last_method": name}
        if vlm is not None:
            ev["vlm_prefer"] = round(vlm, 3)
        return {"candidate": cand, "evidence": ev, "metric_estimate": q}

    # -- label-free adaptations (3D) ----------------------------------------------
    def _bn_reset_3d(self, model):
        """Deepcopy source; reset BN running stats; forward each target volume in
        train() under no_grad so BN re-estimates target mean/var; back to eval()."""
        import torch
        import torch.nn as nn
        t3d = self._t3d()
        m = copy.deepcopy(model).to(self.device)
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.reset_running_stats()
                mod.momentum = None      # cumulative moving average over the target volumes
        m.train()
        with torch.no_grad():
            for vol in self.timgs:
                x = t3d.to_tensor(vol, self.device)
                # full-volume forward feeds BN; sliding-window uses no_grad too but we want
                # the BN modules in train() to update stats, so call the model directly with
                # a center patch to keep memory bounded.
                self._bn_forward(m, x)
        m.eval()
        return m

    def _bn_forward(self, m, x):
        """Forward a single (center-cropped) patch through `m` to feed BN running stats
        without materializing the whole-volume activation map (OOM guard)."""
        patch = (96, 96, 96)
        D, H, W = x.shape[-3:]
        sl = []
        for dim, p in zip((D, H, W), patch):
            if dim <= p:
                sl.append(slice(0, dim))
            else:
                start = (dim - p) // 2
                sl.append(slice(start, start + p))
        xc = x[:, :, sl[0], sl[1], sl[2]]
        import torch.nn.functional as F
        pads = []
        for i in range(3):
            need = patch[2 - i] - xc.shape[4 - i]
            pads += [0, max(0, need)]
        if any(pads):
            xc = F.pad(xc, pads, mode="constant", value=0)
        m(xc)

    def _bn_reset(self, st, **kw):
        cand = self._bn_reset_3d(self.source_model)
        return self._score_candidate("bn_reset", cand)

    def _tent_3d(self, model, steps=3):
        """Entropy-minimization on BN affine params over a few target volumes (center
        patch per volume to bound memory). Mirrors 2D tent intent for 3D."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        t3d = self._t3d()
        m = copy.deepcopy(model).to(self.device)
        for p in m.parameters():
            p.requires_grad_(False)
        params = []
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.train()                  # use batch stats during tent
                if mod.weight is not None:
                    mod.weight.requires_grad_(True); params.append(mod.weight)
                if mod.bias is not None:
                    mod.bias.requires_grad_(True); params.append(mod.bias)
        if not params:
            return m.eval()
        opt = torch.optim.Adam(params, lr=1e-3)
        vols = self.timgs[:min(4, len(self.timgs))]
        for _ in range(steps):
            for vol in vols:
                x = t3d.to_tensor(vol, self.device)
                xc = self._center_patch(x)
                logits = m(xc)
                p = F.softmax(logits, 1)
                ent = -(p * torch.clamp(p, 1e-6, 1).log()).sum(1).mean()
                opt.zero_grad(); ent.backward(); opt.step()
        m.eval()
        return m

    def _center_patch(self, x, patch=(96, 96, 96)):
        import torch.nn.functional as F
        D, H, W = x.shape[-3:]
        sl = []
        for dim, p in zip((D, H, W), patch):
            start = max(0, (dim - p) // 2)
            sl.append(slice(start, start + min(p, dim)))
        xc = x[:, :, sl[0], sl[1], sl[2]]
        pads = []
        for i in range(3):
            pads += [0, max(0, patch[2 - i] - xc.shape[4 - i])]
        if any(pads):
            xc = F.pad(xc, pads, mode="constant", value=0)
        return xc

    def _tent(self, st, steps=3, **kw):
        cand = self._tent_3d(self.source_model, steps=steps)
        return self._score_candidate("tent", cand)

    # -- LABEL-FREE (K=0) input-space / prediction-time / norm-stat tools ----------
    # All produce a label-free candidate scored via _score_candidate (no labels). The
    # reference-needing tools (histogram_match/fda/gamma) build a SOURCE reference ONCE
    # from a few UNLABELED source-domain volumes (self.simgs), windowed [0,1] by the
    # adapter's own to_tensor. The candidate transforms the windowed patch tensor inside
    # forward(), so the whole sliding-window/Dice/stats/VLM chain scores it unchanged.
    def _src_ref(self):
        if not self._source_ref_built:
            t3d = self._t3d()
            srcs = self.simgs if self.simgs else self.timgs   # fall back to target if no source
            try:
                self._source_ref = _lf3d.build_source_ref(t3d.to_tensor, srcs, self.device,
                                                          n=min(4, len(srcs)))
            except Exception:
                self._source_ref = None
            self._source_ref_built = True
        return self._source_ref

    def _lf_histogram_match(self, st, **kw):
        cand, info = _lf3d.make_histogram_match(self.source_model, self._src_ref())
        return self._score_candidate("histogram_match", cand)

    def _lf_clahe(self, st, **kw):
        cand, info = _lf3d.make_clahe(self.source_model)
        return self._score_candidate("clahe", cand)

    def _lf_gamma(self, st, **kw):
        cand, info = _lf3d.make_gamma(self.source_model, self._src_ref())
        return self._score_candidate("gamma", cand)

    def _lf_fda(self, st, **kw):
        cand, info = _lf3d.make_fda(self.source_model, self._src_ref())
        return self._score_candidate("fda", cand)

    def _lf_tta_aug(self, st, **kw):
        cand, info = _lf3d.make_tta_aug(self.source_model)
        return self._score_candidate("tta_aug", cand)

    def _lf_denoise(self, st, **kw):
        cand, info = _lf3d.make_denoise(self.source_model)
        return self._score_candidate("denoise", cand)

    def _lf_denoise_histmatch(self, st, **kw):
        cand, info = _lf3d.make_denoise_histmatch(self.source_model, self._src_ref())
        return self._score_candidate("denoise_histmatch", cand)

    def _lf_norm_stat_tent(self, st, **kw):
        t3d = self._t3d()
        cand, info = _lf3d.make_norm_stat_tent(self.source_model, self.timgs,
                                               t3d.to_tensor, self.device)
        res = self._score_candidate("norm_stat_tent", cand)
        res.setdefault("evidence", {})["norm_stat_info"] = info
        return res

    def _evaluate(self, st, **kw):
        cand = st.best_candidate or self.source_model
        s = self._stats_of(cand)
        q = _quality_score(s, self.fg_prior)
        return {"evidence": {"eval_stats": s, "quality": q}, "metric_estimate": q}

    # -- K-shot labels: reveal, split, rank (mirror 2D exactly) -------------------
    def _ensure_labels(self, K=6):
        if self._labels_revealed:
            return self._train
        K = int(K)
        fg_idx = [i for i in range(len(self.tlbls))
                  if (np.asarray(self.tlbls[i]) >= 1).any()]
        pool = fg_idx if fg_idx else list(range(len(self.timgs)))
        sel = np.linspace(0, len(pool) - 1, min(K, len(pool)), dtype=int)
        idx = [pool[int(j)] for j in sel]
        n = len(idx)
        hold_pos = set(range(0, n, 3)) if n >= 2 else set()
        hold_idx = [idx[p] for p in range(n) if p in hold_pos]
        train_idx = [idx[p] for p in range(n) if p not in hold_pos] or hold_idx
        self._train = ([self.timgs[i] for i in train_idx], [self.tlbls[i] for i in train_idx])
        self._hold = ([self.timgs[i] for i in hold_idx], [self.tlbls[i] for i in hold_idx])
        self.labeled_k += len(idx)
        self._labels_revealed = True
        return self._train

    def _rank_on_holdout(self):
        h_imgs, h_lbls = self._hold
        all_imgs = list(self._train[0]) + list(h_imgs)
        all_lbls = list(self._train[1]) + list(h_lbls)
        ld = {}
        for b in self.built:
            if str(b["name"]).startswith("finetune"):
                ld[b["name"]] = self._dice_of_3d(b["model"], h_imgs, h_lbls)
            else:
                ld[b["name"]] = self._dice_of_3d(b["model"], all_imgs, all_lbls)
        self._labeled_dice = ld
        return ld

    def _estimate_performance(self):
        if self._perf_est is not None:
            return self._perf_est
        fallback = {"sota": 0.85, "usable": 0.65, "rationale": "generic seg default (no LLM)"}
        llm = self._llm()
        if llm is None:
            self._perf_est = fallback; return fallback
        import json, re
        prompt = (f"Medical-image 3D SEGMENTATION task: {self.task_desc}. Metric: volumetric "
                  f"foreground Dice.\nFrom the published literature/benchmarks for THIS structure "
                  f"+ modality, give: the Dice a COMPETENT / near-SOTA model typically reaches, and "
                  f"the MINIMUM clinically-usable Dice. Respond ONLY JSON: {{\"sota\": <0..1>, "
                  f"\"usable\": <0..1>, \"rationale\": \"<=15 words citing the typical benchmark\"}}.")
        try:
            d = json.loads(re.search(r"\{.*\}", llm.chat(prompt), re.S).group(0))
            d = {"sota": float(d.get("sota", 0.85)), "usable": float(d.get("usable", 0.65)),
                 "rationale": str(d.get("rationale", ""))[:160]}
        except Exception:
            d = fallback
        self._perf_est = d
        return d

    def _request(self, st, K=6, **kw):
        if self.tlbls is None:
            return {"k_labels_used": 0, "evidence": {"error": "no labels available here"}}
        # TOOL-ORDER FIX (2026-06): in the NO_FINETUNE / labels-for-SELECTION regime labels can
        # ONLY rank the candidate POOL. If called with an EMPTY pool (only `source` built) and no
        # labels revealed yet, ranking "source vs nothing" wrongly keeps source. Refuse softly,
        # consume NO labels, emit NO labeled_dice, and nudge the agent to build >=1 label-free
        # candidate first (the menu gate in llm_policy._available normally prevents reaching here;
        # this is the defensive backstop). NOT scoped to finetune: there labels TRAIN, so an empty
        # pool is the correct flow and this guard stays silent (it only fires when NO_FINETUNE).
        if (os.environ.get("NO_FINETUNE", "0") != "0" and not self._labels_revealed
                and sum(1 for b in self.built if str(b.get("name")) != "source") < 1):
            untried = [n for n in ("histogram_match", "gamma", "denoise", "denoise_histmatch",
                                   "norm_stat_tent", "clahe", "fda", "tta_aug")]
            return {"k_labels_used": 0, "evidence": {"request_labels_blocked": True,
                    "error": "build >=1 label-free candidate BEFORE requesting labels: in this "
                             "labels-for-SELECTION regime labels only RANK the candidate pool, so "
                             "an empty pool ranks source-vs-nothing and keeps source. Untried "
                             f"label-free tools remain: {', '.join(untried)}. Build a few, THEN "
                             "request_labels to rank them on held-out Dice."}}
        self._ensure_labels(K)
        ld = self._rank_on_holdout()
        best = max(ld, key=ld.get) if ld else None
        est = self._estimate_performance()
        return {"k_labels_used": self.labeled_k,
                "evidence": {"labeled_dice": ld, "best_by_labels": best,
                             "n_train": len(self._train[0]), "n_holdout": len(self._hold[0]),
                             "expected_metric": est["sota"], "usable_floor": est["usable"],
                             "safety_margin": self.safety_margin(ld.get("source", 0.0)),
                             "perf_rationale": est["rationale"],
                             "note": f"HELD-OUT 3D Dice ({len(self._hold[0])} volumes); "
                                     f"task-expected Dice≈{est['sota']} (usable≥{est['usable']}: "
                                     f"{est['rationale']}). Ship '{best}' if it beats source; if best "
                                     f"is well below {est['sota']}, fine-tune to close the gap."}}

    # -- few-shot 3D fine-tune (local, Conv3d-aware; see module docstring) --------
    def _finetune(self, mode, K=6, base="source", steps=None, lr=None, **kw):
        """Few-shot 3D fine-tune. The LLM DESIGNS the recipe (lr=/steps= override the per-budget
        default) and reacts to the returned diagnostics (loss/val/collapse/overfit). Safety floor
        (best-ckpt + never-worse-than-source guard) lives in _finetune_3d."""
        if self.tlbls is None:
            return {"evidence": {"error": "no labels available — cannot finetune"}}
        tr_imgs, tr_lbls = self._ensure_labels(K)
        if len(tr_imgs) == 0:
            return {"evidence": {"error": "no train labels after split (K too small)"}}
        start = self.source_model
        if base == "best" and self.built:
            start = max(self.built, key=lambda b: b.get("quality", 0))["model"]
        try: lr = float(lr) if lr is not None else None
        except Exception: lr = None
        try: steps = int(steps) if steps is not None else None
        except Exception: steps = None
        if lr is None:                                   # budget-aware default; gentler for high-capacity/small-K
            lr = 1e-3 if len(tr_imgs) >= 12 else 5e-4
            if mode in ("full", "partial"):
                lr *= 0.5
        if steps is None:
            steps = int(min(160, max(40, len(tr_imgs) * 8)))
        model, info = self._finetune_3d(start, tr_imgs, tr_lbls, mode=mode, steps=steps, lr=lr)
        name = f"finetune_{mode}"
        self.built = [b for b in self.built if b.get("name") != name]   # clean-replace on re-call
        if self._labeled_dice: self._labeled_dice.pop(name, None)
        res = self._score_candidate(name, model)
        self._rank_on_holdout()
        hd = (self._labeled_dice or {}).get(name)
        lf, ll = info.get("loss_first"), info.get("loss_last")
        sv, bv = info.get("src_val_loss"), info.get("best_val_loss")
        diag = []
        if info.get("collapse_guard"):
            diag.append("COLLAPSE-GUARD fired (no val gain -> returned source; lower lr / smaller capacity)")
        if (lf is not None and ll is not None and lf and ll < 0.2 * lf
                and sv is not None and bv is not None and bv >= sv - 1e-6):
            diag.append("OVERFIT (train loss crashed, val did not improve -> lower lr / fewer steps)")
        if info.get("used_val") is False:
            diag.append("small-K: no internal val (more volumes enable best-ckpt selection)")
        res.setdefault("evidence", {})
        best_by = max(self._labeled_dice, key=self._labeled_dice.get) if self._labeled_dice else None
        res["evidence"].update({"finetune": info, "labeled_dice": self._labeled_dice,
                                "best_by_labels": best_by, "usable_bar": self.USABLE_BAR,
                                "holdout_dice": hd, "trained_on": len(tr_imgs),
                                "recipe_used": info.get("recipe"), "ft_diagnostics": diag,
                                "note": (f"{name}: held-out 3D Dice {hd} on {len(self._hold[0])} vols | "
                                         f"recipe {info.get('recipe')} | loss {lf}->{ll}, val {sv}->{bv} "
                                         f"(pick@{info.get('picked_step')}) | "
                                         f"{'; '.join(diag) if diag else 'clean fit'} | "
                                         f"to REVISE re-call finetune_* with args lr=/steps=")})
        return res

    def _finetune_3d(self, model, imgs, lbls, mode="head", steps=40, lr=1e-3, seed=0, val_frac=0.3):
        """Few-shot 3D fine-tune on a SINGLE labeled patch per volume (memory-bounded). Collapse-safe:
        when >=4 volumes, holds out val_frac patches as an internal val, tracks val-loss, restores the
        BEST checkpoint, early-stops on plateau, and if NO step beats the source's own val-loss returns
        the SOURCE weights (guard). At small K (<4) trains on all patches (no guard) — same as before."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        t3d = self._t3d()
        m = copy.deepcopy(model).to(self.device)
        n_train = self._set_trainable_3d(m, mode)
        m = m.to(self.device)
        if n_train == 0:
            m.eval()
            return m, {"mode": mode, "trainable_params": 0, "note": "no trainable params; no-op",
                       "loss_first": None, "loss_last": None}
        patches = [self._label_patch(t3d, im, lb) for im, lb in zip(imgs, lbls)]
        patches = [p for p in patches if p is not None]
        if not patches:
            m.eval()
            return m, {"mode": mode, "trainable_params": int(n_train),
                       "note": "no usable label patch", "loss_first": None, "loss_last": None}
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(patches), generator=g).tolist()
        use_val = len(patches) >= 4
        n_val = max(1, int(round(len(patches) * val_frac))) if use_val else 0
        val_p = [patches[i] for i in perm[:n_val]]
        tr_p = [patches[i] for i in perm[n_val:]] if n_val and len(perm) - n_val >= 1 else patches

        def _fit(lg, yc):
            yy = yc if lg.shape[-3:] == yc.shape[-3:] else F.interpolate(
                yc.unsqueeze(1).float(), size=lg.shape[-3:], mode="nearest").squeeze(1).long()
            return self._seg_loss_3d(lg, yy)

        def _vloss():
            if not val_p:
                return None
            was = m.training; m.eval(); tot = 0.0
            with torch.no_grad():
                for xc, yc in val_p:
                    tot += float(_fit(m(xc.to(self.device)), yc.to(self.device)))
            if was: m.train()
            return tot / len(val_p)

        def _set_train():
            m.train()
            for mod in m.modules():
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    mod.eval()

        src_val = _vloss() if use_val else None
        best_val = src_val
        best_state = copy.deepcopy(m.state_dict()) if use_val else None
        best_step = 0
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
        eval_every = max(5, steps // 8); patience, bad = 3, 0
        g2 = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(tr_p), generator=g2).tolist()
        losses = []
        for step in range(steps):
            _set_train()
            xc, yc = tr_p[order[step % len(tr_p)]]
            loss = _fit(m(xc.to(self.device)), yc.to(self.device))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
            if use_val and (step + 1) % eval_every == 0:
                v = _vloss()
                if v is not None and (best_val is None or v < best_val - 1e-4):
                    best_val, best_step, bad = v, step + 1, 0
                    best_state = copy.deepcopy(m.state_dict())
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if use_val:
            m.load_state_dict(best_state)
        m.eval()
        guard = bool(use_val and src_val is not None and best_step == 0)
        return m, {"mode": mode, "trainable_params": int(n_train), "steps": steps, "lr": lr,
                   "loss_first": round(losses[0], 4) if losses else None,
                   "loss_last": round(losses[-1], 4) if losses else None,
                   "src_val_loss": None if src_val is None else round(src_val, 4),
                   "best_val_loss": None if best_val is None else round(best_val, 4),
                   "picked_step": best_step, "used_val": use_val,
                   "recipe": f"lr={lr:.1e},steps={steps}", "collapse_guard": guard}

    def _label_patch(self, t3d, img, lbl, patch=(96, 96, 96)):
        """A single 96^3 patch centered on the labeled foreground (so the crop actually
        contains the structure), as (img_tensor[1,1,d,h,w], lbl_tensor[1,d,h,w])."""
        import torch
        lbl = np.asarray(lbl)
        x = t3d.to_tensor(img, self.device)            # [1,1,D,H,W] windowed, on device
        vol = x[0, 0].detach().cpu().numpy()
        # align label grid to the (possibly resampled) tensor grid
        if lbl.shape != vol.shape:
            lt = torch.from_numpy(lbl.astype(np.float32))[None, None]
            lt = torch.nn.functional.interpolate(lt, size=vol.shape, mode="nearest")
            lbl = lt[0, 0].numpy().astype(np.int64)
        fg = np.argwhere(lbl >= 1)
        if len(fg) == 0:
            return None
        cz, cy, cx = fg.mean(0).astype(int)
        D, H, W = vol.shape
        def _rng(c, p, n):
            s = int(np.clip(c - p // 2, 0, max(0, n - p)))
            return slice(s, s + min(p, n))
        sl = (_rng(cz, patch[0], D), _rng(cy, patch[1], H), _rng(cx, patch[2], W))
        xc = x[:, :, sl[0], sl[1], sl[2]]
        yc = torch.from_numpy(np.clip(lbl[sl], 0, self.nc - 1).astype(np.int64))[None]
        import torch.nn.functional as F
        pads = []
        for i in range(3):
            pads += [0, max(0, patch[2 - i] - xc.shape[4 - i])]
        if any(pads):
            xc = F.pad(xc, pads, mode="constant", value=0)
            yc = F.pad(yc, pads, mode="constant", value=0)
        return xc.detach().cpu(), yc.detach().cpu()

    def _set_trainable_3d(self, model, mode, partial_frac=0.4, r=4, alpha=8, tail_frac=0.5):
        """Conv3d/Linear-aware param selection (head < partial < lora). Mirrors the
        capacity tiers of harness.finetune.set_trainable/inject_lora but for 3D modules."""
        import torch.nn as nn
        from harness.finetune import LoRAConv2d  # noqa: F401 (only to confirm module import)

        def leaf_mods(m):
            return [(n, mod) for n, mod in m.named_modules()
                    if list(mod.parameters(recurse=False))]

        def head_mod(m):
            h = None
            for n, mod in m.named_modules():
                if isinstance(mod, nn.Conv3d) and mod.out_channels == self.nc:
                    h = (n, mod)
                elif isinstance(mod, nn.Linear) and mod.out_features == self.nc:
                    h = (n, mod)
            return h

        for p in model.parameters():
            p.requires_grad_(False)
        if mode == "full":
            for p in model.parameters():
                p.requires_grad_(True)
        elif mode == "head":
            h = head_mod(model)
            if h:
                for p in h[1].parameters():
                    p.requires_grad_(True)
        elif mode == "partial":
            mods = leaf_mods(model)
            k = max(1, int(len(mods) * partial_frac))
            for _, mod in mods[-k:]:
                for p in mod.parameters(recurse=False):
                    p.requires_grad_(True)
            h = head_mod(model)
            if h:
                for p in h[1].parameters():
                    p.requires_grad_(True)
        elif mode == "lora":
            return self._inject_lora_3d(model, r=r, alpha=alpha, tail_frac=tail_frac)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _inject_lora_3d(self, model, r=4, alpha=8, tail_frac=0.5):
        """Minimal Conv3d/Linear LoRA: wrap the TAIL tail_frac of conv/linear modules
        with low-rank adapters, freeze the base. Returns #trainable params."""
        import torch.nn as nn

        class LoRAConv3d(nn.Module):
            def __init__(self, base, r, alpha):
                super().__init__()
                self.base = base
                for p in self.base.parameters():
                    p.requires_grad_(False)
                self.A = nn.Conv3d(base.in_channels, r, base.kernel_size, base.stride,
                                   base.padding, bias=False)
                self.B = nn.Conv3d(r, base.out_channels, 1, bias=False)
                nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
                nn.init.zeros_(self.B.weight)
                self.scale = alpha / r

            def forward(self, x):
                return self.base(x) + self.scale * self.B(self.A(x))

        class LoRALinear(nn.Module):
            def __init__(self, base, r, alpha):
                super().__init__()
                self.base = base
                for p in self.base.parameters():
                    p.requires_grad_(False)
                self.A = nn.Linear(base.in_features, r, bias=False)
                self.B = nn.Linear(r, base.out_features, bias=False)
                nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
                nn.init.zeros_(self.B.weight)
                self.scale = alpha / r

            def forward(self, x):
                return self.base(x) + self.scale * self.B(self.A(x))

        def set_sub(m, dotted, new):
            parent = m
            parts = dotted.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            last = parts[-1]
            if last.isdigit():
                parent[int(last)] = new
            else:
                setattr(parent, last, new)

        for p in model.parameters():
            p.requires_grad_(False)
        targets = [(n, mod) for n, mod in model.named_modules()
                   if isinstance(mod, (nn.Conv3d, nn.Linear))]
        k = max(1, int(len(targets) * tail_frac))
        for n, mod in targets[-k:]:
            wrap = (LoRAConv3d(mod, r, alpha) if isinstance(mod, nn.Conv3d)
                    else LoRALinear(mod, r, alpha))
            set_sub(model, n, wrap)
        model.to(self.device)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _seg_loss_3d(self, logit, lbl):
        """3D CE + soft-Dice (5D logit [B,C,D,H,W], lbl [B,D,H,W] long)."""
        import torch.nn.functional as F
        ce = F.cross_entropy(logit, lbl)
        p = F.softmax(logit, 1)
        oh = F.one_hot(lbl, logit.shape[1]).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = (p * oh).sum(dims)
        denom = p.sum(dims) + oh.sum(dims)
        dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
        return ce + dice

    def _recommend_alternative(self, st, **kw):
        from harness.model_search import recommend_alternative_models
        best = max((self._labeled_dice or {}).values(), default=None)
        rec = recommend_alternative_models(
            self._llm(), task_desc=self.task_desc, task_type="segmentation_3d",
            metric_name="Dice", best_metric=best, usable_target=0.70,
            domain_shift=st.evidence.get("domain_shift"),
            model_info=st.evidence.get("model_info"))
        return {"evidence": {"recommendation": rec}}

    def _scaling_eval(self, K=12, modes=("head", "partial"), **kw):
        """Env-gated scaling stub for 3D: reveal K, fine-tune the cheapest mode on the
        full K, report its held-out 3D Dice + a simple annotate->expect plan. (The 2D
        k-fold CV gate in harness.scaling is 2D-Dice-coupled; deferred for 3D.)"""
        if self.tlbls is None:
            return {"evidence": {"error": "no labels available — cannot scale"}}
        self._ensure_labels(K)
        res = self._finetune("head", K=K)
        est = self._estimate_performance()
        hd = (self._labeled_dice or {}).get("finetune_head")
        res.setdefault("evidence", {})
        res["evidence"].update({"expected_metric": est["sota"], "usable_floor": est["usable"],
                                "perf_rationale": est["rationale"],
                                "scaling_plan": f"annotate ~{self.labeled_k} volumes -> 3D Dice ≈ {hd}"})
        return res

    # -- I1 safety: never ship a candidate worse/degenerate vs source (mirror 2D) --
    def safety_margin(self, source_metric) -> float:
        return 0.05  # flat NOISE floor (2026-06-08); was max(0.05,0.10*source) — the 10%-relative term over-rejected real verified gains at high baselines

    def _structural_plausibility(self, cand, s=None):
        s = s if s is not None else self._stats_of(cand)
        ff = float(s.get('fg_frac', 0.0) or 0.0)
        if ff < 0.15 * self.fg_prior:
            return False, 'empty/collapsed foreground'
        if ff > 5.0 * self.fg_prior:
            return False, 'exploded foreground (bleeds into background)'
        snc = int((self._src_stats or {}).get('n_comp', 0) or 0)
        nc = int(s.get('n_comp', 0) or 0)
        if snc > 0 and nc > 5 * snc + 10:
            return False, f'fragmented into {nc} components (source {snc}) — scattered/incoherent'
        return True, 'structurally plausible'

    # ---- PERMISSIVE label-free ship at TRUE K=0 (2026-06-09) ----------------------
    # Policy (user, feedback_budget_aware_ship_policy): at a zero-label budget the VLM
    # CANNOT verify (it saturates to ~0.5 "can't tell" on a hard 3D contrast shift), so a
    # VLM hard-veto ships NOTHING even when a REAL label-free gain exists. The chosen policy
    # is: SHIP the best PLAUSIBLE label-free candidate + an honest UNVERIFIED stamp (core I3
    # forces trust=label_free_estimate; Harness._disclaim stamps it). This helper ranks the
    # already-built label-free candidates by a LABEL-FREE confidence/coherence proxy (NOT the
    # saturated VLM) so the gate can (a) only permit the TOP-ranked candidate to ship and
    # (b) require it to beat source on that proxy — which keeps the non-recoverable shift
    # (no candidate beats source's proxy) on keep_source.  K=0 + 3D ONLY.
    def _labelfree_proxy(self, s):
        """Label-free confidence/coherence score in [0,1] (higher = trust more). Reuses the
        dimension-agnostic _quality_score (low foreground-prediction entropy + plausible fg
        fraction vs fg_prior + low fragmentation) — the exact signal we have at K=0 with no
        VLM and no labels."""
        return _quality_score(s, self.fg_prior)

    def _labelfree_rank(self):
        """Rank every STRUCTURALLY-PLAUSIBLE built candidate (source excluded) by the
        label-free proxy. Returns (best_name, best_score, source_score) or (None, .., ..)
        if nothing plausible was built. A candidate that collapsed/exploded/scattered
        (the tta_aug 0-Dice trap) is dropped here BEFORE it can be ranked best — the
        structural floor is the real K=0 safety floor."""
        src_score = self._labelfree_proxy(self._src_stats) if self._src_stats else 0.0
        scored = []
        for b in self.built:
            if b["name"] == "source" or b["model"] is self.source_model:
                continue
            s = b.get("stats") or self._stats_of(b["model"])
            ok, _why = self._structural_plausibility(b["model"], s)
            if not ok:
                continue                       # drop degenerate (incl. tta_aug all-bg collapse)
            scored.append((b["name"], self._labelfree_proxy(s)))
        if not scored:
            return None, None, src_score
        best_name, best_score = max(scored, key=lambda t: t[1])
        return best_name, best_score, src_score

    def safety_check(self, candidate, source, evidence):
        src = self._src_stats
        if src is None:
            return SafetyVerdict(False, "not deployed yet — no source baseline")
        if candidate is source:
            return SafetyVerdict(True, "source baseline")
        rec = next((b for b in self.built if b["model"] is candidate), None)
        name = rec["name"] if rec else None
        # (1) LABELS ARE GROUND TRUTH: labeled-Dice gate runs FIRST when revealed.
        if self._labeled_dice is not None and name in self._labeled_dice:
            cd = self._labeled_dice[name]; sd = self._labeled_dice.get("source", 0.0)
            margin = self.safety_margin(sd)
            if cd < sd + margin:
                return SafetyVerdict(False, f"labeled Dice {cd} not clearly > source {sd} "
                                            f"(needs +{margin:.3f}; few-label noise guard)")
            return SafetyVerdict(True, f"labeled Dice {cd} clearly > source {sd} (label-verified)")
        # (2) NO labels: label-free degenerate guard (vs EXPECTED anatomy fraction fg_prior).
        # This is the REAL K=0 safety FLOOR — never ship an empty / exploded / scattered mask.
        s = self._stats_of(candidate)
        if s["fg_frac"] < 0.2 * self.fg_prior:
            return SafetyVerdict(False, "candidate collapsed foreground (degenerate)")
        if s["fg_frac"] > 4.0 * self.fg_prior:
            return SafetyVerdict(False, "candidate exploded foreground (degenerate)")
        if os.environ.get("PLAUSIBILITY_GATE", "0") != "0":
            ok, why = self._structural_plausibility(candidate, s)
            if not ok:
                return SafetyVerdict(False, f"implausible output (structural pre-gate): {why}")

        # (3) TRUE K=0 (no labels, no VLM verification possible): PERMISSIVE label-free ship.
        # The user's chosen policy (feedback_budget_aware_ship_policy): at a zero-label budget
        # the VLM cannot verify, so we DO NOT hard-veto on an indecisive VLM; instead we ship
        # the BEST PLAUSIBLE label-free candidate ranked by a label-free proxy (entropy/
        # coherence), with the UNVERIFIED stamp added downstream (core I3 + Harness._disclaim).
        # Two label-free guards keep this honest and prevent shipping a WORSE-than-source mask:
        #   (a) the offered candidate must be the TOP-ranked plausible label-free candidate, and
        #   (b) its label-free proxy must beat source's proxy by a noise margin — so on a
        #       NON-recoverable shift (no candidate beats source's proxy) we keep_source.
        # The structural floor above already dropped any collapsed/exploded/scattered mask
        # (e.g. the tta_aug all-background 0-Dice trap), which is the genuine K=0 risk.
        if self.labeled_k == 0 and self._labeled_dice is None:
            margin = self.safety_margin(self._labelfree_proxy(src))
            cand_score = self._labelfree_proxy(s)
            best_name, best_score, src_score = self._labelfree_rank()
            # (3a) only the top-ranked plausible label-free candidate may ship
            if best_name is not None and name is not None and name != best_name \
                    and cand_score < best_score - 1e-6:
                return SafetyVerdict(False, f"not the best label-free candidate at K=0 "
                                            f"('{best_name}' proxy {best_score:.3f} > "
                                            f"'{name}' {cand_score:.3f}); ship '{best_name}'")
            # (3b) require a label-free improvement over source (else keep_source is correct)
            if cand_score < src_score + margin:
                return SafetyVerdict(False, f"label-free proxy {cand_score:.3f} does not beat "
                                            f"source {src_score:.3f} by {margin:.3f} at K=0 — "
                                            f"no proven label-free gain, keep_source")
            # (3c) CONFIDENT-WORSE VLM still vetoes; an INDECISIVE VLM (~0.5) is permissive-ship.
            # We consult the VLM as a veto-only signal: it can REJECT a clearly-worse candidate
            # but cannot BLOCK a permissive ship when it simply cannot tell (the K=0 reality).
            if self.vlm is not None:
                prefer = rec["vlm_prefer"] if rec and rec.get("vlm_prefer") is not None \
                    else self._vlm_of(candidate)
                if prefer is not None and prefer < 0.35:
                    return SafetyVerdict(False, f"VLM confidently prefers source over candidate "
                                                f"(prefer {prefer:.2f} < 0.35)")
            return SafetyVerdict(True, f"PERMISSIVE K=0 ship: best plausible label-free candidate "
                                       f"'{best_name or name}' (proxy {cand_score:.3f} > source "
                                       f"{src_score:.3f}); UNVERIFIED label-free estimate")

        # (4) K>0 with VLM but no labeled_dice yet (e.g. NO_FINETUNE pre-reveal): keep the
        # original CONFIDENT-VLM gate — labels are affordable so a marginal ship must verify.
        if self.vlm is not None:
            prefer = rec["vlm_prefer"] if rec and rec.get("vlm_prefer") is not None \
                else self._vlm_of(candidate)
            if prefer is not None and prefer < self.VLM_PREFER_THRESH:
                return SafetyVerdict(False, f"VLM prefers source over candidate "
                                            f"(prefer {prefer:.2f} < {self.VLM_PREFER_THRESH})")
        if _quality_score(s, self.fg_prior) < _quality_score(src, self.fg_prior) - 0.05:
            return SafetyVerdict(False, "label-free quality below source")
        return SafetyVerdict(True, "passes safety vs source")
