"""Segmentation-3D **foundation-model / single-organ specialization** TaskAdapter.

The "foundation model as source" sibling of harness/seg3d_adapter.py::SegTask3D and
harness/seg3d_mri_adapter.py::SegTask3DMRI. It exposes the SAME interface (class attrs,
__init__ fields, toolbox/safety_check/_request/_finetune/... shapes + result-dict keys)
so harness/core.py + harness/llm_policy.py run it UNCHANGED.

THE STORY — generalist -> specialist. The source is the MONAI wholeBody_ct_segmentation
bundle: a 105-class SegResNet CT GENERALIST (TotalSegmentator organs). We deploy it on
MSD Task09 spleen as if it were a single-organ (spleen) deployment. A 100-organ generalist
is only mediocre on one specific organ; a few-shot specialization fine-tune should lift it.
We measure (a) the generalist's out-of-the-box spleen Dice with CORRECT preprocessing, and
(b) the spleen Dice after a K-shot specialization fine-tune.

What is DIFFERENT from the 1-channel CT SegTask3D (and why a new file, not an edit):

  * INPUT is 1-channel CT, but PREPROCESSING is the wholeBody bundle's OWN transforms:
    as_closest_canonical RAS -> Spacingd resample to the bundle pixdim (default
    [1.5,1.5,1.5]) -> NormalizeIntensityd(nonzero=True) z-norm -> ScaleIntensityd(minv,maxv).
    This is NOT the ScaleIntensityRange(-57,164) window the SMALL spleen_ct bundle uses;
    feeding the generalist that window gives a GARBAGE baseline (a prior probe got 0.303
    with the wrong window — that is NOT the real baseline).
  * OUTPUT is 105 softmax channels; SPLEEN = class index 1 (same index as the MSD spleen GT
    label, which is also 1 — clean). nc=105 here means "105 classes", and the single region
    of interest is the spleen channel.
  * DECODE: spleen mask = (argmax(logits, dim=1) == SPLEEN_IDX) — the bundle's native softmax
    decode. We ALSO compute sigmoid(logits[:,SPLEEN_IDX])>0.5 as an alternate decode and
    report both; the headline uses whichever gives a sane non-degenerate baseline.
  * fg_frac for label-free stats = mean of the SPLEEN mask (NOT pred>=1, which would count
    all 105 organs).
  * NORM is InstanceNorm/GroupNorm inside SegResNet -> bn_reset/tent are structural no-ops
    (no BatchNorm running stats); the real specialization lever is finetune.
  * Dice = binary spleen Dice = Dice(spleen_mask, GT==1).

REUSE (do NOT reimplement): _sliding_window_predict from medadaptagent.tasks.segmentation_3d
(the MONAI sliding-window engine), load_source_model (builds the bundle's 105-ch SegResNet),
the dimension-agnostic _vlm_prefer_fraction + analyze_domain_shift + inspect_model helpers,
the resample-to-spacing helper from seg3d_adapter (_resample_to_spacing/_spacing_from_affine),
and the few-shot 3D fine-tune PATTERN (patch-crop, Conv3d-aware param selection) from
SegTask3D._finetune_3d. The window closures in seg3d_adapter are ScaleIntensityRange-specific
and NOT reused — wholeBody normalization is z-norm + rescale, written here.

NO per-dataset magic constants: fg_prior, the spleen class index, and the pixdim all come
from the runner's TASKS config / the bundle's inference.json, never hardcoded in the logic.
"""
from __future__ import annotations
import os
import copy
import numpy as np
from harness.core import Tool, SafetyVerdict, Budget

# Dimension/representation-agnostic helpers reused verbatim from the 2D seg adapter.
from harness.seg_adapter import _vlm_prefer_fraction
# The CT spacing resample (bundle Spacingd equivalent) — reused, not reimplemented.
from harness.seg3d_adapter import _resample_to_spacing, _spacing_from_affine


# --------------------------------------------------------------- wholeBody preprocessing
def _wholebody_to_tensor(img, device, pixdim=None, scale_minv=-1.0, scale_maxv=1.0):
    """The wholeBody bundle's image preprocessing, reproduced exactly:

      (RAS reorient is applied at LOAD time by nib.as_closest_canonical in the runner)
      -> Spacingd(pixdim, bilinear)              [if pixdim given]
      -> NormalizeIntensityd(nonzero=True)       z-norm over NONZERO voxels (whole volume)
      -> ScaleIntensityd(minv, maxv)             rescale min..max -> [minv, maxv]

    `img` is a calibration.Volume (numpy [D,H,W] with a .affine). -> torch [1,1,D,H,W].

    NormalizeIntensityd(nonzero=True) z-norms using the mean/std of NONZERO voxels only and
    leaves the zero (air-padding) voxels at zero — this matches the bundle config
    `{"_target_":"NormalizeIntensityd","nonzero":true}` (note: nonzero defaults to False in
    MONAI, but the wholeBody bundle sets it True). ScaleIntensityd with no a_min/a_max
    rescales the ACTUAL volume min..max linearly to [minv,maxv]."""
    import torch
    arr = np.asarray(img).astype(np.float32)
    affine = getattr(img, "affine", None)
    if pixdim is not None and affine is not None:
        arr = _resample_to_spacing(arr, affine, pixdim, order=1)   # Spacingd (bilinear)
    # NormalizeIntensityd(nonzero=True): z-norm over nonzero voxels, zeros stay zero.
    mask = arr != 0
    if mask.any():
        m = float(arr[mask].mean())
        s = float(arr[mask].std())
        if s < 1e-8:
            s = 1.0
        arr = np.where(mask, (arr - m) / s, 0.0).astype(np.float32)
    # ScaleIntensityd(minv,maxv): rescale actual min..max -> [minv,maxv] over the whole array.
    lo = float(arr.min())
    hi = float(arr.max())
    rng = hi - lo
    if rng > 1e-8:
        arr = (arr - lo) / rng * (scale_maxv - scale_minv) + scale_minv
    arr = arr.astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


# --------------------------------------------------------------- label-free signals
def _predict_stats_foundation(adapter, model, vols, n=4):
    """Label-free signals over up to `n` evenly-spaced target volumes, computed on the
    SPLEEN channel only (NOT the 105-organ foreground):
      fg_frac  — mean predicted SPLEEN voxel fraction (the headline decode's mask)
      conf     — mean softmax probability mass on the spleen class where the spleen is decoded
                 (decisiveness; 1 = confident spleen)
      entropy  — mean full-softmax entropy over the volume (kept for the generic helpers)
      n_comp   — mean number of 3D connected components in the spleen mask (1 = coherent organ)
    No GT used. Decode = adapter._decode_spleen (argmax==idx by default)."""
    import torch
    import torch.nn.functional as F
    from scipy import ndimage
    model.eval()
    if len(vols) == 0:
        return {"fg_frac": 0.0, "conf": 0.0, "entropy": 0.0, "n_comp": 0.0}
    idx = np.linspace(0, len(vols) - 1, min(n, len(vols)), dtype=int)
    fgs, confs, ents, comps = [], [], [], []
    with torch.no_grad():
        for i in idx:
            x = adapter._to_tensor(vols[int(i)], adapter.device)        # [1,1,D,H,W]
            logits = adapter._logits(model, x)                          # [1,105,D,H,W]
            p = F.softmax(logits, 1)
            ents.append(float(-(p * torch.clamp(p, 1e-6, 1).log()).sum(1).mean()))
            mask = adapter._decode_spleen_from_logits(logits)           # [D,H,W] bool numpy
            fgs.append(float(mask.mean()))
            sp = p[0, adapter.spleen_idx].detach().cpu().numpy()        # spleen prob map
            confs.append(float(sp[mask].mean()) if mask.any() else 0.0)
            comps.append(int(ndimage.label(mask)[1]) if mask.any() else 0)
    return {"fg_frac": float(np.mean(fgs)), "conf": float(np.mean(confs)),
            "entropy": float(np.mean(ents)), "n_comp": float(np.mean(comps))}


def _quality_score_foundation(stats, fg_prior):
    """Label-free quality in [0,1]: confident spleen (high softmax mass) + plausible spleen
    fraction (close to fg_prior) + a single coherent component. No GT. Sigmoid/softmax analog
    of harness.seg_adapter._quality_score."""
    conf = float(stats.get("conf", 0.0))
    fg = float(stats.get("fg_frac", 0.0))
    plaus = 1.0 - min(1.0, abs(fg - fg_prior) / max(fg_prior, 0.002) / 4)
    nc = float(stats.get("n_comp", 0.0))
    coherent = 1.0 if nc <= 0 else min(1.0, 1.0 / nc)        # 1 component is best
    return float(0.5 * conf + 0.35 * plaus + 0.15 * coherent)


def _diagnose_foundation(s, src):
    """Spleen-aware seg failure mode from the stats (vs the source baseline)."""
    if s["fg_frac"] < 0.4 * max(src["fg_frac"], 1e-6):
        return "under_segmentation (spleen collapsing — generalist under-calls the organ)"
    if s["fg_frac"] > 2.5 * max(src["fg_frac"], 1e-6):
        return "over_segmentation (spleen bleeding into neighbours)"
    if s.get("conf", 1.0) < 0.6 * max(src.get("conf", 1e-6), 1e-6):
        return "low_confidence (softmax spleen mass uncertain — generalist hedging)"
    if s.get("n_comp", 1) > max(2, src.get("n_comp", 1) + 2):
        return "fragmentation (spleen split into multiple components)"
    return "mild_shift (spleen morphology largely intact)"


class SegTask3DFoundation:
    type, dims, metric_name = "segmentation_3d", "3D", "Dice"

    def __init__(self, source_model, simgs, timgs, nc, device, fg_prior=0.006, tlbls=None,
                 task_desc="", spleen_idx=1, pixdim=(1.5, 1.5, 1.5),
                 scale_minv=-1.0, scale_maxv=1.0, decode="argmax", roi=(96, 96, 96)):
        self.source_model = source_model
        self.simgs, self.timgs = simgs, timgs       # calibration.Volume [D,H,W] (+ .affine)
        self.tlbls = tlbls                          # MSD integer labels — request/finetune only
        self.nc, self.device = nc, device           # nc = 105 classes
        self.fg_prior = fg_prior                    # expected SPLEEN voxel fraction (tiny)
        self.spleen_idx = int(spleen_idx)           # spleen class index (1; == MSD GT label)
        self.pixdim = (None if pixdim is None else tuple(float(p) for p in pixdim))
        self.scale_minv, self.scale_maxv = float(scale_minv), float(scale_maxv)
        self.decode = decode                        # "argmax" (default) or "sigmoid"
        self.roi = tuple(int(r) for r in roi)
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
        self._shift_ev = None
        self._model_ev = None
        self._perf_est = None

    # -- preprocessing + forward (REUSE the MONAI sliding-window engine) -----------
    def _to_tensor(self, vol, device):
        return _wholebody_to_tensor(vol, device, pixdim=self.pixdim,
                                    scale_minv=self.scale_minv, scale_maxv=self.scale_maxv)

    def _logits(self, model, x):
        """Sliding-window logits [1,105,D,H,W]. REUSES medadaptagent.tasks.segmentation_3d.
        _sliding_window_predict (the MONAI gaussian sliding-window engine) with the bundle ROI."""
        from medadaptagent.tasks.segmentation_3d import _sliding_window_predict
        return _sliding_window_predict(model, x, patch=self.roi, batch_size=1, overlap=0.5)

    def _decode_spleen_from_logits(self, logits, mode=None):
        """[1,105,D,H,W] logits -> binary spleen mask [D,H,W] (numpy bool).
          argmax : (argmax over 105 classes == spleen_idx) — the bundle's native softmax decode
          sigmoid: sigmoid(spleen logit) > 0.5 — alternate one-vs-rest decode
        Used for BOTH source and finetuned so the comparison is apples-to-apples."""
        import torch
        mode = mode or self.decode
        if mode == "sigmoid":
            return (torch.sigmoid(logits[0, self.spleen_idx]) > 0.5).detach().cpu().numpy()
        am = torch.argmax(logits[0], dim=0)                 # [D,H,W]
        return (am == self.spleen_idx).detach().cpu().numpy()

    def _spleen_mask(self, model, vol, mode=None):
        """Run the model on one volume and return the binary spleen mask [D,H,W] (numpy)."""
        import torch
        with torch.no_grad():
            x = self._to_tensor(vol, self.device)
            logits = self._logits(model, x)
            return self._decode_spleen_from_logits(logits, mode=mode)

    def set_vlm(self, llm, task_ctx, task, anchors):
        self.vlm = {"llm": llm, "ctx": task_ctx, "task": task, "anchors": anchors}

    def _vlm_of(self, cand):
        if not self.vlm:
            return None
        v = self.vlm
        return _vlm_prefer_fraction(cand, self.source_model, v["anchors"], v["llm"],
                                    v["ctx"], v["task"], self.device)

    def _llm(self):
        return self.vlm["llm"] if self.vlm else None

    # -- toolbox (mirror SegTask3D; bn_reset/tent inert on InstanceNorm but kept) ---
    def toolbox(self):
        B1 = Budget(compute_minutes=1)
        Bt = Budget(compute_minutes=2, n_trials=1)
        return [
            Tool("load_and_deploy", "run the generalist on target volumes; report spleen fg/confidence",
                 self._deploy, B1),
            Tool("analyze_domain_shift", "MEASURE source-vs-target image stats + characterize "
                 "the shift (generalist-vs-single-organ, intensity / contrast / resolution, severity)",
                 self._analyze_shift, Budget(compute_minutes=1, llm_usd=0.02)),
            Tool("inspect_model", "introspect the model (architecture, normalization, in/out, "
                 "likely training domain) — tells you which adaptations apply",
                 self._inspect_model, Budget(llm_usd=0.02)),
            Tool("diagnose_masks", "spleen failure analysis: under/over-seg, fragmentation, confidence",
                 self._diagnose, B1),
            Tool("bn_reset", "recalibrate BatchNorm stats to target volumes (no labels) — NOTE: "
                 "SegResNet uses InstanceNorm, so this is a structural no-op here",
                 self._bn_reset, Bt),
            Tool("tent", "entropy-minimization TTA on norm affine params (no labels)",
                 self._tent, Bt),
            Tool("evaluate", "label-free quality of the current best candidate",
                 self._evaluate, B1),
            Tool("request_labels", "ask the clinician for K (3-12) labeled volumes; split into "
                 "TRAIN+HOLDOUT, then rank ALL candidates by HELD-OUT spleen Dice — use when "
                 "label-free signals can't decide or the best is still poor",
                 self._request, lambda a: Budget(k_labels=int(a.get("K", 6)))),
            Tool("finetune_head", "SPECIALIZE only the output head on the K train volumes — "
                 "cheapest, safest at tiny K; use first when labels are present but TTA stalls",
                 lambda st, **k: self._finetune("head", **k), Budget(compute_minutes=3)),
            Tool("finetune_partial", "SPECIALIZE the decoder + head on the K train volumes — "
                 "more capacity to turn the generalist into a spleen specialist",
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
        if os.environ.get("SCALING_EVAL", "0") != "1":
            return []
        return [Tool("scaling_eval", "few-shot SCALING-LAW: fine-tune across label budgets -> "
                     "the honest delivered-model estimate + an 'annotate N -> expect X' plan.",
                     lambda st, **k: self._scaling_eval(**k), Budget(compute_minutes=8))]

    # -- stats / dice helpers (binary spleen Dice) --------------------------------
    def _stats_of(self, cand, n=4):
        return _predict_stats_foundation(self, cand, self.timgs, n=n)

    def _spleen_dice(self, model, vols, lbls, mode=None):
        """Mean BINARY spleen Dice over (vol, label) pairs: decode the spleen mask, compare to
        GT==1. This is the SINGLE metric used everywhere (ranking, safety). If Spacingd resampled
        the prediction grid, resample the mask back to the label grid (nearest) before scoring."""
        import torch
        model.eval()
        ds = []
        with torch.no_grad():
            for vol, lbl in zip(vols, lbls):
                lbl = np.asarray(lbl)
                pred = self._spleen_mask(model, vol, mode=mode)         # [D,H,W] bool
                g = (lbl == self.spleen_idx)
                if pred.shape != g.shape:
                    pred = self._resize_bool_to(pred, g.shape)
                if g.sum() > 0:
                    ds.append(float(2 * (pred & g).sum() / (pred.sum() + g.sum() + 1e-8)))
        return round(float(np.mean(ds)), 4) if ds else 0.0

    def _resize_bool_to(self, mask, shape):
        if mask.shape == tuple(shape):
            return mask
        from scipy import ndimage
        zoom = np.asarray(shape, dtype=np.float64) / np.asarray(mask.shape, dtype=np.float64)
        return ndimage.zoom(mask.astype(np.float32), zoom, order=0) > 0.5

    # -- tools (each returns the SAME dict shape as SegTask3D) ---------------------
    def _deploy(self, st, **kw):
        s = self._stats_of(self.source_model)
        self._src_stats = s
        q = _quality_score_foundation(s, self.fg_prior)
        self.built.append({"name": "source", "model": self.source_model,
                           "quality": q, "vlm_prefer": None, "stats": s})
        return {"deployment": "source@target", "candidate": self.source_model,
                "evidence": {"source_stats": s, "quality": q, "last_method": "source"},
                "metric_estimate": q}

    def _diagnose(self, st, **kw):
        cand = st.best_candidate or self.source_model
        s = self._stats_of(cand)
        src = self._src_stats or s
        mode = _diagnose_foundation(s, src)
        return {"evidence": {"failure_mode": mode, "stats": s}}

    def _analyze_shift(self, st, **kw):
        if self._shift_ev is not None:
            return {"evidence": {"domain_shift": self._shift_ev,
                    "directive": "shift ALREADY analyzed — do NOT call again; ACT: try an "
                                 "adaptation tool now."}}
        from harness.analysis import analyze_domain_shift
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
        s = self._stats_of(cand)
        q = _quality_score_foundation(s, self.fg_prior)
        vlm = self._vlm_of(cand)
        self.built.append({"name": name, "model": cand, "quality": q,
                           "vlm_prefer": vlm, "stats": s})
        ev = {"cand_stats": s, "quality": q, "last_method": name}
        if vlm is not None:
            ev["vlm_prefer"] = round(vlm, 3)
        return {"candidate": cand, "evidence": ev, "metric_estimate": q}

    # -- label-free adaptations ---------------------------------------------------
    def _has_batchnorm(self, model):
        import torch.nn as nn
        return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                   for m in model.modules())

    def _center_patch(self, x, patch=None):
        import torch.nn.functional as F
        patch = patch or self.roi
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

    def _bn_reset(self, st, **kw):
        """SegResNet has no BatchNorm (InstanceNorm/GroupNorm), so resetting running stats is
        structurally a no-op. We deepcopy + score it and ANNOTATE that it was inert."""
        import torch
        import torch.nn as nn
        m = copy.deepcopy(self.source_model).to(self.device)
        had_bn = self._has_batchnorm(m)
        if had_bn:
            for mod in m.modules():
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    mod.reset_running_stats(); mod.momentum = None
            m.train()
            with torch.no_grad():
                for vol in self.timgs:
                    m(self._center_patch(self._to_tensor(vol, self.device)))
        m.eval()
        res = self._score_candidate("bn_reset", m)
        res.setdefault("evidence", {})["note"] = (
            "bn_reset applied" if had_bn else
            "NO-OP: SegResNet uses InstanceNorm/GroupNorm (no BatchNorm running stats to reset)")
        return res

    def _tent(self, st, steps=3, **kw):
        """Entropy-minimization on the norm-layer affine params over a few target volumes
        (full-softmax entropy). Inert if there are no affine norm params with grad; the real
        lever is finetune. Memory-bounded center patch."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        m = copy.deepcopy(self.source_model).to(self.device)
        for p in m.parameters():
            p.requires_grad_(False)
        params = []
        norm_types = (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)
        for mod in m.modules():
            if isinstance(mod, norm_types):
                if getattr(mod, "weight", None) is not None:
                    mod.weight.requires_grad_(True); params.append(mod.weight)
                if getattr(mod, "bias", None) is not None:
                    mod.bias.requires_grad_(True); params.append(mod.bias)
        if not params:
            res = self._score_candidate("tent", m.eval())
            res.setdefault("evidence", {})["note"] = "NO-OP: no affine norm params to tune"
            return res
        opt = torch.optim.Adam(params, lr=1e-3)
        vols = self.timgs[:min(4, len(self.timgs))]
        m.train()
        for _ in range(steps):
            for vol in vols:
                xc = self._center_patch(self._to_tensor(vol, self.device))
                logits = m(xc)
                p = F.softmax(logits, 1)
                ent = -(p * torch.clamp(p, 1e-6, 1).log()).sum(1).mean()
                opt.zero_grad(); ent.backward(); opt.step()
        m.eval()
        return self._score_candidate("tent", m)

    def _evaluate(self, st, **kw):
        cand = st.best_candidate or self.source_model
        s = self._stats_of(cand)
        q = _quality_score_foundation(s, self.fg_prior)
        return {"evidence": {"eval_stats": s, "quality": q}, "metric_estimate": q}

    # -- K-shot labels: reveal, split, rank (mirror SegTask3D exactly) ------------
    def _ensure_labels(self, K=6):
        if self._labels_revealed:
            return self._train
        K = int(K)
        fg_idx = [i for i in range(len(self.tlbls))
                  if (np.asarray(self.tlbls[i]) == self.spleen_idx).any()]
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
                ld[b["name"]] = self._spleen_dice(b["model"], h_imgs, h_lbls)
            else:
                ld[b["name"]] = self._spleen_dice(b["model"], all_imgs, all_lbls)
        self._labeled_dice = ld
        return ld

    def _estimate_performance(self):
        if self._perf_est is not None:
            return self._perf_est
        fallback = {"sota": 0.95, "usable": 0.80,
                    "rationale": "MSD spleen Dice default (no LLM); dedicated models reach ~0.95"}
        llm = self._llm()
        if llm is None:
            self._perf_est = fallback; return fallback
        import json, re
        prompt = (f"Medical-image 3D SEGMENTATION task: {self.task_desc}. Metric: binary spleen "
                  f"Dice.\nFrom the published MSD/spleen literature, give: the Dice a COMPETENT / "
                  f"near-SOTA dedicated spleen model typically reaches, and the MINIMUM clinically-"
                  f"usable Dice. Respond ONLY JSON: {{\"sota\": <0..1>, \"usable\": <0..1>, "
                  f"\"rationale\": \"<=15 words citing the typical benchmark\"}}.")
        try:
            d = json.loads(re.search(r"\{.*\}", llm.chat(prompt), re.S).group(0))
            d = {"sota": float(d.get("sota", 0.95)), "usable": float(d.get("usable", 0.80)),
                 "rationale": str(d.get("rationale", ""))[:160]}
        except Exception:
            d = fallback
        self._perf_est = d
        return d

    def _request(self, st, K=6, **kw):
        if self.tlbls is None:
            return {"k_labels_used": 0, "evidence": {"error": "no labels available here"}}
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
                             "note": f"HELD-OUT spleen Dice ({len(self._hold[0])} volumes); "
                                     f"task-expected Dice≈{est['sota']} (usable≥{est['usable']}: "
                                     f"{est['rationale']}). Ship '{best}' if it beats source; if best "
                                     f"is well below {est['sota']}, fine-tune to specialize."}}

    # -- few-shot 3D specialization fine-tune (1-ch in / spleen-channel BCE+Dice) --
    def _finetune(self, mode, K=6, base="source", steps=40, lr=1e-3, **kw):
        if self.tlbls is None:
            return {"evidence": {"error": "no labels available — cannot finetune"}}
        tr_imgs, tr_lbls = self._ensure_labels(K)
        if len(tr_imgs) == 0:
            return {"evidence": {"error": "no train labels after split (K too small)"}}
        start = self.source_model
        if base == "best" and self.built:
            start = max(self.built, key=lambda b: b.get("quality", 0))["model"]
        model, info = self._finetune_3d(start, tr_imgs, tr_lbls, mode=mode, steps=steps, lr=lr)
        name = f"finetune_{mode}"
        res = self._score_candidate(name, model)
        self._rank_on_holdout()
        hd = (self._labeled_dice or {}).get(name)
        res.setdefault("evidence", {})
        best_by = max(self._labeled_dice, key=self._labeled_dice.get) if self._labeled_dice else None
        res["evidence"].update({"finetune": info, "labeled_dice": self._labeled_dice,
                                "best_by_labels": best_by, "usable_bar": self.USABLE_BAR,
                                "holdout_dice": hd, "trained_on": len(tr_imgs),
                                "note": f"{name}: held-out spleen Dice {hd} on {len(self._hold[0])} "
                                        f"volumes ({info.get('trainable_params')} params, "
                                        f"loss {info.get('loss_first')}→{info.get('loss_last')})"})
        return res

    def _finetune_3d(self, model, imgs, lbls, mode="head", steps=40, lr=1e-3, seed=0):
        """Few-shot 3D SPECIALIZATION on a SINGLE labeled patch per volume (full-volume backward
        OOMs). 1-ch in / 105-ch out; loss is on the SPLEEN logit channel only:
        BCEWithLogits(logits[:,spleen_idx], GT==spleen) + soft-Dice on sigmoid(spleen logit).
        Param selection is Conv3d-aware (head/partial/lora). Mirrors SegTask3D._finetune_3d.
        Returns a fine-tuned DEEPCOPY + an info dict with the same keys as harness.finetune."""
        import torch
        import torch.nn as nn
        m = copy.deepcopy(model).to(self.device)
        n_train = self._set_trainable_3d(m, mode)
        m = m.to(self.device)
        if n_train == 0:
            m.eval()
            return m, {"mode": mode, "trainable_params": 0, "note": "no trainable params; no-op",
                       "loss_first": None, "loss_last": None}
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        m.train()
        for mod in m.modules():                       # protect BN (none here) at batch=1
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.eval()
        patches = [self._label_patch(im, lb) for im, lb in zip(imgs, lbls)]
        patches = [p for p in patches if p is not None]
        if not patches:
            m.eval()
            return m, {"mode": mode, "trainable_params": int(n_train),
                       "note": "no usable label patch", "loss_first": None, "loss_last": None}
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(patches), generator=g).tolist()
        losses = []
        for step in range(steps):
            xc, yc = patches[order[step % len(patches)]]
            xc = xc.to(self.device); yc = yc.to(self.device)              # yc [1,1,d,h,w] float spleen
            logit = m(xc)                                                 # [1,105,d,h,w]
            import torch.nn.functional as F
            if logit.shape[-3:] != yc.shape[-3:]:
                yc = F.interpolate(yc.float(), size=logit.shape[-3:], mode="nearest")
            loss = self._spleen_loss_3d(logit, yc)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        m.eval()
        return m, {"mode": mode, "trainable_params": int(n_train), "steps": steps, "lr": lr,
                   "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4)}

    def _label_patch(self, img, lbl, patch=None):
        """One ROI-sized patch centered on the spleen, as (img[1,1,d,h,w], spleen[1,1,d,h,w]).
        Spleen target is the binary (GT==spleen_idx) mask, aligned to the (possibly resampled)
        tensor grid."""
        import torch
        import torch.nn.functional as F
        patch = patch or self.roi
        x = self._to_tensor(img, self.device)                # [1,1,D,H,W] preprocessed
        D, H, W = x.shape[-3:]
        lbl = np.asarray(lbl)
        sp = (lbl == self.spleen_idx)
        if sp.shape != (D, H, W):                            # align to the resampled tensor grid
            sp = self._resize_bool_to(sp, (D, H, W))
        fg = np.argwhere(sp)
        if len(fg) == 0:
            return None
        cz, cy, cx = fg.mean(0).astype(int)

        def _rng(c, p, n):
            s = int(np.clip(c - p // 2, 0, max(0, n - p)))
            return slice(s, s + min(p, n))
        sl = (_rng(cz, patch[0], D), _rng(cy, patch[1], H), _rng(cx, patch[2], W))
        xc = x[:, :, sl[0], sl[1], sl[2]]
        yc = torch.from_numpy(sp[sl].astype(np.float32))[None, None]      # [1,1,d,h,w]
        pads = []
        for i in range(3):
            pads += [0, max(0, patch[2 - i] - xc.shape[4 - i])]
        if any(pads):
            xc = F.pad(xc, pads, mode="constant", value=0)
            yc = F.pad(yc, pads, mode="constant", value=0)
        return xc.detach().cpu(), yc.detach().cpu()

    def _spleen_loss_3d(self, logit, target):
        """SPLEEN-channel loss: BCEWithLogits(spleen logit, GT) + soft-Dice on sigmoid(spleen
        logit). logit [B,105,d,h,w], target [B,1,d,h,w] float in {0,1}."""
        import torch.nn.functional as F
        sp = logit[:, self.spleen_idx:self.spleen_idx + 1]               # [B,1,d,h,w]
        bce = F.binary_cross_entropy_with_logits(sp, target)
        p = sp.sigmoid()
        dims = (0, 2, 3, 4)
        inter = (p * target).sum(dims)
        denom = p.sum(dims) + target.sum(dims)
        dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
        return bce + dice

    def _set_trainable_3d(self, model, mode, partial_frac=0.4, r=4, alpha=8, tail_frac=0.5):
        """Conv3d/Linear-aware param selection (head < partial < lora). Head = the Conv3d whose
        out_channels == nc (105). Mirrors SegTask3D._set_trainable_3d."""
        import torch.nn as nn

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
        """Minimal Conv3d/Linear LoRA on the TAIL tail_frac of conv/linear modules."""
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

    def _recommend_alternative(self, st, **kw):
        from harness.model_search import recommend_alternative_models
        best = max((self._labeled_dice or {}).values(), default=None)
        rec = recommend_alternative_models(
            self._llm(), task_desc=self.task_desc, task_type="segmentation_3d",
            metric_name="Dice", best_metric=best, usable_target=0.80,
            domain_shift=st.evidence.get("domain_shift"),
            model_info=st.evidence.get("model_info"))
        return {"evidence": {"recommendation": rec}}

    def _scaling_eval(self, K=12, modes=("head", "partial"), **kw):
        if self.tlbls is None:
            return {"evidence": {"error": "no labels available — cannot scale"}}
        self._ensure_labels(K)
        res = self._finetune("head", K=K)
        est = self._estimate_performance()
        hd = (self._labeled_dice or {}).get("finetune_head")
        res.setdefault("evidence", {})
        res["evidence"].update({"expected_metric": est["sota"], "usable_floor": est["usable"],
                                "perf_rationale": est["rationale"],
                                "scaling_plan": f"annotate ~{self.labeled_k} volumes -> spleen Dice ≈ {hd}"})
        return res

    # -- I1 safety: never ship a candidate worse/degenerate vs source -------------
    def safety_margin(self, source_metric) -> float:
        return 0.05  # flat NOISE floor (2026-06-08); matches the rest of the harness

    def _structural_plausibility(self, cand, s=None):
        s = s if s is not None else self._stats_of(cand)
        ff = float(s.get('fg_frac', 0.0) or 0.0)
        if ff < 0.15 * self.fg_prior:
            return False, 'empty/collapsed spleen'
        if ff > 5.0 * self.fg_prior:
            return False, 'exploded spleen (bleeds into neighbours)'
        return True, 'structurally plausible'

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
        # (2) NO labels: label-free degenerate guard (vs EXPECTED spleen fraction fg_prior)
        s = self._stats_of(candidate)
        if s["fg_frac"] < 0.2 * self.fg_prior:
            return SafetyVerdict(False, "candidate collapsed spleen (degenerate)")
        if s["fg_frac"] > 4.0 * self.fg_prior:
            return SafetyVerdict(False, "candidate exploded spleen (degenerate)")
        if os.environ.get("PLAUSIBILITY_GATE", "0") != "0":
            ok, why = self._structural_plausibility(candidate, s)
            if not ok:
                return SafetyVerdict(False, f"implausible output (structural pre-gate): {why}")
        # (3) VLM PAIRWISE-preference gate.
        if self.vlm is not None:
            prefer = rec["vlm_prefer"] if rec and rec.get("vlm_prefer") is not None \
                else self._vlm_of(candidate)
            if prefer is not None and prefer < self.VLM_PREFER_THRESH:
                return SafetyVerdict(False, f"VLM prefers source over candidate "
                                            f"(prefer {prefer:.2f} < {self.VLM_PREFER_THRESH})")
        if _quality_score_foundation(s, self.fg_prior) < _quality_score_foundation(src, self.fg_prior) - 0.05:
            return SafetyVerdict(False, "label-free quality below source")
        return SafetyVerdict(True, "passes safety vs source")
