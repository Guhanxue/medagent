"""Segmentation TaskAdapter for the deployment harness.

Wraps the REAL existing seg capabilities (gate code) as the task's own toolbox +
its own failure analysis. The harness core stays generic; this file is the seg
"brain": which diagnosis and which adaptations apply to segmentation.

Label-free (key-free) quality for this first real run = prediction ENTROPY +
foreground-fraction plausibility + fragmentation. (VLM votes get added later as a
second quality signal once the Azure key is wired.) The TRUE Dice is never shown
to the agent — the runner computes it as a hidden oracle to verify the trajectory.
"""
from __future__ import annotations
import os
import copy
import numpy as np
from harness.core import Tool, SafetyVerdict, Budget

# Excluded from the bridged toolbox: multi_scale_tta (pre-existing impl bug); bn_reset/tent
# are provided by the proven hand-written tools below (avoid a duplicate-name collision).
_BRIDGE_SKIP = {"multi_scale_tta", "bn_reset", "tent"}


# --------------------------------------------------------------- label-free signals
def _predict_stats(model, imgs, device, n=24):
    """Mean softmax entropy, mean fg fraction, mean #connected-components (frag)."""
    import torch
    import torch.nn.functional as F
    from scipy import ndimage
    from scripts.run_baseline_tta import _to_tensor
    model.eval()
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs)), dtype=int)
    ents, fgs, comps = [], [], []
    with torch.no_grad():
        for i in idx:
            logit = model(_to_tensor(imgs[int(i)], device))
            p = F.softmax(logit, 1)
            ents.append(float(-(p * torch.clamp(p, 1e-6, 1).log()).sum(1).mean()))
            pred = logit.argmax(1)[0].cpu().numpy()
            fg = (pred >= 1)
            fgs.append(float(fg.mean()))
            comps.append(int(ndimage.label(fg)[1]) if fg.any() else 0)
    return {"entropy": float(np.mean(ents)), "fg_frac": float(np.mean(fgs)),
            "n_comp": float(np.mean(comps))}


def _quality_score(stats, fg_prior=0.05):
    """Label-free quality in [0,1]: confident (low entropy) + plausible fg + not
    fragmented. Higher is better. fg_prior ~ expected target fg fraction."""
    conf = max(0.0, 1.0 - stats["entropy"])
    plaus = 1.0 - min(1.0, abs(stats["fg_frac"] - fg_prior) / max(fg_prior, 0.05) / 4)
    frag = 1.0 / (1.0 + max(0.0, stats["n_comp"] - 2))      # 1-2 comps ~ fine
    return float(0.5 * conf + 0.35 * plaus + 0.15 * frag)


def _vlm_prefer_fraction(model, source, anchors, llm, ctx, task, device):
    """PAIRWISE preference: fraction of anchors where the VLM judges the ADAPTED model
    BETTER than source, among the anchors where it made a decisive better/worse call.

    Uses the RELATIVE votes from _vlm_iteration_vote (per-sample better|similar|worse,
    the forced source-vs-adapted choice) — NOT the absolute usable/broken rating.
    Rationale (user, 2026-06-03): a VLM is more reliable at the A-vs-B CHOICE than at
    absolute 'is this usable' scoring, which saturates to 'broken' for BOTH masks on a
    hard shift (chaos: source 0.12 and bn_reset 0.39 both got absolute usable=0.17,
    losing the real recovery). 0.5 = tie / no decisive comparison."""
    from medadaptagent.graph.agents.execution_engineer import _vlm_iteration_vote
    _score, votes, _note, _abs = _vlm_iteration_vote(
        current_model=model, source_model=source, anchor_imgs=anchors,
        device=device, llm_client=llm, task_ctx=ctx, k=min(6, len(anchors)), task=task)
    if not votes:
        return 0.5
    nb = sum(1 for v in votes if v == "better")
    nw = sum(1 for v in votes if v == "worse")
    return nb / (nb + nw) if (nb + nw) > 0 else 0.5


def _diagnose_from_stats(s, src):
    """Seg-SPECIFIC failure analysis: name the failure mode from mask morphology."""
    if s["fg_frac"] < 0.4 * max(src["fg_frac"], 1e-6):
        mode = "under_segmentation (foreground collapsing)"
    elif s["fg_frac"] > 2.5 * max(src["fg_frac"], 1e-6):
        mode = "over_segmentation (bleeding into background)"
    elif s["n_comp"] > src["n_comp"] + 3:
        mode = "fragmentation (scattered false positives)"
    elif s["entropy"] > 1.4 * max(src["entropy"], 1e-6):
        mode = "low_confidence (calibration shift, masks uncertain)"
    else:
        mode = "mild_shift (morphology largely intact)"
    return mode


class SegTask:
    type, dims, metric_name = "segmentation", "2D", "Dice"

    def __init__(self, source_model, simgs, timgs, nc, device, fg_prior=0.05, tlbls=None,
                 task_desc=""):
        self.source_model = source_model
        self.simgs, self.timgs = simgs, timgs
        self.tlbls = tlbls         # target GT — used ONLY by request_labels (few-shot)
        self.nc, self.device = nc, device
        self.fg_prior = fg_prior
        self.task_desc = task_desc # free-text modality/anatomy/shift, for recommend_alternative
        self.USABLE_BAR = 0.30     # research heuristic; not a clinical acceptance criterion
        self._src_stats = None     # filled by load_and_deploy
        self.built = []            # [{name, model, quality, vlm_prefer, stats}] for audit
        self.vlm = None            # optional {llm, ctx, task, anchors} set via set_vlm
        self.VLM_PREFER_THRESH = 0.66  # K=0 ship over source needs a CONFIDENT VLM majority,
                                       # not a coin-flip 0.6 (riga shipped a worse model at 0.6);
                                       # marginal preferences must be label-verified instead
        self.labeled_k = 0         # labels revealed so far via request_labels
        self._labeled_dice = None  # {name: HELD-OUT few-shot Dice} once labels revealed
        # K-shot labels are split: TRAIN slices feed the finetune_* tools; HOLDOUT slices
        # rank/safety-check every candidate — so a fine-tuned model is never judged on its
        # own training labels.
        self._labels_revealed = False
        self._train = None         # (imgs, lbls) for fine-tuning
        self._hold = None          # (imgs, lbls) for ranking / safety

    def set_vlm(self, llm, task_ctx, task, anchors):
        """Enable the VLM pairwise-preference signal (requires a configured vision-capable LLM)."""
        self.vlm = {"llm": llm, "ctx": task_ctx, "task": task, "anchors": anchors}

    def _vlm_of(self, cand):
        if not self.vlm:
            return None
        v = self.vlm
        return _vlm_prefer_fraction(cand, self.source_model, v["anchors"], v["llm"],
                                    v["ctx"], v["task"], self.device)

    # -- task-specific toolbox (its OWN diagnosis + the seg-applicable adaptations) --
    def toolbox(self):
        B1 = Budget(compute_minutes=1)
        Bt = Budget(compute_minutes=2, n_trials=1)
        # STRUCTURE-AWARE: hard-exclude bn_reset/tent when the model has NO BatchNorm (provable no-op).
        import torch.nn as _nn
        _bn = sum(1 for _m in self.source_model.modules()
                  if isinstance(_m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)))
        _bn_tools = ([Tool("bn_reset", "recalibrate BatchNorm stats to target (no labels); %d BN "
                           "layer(s) so live." % _bn, self._bn_reset, Bt),
                      Tool("tent", "entropy-minimization TTA (valid for seg; %d BN layer(s))." % _bn,
                           self._tent, Bt)] if _bn > 0 else [])
        return [
            Tool("load_and_deploy", "run the model on target data; report fg/entropy",
                 self._deploy, B1),
            Tool("analyze_domain_shift", "MEASURE source-vs-target image stats + characterize the "
                 "shift (intensity / contrast / modality / resolution / FOV, severity)",
                 self._analyze_shift, Budget(compute_minutes=1, llm_usd=0.02)),
            Tool("inspect_model", "introspect the model (architecture, normalization, in/out, "
                 "likely training domain) — tells you which adaptations apply",
                 self._inspect_model, Budget(llm_usd=0.02)),
            Tool("diagnose_masks", "seg failure analysis: under/over-seg, fragmentation, confidence",
                 self._diagnose, B1),
            Tool("plan", "REASON FIRST: capability inventory (which repairs THIS segmenter admits, "
                 "label costs, budget) -> commit an ordered plan with rationale in ONE call: "
                 "args steps=[...] rationale='...'. Repairs unlock once committed.",
                 self._plan, Budget(llm_usd=0.02)),
            Tool("probe_head", "SEG HEAD REPAIR: refit the final conv as a per-pixel logistic probe "
                 "on frozen features -- K labeled images = tens of thousands of labeled pixels, far "
                 "better conditioned than image-level fine-tuning; cannot damage the backbone. "
                 "Auto-built when labels are revealed. args: K=",
                 self._probe_head, Budget(compute_minutes=2)),
            *_bn_tools,
            Tool("spectral", "match input amplitude spectrum to source", self._spectral, Bt),
            Tool("evaluate", "label-free quality of the current best candidate",
                 self._evaluate, B1),
            Tool("compare_prediction_evidence", "rank zero-label candidates against the source "
                 "using entropy, foreground, structure, and VLM evidence",
                 self._compare_prediction_evidence, B1),
            Tool("request_labels", "ask the clinician for K (3-8) labels; split into TRAIN+"
                 "HOLDOUT, then rank ALL candidates by HELD-OUT few-shot Dice — use when "
                 "label-free signals can't decide or the best is still poor",
                 self._request, lambda a: Budget(k_labels=int(a.get("K", int(os.environ.get("KLAB", "6") or 6))))),
            Tool("finetune_head", "FINE-TUNE only the output head on the K train labels — "
                 "cheapest, safest at tiny K; use first when labels are present but TTA stalls",
                 lambda st, **k: self._finetune("head", **k), Budget(compute_minutes=3)),
            Tool("finetune_partial", "fine-tune the decoder + head on the K train labels — "
                 "more capacity to bridge a real shift; needs a few more labels",
                 lambda st, **k: self._finetune("partial", **k), Budget(compute_minutes=4)),
            Tool("finetune_lora", "parameter-efficient LoRA adapters on the K train labels — "
                 "best capacity-per-label; good when head/partial underfit",
                 lambda st, **k: self._finetune("lora", **k), Budget(compute_minutes=4)),
            Tool("finetune_full", "fine-tune ALL weights on the K train labels — highest "
                 "capacity, overfits at small K; only with a larger label budget",
                 lambda st, **k: self._finetune("full", **k), Budget(compute_minutes=6)),
            Tool("recommend_alternative", "when training-free adaptation CANNOT reach usable "
                 "Dice (a fundamental shift like CT->MRI): get modality-appropriate "
                 "pretrained models to use INSTEAD + a fine-tuning label-budget estimate",
                 self._recommend_alternative, Budget(llm_usd=0.05)),
        ] + self._scaling_tools() + self._bridged_tools()

    def _scaling_tools(self):
        # Component B — env-gated TOOL REGISTRATION (SCALING_EVAL=0 default) so the tool is
        # absent from toolbox/_available/_fallback when off (true inertness, design §3.6).
        import os
        if os.environ.get("SCALING_EVAL", "0") != "1":
            return []
        return [Tool("scaling_eval", "few-shot SCALING-LAW: k-fold CV fine-tune across label "
                     "budgets -> the honest delivered-model estimate (low-variance, fixes "
                     "tiny-holdout under-ship) + an 'annotate N -> expect X' plan. Prefer this "
                     "over the manual finetune ladder once labels are revealed.",
                     lambda st, **k: self._scaling_eval(**k), Budget(compute_minutes=8))]

    # -- the FULL label-free arsenal, pulled from the medadaptagent/tools/ registry
    #    (capability-filtered per model): preprocessing, spatial, test-time aug,
    #    BN/gradient TTA (sar/eata/lora_tta), frequency, postprocessing, meta. --
    def _bridged_tools(self):
        from harness.registry_bridge import bridged_specs
        try:
            specs, caps = bridged_specs("segmentation", self.source_model)
        except Exception:
            return []
        self._caps = caps
        out = []
        for t in specs:
            if t.name in _BRIDGE_SKIP:
                continue
            prof = (t.profile or "").split(".")[0][:140]
            out.append(Tool(f"{t.name}", f"[{t.category}] {prof}",
                            (lambda st, _n=t.name, **k: self._apply_bridged(_n)),
                            Budget(compute_minutes=2, n_trials=1)))
        return out

    def _seg_task_obj(self):
        if getattr(self, "_segtask", None) is None:
            from medadaptagent.tasks import get_task
            self._segtask = get_task("segmentation")
        return self._segtask

    def _apply_bridged(self, name):
        """Apply a registry tool, wrap the resulting ToolContext as a candidate, score it
        label-free (ctx-aware) and add to the tournament. Robust: a tool that throws just
        returns an error result so the agent moves on."""
        from harness.registry_bridge import apply_registry_tool, CtxCandidate
        from scripts.run_baseline_tta import _to_tensor
        fs_idx, fs_lbl = None, None
        if self._labels_revealed and self._train:                # let _ss variants use labels
            fs_idx = list(range(len(self._train[0])))
            fs_lbl = self._train[1]
        try:
            ctx = apply_registry_tool(name, self.source_model, self.timgs, self.nc,
                                      self.device, task=self._seg_task_obj(),
                                      fewshot_idx=fs_idx, fewshot_lbls=fs_lbl)
        except Exception as e:
            return {"evidence": {"error": f"{name} failed: {type(e).__name__}: {str(e)[:120]}"}}
        cand = CtxCandidate(ctx, _to_tensor)
        s = self._stats_of(cand)
        q = _quality_score(s, self.fg_prior)
        self.built.append({"name": name, "model": cand, "ctx": ctx, "quality": q,
                           "vlm_prefer": None, "stats": s})       # VLM skipped for ctx cands
        if self._labels_revealed:
            self._rank_on_holdout()
        return {"candidate": cand, "evidence": {"cand_stats": s, "quality": q,
                "last_method": name, "tools_applied": list(ctx.tools_applied)},
                "metric_estimate": q}

    def _dice_of(self, cand, imgs, lbls):
        from harness.registry_bridge import seg_dice_of
        return seg_dice_of(cand, imgs, lbls, self.nc, self.device)

    def _stats_of(self, cand, n=24):
        from harness.registry_bridge import seg_stats_of
        return seg_stats_of(cand, self.timgs, self.device, n=n, predict_stats=_predict_stats)

    def _recommend_alternative(self, st, **kw):
        """Smart escalation: when training-free adaptation can't bridge the shift, the agent
        shouldn't hand back a marginally-better-but-unusable model. It (a) searches the
        HuggingFace Hub for REAL modality-appropriate models (reading their model + dataset
        cards so the suggestion is grounded + loadable, not a hallucinated repo id), and (b)
        estimates how many labels would fine-tune the existing model to usable Dice."""
        from harness.model_search import recommend_alternative_models
        best = max((self._labeled_dice or {}).values(), default=None)
        rec = recommend_alternative_models(
            self._llm(), task_desc=self.task_desc, task_type="segmentation",
            metric_name="Dice", best_metric=best, usable_target=0.70,
            domain_shift=st.evidence.get("domain_shift"),
            model_info=st.evidence.get("model_info"))
        return {"evidence": {"recommendation": rec}}

    # -- tools (each returns a result dict the harness folds into state) ----------
    def _deploy(self, st, **kw):
        s = _predict_stats(self.source_model, self.timgs, self.device)
        self._src_stats = s
        q = _quality_score(s, self.fg_prior)
        self.built.append({"name": "source", "model": self.source_model, "quality": q, "stats": s})
        return {"deployment": "source@target", "candidate": self.source_model,
                "evidence": {"source_stats": s, "quality": q, "last_method": "source"},
                "metric_estimate": q}

    def _diagnose(self, st, **kw):
        cand = st.best_candidate or self.source_model
        s = self._stats_of(cand)
        src = self._src_stats or s
        mode = _diagnose_from_stats(s, src)
        return {"evidence": {"failure_mode": mode, "stats": s}}

    def _llm(self):
        return self.vlm["llm"] if self.vlm else None

    def _analyze_shift(self, st, **kw):
        # ONE-TIME diagnostic: cache + hard-stop re-analysis (the big toolbox tempts the
        # agent to loop on diagnostics instead of acting).
        if getattr(self, "_shift_ev", None) is not None:
            return {"evidence": {"domain_shift": self._shift_ev,
                    "directive": "shift ALREADY analyzed — do NOT call again; ACT: try an "
                                 "adaptation tool now."}}
        from harness.analysis import analyze_domain_shift
        self._shift_ev = analyze_domain_shift(self.simgs, self.timgs, self._llm(), self.task_desc, model=self.source_model)
        return {"evidence": {"domain_shift": self._shift_ev, **self._retrieve_lessons(st)}}

    def _inspect_model(self, st, **kw):
        if getattr(self, "_model_ev", None) is not None:
            return {"evidence": {"model_info": self._model_ev,
                    "directive": "model ALREADY inspected — do NOT call again; ACT: try an "
                                 "adaptation tool now."}}
        from harness.analysis import inspect_model
        self._model_ev = inspect_model(self.source_model, self._llm(), self.task_desc)
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
                              getattr(self, "_model_ev", None), "segmentation")
    def _adapt(self, name, builder):
        cand = copy.deepcopy(self.source_model)
        builder(cand)
        return self._score_candidate(name, cand)

    def _score_candidate(self, name, cand):
        s = _predict_stats(cand, self.timgs, self.device)
        q = _quality_score(s, self.fg_prior)
        vlm = self._vlm_of(cand)                    # pairwise prefer-over-source, or None
        self.built.append({"name": name, "model": cand, "quality": q,
                           "vlm_prefer": vlm, "stats": s})
        ev = {"cand_stats": s, "quality": q, "last_method": name}
        if vlm is not None:
            ev["vlm_prefer"] = round(vlm, 3)        # fraction VLM judges BETTER than source
        return {"candidate": cand, "evidence": ev, "metric_estimate": q}

    def _bn_reset(self, st, **kw):
        from scripts.run_baseline_tta import baseline_bn_reset
        return self._adapt("bn_reset", lambda m: baseline_bn_reset(m, self.timgs, self.device))

    def _tent(self, st, epochs=5, **kw):
        from scripts.run_baseline_tta import baseline_tent
        return self._adapt("tent", lambda m: baseline_tent(m, self.timgs, self.device, epochs=epochs))

    def _spectral(self, st, **kw):
        from scripts.step4_toolbox_gate import SpectralWrap, _mean_amp
        amp = _mean_amp(self.simgs)
        cand = SpectralWrap(copy.deepcopy(self.source_model), amp).to(self.device).eval()
        return self._score_candidate("spectral", cand)

    def _evaluate(self, st, **kw):
        cand = st.best_candidate or self.source_model
        s = self._stats_of(cand)
        return {"evidence": {"eval_stats": s, "quality": _quality_score(s, self.fg_prior)},
                "metric_estimate": _quality_score(s, self.fg_prior)}

    def _zero_shot_comparison(self, evidence):
        """Construct the source-relative K=0 consensus from stored prediction evidence."""
        import math
        latest = {str(rec.get("name")): rec for rec in self.built}
        source = latest.get("source")
        if source is None:
            return {"recommended_candidate": None,
                    "reason": "source prediction statistics are unavailable", "rows": []}
        src_stats = source.get("stats") or {}
        src_q = float(source.get("quality", 0.0) or 0.0)
        src_fg = max(float(src_stats.get("fg_frac", 0.0) or 0.0), 1e-6)
        src_comp = max(float(src_stats.get("n_comp", 0.0) or 0.0), 1.0)
        shift = ((evidence or {}).get("domain_shift") or {}).get("characterization") or (
            (evidence or {}).get("domain_shift") or {})
        semantic = bool(shift.get("semantic_flag", False))
        rows = []
        for name, rec in latest.items():
            if name == "source":
                continue
            stats = rec.get("stats") or {}
            q = float(rec.get("quality", 0.0) or 0.0)
            q_gain = q - src_q
            vlm = rec.get("vlm_prefer")
            vlm = None if vlm is None else float(vlm)
            fg = max(float(stats.get("fg_frac", 0.0) or 0.0), 1e-6)
            comp = max(float(stats.get("n_comp", 0.0) or 0.0), 1.0)
            fg_ratio, comp_ratio = fg / src_fg, comp / src_comp
            plausible, why = self._structural_plausibility(rec.get("model"), stats)
            score = q_gain
            score += (0.20 * (vlm - 0.5)) if vlm is not None else -0.05
            score -= 0.02 * min(2.0, abs(math.log(fg_ratio)))
            score -= 0.01 * min(2.0, abs(math.log(comp_ratio)))
            consensus = bool(plausible and q_gain >= 0.0 and score >= 0.03)
            if semantic:
                consensus = bool(consensus and vlm is not None and vlm >= 0.67)
            rows.append({
                "candidate": name, "quality": round(q, 4),
                "quality_gain_vs_source": round(q_gain, 4),
                "vlm_prefer": None if vlm is None else round(vlm, 3),
                "fg_ratio_vs_source": round(fg_ratio, 4),
                "component_ratio_vs_source": round(comp_ratio, 4),
                "entropy_delta_vs_source": round(
                    float(stats.get("entropy", 0.0) or 0.0)
                    - float(src_stats.get("entropy", 0.0) or 0.0), 4),
                "structurally_plausible": bool(plausible),
                "plausibility_reason": why, "semantic_shift": semantic,
                "selection_score": round(score, 4), "consensus_pass": consensus,
            })
        eligible = [row for row in rows if row["consensus_pass"]]
        winner = max(eligible, key=lambda row: row["selection_score"]) if eligible else None
        return {
            "source_quality": round(src_q, 4), "semantic_shift": semantic,
            "recommended_candidate": None if winner is None else winner["candidate"],
            "selection_rule": ("source-relative entropy/foreground/fragmentation improvement "
                               "+ VLM support - prediction-shift penalties; semantic shifts "
                               "also require strong VLM agreement"),
            "rows": rows,
        }

    def _compare_prediction_evidence(self, st, **kw):
        return {"evidence": {"zero_shot_comparison":
                             self._zero_shot_comparison(st.evidence)}}

    def _fewshot_dice(self, model, imgs, lbls):
        """Self-contained mean fg Dice over (img,lbl) pairs, using the SAME prediction
        path as _predict_stats (which works). Resizes each GT label to the prediction
        grid; averages only over slices that actually contain foreground."""
        import torch
        from PIL import Image as PILImage
        from scripts.run_baseline_tta import _to_tensor
        model.eval(); ds = []
        with torch.no_grad():
            for img, lbl in zip(imgs, lbls):
                pred = model(_to_tensor(img, self.device)).argmax(1)[0].cpu().numpy()
                g = (np.asarray(lbl) >= 1).astype(np.uint8)
                if g.shape != pred.shape:
                    g = (np.array(PILImage.fromarray(g).resize(
                        (pred.shape[1], pred.shape[0]), PILImage.NEAREST)) > 0)
                g = g.astype(bool); p = (pred >= 1)
                if g.sum() == 0:
                    continue
                ds.append(2.0 * (p & g).sum() / (p.sum() + g.sum() + 1e-8))
        return round(float(np.mean(ds)), 4) if ds else 0.0

    def _capability_inventory(self):
        """WHAT IS AVAILABLE for THIS segmenter -- knowable with zero labels (cls-ledger M1)."""
        import torch.nn as _nn
        m = self.source_model
        bn = sum(1 for x in m.modules() if isinstance(x,(_nn.BatchNorm1d,_nn.BatchNorm2d,_nn.BatchNorm3d)))
        convs = [x for x in m.modules() if isinstance(x,_nn.Conv2d) and x.out_channels == self.nc]
        k_left = int(os.environ.get("KLAB", "0") or 0)
        inv = {"model": {"batchnorm_layers": bn, "final_conv_head": bool(convs),
                         "head_kernel": list(convs[-1].kernel_size) if convs else None},
               "data": {"n_target": len(self.timgs), "n_classes": self.nc, "fg_prior": self.fg_prior},
               "budget": {"labels_available": k_left, "labels_revealed": bool(self._labels_revealed)},
               "repairs": [
                 {"name": "input-space repair (histogram_match/spectral/gamma_correct/fda_warmup/zscore_norm)", "label_cost": 0,
                  "applicable": True, "fixes": "appearance/intensity shift; can re-align a displaced head for free"},
                 {"name": "bn_reset / tent", "label_cost": 0, "applicable": bn > 0,
                  "fixes": (("norm-statistic shift (%d BatchNorm layers)" % bn) if bn>0 else "NOTHING here -- no BatchNorm, structural no-op")},
                 {"name": "probe_head", "label_cost": "reuses the SAME requested labels (no extra)",
                  "applicable": bool(convs),
                  "fixes": "a misread representation at the FINAL conv: refit it as a per-pixel probe on frozen features; K images = tens of thousands of labeled pixels"},
                 {"name": "finetune_{head,partial,lora}", "label_cost": "the requested K images",
                  "applicable": True, "fixes": "genuinely degraded representation; most capacity, most overfit risk"}]}
        self._inventory = inv
        return inv

    def _plan(self, st, **kw):
        """Commit a reasoned natural-language plan and record the tools it mentions.

        Executability is enforced later by the harness action menu; a plan should not be
        rejected merely because a step contains arguments or explanatory prose.
        """
        inv = self._capability_inventory()
        steps = kw.get("steps"); rationale = kw.get("rationale", "")
        if steps is None:
            return {"evidence": {"inventory": inv,
                                 "directive": "call plan again with steps and rationale"}}
        if isinstance(steps, str):
            steps = [x.strip() for x in steps.split(",") if x.strip()]
        steps = [str(x).strip() for x in steps if str(x).strip()]
        if not steps:
            return {"evidence": {"inventory": inv,
                                 "directive": "call plan again with at least one nonempty step"}}
        valid = {t.name for t in self.toolbox()} | {"ship", "keep_source", "escalate",
                                                    "request_labels", "load_and_deploy"}
        recognized = sorted({name for step in steps for name in valid if name in step})
        self._planned = {"steps": steps, "rationale": str(rationale),
                         "recognized_tools": recognized}
        return {"evidence": {"plan": self._planned, "inventory": inv}}

    def _feats_of_head(self, model, imgs):
        """Per-pixel features at the input of the FINAL nc-channel conv (the seg head)."""
        import torch, torch.nn as _nn
        convs=[x for x in model.modules() if isinstance(x,_nn.Conv2d) and x.out_channels==self.nc]
        if not convs: return None, None
        head=convs[-1]; h={}
        hd=head.register_forward_hook(lambda m,i,o: h.__setitem__("f", i[0].detach()))
        from scripts.run_baseline_tta import _to_tensor
        F=[]
        try:
            model.eval()
            with torch.no_grad():
                for im in imgs:
                    model(_to_tensor(im, self.device))
                    F.append(h["f"][0].float().cpu().numpy())
        finally:
            hd.remove()
        return F, head

    def _fit_pixel_probe(self, F_list, lbl_list, max_px=40000):
        """Logistic per-pixel probe on frozen features; class-balanced pixel subsample."""
        import numpy as _np
        from sklearn.linear_model import LogisticRegression
        X=[]; Y=[]
        for F, L in zip(F_list, lbl_list):
            C,H,W = F.shape
            Lr = _np.asarray(L)
            if Lr.shape != (H,W):
                from PIL import Image as _Im
                Lr = _np.array(_Im.fromarray(Lr.astype(_np.uint8)).resize((W,H), _Im.NEAREST))
            X.append(F.reshape(C,-1).T); Y.append(Lr.reshape(-1))
        X=_np.concatenate(X); Y=_np.concatenate(Y).astype(int)
        keep=[]
        rng=_np.random.RandomState(0)
        for c in _np.unique(Y):
            ci=_np.where(Y==c)[0]
            take=min(len(ci), max(2000, max_px//max(1,len(_np.unique(Y)))))
            keep.append(rng.choice(ci, size=take, replace=False))
        k=_np.concatenate(keep)
        if len(_np.unique(Y[k]))<2: return None
        return LogisticRegression(max_iter=1500).fit(X[k], Y[k])

    def _probe_model_from(self, lr):
        """Deep-copied source whose final head is a 1x1 conv carrying the probe weights."""
        import copy as _cp, torch, torch.nn as _nn, numpy as _np
        m=_cp.deepcopy(self.source_model).to(self.device)
        convs=[(nm,x) for nm,x in m.named_modules() if isinstance(x,_nn.Conv2d) and x.out_channels==self.nc]
        nm, head = convs[-1]
        Cin = head.in_channels
        W=_np.zeros((self.nc, Cin), dtype=_np.float32); b=_np.zeros((self.nc,), dtype=_np.float32)
        cls=list(lr.classes_)
        if len(cls)==2 and lr.coef_.shape[0]==1:
            W[int(cls[1])]=lr.coef_[0]; b[int(cls[1])]=lr.intercept_[0]
            W[int(cls[0])]=-lr.coef_[0]; b[int(cls[0])]=-lr.intercept_[0]
        else:
            for i,c in enumerate(cls): W[int(c)]=lr.coef_[i]; b[int(c)]=lr.intercept_[i]
        new=_nn.Conv2d(Cin, self.nc, kernel_size=1).to(self.device)
        with torch.no_grad():
            new.weight.copy_(torch.tensor(W).view(self.nc,Cin,1,1))
            new.bias.copy_(torch.tensor(b))
        parent=m; parts=nm.split(".")
        for q in parts[:-1]: parent=getattr(parent,q)
        setattr(parent, parts[-1], new)
        m.eval(); return m

    def _probe_head(self, st, K=None, **kw):
        """SEG HEAD REPAIR: refit the final conv as a per-pixel logistic probe on frozen features."""
        if K is None: K=int(os.environ.get("KLAB","6") or 6)
        self._ensure_labels(K)
        imgs=list(self._train[0])+list(self._hold[0]); lbls=list(self._train[1])+list(self._hold[1])
        F,head=self._feats_of_head(self.source_model, imgs)
        if F is None: return {"evidence":{"error":"no final nc-channel Conv2d head found"}}
        lr=self._fit_pixel_probe(F,lbls)
        if lr is None: return {"evidence":{"error":"probe fit failed (single class in pixels)"}}
        m=self._probe_model_from(lr)
        self.built=[b for b in self.built if b.get("name")!="probe_head"]
        stt=self._stats_of(m); q=float(stt.get("quality", 0.5)) if isinstance(stt,dict) else 0.5
        self.built.append({"name":"probe_head","model":m,"quality":q,"stats":stt})
        if self._labels_revealed: self._rank_on_holdout()
        hd=(self._labeled_dice or {}).get("probe_head")
        return {"candidate": m, "metric_estimate": q,
                "evidence": {"last_method":"probe_head","quality":q,
                             "probe_head":{"n_label_images":len(imgs),"head_kernel":list(head.kernel_size)},
                             "holdout_dice":hd,"labeled_dice":self._labeled_dice}}

    def _probe_oof_dice(self):
        """LOO out-of-fold per-case Dice deltas for the probe (K cheap logistic refits)."""
        imgs=list(self._train[0])+list(self._hold[0]); lbls=list(self._train[1])+list(self._hold[1])
        if len(imgs)<3: return None
        F,_=self._feats_of_head(self.source_model, imgs)
        if F is None: return None
        out=[]; absd=[]
        for i in range(len(imgs)):
            tr=[j for j in range(len(imgs)) if j!=i]
            lr=self._fit_pixel_probe([F[j] for j in tr],[lbls[j] for j in tr])
            if lr is None: return None
            m=self._probe_model_from(lr)
            d_c=float(self._dice_of_model(m,[imgs[i]],[lbls[i]]))
            d_s=float(self._dice_of_model(self.source_model,[imgs[i]],[lbls[i]]))
            out.append(d_c-d_s); absd.append(d_c)
        self._probe_oof_abs = absd          # UNBIASED per-case dice for ranking/reporting
        return out

    def _dice_of_model(self, model, imgs, lbls):
        """Per-case Dice for a PLAIN model (bypasses the candidate/ctx wrapper)."""
        from scripts.run_baseline_tta import _avg_dice
        return _avg_dice(model, imgs, lbls, self.nc, self.device)

    def _loo_finetune_dice(self, mode):
        """LOO-OOF per-case Dice deltas for a FINE-TUNE: refit leaving each labeled image out with
        the same recipe, score the held-out image. K unbiased paired samples where the 2-slice
        holdout had none. Affordable because seg few-shot refits take seconds."""
        from harness.finetune import finetune
        from scripts.run_baseline_tta import _to_tensor
        imgs=list(self._train[0])+list(self._hold[0]); lbls=list(self._train[1])+list(self._hold[1])
        if len(imgs)<4: return None
        out=[]
        for i in range(len(imgs)):
            tr=[j for j in range(len(imgs)) if j!=i]
            try:
                m,_=finetune(self.source_model,[imgs[j] for j in tr],[lbls[j] for j in tr],
                             task="seg",nc=self.nc,device=self.device,to_tensor=_to_tensor,mode=mode)
            except Exception:
                return None
            d_c=float(self._dice_of_model(m,[imgs[i]],[lbls[i]]))
            d_s=float(self._dice_of_model(self.source_model,[imgs[i]],[lbls[i]]))
            out.append(d_c-d_s)
        return out

    def _refit_full_k(self, mode):
        """After LOO verification passes, SHIP the model refit on ALL K labels (standard
        select-on-CV, refit-on-full practice; matches what a naive fine-tune trains on)."""
        from harness.finetune import finetune
        from scripts.run_baseline_tta import _to_tensor
        imgs=list(self._train[0])+list(self._hold[0]); lbls=list(self._train[1])+list(self._hold[1])
        m,_=finetune(self.source_model,imgs,lbls,task="seg",nc=self.nc,device=self.device,
                     to_tensor=_to_tensor,mode=mode)
        return m

    def _paired_gate(self, name, B=2000, lo_pct=None):
        """Paired per-case Dice verification; objective-conditioned bound (cls ledger 8)."""
        if lo_pct is None:
            obj=getattr(self,"objective",None) or os.environ.get("OBJECTIVE","never_worse")
            lo_pct=25.0 if str(obj)=="max_performance" else 10.0
        rec=next((b for b in self.built if b["name"]==name), None)
        if rec is None or self._hold is None: return None
        if name=="probe_head":
            d=self._probe_oof_dice()
            if not d or len(d)<3: return None
        elif str(name).startswith("finetune"):
            # LOO-OOF verification (the 2-slice holdout has no statistical power; the paired gate
            # abstained and real +0.39 recoveries were refused). On PASS, the shipped candidate is
            # replaced by the full-K refit -- verified like CV, trained like naive.
            mode = name.split("finetune_",1)[-1] or "head"
            d = self._loo_finetune_dice(mode)
            if not d or len(d) < 4: return None
        else:
            imgs=list(self._train[0])+list(self._hold[0]); lbls=list(self._train[1])+list(self._hold[1])
            if len(imgs)<3: return None
            d=[float(self._dice_of(rec["model"],[im],[lb]))-float(self._dice_of(self.source_model,[im],[lb]))
               for im,lb in zip(imgs,lbls)]
        import numpy as _np
        d=_np.asarray(d,dtype=float); n=len(d)
        # POWER GUARD (ledger 10): a bootstrap over <6 samples is not a test -- a fine-tune passed
        # it on noise and shipped a -0.107 collapse the flat gate would have refused. Below n=6 the
        # paired gate ABSTAINS (returns None) so the margin gate stays authoritative.
        # (power guard removed for finetunes here: they now verify via LOO-OOF,
        #  which is unbiased at n=K; the old guard was for the biased 2-slice holdout)
        rng=_np.random.RandomState(0)
        means=_np.array([d[rng.randint(0,n,n)].mean() for _ in range(B)])
        lo=float(_np.percentile(means,lo_pct))
        # practical-significance floor: a +-0.001 tie is not a deployable improvement
        ok = bool(lo > 0.0) and float(d.mean()) >= 0.005
        # (full-K refit swap REMOVED: replacing the model object inside the gate broke the
        #  identity audit -- riga shipped finetune_partial 0.837 but was scored as source. The
        #  verified split-trained model is what ships; refit-on-full is future work.)
        return (ok, lo, float(d.mean()), n, float((d>0).mean()))

    def _best_by_labels_verified(self):
        """Recommend the best labeled candidate that PASSES verification (cls ledger 9)."""
        ld=self._labeled_dice or {}
        cands=[n for n in ld if n!="source"]
        if not cands: return None
        # TIE-BREAK BY PAIRED MEAN (ledger 13): labeled scores saturate/tie at tiny K (many 1.0s)
        # and build-order tie-breaking shipped a +0.115 candidate over a +0.197 one. When scores
        # tie, the paired-gate MEAN (already-computed evidence) orders by actual estimated gain.
        def _pm(n):
            try:
                pg = self._paired_gate(n)
                return pg[2] if pg else 0.0
            except Exception:
                return 0.0
        ranked=sorted(cands,key=lambda n:((ld.get(n) or 0), _pm(n)),reverse=True)
        if os.environ.get("VERIFIER","")!="paired": return ranked[0]
        for n in ranked:
            try: pg=self._paired_gate(n)
            except Exception: pg=None
            if pg is None or pg[0]:
                self._best_verified = bool(pg and pg[0]); return n
        return "source"

    def _ensure_labels(self, K=None):
        if K is None:
            K = int(os.environ.get("KLAB", "6") or 6)   # full authorized budget (cls ledger 6)
        """Reveal K labels ONCE and split them TRAIN (~70%) / HOLDOUT (~30%), both drawn
        from evenly-spaced foreground-bearing slices. TRAIN feeds the finetune_* tools;
        HOLDOUT ranks/safety-checks every candidate (incl. fine-tuned ones) — so a model is
        never judged on its own training labels. Idempotent: later calls reuse the split."""
        if self._labels_revealed:
            return self._train
        K = int(K)
        fg_idx = [i for i in range(len(self.tlbls)) if (np.asarray(self.tlbls[i]) >= 1).any()]
        pool = fg_idx if fg_idx else list(range(len(self.timgs)))
        _draw = os.environ.get("LABEL_DRAW_SEED", "")
        if _draw != "":                     # opt-in random fg-slice draw for robustness studies
            _rng = np.random.RandomState(int(_draw))
            idx = [int(x) for x in np.sort(_rng.choice(pool, size=min(K, len(pool)), replace=False))]
        else:
            sel = np.linspace(0, len(pool) - 1, min(K, len(pool)), dtype=int)
            idx = [pool[int(j)] for j in sel]
        # Holdout = every 3rd slice SPREAD across the fg range (not the tail) so it is
        # representative; tail-only holdouts misjudged genuinely-good candidates.
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
        """Score candidates by labeled Dice (ctx-aware). KEY: a label-free candidate (and
        source) never trained on ANY label, so rank it on ALL K labels (reliable). Only a
        finetune_* candidate trained on the TRAIN split, so it is ranked on the HOLDOUT
        split (bias-free). Using a tiny holdout for label-free candidates underestimated
        them and caused false escalations."""
        h_imgs, h_lbls = self._hold
        all_imgs = list(self._train[0]) + list(h_imgs)
        all_lbls = list(self._train[1]) + list(h_lbls)
        ld = {}
        for b in self.built:
            if b["name"] == "probe_head":
                # the probe TRAINS on all K labels, so holdout ranking would be train-contaminated
                # (user-caught: report quoted a biased 0.807). Rank/report by the UNBIASED LOO-OOF
                # estimate instead (each case scored by a probe that never saw it).
                if getattr(self, "_probe_oof_abs", None) is None:
                    try: self._probe_oof_dice()
                    except Exception: pass
                oof = getattr(self, "_probe_oof_abs", None)
                ld[b["name"]] = round(float(np.mean(oof)), 4) if oof else \
                                round(float(self._dice_of(b["model"], h_imgs, h_lbls)), 4)
            elif str(b["name"]).startswith("finetune"):
                ld[b["name"]] = round(float(self._dice_of(b["model"], h_imgs, h_lbls)), 4)
            else:
                ld[b["name"]] = round(float(self._dice_of(b["model"], all_imgs, all_lbls)), 4)
        self._labeled_dice = ld
        return ld

    def _estimate_performance(self):
        """Look up the EXPECTED performance for THIS task from the literature (via the LLM,
        which has read the benchmarks) — NOT a hardcoded ceiling. e.g. BraTS whole-tumor Dice
        ~0.85, prostate MRI ~0.88, liver ~0.95. The decision uses these task-grounded numbers
        as the 'good-enough' ceiling + minimal-usable floor, so nothing is brewed per dataset."""
        if getattr(self, "_perf_est", None) is not None:
            return self._perf_est
        fallback = {"sota": 0.85, "usable": 0.65, "rationale": "generic seg default (no LLM)"}
        llm = self._llm()
        if llm is None:
            self._perf_est = fallback; return fallback
        import json, re
        prompt = (f"Medical-image SEGMENTATION task: {self.task_desc}. Metric: foreground Dice.\n"
                  f"From the published literature/benchmarks for THIS structure + modality, give: "
                  f"the Dice a COMPETENT / near-SOTA model typically reaches, and the MINIMUM "
                  f"clinically-usable Dice. Respond ONLY JSON: {{\"sota\": <0..1>, "
                  f"\"usable\": <0..1>, \"rationale\": \"<=15 words citing the typical benchmark\"}}.")
        try:
            d = json.loads(re.search(r"\{.*\}", llm.chat(prompt), re.S).group(0))
            d = {"sota": float(d.get("sota", 0.85)), "usable": float(d.get("usable", 0.65)),
                 "rationale": str(d.get("rationale", ""))[:160]}
        except Exception:
            d = fallback
        self._perf_est = d
        return d

    def _request(self, st, K=None, **kw):
        if K is None:
            K = int(os.environ.get("KLAB", "6") or 6)   # full authorized budget (cls ledger 6)
        """Clinician reveals K labels → TRAIN/HOLDOUT split → rank EVERY built candidate by
        HELD-OUT few-shot Dice (the trustworthy disambiguator). Now also the gateway that
        the finetune_* tools train against."""
        if self.tlbls is None:
            return {"k_labels_used": 0, "evidence": {"error": "no labels available here"}}
        self._ensure_labels(K)
        try:                                            # auto-build the cheap head repair (ledger 5/7)
            pr = self._probe_head(st, K=K)
            _perr = (pr.get("evidence", {}) or {}).get("error")
            if _perr: print("[auto-probe seg] NOT BUILT:", _perr, flush=True)
            elif pr.get("candidate") is not None and st is not None:
                try:
                    st.named_candidates["probe_head"] = pr["candidate"]
                    st.candidates_log.append({"method": "probe_head",
                                              "metric_estimate": pr.get("metric_estimate")})
                except Exception:
                    print("[auto-probe seg] state registration failed", flush=True)
        except Exception as _e:
            import traceback; traceback.print_exc()
        ld = self._rank_on_holdout()
        best = self._best_by_labels_verified() if ld else None
        est = self._estimate_performance()        # task-grounded ceiling/floor, not hardcoded
        return {"k_labels_used": self.labeled_k,
                "evidence": {"labeled_dice": ld, "best_by_labels": best, "best_verified": bool(getattr(self, "_best_verified", False)),
                             "n_train": len(self._train[0]), "n_holdout": len(self._hold[0]),
                             "expected_metric": est["sota"], "usable_floor": est["usable"],
                             "safety_margin": self.safety_margin(ld.get("source", 0.0)),
                             "perf_rationale": est["rationale"],
                             "note": f"HELD-OUT few-shot Dice ({len(self._hold[0])} slices); "
                                     f"task-expected Dice≈{est['sota']} (usable≥{est['usable']}: "
                                     f"{est['rationale']}). Ship '{best}' if it beats source; if best "
                                     f"is well below {est['sota']}, fine-tune to close the gap."}}

    def _finetune(self, mode, K=None, base="source", **kw):
        if K is None:
            K = int(os.environ.get("KLAB", "6") or 6)
        """Few-shot FINE-TUNE; the LLM DESIGNS the recipe (kw lr=/steps= override per-budget default)
        and reacts to the returned diagnostics by RE-CALLING with a revised recipe. Safety floor
        (best-ckpt + never-worse-than-source guard) lives in finetune()."""
        from harness.finetune import finetune
        from scripts.run_baseline_tta import _to_tensor
        if self.tlbls is None:
            return {"evidence": {"error": "no labels available — cannot finetune"}}
        tr_imgs, tr_lbls = self._ensure_labels(K)
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
        model, info = finetune(start, tr_imgs, tr_lbls, task="seg", nc=self.nc,
                               device=self.device, to_tensor=_to_tensor, mode=mode, lr=lr, steps=steps)
        name = f"finetune_{mode}"
        self.built = [b for b in self.built if b.get("name") != name]
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
            diag.append("small-K: trained on all labels, no internal val")
        res.setdefault("evidence", {})
        best_by = self._best_by_labels_verified() if self._labeled_dice else None
        res["evidence"].update({"finetune": info, "labeled_dice": self._labeled_dice,
                                "best_by_labels": best_by, "best_verified": bool(getattr(self, "_best_verified", False)), "usable_bar": self.USABLE_BAR,
                                "holdout_dice": hd, "trained_on": len(tr_imgs),
                                "recipe_used": info.get("recipe"), "ft_diagnostics": diag,
                                "note": (f"{name}: held-out Dice {hd} on {len(self._hold[0])} slices | "
                                         f"recipe {info.get('recipe')} | loss {lf}->{ll}, val {sv}->{bv} "
                                         f"(pick@{info.get('picked_step')}) | "
                                         f"{'; '.join(diag) if diag else 'clean fit'} | "
                                         f"to REVISE re-call a finetune_* with args lr=/steps=")})
        return res

    def _scaling_eval(self, K=12, modes=("head", "partial"), **kw):
        """Component B: k-fold CV fine-tune across label budgets. The GATE = full-K k-fold CV
        of the best mode (low-variance estimate of the actually-shipped model — fixes the tiny
        holdout under-ship); the scaling curve is a separate advisory deliverable. Trains the
        shipped artifact on the full K, registers it, sets labeled_dice from the CV gate."""
        from harness.scaling import run_scaling_eval
        from harness.finetune import finetune
        from scripts.run_baseline_tta import _to_tensor, _avg_dice
        import torch
        if self.tlbls is None:
            return {"evidence": {"error": "no labels available — cannot scale"}}
        self._ensure_labels(K)
        imgs = list(self._train[0]) + list(self._hold[0])      # ALL revealed K (CV over them)
        lbls = list(self._train[1]) + list(self._hold[1])

        def fit_metric(m, ei, el):
            return float(_avg_dice(m, ei, el, self.nc, self.device))

        def finetune_fn(mode, ti, tl):
            torch.manual_seed(0); np.random.seed(0)         # finetune() has no seed= param
            m, _ = finetune(self.source_model, ti, tl, task="seg", nc=self.nc,
                            device=self.device, to_tensor=_to_tensor, mode=mode)
            return m

        est = self._estimate_performance()
        rep = run_scaling_eval(source_model=self.source_model, imgs=imgs, lbls=lbls,
                               fit_metric=fit_metric, finetune_fn=finetune_fn,
                               safety_margin_fn=self.safety_margin, modes=tuple(modes),
                               metric_name=self.metric_name, usable_floor=est.get("usable"), seed=0)
        ship_model = finetune_fn(rep.best_mode, imgs, lbls)   # the shipped artifact (full-K)
        res = self._score_candidate(rep.shipped_candidate, ship_model)
        # labeled_dice from the CV GATE (honest, low-variance) — NOT the optimistic full-K-train eval
        self._labeled_dice = {"source": round(rep.source_metric, 4),
                              rep.shipped_candidate: round(rep.shipped_eval_cv_mean, 4)}
        best_by = self._best_by_labels_verified()
        res.setdefault("evidence", {})
        res["evidence"].update({"labeled_dice": self._labeled_dice, "best_by_labels": best_by, "best_verified": bool(getattr(self, "_best_verified", False)),
                                "scaling_law": rep.to_evidence(), "scaling_plan": rep.plan,
                                "safety_margin": rep.margin, "expected_metric": est["sota"],
                                "usable_floor": est["usable"], "perf_rationale": est["rationale"],
                                "note": rep.plan})
        return res

    # -- I1 safety: never ship a candidate worse/degenerate vs source --
    def safety_margin(self, source_metric) -> float:
        """The ONE numeric decision gate (design §A.1) — shared by safety_check, the decision
        spine, and the scaling-law verdict. Few-label noise guard; never a hand-copied constant."""
        return 0.05  # flat NOISE floor (2026-06-08); was max(0.05,0.10*source) — the 10%-relative term over-rejected real verified gains at high baselines

    def _structural_plausibility(self, cand, s=None):
        """STRUCTURAL coherence (NOT quality): veto outputs that are empty, exploded, or
        scattered into far more components than the source — the implausible shapes the
        pairwise VLM can rate as 'better' (e.g. amos finetune collapsed mask, vlm=1.0,
        true=0.0). A coherent-but-low-Dice mask PASSES (no accuracy judgment), so valid
        recoveries (chaos liver 0.39) are never vetoed. Bounds are RELATIVE to source /
        fg_prior, not brewed constants. Returns (ok, reason)."""
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

    def safety_check(self, candidate, source, evidence):
        src = self._src_stats
        if src is None:
            return SafetyVerdict(False, "not deployed yet — no source baseline")
        if candidate is source:                     # keeping source is always safe
            return SafetyVerdict(True, "source baseline")
        rec = next((b for b in self.built if b["model"] is candidate), None)
        name = rec["name"] if rec else None
        # (1) LABELS ARE GROUND TRUTH: when revealed, the labeled-Dice gate is authoritative
        # and runs FIRST. A truly-collapsed model scores ~0 on the labels (cd < source), so
        # it is still caught here — without false-flagging a cautious-but-correct recovery
        # (chaos liver: low MEAN fg yet high Dice) that the label-free collapse heuristic would.
        if self._labeled_dice is not None and name in self._labeled_dice:
            if os.environ.get("VERIFIER", "") == "paired":
                pg = self._paired_gate(name)
                if pg is not None:
                    ok, lo, mean_d, npair, frac = pg
                    return SafetyVerdict(ok, "paired per-case Dice gain mean %+.4f, lo %+.4f, "
                                             "%.0f%% of %d cases improved" % (mean_d, lo, 100*frac, npair))
            cd = self._labeled_dice[name]; sd = self._labeled_dice.get("source", 0.0)
            # few-label NOISE GUARD: an adaptation must CLEARLY beat source on the labels,
            # not a noise-level tie. Margin via the single source of truth (design §A.1).
            margin = self.safety_margin(sd)
            if cd < sd + margin:
                return SafetyVerdict(False, f"labeled Dice {cd} not clearly > source {sd} "
                                            f"(needs +{margin:.3f}; few-label noise guard)")
            return SafetyVerdict(True, f"labeled Dice {cd} clearly > source {sd} (label-verified)")
        # (2) NO labels: require the full source-relative consensus winner. Compute
        # it lazily as a gate invariant even if the planner did not call the comparison tool.
        comparison = (evidence or {}).get("zero_shot_comparison") or \
            self._zero_shot_comparison(evidence or {})
        row = next((r for r in comparison.get("rows", [])
                    if r.get("candidate") == name), None)
        consensus_override = bool(
            row and row.get("consensus_pass")
            and comparison.get("recommended_candidate") == name)
        if comparison.get("rows") and not consensus_override:
            return SafetyVerdict(False, "candidate is not the zero-label consensus winner")
        # Label-free degenerate guard (vs EXPECTED anatomy fraction fg_prior).
        s = self._stats_of(candidate)
        if s["fg_frac"] < 0.2 * self.fg_prior:
            return SafetyVerdict(False, "candidate collapsed foreground (degenerate)")
        if s["fg_frac"] > 4.0 * self.fg_prior:
            return SafetyVerdict(False, "candidate exploded foreground (degenerate)")
        if os.environ.get("PLAUSIBILITY_GATE", "0") != "0":   # BEFORE-gate: structural coherence
            ok, why = self._structural_plausibility(candidate, s)
            if not ok:
                return SafetyVerdict(False, f"implausible output (structural pre-gate): {why}")
        # (3) VLM PAIRWISE-preference gate: ship only if VLM prefers candidate over source.
        # Skipped for ctx (bridged) candidates — the model-based VLM render can't replay
        # their numpy preprocess; their quality + labeled gates still protect safety.
        from harness.registry_bridge import CtxCandidate
        if self.vlm is not None and not isinstance(candidate, CtxCandidate):
            prefer = rec["vlm_prefer"] if rec and rec.get("vlm_prefer") is not None \
                else self._vlm_of(candidate)
            if (prefer is not None and prefer < self.VLM_PREFER_THRESH
                    and not consensus_override):
                return SafetyVerdict(False, f"VLM prefers source over candidate "
                                            f"(prefer {prefer:.2f} < {self.VLM_PREFER_THRESH})")
        if _quality_score(s, self.fg_prior) < _quality_score(src, self.fg_prior) - 0.05:
            return SafetyVerdict(False, "label-free quality below source")
        return SafetyVerdict(True, "passes safety vs source")
