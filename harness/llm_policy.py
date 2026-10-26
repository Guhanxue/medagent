"""The LLM policy — the agent BRAIN that drives the harness loop.

Given the harness view (goal, tools, state, evidence table, guardrail feedback),
it asks an LLM to choose the next action as JSON {tool, args}. The LLM decides
WHAT to do and WHEN to stop — emergent, per-case control flow. The harness still
enforces the four guardrails around whatever the LLM chooses.

The LLM is injected as `call_llm: (prompt:str)->str` so this module has no SDK
dependency and is unit-testable with a mock (no key, no network).
"""
from __future__ import annotations
import json
import os
from typing import Callable

SYSTEM = """You are a deployment agent. A clinician at a new site gave you a trained \
model and their local (mostly unlabeled) data. Your GOAL: deliver the best model for \
THIS site that is provably NOT worse than the source, within budget — or honestly hand \
back if you cannot. You act ONLY by calling one tool per turn.

How to think (be decisive — every turn must make progress):
- First deploy. Then UNDERSTAND the deployment: call analyze_domain_shift and
  inspect_model EXACTLY ONCE each (architecture + normalization tell you which adaptations
  apply: BN-recalibration / tent only help with BatchNorm; for InstanceNorm/LayerNorm they
  are no-ops). These are ONE-TIME diagnostics — NEVER call analyze_domain_shift or
  inspect_model more than once; a repeat returns nothing new and wastes a turn. After the
  two diagnostics, you MUST start ACTING: build candidate adaptations.
- Diagnostics ≠ progress. Build at least 2-3 candidate adaptations (with real adaptation
  tools) before you consider request_labels, keep_source, or escalate. Do NOT loop on
  analyze/inspect/diagnose/evaluate — an unchanged model returns no new information.
- If quality is already high (e.g. >0.9 and VLM-usable), SHIP immediately.
- If quality is poor, diagnose the failure ONCE, then TRY the adaptations that fit the
  diagnosis (each adaptation is a new candidate worth evaluating), and ship the best
  SAFE one.
- Trust the EVIDENCE over your assumptions. The VLM signal is a PAIRWISE preference:
  vlm_prefer = fraction of images where the VLM judges that candidate BETTER than the
  source. Prefer the candidate with the highest vlm_prefer; only ship one with
  vlm_prefer > 0.5 (the VLM judges it better than source).
- TRY LABEL-FREE ADAPTATION FIRST — it needs no labels. Your toolbox has many label-free
  tools (bn_reset, tent, sar, eata; preprocessing like histogram_match/clahe/gamma_correct;
  spatial; frequency; postprocessing). Try the ones your domain-shift + model analysis
  suggest (BN/stats shift → bn_reset; decision drift → tent/sar/eata; intensity/contrast →
  preprocessing; spatial/FOV → spatial). Build a few candidates before reaching for labels.
- ONLY IF several label-free adaptations still leave the best candidate POOR, or the VLM
  preferences are inconclusive (all near 0.5 / nothing clearly beats source), call
  request_labels with NO K argument (it defaults to your FULL authorized budget). That returns labeled_dice (held-out few-shot Dice for EVERY
  candidate).
- ONCE labeled_dice EXISTS it is GROUND TRUTH and OVERRIDES vlm_prefer and quality
  (which are unreliable estimates). Ship the candidate with the HIGHEST labeled_dice
  as long as it exceeds source's labeled_dice, trust_level="label_verified" — EVEN IF
  its vlm_prefer looked low. NEVER keep_source when an adaptation has a higher labeled
  Dice than source: that is leaving a real improvement on the table.
- If even the best model is still UNUSABLE after label-free TTA, the K labels can TRAIN
  the model, not just rank it: call the finetune_* tools (head → partial → lora), which
  train on the TRAIN split and are re-ranked on the HELD-OUT split. Exhaust the
  fine-tune ladder BEFORE concluding the shift is unfixable.
- ORDER OF ESCALATION (do NOT skip steps): (1) try several LABEL-FREE adaptations first;
  (2) if still poor, request_labels and climb the finetune ladder (head→partial→lora);
  (3) only after BOTH label-free TTA and fine-tuning fail to reach usable, call
  recommend_alternative + escalate. Do NOT jump to labels/escalation while untried
  label-free tools remain — and fine-tuning often rescues what analysis calls "unfixable".
- ONLY if even fine-tuning the few labels stays UNUSABLE (a fundamental shift, e.g.
  CT->MRI needing many labels), do NOT ship a worthless "improvement": use
  recommend_alternative to get modality-appropriate pretrained models + a fine-tuning
  label estimate, then ESCALATE with that recommendation embedded in report.summary.
- After ~3 adaptations with no clear winner and no labels left, keep_source if
  acceptable, else ESCALATE. Never loop the same tool; never ship what you can't defend.
- When a LABEL-VERIFIED DECISION or UNFIXABLE directive block is present, FOLLOW IT
  EXACTLY (it is computed from ground truth) — do not re-derive the comparison yourself.

Respond with ONLY a single JSON object: {"tool": "<name>", "args": {...}}
To finish, call a terminal tool (ship | keep_source | escalate) with
args.report = {"trust_level": "label_verified"|"label_free_estimate",
               "confidence": <0..1>, "summary": "<one line>"}.
Claim label_verified ONLY if you actually used K labels."""


def _available(view: dict):
    # PLAN-FIRST (structural): when the task exposes a 'plan' tool, repair tools stay OFF the
    # menu until a plan has been committed (evidence['plan'] exists). Observation tools remain.
    # Lesson from the label-spend failure: prompt-only rules get ignored; the MENU shapes behavior.
    _ev0 = view.get("evidence", {}) or {}
    _menu_names = {t["name"] for t in view.get("tools", [])}
    if "plan" in _menu_names and not _ev0.get("plan"):
        _OBS = {"load_and_deploy", "analyze_domain_shift", "inspect_model", "diagnose_cls",
                "diagnose_masks", "diagnose_failure_mode", "plan", "evaluate",
                "keep_source", "escalate", "ship"}
        view = dict(view, tools=[t for t in view.get("tools", []) if t["name"] in _OBS])
    """STRUCTURAL anti-dither: drop one-time diagnostics from the menu once done, so the
    agent physically cannot loop on them (prompt-only 'don't repeat' rules get ignored)."""
    ev = view.get("evidence", {}) or {}
    done = set()
    if view.get("deployed"):
        done.add("load_and_deploy")
    if ev.get("domain_shift"):
        done.add("analyze_domain_shift")
    if ev.get("model_info"):
        done.add("inspect_model")
    out = [t for t in view["tools"] if t["name"] not in done]
    no_finetune = os.environ.get("NO_FINETUNE", "0") != "0"
    if no_finetune:   # labels-for-SELECTION regime: rank/verify, do NOT train
        out = [t for t in out if not str(t["name"]).startswith("finetune")]
    # TOOL-ORDER FIX (2026-06): in the NO_FINETUNE / labels-for-SELECTION regime, the K labels
    # can ONLY RANK the candidate POOL (finetune is off). If request_labels is revealed while
    # the pool is still thin, the agent reveals labels early, ranks "source vs a couple of weak
    # tools" (e.g. only bn_reset, a no-op on InstanceNorm models), sees nothing beats source,
    # and keeps source -- it NEVER ships the proven label-free recovery (denoise_histmatch
    # +0.147). The robust structural fix: keep request_labels OFF the menu until the label-free
    # adaptation pool is EXHAUSTED -- every label-free tool ON THE MENU has been tried (built)
    # at least once. Then request_labels appears and the agent ranks the FULL pool, exactly as
    # the no-LLM mechanism proof does (build all 8 -> rank -> ship the winner). This mirrors the
    # existing one-time-diagnostic drop pattern (prompt-only "build first" rules get ignored).
    #
    # Regime-scoped: ONLY in NO_FINETUNE. In the finetune regime (request_labels -> finetune)
    # labels TRAIN the model, so requesting them before exhausting label-free TTA is the CORRECT
    # escalation flow (head->partial->lora often rescues what TTA cannot) and this gate must NOT
    # fire. Also inert once labels are revealed (labeled_dice exists) -- re-requesting is a no-op.
    if no_finetune and not ev.get("labeled_dice"):
        menu_now = {t["name"] for t in out}
        tried = {str(c.get("method", "")) for c in (view.get("candidates_tried") or [])}
        # label-free adaptation tools the agent may build (whatever the task put on the menu).
        untried_lf = (menu_now & _LABELFREE_TOOLS) - tried
        # n_trials budget caps how many candidates can be built (each scored adaptation costs
        # one trial); keep request_labels gated only WHILE the agent both still has untried
        # label-free tools AND the trial budget to build them. So the agent spends its whole
        # build budget on candidates BEFORE revealing labels, then ranks the full pool it could
        # afford. Unlock as soon as either is exhausted (avoids an unsatisfiable gate when the
        # trial budget is smaller than the tool pool -> would otherwise force-escalate).
        trials_left = float((view.get("budget_remaining") or {}).get("n_trials", 1e9) or 0.0)
        if untried_lf and trials_left > 0:
            out = [t for t in out if t["name"] != "request_labels"]
    return out


# the label-free ADAPTATION tools (no labels needed); request_labels stays gated in the
# NO_FINETUNE / selection regime until every one of these that is ON THE MENU has been tried,
# so the agent ranks the FULL pool rather than a thin "source vs one weak tool" comparison.
_LABELFREE_TOOLS = frozenset({
    "bn_reset", "tent", "sar", "eata",
    "histogram_match", "clahe", "gamma", "gamma_correct", "percentile_clip",
    "fda", "fda_warmup", "spectral", "tta_aug", "crop_center", "spatial",
    "denoise", "denoise_histmatch", "norm_stat_tent",
    "largest_connected_component", "weight_average_source_adapted",
    # cls INPUT-SPACE color/intensity label-free tools (harness/labelfree_2d_tools.py) —
    # the on-target lever for cross-camera / cross-source COLOR & illumination shifts that
    # bn_reset/tent (BN-stat only) cannot fix. Keep request_labels gated until these are tried.
    "histogram_match_src", "match_intensity_stats", "fda_src", "gamma_src",
    "zscore_to_src", "percentile_normalize",
})


# order in which the fallback auto-tries label-free adaptations when the LLM stalls.
# cls color tools are placed EARLY for cls color/illumination shifts (their on-target lever).
_FALLBACK_ADAPT = ["bn_reset", "tent", "histogram_match_src", "match_intensity_stats",
                   "fda_src", "gamma_src", "zscore_to_src", "percentile_normalize",
                   "sar", "eata", "gamma_correct", "percentile_clip",
                   "histogram_match", "fda_warmup", "spectral", "crop_center", "clahe",
                   "largest_connected_component", "weight_average_source_adapted"]


def _fallback_action(view: dict):
    """When the LLM produces no valid on-menu action (i.e. it tried to dither), auto-advance:
    the next untried label-free adaptation → else request_labels → else None (let escalate)."""
    tried = {c.get("method") for c in (view.get("candidates_tried") or [])}
    names = {t["name"] for t in _available(view)}
    _order = list(_FALLBACK_ADAPT)
    if os.environ.get("TOOL_ADVICE", "0") == "1":   # Component D: diagnosis-aware stall order
        try:
            from harness.tool_advisor import build_tool_advice, reorder_menu
        except ImportError:
            from tool_advisor import build_tool_advice, reorder_menu
        _order = reorder_menu(_order, build_tool_advice(
            view.get("evidence", {}) or {}, [{"name": n} for n in _order]))
    for t in _order:
        if t in names and t not in tried:
            return {"tool": t, "args": {}}
    ev = view.get("evidence", {}) or {}
    # NOTE: we deliberately do NOT auto-request labels here. Spending a clinician's annotation
    # budget must be a reasoned decision (see _label_spend_nudge), never the automatic
    # consequence of the LLM stalling. Returning None lets the loop conclude (keep_source /
    # escalate) instead of silently buying labels the evidence does not justify.
    return None


def _advisory_section(ev: dict) -> str:
    """Design §A.4: a dedicated prompt section for ADVISORY blobs (tool suggestions, retrieved
    lessons, shift rationale), rendered from their OWN evidence keys OUTSIDE the 600-char
    LATEST EVIDENCE dump so they never evict load-bearing keys. Empty (inert) until components
    C/D/E populate those keys; each blob is length-capped."""
    try:                                  # robust to both repo-root and harness/-on-path contexts
        from harness.evidence_keys import get_characterization
    except ImportError:
        from evidence_keys import get_characterization
    lines = []
    ta = ev.get("tool_advice")
    if ta:
        lines.append(f"- suggested tools (advisory, NOT constraints): {str(ta)[:400]}")
    rl = ev.get("retrieved_lessons")
    if rl:
        lines.append(f"- lessons from past runs (advisory priors): {str(rl)[:400]}")
    why = get_characterization(ev).get("why")
    if why:
        lines.append(f"- shift rationale: {str(why)[:400]}")
    if not lines:
        return ""
    return "\n[ADVISORY — diagnosis-guided, NOT constraints]\n" + "\n".join(lines)


def _recipe_design_block(view: dict) -> str:
    """Invite the LLM to DESIGN the fine-tune recipe from the characterization and REVISE from the
    last fine-tune's diagnostics (observe->revise loop). Reasoning guidance, NOT a lookup table —
    no per-shift recipe is hardcoded; the agent decides lr/steps/capacity. Fires only when a
    finetune_* tool is on the menu."""
    ev = view.get("evidence", {}) or {}
    menu = {t.get("name") for t in (view.get("tools") or [])}
    if not any(str(m).startswith("finetune") for m in menu):
        return ""
    try:
        from harness.evidence_keys import get_characterization
        ch = get_characterization(ev) or {}
    except Exception:
        ch = ev.get("domain_shift") or {}
    st = ch.get("shift_type", "?"); sev = ch.get("severity", "?")
    k = view.get("k_labels_used") or 0
    di = ev.get("ft_diagnostics") or []
    s = ("\n*** RECIPE DESIGN — YOU design the fine-tune recipe: pass args lr= and steps= to any "
         "finetune_* tool (and choose the capacity tier head/partial/lora/full); do not accept a "
         f"fixed default. Reason from THIS case (shift={st}/{sev}, labels~{k}) — heuristics to reason "
         "with, NOT rules: noisy/severe shift or few labels -> gentler (lower lr ~1e-4..3e-4, fewer "
         "steps, head/lora); clean large gap + adequate labels -> push harder (full/partial, more steps). "
         "A deterministic best-checkpoint + never-worse-than-source guard protects you, so you can probe.")
    if di:
        s += ("\n    OBSERVE->REVISE: your last fine-tune reported: " + "; ".join(di) +
              ". RE-CALL a finetune_* with a revised recipe (collapse/overfit -> lr/4, fewer steps, or "
              "smaller capacity; underfit -> more steps / higher capacity) before you decide.")
    return s + " ***"


def _labeled_directive(view: dict) -> str:
    """BUDGET x VERIFIED-GAIN decision spine (design Component A). The K>0 regime as a clean
    directive with EXACTLY ONE numeric gate (the shared safety margin) — the interacting
    thresholds (usable_bar / good-enough ceiling / 0.5*usable floor) are deleted. Adequacy and
    escalation are LLM-CATEGORICAL, grounded in the looked-up SOTA/usable. Consumes the scaling
    curve. Reads everything via .get — writes nothing, CANNOT raise."""
    ev = view.get("evidence", {}) or {}
    ld = ev.get("labeled_dice")
    if not isinstance(ld, dict) or not ld:
        return ""
    sd = float(ld.get("source", 0.0) or 0.0)                  # NEVER ld['source']
    best = ev.get("best_by_labels")
    cd = float(ld.get(best, sd) or sd) if best else sd
    margin = float(ev.get("safety_margin", 0.05))   # single source of truth (§A.1): flat noise floor
    # a candidate the VERIFIER passed (paired / LOO-OOF) beats source even if the coarse 2-slice
    # holdout score reads lower (brats: LOO +0.10 mean, holdout 0.347 vs 0.412 -> was kept source)
    beats = bool(best) and best != "source" and (cd >= sd + margin or bool(ev.get("best_verified")))
    sota = ev.get("expected_metric"); usable = ev.get("usable_floor")
    rat = str(ev.get("perf_rationale", "") or "")[:120]
    sl = ev.get("scaling_law") if isinstance(ev.get("scaling_law"), dict) else None
    menu = {t.get("name") for t in (view.get("tools") or [])}
    LADDER = ["finetune_head", "finetune_partial", "finetune_lora"]
    br = view.get("budget_remaining") or {}
    compute_left = float(br.get("compute_minutes", 1e9) or 0.0)     # ladder gated on COMPUTE only
    ladder_left = next((m for m in LADDER if m not in ld and m in menu and compute_left >= 3.0), None)

    target = (f"the looked-up target (near-SOTA {sota}, usable {usable}" + (f": {rat}" if rat else "") + ")"
              if (sota is not None or usable is not None)
              else "the source baseline + the candidate table (no looked-up target available)")

    def b_sl():
        if not sl:
            return ""
        fit = sl.get("fit")
        fit = fit if isinstance(fit, dict) else {}            # defensive — never .get on a str
        N = sl.get("needed_for_usable")
        nphr = (f"to reach usable {usable} annotate ≈{N} more labels" if N is not None
                else "the fitted curve does not reach usable by extrapolation (a better model may be needed)")
        return (f"\n    SCALING LAW [{fit.get('form')}] {fit.get('formula','')}: measured "
                f"{sl.get('measured')}; {nphr}. Hand this curve VERBATIM to the clinician in report.summary.")

    if beats:                                                  # B0 — verified, safe gain
        def _f(x):
            try: return float(x)
            except Exception: return None
        uf, sf = _f(usable), _f(sota)
        headroom = (uf is not None and cd < uf) or (sf is not None and cd < sf)
        stronger = ladder_left or ("scaling_eval" if "scaling_eval" in menu else None)
        if stronger and compute_left >= 3.0:
            climb = (f" BUT compute remains and a STRONGER lever is untried ({stronger}): '{best}' "
                     f"({cd:.3f}) is a SAFE FLOOR, not necessarily the best you can do. Before shipping, "
                     f"DESIGN one stronger fine-tune for THIS shift (choose capacity + args lr=/steps=; a "
                     f"clean large gap can take a higher-capacity/higher-lr recipe) — the best-checkpoint "
                     f"guard keeps you >= source, so it is safe to probe — then ship the BEST candidate."
                     + (" There is headroom to the usable target." if headroom else " Ship now only if you judge it already adequate."))
        else:
            climb = ""
        return (f"\n*** LABEL-VERIFIED GAIN: '{best}' ({cd:.3f}) clearly beats source ({sd:.3f}, "
                f"margin {margin:.3f}) — a real, safe improvement. Compare {cd:.3f} to {target}. If "
                f"you judge it ADEQUATE → ship args {{\"candidate\": \"{best}\", \"report\": "
                f'{{"trust_level": "label_verified", "confidence": 0.85, '
                f'"summary": "recovered via {best}"}}}}.{climb} ***{b_sl()}')
    if "scaling_eval" in menu and not sl:                      # B-SCALE — the HONEST few-shot method
        return (f"\n*** LABEL-VERIFIED: best {cd:.3f}, source {sd:.3f}; nothing yet beats source within "
                f"the margin {margin:.3f}. Before the manual finetune ladder, call scaling_eval now "
                f"(no args): it runs k-fold CROSS-VALIDATION fine-tuning across label budgets for the "
                f"HONEST delivered-model estimate — the manual single-holdout UNDER-estimates the gain "
                f"— plus an 'annotate N -> expect X' curve. Then re-judge with that gate. ***")
    if ladder_left:                                            # B1 — no gain yet, compute remains
        return (f"\n*** LABEL-VERIFIED: best {cd:.3f}, source {sd:.3f}; nothing yet beats source within "
                f"the margin {margin:.3f}. The K labels can TRAIN, not just rank. Compute allows "
                f"{ladder_left}: call it now (no args); held-out re-rank is automatic. Climb "
                f"head→partial→lora, then re-judge. ***{b_sl()}")
    # B2 -- no verified gain, ladder exhausted/unaffordable. Source is the BEST available model;
    # DEFAULT to keep_source. Escalation delivers source ANYWAY but wrongly implies we found
    # something better -- reserve it for a GENUINELY UNUSABLE source (not merely below an
    # aspirational SOTA). Decision-spine robustness fix (2026-06).
    rec = ev.get("recommendation")
    head = (f"\n*** LABEL-VERIFIED, ladder exhausted: nothing beats source (best {cd:.3f} vs source "
            f"{sd:.3f}) within the safety margin -- SOURCE IS THE BEST AVAILABLE MODEL. Target: {target}.\n"
            f"    DEFAULT -> keep_source: deliver the best available; the honest report flags it if it "
            f"is below the usable floor. Do NOT escalate just because source is below an aspirational "
            f"SOTA -- escalation delivers source ANYWAY but wrongly implies you found something better, "
            f"and fine-tuning a good source can DEGRADE it.\n"
            f"    ONLY escalate if source is CLINICALLY UNUSABLE (genuinely broken / near-zero) and "
            f"fine-tuning could not rescue it within budget.")
    if rec:
        return (head + f"\n    A grounded alternative EXISTS: {json.dumps(rec, default=str)[:400]}\n    "
                f'>> if source is UNUSABLE -> escalate, args {{"report": {{"trust_level": "label_verified", '
                f'"confidence": 0.3, "summary": "<verdict + alternative_models + finetune_labels_estimate>"}}}}; '
                f"otherwise keep_source (report it honestly). ***{b_sl()}")
    return (head + f"\n    >> if source is usable-enough -> keep_source NOW (honest report); ONLY if source "
            f"is genuinely unusable -> call recommend_alternative (no args) for a grounded alternative. "
            f"***{b_sl()}")


def _verify_nudge(view: dict) -> str:
    """When labels are still AFFORDABLE and none used yet, nudge the agent to VERIFY a
    label-free candidate on labels before shipping it (a label_verified delivery beats an
    unverified estimate). At a TRUE 0-label budget this is silent -> the permissive
    label-free ship is correct (design: budget-aware ship policy)."""
    if not view.get("deployed") or int(view.get("k_labels_used", 0) or 0) > 0:
        return ""
    k = int((view.get("budget_remaining") or {}).get("k_labels", 0) or 0)
    if k <= 0 or not view.get("candidates_tried"):
        return ""
    return (f"\nVERIFY-BEFORE-SHIP: you have {k} labels available and have used none. Before "
            f"shipping a LABEL-FREE adapted candidate, call request_labels (no K argument = your full budget) to VERIFY it on "
            f"labels \u2014 a label_verified delivery is strictly better than an unverified estimate "
            f"when labels are affordable. (At a TRUE 0-label budget you cannot verify; shipping the "
            f"best label-free candidate as an honest label_free_estimate is then correct.)")


# the HIGH-VALUE input-space color/stain tools — the on-target lever for cross-source COLOR /
# STAIN shifts (histopath H&E, cross-camera). bn_reset/tent (BN-stat only) give a small bump but
# CANNOT fix a stain/color cast; these are the ones that recover the big delta (Camelyon17 phaseA:
# zscore_to_src/macenko_stain/match_intensity_stats ~+0.085 AUROC vs bn_reset ~+0.017).
_STRONG_COLOR_TOOLS = frozenset({
    "zscore_to_src", "match_intensity_stats", "macenko_stain", "reinhard_stain",
    "gamma_src", "histogram_match_src", "fda_src",
})


def _k0_pool_nudge(view: dict) -> str:
    """K=0 (TRUE label-free permissive-ship) build-the-strong-tools nudge. Symmetric partner of
    _selection_pool_nudge (which only fires in NO_FINETUNE). At a 0-label budget the agent ships
    the best LABEL-FREE candidate as an UNVERIFIED estimate — but it tends to terminate on the
    FIRST marginal win (bn_reset +0.05) before building the high-value stain/color tools that
    recover the BIG delta (+0.085). bn_reset/tent only touch BN stats and CANNOT fix a stain/color
    cast; the on-target lever for a color/stain/illumination shift is the input-space tools. So
    while strong color/stain tools remain UNTRIED, push the agent to BUILD them before shipping a
    marginal candidate. Fires ONLY at a true 0-label budget (k_labels<=0) — when labels ARE
    affordable, _selection_pool_nudge / _verify_nudge own the build-then-verify flow. Silent once
    every strong color tool on the menu has been built or once labels are revealed."""
    ev = view.get("evidence", {}) or {}
    if ev.get("labeled_dice"):
        return ""
    k_avail = int((view.get("budget_remaining") or {}).get("k_labels", 0) or 0)
    if k_avail > 0:                      # labels affordable -> the verify/selection nudges drive
        return ""
    menu = {t["name"] for t in view.get("tools", [])}
    tried = {str(c.get("method", "")) for c in (view.get("candidates_tried") or [])}
    built_strong = sorted((_STRONG_COLOR_TOOLS & menu) & tried)
    untried_strong = sorted((_STRONG_COLOR_TOOLS & menu) - tried)
    # is the diagnosed shift a COLOR / STAIN / INTENSITY one (where input-space tools are the lever
    # and BN-stat confidence is an over-sharpened, unreliable proxy)?
    ds = ev.get("domain_shift") or {}
    axis = str(ds.get("dominant_axis", "") or ds.get("shift_type", "")).lower()
    color_shift = any(w in axis for w in ("intensity", "contrast", "color", "stain", "histogram")) \
        or any(w in str(view.get("goal", "")).lower() for w in ("stain", "h&e", "histopath"))
    trials = (view.get("budget_remaining") or {}).get("n_trials", "?")
    cl = "(clahe / percentile_normalize are source-AGNOSTIC contrast ops — they do NOT match the " \
         "source stain and tend to COLLAPSE on histopathology; the safety gate will veto them.)"
    if untried_strong:                   # PHASE 1 — still building: build the strong pool first
        return (f"\nK=0 LABEL-FREE SHIP — BUILD THE STRONG POOL FIRST: you have NO labels (permissive "
                f"ship as an UNVERIFIED estimate). Do NOT terminate on the first marginal win "
                f"(bn_reset/tent touch only BN stats — a small bump that CANNOT fix a stain/color/"
                f"illumination cast). The on-target lever for a COLOR/STAIN shift is the input-space "
                f"tools; build them and ship the BEST. Untried high-value tools (n_trials left="
                f"{trials}): {', '.join(untried_strong)}. Build these BEFORE shipping. {cl}")
    if color_shift and built_strong:     # PHASE 2 — pool built: ship the best COLOR tool BY NAME
        return (f"\nK=0 SHIP-THE-RIGHT-TOOL: the strong input-space color/stain pool is BUILT "
                f"({', '.join(built_strong)}). This is a COLOR/STAIN/INTENSITY shift, so SHIP one of "
                f"these BY NAME (ship args {{\"candidate\": \"<tool>\"}}). Do NOT default to "
                f"bn_reset/tent: BN-stat recalibration SHARPENS the softmax so its label-free "
                f"confidence/quality reads ARTIFICIALLY HIGH (over-confident), but it only touches BN "
                f"statistics and CANNOT correct a stain/color cast — the input-space color tool is the "
                f"real on-target fix here. Prefer the input-space color/stain candidate over a "
                f"BN-stat tool whose only edge is an over-sharpened confidence. {cl}")
    return ""


def _label_spend_nudge(view: dict) -> str:
    """THE MISSING DECISION: is it worth spending the clinician's labels at all?

    Measured problem (2026-06-26): the harness requested labels on 11/11 cls tasks but only 3
    produced a gain -- i.e. it burned annotation budget on 8 tasks where the source was already
    adequate or the shift was unrecoverable, giving it NO label-efficiency advantage over just
    fine-tuning everything. Labels are the one cost a clinician actually pays, so requesting them
    must be a JUSTIFIED decision, not the default next move after the label-free pool empties.

    This surfaces the evidence for/against spending and reminds the model that concluding WITHOUT
    labels (keep_source / escalate) is a legitimate, often correct outcome. Advisory only -- the
    LLM still decides; we never hard-block request_labels."""
    ev = view.get("evidence", {}) or {}
    if ev.get("labeled_dice"):
        return ""                                   # labels already revealed
    names = {t["name"] for t in _available(view)}
    if "request_labels" not in names:
        return ""
    cands = view.get("candidates_tried") or []
    if not cands:
        return ""                                   # nothing built yet; let it explore first
    qs = [(str(c.get("method","")), c.get("quality")) for c in cands if c.get("quality") is not None]
    src_q = next((q for n,q in qs if n == "source"), None)
    best = max([(q,n) for n,q in qs if n != "source"], default=None)
    ds = ev.get("domain_shift") or {}
    sev = str(ds.get("severity","")).lower()
    lines = []
    if src_q is not None and best is not None:
        lines.append(f"source quality {src_q:.3f} vs best label-free candidate '{best[1]}' {best[0]:.3f}")
    if sev:
        lines.append(f"diagnosed shift severity: {sev}")
    return ("\nSPEND LABELS ONLY IF THEY WILL CHANGE THE OUTCOME: request_labels costs a CLINICIAN "
            "real annotation time -- it is a decision, not a free next step. Evidence so far: "
            + ("; ".join(lines) if lines else "(label-free pool built)") + ". Before requesting, state "
            "what you would DO with the labels and why it is likely to pay off: (a) VERIFY a "
            "label-free candidate that already looks better than source, or (b) TRAIN a fine-tune "
            "because the shift looks fixable but no label-free tool fixed it. If NEITHER holds -- the "
            "source already looks healthy on this target, or the diagnosis says the shift is severe "
            "and nothing (including fine-tuning) is likely to recover it -- then do NOT spend the "
            "labels: conclude now with keep_source (source is fine) or escalate (unrecoverable). "
            "Concluding without spending the budget is a GOOD outcome, not a failure.")


def _plan_block(view: dict) -> str:
    """Carry the plan workflow IN THE PROMPT (never via the 600-char evidence dump, which
    truncates): before commit -> render the capability inventory + a one-call commit example;
    after commit -> render the committed plan and require explicit revision."""
    ev = view.get("evidence", {}) or {}
    plan = ev.get("plan")
    if plan:
        steps = plan.get("steps") or []
        dropped = plan.get("dropped_steps") or []
        drop_note = ("\n  NOTE: these planned names were NOT real tools and were dropped: "
                     + ", ".join(map(str, dropped)) + " -- if you meant a real tool (e.g. a stain/"
                     "histogram repair), revise the plan with its exact menu name.") if dropped else ""
        return ("\nYOUR COMMITTED PLAN: " + " -> ".join(map(str, steps)) + drop_note +
                f"\n  rationale: {plan.get('rationale','')}"
                "\n  Follow it. Deviate ONLY by calling plan again with revised steps and what "
                "observation changed your mind -- silent deviation is not allowed.")
    if not any(t["name"] == "plan" for t in view.get("tools", [])):
        return ""
    bad = ev.get("plan_invalid")
    if bad:
        return ("\nYOUR PLAN WAS REJECTED -- it referenced tools that do not exist: "
                + ", ".join(map(str, bad.get("unknown_steps", [])))
                + ". Commit again using ONLY these exact names: "
                + ", ".join(map(str, bad.get("valid_tools", []))))
    inv = ev.get("inventory")
    if not inv:
        return ("\nPLAN FIRST: repair tools are locked until you commit a plan. Call "
                '{"tool": "plan", "args": {}} once to fetch the capability inventory '
                "(which repairs THIS model admits and what each costs in labels); it will be "
                "shown here next turn, then you commit.")
    import json as _json
    rep_lines = []
    for r in inv.get("repairs", []):
        mark = "+" if r.get("applicable") else "x(EXCLUDED)"
        rep_lines.append(f"  [{mark}] {r.get('name')}  labels={r.get('label_cost')}  -- {r.get('fixes')}")
    bud = inv.get("budget", {}); dat = inv.get("data", {})
    return ("\nCAPABILITY INVENTORY (reason from THIS, then COMMIT in ONE call):\n"
            + "\n".join(rep_lines)
            + f"\n  budget: labels_available={bud.get('labels_available')} | data: n={dat.get('n_target')}"
              f" classes={dat.get('n_classes')} minority_frac={dat.get('minority_fraction')}"
            + "\nNOW COMMIT THE PLAN -- one call, exactly this shape:\n"
              '  {"tool": "plan", "args": {"steps": ["<tool1>", "<tool2>", "..."], '
              '"rationale": "<why THIS order fits the diagnosed shift, this architecture, and the '
              'label budget>"}}\n'
              "Order cheap-to-expensive: label-free repairs first; request_labels ONLY if the plan "
              "needs verification or training; concluding without labels (keep_source/escalate) is "
              "a valid plan when the source is adequate or the shift looks unrecoverable. Repair "
              "tools unlock the moment the plan is committed.")


def _selection_pool_nudge(view: dict) -> str:
    """NO_FINETUNE (labels-for-SELECTION) prompt reinforcement of the structural gate in
    _available: you CANNOT fine-tune here, so the K labels can ONLY rank the candidate POOL.
    Build the FULL label-free pool BEFORE revealing labels — request_labels is intentionally
    OFF the menu until you have. Try the input-space tools that match the diagnosis (intensity/
    histogram/noise -> denoise_histmatch, denoise, histogram_match, gamma), not just bn_reset/
    tent (no-ops on InstanceNorm/GroupNorm models). Silent once labels are revealed or in the
    finetune regime."""
    if os.environ.get("NO_FINETUNE", "0") == "0":
        return ""
    ev = view.get("evidence", {}) or {}
    if ev.get("labeled_dice"):
        return ""
    untried = sorted((_LABELFREE_TOOLS & {t["name"] for t in view.get("tools", [])})
                     - {str(c.get("method", "")) for c in (view.get("candidates_tried") or [])})
    if not untried:
        return ""
    trials = (view.get("budget_remaining") or {}).get("n_trials", "?")
    return (f"\nSELECTION REGIME (NO fine-tuning): the {view['budget_remaining'].get('k_labels','?')} "
            f"labels can ONLY RANK candidates — you cannot train. request_labels is therefore HIDDEN "
            f"until you BUILD the label-free pool; build candidates now (n_trials left={trials}). Spend "
            f"your build budget on the tools that fit the diagnosed shift — for an intensity/histogram/"
            f"noise shift try denoise_histmatch, denoise, histogram_match, gamma (NOT just bn_reset/tent, "
            f"which are no-ops on InstanceNorm/GroupNorm). Untried label-free tools: {', '.join(untried)}. "
            f"Build the strong ones, THEN request_labels to rank the full pool and ship the best.")


def build_prompt(view: dict) -> str:
    ev = view.get("evidence", {}) or {}
    menu_tools = _available(view)
    # Component D (advisory tool coupling) -- gated; default OFF = byte-inert. Reorders the
    # PRESENTED recovery menu (STABLE permutation; _available/valid-set untouched) and stashes
    # a diagnosis shortlist for the section-A.4 advisory. NEVER filters / forces / guards.
    _adv_ev = None
    if os.environ.get("TOOL_ADVICE", "0") == "1":
        try:
            from harness.tool_advisor import build_tool_advice, reorder_menu
        except ImportError:
            from tool_advisor import build_tool_advice, reorder_menu
        _adv = build_tool_advice(ev, menu_tools)
        if _adv.family_rank:
            _byn = {t["name"]: t for t in menu_tools}
            menu_tools = [_byn[n] for n in reorder_menu([t["name"] for t in menu_tools], _adv)]
        if _adv.shortlist:
            _adv_ev = _adv.as_evidence()
    tools = "\n".join(f"  - {t['name']}: {t['description']}"
                      + ("  [TERMINAL]" if t.get("terminal") else "")
                      for t in menu_tools)
    cand = view.get("candidates_tried") or []
    cand_tbl = ("\n".join(f"    {c.get('method','?'):<10} "
                          f"quality={c.get('metric_estimate')} "
                          f"vlm_prefer_over_source={c.get('vlm_prefer','n/a')}" for c in cand)
                or "    (none yet)")
    fb = view.get("guardrail_feedback")
    ev = view.get("evidence", {}) or {}
    # K>0 regime: the BUDGET x VERIFIED-GAIN decision spine (Component A). ONE numeric gate
    # (the safety margin); adequacy/escalation are LLM-categorical, grounded in looked-up SOTA.
    ld_block = _labeled_directive(view) if ev.get("labeled_dice") else ""
    parts = [
        SYSTEM,
        f"\nGOAL: {view['goal']}",
        f"TASK: {view['task']['type']} {view['task']['dims']} (metric {view['task']['metric']})",
        f"STEP: {view['step']}   deployed={view['deployed']}   "
        f"best_quality={view.get('best_metric_estimate')}   k_labels_used={view['k_labels_used']}",
        f"BUDGET REMAINING: {view['budget_remaining']}",
        f"CANDIDATES TRIED:\n{cand_tbl}",
        ld_block,
        _recipe_design_block(view),
        _plan_block(view),
        _selection_pool_nudge(view),
        _label_spend_nudge(view),
        _k0_pool_nudge(view),
        _verify_nudge(view),
        f"LATEST EVIDENCE: {json.dumps(ev, default=str)[:600]}",
        f"TOOLS:\n{tools}",
    ]
    if fb:
        parts.append(f"\n!! GUARDRAIL FEEDBACK from your last action: {fb}\n"
                     f"Adjust — do not repeat the blocked action.")
    adv = _advisory_section({**ev, "tool_advice": _adv_ev} if _adv_ev else ev)  # A.4 only; 600-char dump stays on original ev
    if adv:
        parts.append(adv)
    parts.append('\nYour next action (JSON only):')
    return "\n".join(parts)


_ESCALATE = {"tool": "escalate", "args": {"report": {
    "trust_level": "label_free_estimate", "confidence": 0.0,
    "summary": "policy could not produce a valid action"}}}


def parse_action(text: str, valid_tools: set):
    """Return the last complete, valid top-level action in the response.

    Consume each decoded JSON value as a whole, including non-action containers
    and quoted strings. Their nested objects are data, never separate actions.
    Prose and malformed fragments may precede a subsequent complete action.
    """
    if not isinstance(text, str) or not text:
        return None
    decoder = json.JSONDecoder()
    action = None
    pos = 0
    while pos < len(text):
        if text[pos] not in '{["':
            pos += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, pos)
        except (ValueError, RecursionError):
            # Skip the entire malformed container; its nested objects are data.
            start = text[pos]
            if start == '"':
                return action  # an unterminated string has no later top-level value
            stack = [start]
            quoted = escaped = False
            pos += 1
            while pos < len(text) and stack:
                char = text[pos]
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char in "{[":
                    stack.append(char)
                elif char in "}]":
                    if (stack[-1], char) not in (("{", "}"), ("[", "]")):
                        return action
                    stack.pop()
                pos += 1
            continue
        pos = end
        if not isinstance(obj, dict):
            continue
        tool = obj.get("tool")
        args = obj.get("args", {})
        if isinstance(tool, str) and tool in valid_tools and isinstance(args, dict):
            action = {"tool": tool, "args": args}
    return action


def make_llm_policy(call_llm: Callable[[str], str]) -> Callable[[dict], dict]:
    """Wrap an injected LLM with one bounded correction for invalid actions.

    Provider failures immediately escalate. Two invalid responses also escalate;
    neither path substitutes a tool or spends an adaptation/annotation budget.
    """
    def policy(view: dict) -> dict:
        valid = {t["name"] for t in _available(view)}
        prompt = build_prompt(view)
        for attempt in range(2):
            try:
                text = call_llm(prompt)
            except Exception:
                return {"tool": "escalate", "args": {"report": {
                    "trust_level": "label_free_estimate", "confidence": 0.0,
                    "summary": "LLM call failed; no further adaptation was attempted"}}}
            act = parse_action(text, valid)
            if act is not None:
                return act
            if attempt == 0:
                prompt += (
                    "\nACTION FORMAT CORRECTION (one attempt): Your previous response did not "
                    "contain a valid available action. Return exactly one JSON object with "
                    "a string tool name and an object-valued args field: "
                    '{"tool": "<available name>", "args": {}}. '
                    "Choose only from these currently available tools: "
                    + json.dumps(sorted(valid))
                    + ". Do not wrap the action in another object or array."
                )
        return {"tool": "escalate", "args": {"report": {
            "trust_level": "label_free_estimate", "confidence": 0.0,
            "summary": "LLM returned an invalid or unavailable action after one correction "
                       "attempt; no fallback action executed"}}}
    return policy
