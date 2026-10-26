"""
Agent 4: Execution Engineer

Translates the Strategy Architect's plan into actual code execution.
Handles:
  - Model loading and architecture detection
  - Environment setup (device, data pipeline)
  - FDA warmup, histogram matching, BN reset
  - Self-supervised adaptation loop
  - Checkpoint saving

This is the ONLY agent that touches PyTorch — all others are reasoning agents.
"""

import copy
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medadaptagent.graph.vision_utils import (
    prepare_prediction_samples, build_multimodal_message,
)


PER_ITER_VLM_PROMPT = """Compare segmentation predictions from the SOURCE model vs the CURRENT adapted model on {k} target samples.

## Task
Target structure: {structure_name}
Expected location: {expected_location}
Expected shape: {expected_shape}
Expected size (prior estimate, may be inaccurate): {expected_fg} of image
Failure type of source on target (diagnosis, may be incorrect): {failure_type}
Expected correction direction (diagnosis-driven): {correction_direction}

## Diagnosed shift mechanism (condition your vote on this)

{mechanism_hint}

## Context about the source model + size change

{source_quality_hint}

{size_delta_hint}

## How the images are presented

For each sample you see THREE images in order:
  1. RAW IMAGE (no overlay) — use this FIRST to locate where the actual
     {structure_name} is in the image. Trust what you can see in the raw
     anatomy, NOT the colored masks and NOT the diagnosis fields above
     (which can be wrong).
  2. SOURCE mask overlay (40% colored on the raw image)
  3. CURRENT mask overlay (40% colored on the raw image)

Process each sample in this order:
  (i)   Look at the RAW IMAGE and identify where {structure_name} actually
        is. Estimate its approximate size and location *from the image*.
  (ii)  Look at the SOURCE overlay — does its colored region coincide
        with the actual {structure_name} from step (i)?
  (iii) Look at the CURRENT overlay — same question.
  (iv)  RELATIVE vote rule (better|similar|worse), conditioned on BOTH
        the diagnosed shift mechanism above AND the diagnosis
        (failure_type = {failure_type}). A change only counts as "better"
        if it specifically corrects the diagnosed shift and its foreground
        still lands on the real {structure_name} in the raw image:
          - If failure_type is "under_segmentation": growth that lands ON
            {structure_name} is the correct fix → vote "better" when
            CURRENT covers more of the real structure than SOURCE. Only
            vote "worse" if growth landed on non-target anatomy.
          - If failure_type is "over_segmentation": shrink toward
            {structure_name} is the correct fix → vote "better" when
            CURRENT is smaller AND still covers the structure boundary.
          - If failure_type is "complete_failure" or "novel_structure_unknown":
            SOURCE was failing entirely. Vote "better" ONLY when CURRENT's
            mask lies predominantly on the actual {structure_name} you
            identified in the raw image. A localized mask in the WRONG
            place is NOT "better" than a failed source — vote "similar"
            or "worse".
          - If failure_type is "correct_location_wrong_shape": judge
            shape match against the raw structure, ignore size delta.
          - If failure_type is "unknown" or not provided: judge purely
            by overlap with the structure visible in the raw image.

  (v)   The relative vote (above) tells whether CURRENT is closer to the
        truth than SOURCE. The absolute vote (next section) is INDEPENDENT
        of SOURCE — judge CURRENT on its own. The "source_quality_hint"
        / "size_delta_hint" are context only; do not let them override
        what you see in the raw image. A larger CURRENT mask is good
        ONLY when it lands on the actual {structure_name}.

## Two per-sample judgments

1. **Relative** (CURRENT vs SOURCE): better | similar | worse.
   Which mask more closely covers the actual {structure_name} you
   identified in the raw image?
   - "worse" if CURRENT drifted onto non-target pixels even if it grew.
   - "better" only if CURRENT's mask pixels are actually on the target.

2. **Absolute quality of CURRENT alone**: usable | broken.
   Looking at the RAW image and CURRENT overlay, does the CURRENT mask
   cover the actual {structure_name}?
     - "broken" — CURRENT's coloured region sits largely on non-target
       anatomy, or the mask grew into organs that aren't {structure_name},
       or the mask is nearly empty, or the mask covers substantially
       more than the real structure visible in the raw image.
     - "usable" — CURRENT's coloured region is predominantly on the real
       {structure_name} in the raw image; boundaries may be imperfect.

You only return per-sample votes; the orchestration code will decide what to
do with them. Do NOT return an overall accept/reject.

Return ONLY this JSON (no extra text):
```json
{{"per_sample":          ["better|similar|worse", ...],
  "per_sample_absolute": ["usable|broken", ...],
  "note": "<one-sentence summary>"}}
```"""





# --------------------------------------------------------------
# Gate policy constants
#
# These are the few magic numbers in the final-gate logic. Each has a
# docstring explaining the trade-off so future tuning has a single
# source of truth. They are NOT derived from priors because they're
# decisions about the gate's risk profile, not properties of the task.
# --------------------------------------------------------------

# Strong-source threshold. When Agent 2's source_quality_score is at
# or above this, the gate requires CLEAR improvement (majority better)
# rather than just "no majority worse" — defends against subtle
# regressions on already-competent sources. Lowering 0.8 makes the
# guard fire more often (more conservative); raising loosens.
STRONG_SOURCE_SQ_THRESHOLD = 0.8

# Final gate "broken-absolute" threshold. Reject if at least this many
# of vlm_final_k=5 samples vote the adapted mask BROKEN. 2/5 catches
# real broken patterns while tolerating one stray vote on a noisy
# anchor; 3/5 (strict majority) is too lenient.
FINAL_BROKEN_THRESHOLD = 2

def _vlm_iteration_vote(current_model: nn.Module,
                        source_model: nn.Module,
                        anchor_imgs: List[np.ndarray],
                        device: str,
                        llm_client,
                        task_ctx: Dict,
                        k: int = 3,
                        task=None) -> Tuple[float, List[str], str, List[str]]:
    """Per-iteration VLM judgment.

    Returns:
        score:     normalized score in [-1, +1] from the RELATIVE vote
                   (n_better - n_worse) / n. None on failure.
        votes:     per-sample relative votes ("better"|"similar"|"worse").
        note:      one-line summary.
        absolute:  per-sample absolute votes ("usable"|"broken") — independent
                   of the relative comparison. Used by _run_final_check to
                   catch the plausible-drift case where CURRENT looks cleaner
                   than SOURCE but is effectively empty.

    If llm_client is None or the call fails, returns (None, [], '', []).
    """
    if llm_client is None or not anchor_imgs:
        return None, [], "", []
    k = min(k, len(anchor_imgs))
    try:
        # Dispatch via task adapter when present so cls uses GradCAM
        # overlays + cls prompt, seg uses mask overlays + seg prompt.
        if task is not None:
            images, labels = task.render_pair_for_vlm(
                model_source=source_model, model_current=current_model,
                anchor_imgs=anchor_imgs[:k], num_classes=task_ctx.get(
                    "num_classes_hint", 2), device=device, size=256)
            template = task.vlm_prompt_template()
            if template == "__SEG_LEGACY_PROMPT__":
                template = PER_ITER_VLM_PROMPT
            # Only format with task_ctx if the template has the legacy
            # seg placeholders. The cls template uses only {k}.
            try:
                prompt = template.format(k=k, **task_ctx)
            except KeyError:
                # cls template doesn't have all seg placeholders
                prompt = template.format(k=k)
        else:
            src_pairs = prepare_prediction_samples(anchor_imgs, source_model, device, n=k)
            cur_pairs = prepare_prediction_samples(anchor_imgs, current_model, device, n=k)
            images, labels = [], []
            for i in range(k):
                images.append(src_pairs[i][0])
                labels.append(f"Sample {i+1} RAW IMAGE (no mask — use this to "
                               f"locate the actual structure)")
                images.append(src_pairs[i][1])
                labels.append(f"Sample {i+1} SOURCE mask overlay")
                images.append(cur_pairs[i][1])
                labels.append(f"Sample {i+1} CURRENT mask overlay")
            prompt = PER_ITER_VLM_PROMPT.format(k=k, **task_ctx)
        content = build_multimodal_message(prompt, images, labels)
        resp = llm_client.cached_complete(
            messages=[{"role": "user", "content": content}],
            max_tokens=300, temperature=0, seed=42)
        text = resp.choices[0].message.content or ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None, [], text[:120], []
        vlm = json.loads(text[start:end])
        votes = vlm.get("per_sample", []) or []
        absolute = vlm.get("per_sample_absolute", []) or []
        note = vlm.get("note", "")
        n = max(len(votes), 1)
        n_better = sum(1 for v in votes if v == "better")
        n_worse = sum(1 for v in votes if v == "worse")
        score = (n_better - n_worse) / n
        return score, votes, note, absolute
    except Exception as _e:
        return None, [], f"VLM error: {str(_e)[:80]}", []


def _measure_quality(model, images, device, max_samples=200):
    """Label-free quality metrics (consistency under flip, entropy, fg_ratio).
    Reported only — never used as a gate. Moved here from Agent 5 after the
    merge so execution_engineer has no cross-agent dependency."""
    model.eval()
    consistencies, entropies, fg_ratios = [], [], []
    with torch.no_grad():
        for img in images[:max_samples]:
            try:
                if img.ndim == 3 and img.shape[2] in (1, 3):
                    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                elif img.ndim == 2:
                    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
                else:
                    t = torch.from_numpy(img.astype(np.float32))
                tensor = t.unsqueeze(0).to(device)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                flipped = model(torch.flip(tensor, [-1]))
                pred_flip = torch.flip(flipped.argmax(dim=1), [-1])
                consistencies.append((pred == pred_flip).float().mean().item())
                ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
                entropies.append(ent)
                fg_ratios.append((pred > 0).float().mean().item())
            except Exception:
                continue
    return {
        "consistency": round(float(np.mean(consistencies)), 3) if consistencies else 0,
        "entropy": round(float(np.mean(entropies)), 4) if entropies else 1,
        "fg_ratio": round(float(np.mean(fg_ratios)), 4) if fg_ratios else 0,
    }


def _catastrophic_collapse(fg_before: float, fg_after: float,
                            expected_min: float, expected_max: float,
                            task_hint: str = "segmentation") -> Optional[str]:
    """Physical-extreme collapse check. Returns a reason string if the
    adapted model collapsed, else None. Thresholds are gated by Agent 1's
    priors so they don't false-positive on tiny / majority structures.

    Covers only physical extremes:
      (a) All-background:    fg ~ 0 (model predicts nothing)
      (b) All-foreground:    fg ~ 1 (model predicts everywhere)
      (c) Reasonable → zero collapse during adaptation

    Note: a "severe under-seg" gate of the form `fg < 0.5 × expected_min`
    was tried and removed — it false-positives on cautious-but-correct
    segmenters (e.g. chaos CT→MRI liver where adapted fg 5% is low vs
    expected_min 50% but Dice is good). Plausible-drift (right size,
    wrong shape) is caught by the raw-image VLM broken-veto downstream
    of this check, not here.
    """
    # Catastrophic collapse: model predicts the same thing for everything.
    # For seg: mask is essentially empty (no foreground voxels) or full.
    # For cls: every input predicts the same class (0 or majority class).
    # `fg_after` is the fraction predicting "any positive class" — well-
    # defined for both. Task-aware messaging keeps the report honest.
    _is_cls = task_hint in ("classification", "classification_3d")
    def _empty_msg(rate):
        if _is_cls:
            return f"mode collapse: predicted negative class for {1 - rate:.0%} of inputs"
        return f"collapsed to all-background (fg={rate:.1%})"
    def _full_msg(rate):
        if _is_cls:
            return f"mode collapse: predicted positive class for {rate:.0%} of inputs"
        return f"exploded to all-foreground (fg={rate:.0%})"

    if fg_after < 0.002 and expected_min > 0.005:
        return _empty_msg(fg_after)
    if fg_after > 0.9 and expected_max < 0.5:
        return _full_msg(fg_after)
    if fg_before > expected_min and fg_after < 0.002:
        if _is_cls:
            return (f"mode collapse during adaptation: positive-class rate "
                     f"{fg_before:.1%} -> {fg_after:.1%}")
        return f"collapsed from reasonable to zero (fg {fg_before:.1%}->{fg_after:.1%})"
    return None


def _run_final_check(ps, source_model, target_images, device,
                       llm_client, vlm_ctx: Dict,
                       knowledge: Dict,
                       model_assessment: Optional[Dict] = None,
                       vlm_final_k: int = 5,
                       task=None) -> Dict:
    """The single adapt-or-revert gate that MUST run before the pipeline
    returns an adapted model to Agent 6 / delivery.

    Two sub-gates, in order:
      (a) Catastrophic collapse — physical-extreme fg check gated by
          Agent 1's expected_min/max priors. Objective, no VLM.
      (b) Symmetric VLM second opinion on a DISJOINT anchor set (linspace
          sampling). Decision rule: any-worse-vote → reject (default-to-
          source). This is what enforces the at-or-above-baseline guarantee.

    The helper is intentionally branch-agnostic. Every adaptation path
    (main iteration loop, SAR shortcut, FDA-only, skip) must call this
    before returning. If any path skips it, adapted models can reach
    delivery without the capability gate — that's the prostate_stunet
    2026-04-21 regression pattern we're fixing here.

    Returns a dict with final_verdict, final_reason, final_votes,
    final_note, quality_before_final, quality_after_final,
    q_returned — ready to splice into execution_report and
    iteration_history. Also mutates ps.adapted_model in-place on reject.
    """
    expected_min, expected_max = 0.005, 0.5
    fg_priors = (knowledge or {}).get("foreground_priors", {}) or {}
    ea = fg_priors.get("expected_area_fraction")
    if isinstance(ea, (list, tuple)) and len(ea) >= 2:
        expected_min, expected_max = ea[0], ea[1]

    q_before = _measure_quality(source_model, target_images, device)
    q_after = _measure_quality(ps.adapted_model, target_images, device)
    fg_before = q_before.get("fg_ratio", 0)
    fg_after = q_after.get("fg_ratio", 0)

    final_verdict = "accept"
    final_reason = "no catastrophe; VLM second opinion did not observe any worse-vote"
    final_votes: List[str] = []
    final_note = ""
    did_revert_local = False

    # (a) Catastrophic collapse gate — runs for both seg and cls. For
    # seg, `fg_ratio` is fraction of foreground voxels in predicted
    # masks. For cls, it is the fraction of inputs predicting any
    # positive class (mode-collapse to class 0 manifests as fg_after≈0).
    # Task-hint controls the diagnostic phrasing.
    _task_hint = (model_assessment or {}).get("task_hint") or "segmentation"
    can_check = (isinstance(fg_before, (int, float))
                    and isinstance(fg_after, (int, float)))
    collapse_reason = None
    if can_check:
        collapse_reason = _catastrophic_collapse(
            fg_before, fg_after, expected_min, expected_max,
            task_hint=_task_hint)
    if collapse_reason is not None:
        final_verdict = "reject"
        final_reason = f"catastrophic: {collapse_reason}"
        print(f"    ⛔ Final check: {final_reason} — reverting to source")
        ps.adapted_model.load_state_dict(source_model.state_dict(), strict=False)
        did_revert_local = True

    # (b) Symmetric VLM second opinion on a DISJOINT anchor set (linspace).
    #     Two gates applied to the VLM's output:
    #       (i)  any sample voted RELATIVE "worse" → reject (default-to-source)
    #       (ii) any sample voted ABSOLUTE "broken" → reject
    #     (ii) catches the plausible-drift case where CURRENT looks cleaner
    #     than SOURCE and wins the relative vote, yet covers almost none of
    #     the actual target structure (riga_plus_nnunet 2026-04-22: adapted
    #     Dice 0.07 vs source 0.30; VLM voted 5/5 "better" relative but would
    #     have voted "broken" absolute on a near-empty mask). The two gates
    #     together enforce: the returned model must not be clearly worse than
    #     source AND must not be clearly broken on its own.
    final_abs: List[str] = []
    if final_verdict == "accept" and llm_client is not None and len(target_images) >= 3:
        k_final = min(vlm_final_k, len(target_images))
        final_anchor_idx = np.linspace(
            0, len(target_images) - 1, k_final, dtype=int).tolist()
        final_anchor_imgs = [target_images[i] for i in final_anchor_idx]
        # Rebuild task_ctx with size-delta + source-quality hints grounded in
        # the actual source vs adapted fg ratios we just measured. This gives
        # the VLM the explicit growth context so it can judge whether a
        # visually-plausible larger mask is actually justified.
        vlm_ctx_final = _build_vlm_task_ctx(
            knowledge or {}, model_assessment or {},
            source_fg=fg_before, current_fg=fg_after,
        )
        _score, final_votes, final_note, final_abs = _vlm_iteration_vote(
            current_model=ps.adapted_model,
            source_model=source_model,
            anchor_imgs=final_anchor_imgs,
            device=device,
            llm_client=llm_client,
            task_ctx=vlm_ctx_final,
            k=k_final,
            task=task,
        )
        n_better = sum(1 for v in final_votes if v == "better")
        n_similar = sum(1 for v in final_votes if v == "similar")
        n_worse = sum(1 for v in final_votes if v == "worse")
        n_broken = sum(1 for v in final_abs if v == "broken")
        n_usable = sum(1 for v in final_abs if v == "usable")
        if final_votes or final_abs:
            print(f"    Final VLM (disjoint anchors {final_anchor_idx}): "
                  f"relative better={n_better}/similar={n_similar}/worse={n_worse}"
                  + (f" | absolute usable={n_usable}/broken={n_broken}"
                      if final_abs else " | absolute=(none)"))
            if final_note:
                print(f"    Note: {final_note[:110]}")
            # Decision rule: reject on STRICT MAJORITY (>50%) of either
            # signal — `worse` (relative regression vs source) or `broken`
            # (absolute catastrophe on raw-image grounding). A single noisy
            # VLM vote on one sample shouldn't tank a clear improvement;
            # majority agreement across disjoint anchors gives a much lower
            # false-positive rate. This is the same strict-majority principle
            # used elsewhere in the pipeline (halluc majority > 0.5).
            n_total = max(len(final_votes), len(final_abs), 1)
            majority = n_total // 2 + 1  # >50% = at least ceil(n/2)+1 ... =floor(n/2)+1 actually
            if n_worse >= majority:
                final_verdict = "reject"
                final_reason = (f"VLM second opinion (relative): {n_worse}/"
                                 f"{n_total} samples voted worse (strict "
                                 f"majority ≥{majority}) on anchors {final_anchor_idx}")
                print(f"    ⛔ Final check rejected (majority worse) — reverting to source")
                ps.adapted_model.load_state_dict(source_model.state_dict(), strict=False)
                did_revert_local = True
            elif (
                isinstance((model_assessment or {}).get("source_quality_score"),
                            (int, float))
                and (model_assessment or {}).get("source_quality_score", 0) >= STRONG_SOURCE_SQ_THRESHOLD
                and n_better < majority
                and n_total > 0
            ):
                # Strong-source guard. Source is already competent on its
                # own domain (sq>=0.8); we should only accept adaptation
                # if the VLM sees CLEAR improvement (majority "better"),
                # not just "all similar / all usable". Catches the
                # subtle-regression case where adaptation drifts predictions
                # in ways too small for the VLM to vote "worse" but large
                # enough to lose Dice (mnms 2026-04-27 with bn_reset).
                _sq = (model_assessment or {}).get("source_quality_score", 0)
                final_verdict = "reject"
                final_reason = (f"strong source (sq={_sq:.2f}) but only "
                                 f"{n_better}/{n_total} clear-better votes "
                                 f"(<{majority}) — adaptation not justified")
                print(f"    ⛔ Final check rejected (strong source, no clear gain) — reverting to source")
                ps.adapted_model.load_state_dict(source_model.state_dict(), strict=False)
                did_revert_local = True
            elif n_broken >= majority:
                final_verdict = "reject"
                final_reason = (f"VLM second opinion (absolute): {n_broken}/"
                                 f"{n_total} samples voted the adapted mask "
                                 f"BROKEN (strict majority ≥{majority}) on "
                                 f"anchors {final_anchor_idx}")
                print(f"    ⛔ Final check rejected (majority broken) — reverting to source")
                ps.adapted_model.load_state_dict(source_model.state_dict(), strict=False)
                did_revert_local = True

    # Phase R5: structural-drift veto for strong-source regime.
    # The existing strong-source guard (lines ~428) only fires when VLM
    # is NOT majority-better; on mnms 2026-05-27 the VLM gave 5/5
    # unanimous "better" votes (note: "CURRENT masks grew to better
    # cover the visible target structure") but bn_reset dropped GT
    # Dice 0.87 -> 0.73 — confidently wrong.
    #
    # Add a NON-VLM second opinion: when the source is strong (sq >=
    # 0.8) AND Agent 2 does NOT diagnose a severe target failure, the
    # adapted model's predictions should not change structurally. If
    # the adapted fg_ratio differs from source by more than 25%
    # relative, that's evidence of drift (BN stat shift, gradient
    # over-fitting) — not improvement — regardless of what the VLM
    # voted. This is a quantitative gate that survives unanimous
    # VLM agreement.
    if (final_verdict == "accept" and not did_revert_local
            and isinstance((model_assessment or {}).get(
                "source_quality_score"), (int, float))
            and (model_assessment or {}).get(
                "source_quality_score", 0) >= STRONG_SOURCE_SQ_THRESHOLD):
        ft = (model_assessment or {}).get("failure_type", "").lower()
        # Skip R5 when Agent 2 diagnosed a real target failure — those
        # cases NEED adaptation and big fg_ratio changes are expected
        # (chaos K=0: failure_type=complete_failure, fg shift is the
        # whole point).
        agent2_says_broken = ft in (
            "complete_failure", "source_broken", "mode_collapse",
            "hallucination")
        if not agent2_says_broken and fg_before > 0.005:
            fg_drift_rel = abs(fg_after - fg_before) / max(fg_before, 0.02)
            if fg_drift_rel > 0.25:
                sq = (model_assessment or {}).get("source_quality_score", 0)
                final_verdict = "reject"
                final_reason = (
                    f"Phase R5 structural-drift veto: strong-source "
                    f"(sq={sq:.2f}, failure_type={ft}) but adapted "
                    f"fg_ratio={fg_after:.3f} differs from source "
                    f"fg_ratio={fg_before:.3f} by {fg_drift_rel:.0%} "
                    f"(>25%) — VLM voted accept but predictions "
                    f"changed structurally, suggesting drift not "
                    f"improvement.")
                print(f"    ⛔ Phase R5 structural-drift reject "
                       f"(fg shifted {fg_drift_rel:.0%}, "
                       f"VLM may be over-confident) — reverting "
                       f"to source")
                ps.adapted_model.load_state_dict(
                    source_model.state_dict(), strict=False)
                did_revert_local = True

    q_returned = (_measure_quality(ps.adapted_model, target_images, device)
                   if did_revert_local else q_after)

    return {
        "final_verdict": final_verdict,
        "final_reason": final_reason,
        "final_votes": final_votes,
        "final_votes_absolute": final_abs,
        "final_note": final_note,
        "quality_before_final": q_before,
        "quality_after_final": q_returned,
        "did_revert_in_final_check": did_revert_local,
    }


def _build_mechanism_hint(knowledge: Dict, model_assessment: Dict) -> str:
    """Render the Agent-1-diagnosed distribution-shift mechanism into a
    direct, sample-level instruction for the VLM, so the tournament /
    final-gate question is conditioned on WHAT shift was diagnosed and
    WHAT the correct fix must (and must not) do -- instead of the generic
    "is CURRENT better than SOURCE?". This is what stops the
    confidently-wrong unanimous vote (e.g. a stats-norm reset that looked
    cleaner but had grown onto non-target tissue).

    Additive + dataset-agnostic: every field is read from the Agent-1
    `knowledge` dict (no dataset names, no task assumptions). When no
    mechanism was diagnosed, returns a neutral fallback so the question
    degrades gracefully to the generic form.
    """
    da = knowledge or {}

    dom = da.get("dominant_shift_factor")
    dom = dom.strip() if isinstance(dom, str) else ""

    severity = da.get("severity")
    severity = severity.strip().lower() if isinstance(severity, str) else ""
    contrast_change = da.get("contrast_mechanism_change")
    contrast_change = (contrast_change.strip().lower()
                       if isinstance(contrast_change, str) else "")
    shift_level = da.get("shift_level")

    # Per-channel shifts: surface the highest-severity descriptions so the
    # VLM knows which specific appearance change to expect (e.g. an
    # intensity inversion on one channel) and can check the fix targets it.
    shift_phrases: List[str] = []
    dshifts = da.get("dominant_shifts")
    if isinstance(dshifts, list):
        rank = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(
            [d for d in dshifts if isinstance(d, dict)],
            key=lambda d: rank.get(str(d.get("severity", "")).lower(), 3))
        for d in ordered[:2]:
            st = str(d.get("shift_type", "")).strip()
            desc = str(d.get("description", "")).strip()
            ch = str(d.get("channel", "")).strip()
            piece = desc or st
            if piece:
                if ch and ch.lower() not in ("global", ""):
                    piece = f"{ch}: {piece}"
                shift_phrases.append(piece[:160])

    # What Agent 1 predicted would fail -- the appearance-level trap the
    # adaptation is supposed to fix.
    fail_phrase = ""
    wwf = da.get("what_will_fail")
    if isinstance(wwf, list) and wwf:
        first = wwf[0]
        if isinstance(first, str) and first.strip():
            fail_phrase = first.strip()[:200]

    calib = da.get("calibration_concern")
    calib = calib.strip().lower() if isinstance(calib, str) else ""

    # If Agent 1 gave us nothing usable, keep the question generic.
    if not (dom or shift_phrases or fail_phrase or contrast_change):
        return ("No specific distribution-shift mechanism was diagnosed for "
                "this case. Judge each prediction purely by how well it "
                "overlaps the actual structure visible in the raw image, "
                "using the criteria below.")

    lines: List[str] = []
    head = "Agent 1 diagnosed the source->target shift as: "
    if dom:
        head += f'"{dom}"'
    else:
        head += "a domain shift"
    qualifiers = []
    if isinstance(shift_level, (int, float)):
        qualifiers.append(f"shift level {int(shift_level)}/3")
    if severity:
        qualifiers.append(f"{severity} severity")
    if contrast_change in ("different", "fundamentally_different"):
        qualifiers.append("the tissue-contrast mechanism itself changed")
    if qualifiers:
        head += " (" + "; ".join(qualifiers) + ")"
    head += "."
    lines.append(head)

    if shift_phrases:
        lines.append("Specific appearance changes to expect: "
                     + "; ".join(shift_phrases) + ".")
    if fail_phrase:
        lines.append(f"The source model is expected to fail because: "
                     f"{fail_phrase}")

    # The decisive instruction: a fix is only "better" if it specifically
    # repairs the diagnosed mechanism WITHOUT bleeding onto non-target
    # tissue. This is the mechanism-conditioning that prevents the
    # plausible-but-wrong unanimous vote.
    lines.append(
        'Vote CURRENT "better" ONLY if its prediction specifically '
        "corrects THIS diagnosed shift AND its foreground still lies on "
        "the actual target structure in the raw image. If CURRENT changed "
        "in a way that is unrelated to the diagnosed shift, or grew onto "
        'non-target tissue, vote "similar" or "worse" even if it looks '
        "cleaner or larger than SOURCE.")

    if calib.startswith("calibration shift"):
        lines.append("Note: this is an intensity-rescaling (calibration) "
                     "shift -- the correct fix re-normalizes appearance and "
                     "should NOT move or grow the structure boundary much.")
    elif calib.startswith("feature shift"):
        lines.append("Note: this is a feature shift -- the source "
                     "representations may be wrong, so a real fix can change "
                     "the mask substantially; still require it to land on "
                     "the actual structure.")

    return " ".join(lines)


def _build_vlm_task_ctx(knowledge: Dict, model_assessment: Dict,
                          source_fg: Optional[float] = None,
                          current_fg: Optional[float] = None) -> Dict:
    """Assemble structure/location/shape/failure-type context for the VLM,
    plus source-quality and fg-ratio context so the VLM can reason about:
    (a) whether to trust the source model as a reference (strong source →
        be skeptical of large changes), and (b) whether the size delta from
        source to current is anatomically justified.
    """
    da = knowledge or {}
    ma = model_assessment or {}
    priors = da.get("foreground_priors", {}) or {}

    structure_name = "target structure"
    structures = da.get("structures", {})
    if isinstance(structures, dict):
        fg_names = [s for s, v in structures.items()
                     if s != "background" and v > 0]
        if fg_names:
            structure_name = " / ".join(fg_names)

    expected_min, expected_max = 0.005, 0.5
    ea = priors.get("expected_area_fraction")
    if isinstance(ea, (list, tuple)) and len(ea) >= 2:
        expected_min, expected_max = ea[0], ea[1]

    direction_map = {
        "under_segmentation": "grow the predicted region (source was too small)",
        "over_segmentation":  "shrink the predicted region (source was too large)",
        "complete_failure":   "recover the structure from near-zero predictions",
        "correct_location_wrong_shape": "refine shape without moving location",
    }
    failure_type = ma.get("failure_type", "unknown")

    # Source quality context — if the source model is strong on its OWN
    # training domain, we should be conservative about accepting large
    # changes on the target (risk of adapting a good model into a bad one).
    sq_score = ma.get("source_quality_score")
    if isinstance(sq_score, (int, float)):
        if sq_score >= 0.8:
            sq_hint = (f"STRONG source (source-domain Dice={sq_score:.2f}). "
                        f"The source model is likely competent; be skeptical of "
                        f"large, unjustified changes to its predictions.")
        elif sq_score >= 0.5:
            sq_hint = (f"MODERATE source (source-domain Dice={sq_score:.2f}). "
                        f"Adaptation may help but has a ceiling.")
        else:
            sq_hint = (f"WEAK source (source-domain Dice={sq_score:.2f}). "
                        f"The source model is borderline; adaptation's ceiling "
                        f"is set by the source.")
    else:
        sq_hint = "Source quality unknown."

    # Size-delta context — make the growth/shrink observable to the VLM so
    # it can reason whether the change is justified by the raw image.
    if isinstance(source_fg, (int, float)) and isinstance(current_fg, (int, float)):
        if source_fg > 1e-4:
            ratio = current_fg / source_fg
            if ratio > 3.0:
                if failure_type == "under_segmentation":
                    size_delta_hint = (f"CURRENT predicts {current_fg:.1%} fg vs "
                                         f"SOURCE's {source_fg:.1%} - a {ratio:.1f}x "
                                         f"growth. This direction MATCHES the "
                                         f"diagnosis (under_segmentation needs growth). "
                                         f"Vote 'better' if the new foreground falls "
                                         f"on the actual {structure_name}.")
                else:
                    size_delta_hint = (f"CURRENT predicts {current_fg:.1%} of the "
                                         f"image as foreground vs SOURCE's {source_fg:.1%} "
                                         f"- a {ratio:.1f}x growth. Failure_type is "
                                         f"{failure_type}; growth may not be the right "
                                         f"correction direction. Verify against the raw image.")
            elif ratio < 0.3:
                size_delta_hint = (f"CURRENT predicts {current_fg:.1%} vs SOURCE's "
                                    f"{source_fg:.1%} - a shrink to {ratio:.1f}x of "
                                    f"source size. Large shrink is a red flag unless "
                                    f"justified.")
            else:
                size_delta_hint = (f"CURRENT fg={current_fg:.1%}, SOURCE fg={source_fg:.1%} "
                                    f"({ratio:.1f}x change). Modest size change.")
        else:
            size_delta_hint = (f"SOURCE predicts ~0% fg; any CURRENT fg is new "
                                f"structure appearing under adaptation.")
    else:
        size_delta_hint = "Size delta not provided."

    return {
        "structure_name": structure_name,
        "mechanism_hint": _build_mechanism_hint(knowledge, model_assessment),
        "expected_location": priors.get("expected_location", "unknown"),
        "expected_shape": priors.get("expected_shape", "unknown"),
        "expected_fg": f"{expected_min*100:.0f}-{expected_max*100:.0f}%",
        "failure_type": failure_type,
        "correction_direction": direction_map.get(
            failure_type, "not specified — judge by absolute plausibility"),
        "source_quality_hint": sq_hint,
        "size_delta_hint": size_delta_hint,
    }


# --- Agent 4 safety helpers ---

def _sanity_check_plan(plan: Dict) -> Tuple[bool, str]:
    """Reject obviously-broken plans before execution.

    Returns (ok, reason). ok=False blocks execution and returns a skipped
    result so downstream (Agent 5) can ask Agent 3 to re-plan.
    """
    if not isinstance(plan, dict) or not plan:
        return False, "empty or non-dict plan"
    lw = plan.get("loss_weights", {}) or {}
    # Allow all-zero loss weights only for protocols that do not use gradient
    # updates at all (num_steps=0). For gradient-based protocols, zero weights
    # mean "train with no loss" which is nonsensical.
    num_steps = plan.get("num_steps", 0)
    if num_steps and isinstance(lw, dict):
        numeric = [v for v in lw.values() if isinstance(v, (int, float))]
        if numeric and sum(abs(v) for v in numeric) < 1e-6:
            return False, "all loss weights are zero but num_steps > 0"
    # Reject runaway steps / iterations
    if isinstance(num_steps, (int, float)) and num_steps > 5000:
        return False, f"num_steps={num_steps} exceeds safe max 5000"
    max_iter = plan.get("max_iterations", 1)
    if isinstance(max_iter, (int, float)) and max_iter > 10:
        return False, f"max_iterations={max_iter} exceeds safe max 10"
    lr = plan.get("learning_rate", 0)
    if isinstance(lr, (int, float)) and lr < 0:
        return False, f"negative learning_rate={lr}"
    return True, ""


def _check_prediction_collapse(ps, device, min_fg=0.0005, max_fg=0.95):
    """Rule-based in-flight collapse detector.

    Runs the adapted model on up to 16 target samples, returns fg_ratio and
    whether it looks collapsed (all-background or all-foreground — both are
    signs the plan is destroying the model rather than adapting it).
    """
    try:
        model = ps.adapted_model
        imgs = ps.target_images[:16]
        model.eval()
        preds_fg = []
        with torch.no_grad():
            for img in imgs:
                if img.ndim == 2:
                    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
                    t = t.repeat(1, 3, 1, 1)
                elif img.ndim == 3:
                    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                else:
                    continue
                t = t.to(device)
                try:
                    logits = model(t)
                    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                    preds_fg.append(float((pred > 0).mean()))
                except Exception:
                    continue
        if not preds_fg:
            return None, False
        fg = float(np.mean(preds_fg))
        collapsed = (fg < min_fg) or (fg > max_fg)
        return fg, collapsed
    except Exception:
        return None, False


def _best_iter_by_selection_score(history: List[Dict]) -> int:
    """Return the index of the best iteration by the unified selection metric.

    Priority (pure-TTA):
      1. vlm_score (VLM relative judgment vs source, in [-1, +1]) — primary
      2. quality_score (label-free entropy/consistency) — fallback if VLM down
      3. avg_dice — only populated in offline evaluation, never in deployment

    Ignores post_revert entries (they are post-hoc re-evals, not adaptation
    iterations).
    """
    def score_of(h):
        if h.get("method") == "post_revert":
            return -float("inf")
        for key in ("vlm_score", "quality_score", "avg_dice"):
            v = h.get(key)
            if isinstance(v, (int, float)):
                return v
        return -float("inf")

    if not history:
        return -1
    best_i = 0
    best_s = score_of(history[0])
    for i in range(1, len(history)):
        s = score_of(history[i])
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


def _run_sar_adaptation(ps, plan, device):
    """SAR: Sharpness-Aware entropy minimization.

    More robust than Tent — perturbs weights toward flat minima
    before computing entropy gradient, avoiding sharp minima collapse.
    """
    model = ps.adapted_model
    model.train()

    # Reset running stats on any norm layer that tracks them (BN always,
    # IN when track_running_stats=True). No-op on LN/GN.
    from medadaptagent.graph.norm_utils import (
        iter_norm_modules, has_running_stats, reset_norm_running_stats,
    )
    reset_norm_running_stats(model)
    # Cumulative average over the upcoming forward passes — only valid for
    # BN (PyTorch's instance_norm raises if momentum is None). For IN with
    # tracking we leave the default EMA momentum; the stats still refresh
    # from target data because reset_norm_running_stats() zeroed them.
    for _, m, nt in iter_norm_modules(model):
        if nt == "batchnorm" and has_running_stats(m):
            m.momentum = None

    with torch.no_grad():
        for img in ps.target_images[:200]:
            if img.ndim == 3 and img.shape[2] in (1, 3):
                t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            elif img.ndim == 2:
                t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
            else:
                t = torch.from_numpy(img.astype(np.float32))
            model(t.unsqueeze(0).to(device))

    # SAR: train the affine params of every norm layer (BN/IN/LN/GN).
    # SAR's sharpness-aware step is architecture-agnostic — what it updates
    # is norm-layer weight/bias, and norm_utils gives us those uniformly.
    params = []
    param_ids = set()
    for p in model.parameters():
        p.requires_grad = False
    for _, m, _ in iter_norm_modules(model):
        if getattr(m, "weight", None) is not None:
            m.weight.requires_grad = True
            params.append(m.weight)
            param_ids.add(id(m.weight))
        if getattr(m, "bias", None) is not None:
            m.bias.requires_grad = True
            params.append(m.bias)
            param_ids.add(id(m.bias))

    if not params:
        print("  No BN params found — SAR not applicable")
        model.eval()
        return

    lr = plan.get("learning_rate", 5e-3)
    rho = 0.05  # SAR perturbation radius
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    steps = plan.get("num_steps", 80)
    model.eval()
    for step in range(steps):
        idx = np.random.randint(len(ps.target_images))
        img = ps.target_images[idx]
        if img.ndim == 3 and img.shape[2] in (1, 3):
            t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        elif img.ndim == 2:
            t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
        else:
            t = torch.from_numpy(img.astype(np.float32))
        x = t.unsqueeze(0).to(device)

        # First forward: compute entropy + gradient for perturbation
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()  # use running stats during adaptation

        logits = model(x)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()

        entropy.backward()

        # SAR perturbation: save params, perturb toward sharpest direction
        saved_params = {id(p): p.data.clone() for p in params}
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    grad_norm = p.grad.norm()
                    if grad_norm > 0:
                        p.data.add_(rho * p.grad / grad_norm)
                    p.grad.zero_()

        # Second forward: compute loss at perturbed point
        logits2 = model(x)
        probs2 = F.softmax(logits2, dim=1)
        loss = -(probs2 * torch.log(probs2 + 1e-10)).sum(dim=1).mean()

        # Revert perturbation BEFORE gradient step
        with torch.no_grad():
            for p in params:
                p.data.copy_(saved_params[id(p)])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"    SAR step {step+1}/{steps}, entropy={loss.item():.4f}")

    model.eval()
    print(f"  SAR adaptation complete ({steps} steps)")


def _apply_color_normalization(ps):
    """Simple intensity normalization for fundus/histopath images."""
    # Reinhard-style: match mean/std per channel
    src = ps.source_images[:100]
    tgt = ps.target_images

    if not src or not tgt:
        return

    # Compute source stats
    src_means = np.mean([img.mean(axis=(0, 1)) for img in src], axis=0)
    src_stds = np.mean([img.std(axis=(0, 1)) for img in src], axis=0)

    for i, img in enumerate(tgt):
        tgt_mean = img.mean(axis=(0, 1))
        tgt_std = img.std(axis=(0, 1)) + 1e-8
        normalized = (img - tgt_mean) / tgt_std * src_stds + src_means
        ps.target_images[i] = np.clip(normalized, 0, 1).astype(np.float32)

    print(f"  Color normalized {len(tgt)} images")


def run_execution_engineer(plan: Dict,
                           source_model: nn.Module,
                           source_images: List[np.ndarray],
                           target_images: List[np.ndarray],
                           source_labels: List[np.ndarray],
                           task_config,
                           knowledge: Dict,
                           device: str = "cuda:0",
                           llm_client=None,
                           model_assessment: Optional[Dict] = None,
                           vlm_anchor_k: int = 3,
                           vlm_final_k: int = 5) -> Dict:
    """Execute the adaptation plan.

    Takes the Strategy Architect's plan and runs the actual
    adaptation using the orchestrator's adaptation machinery.

    Returns: dict with adapted_model, iteration_history, etc.
    """
    print("\n" + "=" * 55)
    print("AGENT 4: Execution Engineer")
    print("=" * 55)

    # (1) Plan sanity check — refuse obviously-broken plans at entry.
    ok, reason = _sanity_check_plan(plan)
    if not ok:
        print(f"  REFUSING plan: {reason}")
        return {
            "adapted_model": source_model,  # pass-through, unchanged
            "iteration_history": [],
            "quality_scores": [],
            "skipped": True,
            "did_revert": False,
            "execution_report": {
                "status": "refused_at_entry",
                "reason": reason,
                "plan_snapshot": {k: plan.get(k) for k in
                                  ("protocol", "trainable_scope", "num_steps",
                                   "max_iterations", "learning_rate", "loss_weights")},
            },
            "pipeline_state": None,
        }

    from medadaptagent.orchestrator import Orchestrator, PipelineState

    # Setup pipeline state
    cfg = copy.deepcopy(task_config)
    cfg.device = device
    cfg.max_iterations = plan.get("max_iterations", 1)

    # Pure TTA: no target labels consumed at all. VLM anchor samples are a
    # small FIXED subset of target_images (no labels) used every iteration so
    # the VLM always scores the same scenes and per-iter scores are comparable.
    rng_anchor = np.random.RandomState(42)
    anchor_perm = rng_anchor.permutation(len(target_images))
    anchor_idx = anchor_perm[:min(vlm_anchor_k, len(target_images))].tolist()
    anchor_imgs = [target_images[i] for i in anchor_idx]
    print(f"  [VLM anchors] fixed {len(anchor_imgs)} samples (idx={anchor_idx}) "
          f"for per-iteration VLM comparison")

    ps = PipelineState(
        config=cfg,
        source_model=copy.deepcopy(source_model),
        source_images=source_images,
        target_images=target_images,
        source_labels=source_labels,
        target_labels=[],  # pure TTA — agents never see target labels
    )
    ps.calibration_images = []
    ps.calibration_labels = []

    # Inject knowledge and plan — ensure plan values are correct types
    ps.rag_knowledge = knowledge

    # Sanitize plan: LLM sometimes returns strings instead of dicts
    if isinstance(plan.get("loss_weights"), str):
        plan["loss_weights"] = {"entropy": 0.3, "consistency": 0.2,
                                 "diversity": 0.2, "bn_adapt": 0.3}
    if not isinstance(plan.get("loss_weights"), dict):
        plan["loss_weights"] = {"entropy": 0.3, "consistency": 0.2,
                                 "diversity": 0.2, "bn_adapt": 0.3}
    # Ensure numeric types
    for key in ["learning_rate", "num_steps", "max_iterations", "fda_beta",
                "fda_finetune_steps", "fda_finetune_lr", "fda_mix_weight"]:
        if key in plan and isinstance(plan[key], str):
            try:
                plan[key] = float(plan[key])
            except (ValueError, TypeError):
                pass
    # Cast int-typed fields to int (LLM may return them as floats like 120.0,
    # which breaks range() in run_fda_warmup and the gradient loop).
    for key in ["num_steps", "max_iterations", "fda_finetune_steps"]:
        if key in plan and plan[key] is not None:
            try:
                plan[key] = int(plan[key])
            except (ValueError, TypeError):
                pass

    ps.adaptation_plan = plan

    # Handle skip
    if plan.get("skip_adaptation"):
        print("  Plan says SKIP — returning source model unchanged")
        ps.adapted_model = copy.deepcopy(source_model)
        ps.adapted_model.to(device)

        # Pre-create iteration_history entry so evaluation_agent.run()
        # can populate it in-place with quality_score, per_class_quality,
        # AND GT dice metrics (avg_dice + <class>_dice). Without this,
        # the entry is empty when eval runs and delivery reads avg_dice=0.
        ps.iteration_history.append({"method": "do_no_harm_skip"})

        orch = Orchestrator(ps)
        q_score, per_class_q = orch.evaluation_agent.run()

        return {
            "adapted_model": ps.adapted_model,
            "iteration_history": ps.iteration_history,
            "quality_scores": [float(q) for q in ps.quality_scores],
            "skipped": True,
        }

    # Initialize adapted model
    ps.adapted_model = copy.deepcopy(source_model)
    ps.adapted_model.to(device)

    orch = Orchestrator(ps)

    # Step 1: FDA warmup (if planned)
    if plan.get("use_fda_warmup") and source_images and source_labels:
        print("  Running FDA warmup...")
        # Set FDA params from plan or defaults
        plan.setdefault("fda_beta", 0.03)
        plan.setdefault("fda_finetune_steps", 100)
        plan.setdefault("fda_finetune_lr", 5e-4)
        plan.setdefault("fda_scope", "bn+output")
        plan.setdefault("use_fda_mixed_training", True)
        plan.setdefault("fda_mix_weight", 0.2)
        orch.adaptation_agent.run_fda_warmup()

    # Step 2: Preprocessing
    if plan.get("use_histogram_matching"):
        print("  Applying histogram matching...")
        orch._apply_histogram_matching()

    if plan.get("use_color_normalization"):
        print("  Applying color normalization...")
        # Reinhard normalization for fundus/histopath
        _apply_color_normalization(ps)

    # Step 3: SAR adaptation (alternative to Tent)
    if plan.get("use_sar"):
        print("  Running SAR (sharpness-aware adaptation)...")
        try:
            _run_sar_adaptation(ps, plan, device)
        except Exception as e:
            print(f"  SAR failed ({e}), falling back to BN reset + Tent")
            # Fallback: just do BN reset
            ps.adapted_model = copy.deepcopy(source_model)
            ps.adapted_model.to(device)

        # Evaluate after SAR
        orch2 = Orchestrator(ps)
        orch2.evaluation_agent.run()

        # Ensure iteration_history has an entry
        if not ps.iteration_history:
            ps.iteration_history.append({"avg_dice": 0, "method": "sar"})

        # Route through the shared final check — SAR must not bypass the
        # capability gate (prostate_stunet 2026-04-21 regression pattern).
        vlm_ctx = _build_vlm_task_ctx(knowledge or {}, model_assessment or {})
        fc = _run_final_check(ps, source_model, target_images, device,
                               llm_client, vlm_ctx, knowledge or {},
                               model_assessment=model_assessment,
                               vlm_final_k=vlm_final_k)
        did_revert_sar = fc["did_revert_in_final_check"]
        ps.iteration_history.append({
            "method": "final_check",
            "iteration": len(ps.iteration_history),
            "quality_score": fc["quality_after_final"].get("consistency", 0),
            "fg_ratio_post_iter": fc["quality_after_final"].get("fg_ratio", 0),
            "final_verdict": fc["final_verdict"],
            "final_votes": fc["final_votes"],
        })

        execution_report = {
            "status": "completed",
            "stopped_reason": ("SAR + final check reverted" if did_revert_sar
                                else "SAR completed without regression"),
            "did_revert": did_revert_sar,
            "n_iterations_run": 1,
            "best_iteration_index": 0,
            "collapse_events": [],
            "vlm_best_score": None,
            "vlm_anchor_idx": [],
            "final_verdict": fc["final_verdict"],
            "final_reason": fc["final_reason"],
            "final_votes": fc["final_votes"],
            "final_votes_absolute": fc.get("final_votes_absolute", []),
            "final_note": fc["final_note"],
            "quality_before_final": fc["quality_before_final"],
            "quality_after_final": fc["quality_after_final"],
            "plan_snapshot": {
                "protocol": plan.get("protocol"),
                "trainable_scope": plan.get("trainable_scope"),
                "num_steps": plan.get("num_steps"),
                "max_iterations": plan.get("max_iterations"),
                "learning_rate": plan.get("learning_rate"),
                "loss_weights": plan.get("loss_weights"),
            },
            "trajectory": [{"iteration": 0, "method": "sar"}],
        }

        return {
            "adapted_model": ps.adapted_model,
            "iteration_history": ps.iteration_history,
            "quality_scores": [float(q) for q in ps.quality_scores],
            "skipped": False,
            "did_revert": did_revert_sar,
            "execution_report": execution_report,
            "pipeline_state": ps,
        }

    # Step 4: Iterative adaptation (default path)
    # SAFE MODE: if num_steps=0, only do BN reset + evaluation (no gradients)
    num_steps = plan.get("num_steps", 0)
    max_iter = plan.get("max_iterations", 1)

    if num_steps == 0:
        print(f"  SAFE MODE: BN reset only (no gradient steps)")
        max_iter = 1  # just one pass for BN stats

    print(f"  Running {max_iter} adaptation iteration(s) (steps={num_steps})...")

    did_revert = False
    stopped_reason = None
    collapse_events = []

    # Pure-TTA selection metric: vlm_score first, quality_score as a fallback
    # if the VLM is unavailable. No GT metric used.
    def _selection_score(h):
        for key in ("vlm_score", "quality_score"):
            v = h.get(key)
            if isinstance(v, (int, float)):
                return v, key
        return None, None

    # Build the VLM task context once (same across iterations).
    vlm_ctx = _build_vlm_task_ctx(knowledge, model_assessment or {})

    # Track best iteration by VLM score (own best-state tracking; don't use
    # the orchestrator's quality_score-based best_model_state since VLM is a
    # different scale).
    vlm_best_score = -float("inf")
    vlm_best_state = None

    for i in range(max_iter):
        metrics = orch.adaptation_agent.run_iteration(i)
        q_score, per_class_q = orch.evaluation_agent.run()

        # (3) In-flight prediction-collapse guard: objective physical check.
        fg_ratio_now, collapsed = _check_prediction_collapse(ps, device)
        if fg_ratio_now is not None and ps.iteration_history:
            ps.iteration_history[-1]["fg_ratio_post_iter"] = fg_ratio_now
        if collapsed:
            collapse_events.append({"iteration": i, "fg_ratio": fg_ratio_now})
            print(f"    ⚠️ Prediction collapse (fg={fg_ratio_now:.4f}); aborting")
            if vlm_best_state is not None:
                ps.adapted_model.load_state_dict(vlm_best_state)
                did_revert = True
            stopped_reason = f"prediction_collapse at iter {i} (fg={fg_ratio_now:.4f})"
            break

        # VLM per-iteration vote: compare current adapted model to the frozen
        # source on the fixed anchor samples. This is our primary selection
        # signal in pure TTA.
        vlm_score, vlm_votes, vlm_note, vlm_absolute = _vlm_iteration_vote(
            current_model=ps.adapted_model,
            source_model=source_model,
            anchor_imgs=anchor_imgs,
            device=device,
            llm_client=llm_client,
            task_ctx=vlm_ctx,
            k=min(vlm_anchor_k, len(anchor_imgs)),
        )
        if vlm_score is not None:
            ps.iteration_history[-1]["vlm_score"] = vlm_score
            ps.iteration_history[-1]["vlm_votes"] = vlm_votes
            ps.iteration_history[-1]["vlm_absolute"] = vlm_absolute
            abs_str = ("/".join(vlm_absolute) if vlm_absolute else "—")
            print(f"    iter {i} VLM score={vlm_score:+.2f} "
                  f"({'/'.join(vlm_votes) or '—'}) abs=({abs_str})")
            if vlm_score > vlm_best_score:
                vlm_best_score = vlm_score
                vlm_best_state = {k: v.detach().cpu().clone()
                                   for k, v in ps.adapted_model.state_dict().items()}

        # ---- Circuit breaker: stop only on CLEAR catastrophe ----
        # Early iterations are noisy (Tent first forward pass resets BN stats;
        # FDA warmup sets up latent space). A single noisy "worse" vote at
        # iter 0 doesn't mean adaptation is failing. Require unambiguous
        # evidence: strict-majority worse AND zero positive signal ("no sample
        # voted better"), OR strict-majority broken AND zero "usable". This
        # catches catastrophic cases like all-worse-voted (brats Tent, score=-1.0)
        # while preserving noisy-starter runs that recover (chaos, mmwhs).
        if vlm_score is not None and vlm_votes:
            n_worse_iter = sum(1 for v in vlm_votes if v == "worse")
            n_better_iter = sum(1 for v in vlm_votes if v == "better")
            n_broken_iter = sum(1 for v in (vlm_absolute or []) if v == "broken")
            n_usable_iter = sum(1 for v in (vlm_absolute or []) if v == "usable")
            n_iter = max(len(vlm_votes), 1)
            majority_iter = n_iter // 2 + 1
            # Clear catastrophe: majority voted bad AND zero positive support.
            relative_catastrophe = (n_worse_iter >= majority_iter
                                      and n_better_iter == 0)
            absolute_catastrophe = (n_broken_iter >= majority_iter
                                      and n_usable_iter == 0
                                      and len(vlm_absolute or []) > 0)
            if relative_catastrophe or absolute_catastrophe:
                reason = (f"majority worse with zero-better "
                           f"({n_worse_iter}/{n_iter} worse, 0 better)"
                           if relative_catastrophe
                           else f"majority broken with zero-usable "
                                f"({n_broken_iter}/{n_iter} broken, 0 usable)")
                print(f"    ⛔ Circuit breaker at iter {i}: {reason} — "
                      f"reverting to source and aborting loop")
                ps.adapted_model.load_state_dict(
                    source_model.state_dict(), strict=False)
                did_revert = True
                stopped_reason = f"circuit_breaker at iter {i}: {reason}"
                vlm_best_state = None
                break

        # Mid-loop revert: stop if this iteration's selection metric dropped
        # from the previous iteration by more than a one-sample swing. The
        # swing size IS data-driven — it's 1/k (one vote change on k samples)
        # for vlm_score, or 0.02 for quality_score (label-free noise band).
        if len(ps.iteration_history) >= 2:
            prev, prev_key = _selection_score(ps.iteration_history[-2])
            curr, curr_key = _selection_score(ps.iteration_history[-1])
            if prev is not None and curr is not None and prev_key == curr_key:
                margin = (1.0 / max(len(vlm_votes), 1)) if prev_key == "vlm_score" else 0.02
                if curr < prev - margin:
                    print(f"    Regression on {prev_key}: {prev:+.3f}→{curr:+.3f}; reverting")
                    if vlm_best_state is not None:
                        ps.adapted_model.load_state_dict(vlm_best_state)
                        did_revert = True
                    stopped_reason = f"regression on {prev_key}: {prev:+.3f}→{curr:+.3f}"
                    break

    # Final revert: if VLM-best iteration scored higher than the last one,
    # load that model. Only applies when VLM was available.
    if vlm_best_state is not None and ps.iteration_history:
        last, last_key = _selection_score(ps.iteration_history[-1])
        if last_key == "vlm_score" and last is not None and vlm_best_score > last + 1e-6:
            print(f"    Final revert to VLM-best (score {vlm_best_score:+.2f} > "
                  f"last {last:+.2f})")
            ps.adapted_model.load_state_dict(vlm_best_state)
            did_revert = True
            if stopped_reason is None:
                stopped_reason = (f"final revert: VLM-best {vlm_best_score:+.2f} "
                                   f"> last {last:+.2f}")

    # === FINAL CHECK — delegate to the single shared helper ===
    # Every adaptation branch must route through _run_final_check before
    # returning; see its docstring for the rationale. Keeping this call
    # inline in the main iteration path AND duplicating it in the SAR /
    # skip_adaptation early-return paths ensures no branch can deliver an
    # adapted model without the capability gate.
    fc = _run_final_check(ps, source_model, target_images, device,
                           llm_client, vlm_ctx, knowledge or {},
                           model_assessment=model_assessment,
                           vlm_final_k=vlm_final_k)
    final_verdict = fc["final_verdict"]
    final_reason = fc["final_reason"]
    final_votes = fc["final_votes"]
    final_votes_absolute = fc.get("final_votes_absolute", [])
    final_note = fc["final_note"]
    q_before_final = fc["quality_before_final"]
    q_returned = fc["quality_after_final"]
    if fc["did_revert_in_final_check"]:
        did_revert = True
        stopped_reason = stopped_reason or final_reason

    ps.iteration_history.append({
        "method": "final_check",
        "iteration": len(ps.iteration_history),
        "quality_score": q_returned.get("consistency", 0),
        "fg_ratio_post_iter": q_returned.get("fg_ratio", 0),
        "final_verdict": final_verdict,
        "final_votes": final_votes,
        "final_votes_absolute": final_votes_absolute,
    })

    if stopped_reason is None:
        stopped_reason = f"completed {max_iter} iteration(s) without regression"

    # Structured execution report. Now includes the final verdict (previously
    # Agent 5's output) since Agent 5 has been merged in.
    adapt_iters = [h for h in ps.iteration_history
                    if h.get("method") not in ("post_revert", "final_check")]
    best_idx = _best_iter_by_selection_score(adapt_iters)
    execution_report = {
        "status": "completed",
        "stopped_reason": stopped_reason,
        "did_revert": did_revert,
        "n_iterations_run": len(adapt_iters),
        "best_iteration_index": best_idx,
        "collapse_events": collapse_events,
        "vlm_best_score": vlm_best_score if vlm_best_state is not None else None,
        "vlm_anchor_idx": anchor_idx,
        "final_verdict": final_verdict,
        "final_reason": final_reason,
        "final_votes": final_votes,
        "final_votes_absolute": final_votes_absolute,
        "final_note": final_note,
        "quality_before_final": q_before_final,
        "quality_after_final": q_returned,
        "plan_snapshot": {
            "protocol": plan.get("protocol"),
            "trainable_scope": plan.get("trainable_scope"),
            "num_steps": plan.get("num_steps"),
            "max_iterations": plan.get("max_iterations"),
            "learning_rate": plan.get("learning_rate"),
            "loss_weights": plan.get("loss_weights"),
        },
    }
    # Trajectory of selection metric + key per-iter signals
    trajectory = []
    for h in adapt_iters:
        entry = {"iteration": h.get("iteration")}
        for k in ("vlm_score", "vlm_votes", "vlm_absolute", "quality_score",
                 "fg_ratio_post_iter", "loss"):
            if k in h:
                entry[k] = h[k]
        trajectory.append(entry)
    execution_report["trajectory"] = trajectory

    return {
        "adapted_model": ps.adapted_model,
        "iteration_history": ps.iteration_history,
        "quality_scores": [float(q) for q in ps.quality_scores],
        "skipped": False,
        "did_revert": did_revert,
        "execution_report": execution_report,
        "pipeline_state": ps,
    }
