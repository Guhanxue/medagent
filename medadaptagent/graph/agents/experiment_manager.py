"""Agent 5: Experiment Manager — the outer research loop.

Wraps the existing Agent 3 v2 + Agent 4 v2 (the "inner experiment") with:

  - Iterative replan on failure: when the inner experiment's gate
    rejects all candidates (or accepts one but doesn't reach the goal),
    Manager tells Agent 3 explicitly WHAT was tried and WHY it failed,
    and asks for NEW candidates from a different mechanism family.

  - Goal tracking: optional `target_dice` (when K labels available).
    Loop ends when goal met or budget exhausted.

  - Budget tracking: `max_iterations` (default 3) hard-caps the loop.

  - Best-so-far tracking: across iterations, keep the highest-scoring
    accepted model. On budget-exhaust, deliver the best one (not just
    the last one).

  - History: every attempt is recorded with tools, verdict, reason,
    delivered Dice (if K labels). Returned to caller for audit.

  - Explicit failure mode: if no iteration produces an acceptable
    delivered model, return source with `recommendation` field
    suggesting next steps (more labels, different source model,
    document as un-deployable).

This is the "auto-fix until it works" sandbox: pipeline keeps trying
strategies within budget, learning from each attempt, until it finds
one that works or honestly reports that pure TTA can't bridge the gap.
"""
from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn

from medadaptagent.graph.agents.strategy_architect_tools import (
    run_strategy_architect_tools,
)
from medadaptagent.graph.agents.execution_engineer_tools import (
    run_execution_engineer_tools, _fewshot_dice,
)
from medadaptagent.memory_store import (
    build_fingerprint, query_similar, render_memory_context,
    append_record, make_record, DEFAULT_MEMORY_PATH,
)


@dataclass
class ExperimentOutcome:
    """Record of one inner-experiment iteration."""
    iteration: int
    plan: Dict  # Agent 3's plan (candidates + reasons)
    tools_winner: List[str]  # variant the tournament picked
    verdict: str  # "accept" | "reject"
    did_revert: bool
    final_reason: str
    delivered_model: Optional[nn.Module] = None  # the model returned by Agent 4
    delivered_metric: Optional[float] = None  # Task primary metric (Dice for seg, accuracy for cls) on K labels
    source_metric: Optional[float] = None  # Same metric on source model
    delta: Optional[float] = None  # delivered_metric - source_metric
    elapsed_s: float = 0.0
    full_exec_report: Dict = field(default_factory=dict)
    # Hypothesis schema (b-3): the chosen candidate's prediction text
    expected_outcome: str = ""


def _outcome_score(outcome: ExperimentOutcome) -> float:
    """Sort key for picking best-so-far. Higher = better.
    - If GT Dice available: use delivered_metric (absolute Dice)
    - Else: 0 if reject, 1 if accept (binary)"""
    if outcome.delivered_metric is not None:
        return outcome.delivered_metric
    return 1.0 if outcome.verdict == "accept" else 0.0


def _detect_a1_a2_disagreement(domain_analysis: Dict,
                                    model_assessment: Dict,
                                    src_metric_baseline: Optional[float]
                                    ) -> Optional[Dict[str, str]]:
    """Detect cases where Agent 1 and Agent 2 produce inconsistent
    diagnoses. Returns a {summary, prompt_block} dict when found, else
    None.

    Three conflict patterns we currently catch:

    1. A2 says "complete_failure" / "mode_collapse" / "source_broken"
       BUT A1 says shift_level<=1 (mild) — likely A2 over-labeling.
    2. A2 says severe failure_type BUT source_quality>0.8 — model is
       clearly working on its own data, so "complete_failure" on
       target is suspicious.
    3. K-label metric is already strong (>=0.85) BUT A2 says severe —
       quantitative evidence contradicts A2's categorical label.

    For each conflict, the prompt block tells Agent 3 to weight the
    QUANTITATIVE signals (A1's specific shift diagnosis, source_quality
    score, K-label metric) over A2's coarse categorical label."""
    da = domain_analysis or {}
    ma = model_assessment or {}
    sev_labels = {"complete_failure", "mode_collapse", "source_broken",
                    "over_confident_wrong"}
    ft = str(ma.get("failure_type", "")).lower()
    if ft not in sev_labels:
        return None
    conflicts = []
    a1_shift = da.get("shift_level")
    if isinstance(a1_shift, (int, float)) and a1_shift <= 1:
        conflicts.append(
            f"A1 reports shift_level={a1_shift} (mild) while A2 reports "
            f"failure_type='{ft}' (severe)")
    sq = ma.get("source_quality_score")
    if isinstance(sq, (int, float)) and sq >= 0.8:
        conflicts.append(
            f"A2 reports failure_type='{ft}' (severe) but source_quality "
            f"= {sq:.3f} (the model works on its own data — failure on "
            f"target is likely moderate, not complete)")
    if isinstance(src_metric_baseline, (int, float)) and src_metric_baseline >= 0.85:
        conflicts.append(
            f"K-label metric is already {src_metric_baseline:.3f} on the "
            f"source model — quantitative evidence contradicts A2's "
            f"'{ft}' label")
    if not conflicts:
        return None
    summary = " | ".join(conflicts[:2])  # short stdout
    prompt_block = (
        "## Dynamic-Topology Note: Agent 1 / Agent 2 Disagreement Detected\n"
        "\n"
        "The pipeline detected the following inconsistencies between "
        "Agent 1's domain diagnosis and Agent 2's failure assessment:\n"
        + "\n".join(f"  - {c}" for c in conflicts) + "\n\n"
        "GUIDANCE FOR STRATEGY: Weight Agent 1's specific shift diagnosis "
        "(dominant_shift_factor) and quantitative signals (source_quality "
        "score, K-label baseline) MORE heavily than Agent 2's categorical "
        "failure label. Do NOT escalate to aggressive tools (finetune_partial, "
        "finetune_lora) just because A2's label is severe — prefer "
        "lightweight tools matched to A1's specific shift unless the "
        "quantitative evidence ALSO supports aggressive intervention."
    )
    return {"summary": summary, "prompt_block": prompt_block}


def _maybe_clarification_block(history: List["ExperimentOutcome"]) -> str:
    """When a previous iteration's plan emitted a clarification_request,
    surface it so the next Agent 3 call can refine accordingly. This is
    the multi-turn beat: Agent 3 can ask, and the next invocation has
    the question in context."""
    if not history:
        return ""
    pending = []
    for h in history:
        cr = (h.plan or {}).get("clarification_request")
        if cr and isinstance(cr, str) and cr.strip():
            pending.append((h.iteration, cr.strip()))
    if not pending:
        return ""
    lines = [
        "## Multi-Turn: Prior clarification requests from THIS run",
        "",
        "You (Agent 3) raised the following questions in previous "
        "iterations. The evidence may not have been re-examined yet — "
        "treat as a self-note to refine your strategy this iteration:",
        "",
    ]
    for it, q in pending[-3:]:  # only show the most recent 3
        lines.append(f"  - iter {it}: \"{q[:200]}\"")
    lines.append("")
    return "\n".join(lines)


TOOL_FAMILY = {
    "bn_reset": "stats_norm",
    "tent": "gradient_norm", "sar": "gradient_norm",
    "eata": "gradient_norm", "tent_ss": "gradient_norm",
    "sar_ss": "gradient_norm", "eata_ss": "gradient_norm",
    "lora_tta": "gradient_norm",
    "finetune_head": "gradient_supervised",
    "finetune_partial": "gradient_supervised",
    "finetune_lora": "gradient_supervised",
    "histogram_match": "intensity",
    "percentile_clip": "intensity",
    "clahe": "intensity",
    "fda_warmup": "frequency",
    "crop_center": "spatial",
    "tta_aug": "ensemble",
    "noop": "noop",
}


def _families_of(tools):
    return sorted({TOOL_FAMILY.get(t, "other") for t in tools})


def _diagnose_failure_mechanism(
        tools_winner: List[str], rel_drop: float, abs_drop: float,
        src_metric: float, delivered: float,
        knowledge: Dict, model_assessment: Dict) -> str:
    """Phase D1: causal interpretation of a Q3-catastrophic iter.

    Given the tools that failed + the source/adapted metrics + Agent 1's
    diagnosis + Agent 2's assessment, return a one-sentence mechanism
    hypothesis explaining WHY the family failed, not just THAT it
    failed. This becomes a directive for Agent 3 in the next iter."""
    fams = set(_families_of(tools_winner))
    sq = (model_assessment or {}).get("source_quality_score", 0)
    shift = (knowledge or {}).get("shift_level", 1)
    ft = (model_assessment or {}).get("failure_type", "")

    # Per-family causal templates, gated by Agent 1/2 signals
    if "gradient_norm" in fams:  # tent / sar / eata / tent_ss
        if isinstance(sq, (int, float)) and sq >= 0.85:
            return (
                "Gradient-norm family (Tent/SAR family) over-fit on "
                "high-confidence predictions. Source is competent on "
                f"source domain (sq={sq:.2f}); entropy minimization "
                "amplified spurious confidence on target samples that "
                "the model was already classifying correctly. "
                "MECHANISM: low-LR entropy + reliability filter, or "
                "drop gradient-norm entirely.")
        return (
            "Gradient-norm family collapsed predictions. LR too "
            "aggressive for this shift severity. MECHANISM: try "
            "lower LR (1e-5 to 5e-5) or SAR with high entropy "
            "margin (>0.8) to filter unreliable updates.")
    if "stats_norm" in fams:  # bn_reset
        if shift and int(shift) <= 1:
            return (
                "Stats-norm family (bn_reset) re-estimated channel "
                "statistics on a target whose shift was MILD; the "
                "old BN stats were closer to optimal than the new "
                f"target stats (Agent 1 shift_level={shift}). "
                "MECHANISM: skip BN reset entirely, or use partial "
                "BN reset with running-stats momentum 0.99 instead "
                "of full reset.")
        return (
            "Stats-norm family (bn_reset) re-estimated stats on a "
            "heterogeneous target set; the new BN stats are biased "
            "toward outlier samples. MECHANISM: try per-sample BN "
            "or accumulate stats over more target samples before "
            "freezing.")
    if "gradient_supervised" in fams:  # finetune_head / partial / lora
        return (
            "Gradient-supervised family overfit on K labels (small "
            "supervised set + high-capacity model). "
            f"src={src_metric:.3f} -> adapted={delivered:.3f}. "
            "MECHANISM: lower LR by 10x, freeze more layers "
            "(use finetune_head not finetune_partial), or use LoRA "
            "with rank<=4 for low-K regimes.")
    if "intensity" in fams:  # histogram_match / clahe / percentile_clip
        return (
            "Intensity family (histogram_match etc.) remapped "
            "target intensities in a way that violates the source's "
            "learned feature space. MECHANISM: try a different "
            "preprocessing chain (FDA frequency-domain warmup, "
            "percentile_clip with looser range), or pair with "
            "bn_reset to absorb the residual stats shift.")
    if "frequency" in fams:  # fda_warmup
        return (
            "Frequency family (FDA) over-warped low-freq target "
            "components; L too high for this shift. MECHANISM: try "
            "L=0.005 or L=0.01 (smaller warmup region).")
    if "spatial" in fams:  # crop_center
        return (
            "Spatial family (crop_center) cropped away the actual "
            "target structure (anatomy was at edges). MECHANISM: "
            "skip cropping or use crop_fraction>=0.85.")
    # default
    return (
        f"Family {sorted(fams)} caused {rel_drop:.0%} relative "
        f"Dice drop on K-shot eval. MECHANISM: unknown — try a "
        f"completely different family this iter.")


def _build_exploration_directive(history: List[ExperimentOutcome],
                                     iteration: int,
                                     catastrophic_families: set,
                                     mechanism_diagnoses: List[str] = None) -> str:
    """Phase Q1+Q3: when entering iter >= 1, render an explicit
    exploration directive listing past WINNER tool-sets and forbidding
    repetition. Also, if Phase Q3 flagged any tool families as
    catastrophic this run, list them as banned."""
    if iteration == 0 and not catastrophic_families:
        return ""
    lines = ["## Exploration directive (Phase Q1)", ""]
    if history:
        past_winners = []
        past_fams = set()
        for h in history:
            tk = tuple(h.tools_winner) if h.tools_winner else ("(no-op)",)
            past_winners.append(tk)
            past_fams.update(_families_of(h.tools_winner))
        lines.append("Past iter winners in THIS run:")
        for i, tk in enumerate(past_winners):
            lines.append(f"  iter {i}: tools={list(tk)} "
                          f"families={_families_of(list(tk))}")
        lines.append("")
        lines.append("HARD RULES for this iteration:")
        lines.append("  1. Do NOT propose any candidate whose tool-set "
                      "equals any past iter's winner above. Tournament "
                      "validator will reject exact-repeat candidates.")
        lines.append(f"  2. Past iterations covered families: "
                      f"{sorted(past_fams)}. Your NEW candidates must "
                      "explore AT LEAST ONE family NOT in this set "
                      "(unless you have a strong reason to stay).")
        lines.append("  3. If a past winner showed marginal delta "
                      "(|delta|<0.05), one of your candidates should be "
                      "a HYPERPARAMETER SWEEP variant of that tool "
                      "(non-default tool_kwargs).")
    if catastrophic_families:
        lines.append("")
        lines.append("BANNED families (caused >=50% Dice collapse "
                      "earlier this run):")
        for f in sorted(catastrophic_families):
            lines.append(f"  - {f}")
        lines.append("  -> Do NOT propose any tool from these "
                      "families this iteration.")
    if mechanism_diagnoses:
        # Phase D1: causal interpretation of past failures.
        lines.append("")
        lines.append("MECHANISTIC HYPOTHESES (Phase D1) — read these "
                      "before proposing the next iteration:")
        for i, d in enumerate(mechanism_diagnoses):
            lines.append(f"  [{i}] {d}")
        lines.append("  -> Your new candidates should ADDRESS these "
                      "mechanisms, not just propose unrelated tools. "
                      "If a mechanism suggests 'lower LR', pass that "
                      "via tool_kwargs.")
    return "\n".join(lines)


def _build_replan_context(history: List[ExperimentOutcome]) -> str:
    """Render past attempts as a prompt block Agent 3 can read in
    its NEXT iteration. Includes (a) what was tried, (b) the hypothesis
    Agent 3 had for it, and (c) what actually happened. The
    prediction-vs-reality comparison is the calibration signal: if
    Agent 3's predictions are systematically wrong, it should adjust."""
    if not history:
        return ""
    lines = [
        "## Replan context — past attempts in this run",
        "",
        "Each entry shows the strategy tried, what you predicted would "
        "happen, and what actually happened. Use the gap between "
        "prediction and reality as a calibration signal. Avoid "
        "proposing the same mechanism family again unless you have a "
        "strong, specific reason; prefer a DIFFERENT cause-class "
        "hypothesis this round.",
        "",
    ]
    for h in history:
        gt = (f"GT delta={h.delta:+.4f}" if h.delta is not None
              else "no GT signal")
        lines.append(
            f"  iter {h.iteration}: tools={h.tools_winner}")
        if h.expected_outcome:
            lines.append(f"    YOU PREDICTED: {h.expected_outcome[:200]}")
        lines.append(
            f"    ACTUAL: verdict={h.verdict}, "
            f"did_revert={h.did_revert}, {gt}")
        lines.append(f"    final_reason: {h.final_reason[:160]}")
        lines.append("")
    return "\n".join(lines)


def run_experiment_manager(
    source_model: nn.Module,
    target_images: List[np.ndarray],
    knowledge: Dict,
    model_assessment: Dict,
    model_info: Dict,
    llm_client,
    num_classes: int,
    device: str = "cuda:0",
    fewshot_indices: Optional[List[int]] = None,
    fewshot_labels: Optional[List[np.ndarray]] = None,
    target_dice: Optional[float] = None,
    max_iterations: int = 3,
    vlm_final_k: int = 5,
    vlm_tournament_k: int = 3,
    dataset_name: Optional[str] = None,
    memory_path: Optional[str] = None,
    use_memory: bool = True,
    arch: str = "unet",
    task=None,
    metric: Optional[str] = None,
    goal: str = "ship",
    agent1_recall_fn=None,
) -> Dict:
    """Outer research loop. Returns a dict with the delivered model
    plus full history and recommendation."""
    print("\n" + "=" * 55)
    print(f"AGENT 5: Experiment Manager (max_iterations={max_iterations})")
    print(f"  goal: {goal}")
    if target_dice is not None:
        print(f"  target metric >= {target_dice:.3f}")
    if fewshot_indices:
        print(f"  K={len(fewshot_indices)} labels available for verification")
    print("=" * 55)

    # Goal: diagnose — run A1+A2 only, skip A3/A4/A5 entirely. Return
    # diagnosis report without attempting any adaptation. Used when the
    # user wants to know "what's wrong" before deciding to spend GPU on
    # adaptation tools or label collection.
    if goal == "diagnose":
        print("  GOAL=diagnose: returning A1+A2 diagnosis without adaptation")
        recommendation = "diagnose_only_no_adaptation_attempted"
        if (model_assessment or {}).get("source_quality_score", 0) >= 0.85:
            recommendation = "source_already_strong_no_adaptation_needed"
        elif (model_assessment or {}).get("failure_type") in (
                "source_broken", "mode_collapse"):
            recommendation = ("model_broken_supervised_training_required")
        fingerprint_for_record = build_fingerprint(
            knowledge, model_assessment, model_info,
            task_kind=("cls" if getattr(task, "name", "").startswith("classification")
                          else "seg"),
            metric_name=metric)
        return {
            "n_iterations": 0,
            "history": [],
            "best_iteration": None,
            "delivered_model": source_model,  # untouched
            "delivered_metric": None,
            "source_metric": None,
            "metric_name": metric,
            "delta": None,
            "verdict": "diagnose_only",
            "recommendation": recommendation,
            "tools_applied": [],
            "goal": goal,
            "diagnosis": {
                "domain_analysis": knowledge,
                "model_assessment": model_assessment,
                "fingerprint": fingerprint_for_record,
            },
        }

    history: List[ExperimentOutcome] = []
    src_metric_baseline = None

    # Query persistent memory once at the start (top-K similar past runs).
    # Re-used across all iterations of this run as a prior for Agent 3.
    # Derive task_kind from the task adapter so seg/cls memories stay
    # in their own buckets.
    _task_name = getattr(task, "name", "") if task is not None else ""
    _task_kind = ("cls" if _task_name.startswith("classification")
                    else "seg" if _task_name.startswith("segmentation")
                    else "unknown")
    _metric_fp = metric or (getattr(task, "metric_name", "unknown") if task else "unknown")
    fingerprint = build_fingerprint(knowledge, model_assessment, model_info,
                                          task_kind=_task_kind,
                                          metric_name=_metric_fp)
    memory_ctx_str = ""
    if use_memory:
        path = memory_path or DEFAULT_MEMORY_PATH
        nbrs = query_similar(fingerprint, current_dataset=dataset_name,
                                k=3, path=path)
        if nbrs:
            memory_ctx_str = render_memory_context(nbrs)
            print(f"  memory: {len(nbrs)} similar past run(s) loaded")
        else:
            print("  memory: no similar past runs (fresh task)")
    if fewshot_indices and fewshot_labels:
        src_metric_baseline = _fewshot_dice(
            source_model, target_images, fewshot_indices, fewshot_labels,
            num_classes, device, task=task, metric=metric)
        _mname = (metric or getattr(task, "metric_name", "dice"))
        print(f"  source few-shot {_mname} = {src_metric_baseline:.4f}")

    best_outcome: Optional[ExperimentOutcome] = None
    a1_recalled = False  # Phase P: only allow one A3->A1 recall per run
    catastrophic_families: set = set()  # Phase Q3
    mechanism_diagnoses: List[str] = []   # Phase D1

    # Dynamic agent topology: detect A1/A2 disagreement once at the top
    # of the loop (not per-iter since A1+A2 outputs are stable within a
    # run). When detected, inject a reconciliation directive into
    # Agent 3's context so it weights the evidence appropriately rather
    # than blindly trusting A2's categorical label.
    disagreement_ctx = _detect_a1_a2_disagreement(knowledge, model_assessment,
                                                          src_metric_baseline)
    if disagreement_ctx:
        print(f"  [dynamic] A1/A2 disagreement: {disagreement_ctx['summary']}")

    for iteration in range(max_iterations):
        print(f"\n--- ITERATION {iteration} ---")

        # 1. PROPOSE: Agent 3 with replan context if any
        replan_ctx = _build_replan_context(history)
        # Multi-turn: surface prior clarification_request from past
        # iters so Agent 3 self-dialogues across iterations.
        clarification_ctx = _maybe_clarification_block(history)
        if clarification_ctx and iteration > 0:
            print(f"  [multi-turn] forwarding {clarification_ctx.count('  - iter')} "
                   f"prior clarification request(s) to Agent 3")
        # Phase Q1: exploration directive listing past winners +
        # banned families (Phase Q3 catastrophe feedback) +
        # mechanism diagnoses (Phase D1).
        exploration_ctx = _build_exploration_directive(
            history, iteration, catastrophic_families,
            mechanism_diagnoses=mechanism_diagnoses)
        # Combine memory context (cross-run), replan context (within-run),
        # disagreement context (dynamic-topology arbitration), and
        # clarification context (multi-turn self-dialogue).
        ctx_parts = []
        if memory_ctx_str:
            ctx_parts.append(memory_ctx_str)
        if disagreement_ctx:
            ctx_parts.append(disagreement_ctx["prompt_block"])
        if clarification_ctx:
            ctx_parts.append(clarification_ctx)
        if exploration_ctx:
            ctx_parts.append(exploration_ctx)
        if replan_ctx:
            ctx_parts.append(replan_ctx)
        combined_ctx = "\n\n".join(ctx_parts) if ctx_parts else None
        # Phase Q1: collect past winner tool-sets for the validator to
        # reject exact-repeat candidates.
        past_winner_toolsets = [tuple(h.tools_winner) for h in history
                                  if h.tools_winner]
        plan = run_strategy_architect_tools(
            domain_analysis=knowledge,
            model_assessment=model_assessment,
            llm_client=llm_client,
            model_info=model_info,
            replan_context=combined_ctx,
            past_winner_toolsets=past_winner_toolsets,
            catastrophic_families=catastrophic_families,
        )

        # Phase P: TRUE multi-turn — if Agent 3 emitted a clarification
        # request AND we have a callback to re-call Agent 1 AND we
        # haven't already done a recall this run, invoke A1 with the
        # question and replace `knowledge` for the next iteration.
        # Cap at 1 recall per run to prevent infinite loops.
        if (agent1_recall_fn is not None
                and plan.get("clarification_request")
                and not a1_recalled):
            q = plan["clarification_request"]
            print(f"  [Phase P] Agent 3 -> Agent 1 callback: \"{q[:100]}\"")
            try:
                refined = agent1_recall_fn(q)
                if refined and isinstance(refined, dict):
                    knowledge = refined
                    a1_recalled = True
                    print(f"  [Phase P] A1 refined diagnosis: "
                           f"dominant_shift={refined.get('dominant_shift_factor', '?')[:60]}")
            except Exception as _e:
                print(f"  [Phase P] A1 recall failed ({_e}); proceeding "
                       f"with original diagnosis")

        # 2. EXECUTE: Agent 4 with full inner pipeline
        t0 = time.time()
        exec_result = run_execution_engineer_tools(
            plan=plan,
            source_model=source_model,
            target_images=target_images,
            task_config=None,
            knowledge=knowledge,
            device=device,
            llm_client=llm_client,
            model_assessment=model_assessment,
            num_classes=num_classes,
            vlm_final_k=vlm_final_k,
            vlm_tournament_k=vlm_tournament_k,
            fewshot_indices=fewshot_indices,
            fewshot_labels=fewshot_labels,
            task=task,
        )
        elapsed = time.time() - t0

        # 3. RECORD
        er = exec_result.get("execution_report", {})
        # Find which Agent-3 candidate won (by tool sequence match)
        winner_tools = er.get("tools_applied", [])
        chosen_expected = ""
        for c in (plan.get("candidates") or []):
            if list(c.get("tools", [])) == list(winner_tools):
                chosen_expected = c.get("expected_outcome", "")
                break
        outcome = ExperimentOutcome(
            iteration=iteration,
            plan=plan,
            tools_winner=winner_tools,
            verdict=er.get("final_verdict", "?"),
            did_revert=bool(er.get("did_revert", False)),
            final_reason=er.get("final_reason", "")[:300],
            delivered_model=exec_result.get("adapted_model"),
            elapsed_s=elapsed,
            full_exec_report=er,
            expected_outcome=chosen_expected,
        )
        if src_metric_baseline is not None and outcome.delivered_model is not None:
            outcome.source_metric = src_metric_baseline
            outcome.delivered_metric = _fewshot_dice(
                outcome.delivered_model, target_images,
                fewshot_indices, fewshot_labels,
                num_classes, device, task=task, metric=metric)
            outcome.delta = outcome.delivered_metric - src_metric_baseline
            print(f"  iter {iteration}: tools={outcome.tools_winner} "
                  f"verdict={outcome.verdict} delta={outcome.delta:+.4f}")
            # Phase Q3: catastrophe detection on few-shot dice. 50%+
            # relative drop OR absolute drop > 0.15 = catastrophic.
            # Tag the tool families involved so the next iteration
            # avoids them via the exploration directive.
            if src_metric_baseline > 0.10:
                rel_drop = (src_metric_baseline - outcome.delivered_metric) / src_metric_baseline
                abs_drop = src_metric_baseline - outcome.delivered_metric
                if rel_drop > 0.50 or abs_drop > 0.15:
                    fams = _families_of(outcome.tools_winner)
                    catastrophic_families.update(fams)
                    print(f"  [Phase Q3] CATASTROPHE: tools={outcome.tools_winner} "
                           f"families={fams} dropped few-shot dice "
                           f"{src_metric_baseline:.4f} -> {outcome.delivered_metric:.4f} "
                           f"(rel={rel_drop:+.2%}). Banning these families next iter.")
                    # Phase D1: causal mechanism interpretation. Feeds
                    # the WHY (not just the WHAT) into Agent 3 next
                    # iter, so it can propose a tool that addresses
                    # the actual failure mode, not just a different
                    # random family.
                    diag = _diagnose_failure_mechanism(
                        outcome.tools_winner, rel_drop, abs_drop,
                        src_metric_baseline, outcome.delivered_metric,
                        knowledge, model_assessment)
                    mechanism_diagnoses.append(diag)
                    print(f"  [Phase D1] mechanism: {diag[:200]}")
        else:
            print(f"  iter {iteration}: tools={outcome.tools_winner} "
                  f"verdict={outcome.verdict}")
        history.append(outcome)

        # 4. BEST-SO-FAR
        if best_outcome is None or _outcome_score(outcome) > _outcome_score(best_outcome):
            best_outcome = outcome
            print(f"  iter {iteration}: new best (score={_outcome_score(outcome):.4f})")

        # 5. GOAL CHECK
        if (target_dice is not None and outcome.delivered_metric is not None
                and outcome.delivered_metric >= target_dice):
            print(f"  GOAL MET at iter {iteration} "
                  f"(delivered={outcome.delivered_metric:.4f} >= {target_dice:.3f})")
            break

        # 6. EARLY-STOP: 2 consecutive rejects + no GT Dice signal AND we've
        #    already tried 2 distinct mechanism families => give up.
        if iteration >= 1 and not history[-1].verdict == "accept" \
                and not history[-2].verdict == "accept" \
                and src_metric_baseline is None:
            recent_tools = (set(history[-1].tools_winner) |
                              set(history[-2].tools_winner))
            if len(recent_tools) >= 3:
                print(f"  Two consecutive rejects with diverse tools "
                      f"({recent_tools}); stopping")
                break

    # 7. DELIVER best-so-far + recommendation
    print(f"\n--- LOOP COMPLETE: {len(history)} iterations ---")

    # Phase Q3 (run-level safety net): if we have GT signal AND best is
    # worse than source, replace best with a "source-keep" outcome.
    # Closes the mmwhs-style collapse hole where the agent shipped
    # delivered=0.0 below source=0.04. This NEVER fires when there's no
    # GT signal (K=0); Phase Q4 handles that case differently.
    if (best_outcome is not None and src_metric_baseline is not None
            and best_outcome.delivered_metric is not None
            and best_outcome.delivered_metric < src_metric_baseline):
        print(f"  [Phase Q3] Best iter delivered "
               f"{best_outcome.delivered_metric:.4f} < source "
               f"{src_metric_baseline:.4f} — replacing best with "
               f"source-keep (never ship worse than source).")
        # Replace with a synthetic source-keep outcome
        src_keep = ExperimentOutcome(
            iteration=-1, plan={}, tools_winner=[],
            verdict="reject_source_keep", did_revert=True,
            final_reason=(f"Phase Q3 safety: best delivered "
                            f"{best_outcome.delivered_metric:.4f} < "
                            f"source {src_metric_baseline:.4f}; "
                            f"shipping source instead"),
            delivered_model=source_model,
            elapsed_s=0.0, full_exec_report={},
            expected_outcome="",
        )
        src_keep.source_metric = src_metric_baseline
        src_keep.delivered_metric = src_metric_baseline
        src_keep.delta = 0.0
        best_outcome = src_keep

    if best_outcome is None:
        # Should not happen — always at least 1 iteration
        recommendation = "no_attempts"
    elif (src_metric_baseline is not None and src_metric_baseline >= 0.95):
        # Source is already saturated on K labels — no adaptation needed.
        # The K labels we drew happen to all be correct (lucky draw or
        # source genuinely strong on this target). Either way, more
        # labels won't help: ship source.
        recommendation = "ship_source_already_strong"
    elif best_outcome.verdict == "accept" and (
            best_outcome.delta is None or best_outcome.delta > 0):
        recommendation = "ship"
    elif best_outcome.delta is not None and best_outcome.delta > 0:
        # GT signal positive but verdict=reject (gate too conservative)
        recommendation = "ship_with_caveat"
    elif src_metric_baseline is None:
        recommendation = "label_recommended_K=5"
    else:
        # Source is sub-optimal AND pipeline couldn't beat it with this
        # K. More labels (semi-supervised) is the principled escalation.
        recommendation = "label_recommended_K=20_for_semi_supervised"

    print(f"  best iter: {best_outcome.iteration}, verdict="
          f"{best_outcome.verdict}, "
          f"delivered_metric={best_outcome.delivered_metric}")
    print(f"  recommendation: {recommendation}")

    # Persist this run to memory (skip if dataset_name is None)
    if use_memory and dataset_name is not None:
        path = memory_path or DEFAULT_MEMORY_PATH
        # Build a result dict that matches make_record's expected fields
        record_input = {
            "tools_applied": best_outcome.tools_winner,
            "verdict": best_outcome.verdict,
            "delta": best_outcome.delta,
            "source_metric": best_outcome.source_metric,
            "delivered_metric": best_outcome.delivered_metric,
            "n_iterations": len(history),
            "recommendation": recommendation,
        }
        append_record(make_record(dataset_name, arch, fingerprint,
                                       record_input), path=path)
        print(f"  memory: appended record for {dataset_name}")

    return {
        "n_iterations": len(history),
        "history": [
            {"iteration": h.iteration, "tools": h.tools_winner,
             "verdict": h.verdict, "did_revert": h.did_revert,
             "delta": h.delta, "delivered_metric": h.delivered_metric,
             "elapsed_s": h.elapsed_s, "reason": h.final_reason,
             "expected_outcome": h.expected_outcome}
            for h in history
        ],
        "best_iteration": best_outcome.iteration,
        "delivered_model": best_outcome.delivered_model,
        "delivered_metric": best_outcome.delivered_metric,
        "source_metric": src_metric_baseline,
        "metric_name": (metric or getattr(task, "metric_name", "dice")),
        "delta": best_outcome.delta,
        "verdict": best_outcome.verdict,
        "recommendation": recommendation,
        "tools_applied": best_outcome.tools_winner,
        "final_reason": best_outcome.final_reason,
        "goal": goal,
    }
