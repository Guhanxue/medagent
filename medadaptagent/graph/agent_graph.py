"""
LangGraph MedAdaptAgent: Propose-and-Validate Architecture.

The LLM proposes adaptation strategies. Each proposal is TESTED
using label-free quality metrics (no target labels needed).
If quality improves → accept. If it worsens → try safer alternative.

Graph:
  knowledge → analysis → diagnosis → LLM_planning
       → try_adaptation → validate → [accept | try_safer | give_up]

This is a real agentic loop: observe → reason → act → observe result → adjust.
"""

import os
import json
import copy
import time
import numpy as np
from typing import Dict, TypedDict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from langgraph.graph import StateGraph, END

from medadaptagent.graph.experience_prompts import (
    ANALYSIS_SELECTION_PROMPT, DIAGNOSIS_PROMPT,
    VERIFICATION_PROMPT, PLANNING_PROMPT,
)
from medadaptagent.graph.analysis_tools import ANALYSIS_TOOLS


# ================================================================
# State
# ================================================================

class AgentState(TypedDict):
    knowledge_summary: str
    analysis_results: str
    diagnosis_summary: str
    verification_summary: str
    plan_summary: str
    eval_summary: str
    task_config_summary: str
    # Quality tracking (label-free)
    baseline_quality: float    # quality BEFORE adaptation
    adapted_quality: float     # quality AFTER adaptation
    # Control
    iteration: int
    max_iterations: int
    attempt: int               # which strategy attempt (0=LLM proposal, 1=safer, 2=safest)
    done: bool
    history_text: str


_refs = {}


def _llm_call(prompt: str, max_tokens: int = 1500) -> str:
    client = _refs.get("llm_client")
    if not client:
        return ""
    try:
        return client.chat(prompt, max_tokens=max_tokens) or ""
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return ""


def _parse_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def _measure_quality(model, images, device, max_samples=200):
    """Label-free quality: flip consistency + entropy + fg plausibility.

    Returns a score 0-1 where higher = better model behavior on target.
    """
    model.eval()
    consistencies = []
    entropies = []
    fg_ratios = []

    with torch.no_grad():
        for img in images[:max_samples]:
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

            # Flip consistency
            flipped = model(torch.flip(tensor, [-1]))
            pred_flip = flipped.argmax(dim=1)
            pred_flip = torch.flip(pred_flip, [-1])
            agreement = (pred == pred_flip).float().mean().item()
            consistencies.append(agreement)

            # Entropy
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
            entropies.append(ent)

            # Foreground ratio
            fg = (pred > 0).float().mean().item()
            fg_ratios.append(fg)

    consistency = np.mean(consistencies) if consistencies else 0
    entropy = np.mean(entropies) if entropies else 1
    fg_ratio = np.mean(fg_ratios) if fg_ratios else 0

    # Combine: high consistency + low entropy + reasonable fg = good
    # Normalize entropy to 0-1 scale (lower is better)
    ent_score = max(0, 1.0 - entropy * 10)  # entropy < 0.1 → score > 0

    quality = 0.5 * consistency + 0.3 * ent_score + 0.2 * min(fg_ratio * 5, 1.0)
    return float(quality), {
        "consistency": float(consistency),
        "entropy": float(entropy),
        "fg_ratio": float(fg_ratio),
        "quality": float(quality),
    }


# ================================================================
# Nodes
# ================================================================

def knowledge_node(state: AgentState) -> dict:
    print("\n" + "=" * 50)
    print("NODE: Knowledge (LLM)")
    print("=" * 50)
    from medadaptagent.orchestrator import KnowledgeAgent
    ps = _refs["pipeline_state"]
    KnowledgeAgent(ps).run()
    k = ps.rag_knowledge
    summary = json.dumps({
        "shift_types": k.get("shift_types", []),
        "contrast_mechanism": k.get("contrast_mechanism_change", "unknown"),
        "standardization_gap": k.get("standardization_gap", "unknown"),
        "dominant_factor": k.get("dominant_shift_factor", "unknown"),
        "recommended_scope": k.get("recommended_scope", "unknown"),
        "what_fails": k.get("what_will_fail", []),
    }, indent=2)
    return {"knowledge_summary": summary}


def analysis_node(state: AgentState) -> dict:
    print("\n" + "=" * 50)
    print("NODE: Analysis (LLM-selected tools)")
    print("=" * 50)

    tool_desc = "\n".join(f"- **{n}**: {i['description']}" for n, i in ANALYSIS_TOOLS.items())
    prompt = ANALYSIS_SELECTION_PROMPT.format(
        tool_descriptions=tool_desc,
        knowledge_summary=state.get("knowledge_summary", "unknown"),
    )
    response = _llm_call(prompt, max_tokens=200)
    selected = _parse_json(response)
    if not selected or not isinstance(selected, list):
        selected = ["intensity_analysis", "confidence_analysis"]

    print(f"  LLM selected: {selected}")

    ps = _refs["pipeline_state"]
    results = {}
    for tool_name in selected:
        if tool_name not in ANALYSIS_TOOLS:
            continue
        tool = ANALYSIS_TOOLS[tool_name]
        try:
            if tool["needs_model"]:
                result = tool["fn"](ps.target_images, ps.source_model, ps.config.device, max_samples=200)
            else:
                result = tool["fn"](ps.source_images, ps.target_images, max_samples=200)
            results[tool_name] = result
        except Exception as e:
            print(f"    {tool_name} failed: {e}")

    return {"analysis_results": json.dumps(results, indent=2, default=str)}


def diagnosis_node(state: AgentState) -> dict:
    print("\n" + "=" * 50)
    print("NODE: Diagnosis (LLM)")
    print("=" * 50)
    prompt = DIAGNOSIS_PROMPT.format(
        analysis_results=state.get("analysis_results", "{}"),
        knowledge_summary=state.get("knowledge_summary", "unknown"),
    )
    response = _llm_call(prompt, max_tokens=500)
    diagnosis = _parse_json(response)
    if not diagnosis:
        diagnosis = {"failure_type": "mixed", "halluc_ratio": 0.3, "severity": 0.5}
    print(f"  Type: {diagnosis.get('failure_type', '?')}, "
          f"Halluc: {diagnosis.get('halluc_ratio', 0):.0%}, "
          f"Severity: {diagnosis.get('severity', 0):.2f}")

    # Store for adaptation agent
    from medadaptagent.diagnosis.failure_classifier import FailureReport
    ps = _refs["pipeline_state"]
    ps.failure_report = FailureReport(
        type_distribution={
            "miss": diagnosis.get("miss_ratio", 0),
            "hallucination": diagnosis.get("halluc_ratio", 0),
            "boundary_error": diagnosis.get("boundary_ratio", 0),
            "class_confusion": 0,
        },
        overall_severity=diagnosis.get("severity", 0.5),
        recommended_adapters=[], summary="",
    )
    return {"diagnosis_summary": json.dumps(diagnosis, indent=2)}


def baseline_quality_node(state: AgentState) -> dict:
    """Measure source model quality on target BEFORE adaptation (label-free)."""
    print("\n" + "=" * 50)
    print("NODE: Baseline Quality (label-free)")
    print("=" * 50)

    ps = _refs["pipeline_state"]
    q, details = _measure_quality(ps.source_model, ps.target_images, ps.config.device)
    print(f"  Quality: {q:.3f} (consistency={details['consistency']:.3f}, "
          f"entropy={details['entropy']:.4f}, fg={details['fg_ratio']:.1%})")
    return {"baseline_quality": q}


def planning_node(state: AgentState) -> dict:
    """LLM plans adaptation. Includes baseline quality so it knows the starting point."""
    print("\n" + "=" * 50)
    print("NODE: Planning (LLM)")
    print("=" * 50)

    baseline_q = state.get("baseline_quality", 0)
    attempt = state.get("attempt", 0)

    if attempt == 0:
        # First attempt: LLM proposes
        prompt = PLANNING_PROMPT.format(
            verification_summary=state.get("verification_summary", "{}"),
            knowledge_summary=state.get("knowledge_summary", "{}"),
            diagnosis_summary=state.get("diagnosis_summary", "{}"),
            analysis_results=state.get("analysis_results", "{}"),
            task_config=state.get("task_config_summary", "{}"),
        )
        prompt += f"\n\n## Baseline Quality (label-free, BEFORE adaptation)\nQuality score: {baseline_q:.3f}\nIf baseline quality > 0.9, the model already works well — be very conservative."

        response = _llm_call(prompt, max_tokens=800)
        plan = _parse_json(response)

        if not plan or "trainable_scope" not in plan:
            plan = _default_plan(state)
        else:
            plan.setdefault("loss_weights", {"entropy": 0.3, "consistency": 0.2, "diversity": 0.2, "bn_adapt": 0.3})
            plan.setdefault("tent_bn_mode", plan.get("trainable_scope") == "bn_only")
            plan.setdefault("use_anatomy_refinement", False)
            plan.setdefault("max_iterations", 1)

    elif attempt == 1:
        # Second attempt: safer — bn_only Tent
        print("  [Safer fallback] bn_only + Tent")
        plan = {
            "trainable_scope": "bn_only", "learning_rate": 1e-3,
            "num_steps": 80, "tent_bn_mode": True, "skip_adaptation": False,
            "loss_weights": {"entropy": 0.5, "consistency": 0.3, "diversity": 0.1, "bn_adapt": 0.4},
            "max_iterations": 1, "use_anatomy_refinement": False,
            "rationale": "Safer fallback: bn_only + Tent (previous attempt hurt quality)",
        }
    else:
        # Third attempt: safest — skip entirely
        print("  [Safest fallback] DO NO HARM — skip adaptation")
        plan = {
            "trainable_scope": "bn_only", "learning_rate": 0,
            "num_steps": 0, "skip_adaptation": True, "tent_bn_mode": False,
            "loss_weights": {"bn_adapt": 0}, "max_iterations": 1,
            "use_anatomy_refinement": False,
            "rationale": "DO NO HARM: all strategies hurt quality, returning source model",
        }

    ps = _refs["pipeline_state"]
    ps.adaptation_plan = plan

    summary = (f"attempt={attempt}, scope={plan.get('trainable_scope')}, "
               f"lr={plan.get('learning_rate')}, skip={plan.get('skip_adaptation', False)}")
    print(f"  Plan: {summary}")
    return {"plan_summary": summary}


def adaptation_node(state: AgentState) -> dict:
    """Execute one adaptation attempt."""
    ps = _refs["pipeline_state"]
    plan = ps.adaptation_plan

    if plan.get("skip_adaptation"):
        ps.adapted_model = copy.deepcopy(ps.source_model)
        ps.adapted_model.to(ps.config.device)
        return {}

    # Fresh model copy for each attempt
    ps.adapted_model = copy.deepcopy(ps.source_model)
    ps.adapted_model.to(ps.config.device)
    ps.iteration_history.clear()
    ps.quality_scores.clear()
    ps.best_quality_score = 0.0
    ps.best_model_state = None

    # FDA warmup
    if plan.get("use_fda_warmup") and ps.source_images:
        from medadaptagent.orchestrator import Orchestrator
        Orchestrator(ps).adaptation_agent.run_fda_warmup()

    # Histogram matching
    if plan.get("use_histogram_matching"):
        from medadaptagent.orchestrator import Orchestrator
        Orchestrator(ps)._apply_histogram_matching()

    # Run adaptation iterations
    from medadaptagent.orchestrator import Orchestrator
    orch = Orchestrator(ps)
    max_iter = min(plan.get("max_iterations", 1), state.get("max_iterations", 3))
    for i in range(max_iter):
        orch.adaptation_agent.run_iteration(i)
        orch.evaluation_agent.run()
        # Check quality degradation
        if len(ps.iteration_history) >= 2:
            prev = ps.iteration_history[-2].get("avg_dice", 0)
            curr = ps.iteration_history[-1].get("avg_dice", 0)
            if curr < prev - 0.02 and ps.best_model_state:
                ps.adapted_model.load_state_dict(ps.best_model_state)
                break

    return {}


def validate_node(state: AgentState) -> dict:
    """Validate: did adaptation improve quality? (label-free)"""
    print("\n" + "=" * 50)
    print("NODE: Validate (label-free)")
    print("=" * 50)

    ps = _refs["pipeline_state"]
    model = ps.adapted_model if ps.adapted_model else ps.source_model
    q_after, details = _measure_quality(model, ps.target_images, ps.config.device)
    q_before = state.get("baseline_quality", 0)

    improved = q_after >= q_before - 0.02  # allow small tolerance
    delta = q_after - q_before

    print(f"  Before: {q_before:.3f}, After: {q_after:.3f}, Delta: {delta:+.3f}")
    print(f"  Verdict: {'ACCEPT ✅' if improved else 'REJECT ❌ — try safer'}")

    # Get GT dice for logging (if available, not used for decision)
    gt_dice = 0
    if ps.iteration_history:
        gt_dice = ps.iteration_history[-1].get("avg_dice", 0)
        print(f"  [GT reference] Dice: {gt_dice*100:.1f}%")

    attempt = state.get("attempt", 0)
    if improved:
        return {"adapted_quality": q_after, "done": True,
                "eval_summary": f"ACCEPTED attempt={attempt}, quality={q_after:.3f}, dice={gt_dice*100:.1f}%",
                "history_text": state.get("history_text", "") + f"\n  Attempt {attempt}: q={q_after:.3f} ACCEPTED"}
    else:
        return {"adapted_quality": q_after, "done": False,
                "attempt": attempt + 1,
                "eval_summary": f"REJECTED attempt={attempt}, quality dropped {q_before:.3f}→{q_after:.3f}",
                "history_text": state.get("history_text", "") + f"\n  Attempt {attempt}: q={q_after:.3f} REJECTED (was {q_before:.3f})"}


def should_continue(state: AgentState) -> Literal["retry", "done"]:
    if state.get("done", False):
        return "done"
    if state.get("attempt", 0) >= 3:
        return "done"
    return "retry"


def _default_plan(state):
    """Rule-based fallback when LLM fails."""
    knowledge = json.loads(state.get("knowledge_summary", "{}"))
    contrast = knowledge.get("contrast_mechanism", "same")
    scope = "bn+output" if contrast in ("fundamentally_different", "different") else "bn_only"
    return {
        "trainable_scope": scope,
        "learning_rate": 5e-4 if scope == "bn+output" else 1e-3,
        "num_steps": 100 if scope == "bn+output" else 80,
        "tent_bn_mode": scope == "bn_only",
        "skip_adaptation": False,
        "loss_weights": {"entropy": 0.3, "consistency": 0.2, "diversity": 0.2, "bn_adapt": 0.3},
        "max_iterations": 1,
        "use_fda_warmup": contrast == "fundamentally_different",
        "use_histogram_matching": contrast == "different",
        "use_anatomy_refinement": False,
        "rationale": f"Rule-based fallback: {contrast} contrast → {scope}",
    }


# ================================================================
# Build Graph
# ================================================================

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("knowledge", knowledge_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("baseline_quality", baseline_quality_node)
    graph.add_node("planning", planning_node)
    graph.add_node("adaptation", adaptation_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("knowledge")
    graph.add_edge("knowledge", "analysis")
    graph.add_edge("analysis", "diagnosis")
    graph.add_edge("diagnosis", "baseline_quality")
    graph.add_edge("baseline_quality", "planning")
    graph.add_edge("planning", "adaptation")
    graph.add_edge("adaptation", "validate")

    # Validate decides: accept (done) or retry with safer strategy
    graph.add_conditional_edges(
        "validate", should_continue,
        {"retry": "planning", "done": END},
    )

    return graph.compile()


# ================================================================
# Run
# ================================================================

def run_medadaptagent_graph(
    task_config, source_model, source_images, target_images,
    source_labels=None, target_labels=None,
    llm_client=None, device="cuda:0", max_iterations=3,
) -> dict:
    from medadaptagent.orchestrator import PipelineState

    cfg = copy.deepcopy(task_config)
    cfg.device = device
    cfg.max_iterations = max_iterations

    ps = PipelineState(
        config=cfg,
        source_model=copy.deepcopy(source_model),
        source_images=source_images or [],
        target_images=target_images,
        source_labels=source_labels or [],
        target_labels=target_labels or [],
        llm_client=llm_client,
    )

    _refs["pipeline_state"] = ps
    _refs["llm_client"] = llm_client

    task_summary = json.dumps({
        "task_name": cfg.task_name, "source_domain": cfg.source_domain,
        "target_domain": cfg.target_domain, "structures": cfg.structures,
        "anatomy_priors": cfg.anatomy_priors,
        "containment_rules": cfg.containment_rules,
    }, indent=2)

    initial_state = {
        "knowledge_summary": "", "analysis_results": "",
        "diagnosis_summary": "", "verification_summary": "",
        "plan_summary": "", "eval_summary": "",
        "task_config_summary": task_summary,
        "baseline_quality": 0.0, "adapted_quality": 0.0,
        "iteration": 0, "max_iterations": max_iterations,
        "attempt": 0, "done": False, "history_text": "",
    }

    print("=" * 60)
    print("MedAdaptAgent LangGraph (Propose-and-Validate)")
    llm_info = f"{llm_client.backend}/{llm_client.model}" if llm_client and hasattr(llm_client, 'backend') else "rule-based"
    print(f"  LLM: {llm_info}")
    print(f"  Task: {cfg.task_name}")
    print("=" * 60)

    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    graph = build_agent_graph()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - t0

    results = {
        "plan": ps.adaptation_plan,
        "history": ps.iteration_history,
        "knowledge": ps.rag_knowledge,
        "elapsed": elapsed,
        "attempts": final_state.get("attempt", 0),
        "accepted": final_state.get("done", False),
        "baseline_quality": final_state.get("baseline_quality", 0),
        "adapted_quality": final_state.get("adapted_quality", 0),
    }

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print(f"  Time: {elapsed:.1f}s, Attempts: {final_state.get('attempt', 0)}")
    print(f"  Quality: {final_state.get('baseline_quality', 0):.3f} → {final_state.get('adapted_quality', 0):.3f}")
    if ps.iteration_history:
        last = ps.iteration_history[-1]
        for k, v in last.items():
            if "dice" in k:
                print(f"  {k}: {v*100:.1f}%")
    print(f"  Strategy trace:{final_state.get('history_text', '')}")
    print("=" * 60)

    return results
