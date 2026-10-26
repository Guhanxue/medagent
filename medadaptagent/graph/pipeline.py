"""
MedAdaptAgent v2: 6-Agent LangGraph Pipeline.

The pipeline orchestrates 6 specialized agents through a propose-and-validate loop.
Each reasoning agent uses multimodal LLM (sees images + reads statistics).
Each decision is backed by visual + numerical evidence.

Usage:
    from medadaptagent.graph.pipeline import run_pipeline

    result = run_pipeline(
        source_model=model,
        source_images=src_imgs,
        target_images=tgt_imgs,
        task_config=config,
        llm_client=llm,
        device="cuda:0",
    )
"""

import copy
import json
import time
import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from medadaptagent.graph.agents.domain_expert import run_domain_expert
from medadaptagent.graph.agents.model_analyst import run_model_analyst
from medadaptagent.graph.agents.strategy_architect import run_strategy_architect
from medadaptagent.graph.agents.execution_engineer import run_execution_engineer
from medadaptagent.graph.agents.delivery_agent import run_delivery_agent


MAX_STRATEGY_ATTEMPTS = 3


def run_pipeline(
    source_model: nn.Module,
    source_images: List[np.ndarray],
    target_images: List[np.ndarray],
    task_config,
    llm_client,
    source_labels: List[np.ndarray] = None,
    target_labels: List[np.ndarray] = None,
    device: str = "cuda:0",
    save_dir: str = None,
) -> Dict:
    """Run the full 6-agent MedAdaptAgent pipeline.

    Args:
        source_model: trained source segmentation model
        source_images: list of source images (H,W,C numpy)
        target_images: list of target images
        task_config: TaskConfig object
        llm_client: UnifiedLLMClient (GPT-4.1/5.2 via Azure)
        source_labels: optional source ground truth
        target_labels: optional target ground truth (for evaluation only)
        device: torch device
        save_dir: optional directory to save output

    Returns:
        Dict with adapted_model, reasoning_trace, scaling_recommendation
    """
    print("\n" + "=" * 60)
    print("MedAdaptAgent v2: 6-Agent Pipeline")
    llm_info = f"{llm_client.backend}/{llm_client.model}" if hasattr(llm_client, 'backend') else "unknown"
    print(f"  LLM: {llm_info}")
    print(f"  Task: {task_config.task_name}")
    print(f"  Data: {len(source_images)} source, {len(target_images)} target")
    print("=" * 60)

    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================
    # Agent 1: Domain Expert
    # ============================================================
    # Look at images + search knowledge base + analyze shift
    kb_knowledge = _get_kb_knowledge(task_config.task_name)

    domain_analysis = run_domain_expert(
        source_images=source_images,
        target_images=target_images,
        llm_client=llm_client,
        task_name=task_config.task_name,
        source_domain=task_config.source_domain,
        target_domain=task_config.target_domain,
        kb_knowledge=kb_knowledge,
        channel_names=getattr(task_config, "channel_names", None),
    )

    # Ensure domain_analysis is a dict (LLM sometimes returns string)
    if not isinstance(domain_analysis, dict):
        print(f"  Warning: domain_analysis is {type(domain_analysis)}, using fallback")
        from medadaptagent.graph.agents.domain_expert import _fallback_analysis
        domain_analysis = _fallback_analysis(task_config.source_domain, task_config.target_domain)

    # Auto-fill config fields from analysis
    if not task_config.source_domain:
        task_config.source_domain = domain_analysis.get("modality_source", "")
    if not task_config.target_domain:
        task_config.target_domain = domain_analysis.get("modality_target", "")

    # ============================================================
    # Agent 2: Model Analyst
    # ============================================================
    # Reason about model architecture, training conditions, statistics,
    # and prediction overlays — then synthesize a causal failure story.
    agent2_model_info = _get_model_info(source_model)
    agent2_model_info["input_size"] = getattr(task_config, "image_size", "unknown")
    model_assessment = run_model_analyst(
        target_images=target_images,
        model=source_model,
        device=device,
        llm_client=llm_client,
        domain_analysis=domain_analysis,
        source_images=source_images,
        model_info=agent2_model_info,
        task_config=task_config,
    )

    # ============================================================
    # Propose-and-Validate Loop (Agents 3-5)
    # ============================================================
    feedback = None
    adapted_model = None
    iteration_history = []
    final_inspection = None
    tried_protocols = set()  # Track to avoid repeating failed protocols

    for attempt in range(MAX_STRATEGY_ATTEMPTS):
        print(f"\n{'─' * 40}")
        print(f"Strategy Attempt {attempt + 1}/{MAX_STRATEGY_ATTEMPTS}")
        print(f"{'─' * 40}")

        # Agent 3: Strategy Architect — selects protocol with model awareness
        model_info = _get_model_info(source_model)
        plan = run_strategy_architect(
            domain_analysis=domain_analysis,
            model_assessment=model_assessment,
            llm_client=llm_client,
            feedback=feedback,
            has_source_images=bool(source_images),
            model_info=model_info,
        )

        # Agent 4: Execution Engineer (now includes the merged-in final
        # check previously handled by Agent 5).
        exec_result = run_execution_engineer(
            plan=plan,
            source_model=source_model,
            source_images=source_images,
            target_images=target_images,
            source_labels=source_labels or [],
            task_config=task_config,
            knowledge=domain_analysis,
            device=device,
            llm_client=llm_client,
            model_assessment=model_assessment,
        )

        adapted_model = exec_result["adapted_model"]
        iteration_history = exec_result["iteration_history"]

        # Final verdict + VLM second-opinion votes come from A4's execution_report
        er = exec_result.get("execution_report", {}) or {}
        inspection = {
            "verdict": er.get("final_verdict", "accept"),
            "reasoning": er.get("final_reason", ""),
            "per_sample_votes": er.get("final_votes", []),
            "quality_before": er.get("quality_before_final", {}),
            "quality_after": er.get("quality_after_final", {}),
        }
        final_inspection = inspection

        if inspection.get("verdict") == "accept":
            print(f"\n  ✅ Strategy ACCEPTED on attempt {attempt + 1}")
            break
        else:
            # Adaptation didn't help. Should we try again or give up?
            #
            # Only give up (keep original) on LAST attempt.
            # On earlier attempts: ALWAYS try a different approach.
            # The agent should exhaust all reasonable options before
            # accepting that the source model is the best we can do.

            if attempt < MAX_STRATEGY_ATTEMPTS - 1:
                # NOT the last attempt → try a different approach
                prev_protocol = plan.get("protocol", "unknown")
                q_before = inspection.get("quality_before", {})
                q_after = inspection.get("quality_after", {})

                # Determine if adaptation HURT: compare consistency + entropy
                # If adaptation decreased consistency or increased entropy, it hurt
                cons_before = q_before.get("consistency", 0)
                cons_after = q_after.get("consistency", 0)
                ent_before = q_before.get("entropy", 1)
                ent_after = q_after.get("entropy", 1)
                fg_before = q_before.get("fg_ratio", 0)
                fg_after = q_after.get("fg_ratio", 0)

                ent_ratio = ent_after / max(ent_before, 1e-6)
                adaptation_hurt = (cons_after < cons_before - 0.005) and (ent_ratio > 1.5)
                severity = domain_analysis.get("severity", "unknown")
                is_mild = severity in ("mild", "very_mild")

                if adaptation_hurt and is_mild:
                    # Mild shift + adaptation made things worse → KEEP SOURCE MODEL
                    # Don't ask LLM again — it might ignore the "skip" instruction.
                    # Force-keep the source model since it's already better.
                    ent_ratio_val = ent_after / max(ent_before, 1e-6)
                    print(f"\n  ⏹️ Adaptation HURT on mild shift "
                          f"(consistency {cons_before:.3f}→{cons_after:.3f}, "
                          f"entropy {ent_ratio_val:.1f}x) → keeping SOURCE MODEL")

                    # Use do_no_harm plan to execute skip through normal path
                    from medadaptagent.graph.empirical_protocols import get_protocol
                    plan = get_protocol("do_no_harm")
                    plan["reasoning"] = (
                        f"Adaptation with {prev_protocol} degraded quality on mild shift. "
                        f"Source model kept — it was already well-calibrated for this domain."
                    )
                    plan["rationale"] = plan["reasoning"]

                    # Run execution engineer with skip (handles eval properly)
                    exec_result = run_execution_engineer(
                        plan=plan,
                        source_model=source_model,
                        source_images=source_images,
                        target_images=target_images,
                        source_labels=source_labels or [],
                        target_labels=target_labels or [],
                        task_config=task_config,
                        knowledge=domain_analysis,
                        device=device,
                    )
                    adapted_model = exec_result["adapted_model"]
                    iteration_history = exec_result["iteration_history"]
                    final_inspection = inspection
                    break
                else:
                    # Significant shift or adaptation had no effect → try different approach
                    tried_protocols.add(prev_protocol)

                    # Mechanism-aware alternatives: when a protocol fails,
                    # suggest an alternative with SIMILAR or MORE aggressive
                    # mechanism — never strictly milder.
                    # Each entry: ranked alternatives (most similar mechanism first).
                    alternatives_map = {
                        "mild_same_modality": ["sar_robust", "moderate_cross_device", "contrast_change"],
                        "sar_robust": ["moderate_cross_device", "contrast_change"],
                        "moderate_cross_device": ["contrast_change", "cross_modality_fda", "sar_robust"],
                        "contrast_change": ["moderate_cross_device", "cross_modality_fda"],
                        "cross_modality_fda": ["contrast_change", "moderate_cross_device"],
                        "do_no_harm": ["mild_same_modality", "moderate_cross_device"],
                        "novel_unknown": ["mild_same_modality", "moderate_cross_device"],
                    }
                    candidates = alternatives_map.get(prev_protocol, ["moderate_cross_device"])
                    next_suggestion = None
                    for proto in candidates:
                        if proto not in tried_protocols:
                            next_suggestion = proto
                            break
                    if next_suggestion is None:
                        # All alternatives exhausted — keep best model we have
                        print(f"\n  ⏹️ All alternatives for '{prev_protocol}' exhausted → keeping best so far")
                        final_inspection = inspection
                        break

                    avoided = ", ".join(f"'{p}'" for p in tried_protocols)
                    feedback = (
                        f"Attempt {attempt+1} with protocol '{prev_protocol}' didn't improve quality. "
                        f"Inspector feedback: {inspection.get('reasoning', inspection.get('feedback_for_architect', 'no improvement'))}. "
                        f"Quality before: {q_before}, after: {q_after}. "
                        f"Already tried (DO NOT repeat): {avoided}. "
                        f"MUST pick a DIFFERENT approach with similar or more aggressive mechanism. "
                        f"DO NOT pick a strictly milder protocol than '{prev_protocol}'. "
                        f"Recommended: '{next_suggestion}'."
                    )
                    print(f"\n  🔄 Attempt {attempt+1} ({prev_protocol}) failed → suggesting {next_suggestion}")
            else:
                # LAST attempt — give up, keep best model we found
                # Use label-free quality score (not GT dice) to decide
                best_q = 0
                for hist_entry in iteration_history:
                    q = hist_entry.get("quality_score", 0)
                    if q > best_q:
                        best_q = q

                if best_q > 0 and adapted_model is not None:
                    # Some adaptation ran — keep the adapted model
                    # (best_model_state already selected by label-free quality)
                    print(f"\n  ⚠️ All {MAX_STRATEGY_ATTEMPTS} attempts done. Keeping adapted (q={best_q:.3f})")
                else:
                    # Nothing ran — keep source model
                    print(f"\n  ↩️ All {MAX_STRATEGY_ATTEMPTS} attempts failed → keeping original")
                    adapted_model = copy.deepcopy(source_model)
                    adapted_model.to(device)

                # Evaluate for delivery report
                try:
                    from medadaptagent.orchestrator import Orchestrator, PipelineState
                    ps_eval = PipelineState(
                        config=copy.deepcopy(task_config),
                        source_model=copy.deepcopy(source_model),
                        target_images=target_images,
                        target_labels=target_labels or [],
                    )
                    ps_eval.config.device = device
                    ps_eval.adapted_model = copy.deepcopy(adapted_model)
                    ps_eval.adapted_model.to(device)
                    # Pre-populate so eval can store GT dice
                    ps_eval.iteration_history.append({"method": "last_attempt_eval"})
                    Orchestrator(ps_eval).evaluation_agent.run()
                    iteration_history = ps_eval.iteration_history
                except Exception as e:
                    print(f"  Eval failed: {e}")
                    if not iteration_history:
                        iteration_history = [{"avg_dice": 0}]

    # ============================================================
    # Agent 6: Delivery
    # ============================================================
    model_info = _get_model_info(source_model)
    delivery = run_delivery_agent(
        adapted_model=adapted_model,
        domain_analysis=domain_analysis,
        model_assessment=model_assessment,
        plan=plan,
        inspection_result=final_inspection or {},
        iteration_history=iteration_history,
        task_config=task_config,
        model_info=model_info,
        n_target_images=len(target_images),
        llm_client=llm_client,
        save_dir=save_dir,
    )

    # Agent 7 (Few-Shot Specialist) is available but currently disabled.
    # Wire to CLI --n-shots flag when ready.

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    return {
        "adapted_model": adapted_model,
        "domain_analysis": domain_analysis,
        "model_assessment": model_assessment,
        "plan": plan,
        "inspection": final_inspection,
        "delivery": delivery,
        "iteration_history": iteration_history,
        "elapsed": elapsed,
    }


def _get_model_info(model: nn.Module) -> Dict:
    """Extract model architecture info for Agents 2 and 3.

    Norm-type detection and capability set come from the single source of
    truth in `medadaptagent.graph.norm_utils` — one dispatch handles BN / IN
    / LN / GN identically. Adding a new architecture does not touch this
    function; the norm_utils registry picks it up automatically.
    """
    from medadaptagent.graph.norm_utils import (
        detect_norm_type, count_norm_layers, model_capabilities,
    )
    total_params = sum(p.numel() for p in model.parameters())
    norm_type = detect_norm_type(model)
    capabilities = model_capabilities(model)

    # Architecture family detection (heuristic, not decision-gating — used for
    # display / prompt context only).
    module_names = [name for name, _ in model.named_modules()]
    names_str = " ".join(module_names).lower()
    class_name = type(model).__name__.lower()
    if "swin" in names_str:
        arch = "swin_unetr"
    elif ("inc" in names_str and "down" in names_str) or \
         ("enc" in names_str and "dec" in names_str) or \
         "unet" in class_name:
        arch = "unet"
    else:
        arch = "unknown"

    return {
        "architecture": arch,
        "total_params": total_params,
        "norm_type": norm_type,
        "has_batchnorm": norm_type == "batchnorm",  # legacy key for callers
        "capabilities": sorted(capabilities),
        "norm_layer_counts": count_norm_layers(model),
    }


def _get_kb_knowledge(task_name: str) -> Optional[Dict]:
    """Look up knowledge base for this task."""
    import os
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rag", "domain_knowledge.json"
    )
    if not os.path.exists(kb_path):
        return None
    try:
        import json
        with open(kb_path) as f:
            kb = json.load(f)
        for key, entry in kb.items():
            if key == "_meta":
                continue
            if key == task_name or task_name.startswith(key):
                return entry
        return None
    except Exception:
        return None
