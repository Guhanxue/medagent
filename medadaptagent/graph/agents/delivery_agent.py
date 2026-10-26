"""
Agent 6: Delivery Agent (LLM-powered)

Analyzes the adaptation results and provides:
  1. Adapted model checkpoint
  2. Full reasoning trace from all agents
  3. LLM-reasoned scaling recommendation (not a static table)
  4. Few-shot execution if user provides annotations
  5. Deployment guidance
"""

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn


DELIVERY_PROMPT = """You are a deployment advisor for a medical image segmentation model that was just adapted to a new domain.

## Adaptation Summary
- Task: {task_name}
- Source: {source_domain} → Target: {target_domain}
- Shift type: {shift_type}, severity: {severity}
- Protocol used: {protocol}
- Zero-shot adapted Dice: {zero_shot_dice}%

## Model Information
- Architecture: {architecture}
- Parameters: {total_params}M
- Norm type: {norm_type}

## Target Dataset
- Number of images: {n_target}
- Estimated volumes: ~{n_volumes}
- Average slices per volume: ~{slices_per_vol}

## What the Adaptation Did
{adaptation_summary}

## Quality Assessment
{quality_assessment}

## Your Analysis

Based on ALL the above, provide:

1. **Performance assessment**: Is the current zero-shot result clinically usable? What's the gap?

2. **Scaling prediction**: How would performance improve with labeled target data?
   Consider:
   - Model size ({total_params}M params): larger models need more data to fine-tune safely
   - Shift severity ({severity}): severe shifts benefit more from annotations
   - Current performance ({zero_shot_dice}%): how much room to improve?
   - Data available ({n_target} images, ~{n_volumes} volumes): how many can realistically be annotated?

3. **Annotation recommendation**: How many volumes should the user annotate?
   - Cost-benefit: each volume takes ~15-30 min for a radiologist
   - Diminishing returns: going from N=5→10 helps more than N=10→20

4. **Per-class analysis** (if multi-class segmentation):
   - Which classes are strongest/weakest?
   - Which would benefit most from targeted annotation?
   - Are there class-specific failure modes?

5. **Deployment advice**: Any caveats for clinical use?

Return JSON:
```json
{{
    "performance_assessment": "<is this clinically usable? what's missing?>",
    "scaling_prediction": {{
        "n_0": {zero_shot_dice},
        "n_2": <estimated with 2 annotated volumes>,
        "n_5": <estimated with 5>,
        "n_10": <estimated with 10>,
        "n_20": <estimated with 20>,
        "reasoning": "<how you estimated each N, citing model size and shift type>"
    }},
    "recommended_annotations": {{
        "n_volumes": <recommended number>,
        "estimated_time_hours": <radiologist time>,
        "expected_performance": <Dice % after annotation>,
        "rationale": "<why this many>"
    }},
    "deployment_caveats": ["<caveat 1>", "<caveat 2>"],
    "overall_summary": "<2-3 sentence summary for the user>"
}}
```"""


def run_delivery_agent(adapted_model: nn.Module,
                       domain_analysis: Dict,
                       model_assessment: Dict,
                       plan: Dict,
                       inspection_result: Dict,
                       iteration_history: list,
                       task_config=None,
                       model_info: Dict = None,
                       n_target_images: int = 0,
                       llm_client=None,
                       save_dir: str = None) -> Dict:
    """Package and deliver with LLM-powered analysis."""
    print("\n" + "=" * 55)
    print("AGENT 6: Delivery (LLM Analysis)")
    print("=" * 55)

    # Extract final dice
    final_dice = 0
    if iteration_history:
        final_dice = iteration_history[-1].get("avg_dice", 0)

    # Model info
    if model_info is None:
        model_info = {}
    total_params = model_info.get("total_params", sum(p.numel() for p in adapted_model.parameters()))
    arch = model_info.get("architecture", "unknown")
    norm = model_info.get("norm_type", "unknown")

    # Estimate volumes
    slices_per_vol = 20  # default estimate
    n_volumes = max(1, n_target_images // slices_per_vol)

    # Config info
    task_name = ""
    source_domain = ""
    target_domain = ""
    if task_config:
        task_name = getattr(task_config, "task_name", "")
        source_domain = getattr(task_config, "source_domain", "")
        target_domain = getattr(task_config, "target_domain", "")

    # Build adaptation summary
    adapt_parts = []
    if plan.get("use_fda_warmup"):
        adapt_parts.append("FDA warmup (bridged appearance gap)")
    if plan.get("use_histogram_matching"):
        adapt_parts.append("Histogram matching (corrected intensity mapping)")
    if plan.get("skip_adaptation"):
        adapt_parts.append("DO NO HARM (model preserved unchanged)")
    elif plan.get("tent_bn_mode"):
        adapt_parts.append(f"Tent per-image BN adaptation ({plan.get('num_steps')} steps)")
    elif plan.get("num_steps", 0) > 0:
        adapt_parts.append(f"Batch adaptation (scope={plan.get('trainable_scope')}, "
                          f"{plan.get('num_steps')} steps)")
    else:
        adapt_parts.append("BN statistics reset (SAFE MODE)")
    adaptation_summary = "; ".join(adapt_parts) if adapt_parts else "No adaptation applied"

    # Quality assessment
    quality_parts = []
    if inspection_result:
        quality_parts.append(f"Verdict: {inspection_result.get('verdict', '?')}")
        if inspection_result.get("visual_evidence"):
            quality_parts.append(f"Visual: {inspection_result['visual_evidence'][:100]}")
        if inspection_result.get("quality_before") and inspection_result.get("quality_after"):
            quality_parts.append(f"Before: {inspection_result['quality_before']}")
            quality_parts.append(f"After: {inspection_result['quality_after']}")
    quality_assessment = "\n".join(quality_parts) if quality_parts else "No assessment available"

    # LLM analysis
    scaling_rec = None
    if llm_client:
        try:
            prompt = DELIVERY_PROMPT.format(
                task_name=task_name,
                source_domain=source_domain,
                target_domain=target_domain,
                shift_type=domain_analysis.get("severity", "unknown"),
                severity=domain_analysis.get("severity", "unknown"),
                protocol=plan.get("protocol", "unknown"),
                zero_shot_dice=round(final_dice * 100, 1),
                architecture=arch,
                total_params=round(total_params / 1e6, 1),
                norm_type=norm,
                n_target=n_target_images,
                n_volumes=n_volumes,
                slices_per_vol=slices_per_vol,
                adaptation_summary=adaptation_summary,
                quality_assessment=quality_assessment,
            )

            response = llm_client.chat(prompt, max_tokens=800)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                scaling_rec = json.loads(response[start:end])
                print(f"  LLM analysis complete")
        except Exception as e:
            print(f"  LLM analysis failed: {e}")

    if not scaling_rec:
        scaling_rec = _default_recommendation(domain_analysis, final_dice,
                                               total_params, n_target_images)

    # Build reasoning trace
    reasoning_trace = {
        "1_domain_expert": {
            "severity": domain_analysis.get("severity"),
            "contrast": domain_analysis.get("contrast_mechanism_change"),
            "dominant_factor": domain_analysis.get("dominant_shift_factor"),
            "foreground_priors": domain_analysis.get("foreground_priors"),
            "evidence_sources": domain_analysis.get("evidence_sources"),
        },
        "2_model_analyst": {
            "failure_type": model_assessment.get("failure_type"),
            "hallucination": model_assessment.get("halluc_ratio"),
            "evidence_agreement": model_assessment.get("evidence_agreement"),
        },
        "3_strategy_architect": {
            "protocol": plan.get("protocol"),
            "scope": plan.get("trainable_scope"),
            "rationale": plan.get("rationale"),
        },
        "4_execution": {
            "iterations": len(iteration_history),
            "skipped": plan.get("skip_adaptation", False),
            "adaptation_summary": adaptation_summary,
        },
        "5_quality_inspection": {
            "verdict": inspection_result.get("verdict") if inspection_result else None,
        },
        "6_delivery": {
            "final_dice": round(final_dice * 100, 1),
            "scaling_recommendation": scaling_rec,
        },
    }

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(adapted_model.state_dict(), os.path.join(save_dir, "adapted_model.pt"))
        with open(os.path.join(save_dir, "adaptation_report.json"), "w") as f:
            json.dump(reasoning_trace, f, indent=2, default=str)
        print(f"  Saved to {save_dir}/")

    # Print summary
    print(f"\n  {'='*45}")
    print(f"  DELIVERY SUMMARY")
    print(f"  {'='*45}")
    print(f"  Adapted Dice: {final_dice*100:.1f}%")
    print(f"  Protocol: {plan.get('protocol', '?')}")

    if scaling_rec.get("overall_summary"):
        print(f"\n  {scaling_rec['overall_summary']}")

    sp = scaling_rec.get("scaling_prediction", {})
    if sp:
        print(f"\n  Scaling Prediction:")
        for k in ["n_0", "n_2", "n_5", "n_10", "n_20"]:
            v = sp.get(k)
            if v is not None:
                print(f"    {k}: ~{v}%")
        if sp.get("reasoning"):
            print(f"    Reasoning: {sp['reasoning'][:120]}")

    ra = scaling_rec.get("recommended_annotations", {})
    if ra:
        print(f"\n  Recommendation: annotate {ra.get('n_volumes', '?')} volumes")
        print(f"    Time: ~{ra.get('estimated_time_hours', '?')} hours")
        print(f"    Expected: ~{ra.get('expected_performance', '?')}%")
        if ra.get("rationale"):
            print(f"    Reason: {ra['rationale'][:100]}")

    return {
        "reasoning_trace": reasoning_trace,
        "scaling_recommendation": scaling_rec,
        "final_dice": final_dice,
    }


def _default_recommendation(domain_analysis, zero_shot_dice, total_params, n_target):
    """Fallback when LLM unavailable."""
    severity = domain_analysis.get("severity", "moderate")
    zs = round(zero_shot_dice * 100, 1)
    is_large_model = total_params > 50e6

    if severity == "mild" or zs > 75:
        n_rec = 2 if zs < 85 else 0
    elif severity == "severe":
        n_rec = 10
    else:
        n_rec = 5

    return {
        "scaling_prediction": {
            "n_0": zs, "n_2": min(zs + 10, 95),
            "n_5": min(zs + 20, 95), "n_10": min(zs + 30, 95),
        },
        "recommended_annotations": {
            "n_volumes": n_rec,
            "estimated_time_hours": n_rec * 0.5,
            "expected_performance": min(zs + n_rec * 3, 95),
            "rationale": f"{severity} shift, {zs}% zero-shot",
        },
        "overall_summary": f"Zero-shot: {zs}%. Recommend {n_rec} annotated volumes for improvement.",
    }
