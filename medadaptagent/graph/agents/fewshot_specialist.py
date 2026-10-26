"""
Agent 7: Few-Shot Specialist

When the user provides annotated target volumes, this agent:
  1. Analyzes: how many labels? what quality? which volumes?
  2. Selects strategy per N (different N → different approach)
  3. Executes few-shot fine-tuning
  4. Reports scaling curve

Strategy varies by N:
  N=1:  Too noisy for training → output-layer calibration only
  N=2:  Careful training → decoder+output, low LR, heavy source mixing
  N=5:  Standard training → decoder+output or all, moderate LR
  N=10: Full training → all params, standard LR
  N=20: Aggressive → all params, more epochs

The specialist also considers:
  - Model architecture (large model needs more data per param)
  - Shift type (cross-modality benefits more from FDA augmentation)
  - Zero-shot quality (better starting point → less annotation needed)
"""

import json
import copy
import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn as nn


FEWSHOT_PROMPT = """You are a few-shot fine-tuning specialist. The user has provided {n_shots} annotated target volumes (~{n_slices} 2D slices).

## Context
- Zero-shot adapted Dice: {zero_shot_dice}%
- Shift type: {shift_type}
- Model: {architecture}, {total_params}M params, {norm_type}
- Domain gap: {severity}

## Available Strategies (choose ONE per N)

**Strategy: calibrate_output**
  What: Fine-tune only the output layer (~130 params)
  When: N=1 (too few for deeper training)
  Safe, but limited improvement

**Strategy: train_decoder_conservative**
  What: decoder+output scope, low LR (5e-5), heavy source mixing (0.8-0.9)
  When: N=2-4 (enough for careful training but high overfitting risk)
  Source mixing regularizes against overfitting

**Strategy: train_decoder_standard**
  What: decoder+output scope, standard LR (1e-4), moderate source mixing (0.3-0.5)
  When: N=5-9 (good balance of data and capacity)

**Strategy: train_all**
  What: all params, LR=5e-4, light source mixing (0.1-0.3)
  When: N≥10 (enough data for full model)

**Strategy: train_all_aggressive**
  What: all params, higher LR, more epochs, minimal source mixing
  When: N≥20 (abundant data)

## Adjustments to Consider
- Model has {total_params}M params. At N={n_shots} ({n_slices} slices), params_per_sample = {params_per_sample:.0f}. If >500K, restrict scope.
- Shift is {severity}. Cross-modality → use FDA augmentation with source mixing. Same-modality → standard augmentation.
- If halluc_ratio was high in zero-shot: use source mixing to regularize, avoid Mean Teacher.

## Select and Customize

Return JSON:
```json
{{
    "strategy": "<strategy name>",
    "scope": "output_only | decoder+output | all",
    "learning_rate": <float>,
    "epochs": <int>,
    "source_mix_ratio": <float 0-1>,
    "use_fda_augment": <true for cross-modality>,
    "use_mean_teacher": <true if halluc_ratio < 0.3>,
    "reasoning": "<why this strategy for N={n_shots} with this model/shift>"
}}
```"""


def run_fewshot_specialist(adapted_model: nn.Module,
                           target_images: List[np.ndarray],
                           target_labels: List[np.ndarray],
                           source_images: List[np.ndarray],
                           source_labels: List[np.ndarray],
                           domain_analysis: Dict,
                           model_assessment: Dict,
                           task_config,
                           llm_client,
                           model_info: Dict = None,
                           n_shots: int = 5,
                           device: str = "cuda:0") -> Dict:
    """Run few-shot fine-tuning with LLM-selected strategy.

    Args:
        adapted_model: the zero-shot adapted model (starting point)
        target_images/labels: target data (labels for first n_shots volumes)
        source_images/labels: source data (for mixing)
        domain_analysis: from Agent 1
        model_assessment: from Agent 2
        task_config: TaskConfig
        llm_client: for strategy reasoning
        n_shots: number of annotated target volumes
        device: torch device

    Returns:
        Dict with fine-tuned model, scaling results, strategy trace
    """
    print(f"\n" + "=" * 55)
    print(f"AGENT 7: Few-Shot Specialist (N={n_shots})")
    print("=" * 55)

    if model_info is None:
        model_info = {}
    total_params = model_info.get("total_params",
                    sum(p.numel() for p in adapted_model.parameters()))
    arch = model_info.get("architecture", "unknown")
    norm = model_info.get("norm_type", "unknown")

    # Estimate slices per volume
    avg_slices = 20  # default
    n_slices = n_shots * avg_slices
    params_per_sample = total_params / max(n_slices, 1)

    # Get zero-shot dice
    zero_shot_dice = 0
    if model_assessment.get("statistics", {}).get("confidence_analysis"):
        # Estimate from fg_ratio as proxy
        zero_shot_dice = model_assessment.get("statistics", {}).get(
            "confidence_analysis", {}).get("fg_ratio", 0) * 100

    severity = domain_analysis.get("severity", "moderate")
    shift_type = domain_analysis.get("contrast_mechanism_change", "same")

    # LLM selects strategy
    strategy = None
    if llm_client:
        prompt = FEWSHOT_PROMPT.format(
            n_shots=n_shots, n_slices=n_slices,
            zero_shot_dice=round(zero_shot_dice, 1),
            shift_type=shift_type, severity=severity,
            architecture=arch, total_params=round(total_params/1e6, 1),
            norm_type=norm, params_per_sample=params_per_sample,
        )
        try:
            response = llm_client.chat(prompt, max_tokens=400)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                strategy = json.loads(response[start:end])
                print(f"  LLM strategy: {strategy.get('strategy', '?')}")
                print(f"  Scope: {strategy.get('scope')}, LR: {strategy.get('learning_rate')}")
                if strategy.get("reasoning"):
                    print(f"  Reasoning: {strategy['reasoning'][:120]}")
        except Exception as e:
            print(f"  LLM failed: {e}")

    if not strategy:
        strategy = _default_fewshot_strategy(n_shots, total_params, severity)
        print(f"  Fallback: {strategy.get('strategy', '?')}")

    # Execute fine-tuning
    print(f"\n  Executing: {strategy.get('strategy')} on N={n_shots}...")
    from medadaptagent.adaptation.few_shot import (
        FewShotAdapter, group_by_volume, evaluate_model,
    )

    # Build analysis dict for FewShotAdapter
    analysis = {
        "contrast_mechanism_change": shift_type,
        "verified_severity": 0.5 if severity == "moderate" else (0.8 if severity == "severe" else 0.2),
        "halluc_ratio": model_assessment.get("halluc_ratio", 0),
        "used_fda": shift_type == "fundamentally_different",
        "avg_slices_per_vol": avg_slices,
        "num_classes": len(getattr(task_config, "structures", {0: 0})),
        "has_gpu": True,
    }

    adapter = FewShotAdapter(
        adapted_model=copy.deepcopy(adapted_model),
        analysis=analysis,
        device=device,
        num_classes=analysis["num_classes"],
        class_names=list(getattr(task_config, "fg_classes", {"fg": 1}).keys()),
        source_images=source_images,
        source_labels=source_labels,
        model_info=model_info,
    )

    # Run scaling law
    try:
        vol_ids = None  # auto-detect from filenames
        scaling = adapter.run_scaling_law(
            target_images=target_images,
            target_labels=target_labels,
            volume_ids=vol_ids,
            shot_counts=[n_shots],
            n_seeds=2,
        )
        print(f"  Result: {scaling}")
    except Exception as e:
        print(f"  Fine-tuning failed: {e}")
        scaling = {}

    return {
        "strategy": strategy,
        "scaling_results": scaling,
        "n_shots": n_shots,
    }


def _default_fewshot_strategy(n_shots, total_params, severity):
    """Default strategy when LLM unavailable."""
    if n_shots <= 1:
        return {"strategy": "calibrate_output", "scope": "output_only",
                "learning_rate": 1e-3, "epochs": 20, "source_mix_ratio": 0}
    elif n_shots <= 4:
        return {"strategy": "train_decoder_conservative", "scope": "decoder+output",
                "learning_rate": 5e-5, "epochs": 60, "source_mix_ratio": 0.9}
    elif n_shots <= 9:
        return {"strategy": "train_decoder_standard", "scope": "decoder+output",
                "learning_rate": 1e-4, "epochs": 50, "source_mix_ratio": 0.3}
    elif n_shots <= 19:
        return {"strategy": "train_all", "scope": "all",
                "learning_rate": 5e-4, "epochs": 40, "source_mix_ratio": 0.1}
    else:
        return {"strategy": "train_all_aggressive", "scope": "all",
                "learning_rate": 5e-4, "epochs": 30, "source_mix_ratio": 0}
