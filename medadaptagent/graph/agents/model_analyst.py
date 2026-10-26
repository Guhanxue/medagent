"""
Agent 2: Model Analyst

Understands HOW the model fails on target by combining:
  - Visual inspection of prediction overlays (multimodal LLM)
  - Statistical tools to confirm (confidence, entropy, hallucination detection)
  - Can request Agent 1 for clarification

The analyst thinks like a quality control specialist: LOOK at the
model's actual output, then MEASURE to confirm what you see.
"""

import json
import numpy as np
import torch
from typing import Dict, List

from medadaptagent.graph.vision_utils import (
    prepare_prediction_samples, build_multimodal_message,
)
from medadaptagent.graph.analysis_tools import ANALYSIS_TOOLS


def _assess_source_quality(model, device, source_images, source_labels,
                            source_fg_ratio, expected_area,
                            task=None,
                            source_confidence=None):
    """Check whether the model actually works on source before claiming to adapt.

    For classification tasks:
        Priority A: source accuracy via task.avg_metric on source_labels.
        Priority B: mean source confidence (high conf ≈ model knows its
                     training distribution).
        fg_ratio is NOT meaningful for cls — it's a 1-bit signal
        sensitive to sample size and class balance.

    For segmentation tasks (legacy):
        Priority 1: Dice on source_labels.
        Priority 2: source_fg_ratio vs Agent 1's expected_area ratio.

    Returns dict with score in [0, 1], method tag, and reason."""
    is_cls = task is not None and getattr(task, "name", "").startswith(
        "classification")

    # Cls path
    if is_cls:
        # Priority A: accuracy on source labels
        if source_images and source_labels and len(source_labels) > 0:
            try:
                num_classes = 2
                if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
                    num_classes = int(model.fc.out_features)
                acc = task.avg_metric(
                    model=model, images=source_images,
                    labels=source_labels, num_classes=num_classes,
                    device=device,
                )
                if acc is not None:
                    return {
                        "method": "cls_accuracy",
                        "score": round(float(acc), 3),
                        "n_samples": len(source_labels),
                        "reason": f"Source accuracy = {acc:.3f} across "
                                    f"{len(source_labels)} samples",
                    }
            except Exception:
                pass
        # Priority B: source confidence proxy
        if isinstance(source_confidence, (int, float)) and source_confidence > 0:
            return {
                "method": "cls_confidence_proxy",
                "score": round(float(source_confidence), 3),
                "reason": f"Mean source confidence = {source_confidence:.3f} "
                            f"(proxy; no source labels available)",
            }
        return {
            "method": "none",
            "score": None,
            "reason": "Cls: no source_labels or source_confidence — "
                       "can't estimate source quality",
        }
    # Seg path (legacy)
    # Priority 1: Dice from labels
    if source_images and source_labels and len(source_labels) > 0:
        try:
            from medadaptagent.evaluation.eval_metrics import dice_coefficient
            import torch as _torch
            model.eval()
            dices = []
            n = min(len(source_images), len(source_labels), 50)
            with _torch.no_grad():
                for i in range(n):
                    img = source_images[i]
                    lbl = source_labels[i]
                    if img.ndim == 2:
                        t = _torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
                        t = t.repeat(1, 3, 1, 1)
                    elif img.ndim == 3:
                        t = _torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                    else:
                        continue
                    t = t.to(device)
                    try:
                        logits = model(t)
                        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                    except Exception:
                        continue
                    # Binarize: any non-background class as foreground
                    pred_bin = (pred > 0)
                    gt_bin = (np.asarray(lbl) > 0)
                    # Handle shape mismatch
                    if pred_bin.shape != gt_bin.shape:
                        continue
                    d = dice_coefficient(pred_bin, gt_bin)
                    dices.append(d)
            if dices:
                mean_dice = float(np.mean(dices))
                return {
                    "method": "dice",
                    "score": round(mean_dice, 3),
                    "n_samples": len(dices),
                    "reason": f"Source Dice = {mean_dice:.3f} across {len(dices)} samples",
                }
        except Exception as e:
            # Fall through to heuristic
            pass

    # Priority 2: fg_ratio heuristic. Score = fg_coverage clamped to [0, 1]
    # (proxy for Dice when no labels are available — high coverage ≈ model
    # finds the structure, low coverage ≈ model misses it).
    if (isinstance(source_fg_ratio, (int, float)) and
        isinstance(expected_area, (list, tuple)) and len(expected_area) >= 2):
        expected_min = expected_area[0]
        if expected_min > 0:
            ratio = source_fg_ratio / expected_min
            return {
                "method": "fg_ratio_heuristic",
                "score": round(min(1.0, max(0.0, ratio)), 3),
                "reason": (f"Source fg = {source_fg_ratio:.1%} vs Agent 1 expected_min "
                           f"{expected_min:.0%} → coverage {ratio:.1%} "
                           f"(heuristic proxy, no source labels available)"),
            }

    return {
        "method": "none",
        "score": None,
        "reason": "No source_labels or expected_area_fraction available",
    }


def _compute_norm_divergence(model, target_images, device, max_samples=100,
                                task=None):
    """Measure norm-statistics divergence between source (running stats) and
    target (batch stats via forward pass). Works for any norm type that tracks
    running stats (BatchNorm always; InstanceNorm when track_running_stats=True).
    Returns `None` for LN/GN models (which have no running stats, so the
    severity pipeline falls back to fg_severity alone). The old key
    `bn_divergence` is preserved for backward compatibility with downstream
    severity formulas.

    For each eligible layer, computes a simplified KL-like divergence:
        d = mean((running_mean − batch_mean)^2 / running_var)
          + mean(log(running_var / batch_var) + batch_var/running_var − 1)
    Then averages across layers.
    """
    from medadaptagent.graph.norm_utils import iter_norm_modules, has_running_stats

    eligible = [(name, mod, nt) for name, mod, nt in iter_norm_modules(model)
                 if has_running_stats(mod)]
    if not eligible:
        # Pure LN / GN / or IN without tracking — no running-stat signal.
        return None

    # Save source running stats so the operation is non-destructive.
    saved = {name: {"running_mean": mod.running_mean.clone(),
                     "running_var": mod.running_var.clone()}
              for name, mod, _ in eligible}

    # Collect target batch stats. Switch to train mode on eligible layers and
    # reset running stats so a forward pass accumulates fresh numbers.
    prev_mode = model.training
    model.train()
    saved_momentum = {}
    for name, mod, _ in eligible:
        saved_momentum[name] = mod.momentum
        mod.reset_running_stats()
        mod.momentum = None  # cumulative average over passes

    imgs = target_images[:max_samples]
    with torch.no_grad():
        for img in imgs:
            try:
                if task is not None:
                    t = task.to_tensor(img, device)
                else:
                    if img.ndim == 2:
                        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
                        t = t.repeat(1, 3, 1, 1)
                    elif img.ndim == 3:
                        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                    else:
                        continue
                    t = t.to(device)
                model(t)
            except Exception:
                pass  # occasional shape issue on a single image is tolerable

    per_layer = {}
    divergences = []
    eps = 1e-6
    for name, mod, _ in eligible:
        src_mean = saved[name]["running_mean"]
        src_var = saved[name]["running_var"]
        tgt_mean = mod.running_mean
        tgt_var = mod.running_var
        mean_div = ((src_mean - tgt_mean) ** 2 / (src_var + eps)).mean().item()
        var_ratio = (tgt_var + eps) / (src_var + eps)
        var_div = (torch.log(1.0 / var_ratio) + var_ratio - 1).mean().item()
        layer_div = mean_div + var_div
        per_layer[name] = round(layer_div, 4)
        divergences.append(layer_div)

    # Restore source state.
    for name, mod, _ in eligible:
        mod.running_mean.copy_(saved[name]["running_mean"])
        mod.running_var.copy_(saved[name]["running_var"])
        mod.momentum = saved_momentum[name]
    if not prev_mode:
        model.eval()

    mean_div = float(np.mean(divergences))
    max_div = float(np.max(divergences))

    # Primary norm type represented in the eligible set.
    nt_counts = {}
    for _, _, nt in eligible:
        nt_counts[nt] = nt_counts.get(nt, 0) + 1
    dominant_nt = max(nt_counts.items(), key=lambda kv: kv[1])[0]

    return {
        "bn_divergence": round(mean_div, 4),        # legacy key
        "norm_divergence": round(mean_div, 4),
        "bn_divergence_max": round(max_div, 4),     # legacy key
        "norm_divergence_max": round(max_div, 4),
        "n_layers": len(eligible),
        "has_batchnorm": dominant_nt == "batchnorm",
        "norm_type": dominant_nt,
        "per_layer": per_layer,
    }


# Backward-compat alias so any external callers keep working.
_compute_bn_divergence = _compute_norm_divergence


MODEL_ANALYST_PROMPT = """You are a quality control specialist evaluating a segmentation model on a NEW dataset.

## Domain Expert's Findings (Agent 1)
Shift level: {shift_level} ({shift_severity})
Imaging: {modality_source} → {modality_target}
Key difference: {dominant_factor}
Calibration concern: {calibration_concern}
Per-channel shift breakdown:
{per_channel_breakdown}
Warnings: {agent1_warnings}

## What to Segment
Structure: {structure_name}
Expected location: {expected_location}
Expected size: {expected_area} of image
Expected shape: {expected_shape}

## Model Architecture
Architecture: {model_architecture}
Normalization layers: {norm_type}
Parameters: {total_params}
Input resolution: {input_size}

## Model Behavior Shift (source vs target)
Source fg_ratio: {source_fg_ratio} (model on source — baseline when model works)
Target fg_ratio: {measured_fg} (model on target — current performance)
fg_drop: {fg_drop}
Source confidence: {source_confidence} → Target confidence: {target_confidence}
Source entropy: {source_entropy} → Target entropy: {target_entropy}

## Source Quality Check
Source quality score: {source_quality_score} ({source_quality_reason})
This score is a continuous measure in [0, 1] of how well the source model
performs on its OWN training domain. It is the soft ceiling for adaptation —
target Dice cannot meaningfully exceed source Dice. Reflect the score (not a
bucket) in your recommendation_for_agent3.

## BatchNorm Divergence
{bn_divergence_section}

## Model Predictions on Target
{prediction_section}

## Statistical Measurements
{statistical_results}

## Your Assessment — DUAL EVIDENCE CHECKLIST

For EACH question, provide BOTH visual evidence (from overlays) AND statistical evidence (from measurements). Then state your conclusion.

For each check, OBSERVE what you see → COMPARE with what's expected → JUDGE.

### Check 1: Foreground Amount
OBSERVE: Look at the overlays. How much of the image has colored predictions?
COMPARE: Predicted fg = {measured_fg}. Expected for {structure_name}: {expected_area}.
JUDGE: Is this reasonable, too little (under-segmenting), too much (over-segmenting), or near-zero (complete failure)?

### Check 2: Location and Shape
OBSERVE: Where are the predictions? What shape are they?
COMPARE: {structure_name} should be at {expected_location}, shaped like {expected_shape}.
JUDGE: Right place and shape? Or wrong location / impossible shape / random blobs?

### Check 3: Hallucination vs Under-segmentation
OBSERVE: Are there predicted regions in areas where {structure_name} clearly isn't?
COMPARE: High confidence ({measured_fg} predicted) but wrong location = hallucination.
IMPORTANT: If the model predicts LESS than expected (fg < expected minimum), it is UNDER-SEGMENTING, not hallucinating. Under-segmentation = model finds too little. Hallucination = model predicts WRONG things confidently. These require DIFFERENT treatment.
JUDGE: None / moderate / severe hallucination?

### Check 4: Clinical Usability
OBSERVE: As a radiologist, would you accept this segmentation?
JUDGE one of these:
- **model_works_ok**: correct structure, reasonable size and shape, acceptable boundaries — usable as-is
- **boundary_error**: right structure and size, but edges imprecise — needs mild correction
- **under_segmentation**: finds the structure but too small/incomplete — needs more adaptation
- **hallucination**: predicts wrong things confidently — DANGEROUS, needs careful approach
- **complete_failure**: nothing useful — needs aggressive intervention

NOTE: "model_works_ok" means a radiologist would USE this. Not just "produces some output."

Now classify the failure:

- **complete_failure**: Model predicts all background or all wrong. Need aggressive intervention.
- **hallucination**: Model confidently predicts structures that aren't there. DANGEROUS for entropy-based methods.
- **boundary_error**: Model finds right structures but boundaries are imprecise. MILD — normalization reset often fixes this.
- **class_confusion**: Model segments the right region but assigns wrong class labels.
- **model_works_ok**: Predictions actually look reasonable despite the domain shift.

ARCHITECTURE-AWARE RULES for recommendation_for_agent3:
- If norm_type is "batchnorm" AND bn_divergence is high (>1.0): BN statistics reset is strongly indicated (large internal representation mismatch).
- If norm_type is "batchnorm" AND bn_divergence is low (<0.5): BN reset may help but the shift is mild — lighter methods (intensity normalization, TTA) may suffice.
- If norm_type is "instancenorm" or "layernorm": BN reset is NOT applicable — recommend feature alignment, histogram matching, or test-time augmentation instead.
- If fg_drop is large (>50%) but target confidence stays high: the model is CONFIDENTLY WRONG — hallucination risk. DO NOT use entropy-based methods.
- If fg_drop is large AND target entropy increased: the model KNOWS it is uncertain — entropy-based selection/Tent may help.
- If the calibration_concern from Agent 1 says "calibration shift": intensity rescaling/histogram matching is sufficient — do NOT default to heavier methods.
- Your recommendation MUST be grounded in the architecture and model behavior shift above. Do not guess.

Return JSON:
```json
{{
    "checks": {{
        "foreground": {{"observe": "<what you see>", "compare": "<vs expected>", "judge": "reasonable|under_seg|over_seg|near_zero"}},
        "location_shape": {{"observe": "<where/shape>", "compare": "<vs expected>", "judge": "correct|wrong_location|wrong_shape|none"}},
        "hallucination": {{"observe": "<any false predictions?>", "judge": "none|moderate|severe"}},
        "usability": {{"observe": "<overall impression>", "judge": "model_works_ok|boundary_error|under_segmentation|hallucination|complete_failure"}}
    }},
    "failure_type": "complete_failure | hallucination | under_segmentation | boundary_error | class_confusion | model_works_ok",
    "halluc_ratio": <float 0-1, how much of predictions are hallucinated>,
    "miss_ratio": <float 0-1, how much is missed>,
    "boundary_ratio": <float 0-1, how much is boundary error>,
    "recommendation_for_agent3": "<SPECIFIC: what Agent 3 needs to know, e.g., 'BN stats mismatch causes under-segmentation — BN reset should help' or 'DO NOT use entropy — hallucination'>",
    "confidence": "high|medium|low"
}}
```"""


MODEL_ANALYST_CLS_PROMPT = """You are a quality control specialist evaluating a CLASSIFICATION model on a NEW dataset.

This is a CLS task. The model outputs a class probability vector, NOT a spatial
mask. The vocabulary "foreground", "boundary", "hallucination" from
segmentation does NOT apply directly — use the cls-appropriate checks below.

## Domain Expert's Findings (Agent 1)
Shift level: {shift_level} ({shift_severity})
Imaging: {modality_source} → {modality_target}
Key difference: {dominant_factor}
Calibration concern: {calibration_concern}
Per-channel shift breakdown:
{per_channel_breakdown}
Warnings: {agent1_warnings}

## Task
Classes: {structure_name}
Expected positive rate (Agent 1 prior): {expected_area}

## Model Architecture
Architecture: {model_architecture}
Normalization layers: {norm_type}
Parameters: {total_params}
Input resolution: {input_size}

## Model Behavior Shift (source vs target)
Source positive prediction rate: {source_fg_ratio}
Target positive prediction rate: {measured_fg}
Rate drift: {fg_drop} (NOT "loss" — could be drift in either direction)
Source mean confidence: {source_confidence} → Target mean confidence: {target_confidence}
Source mean entropy: {source_entropy} → Target mean entropy: {target_entropy}

## Source Quality Check
Source quality score: {source_quality_score} ({source_quality_reason})
For cls this is either source accuracy on labels (preferred) or mean source
confidence as a proxy. Score >= 0.85 means source model is genuinely strong;
score < 0.5 means model is broken on its own training distribution.

## Statistical Measurements
{statistical_results}

## Cls-Specific Assessment Checks

For each check, OBSERVE → COMPARE → JUDGE.

### Check 1: Prediction Distribution
OBSERVE: What is the model's positive prediction rate on target?
COMPARE: Source rate {source_fg_ratio} → target rate {measured_fg}. Expected
prevalence (Agent 1 prior): {expected_area}.
JUDGE: matched|mild_drift|moderate_drift|mode_collapse_to_one_class

### Check 2: Confidence Calibration
OBSERVE: How confident is the model on target predictions?
COMPARE: Source confidence {source_confidence} → target {target_confidence}.
JUDGE: well_calibrated|over_confident|under_confident|confidence_collapse

### Check 3: Source Model Quality
OBSERVE: Did the source model actually work on its training distribution?
COMPARE: source_quality_score = {source_quality_score}.
JUDGE: source_strong (>=0.85)|source_moderate (0.5-0.85)|source_broken (<0.5)

### Check 4: Clinical Usability
OBSERVE: From the prediction overlays and the numbers above, would a clinician
trust this model's outputs on target data?
JUDGE one of:
- **model_strong**: source is strong AND predictions look calibrated on target — likely usable as-is
- **mild_drift**: small prediction-rate or confidence change, likely benign — light adaptation may help
- **calibration_drift**: predictions distributed reasonably but confidence is over/under-stated — needs temperature scaling or BN reset
- **over_confident_wrong**: high confidence + likely wrong predictions on target — DANGEROUS, do NOT use entropy minimization
- **mode_collapse**: predicts the same class for everything (typically negative class on imbalanced data) — needs aggressive intervention or labels
- **source_broken**: source model never worked, adaptation cannot fix — refer to supervised training

ARCHITECTURE-AWARE RULES for recommendation_for_agent3:
- BatchNorm + high bn_divergence: BN reset is strongly indicated.
- LayerNorm / no running stats: BN reset N/A — recommend feature alignment, finetune_head, or temperature scaling.
- Imbalanced data + small K: warn Agent 3 that K-label gate may be noisy.
- If source_quality < 0.5: NO TTA will fix this — recommend label collection.
- If `over_confident_wrong`: explicitly forbid entropy-minimization tools (tent, sar, eata) — confidence is misleading.

Return JSON:
```json
{{
    "checks": {{
        "prediction_distribution": {{"observe": "<rate/distribution>", "judge": "matched|mild_drift|moderate_drift|mode_collapse_to_one_class"}},
        "confidence_calibration": {{"observe": "<confidence levels>", "judge": "well_calibrated|over_confident|under_confident|confidence_collapse"}},
        "source_quality": {{"observe": "<source perf>", "judge": "source_strong|source_moderate|source_broken"}},
        "usability": {{"observe": "<overall>", "judge": "model_strong|mild_drift|calibration_drift|over_confident_wrong|mode_collapse|source_broken"}}
    }},
    "failure_type": "model_strong | mild_drift | calibration_drift | over_confident_wrong | mode_collapse | source_broken",
    "halluc_ratio": <float 0-1, over-confident-wrong rate>,
    "miss_ratio": <float 0-1, not really applicable to cls; use 0 unless mode collapse>,
    "boundary_ratio": 0.0,
    "recommendation_for_agent3": "<SPECIFIC: e.g. 'source strong, no adaptation needed' or 'mode collapse to neg class — finetune_head with K labels' or 'over-confident wrong — DO NOT use entropy methods, try crop_center + histogram_match'>",
    "confidence": "high|medium|low"
}}
```"""


def run_model_analyst(target_images: List[np.ndarray],
                      model, device: str,
                      llm_client,
                      domain_analysis: Dict,
                      source_images: List[np.ndarray] = None,
                      source_labels: List[np.ndarray] = None,
                      model_info: Dict = None,
                      task_config=None,
                      n_samples: int = 3,
                      task=None) -> Dict:
    """Run the Model Analyst agent.

    Looks at model predictions on target + runs statistical tools.

    source_images: when provided, used for source-side model baseline.
    source_labels: when provided (alongside source_images), used to compute
        source Dice — a ground-truth measure of whether the model works on
        its training domain. Surfaces as `source_quality_score` (continuous
        in [0, 1]); Agent 3 uses it as a soft ceiling estimate for what TTA
        can plausibly achieve on the target.
    model_info: dict from _get_model_info() with architecture, norm_type, etc.
    task_config: TaskConfig object; used for input_size.
    """
    print("\n" + "=" * 55)
    print("AGENT 2: Model Analyst (Visual + Statistical)")
    print("=" * 55)

    # Get prediction overlays. Pass `task` so non-2D tasks (cls_3d,
    # seg_3d) get rendered via task.render_prediction_panel instead of
    # the legacy 2D path.
    print(f"  Generating prediction overlays ({n_samples} samples)...")
    pred_pairs = prepare_prediction_samples(
        target_images, model, device, n=n_samples, task=task)

    # Run statistical tools (selected based on domain analysis)
    # NOTE: raw-image intensity_analysis is NOT run here — that's Agent 1's
    # job (per-channel stats). Agent 2 only runs MODEL-level tools.
    print("  Running statistical confirmation tools...")
    stat_results = {}

    # Confidence analysis on target (model-level). Task-aware: defers
    # tensor prep + forward to the adapter for 3D / cls tasks.
    if "confidence_analysis" in ANALYSIS_TOOLS:
        try:
            result = ANALYSIS_TOOLS["confidence_analysis"]["fn"](
                target_images, model, device, max_samples=200, task=task)
            stat_results["confidence_analysis"] = result
            print(f"    confidence_analysis (target): OK")
        except Exception as e:
            print(f"    confidence_analysis (target): failed ({e})")

    # Source-side model baseline: run the same confidence analysis on source
    # images so we can measure HOW MUCH the model's behavior changed, not
    # just what it does on target.
    if source_images and "confidence_analysis" in ANALYSIS_TOOLS:
        try:
            src_result = ANALYSIS_TOOLS["confidence_analysis"]["fn"](
                source_images, model, device, max_samples=200, task=task)
            stat_results["confidence_analysis_source"] = src_result
            src_fg = src_result.get("fg_ratio", 0)
            tgt_fg = stat_results.get("confidence_analysis", {}).get("fg_ratio", 0)
            fg_drop = 1 - (tgt_fg / src_fg) if src_fg > 0 else 1.0
            stat_results["model_behavior_shift"] = {
                "source_fg_ratio": src_fg,
                "target_fg_ratio": tgt_fg,
                "fg_drop": round(fg_drop, 4),
                "source_confidence": src_result.get("mean_confidence", 0),
                "target_confidence": stat_results.get("confidence_analysis", {}).get("mean_confidence", 0),
                "source_entropy": src_result.get("mean_entropy", 0),
                "target_entropy": stat_results.get("confidence_analysis", {}).get("mean_entropy", 0),
            }
            print(f"    confidence_analysis (source): OK  "
                  f"fg {src_fg:.1%} -> {tgt_fg:.1%} (drop {fg_drop:.0%})")
        except Exception as e:
            print(f"    confidence_analysis (source): failed ({e})")

    # ── Source quality check ──
    # Before we claim to adapt, verify the model actually works on source.
    # If it doesn't, domain adaptation is out of scope (training issue, not
    # adaptation issue). Use source_labels for real Dice when available,
    # fall back to src_fg vs Agent 1's expected_area_fraction ratio.
    fg_priors = domain_analysis.get("foreground_priors", {}) or {}
    expected_area = fg_priors.get("expected_area_fraction", None)
    # For cls tasks, supply source-side mean confidence as a proxy when
    # source_labels aren't available — fg_ratio is a 1-bit / sample-size-
    # sensitive signal that misleads on classification.
    _src_conf = (stat_results.get("confidence_analysis_source") or {}).get(
        "mean_confidence")
    source_quality = _assess_source_quality(
        model=model, device=device,
        source_images=source_images, source_labels=source_labels,
        source_fg_ratio=(stat_results.get("model_behavior_shift") or {}).get("source_fg_ratio"),
        expected_area=expected_area,
        task=task,
        source_confidence=_src_conf,
    )
    if source_quality:
        stat_results["source_quality"] = source_quality
        method = source_quality.get("method", "?")
        score = source_quality.get("score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        print(f"    source_quality: score={score_s} (method={method})")

    # Norm-statistics divergence: measures internal representation mismatch
    # between source (running stats) and target (batch stats). Works for any
    # norm type that actually tracks running stats (BN always, IN when
    # track_running_stats=True). For LN / GN / IN-without-tracking the signal
    # is unavailable and severity falls back to fg_severity alone — graceful,
    # no hardcoded fallback value.
    try:
        nd_result = _compute_norm_divergence(model, target_images, device,
                                              max_samples=100, task=task)
        if nd_result is not None:
            # Key kept as `bn_divergence` for backward compat with severity
            # formulas; also exposed as `norm_divergence` alongside.
            stat_results["bn_divergence"] = nd_result
            print(f"    norm_divergence ({nd_result['norm_type']}): "
                  f"{nd_result['norm_divergence']:.3f} "
                  f"(max={nd_result['norm_divergence_max']:.3f}, "
                  f"{nd_result['n_layers']} layers)")
        else:
            print(f"    norm_divergence: unavailable (model has no running-stat norm layers)")
    except Exception as e:
        print(f"    norm_divergence: failed ({e})")

    # Run hallucination detection if EITHER signal suggests risk:
    #   - contrast_mechanism_change in different/fundamentally_different (legacy)
    #   - calibration_concern indicates feature-level break (new from upgraded Agent 1)
    # Both are independent grounds to expect representation drift.
    contrast = domain_analysis.get("contrast_mechanism_change", "same")
    cal_concern = str(domain_analysis.get("calibration_concern", "") or "").lower()
    feature_shift_signal = "feature shift" in cal_concern
    if contrast in ("different", "fundamentally_different") or feature_shift_signal:
        try:
            result = ANALYSIS_TOOLS["hallucination_detection"]["fn"](
                target_images, model, device, max_samples=100, task=task)
            stat_results["hallucination_detection"] = result
            trigger = "contrast" if contrast in ("different", "fundamentally_different") else "calibration_concern"
            print(f"    hallucination_detection: OK (triggered by {trigger})")
        except Exception as e:
            print(f"    hallucination_detection: failed ({e})")

    # Extract foreground priors from domain analysis
    fg_priors = domain_analysis.get("foreground_priors", {})
    expected_location = fg_priors.get("expected_location", "unknown")
    expected_area = fg_priors.get("expected_area_fraction", [0.01, 0.30])
    expected_shape = fg_priors.get("expected_shape", "unknown")

    # Measured fg ratio from confidence analysis
    measured_fg = stat_results.get("confidence_analysis", {}).get("fg_ratio",
                  stat_results.get("hallucination_detection", {}).get("mean_fg_ratio", "unknown"))

    # Extract Agent 1's key findings for readable prompt
    shift_level = domain_analysis.get("shift_level", "?")
    shift_severity = domain_analysis.get("severity", "unknown")
    modality_source = domain_analysis.get("modality_source", domain_analysis.get("protocol_source", "unknown"))
    modality_target = domain_analysis.get("modality_target", domain_analysis.get("protocol_target", "unknown"))
    dominant_factor = domain_analysis.get("dominant_shift_factor", "unknown")
    warnings = domain_analysis.get("what_will_fail", [])
    warnings_text = "; ".join(str(w)[:60] for w in warnings[:3]) if warnings else "none"

    # New from upgraded Agent 1: structured per-channel shift list and
    # calibration_concern enum. Build a readable bullet block when present;
    # fall back to "(not provided by Agent 1)" so the prompt template stays
    # well-formed for older cached domain analyses.
    calibration_concern_text = str(
        domain_analysis.get("calibration_concern", "") or "(not specified)")
    dominant_shifts_list = domain_analysis.get("dominant_shifts") or []
    if isinstance(dominant_shifts_list, list) and dominant_shifts_list:
        bullets = []
        for d in dominant_shifts_list[:6]:
            if not isinstance(d, dict):
                continue
            ch = str(d.get("channel", "?"))
            st = str(d.get("shift_type", "?"))
            sv = str(d.get("severity", "?")).upper()
            desc = str(d.get("description", ""))[:120]
            bullets.append(f"  - [{sv}] {ch} ({st}): {desc}")
        per_channel_breakdown = "\n".join(bullets) if bullets else "  (none)"
    else:
        per_channel_breakdown = "  (not provided by Agent 1)"

    # Structure name from domain analysis
    structures = domain_analysis.get("structures", {})
    if isinstance(structures, dict):
        fg_names = [k for k in structures if k != "background"]
        structure_name = " / ".join(fg_names) if fg_names else "target structure"
    else:
        structure_name = "target structure"

    stat_text = json.dumps(stat_results, indent=2, default=str)[:800]

    # Extract model architecture info (from _get_model_info when provided)
    mi = model_info or {}
    model_architecture = mi.get("architecture", "unknown")
    norm_type = mi.get("norm_type", "unknown")
    total_params_val = mi.get("total_params", 0)
    total_params_str = f"{total_params_val / 1e6:.1f}M" if total_params_val else "unknown"
    input_size = getattr(task_config, "image_size", None) or mi.get("input_size", "unknown")

    # Extract model behavior shift for the prompt (source vs target model outputs)
    mbs = stat_results.get("model_behavior_shift", {})
    source_fg = mbs.get("source_fg_ratio")
    fg_drop = mbs.get("fg_drop")
    src_conf = mbs.get("source_confidence")
    tgt_conf = mbs.get("target_confidence")
    src_ent = mbs.get("source_entropy")
    tgt_ent = mbs.get("target_entropy")

    # BN divergence section for prompt
    bn_stats = stat_results.get("bn_divergence", {})
    bn_div_val = bn_stats.get("bn_divergence")
    if bn_div_val is not None:
        bn_div_section = (f"BN divergence (mean across {bn_stats.get('n_layers',0)} layers): "
                          f"{bn_div_val:.3f}  (max layer: {bn_stats.get('bn_divergence_max',0):.3f})\n"
                          f"Interpretation: >1.0 = large mismatch (BN reset strongly indicated), "
                          f"<0.5 = mild, <0.1 = negligible")
    else:
        bn_div_section = "(not computed — model does not use BatchNorm)"

    # Source quality for prompt: continuous score, no categorical flag
    sq = stat_results.get("source_quality", {}) or {}
    source_quality_score = sq.get("score")
    source_quality_score_str = (f"{source_quality_score:.3f}"
                                  if isinstance(source_quality_score, (int, float))
                                  else "not computed")
    source_quality_reason = sq.get("reason", "not computed")

    def _fmt(v, fmt=".3f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else "(not available)"

    # Pick task-appropriate prompt template. cls and cls_3d use the
    # cls-specific failure_type enum (model_strong / mild_drift /
    # calibration_drift / over_confident_wrong / mode_collapse /
    # source_broken). Seg uses the original enum with foreground/
    # boundary/hallucination vocabulary.
    _task_name = getattr(task, "name", "") if task is not None else ""
    _tmpl = (MODEL_ANALYST_CLS_PROMPT
              if _task_name.startswith("classification")
              else MODEL_ANALYST_PROMPT)
    prompt = _tmpl.format(
        shift_level=shift_level,
        shift_severity=shift_severity,
        modality_source=modality_source,
        modality_target=modality_target,
        dominant_factor=dominant_factor[:100],
        calibration_concern=calibration_concern_text,
        per_channel_breakdown=per_channel_breakdown,
        agent1_warnings=warnings_text,
        structure_name=structure_name,
        expected_location=expected_location,
        expected_area=f"{expected_area[0]*100:.0f}-{expected_area[1]*100:.0f}%" if isinstance(expected_area, (list, tuple)) else str(expected_area),
        expected_shape=expected_shape,
        model_architecture=model_architecture,
        norm_type=norm_type,
        total_params=total_params_str,
        input_size=input_size,
        source_fg_ratio=f"{source_fg:.1%}" if isinstance(source_fg, (int, float)) else "(not available)",
        fg_drop=f"{fg_drop:.0%}" if isinstance(fg_drop, (int, float)) else "(not available)",
        source_confidence=_fmt(src_conf),
        target_confidence=_fmt(tgt_conf),
        source_entropy=_fmt(src_ent),
        target_entropy=_fmt(tgt_ent),
        bn_divergence_section=bn_div_section,
        source_quality_score=source_quality_score_str,
        source_quality_reason=source_quality_reason,
        prediction_section="(see image pairs below)",
        statistical_results=stat_text,
        measured_fg=f"{measured_fg:.1%}" if isinstance(measured_fg, (int, float)) else str(measured_fg),
    )

    # Build image list: alternate original + overlay
    images = []
    labels = []
    for i, (img_b64, overlay_b64) in enumerate(pred_pairs):
        images.extend([img_b64, overlay_b64])
        labels.extend([f"Target Image {i+1}", f"Prediction Overlay {i+1}"])

    content = build_multimodal_message(prompt, images, labels)

    # Call LLM
    print(f"  Showing {len(pred_pairs)} prediction pairs to LLM...")
    try:
        response = llm_client.cached_complete(
            messages=[{"role": "user", "content": content}],
            max_tokens=800,
            temperature=0,
            seed=42,
        )
        text = response.choices[0].message.content or ""

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            assessment = json.loads(text[start:end])
        else:
            assessment = {"failure_type": "mixed", "halluc_ratio": 0.3,
                          "severity": 0.5, "raw_analysis": text}
    except Exception as e:
        print(f"  LLM failed ({e}), using statistical fallback")
        assessment = _statistical_fallback(stat_results, domain_analysis)

    # Merge statistical evidence
    assessment["statistics"] = stat_results

    # ── Programmatic severity ──
    # Two independent signals, take the max:
    #
    # 1) fg_severity: how much foreground the model lost on target vs source.
    #    severity = 1 - (target_fg / source_fg), clamped [0, 1].
    #    Falls back to Agent 1's expected_area if no source baseline.
    #
    # 2) bn_severity: how much the BN internal statistics diverged.
    #    Maps bn_divergence to [0, 1] via sigmoid-like scaling:
    #    bn_severity = bn_div / (bn_div + 1)  (0→0, 1→0.5, 5→0.83, 10→0.91)
    #
    # Combined: severity = max(fg_severity, bn_severity).
    # This ensures datasets with equal fg but large BN mismatch (e.g., prostate)
    # still get non-zero severity.

    source_fg = (stat_results.get("model_behavior_shift") or {}).get("source_fg_ratio")

    # fg_severity — use absolute deviation from baseline. Both
    # under-prediction (coverage < 1) AND over-prediction (coverage > 1)
    # signal a broken model. The old `1 - coverage` clamped to [0, 1]
    # silently gave 0 severity for over-predicting models, which
    # masked real failures (audit found coverage=161% → severity=0).
    fg_severity = None
    fg_coverage = None
    if isinstance(measured_fg, (int, float)):
        if isinstance(source_fg, (int, float)) and source_fg > 0.001:
            fg_coverage = measured_fg / source_fg
            fg_severity = round(min(1.0, abs(1.0 - fg_coverage)), 3)
        elif isinstance(expected_area, (list, tuple)):
            expected_mid = (expected_area[0] + expected_area[1]) / 2
            if expected_mid > 0:
                fg_coverage = measured_fg / expected_mid
                fg_severity = round(min(1.0, abs(1.0 - fg_coverage)), 3)

    # bn_severity
    bn_div = (stat_results.get("bn_divergence") or {}).get("bn_divergence")
    bn_severity = None
    if isinstance(bn_div, (int, float)):
        bn_severity = round(bn_div / (bn_div + 1.0), 3)

    # Combine
    severities = [s for s in [fg_severity, bn_severity] if s is not None]
    if severities:
        computed_severity = max(severities)
        llm_severity = assessment.get("severity")
        assessment["severity"] = computed_severity
        assessment["_severity_llm"] = llm_severity
        assessment["_severity_fg"] = fg_severity
        assessment["_severity_bn"] = bn_severity
        assessment["_severity_method"] = "fg" if fg_severity == computed_severity else "bn"
        if fg_coverage is not None:
            assessment["fg_coverage"] = round(fg_coverage, 4)

    # Surface source_quality at the top level so Agent 3 can read it directly.
    # Continuous score only — no categorical flag.
    sq_result = stat_results.get("source_quality") or {}
    if sq_result:
        assessment["source_quality_score"] = sq_result.get("score")
        assessment["source_quality_method"] = sq_result.get("method")
        assessment["source_quality_reason"] = sq_result.get("reason")

    print(f"  Failure: {assessment.get('failure_type', '?')}")
    print(f"  Hallucination: {assessment.get('halluc_ratio', 0):.0%}")
    fg_s = f"fg={assessment.get('_severity_fg', '?')}"
    bn_s = f"bn={assessment.get('_severity_bn', '?')}"
    print(f"  Severity: {assessment.get('severity', 0):.3f}  ({fg_s}, {bn_s})")
    if assessment.get("fg_coverage") is not None:
        print(f"  fg_coverage: {assessment['fg_coverage']:.1%} of baseline")
    if assessment.get("source_quality_score") is not None:
        print(f"  source_quality_score: {assessment['source_quality_score']:.3f}  "
              f"(method={assessment.get('source_quality_method')})")

    return assessment


def _statistical_fallback(stats: Dict, domain_analysis: Dict = None) -> Dict:
    """Fallback when LLM is unavailable.

    Uses the IDENTICAL severity formulas as the main-path programmatic
    override, so `severity` is on the same [0, 1] scale regardless of
    whether the LLM ran:
        fg_coverage = target_fg / source_fg            (primary; source_fg > 0.001)
                    = target_fg / expected_mid         (fallback)
        fg_severity = clamp01(1 - fg_coverage)
        bn_severity = bn_div / (bn_div + 1)
        severity    = max(fg_severity, bn_severity)    (if either exists)

    Categorical `failure_type` is indexed off the continuous severity via
    natural breakpoints (strict-majority / near-zero / half-lost / at-
    baseline), not hand-tuned thresholds on raw fg_ratio / confidence.
    """
    conf = stats.get("confidence_analysis", {})
    halluc = stats.get("hallucination_detection", {})
    mbs = stats.get("model_behavior_shift", {}) or {}
    bn_info = stats.get("bn_divergence", {}) or {}

    halluc_ratio = halluc.get("halluc_ratio", 0.0) or 0.0
    measured_fg = conf.get("fg_ratio", halluc.get("mean_fg_ratio", 0)) or 0.0
    source_fg = mbs.get("source_fg_ratio")

    expected_area = None
    if domain_analysis and "foreground_priors" in domain_analysis:
        expected_area = domain_analysis["foreground_priors"].get(
            "expected_area_fraction", [0.01, 0.5])

    # fg_severity — mirrors the main-path priority (source_fg first, then
    # expected_mid).
    fg_severity = None
    fg_coverage = None
    if isinstance(measured_fg, (int, float)):
        if isinstance(source_fg, (int, float)) and source_fg > 0.001:
            fg_coverage = measured_fg / source_fg
            fg_severity = round(min(1.0, abs(1.0 - fg_coverage)), 3)
        elif isinstance(expected_area, (list, tuple)) and len(expected_area) >= 2:
            expected_mid = (expected_area[0] + expected_area[1]) / 2
            if expected_mid > 0:
                fg_coverage = measured_fg / expected_mid
                fg_severity = round(min(1.0, abs(1.0 - fg_coverage)), 3)

    # bn_severity
    bn_div = bn_info.get("bn_divergence")
    bn_severity = None
    if isinstance(bn_div, (int, float)):
        bn_severity = round(bn_div / (bn_div + 1.0), 3)

    # Combine
    severities = [s for s in (fg_severity, bn_severity) if s is not None]
    severity = round(max(severities), 3) if severities else 0.0
    severity_method = None
    if severities:
        severity_method = ("fg" if fg_severity is not None
                            and fg_severity == max(severities) else "bn")

    # Categorical failure_type from severity + halluc (natural breakpoints):
    #   hallucination:       halluc > 0.5          (strict majority)
    #   complete_failure:    fg_severity > 0.95     (< 5% of expected coverage)
    #   under_segmentation:  fg_severity > 0.5      (< 50% of expected coverage)
    #   model_works_ok:      fg_severity < 0.1 AND bn_severity < 0.1
    #   boundary_error:      everything else
    fg_sev_val = fg_severity if fg_severity is not None else 0.0
    bn_sev_val = bn_severity if bn_severity is not None else 0.0
    if halluc_ratio > 0.5:
        failure_type = "hallucination"
        rec = "Majority of samples hallucinating — DO NOT use entropy loss."
    elif fg_sev_val > 0.95:
        failure_type = "complete_failure"
        rec = (f"Complete failure: target fg={measured_fg:.1%} is <5% of the "
                f"source/expected coverage. Aggressive adaptation required.")
    elif fg_sev_val > 0.5:
        failure_type = "under_segmentation"
        rec = (f"Under-segmenting: target fg={measured_fg:.1%}, fg_severity="
                f"{fg_sev_val:.2f}. BN reset + Tent should help.")
    elif fg_sev_val < 0.1 and bn_sev_val < 0.1:
        failure_type = "model_works_ok"
        rec = ("Predictions at baseline coverage and BN stats are aligned — "
                "risk of adaptation outweighs benefit.")
    else:
        failure_type = "boundary_error"
        rec = (f"Moderate errors (fg_sev={fg_sev_val:.2f}, "
                f"bn_sev={bn_sev_val:.2f}) — BN reset should help.")

    return {
        "failure_type": failure_type,
        "halluc_ratio": halluc_ratio,
        "severity": severity,
        "_severity_fg": fg_severity,
        "_severity_bn": bn_severity,
        "_severity_method": severity_method,
        "recommendation_for_agent3": rec,
    }
