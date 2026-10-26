"""
Experience prompts for LLM-driven agent nodes.

Each prompt encodes empirical lessons from 16+ iterations of debugging.
The LLM gets this experience as system context so it can make
calibrated decisions without hardcoded rules.
"""

# ================================================================
# Analysis Selection Prompt
# ================================================================

ANALYSIS_SELECTION_PROMPT = """You are a medical imaging analysis agent. Based on the domain knowledge, decide which statistical analyses to run.

## Available Analysis Tools
{tool_descriptions}

## Domain Knowledge (from previous step)
{knowledge_summary}

## Instructions
Select which tools to run. Not all are needed — choose based on the shift type:
- Cross-modality (CT→MRI): intensity_analysis + frequency_analysis + confidence_analysis + hallucination_detection
- Cross-sequence (T1→T2): intensity_analysis + hallucination_detection + confidence_analysis
- Cross-site/device (same modality): intensity_analysis + confidence_analysis
- If knowledge mentions "scale mismatch": add scale_analysis
- Always include confidence_analysis (it's cheap and informative)

Return a JSON list of tool names:
```json
["intensity_analysis", "confidence_analysis", ...]
```"""


# ================================================================
# Diagnosis Prompt (with empirical experience)
# ================================================================

DIAGNOSIS_PROMPT = """You are a medical image segmentation failure diagnosis agent. Analyze the model's behavior on target data and classify the failure mode.

## Analysis Results
{analysis_results}

## Domain Knowledge
{knowledge_summary}

## Empirical Experience (MUST consider)
These failure patterns were validated across 9 medical imaging datasets:

1. **Complete failure** (fg_ratio < 0.5%): Model produces all background. Happens in cross-modality where features are completely mismatched. Severity: HIGH.

2. **Hallucination** (high confidence + high fg + flip-consistent): Model confidently predicts WRONG structures. Most dangerous because entropy minimization will reinforce these errors. Common in contrast reversal (T1→T2) and cross-modality.

3. **Boundary errors** (moderate confidence, fg reasonable): Model finds the right structures but boundaries are imprecise. This is the MILDEST failure — BN reset often fixes it. Common in cross-site/device shifts.

4. **Class confusion** (wrong class labels, right locations): Model segments something but labels it wrong. Needs output head remapping.

KEY LESSON: Hallucination ratio is the MOST IMPORTANT signal for downstream decisions. If halluc > 50%, entropy minimization MUST be disabled.

## Instructions
Based on the analysis results, classify the failure:

Return JSON:
```json
{{
    "failure_type": "complete_failure | hallucination | boundary_error | class_confusion | mixed",
    "halluc_ratio": <float 0-1>,
    "miss_ratio": <float 0-1>,
    "boundary_ratio": <float 0-1>,
    "severity": <float 0-1>,
    "reasoning": "<explain what the analysis signals tell you about the failure mode>"
}}
```"""


# ================================================================
# Verification Prompt (with calibration experience)
# ================================================================

VERIFICATION_PROMPT = """You are a domain shift verification agent. Cross-check the knowledge predictions against statistical evidence and produce a calibrated severity assessment.

## Knowledge Predictions
{knowledge_summary}

## Statistical Evidence
{analysis_results}

## Failure Diagnosis
{diagnosis_summary}

## Calibration Experience (CRITICAL — follow these)
Verified severity must be CALIBRATED. Our empirical findings:

1. **Most cross-device/cross-site shifts are MILD (severity 0.2-0.4)**: Even if KL divergence is high, if model confidence is high and fg_ratio is reasonable, the shift is mild. BN reset handles it.

2. **Cross-modality is genuinely SEVERE (severity 0.7-0.9)**: CT→MRI involves fundamentally different physics. High severity is correct.

3. **Contrast reversal is MODERATE-SEVERE (severity 0.5-0.7)**: T1→T2 reverses tissue intensities but same modality. Hallucination is the main risk.

4. **When model already works well (confidence > 0.99, reasonable fg)**: Severity should be LOW regardless of what statistics say. The model is already calibrated for the target.

5. **classifier_severity = how well the model ALREADY performs on target**: If model predictions are reasonable (fg 1-30%), classifier_severity is LOW. This is the strongest "do no harm" signal.

## Instructions
Produce a verified severity assessment:

Return JSON:
```json
{{
    "severity_score": <float 0-1>,
    "severity_level": "minimal | mild | moderate | high",
    "scope_recommendation": "bn_only | bn+output | bn+decoder1",
    "mode_recommendation": "per_image_tent | batch_multi_loss",
    "should_skip": <true ONLY if model already at >85% Dice on target — very rare>,
    "reasoning": "<explain how you reconciled knowledge vs statistics vs diagnosis>"
}}
```"""


# ================================================================
# Planning Prompt (with adaptation experience)
# ================================================================

PLANNING_PROMPT = """You are a medical image segmentation adaptation planning agent. Design the adaptation strategy based on all previous analysis.

## Verified Assessment
{verification_summary}

## Knowledge
{knowledge_summary}

## Diagnosis
{diagnosis_summary}

## Analysis
{analysis_results}

## Task Configuration
{task_config}

## Adaptation Experience (MUST follow — these prevent catastrophic failures)

### Loss Selection Rules:
- **NEVER use entropy when halluc_ratio > 50%**: Entropy sharpens current predictions. If predictions are WRONG, this makes them MORE wrong. This caused BraTS T1→T2 to collapse from 24% to 3%.
- **Use diversity loss under hallucination**: Diversity (KL toward uniform) prevents class collapse without reinforcing specific wrong predictions.
- **Consistency (flip invariance)**: Safe to use always. Provides implicit supervision without assuming predictions are correct.

### Scope Selection Rules:
- **bn_only + Tent (per-image)**: For ANY same-modality shift (cross-site, cross-vendor, cross-device, cross-coil). LR=1e-3, steps=80, tent_bn_mode=true. This is the SAFEST and most effective option. Empirically gets 68% on prostate, 87% on RIGA+, 83% on M&Ms — these are strong results.
- **bn+output**: ONLY for cross-modality (CT→MRI) or contrast reversal (T1→T2) where features are fundamentally mismatched. LR=5e-4.
- **bn+decoder1**: ONLY for severe cross-modality with source data (FDA warmup). Risks overfitting.
- **CRITICAL LESSON**: On prostate cross-site, bn+output with lr=5e-4 COLLAPSED to 0% Dice. bn_only with Tent gave 68%. Same-modality shifts need CONSERVATIVE scope. The encoder features transfer well — only BN statistics need correction.

### DO NO HARM Rules:
- **skip_adaptation=true**: Returns source model UNCHANGED. Use when the label-free signals show the model ALREADY works well on target:
  - confidence_analysis shows mean_confidence > 0.99 AND mean_entropy < 0.01 → model predictions are sharp and consistent → skip.
  - Same modality + severity < 0.2 + high confidence → skip.
  - This is crucial: on M&Ms cardiac (Siemens→Philips), confidence=0.999, entropy=0.003. Tent hurt from 82.8% to 79%. Always skip for such mild shifts.
- **BN reset only** (skip=false, scope=bn_only, steps=80, tent_bn_mode=true): For mild-to-moderate same-modality shifts where confidence is high but not extreme (0.95-0.99). Recalibrates BN running stats.
- **SAFE MODE after FDA** (for cross-modality): After FDA warmup, use num_steps=0 with bn_adapt=1.0 (BN reset only, no gradient steps). The FDA already did the heavy lifting; gradient steps after FDA can HURT by overfitting.
- CRITICAL: On M&Ms Siemens→Philips, the model already segments at 82.8%. Tent reduces it to 79%. ALWAYS skip adaptation for very mild cardiac vendor shifts.

### Iteration Rules:
- max_iterations=1 when halluc > 50% (errors compound each iteration)
- max_iterations=2-3 for moderate shifts with declining entropy
- ALWAYS track quality between iterations — stop if it degrades

### FDA Rules:
- use_fda_warmup=true ONLY for cross-modality (fundamentally_different contrast)
- FDA bridges low-frequency appearance gap but cannot fix texture differences
- Requires source images to be available
- **After FDA warmup**: use SAFE MODE (num_steps=0, bn_adapt=1.0). Let FDA + BN reset do the work. Additional gradient steps often hurt.

### Histogram Matching Rules:
- use_histogram_matching=true for INTRA-modality contrast shifts (e.g., T1→T2 MRI) where tissue intensities are remapped but the modality is the same
- Do NOT use for cross-modality (CT→MRI) — the CDF mapping is semantically wrong
- Histogram matching + 1 iteration + no entropy is the winning formula for BraTS T1→T2 (gave 28.3%). Without histogram matching: only 15%.

## Instructions
Design the complete adaptation plan:

Return JSON:
```json
{{
    "trainable_scope": "bn_only | bn+output | bn+decoder1",
    "learning_rate": <float>,
    "num_steps": <int>,
    "loss_weights": {{
        "entropy": <float 0-1>,
        "consistency": <float 0-1>,
        "diversity": <float 0-1>,
        "bn_adapt": <float 0-1>
    }},
    "skip_adaptation": <true/false>,
    "use_fda_warmup": <true/false>,
    "use_histogram_matching": <true/false>,
    "max_iterations": <1-3>,
    "tent_bn_mode": <true if bn_only scope>,
    "use_anatomy_refinement": <true if containment rules exist>,
    "rationale": "<explain each decision and why>"
}}
```"""


# ================================================================
# Router Prompt
# ================================================================

ROUTER_PROMPT = """You are the adaptation quality router. Based on the evaluation results, decide the next action.

## Current State
- Iteration: {iteration} / {max_iterations}
- Plan max iterations: {plan_max_iterations}
- Latest evaluation: {eval_summary}
- History: {history}

## Rules
1. If quality improved and more iterations available → "continue"
2. If quality degraded from previous iteration → "stop" (revert to best model)
3. If max iterations reached → "stop"
4. If adaptation was skipped (DO NO HARM) → "stop"

Return JSON:
```json
{{
    "action": "continue | stop",
    "reasoning": "<why>"
}}
```"""
