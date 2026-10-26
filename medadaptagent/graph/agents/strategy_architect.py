"""
Agent 3: Strategy Architect

Designs adaptation strategy by reasoning about tool MECHANISMS
and adjusting from a reference protocol library.

The agent thinks like an engineer consulting a handbook:
  1. Read Agent 1+2 findings to understand the problem
  2. Look up similar cases in the reference library
  3. Select the closest reference, then ADJUST parameters
     based on the specific situation
  4. Output a complete plan with all parameters

The LLM has REAL influence — it can adjust lr, steps, scope,
loss weights, and preprocessing choices. Safety bounds prevent
catastrophic parameter choices.
"""

import json
from typing import Dict

from medadaptagent.graph.empirical_protocols import (
    PROTOCOLS, PROTOCOL_CAPABILITIES,
    get_protocol, get_fallback_protocol, get_available_protocols,
    scale_protocol_by_severity,
)


# Safety bounds — LLM adjustments are clipped to these ranges
SAFE_BOUNDS = {
    "learning_rate": (0, 0.01),
    "num_steps": (0, 300),
    "max_iterations": (1, 5),
    "fda_beta": (0.01, 0.15),
    "fda_finetune_steps": (50, 200),
    "fda_finetune_lr": (1e-5, 1e-3),
}

SAFE_LOSS_BOUNDS = {
    "entropy": (0.0, 1.0),
    "consistency": (0.0, 1.0),
    "diversity": (0.0, 1.0),
    "bn_adapt": (0.0, 1.0),
    "anatomy": (0.0, 1.0),
    "pseudo_label": (0.0, 1.0),
}

VALID_SCOPES = ["bn_only", "bn+output"]


STRATEGY_PROMPT = """You are an ML engineer deciding how to fix a segmentation model that doesn't work well on new data.

## The Problem

Agent 1 (Domain Expert) analyzed the domain shift:
  Shift: {shift_description}
  Key difference: {dominant_factor}
  Structured shift type (top driver): {structured_shift_type}
  Calibration concern: {calibration_concern}
  Per-channel shifts:
{per_channel_shifts}
  Target structure: {structure_name}

Agent 2 (Model Analyst) analyzed how the model fails:
  Failure mode: {failure_type}
  Hallucination risk: {halluc_level}
  Assessment: {agent2_recommendation}

## Your Toolbox — What Each Tool DOES to the Model

Think about each tool as a mechanism that changes something specific in the model:

**BN Reset** — Recalculates BatchNorm running mean/variance using target data.
  WHAT it fixes: intensity distribution mismatch between source and target.
  WHEN it helps: any shift that changes image brightness/contrast distribution.
  WHY: the model's BN layers expect source-like intensity stats. Target has different stats → BN outputs are wrong → predictions are wrong. Resetting fixes this.
  RISK: almost none. No gradient updates, just statistics recalculation.

**Tent (per-image entropy minimization)** — Adjusts BN affine parameters to make predictions sharper.
  WHAT it fixes: uncertain/blurry predictions, boundary imprecision.
  WHEN it helps: model roughly finds the right structure but isn't confident.
  WHY: entropy minimization pushes the model toward more decisive predictions.
  RISK: if model is WRONG (hallucinating), Tent makes it more confident in wrong predictions → catastrophic. Only safe when model predictions are structurally correct.

**SAR (Sharpness-Aware)** — Like Tent but seeks flat minima instead of sharp ones.
  WHAT it fixes: same as Tent (uncertain predictions) but more robustly.
  WHEN it helps: when Tent is too aggressive and causes instability.
  WHY: flat minima generalize better across domain shifts.
  RISK: lower than Tent. Slower but more stable.

**FDA (Fourier Domain Adaptation)** — Swaps low-frequency appearance between source→target.
  WHAT it fixes: completely different appearance (different imaging physics).
  WHEN it helps: cross-modality (CT→MRI) where everything looks different.
  WHY: transfers the "style" of target images to source domain, creating training data that looks like target but has source labels. Then supervised fine-tuning bridges the gap.
  RISK: requires source images and labels. Only helps for large appearance gaps.

**Histogram Matching** — Maps target intensity distribution to match source.
  WHAT it fixes: intensity remapping between same-modality different-contrast images.
  WHEN it helps: same modality but tissue brightness relationships changed (T1→T2, pre→post contrast, different MRI sequences).
  WHY: aligns the CDF so same tissues have similar intensity.
  RISK: WRONG for cross-modality (CT→MRI) where CDF mapping is meaningless.

**Color Normalization** — Reinhard-style per-channel mean/std matching from source to target.
  WHAT it fixes: color and brightness drift between imaging devices (different cameras, staining protocols).
  WHEN it helps: fundus photography, histopathology, dermatology — domains where color variation is the main shift.
  WHY: normalizes per-channel statistics so the model sees source-like color distributions.
  RISK: low. Only adjusts first-order statistics. Requires source images.

**Self-supervised losses** (consistency + diversity):
  Consistency: predictions should be same under image flipping → implicit supervision. Always safe.
  Diversity: prevents model from collapsing to predict one class for everything. Always safe.
  Entropy: sharpens predictions. DANGEROUS if model is hallucinating — zero this when hallucination risk is high.

**Anatomy containment loss** — Penalizes anatomically impossible predictions using containment rules (e.g., optic cup must be inside optic disc).
  WHAT it fixes: structurally implausible segmentations where child structures leak outside parent structures.
  WHEN it helps: multi-class tasks with known spatial relationships (eye: cup inside disc, abdomen: vessels inside organ).
  WHY: adds a soft anatomical constraint as a loss term during adaptation.
  RISK: very low. Only activates when containment_rules are defined for the task. Set anatomy loss weight > 0 to enable.

**Pseudo-labeling** — Self-training with filtered high-confidence predictions as supervision.
  WHAT it fixes: under-segmentation by providing direct supervision from the model's own best guesses.
  WHEN it helps: moderate shifts where model finds some correct structure but misses parts; combined with histogram matching to boost initial pseudo-label quality.
  WHY: converts confident correct predictions into training signal for uncertain regions.
  RISK: moderate. Bad pseudo-labels propagate errors. Only use when model partially works. Set pseudo_label loss weight > 0 to enable.

**Wider scope (bn+output)** — Train more parameters beyond just BN.
  WHAT it fixes: when BN alone isn't enough, the output head needs remapping.
  WHEN it helps: the feature-to-class mapping changed, not just the statistics.
  RISK: more parameters = more overfitting risk on unlabeled data.

## Reference Protocol Library

Pick a reference based on the SHIFT TYPE described by Agent 1. The numeric
parameters (lr, num_steps, max_iterations) are filled in by the system from
mechanism-class defaults and Agent 2's severity score AFTER you choose — do
not hand-tune them in your output unless you have a specific mechanistic
reason.

Only protocols compatible with THIS model's norm layers are listed — any
protocol that would require a capability the model doesn't have (e.g. BN
running stats on an InstanceNorm / LayerNorm network) has been removed from
the choice set. You cannot pick a reference not listed here.

{protocol_library}

## Your Task

1. **Reason** about what's wrong, why, and which tool mechanism fixes it
2. **Select** the closest reference protocol as your starting point
3. **Adjust** parameters if this case differs from the reference.

**CRITICAL RULE - protocol vs severity are ORTHOGONAL:**
- Pick `base_reference` purely from SHIFT TYPE (what Agent 1 says about modality and contrast):
   - Same modality + same contrast (cross-site, cross-vendor, cross-field) -> mild_same_modality
   - Same modality + different contrast (T1<->T2, pre<->post) -> contrast_change
   - Same modality + different camera/sensor optics -> moderate_cross_device
   - Cross-modality (CT<->MRI, MRI<->US) -> cross_modality_fda
   - Model already works -> do_no_harm
- DO NOT switch base_reference because severity is high. A severe mild_same_modality
  shift is still mild_same_modality, just with more iterations.
- Agent 2's severity score is automatically used downstream to scale `num_steps`
  and `max_iterations` after you return. You do NOT need to adjust those dials.
- YOU are responsible for `trainable_scope`. Pick it based on ALL evidence:
   - `bn_only`: mild shift, calibration-only issue, BN divergence low, model working
     partially correctly. Safe default for same-modality shifts.
   - `bn+output`: the output head needs remapping too — use when contrast changed,
     cross-modality, failure_type is complete_failure, or feature shift indicated.
   - `full`: almost never; only for catastrophic failure with large source+target
     corpus and very low overfitting risk.
- The ONLY adjustments you should make on top of the reference are:
   - High hallucination risk -> reduce/zero entropy, increase diversity
   - Contrast reversal (T1<->T2) -> use histogram matching, zero entropy
   - Cross-modality -> the cross_modality_fda reference already handles this
   - Anatomy rules available -> add anatomy loss weight
   - Scope needs adjustment vs reference -> set trainable_scope explicitly

**STRUCTURED SHIFT HINTS (from Agent 1's `dominant_shifts` + `calibration_concern`):**
Use these as a structured cross-check on your reference selection:
- `structured_shift_type` of `intensity` only -> mild_same_modality (BN reset handles it)
- `structured_shift_type` of `contrast` -> contrast_change (histogram matching helps)
- `structured_shift_type` of `resolution` or `fov` -> moderate_cross_device
- `structured_shift_type` of `acquisition` only -> mild_same_modality
- Agent 1 only flags channels with `severity=low` and calibration_concern says
  "calibration shift" -> stay with bn_only scope; recalibration is sufficient
- Calibration concern says "feature shift" -> the model's representations are wrong,
  not just rescaled; allow trainable_scope=bn+output even if shift_level is 1
- Calibration concern says "mixed" -> start with bn_only but consider bn+output
  if fg_drop > 50% or BN divergence is large (you pick, not a downstream rule)

These hints are SECONDARY to the modality/contrast routing above. If they conflict
with the modality routing (e.g., structured says "intensity" but Agent 1 also
identified cross-modality), the modality routing wins.

{previous_feedback}

## Model Architecture
{architecture_info}

## Source Quality (scope check from Agent 2)
{source_quality_section}

The score above is a continuous measure in [0, 1] of how well the source model
performs on its OWN training domain (Dice when source labels are available,
or fg-coverage proxy otherwise). Treat it as a soft ceiling on what TTA can
plausibly recover on the target side: target Dice cannot meaningfully exceed
source Dice.

If the score is very low (the source model is broken on its own domain),
adaptation is out of scope — pick `do_no_harm` and explain in
`source_quality_caveat`. Otherwise, scale your aggressiveness with the
margin between the score and 1.0: a high score means you have room to push,
a moderate score means stay conservative. Reflect this judgment in the
caveat in your own words — do NOT use bucketed labels like "good/marginal".

## Return

You only decide MECHANISM-level knobs. The numeric dials (`learning_rate`,
`num_steps`, `max_iterations`) are derived by the system from mechanism-class
defaults and Agent 2's severity score — do NOT emit them.

```json
{{
    "reasoning": "<step-by-step: what's wrong → why → which tool → what adjustments>",
    "base_reference": "<which reference protocol you started from>",
    "plan": {{
        "trainable_scope": "bn_only | bn+output",
        "use_fda_warmup": <bool>,
        "use_histogram_matching": <bool>,
        "use_color_normalization": <bool>,
        "tent_bn_mode": <bool>,
        "skip_adaptation": <bool>,
        "loss_weights": {{
            "entropy": <float 0-1>,
            "consistency": <float 0-1>,
            "diversity": <float 0-1>,
            "bn_adapt": <float 0-1>,
            "anatomy": <float 0-1, optional, enable anatomy containment loss>,
            "pseudo_label": <float 0-1, optional, enable pseudo-label self-training>
        }}
    }},
    "adjustment_rationale": "<why you changed from reference, or 'none — reference fits well'>",
    "source_quality_caveat": "<REQUIRED: reflect the source_quality score. If low, note the improvement ceiling. If very low, mark out of scope and pick do_no_harm.>"
}}
```
"""


def _clip_plan_to_safe_ranges(plan: Dict) -> Dict:
    """Clip LLM-generated plan parameters to safe ranges."""
    for key, (lo, hi) in SAFE_BOUNDS.items():
        if key in plan:
            try:
                val = float(plan[key])
                plan[key] = max(lo, min(hi, val))
            except (ValueError, TypeError):
                pass

    # Clip loss weights
    lw = plan.get("loss_weights", {})
    if isinstance(lw, dict):
        for key, (lo, hi) in SAFE_LOSS_BOUNDS.items():
            if key in lw:
                try:
                    val = float(lw[key])
                    lw[key] = max(lo, min(hi, val))
                except (ValueError, TypeError):
                    pass

    # Validate scope
    scope = plan.get("trainable_scope", "bn_only")
    if scope not in VALID_SCOPES:
        plan["trainable_scope"] = "bn_only"

    # Ensure int types
    for key in ["num_steps", "max_iterations"]:
        if key in plan:
            try:
                plan[key] = int(plan[key])
            except (ValueError, TypeError):
                pass

    # Ensure bool types
    for key in ["use_fda_warmup", "use_histogram_matching", "use_color_normalization",
                "tent_bn_mode", "skip_adaptation"]:
        if key in plan and not isinstance(plan[key], bool):
            plan[key] = bool(plan[key])

    return plan


def run_strategy_architect(domain_analysis: Dict,
                           model_assessment: Dict,
                           llm_client,
                           feedback: str = None,
                           has_source_images: bool = True,
                           model_info: Dict = None) -> Dict:
    """Design adaptation strategy: LLM reasons + adjusts from reference library."""
    print("\n" + "=" * 55)
    print("AGENT 3: Strategy Architect (Reasoning + Reference)")
    print("=" * 55)

    if model_info is None:
        model_info = {}

    da = domain_analysis or {}
    ma = model_assessment or {}

    # Capabilities filter — only offer the LLM protocols this model can run.
    # Derived from norm_utils.model_capabilities upstream; passed via
    # model_info["capabilities"]. Falls back to deriving from `has_batchnorm`
    # when `capabilities` isn't present (backward compat with old callers).
    capabilities = model_info.get("capabilities")
    if capabilities is None:
        capabilities = set()
        if model_info.get("has_batchnorm", True):
            capabilities = {"tent_affine", "bn_running_stats"}
        elif model_info.get("norm_type") in ("instancenorm", "layernorm", "groupnorm"):
            capabilities = {"tent_affine"}
    available_protocols = set(get_available_protocols(capabilities))
    print(f"  Available protocols (filtered by model capabilities "
          f"{sorted(capabilities) if capabilities else '—'}): "
          f"{sorted(available_protocols)}")

    # Extract readable fields
    shift_desc = f"Level {da.get('shift_level', '?')} ({da.get('severity', 'unknown')})"
    dominant = da.get("dominant_shift_factor", "unknown")[:120]
    structures = da.get("structures", {})
    if isinstance(structures, dict):
        fg_names = [k for k in structures if k != "background"]
        structure_name = " / ".join(fg_names) if fg_names else "target structure"
    else:
        structure_name = "target structure"

    # New from upgraded Agent 1: structured per-channel shift list and
    # calibration_concern enum. Extract the highest-severity entry as the
    # "top driver" and format the full list as a bullet block. Both have
    # safe fallbacks so older cached Agent 1 responses still produce a
    # well-formed prompt.
    calibration_concern_text = str(
        da.get("calibration_concern", "") or "(not specified)")
    dominant_shifts_list = da.get("dominant_shifts") or []
    structured_shift_type = "(not provided)"
    per_channel_shifts_text = "    (not provided by Agent 1)"
    if isinstance(dominant_shifts_list, list) and dominant_shifts_list:
        sev_rank = {"high": 3, "medium": 2, "low": 1}
        ranked = sorted(
            [d for d in dominant_shifts_list if isinstance(d, dict)],
            key=lambda d: sev_rank.get(str(d.get("severity", "")).lower(), 0),
            reverse=True,
        )
        if ranked:
            top = ranked[0]
            top_type = str(top.get("shift_type", "?"))
            top_ch = str(top.get("channel", "?"))
            top_sv = str(top.get("severity", "?")).upper()
            structured_shift_type = f"{top_type} (channel={top_ch}, severity={top_sv})"
        bullets = []
        for d in ranked[:6]:
            ch = str(d.get("channel", "?"))
            st = str(d.get("shift_type", "?"))
            sv = str(d.get("severity", "?")).upper()
            desc = str(d.get("description", ""))[:120]
            bullets.append(f"    - [{sv}] {ch} ({st}): {desc}")
        if bullets:
            per_channel_shifts_text = "\n".join(bullets)

    failure_type = ma.get("failure_type", "unknown")
    halluc_ratio = ma.get("halluc_ratio", 0)
    halluc_level = f"{halluc_ratio:.0%}" if isinstance(halluc_ratio, (int, float)) else str(halluc_ratio)
    agent2_rec = ma.get("recommendation_for_agent3", ma.get("recommendation", ""))[:150]

    # Foreground ratio context
    stats = ma.get("statistics", {})
    conf_stats = stats.get("confidence_analysis", {})
    measured_fg = conf_stats.get("fg_ratio", 0)
    fg_priors = da.get("foreground_priors", {})
    expected_area = fg_priors.get("expected_area_fraction", [0.01, 0.5])
    if isinstance(expected_area, (list, tuple)) and len(expected_area) >= 2:
        expected_min, expected_max = expected_area[0], expected_area[1]
    else:
        expected_min, expected_max = 0.01, 0.5

    # Model-grounded signals from Agent 2 (source vs target model behavior)
    mbs = stats.get("model_behavior_shift", {}) or {}
    src_fg = mbs.get("source_fg_ratio")
    tgt_fg = mbs.get("target_fg_ratio", measured_fg)
    fg_drop = mbs.get("fg_drop")
    bn_div = (stats.get("bn_divergence") or {}).get("bn_divergence")

    fg_context = ""
    if isinstance(src_fg, (int, float)) and isinstance(tgt_fg, (int, float)):
        fg_context += (f"\n  Source fg: {src_fg:.1%}, Target fg: {tgt_fg:.1%}"
                       f"{f', fg_drop: {fg_drop:.0%}' if isinstance(fg_drop,(int,float)) else ''}")
    elif measured_fg > 0 and expected_min > 0:
        fg_context += f"\n  Foreground: model predicts {measured_fg:.1%} (Agent 1 expected {expected_min:.0%}-{expected_max:.0%})."
    if isinstance(bn_div, (int, float)):
        fg_context += f"\n  BN divergence: {bn_div:.3f} (>1.0 = large representation shift)"

    has_bn = model_info.get("has_batchnorm", True)
    arch_text = (
        f"Architecture: {model_info.get('architecture', 'unknown')}, "
        f"Params: {model_info.get('total_params', 0)/1e6:.1f}M, "
        f"Norm: {model_info.get('norm_type', 'unknown')}, "
        f"Has BatchNorm: {'YES' if has_bn else 'NO (Tent/BN reset not viable)'}"
        f"{fg_context}"
    )

    # Source quality block — continuous score, no categorical bucket.
    sq_score = ma.get("source_quality_score")
    sq_method = ma.get("source_quality_method", "")
    sq_reason = ma.get("source_quality_reason", "")
    if isinstance(sq_score, (int, float)):
        sq_section = (f"Score: {sq_score:.3f}  (method={sq_method})\n"
                      f"Detail: {sq_reason}")
    else:
        sq_section = "Score: not computed (no source data or labels available)"

    feedback_text = ""
    if feedback:
        feedback_text = f"\n## PREVIOUS ATTEMPT FAILED\n{feedback}\nDesign a DIFFERENT plan."

    # Render the reference-protocol table with only the protocols the model
    # supports (capability-filtered). The rows are stable one-liners derived
    # from the registry so adding a new protocol = one row appears here
    # automatically; no per-architecture branching.
    _ROW_TMPL = {
        "mild_same_modality":    "| mild_same_modality | Same modality, minor appearance shift (cross-site / cross-vendor / cross-field) | bn_only | Tent (per-image entropy) | entropy=0.5, consistency=0.3, diversity=0.1 |",
        "moderate_cross_device": "| moderate_cross_device | Same modality, larger gap (different camera/sensor optics) | bn+output | Tent + mean teacher | entropy=0.3, consistency=0.2, diversity=0.2 |",
        "contrast_change":       "| contrast_change | Same modality, contrast mechanism changed (T1↔T2, pre↔post contrast) | bn+output | Histogram match + mean teacher (no entropy) | entropy=0, consistency=0.2, diversity=0.3 |",
        "cross_modality_fda":    "| cross_modality_fda | Different modality entirely (CT↔MRI, MRI↔US) | bn+output | FDA warmup + BN reset (requires BN running stats) | FDA handles it |",
        "sar_robust":            "| sar_robust | Same modality, Tent too aggressive or unstable | bn_only | SAR (sharpness-aware) | entropy=0.5, consistency=0.3 |",
        "do_no_harm":            "| do_no_harm | Model already works well — don't touch it | — | skip | none |",
        "novel_unknown":         "| novel_unknown | Shift type doesn't match any known protocol — conservative fallback | bn_only | Tent (safe defaults) | entropy=0.2, consistency=0.3, diversity=0.2 |",
    }
    header = ("| Reference | When to Use | Scope | Mechanism | Key Losses |\n"
              "|-----------|------------|-------|-----------|------------|")
    rows = [_ROW_TMPL[n] for n in PROTOCOLS.keys()
             if n in available_protocols and n in _ROW_TMPL]
    protocol_library = header + "\n" + "\n".join(rows)

    prompt = STRATEGY_PROMPT.format(
        shift_description=shift_desc,
        dominant_factor=dominant,
        structured_shift_type=structured_shift_type,
        calibration_concern=calibration_concern_text,
        per_channel_shifts=per_channel_shifts_text,
        structure_name=structure_name,
        failure_type=failure_type,
        halluc_level=halluc_level,
        agent2_recommendation=agent2_rec,
        previous_feedback=feedback_text,
        architecture_info=arch_text,
        source_quality_section=sq_section,
        protocol_library=protocol_library,
    )

    if not has_source_images:
        prompt += "\n\nNOTE: No source images available. FDA requires source images — do not use it."

    # LLM reasons and outputs full plan
    plan = None
    base_ref = None
    if llm_client:
        try:
            response = llm_client.chat(prompt, max_tokens=1000)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])

                reasoning = result.get("reasoning", "")
                base_ref = result.get("base_reference", "")
                adjustment = result.get("adjustment_rationale", "none")
                source_quality_caveat = result.get("source_quality_caveat", "")
                llm_plan = result.get("plan", {})

                if reasoning:
                    print(f"  Reasoning: {reasoning[:200]}")
                print(f"  Base reference: {base_ref}")
                if adjustment and adjustment != "none":
                    print(f"  Adjustments: {adjustment[:150]}")
                if source_quality_caveat:
                    print(f"  Source quality caveat: {source_quality_caveat[:150]}")

                # Numeric dials (lr / num_steps / max_iterations) are
                # derived from mechanism defaults + severity scaling, not
                # from the LLM. Strip them if the LLM emits them anyway.
                NUMERIC_DIALS = {"learning_rate", "num_steps", "max_iterations"}
                if isinstance(llm_plan, dict) and llm_plan:
                    for k in list(llm_plan.keys()):
                        if k in NUMERIC_DIALS:
                            del llm_plan[k]

                # Capability guard: if the LLM picked a protocol this model
                # can't run (e.g., cross_modality_fda on an IN/LN model),
                # override with the capability-respecting fallback.
                if base_ref and base_ref not in available_protocols:
                    print(f"  ⚠️ LLM picked '{base_ref}' but it requires "
                          f"unsupported capabilities; overriding with fallback")
                    base_ref = get_fallback_protocol(da, ma, capabilities=capabilities)

                if isinstance(llm_plan, dict) and llm_plan:
                    # LLM provided a plan — start from reference and apply adjustments
                    if base_ref and base_ref in PROTOCOLS:
                        plan = get_protocol(base_ref)
                    else:
                        base_proto = get_fallback_protocol(
                            da, ma, capabilities=capabilities)
                        plan = get_protocol(base_proto)
                    for key, val in llm_plan.items():
                        if key == "loss_weights" and isinstance(val, dict):
                            ref_lw = plan.get("loss_weights", {})
                            ref_lw.update(val)
                            plan["loss_weights"] = ref_lw
                        else:
                            plan[key] = val

                    # Guardrail: if LLM zeroed out all loss weights, restore
                    # the reference defaults. The LLM sometimes returns explicit
                    # zeros even when it says "no adjustments", which would make
                    # finetuning compute zero loss / zero gradients.
                    lw = plan.get("loss_weights", {})
                    if isinstance(lw, dict) and sum(abs(v) for v in lw.values() if isinstance(v, (int, float))) < 1e-6:
                        ref_proto = PROTOCOLS.get(base_ref, {})
                        ref_lw = ref_proto.get("params", {}).get("loss_weights", {})
                        if ref_lw:
                            plan["loss_weights"] = dict(ref_lw)
                            print(f"  ⚠️ LLM zeroed all loss weights — restored reference defaults from {base_ref}")

                    plan["reasoning"] = reasoning
                    plan["rationale"] = reasoning
                    plan["adjustment_rationale"] = adjustment
                    plan["source_quality_caveat"] = source_quality_caveat

                elif base_ref and base_ref in PROTOCOLS:
                    # LLM provided reference but no plan adjustments — use reference as-is
                    plan = get_protocol(base_ref)
                    plan["reasoning"] = reasoning
                    plan["rationale"] = reasoning
                    plan["source_quality_caveat"] = source_quality_caveat

        except Exception as e:
            print(f"  LLM failed: {e}")

    # Fallback: rule-based protocol when LLM unavailable or failed
    if not plan:
        proto_name = get_fallback_protocol(da, ma, capabilities=capabilities)
        plan = get_protocol(proto_name)
        plan["reasoning"] = f"Fallback: {proto_name}"
        plan["rationale"] = plan["reasoning"]
        print(f"  Fallback: {proto_name}")

    # Apply severity-based parameter scaling (Option C):
    # The base protocol determines WHICH tools to use (shift type),
    # Agent 2's severity determines HOW HARD to apply them.
    severity_score = ma.get("severity", None)
    if severity_score is None:
        # Derive from failure_type if not explicit
        failure = ma.get("failure_type", "")
        severity_score = {
            "model_works_ok": 0.0,
            "boundary_error": 0.3,
            "under_segmentation": 0.6,
            "hallucination": 0.7,
            "complete_failure": 1.0,
        }.get(failure, 0.5)

    try:
        sev = float(severity_score)
        plan = scale_protocol_by_severity(
            plan, severity=sev, fg_ratio=measured_fg, expected_min=expected_min
        )
        if plan.get("severity_scaled"):
            print(f"  Severity-scaled (sev={sev:.2f}): "
                  f"steps={plan.get('num_steps')}, iter={plan.get('max_iterations')}, "
                  f"scope={plan.get('trainable_scope')}")
    except (ValueError, TypeError):
        pass

    # Clip to safe ranges (transparent safety, not hidden override)
    plan = _clip_plan_to_safe_ranges(plan)

    # Ensure loss_weights is a proper dict
    if isinstance(plan.get("loss_weights"), str):
        plan["loss_weights"] = {"entropy": 0.3, "consistency": 0.2,
                                 "diversity": 0.2, "bn_adapt": 0.3}
    if not isinstance(plan.get("loss_weights"), dict):
        plan["loss_weights"] = {"entropy": 0.3, "consistency": 0.2,
                                 "diversity": 0.2, "bn_adapt": 0.3}

    # Safety: if a strict majority of samples are hallucinating and the LLM
    # didn't zero entropy, warn + cap. Threshold = 0.5 (strict majority), not
    # a tuned constant.
    if isinstance(halluc_ratio, (int, float)) and halluc_ratio > 0.5:
        w = plan.get("loss_weights", {})
        if isinstance(w, dict) and w.get("entropy", 0) > 0.2:
            print(f"  ⚠️ Majority hallucinating ({halluc_ratio:.0%}) but entropy={w['entropy']:.1f}. "
                  f"Capping entropy to 0.1, raising diversity floor to 0.3.")
            w["entropy"] = min(w["entropy"], 0.1)
            w["diversity"] = max(w.get("diversity", 0), 0.3)

    proto = plan.get("protocol", base_ref or "custom")
    print(f"\n  Final: protocol={proto}, scope={plan.get('trainable_scope')}, "
          f"lr={plan.get('learning_rate')}, steps={plan.get('num_steps')}, "
          f"iterations={plan.get('max_iterations')}")
    lw = plan.get("loss_weights", {})
    print(f"  Losses: entropy={lw.get('entropy', '?')}, cons={lw.get('consistency', '?')}, "
          f"div={lw.get('diversity', '?')}, bn={lw.get('bn_adapt', '?')}")

    return plan
