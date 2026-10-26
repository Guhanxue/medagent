"""
Empirical Adaptation Protocols.

A small library of TTA mechanism-bundles (which losses, which scope, FDA on/off,
mean-teacher on/off, SAR on/off). Agent 3 picks one based on shift type; the
numeric dials (learning_rate, num_steps, max_iterations) are NOT tuned per
protocol — they come from `MECHANISM_DEFAULTS` (literature defaults per
algorithm) and are scaled by Agent 2's severity score in
`scale_protocol_by_severity`. This keeps any-dataset-specific tuning out of
the registry.
"""

from typing import Dict


# ================================================================
# Mechanism-class defaults (literature, not dataset-tuned)
# ================================================================
# Each TTA mechanism has a baseline learning-rate from its original paper:
#   - Tent (Wang et al.) recommends lr ≈ 1e-3 on BN affine params
#   - SAR (Niu et al.) uses lr ≈ 5e-3 with sharpness perturbation
#   - FDA fine-tuning (Yang & Soatto) uses lr ≈ 5e-4 with mixed loss
# `num_steps_base` is a uniform default; the actual step count is set
# downstream as base * (1 + 2 * severity).
MECHANISM_DEFAULTS = {
    "tent":   {"learning_rate": 1e-3, "num_steps_base": 80},
    "sar":    {"learning_rate": 5e-3, "num_steps_base": 80},
    "fda_ft": {"learning_rate": 5e-4, "num_steps_base": 80},
    "noop":   {"learning_rate": 0.0,  "num_steps_base": 0},
}


# ================================================================
# Capability model
# ================================================================
# Each protocol declares which model capabilities it needs. Capabilities are
# derived from the model's norm layers in `norm_utils.model_capabilities`.
#   "tent_affine"       — model has norm layers with learnable weight/bias
#                          (any of BN/IN/LN/GN). Needed by Tent/SAR protocols.
#   "bn_running_stats"  — model has norm layers that track running_mean/var
#                          (BN always, IN when track_running_stats=True).
#                          Needed only by protocols that do a BN-reset step.
# When a protocol's required_capabilities are not satisfied by the model,
# `get_available_protocols` drops it from Agent 3's choice set entirely.
PROTOCOL_CAPABILITIES = {
    "cross_modality_fda":    frozenset({"tent_affine", "bn_running_stats"}),
    "contrast_change":       frozenset({"tent_affine"}),
    "mild_same_modality":    frozenset({"tent_affine"}),
    "moderate_cross_device": frozenset({"tent_affine"}),
    "sar_robust":            frozenset({"tent_affine"}),
    "do_no_harm":            frozenset(),
    "novel_unknown":         frozenset({"tent_affine"}),
}


# ================================================================
# Protocol Definitions
# ================================================================

# Each protocol specifies: (a) which mechanism-class it belongs to — that sets
# lr + num_steps_base from MECHANISM_DEFAULTS, (b) which mechanism bundle to
# enable (FDA? histogram matching? mean teacher? SAR? BN-only scope?), and
# (c) a loss-weight pattern tied to the diagnosed failure mode. No per-dataset
# tuned numbers remain in the params blocks — numerics come from mechanism
# defaults + Agent 2 severity scaling.
PROTOCOLS = {
    "cross_modality_fda": {
        "description": "Cross-modality with fundamentally different physics (CT↔MRI, MRI↔US)",
        "when": "Agent 1 says contrast_mechanism=fundamentally_different AND source images available",
        "mechanism_class": "noop",  # FDA fine-tune does the work; main gradient loop is a no-op
        "params": {
            "trainable_scope": "bn+output",
            "max_iterations": 1,
            "use_fda_warmup": True,
            "fda_beta": 0.03,
            "fda_finetune_steps": 120,
            "fda_finetune_lr": 5e-4,
            "fda_scope": "bn+output",
            "use_fda_mixed_training": True,
            "fda_mix_weight": 0.2,
            "tent_bn_mode": False,
            "skip_adaptation": False,
            "use_histogram_matching": False,
            "use_anatomy_refinement": False,
            "loss_weights": {
                "entropy": 0.2, "consistency": 0.2,
                "diversity": 0.3, "bn_adapt": 0.3,
            },
        },
    },

    "contrast_change": {
        "description": "Same modality but contrast mechanism changed — intensity relationships between tissues are different. Includes: T1↔T2↔FLAIR MRI, non-contrast↔contrast-enhanced CT/MRI, different echo times, different weightings.",
        "when": "Agent 1 says contrast_mechanism=different AND same base modality (both MRI or both CT)",
        "mechanism_class": "fda_ft",  # FDA-style learning rate for histogram-matched fine-tune
        "params": {
            "trainable_scope": "bn+output",
            "max_iterations": 1,  # errors compound under hallucination
            "use_fda_warmup": False,
            "use_histogram_matching": True,
            "use_mean_teacher": True,  # EMA teacher for stable pseudo-labels
            "tent_bn_mode": False,
            "skip_adaptation": False,
            "use_anatomy_refinement": False,
            "loss_weights": {
                "entropy": 0.0,  # NO entropy — model is confidently wrong
                "consistency": 0.2,
                "diversity": 0.3,
                "bn_adapt": 0.4,
                "pseudo_label": 0.3,
            },
        },
    },

    "mild_same_modality": {
        "description": "Same modality, mild shift (cross-site, cross-vendor, cross-field)",
        "when": "Agent 1 says same/similar contrast AND Agent 2 says boundary_error or model_works_ok",
        "mechanism_class": "tent",
        "params": {
            "trainable_scope": "bn_only",
            "max_iterations": 1,
            "use_fda_warmup": False,
            "use_histogram_matching": False,
            "tent_bn_mode": True,
            "skip_adaptation": False,
            "use_anatomy_refinement": False,
            # Published Tent (Wang et al. ICLR 2021) uses pure entropy
            # minimization on BN affine params. No consistency / diversity /
            # bn_adapt admixture. Using only entropy matches the published
            # formulation and avoids diluting the signal on mild shifts
            # where Tent is demonstrably effective.
            "loss_weights": {
                "entropy": 1.0, "consistency": 0.0,
                "diversity": 0.0, "bn_adapt": 0.0,
            },
        },
    },

    "moderate_cross_device": {
        "description": "Same modality but larger appearance gap (different camera/sensor optics). Uses batch multi-loss with bn+output scope and multiple iterations to progressively improve.",
        "when": "Agent 1 says same contrast but moderate standardization gap, or mild_same_modality was rejected",
        "mechanism_class": "tent",  # Tent default lr; severity scales steps, iterations
        "params": {
            "trainable_scope": "bn+output",
            "max_iterations": 3,
            "use_fda_warmup": False,
            "use_histogram_matching": False,
            "use_mean_teacher": True,
            "tent_bn_mode": False,
            "skip_adaptation": False,
            "use_anatomy_refinement": False,
            "loss_weights": {
                "entropy": 0.3, "consistency": 0.2,
                "diversity": 0.2, "bn_adapt": 0.3,
                "anatomy": 0.2,
            },
        },
    },

    "do_no_harm": {
        "description": "Model already works well — any adaptation risks hurting",
        "when": "Agent 2 says model_works_ok with high confidence (low halluc, low entropy)",
        "mechanism_class": "noop",
        "params": {
            "trainable_scope": "bn_only",
            "max_iterations": 1,
            "use_fda_warmup": False,
            "use_histogram_matching": False,
            "tent_bn_mode": False,
            "skip_adaptation": True,
            "use_anatomy_refinement": False,
            "loss_weights": {
                "entropy": 0, "consistency": 0,
                "diversity": 0, "bn_adapt": 0,
            },
        },
    },

    "sar_robust": {
        "description": "SAR (Sharpness-Aware) adaptation — more robust than Tent for moderate shifts. Avoids sharp minima that cause collapse. Good for cross-device shifts where Tent is too aggressive.",
        "when": "Same modality shift where mild_same_modality (Tent) didn't work or is too risky",
        "mechanism_class": "sar",
        "params": {
            "trainable_scope": "bn_only",
            "max_iterations": 1,
            "use_fda_warmup": False,
            "use_histogram_matching": False,
            "use_sar": True,
            "tent_bn_mode": False,
            "skip_adaptation": False,
            "use_anatomy_refinement": False,
            # Published SAR (Niu et al. ICLR 2023) uses pure entropy
            # minimization with a sharpness-aware gradient step; no
            # admixture with consistency / diversity.
            "loss_weights": {
                "entropy": 1.0, "consistency": 0.0,
                "diversity": 0.0, "bn_adapt": 0.0,
            },
        },
    },

    "novel_unknown": {
        "description": "Shift type doesn't match any known protocol — conservative fallback",
        "when": "Agent 3 is uncertain about classification",
        "mechanism_class": "tent",
        "params": {
            "trainable_scope": "bn_only",
            "max_iterations": 1,
            "use_fda_warmup": False,
            "use_histogram_matching": False,
            "tent_bn_mode": True,
            "skip_adaptation": False,
            "use_anatomy_refinement": False,
            # Conservative default: pure Tent loss (matches the fallback
            # mechanism_class "tent"). The safety comes from the final VLM
            # gate, not from diluting the loss.
            "loss_weights": {
                "entropy": 1.0, "consistency": 0.0,
                "diversity": 0.0, "bn_adapt": 0.0,
            },
        },
    },
}


# ================================================================
# Protocol Selection (for Agent 3)
# ================================================================

PROTOCOL_SELECTION_PROMPT = """You are selecting an adaptation protocol based on the diagnosis.

## Domain Expert's Analysis (Agent 1)
{domain_analysis}

## Model Analyst's Assessment (Agent 2)
{model_assessment}

## Model Architecture
{model_architecture_info}

## Available Protocols

{protocol_descriptions}

## Instructions

Based on Agent 1 and Agent 2's findings, which protocol fits best?

## Step 1: REASON about what's happening

Read Agent 1 and Agent 2 carefully. In your own words, describe:
- What is the domain shift? (physics-level understanding)
- How is the model failing? (what Agent 2 observed)
- What kind of intervention makes sense? (your clinical judgment)

Think like a doctor examining a patient — understand the problem FIRST, then decide on treatment.

## Step 2: CHECK your reasoning against the protocol reference

After forming your own judgment, verify it against these validated protocols:

**Protocol Reference Table** (use to VALIDATE your reasoning, not as primary decision):

| Your reasoning says... | Matching protocol |
|---|---|
| Different imaging physics (CT vs MRI, MRI vs US) | cross_modality_fda |
| Same modality, different sequence/weighting (T1↔T2, pre↔post contrast) | contrast_change |
| Same modality & sequence, model has minor boundary errors | mild_same_modality |
| Same modality & sequence, model under-segments or needs more capacity | moderate_cross_device |
| Same modality, Tent too aggressive, need robust adaptation | sar_robust |
| Model predictions already clinically acceptable | do_no_harm |
| Unsure / unusual situation | novel_unknown |

**Escalation principle**: If a previous attempt FAILED, choose a DIFFERENT protocol:
  mild_same_modality (Tent) failed → try sar_robust (more robust than Tent)
  sar_robust failed → try moderate_cross_device (wider scope, batch mode)
  moderate_cross_device failed → try contrast_change (different approach)

  CRITICAL EXCEPTION: If a previous attempt MADE THINGS WORSE (quality metrics decreased),
  AND the shift is mild, the correct response is do_no_harm (skip).
  When adaptation hurts a model that was already working reasonably well,
  the source model IS the best answer. Don't keep trying harder — stop.

**Architecture constraint**: If model has no BatchNorm, mild_same_modality (Tent) won't work — use moderate_cross_device instead.
Architecture info: {model_architecture_info}

**Key principle**: vendor/coil/scanner/calibration differences do NOT change tissue contrast relationships. Same sequence on different scanners → same tissues are bright/dark → mild shift.

Return JSON:
```json
{{
    "selected_protocol": "<protocol name>",
    "reasoning": "<step-by-step reasoning citing Agent 1 and Agent 2 findings>",
    "confidence": "high | medium | low",
    "alternative_protocol": "<backup if first choice fails>"
}}
```"""


def get_protocol_descriptions(capabilities=None) -> str:
    """Format protocol descriptions for the LLM prompt. If `capabilities` is
    provided, only protocols whose `required_capabilities` are satisfied by
    the model are included — the LLM literally cannot see disabled choices.
    Mechanism-only wording — no dataset names or per-dataset validation scores.
    """
    names = get_available_protocols(capabilities) if capabilities is not None else list(PROTOCOLS.keys())
    lines = []
    for name in names:
        proto = PROTOCOLS[name]
        lines.append(f"**{name}**")
        lines.append(f"  {proto['description']}")
        lines.append(f"  Use when: {proto['when']}")
        lines.append(f"  Mechanism class: {proto.get('mechanism_class', 'tent')}")
        lines.append("")
    return "\n".join(lines)


def get_available_protocols(capabilities) -> list:
    """Return the list of protocol names that the given model `capabilities`
    set supports. Protocol is available iff its required_capabilities are a
    subset of the model's capabilities. `capabilities` must be iterable/set.

    When `capabilities` is None or empty, returns every protocol (callers that
    don't care about filtering — e.g., standalone inspection scripts).
    """
    if not capabilities:
        return list(PROTOCOLS.keys())
    caps = frozenset(capabilities)
    return [name for name in PROTOCOLS
             if PROTOCOL_CAPABILITIES.get(name, frozenset()).issubset(caps)]


def get_protocol(protocol_name: str) -> Dict:
    """Get parameters for a protocol.

    Stamps in mechanism-default lr and num_steps_base from
    MECHANISM_DEFAULTS (literature, not dataset-tuned). Returns a DEEP copy
    so callers can mutate freely without corrupting the global PROTOCOLS
    registry (e.g., LLM loss_weight overrides).
    """
    import copy
    if protocol_name not in PROTOCOLS:
        print(f"  Warning: unknown protocol '{protocol_name}', using novel_unknown")
        protocol_name = "novel_unknown"

    proto = PROTOCOLS[protocol_name]
    params = copy.deepcopy(proto["params"])
    params["protocol"] = protocol_name
    params["protocol_description"] = proto["description"]

    # Stamp mechanism-default lr + num_steps_base (literature constants,
    # not per-dataset tuning). Severity scaling happens separately in
    # scale_protocol_by_severity.
    mech = proto.get("mechanism_class", "tent")
    mech_defaults = MECHANISM_DEFAULTS.get(mech, MECHANISM_DEFAULTS["tent"])
    params.setdefault("learning_rate", mech_defaults["learning_rate"])
    params.setdefault("num_steps", mech_defaults["num_steps_base"])
    params["mechanism_class"] = mech
    return params


def get_fallback_protocol(domain_analysis: Dict, model_assessment: Dict,
                            capabilities=None) -> str:
    """Rule-based protocol selection when LLM is unavailable. Respects
    `capabilities` if provided — never returns a protocol the model cannot
    run, silently substitutes the safest available alternative."""
    available = set(get_available_protocols(capabilities))

    def _pick(name):
        """Return `name` if available, else fall back along a safe chain."""
        if name in available:
            return name
        # Fallback order: cross_modality_fda → contrast_change
        #                 contrast_change    → novel_unknown
        #                 mild_same_modality → novel_unknown
        # do_no_harm always available (empty required_capabilities).
        chain = {
            "cross_modality_fda": ["contrast_change", "novel_unknown", "do_no_harm"],
            "contrast_change":    ["novel_unknown", "do_no_harm"],
            "mild_same_modality": ["novel_unknown", "do_no_harm"],
            "moderate_cross_device": ["novel_unknown", "do_no_harm"],
            "sar_robust":         ["novel_unknown", "do_no_harm"],
            "novel_unknown":      ["do_no_harm"],
            "do_no_harm":         [],
        }
        for alt in chain.get(name, []):
            if alt in available:
                return alt
        return "do_no_harm"

    contrast = domain_analysis.get("contrast_mechanism_change",
               domain_analysis.get("contrast_mechanism", "same"))
    failure = model_assessment.get("failure_type", "mixed")
    halluc = model_assessment.get("halluc_ratio", 0)

    if failure == "model_works_ok":
        return _pick("do_no_harm")
    if failure == "under_segmentation":
        if contrast in ("different",):
            return _pick("contrast_change")
        return _pick("mild_same_modality")

    shift_level = domain_analysis.get("shift_level", 0)
    if shift_level == 3:
        return _pick("cross_modality_fda")
    elif shift_level == 2:
        return _pick("contrast_change")
    elif shift_level == 1:
        return _pick("mild_same_modality")

    if contrast == "fundamentally_different":
        return _pick("cross_modality_fda")
    elif contrast == "different":
        return _pick("contrast_change")
    elif contrast in ("same", "similar", "approximately_same"):
        return _pick("mild_same_modality")
    else:
        return _pick("novel_unknown")


# ================================================================
# Severity-Based Parameter Scaling
# ================================================================

def scale_protocol_by_severity(plan: Dict, severity: float,
                                fg_ratio: float = None,
                                expected_min: float = 0.05) -> Dict:
    """Scale protocol parameters by failure severity from Agent 2.

    The base protocol determines WHICH TOOLS to use (shift type).
    Agent 2's severity score determines HOW HARD to apply them.

    Scaling formulas:
        - num_steps      *= 1 + 2*severity     (e.g., 80 -> 80~240)
        - max_iterations  = round(1 + 2*severity), capped at 3
        - scope widens to bn+output if severity > 0.6
        - learning rate untouched (protocol-specific)

    Special cases:
        - If protocol is `do_no_harm` or `cross_modality_fda` (which have
          num_steps=0 by design), DO NOT scale.
        - If model fg_ratio is severely under-segmenting (< expected_min/2),
          force wider scope regardless of severity.

    Args:
        plan: Protocol plan dict from get_protocol().
        severity: Failure severity score from Agent 2 (0.0-1.0).
        fg_ratio: Measured target foreground ratio (optional, for scope override).
        expected_min: Expected minimum foreground fraction.

    Returns:
        Modified plan with scaled parameters.
    """
    proto_name = plan.get("protocol", "")

    # Don't scale skip protocols or pure-FDA protocols
    if proto_name in ("do_no_harm",) or plan.get("skip_adaptation"):
        return plan
    if plan.get("num_steps", 0) == 0:
        # cross_modality_fda runs FDA only, no gradient steps
        return plan

    # Clamp severity to [0, 1]
    sev = max(0.0, min(1.0, float(severity)))

    # Scale num_steps: linear in severity
    base_steps = int(plan.get("num_steps", 80))
    plan["num_steps"] = max(20, int(base_steps * (1 + 2 * sev)))

    # Scale max_iterations: 1 (sev<0.5) -> 2 (0.5<=sev<1.0) -> 3 (sev=1.0)
    # Uses floor so iterations only escalate at clear severity thresholds.
    import math
    base_iter = int(plan.get("max_iterations", 1))
    scaled_iter = max(1, min(3, 1 + math.floor(2 * sev)))
    plan["max_iterations"] = max(base_iter, scaled_iter)  # never lower

    # Note: trainable_scope is chosen by Agent 3's LLM with full context
    # (severity, fg_drop, failure_type, BN divergence, architecture).
    # No post-hoc scope override here — the pattern across the pipeline is
    # that the LLM owns categorical decisions; rules only scale quantitative
    # dials (num_steps, max_iterations, learning_rate).

    plan["severity_scaled"] = True
    plan["scaling_severity"] = sev

    return plan
