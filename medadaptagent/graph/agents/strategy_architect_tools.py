"""Agent 3 v2 — Strategy Architect over the unified tool registry.

Replaces the mechanism-class abstraction (mild_same_modality, etc.)
with direct reasoning over `medadaptagent.tools.TOOL_REGISTRY`. Output
contract (multi-candidate, AutoResearch-style):

    {
      "candidates": [
        {"tools": ["tent"], "tool_kwargs": {}, "reason": "...", "confidence": 0.7},
        {"tools": ["bn_reset"], "tool_kwargs": {}, "reason": "...", "confidence": 0.5},
        {"tools": ["histogram_match", "tent"], ...}
      ],
      "skip_adaptation": false
    }

Agent 4 v2 then runs all candidates and uses a VLM tournament to pick
the winner. This is more robust than committing to a single chain when
the LLM's evidence is ambiguous. Capabilities filter is enforced
upstream via `tools.available_tools(caps)` so the LLM only sees viable
options. Each candidate is validated independently.
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional

from medadaptagent import tools as T
from medadaptagent.tools import composition as _composition


# Maximum chain length per candidate. Longer chains are truncated.
MAX_CHAIN_LEN = 4
# Maximum number of candidates in the tournament. More candidates =
# more VLM calls. 3 is a good tradeoff: covers conservative/moderate/
# aggressive in one round.
MAX_CANDIDATES = 3


# Phase Q2: structured kwarg schemas per tool. Surfaces to Agent 3 the
# concrete value ranges it can sweep over. Pulled from the actual tool
# implementations in medadaptagent/tools/*.py; widening these slightly
# beyond the published-default neighborhood to encourage exploration.
TOOL_KWARG_SCHEMA = {
    "tent": {"lr": [1e-4, 5e-4, 1e-3, 5e-3],
              "epochs": [1, 3, 5]},
    "sar": {"lr": [1e-3, 5e-3, 1e-2],
             "rho": [0.01, 0.05, 0.1],
             "ent_margin": [0.4, 0.6, 0.8]},
    "eata": {"lr": [1e-4, 5e-4, 1e-3],
              "ent_margin": [0.4, 0.6, 0.8]},
    "tent_ss": {"lr": [1e-4, 5e-4, 1e-3],
                 "steps": [5, 20, 50]},
    "sar_ss": {"lr": [1e-3, 5e-3, 1e-2],
                 "steps": [5, 20, 50]},
    "eata_ss": {"lr": [1e-4, 5e-4, 1e-3],
                  "steps": [5, 20, 50]},
    "finetune_head": {"lr": [1e-4, 5e-4, 1e-3, 5e-3],
                          "epochs": [3, 5, 10]},
    "finetune_partial": {"lr": [1e-5, 1e-4, 5e-4, 1e-3],
                              "epochs": [3, 5, 10],
                              "freeze_ratio": [0.3, 0.5, 0.7]},
    "finetune_lora": {"lr": [1e-4, 5e-4, 1e-3],
                          "epochs": [5, 10, 20],
                          "rank": [4, 8, 16]},
    "lora_tta": {"lr": [1e-3, 5e-3, 1e-2],
                  "epochs": [1, 3, 5],
                  "rank": [4, 8, 16],
                  "ent_margin": [0.0, 0.4, 0.6],
                  "alpha": [0.5, 1.0, 2.0]},
    "crop_center": {"crop_fraction": [0.5, 0.6, 0.7, 0.8, 0.9]},
    "fda_warmup": {"L": [0.005, 0.01, 0.02, 0.05]},
}


# Phase Q1: tool family taxonomy for diversity enforcement. Must stay in
# sync with experiment_manager.TOOL_FAMILY.
TOOL_FAMILY = {
    "bn_reset": "stats_norm",
    "tent": "gradient_norm", "sar": "gradient_norm",
    "eata": "gradient_norm", "tent_ss": "gradient_norm",
    "sar_ss": "gradient_norm", "eata_ss": "gradient_norm",
    "lora_tta": "gradient_norm",
    "finetune_head": "gradient_supervised",
    "finetune_partial": "gradient_supervised",
    "finetune_lora": "gradient_supervised",
    "histogram_match": "intensity", "percentile_clip": "intensity",
    "clahe": "intensity",
    "fda_warmup": "frequency",
    "crop_center": "spatial",
    "tta_aug": "ensemble",
    "noop": "noop",
}


def _extract_capabilities(model_info: Dict) -> List[str]:
    caps = model_info.get("capabilities") if model_info else None
    if caps:
        return sorted(caps)
    # Derive from has_batchnorm fallback
    if (model_info or {}).get("has_batchnorm", True):
        return ["bn_running_stats", "tent_affine"]
    if (model_info or {}).get("norm_type") in (
            "instancenorm", "layernorm", "groupnorm"):
        return ["tent_affine"]
    return []


PROMPT_TEMPLATE = """You are the Strategy Architect for a medical-imaging \
test-time adaptation pipeline. You are given:

  • Agent 1's domain-gap analysis (modality, intensity/contrast, FOV, \
calibration concerns)
  • Agent 2's failure-type assessment (severity 0..1, failure_type, \
source_quality)
  • The model's architecture capabilities

Your job: propose your TOP {max_candidates} candidate adaptation strategies \
from the menu below. A downstream VLM will look at the actual outputs of \
each strategy on this target dataset and pick the winner — your job is \
to nominate strategies that are individually plausible AND collectively \
diverse, so the VLM has real choices.

Each candidate is a SHORT ordered chain of 1–{max_chain} tools.

Diversity rule: the {max_candidates} candidates should span different \
risk/scope tradeoffs:
  • At least one CONSERVATIVE option (small footprint: `noop`, `bn_reset`, \
or a single preprocessing tool alone). This is the "do less" floor.
  • At least one MODERATE option (e.g., `tent` alone, or one preprocessing \
tool chained with `tent`/`eata`).
  • At most one AGGRESSIVE option (longer chain or `sar`/longer adaptation). \
Only include this if Agent 2 indicates large severity AND a clear gap path.

Do NOT submit three nearly-identical chains — that wastes the tournament. \
If two of your candidates would be the same tools in different order, drop \
one and propose something genuinely different.

## Tool menu (capability-filtered for this model)

{tool_menu}

## Reasoning protocol (apply per candidate)

You are a mechanistic strategist, NOT a recipe-matcher. For each candidate,
reason from cause to mechanism in this order:

### Step 1 — Diagnose the cause class

From Agent 1's domain analysis and Agent 2's failure_type + structure, name
the LIKELIEST cause of the failure. Common cause classes:

  • **Input-distribution shift**: target's pixel statistics (intensity range,
    contrast, scanner gain, bias field) differ from source's training set.
    Symptom: source's predictions are "miscalibrated" but in roughly the right
    place; foreground ratio off; mask edges fuzzy.

  • **Decision-boundary drift**: model is uncertain on slightly-shifted
    target; predictions are low-entropy but in the wrong direction. Symptom:
    confident-but-wrong; under_segmentation often falls here.

  • **Channel-statistics shift**: BN running stats from source no longer
    match target's per-channel mean/var. Symptom: characteristic of cross-
    site/cross-vendor data with otherwise similar appearance.

  • **Severe shift / structural collapse**: source produces near-zero
    foreground or wildly off-target masks. complete_failure usually fits.

  • **Geometric mismatch**: target FOV / orientation / pixel-spacing differs.
    Symptom: structure is at unexpected location/scale.

  • **Mask-topology artifact**: predictions are roughly correct but contain
    holes, specks, or fragmented components. Postprocessing fix.

### Step 2 — Map cause class to mechanism

Each tool corrects ONE mechanism. Read the mechanism in its profile.
Match cause → mechanism:

  - input-distribution shift → **preprocessing tools** (histogram_match,
    clahe, gamma_correct, percentile_clip, n4_bias_correction)
  - channel-statistics shift → **bn_reset** (BN models only)
  - decision-boundary drift → **gradient TTA on norm-affine**
    (tent / sar / eata)
  - severe shift → either gradient TTA, frequency-domain (fda_warmup), or
    `noop` if the gap is too large for label-free fixes
  - geometric mismatch → **spatial tools** (crop_center, resize_match,
    flip_detect)
  - mask-topology artifact → **postprocessing**
    (largest_connected_component, hole_fill, morph_open_close,
    small_component_filter)
  - high diagnostic uncertainty → **noop** or
    **weight_average_source_adapted** for safety

### Step 3 — Pick {max_candidates} candidates spanning DIFFERENT mechanisms

The tournament discriminates between mechanisms. Three candidates that all
target the same mechanism is wasted compute. Concretely:

  • At least one candidate should test the SINGLE strongest mechanism for
    the diagnosed cause (e.g., `tent` alone for decision-boundary drift,
    `bn_reset` alone for channel-stats shift, `histogram_match` alone for
    pure intensity shift).

  • A second candidate should test a DIFFERENT plausible cause class
    (your second-best diagnosis). E.g., if you're 70% sure it's
    decision-boundary drift but 30% input-distribution, candidate 2
    should target the input-distribution.

  • A third candidate is the SAFETY floor: `noop`, `bn_reset` (if BN), or
    `weight_average_source_adapted`.

### Step 4 — Reason explicitly per candidate

In each candidate's "reason" field, state:
  (a) which cause class this candidate targets,
  (b) the specific mechanism (in your own words, not just "tent"),
  (c) why this is plausible given Agent 1/2.

### Semi-supervised mode (when K target labels are available)

If your capabilities list contains "fewshot_labels", the user has supplied
K labeled target samples. Several tools actually USE the labels for
gradient signal: the semi-supervised TTA variants (`tent_ss`,
`sar_ss`, `eata_ss`) add a cross-entropy term on top of
entropy/SAR/EATA, and the two finetune tools (`finetune_head`,
`finetune_partial`) train weights directly on the labels. When this
capability is present:

  - **ALWAYS include at least one label-using tool** as a candidate,
    regardless of severity. The held-out-Dice tournament will decide
    whether it beats the unsupervised baseline. Empirically, even 1-2
    labels can give +10pp Dice on top of pure TTA; leaving these tools
    out wastes the user-supplied supervision.

  - Map severity to which label-using tool fits:
      * Low severity / mostly-fine source (sq >= 0.8) -> `finetune_head`
        (smallest finetune surface, low overfit risk, recalibrates the
        output mapping only)
      * Medium severity (0.3 < sev < 0.6) -> `tent_ss` or
        `finetune_partial` (norm-only vs last-30% trainable params)
      * High severity (sev >= 0.6 or sq < 0.7) -> `finetune_partial` or
        `eata_ss` (more parameters or robust SS variant)

  - Pure-TTA variants (no _ss / not finetune) belong in the SAFETY-floor
    slot as a hedge against label-using overfit.

  - The user-paid-for-labels have most value in a gradient-signal role;
    the gate uses them for verification regardless.

### Hard constraints

  - Never chain two gradient-based TTA tools (e.g., tent then sar).
  - Do not include `chain` itself in tools list; chain is implicit.
  - Postprocessing tools only when Agent 1's structures field clearly
    indicates the topology constraint applies (e.g., a single compact
    foreground class for largest_connected_component).
  - Single tools are valid candidates and often beat chains in
    label-free TTA. Do NOT pad a candidate with preprocessing just to
    make the chain look richer.

## Inputs

### Agent 1 — domain analysis
{agent1}

### Agent 2 — model assessment
{agent2}

### Model capabilities
{capabilities}

## Output — TWO-LEVEL HIERARCHICAL PLAN (Phase D2)

Return ONLY a JSON object with this schema; no commentary.

You must reason at TWO levels: first WHICH mechanism classes to
address (the WHY), then WHICH specific tools + kwargs within those
classes (the HOW). This separation lets us ablate each level
independently and gives the tournament a structured search space.

Valid mechanism_class values:
  - "stats_norm"            : BN running-stat shift (use bn_reset)
  - "gradient_norm"         : norm-affine entropy/SAR/EATA family
  - "gradient_supervised"   : finetune_head / partial / lora
  - "intensity"             : histogram_match / percentile_clip / clahe
  - "spatial"               : crop_center
  - "frequency"             : fda_warmup
  - "ensemble"              : tta_aug
  - "noop"                  : keep source

{{
  "mechanism_analysis": {{
    "primary": "<class>",
    "secondary": "<class or null>",
    "tertiary": "<class or null>",
    "reasoning": "1-2 sentences. WHY these classes specifically, citing Agent 1's shift mechanism + Agent 2's failure_type. e.g. 'A1 reports CT->MR intensity shift as dominant; A2 says mild_drift not source_broken so stats_norm secondary suffices.'"
  }},
  "candidates": [
    {{
      "mechanism_class": "<one of the classes named in mechanism_analysis>",
      "tools": ["<tool_name>", ...],
      "tool_kwargs": {{"<tool_name>": {{"key": value}}, ...}},
      "reason": "1-2 sentences. WHY this specific tool addresses the chosen mechanism class.",
      "expected_outcome": "1 sentence prediction of what the gate / VLM / GT should see if this strategy works.",
      "confidence": 0.0..1.0,
      "stance": "conservative" | "moderate" | "aggressive"
    }},
    ... up to {max_candidates}
  ],
  "skip_adaptation": false,
  "clarification_request": null
}}

HIERARCHY RULES (validator enforces):
  H1. Every candidate's mechanism_class MUST be one of
      mechanism_analysis.{{primary, secondary, tertiary}} (or "noop").
  H2. At least 2 DIFFERENT mechanism classes must be represented
      across the candidates (no all-same-class plans).
  H3. The primary class MUST appear in at least one candidate.

## Multi-Turn: Requesting Clarification (OPTIONAL)

If Agent 1's diagnosis or Agent 2's assessment leaves a CRITICAL
ambiguity that would change your tool selection, you may emit a
"clarification_request" string instead of leaving it null. Examples:

- "Agent 1 reports FOV shift +12pp but does not specify if it is
  L-R or A-P — affects crop_center axis selection"
- "Agent 2 says complete_failure but source_quality_score=0.85 —
  please confirm if the failure is in the OUTPUT or just BN stats"
- "Agent 1 expects spleen at 1-3% area but target predicts 12% —
  is the target labeled differently (e.g., includes liver)?"

The next iteration of this agent will receive your request in the
replan context and refine its strategy accordingly. Use sparingly —
only when ambiguity would meaningfully change your tool choice.
Leave null when the existing evidence is sufficient.

## Tool Parameter Override Examples (USE tool_kwargs WHEN APPROPRIATE)

For most cases, default tool params work. But when Agent 1/2 evidence
suggests a non-default would help, OVERRIDE via tool_kwargs:

- tent_ss with severe shift -> more steps:
  "tools": ["tent_ss"],
  "tool_kwargs": {{"tent_ss": {{"steps": 20, "lr": 5e-4}}}}

- finetune_head with very low source_quality -> more aggressive lr:
  "tools": ["finetune_head"],
  "tool_kwargs": {{"finetune_head": {{"lr": 1e-3, "epochs": 5}}}}

- finetune_partial with abundant K labels (K >= 20) -> more epochs:
  "tools": ["finetune_partial"],
  "tool_kwargs": {{"finetune_partial": {{"epochs": 8, "freeze_ratio": 0.5}}}}

- crop_center with extreme FOV mismatch (background fraction >25pp) ->
  tighter crop:
  "tools": ["crop_center"],
  "tool_kwargs": {{"crop_center": {{"crop_fraction": 0.7}}}}

LEAVE tool_kwargs EMPTY ({{}}) when default params are appropriate.
Over-parameterizing is risky — only override when you have specific
evidence that the default won't work.

## Composed pipelines (OPTIONAL — Phase D4)

A candidate is normally a `tools` list with a name-keyed `tool_kwargs`.
For most candidates that is enough. But you are NOT restricted to the
published single-tool baselines: you may instead declare an explicit,
ORDERED pipeline of atoms via a `composition` field. Each atom is one
tool plus its own bound kwargs, and the pipeline runs left-to-right
over one model. This lets you (a) fix an exact order and (b) repeat a
tool with different kwargs at different positions — neither of which a
name-keyed `tool_kwargs` can express. Use it to construct strategies
that are NOT any one published baseline, e.g. clip the intensity, reset
the normalization buffers, then entropy-adapt the affines:

  "composition": [
    {{"tool": "percentile_clip", "kwargs": {{}}}},
    {{"tool": "bn_reset", "kwargs": {{}}}},
    {{"tool": "tent", "kwargs": {{"lr": 5e-4, "epochs": 3}}}}
  ]

Ordering rules the validator enforces (illegal atoms are dropped):
  - Never put two gradient-adapt steps (tent/sar/eata/_ss) back-to-back
    — they double-update the same parameters.
  - Put every input-space transform (preprocessing / spatial /
    frequency) BEFORE any model-weight step; one placed after a weight
    step is dropped because it invalidates the adaptation just run.
  - A non-gradient stats step (bn_reset) BEFORE a gradient step is fine.
  - At most {max_chain} atoms.

When you provide `composition`, you may omit `tools`/`tool_kwargs` —
they are derived from it. Use a plain `tools` list when a single tool
(or a simple ordered chain with at most one kwarg set per tool) is all
you need; reach for `composition` when the strategy genuinely needs an
explicit order or a repeated tool. `mechanism_class` still applies and
should name the dominant mechanism of the pipeline.
"""


def _summarize_agent1(da: Dict) -> str:
    if not da:
        return "(unavailable)"
    keys = ["modality", "shift_level", "severity", "dominant_shift_factor",
            "calibration_concern", "foreground_priors", "structures"]
    out = {}
    for k in keys:
        if k in da:
            v = da[k]
            if isinstance(v, str) and len(v) > 200:
                v = v[:200] + "..."
            out[k] = v
    if "dominant_shifts" in da and isinstance(da["dominant_shifts"], list):
        out["dominant_shifts"] = da["dominant_shifts"][:5]
    return json.dumps(out, indent=2, default=str)


def _summarize_agent2(ma: Dict) -> str:
    if not ma:
        return "(unavailable)"
    keys = ["failure_type", "severity_score", "source_quality_score",
            "fg_ratio", "expected_min", "expected_max",
            "summary"]
    out = {k: ma[k] for k in keys if k in ma}
    return json.dumps(out, indent=2, default=str)


def _render_kwarg_schema(capabilities: List[str]) -> str:
    """Phase Q2: render tool kwarg schemas as a structured menu the
    LLM can reason over when proposing tool_kwargs sweeps."""
    avail_names = {t.name for t in T.available_tools(capabilities)}
    rows = []
    for name in sorted(TOOL_KWARG_SCHEMA):
        if name not in avail_names:
            continue
        kw_schema = TOOL_KWARG_SCHEMA[name]
        parts = [f"{k}={kw_schema[k]}" for k in kw_schema]
        rows.append(f"  {name}: {', '.join(parts)}")
    if not rows:
        return ""
    return ("## Tool kwarg sweep menu (Phase Q2)\n\n"
             "These tools accept the kwarg ranges below. When proposing "
             "a variant, set non-default values via `tool_kwargs` to "
             "explore the space — defaults are NOT necessarily best "
             "for your dataset.\n\n" + "\n".join(rows))


def _build_prompt(da: Dict, ma: Dict, capabilities: List[str],
                    replan_context: Optional[str] = None) -> str:
    menu = T.render_tool_menu(capabilities=capabilities)
    base = PROMPT_TEMPLATE.format(
        max_chain=MAX_CHAIN_LEN,
        max_candidates=MAX_CANDIDATES,
        tool_menu=menu,
        agent1=_summarize_agent1(da),
        agent2=_summarize_agent2(ma),
        capabilities=", ".join(capabilities) if capabilities else "(none reported)",
    )
    # Phase Q2: append the kwarg schema after the menu so Agent 3 has
    # explicit ranges to choose from when populating tool_kwargs.
    kwarg_block = _render_kwarg_schema(capabilities)
    if kwarg_block:
        marker = "## Inputs"
        if marker in base:
            base = base.replace(marker, kwarg_block + "\n\n" + marker, 1)
        else:
            base = base + "\n\n" + kwarg_block
    if replan_context:
        # Inject before the "## Output" section so the LLM sees past
        # attempts and reasons before deciding what to propose next.
        marker = "## Output"
        if marker in base:
            base = base.replace(marker, replan_context + "\n\n" + marker, 1)
        else:
            base = base + "\n\n" + replan_context
    return base


def _parse_json_object(text: str) -> Dict:
    """Tolerant JSON extraction: pull the first {...} block out of the
    LLM response, handle ```json fences, return {} on failure."""
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Try trimming trailing comma issues
        cleaned = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
        try:
            return json.loads(cleaned)
        except Exception:
            return {}


VALID_MECHANISM_CLASSES = {
    "stats_norm", "gradient_norm", "gradient_supervised",
    "intensity", "spatial", "frequency", "ensemble", "noop",
}


def _validate_candidate(cand: Dict, caps: set, idx: int) -> Optional[Dict]:
    """Validate a single candidate. Returns None if it has zero usable
    tools (caller should drop it)."""
    out = {
        "tools": [], "tool_kwargs": {},
        "mechanism_class": str(cand.get("mechanism_class", "")).strip().lower(),
        "reason": str(cand.get("reason", ""))[:600],
        "expected_outcome": str(cand.get("expected_outcome", ""))[:400],
        "stance": str(cand.get("stance", "moderate")),
        "confidence": 0.5, "warnings": [],
    }
    # Phase D2: validate mechanism_class declared by the candidate.
    # If LLM omitted or used an unknown class, infer from the tools.
    if out["mechanism_class"] not in VALID_MECHANISM_CLASSES:
        raw_tools_for_infer = (cand.get("tools", []) or [])
        if isinstance(raw_tools_for_infer, str):
            raw_tools_for_infer = [raw_tools_for_infer]
        inferred = None
        for t in raw_tools_for_infer:
            fam = TOOL_FAMILY.get(t)
            if fam in VALID_MECHANISM_CLASSES:
                inferred = fam; break
        if inferred:
            out["mechanism_class"] = inferred
            out["warnings"].append(
                f"cand[{idx}]: mechanism_class inferred as '{inferred}' "
                f"from tools")
        else:
            out["mechanism_class"] = "noop"
            out["warnings"].append(
                f"cand[{idx}]: mechanism_class missing/unknown, "
                f"defaulted to 'noop'")
    raw_tools = cand.get("tools", [])
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    for name in raw_tools[:MAX_CHAIN_LEN]:
        if name == "chain":
            out["warnings"].append(f"cand[{idx}]: dropped 'chain' (implicit)")
            continue
        if name not in T.TOOL_REGISTRY:
            out["warnings"].append(
                f"cand[{idx}]: dropped unknown tool '{name}'")
            continue
        tool = T.TOOL_REGISTRY[name]
        if any(r not in caps for r in tool.requires):
            out["warnings"].append(
                f"cand[{idx}]: dropped '{name}' "
                f"(requires {tool.requires})")
            continue
        # 3D task: skip tools whose implementation assumes 2D inputs
        if ("volumetric_predictions" in caps
                and not getattr(tool, "supports_3d", True)):
            out["warnings"].append(
                f"cand[{idx}]: dropped '{name}' (2D-only impl; "
                f"not 3D-safe)")
            continue
        out["tools"].append(name)

    raw_kw = cand.get("tool_kwargs", {})
    if isinstance(raw_kw, dict):
        out["tool_kwargs"] = {
            k: v for k, v in raw_kw.items()
            if k in out["tools"] and isinstance(v, dict)}

    # Phase D4: tool composition from atoms. If the candidate declares an
    # explicit ordered `composition` (list of {tool, kwargs} atoms), that
    # is authoritative — it can repeat a tool with different kwargs and
    # fix an explicit order that a name-keyed tool_kwargs cannot express.
    # Otherwise we synthesize the canonical atom list from the legacy
    # tools+tool_kwargs we just validated, so EVERY candidate downstream
    # carries an `composition` and the executor has one code path. The
    # composition is then validated for illegal orderings (two gradient-
    # adapt steps in a row, an input-space transform after a model
    # mutation, capability / 3D-safety) and projected back to legacy
    # tools+tool_kwargs so all existing consumers keep working.
    raw_comp = cand.get("composition")
    if isinstance(raw_comp, list) and raw_comp:
        atoms = _composition.normalize_composition({"composition": raw_comp})
        # Strip the implicit-only 'chain' meta-tool if the LLM included it.
        atoms = [a for a in atoms if a.tool != "chain"]
    else:
        atoms = _composition.normalize_composition(
            {"tools": out["tools"], "tool_kwargs": out["tool_kwargs"]})
    vres = _composition.validate_composition(
        atoms, capabilities=sorted(caps), max_len=MAX_CHAIN_LEN)
    for w in vres["warnings"]:
        out["warnings"].append(f"cand[{idx}]: composition {w}")
    valid_atoms = vres["atoms"]
    # Project the validated atoms back to the legacy fields (loss-free
    # except a repeated tool keeps its last kwargs in tool_kwargs; the
    # executor still honours every per-atom kwargs via `composition`).
    leg_tools, leg_kwargs = _composition.composition_to_legacy(valid_atoms)
    out["tools"] = leg_tools
    out["tool_kwargs"] = leg_kwargs
    out["composition"] = [a.to_dict() for a in valid_atoms]

    try:
        out["confidence"] = float(cand.get("confidence", 0.5))
    except (TypeError, ValueError):
        out["confidence"] = 0.5
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))

    if not out["tools"]:
        return None  # caller drops empty candidate
    return out


def _validate_mechanism_analysis(raw: Dict) -> Dict:
    """Phase D2: validate the top-level mechanism shortlist.

    Returns a dict with normalized {primary, secondary, tertiary,
    reasoning, allowed_classes}. Unknown classes get filtered;
    primary defaulted to 'noop' if missing entirely."""
    out = {"primary": "noop", "secondary": None, "tertiary": None,
            "reasoning": "", "allowed_classes": set()}
    if not isinstance(raw, dict):
        return out
    out["reasoning"] = str(raw.get("reasoning", ""))[:600]
    for k in ("primary", "secondary", "tertiary"):
        v = raw.get(k)
        if isinstance(v, str):
            v = v.strip().lower()
            if v in VALID_MECHANISM_CLASSES:
                out[k] = v
    out["allowed_classes"] = {c for c in
                                  (out["primary"], out["secondary"],
                                   out["tertiary"]) if c}
    # Always allow "noop" so the agent can opt out of any class
    out["allowed_classes"].add("noop")
    return out


def _validate_plan(plan: Dict, capabilities: List[str],
                      past_winner_toolsets: Optional[List[tuple]] = None,
                      catastrophic_families: Optional[set] = None) -> Dict:
    """Validate the multi-candidate plan: drop empty / capability-bad
    candidates, dedupe by tool sequence, cap to MAX_CANDIDATES, ensure
    a non-empty result."""
    caps = set(capabilities or [])
    out = {
        "candidates": [], "skip_adaptation": False, "warnings": [],
    }
    out["skip_adaptation"] = bool(plan.get("skip_adaptation", False))

    # Phase D2: validate the mechanism_analysis shortlist BEFORE
    # validating candidates, because candidate-validation enforces
    # mechanism_class ∈ allowed_classes.
    mech = _validate_mechanism_analysis(plan.get("mechanism_analysis", {}))
    out["mechanism_analysis"] = mech
    # Multi-turn (Phase M): preserve the optional clarification_request
    # string so the Manager can surface it in the next iter's replan
    # context. None means "no clarification needed".
    cr = plan.get("clarification_request")
    if isinstance(cr, str) and cr.strip():
        out["clarification_request"] = cr.strip()[:400]
        print(f"  Agent 3 requested clarification: \"{out['clarification_request'][:120]}\"")
    else:
        out["clarification_request"] = None

    raw_cands = plan.get("candidates", [])
    if not isinstance(raw_cands, list):
        raw_cands = []

    past_winners_set = set(past_winner_toolsets or [])
    banned_fams = set(catastrophic_families or [])

    seen_keys = set()
    for i, c in enumerate(raw_cands[:MAX_CANDIDATES * 2]):
        if not isinstance(c, dict):
            continue
        v = _validate_candidate(c, caps, i)
        if v is None:
            out["warnings"].append(f"cand[{i}]: empty after validation")
            continue
        key = tuple(v["tools"])  # ordered chain identity
        if key in seen_keys:
            out["warnings"].append(
                f"cand[{i}]: duplicate chain {key}, dropped")
            continue
        # Phase Q1: reject candidates whose tool-set EXACTLY matches a
        # past iter's winner. Forces the agent to explore new territory.
        if key in past_winners_set:
            out["warnings"].append(
                f"cand[{i}]: tool-set {list(key)} already tried as a "
                f"past iter winner — dropped (Phase Q1 anti-repeat).")
            continue
        # Phase Q3: drop candidates whose tools touch a catastrophe-
        # tagged family.
        if banned_fams:
            cand_fams = {TOOL_FAMILY.get(t, "other") for t in v["tools"]}
            hit = cand_fams & banned_fams
            if hit:
                out["warnings"].append(
                    f"cand[{i}]: tools={v['tools']} touch banned "
                    f"families {hit} — dropped (Phase Q3 safety).")
                continue
        # Phase D2 / H1: candidate's mechanism_class must be in the
        # shortlist that mechanism_analysis declared. If the agent
        # proposed a tool outside the shortlist, drop it — the
        # hierarchical contract is that the WHY constrains the HOW.
        if (v["mechanism_class"] not in mech["allowed_classes"]):
            out["warnings"].append(
                f"cand[{i}]: mechanism_class='{v['mechanism_class']}' "
                f"not in shortlist {sorted(mech['allowed_classes'])} "
                f"— dropped (Phase D2 H1)")
            continue
        seen_keys.add(key)
        out["candidates"].append(v)
        if len(out["candidates"]) >= MAX_CANDIDATES:
            break
        out["warnings"].extend(v.pop("warnings", []))

    # Phase D2 / H2: at least 2 distinct mechanism classes across the
    # surviving candidates (avoids all-same-class plans, which collapses
    # the tournament back to a parameter sweep).
    surviving_classes = {c["mechanism_class"] for c in out["candidates"]
                            if c["mechanism_class"] != "noop"}
    if len(out["candidates"]) >= 2 and len(surviving_classes) < 2:
        out["warnings"].append(
            f"Phase D2 H2: only {len(surviving_classes)} mechanism class "
            f"represented across {len(out['candidates'])} candidates "
            f"(want >=2). Tournament will run anyway but diversity "
            f"signal is weak.")

    # Phase D2 / H3: the primary class declared by mechanism_analysis
    # should appear in at least one surviving candidate. If not (LLM
    # discarded its own primary), log a warning — we don't reject,
    # because the existing Q1/Q3 filters may have caused this.
    if (mech["primary"] not in surviving_classes
            and mech["primary"] != "noop"
            and out["candidates"]):
        out["warnings"].append(
            f"Phase D2 H3: primary class '{mech['primary']}' not "
            f"represented in surviving candidates "
            f"{sorted(surviving_classes)}. Mechanism reasoning may "
            f"be inconsistent with tool choices.")

    # Safety: if no candidates and not skip → default to a single noop
    if not out["candidates"] and not out["skip_adaptation"]:
        out["candidates"].append({
            "tools": ["noop"], "tool_kwargs": {},
            "mechanism_class": "noop",
            "reason": "empty plan → safe default",
            "stance": "conservative", "confidence": 0.3,
        })
        out["warnings"].append("empty candidate list → defaulted to [noop]")
    return out


def run_strategy_architect_tools(domain_analysis: Dict,
                                   model_assessment: Dict,
                                   llm_client,
                                   model_info: Optional[Dict] = None,
                                   replan_context: Optional[str] = None,
                                   past_winner_toolsets: Optional[List[tuple]] = None,
                                   catastrophic_families: Optional[set] = None,
                                   ) -> Dict:
    """LLM-reasoned tool selection from the unified registry."""
    print("\n" + "=" * 55)
    print("AGENT 3 v2: Strategy Architect over Tool Registry")
    print("=" * 55)
    capabilities = _extract_capabilities(model_info or {})
    print(f"  capabilities: {capabilities}")
    available = [t.name for t in T.available_tools(capabilities)]
    print(f"  available tools: {len(available)}")

    prompt = _build_prompt(domain_analysis or {}, model_assessment or {},
                              capabilities,
                              replan_context=replan_context)

    if llm_client is None:
        # Capability-aware fallback: a small, diverse default tournament
        print("  [no LLM client] falling back to default 3-candidate set")
        cands = [{"tools": ["noop"], "stance": "conservative",
                   "reason": "no LLM; safe default", "confidence": 0.3}]
        if "bn_running_stats" in capabilities:
            cands.append({"tools": ["bn_reset"], "stance": "conservative",
                           "reason": "no LLM; cheap stats fix", "confidence": 0.3})
        if "tent_affine" in capabilities:
            cands.append({"tools": ["tent"], "stance": "moderate",
                           "reason": "no LLM; standard Tent", "confidence": 0.3})
        return _validate_plan({"candidates": cands}, capabilities,
                                  past_winner_toolsets=past_winner_toolsets,
                                  catastrophic_families=catastrophic_families)

    text = llm_client.chat(prompt, max_tokens=900) or ""
    plan = _parse_json_object(text)
    if not plan:
        print("  [LLM returned non-JSON; falling back to noop only]")
        return _validate_plan(
            {"candidates": [{"tools": ["noop"], "reason": "LLM parse fail",
                                "stance": "conservative", "confidence": 0.2}]},
            capabilities,
            past_winner_toolsets=past_winner_toolsets,
            catastrophic_families=catastrophic_families)
    validated = _validate_plan(plan, capabilities,
                                  past_winner_toolsets=past_winner_toolsets,
                                  catastrophic_families=catastrophic_families)
    # Phase D2: surface the mechanism shortlist BEFORE listing candidates
    # so the paper-figure reader sees the WHY before the HOW.
    mech = validated.get("mechanism_analysis") or {}
    if mech.get("primary"):
        shortlist = [c for c in (mech.get("primary"),
                                       mech.get("secondary"),
                                       mech.get("tertiary")) if c]
        print(f"  [Phase D2] mechanism shortlist: {shortlist}")
        if mech.get("reasoning"):
            print(f"  [Phase D2] reasoning: {mech['reasoning'][:200]}")
    print(f"  candidates ({len(validated['candidates'])}):")
    for i, c in enumerate(validated["candidates"]):
        mc = c.get("mechanism_class", "?")
        print(f"    [{i}] {c['stance']:12s}  mech={mc:<22}  "
              f"tools={c['tools']}  conf={c['confidence']:.2f}")
        print(f"        reason: {c['reason'][:150]}")
        kw = c.get("tool_kwargs") or {}
        non_empty = {t: p for t, p in kw.items() if p}
        if non_empty:
            print(f"        tool_kwargs: {non_empty}")
    if validated["warnings"]:
        for w in validated["warnings"]:
            print(f"    warn: {w}")
    return validated
