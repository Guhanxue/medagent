"""Tool composition layer (Phase D4) — composing atoms into pipelines.

This module sits ON TOP of the existing TOOL_REGISTRY (purely additive;
the registry is untouched). It gives Agent 3 a way to declare an
explicit, ORDERED pipeline of *atoms* — each atom being one already-
registered tool bound to a specific kwarg dict — and gives Agent 4 a
single, well-typed object to run left-to-right over one ToolContext.

Why this exists
---------------
The legacy candidate contract was ``{"tools": [...], "tool_kwargs":
{tool_name: {...}}}``. Because ``tool_kwargs`` is keyed by tool *name*,
a tool could appear at most once with one bound kwarg set, and ordering
was only ever the order of the ``tools`` list. That ceiling means the
agent's reachable strategy set collapses onto the published-baseline
tool set. A *composition* lifts that ceiling: the agent can build an
ordered pipeline such as::

    percentile_clip -> bn_reset -> tent(steps=5, lr=5e-4) -> multi_scale_tta

which is not any single published baseline, and it can even repeat a
tool with different kwargs at different positions.

Representation
--------------
- ``Atom``        : one registered tool name + a bound kwargs dict.
- ``Composition`` : an ordered ``List[Atom]`` applied left-to-right.

Full back-compatibility
-----------------------
A plain ``tools`` list with empty ``tool_kwargs`` round-trips through
``normalize_composition`` to the identical atom sequence it described
before, so candidates that never use the new ``composition`` field
behave exactly as they did. ``composition_to_legacy`` projects an atom
list back to the ``(tools, tool_kwargs)`` pair so every existing
consumer (serialization, scoreboards, family diagnosis) keeps working
unchanged — the projection is loss-free unless a tool repeats, in which
case the last kwarg set wins for the legacy view only (the executor
still honours every per-atom kwarg).

This module is fully dataset-agnostic: it references only tool names and
their registry metadata (``category`` / ``side_effects`` / ``requires``
/ ``supports_3d``). No dataset name ever appears here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import TOOL_REGISTRY


# Tool categories that mutate the model (a gradient step OR a running-
# stat re-estimation). Used to forbid input-space transforms AFTER a
# model mutation: a preprocess that changes the very signal the model
# was just adapted to invalidates the adaptation.
MODEL_WEIGHT_CATEGORIES = {"stats_adapt", "gradient_adapt"}

# Categories that take a GRADIENT step on the norm-affine parameters.
# Two of these back-to-back would double-update the SAME affines on a
# stale optimizer state — a known instability. (stats_adapt, e.g.
# bn_reset, re-estimates running-stat BUFFERS, not affine weights, so a
# stats_adapt -> gradient_adapt sequence is legitimate and not blocked.)
GRADIENT_ADAPT_CATEGORIES = {"gradient_adapt"}

# Categories that transform the model INPUT (target image space). These
# belong before any weight mutation so the adaptation sees the final
# preprocessed signal.
INPUT_SPACE_CATEGORIES = {"preprocessing", "spatial", "frequency"}

# Categories that operate purely on predictions after the forward pass.
# They are order-free relative to weight mutations but, by convention,
# come last in a pipeline.
POSTPROCESS_CATEGORIES = {"postprocessing"}


@dataclass
class Atom:
    """One pipeline step: a registered tool bound to a kwargs dict.

    The kwargs are the exact per-invocation overrides the executor will
    inject into ``ctx.notes[tool]`` immediately before applying the
    tool, so the same tool can appear twice with different kwargs.
    """
    tool: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def category(self) -> str:
        t = TOOL_REGISTRY.get(self.tool)
        return t.category if t is not None else "unknown"

    def is_model_weight(self) -> bool:
        return self.category() in MODEL_WEIGHT_CATEGORIES

    def is_input_space(self) -> bool:
        return self.category() in INPUT_SPACE_CATEGORIES

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "kwargs": dict(self.kwargs)}


def _coerce_atom(raw: Any) -> Optional[Atom]:
    """Coerce one raw composition entry into an Atom. Accepts:
      - {"tool": name, "kwargs": {...}}        (canonical)
      - {"tool": name}                          (no kwargs)
      - {"name": name, "kwargs": {...}}         (alias for tool)
      - "tool_name"                             (bare string atom)
    Returns None for anything unrecognisable (caller drops it)."""
    if isinstance(raw, str):
        name = raw.strip()
        return Atom(tool=name, kwargs={}) if name else None
    if isinstance(raw, dict):
        name = raw.get("tool") or raw.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        kwargs = raw.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        return Atom(tool=name.strip(), kwargs=dict(kwargs))
    return None


def normalize_composition(candidate: Dict) -> List[Atom]:
    """Build the canonical ordered atom list for a candidate.

    Precedence:
      1. If the candidate carries an explicit ``composition`` list, that
         is authoritative (each entry coerced via ``_coerce_atom``).
      2. Otherwise fall back to the legacy ``tools`` + ``tool_kwargs``
         pair: one atom per tool, in list order, each bound to
         ``tool_kwargs.get(tool, {})``.

    This is the single bridge between the new and old candidate shapes,
    so every downstream stage can operate on atoms uniformly while plain
    ``tools`` candidates keep working byte-for-byte.
    """
    comp = candidate.get("composition")
    if isinstance(comp, list) and comp:
        atoms: List[Atom] = []
        for raw in comp:
            a = _coerce_atom(raw)
            if a is not None:
                atoms.append(a)
        return atoms
    # Legacy path: tools + tool_kwargs.
    tools = candidate.get("tools") or []
    if isinstance(tools, str):
        tools = [tools]
    tool_kwargs = candidate.get("tool_kwargs") or {}
    if not isinstance(tool_kwargs, dict):
        tool_kwargs = {}
    atoms = []
    for name in tools:
        if not isinstance(name, str):
            continue
        kw = tool_kwargs.get(name, {})
        atoms.append(Atom(tool=name, kwargs=dict(kw) if isinstance(kw, dict) else {}))
    return atoms


def composition_to_legacy(atoms: List[Atom]) -> Tuple[List[str], Dict[str, Dict]]:
    """Project an atom list back to the legacy ``(tools, tool_kwargs)``
    pair so existing serializers / scoreboards / family-diagnosis code
    (which read ``candidate['tools']``) keep working unchanged. When a
    tool repeats, the legacy ``tool_kwargs`` view keeps the LAST kwargs
    for that tool (the executor still honours every per-atom kwargs)."""
    tools: List[str] = []
    tool_kwargs: Dict[str, Dict] = {}
    for a in atoms:
        tools.append(a.tool)
        if a.kwargs:
            tool_kwargs[a.tool] = dict(a.kwargs)
    return tools, tool_kwargs


def validate_composition(atoms: List[Atom],
                          capabilities: Optional[List[str]] = None,
                          max_len: int = 4) -> Dict:
    """Validate + repair an ordered atom list. Drops illegal atoms and
    flags illegal orderings, returning a cleaned atom list plus warnings.

    Rules enforced (each rule that fires emits a human-readable warning):
      R0  Unknown tools are dropped.
      R1  Capability gate: an atom whose tool ``requires`` a capability
          not in ``capabilities`` is dropped.
      R2  3D-safety: on a volumetric task (``volumetric_predictions`` in
          capabilities) an atom whose tool is ``supports_3d=False`` is
          dropped.
      R3  No two gradient-adapt atoms may be adjacent — that would
          double-update the same norm-affine parameters on stale
          optimizer state. The second of the pair is dropped. (A
          stats_adapt step such as bn_reset BEFORE a gradient-adapt step
          is fine — it re-estimates buffers, not affine weights.)
      R4  No input-space atom (preprocessing / spatial / frequency) may
          appear AFTER a model-weight atom: changing the input the
          weights were just adapted to invalidates the adaptation. The
          out-of-order input-space atom is dropped.
      R5  Length is capped to ``max_len`` (excess atoms dropped).

    Returns ``{"atoms": [Atom, ...], "warnings": [str, ...],
               "ok": bool}`` where ``ok`` is True iff no rule fired.
    """
    caps = set(capabilities or [])
    # Phase D4 cap-gate: capabilities=None (exec-time call site) means
    # "trust Agent 3's authoritative R1/R2 filtering; enforce only the
    # ordering rules here". Without this, caps=None -> empty set -> every
    # atom with non-empty requires is dropped, silently turning a composed
    # chain into a partial no-op (and legacy tools=['tent'] into a no-op).
    # Agent 3 always passes real caps, so R1/R2 stay authoritative there.
    cap_gate = capabilities is not None
    is_3d = "volumetric_predictions" in caps
    warnings: List[str] = []
    cleaned: List[Atom] = []
    seen_model_weight = False
    prev_was_gradient_adapt = False

    for a in atoms:
        tool = TOOL_REGISTRY.get(a.tool)
        # R0: unknown tool.
        if tool is None:
            warnings.append(f"dropped unknown tool '{a.tool}'")
            continue
        # R1: capability gate.
        missing = [r for r in tool.requires if r not in caps]
        if cap_gate and missing:
            warnings.append(
                f"dropped '{a.tool}' (requires {missing} not in capabilities)")
            continue
        # R2: 3D-safety.
        if cap_gate and is_3d and not getattr(tool, "supports_3d", True):
            warnings.append(
                f"dropped '{a.tool}' (2D-only implementation; not 3D-safe)")
            continue
        cat = tool.category
        is_weight = (cat in MODEL_WEIGHT_CATEGORIES
                     or "model_weights" in getattr(tool, "side_effects", []))
        is_gradient = cat in GRADIENT_ADAPT_CATEGORIES
        is_input = cat in INPUT_SPACE_CATEGORIES
        # R3: two gradient-adapt atoms in a row.
        if is_gradient and prev_was_gradient_adapt:
            warnings.append(
                f"dropped '{a.tool}': two gradient-adapt steps in a row "
                f"would double-update norm-affine parameters (R3)")
            continue
        # R4: input-space transform after a model mutation.
        if is_input and seen_model_weight:
            warnings.append(
                f"dropped '{a.tool}': input-space transform after a model-"
                f"weight step invalidates the adaptation it just ran (R4)")
            continue
        # R5: length cap.
        if len(cleaned) >= max_len:
            warnings.append(
                f"dropped '{a.tool}': composition exceeds max length "
                f"{max_len} (R5)")
            continue
        cleaned.append(a)
        if is_weight:
            seen_model_weight = True
        prev_was_gradient_adapt = is_gradient

    return {"atoms": cleaned, "warnings": warnings, "ok": not warnings}


def render_composition(atoms: List[Atom]) -> str:
    """One-line human-readable rendering, e.g.
    ``percentile_clip -> bn_reset -> tent{lr=0.0005,epochs=3}``."""
    parts = []
    for a in atoms:
        if a.kwargs:
            kv = ",".join(f"{k}={v}" for k, v in a.kwargs.items())
            parts.append(f"{a.tool}{{{kv}}}")
        else:
            parts.append(a.tool)
    return " -> ".join(parts) if parts else "(empty)"
