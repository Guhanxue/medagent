"""Component D — advisory diagnosis->tool coupling (design §5).

Makes label-free tool SELECTION track the diagnosis, ADVISORY only:
  - render a diagnosis-prioritized shortlist + rationale in the §A.4 prompt section, and
  - present the recovery menu reordered (a STABLE PERMUTATION — every tool still reachable).
It NEVER filters, forces, or guards the menu — that forced-routing is what sank chaos &
fundus_roi before. `_available()` (the valid-action set) is untouched; this only changes the
ORDER tools are shown + a text suggestion the LLM may ignore.

EMPIRICAL DEVIATION from the original spec, on purpose: the spec led the modality/sequence
priors with PROTO_FEAT. Our evidence says otherwise — `bn_reset` (NORMALIZATION) is the
cheapest universal first move and it RECOVERS even the modality case (chaos CT->MRI 0.12->0.39),
while the PROTO_FEAT family is thin in this registry. Leading modality/sequence away from
bn_reset is precisely the chaos regression we must not reintroduce. So NORMALIZATION leads for
EVERY shift; the genuine diagnosis signal lives in `expect_tta_sufficient` (modality/sequence ->
False -> the rationale tells the agent to try bn_reset once then prefer a few labels over
exhausting every label-free tool). "Regression -> SHORTEN the family list, never add a guard."

Pure stdlib + the §A.2 accessor. Never raises.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

try:                                  # robust to repo-root vs harness/-on-path
    from harness.evidence_keys import get_characterization
except ImportError:
    from evidence_keys import get_characterization

# ── tool families (mechanism classes) ───────────────────────────────────────────────
NORMALIZATION = "normalization"   # intensity / histogram / BN-stat recalibration
SPECTRAL = "spectral"             # frequency-domain amplitude alignment
ENTROPY_TTA = "entropy_tta"       # entropy-min / norm-param test-time training
PROTO_FEAT = "proto_feat"         # feature / weight-level alignment toward source
OTHER = "other"                   # postproc / geometry / finetune / terminal — never reordered

_FAMILY = {
    "bn_reset": NORMALIZATION, "zscore_norm": NORMALIZATION, "percentile_clip": NORMALIZATION,
    "gamma_correct": NORMALIZATION, "clahe": NORMALIZATION, "histogram_match": NORMALIZATION,
    "n4_bias_correction": NORMALIZATION,
    "spectral": SPECTRAL, "fda_warmup": SPECTRAL,
    "tent": ENTROPY_TTA, "sar": ENTROPY_TTA, "eata": ENTROPY_TTA,
    "tent_ss": ENTROPY_TTA, "sar_ss": ENTROPY_TTA, "eata_ss": ENTROPY_TTA,
    "weight_average_source_adapted": PROTO_FEAT,
}

# Per-shift family ranking (lower index = shown earlier). Keyed on the REAL shift_type enum
# {intensity, contrast, modality, sequence, resolution, FOV, mixed, unknown}. mixed/unknown ->
# empty -> NO reorder. NORMALIZATION leads everywhere (see module docstring).
SHIFT_FAMILY_PRIOR = {
    "intensity":  [NORMALIZATION, SPECTRAL, ENTROPY_TTA, PROTO_FEAT],
    "contrast":   [NORMALIZATION, SPECTRAL, ENTROPY_TTA, PROTO_FEAT],
    "modality":   [NORMALIZATION, ENTROPY_TTA, SPECTRAL, PROTO_FEAT],   # tta_sufficient=False
    "sequence":   [NORMALIZATION, ENTROPY_TTA, SPECTRAL, PROTO_FEAT],   # tta_sufficient=False
    "resolution": [NORMALIZATION, PROTO_FEAT, ENTROPY_TTA, SPECTRAL],
    "FOV":        [NORMALIZATION, PROTO_FEAT, SPECTRAL, ENTROPY_TTA],
    "mixed":      [],
    "unknown":    [],
}


def tool_family(tool_name) -> str:
    return _FAMILY.get(str(tool_name), OTHER)


@dataclass
class ToolAdvice:
    shift_type: str = "unknown"
    shortlist: list = field(default_factory=list)      # tool names, priority order, in menu
    rationale: str = ""
    expect_tta_sufficient: bool = True
    family_rank: dict = field(default_factory=dict)    # tool_name -> (family_idx, menu_idx)

    def as_evidence(self) -> dict:
        return {"suggest": self.shortlist[:5], "why": self.rationale,
                "tta_sufficient": self.expect_tta_sufficient}

    def __str__(self):
        return json.dumps(self.as_evidence())


def _names(menu):
    return [(t.get("name") if isinstance(t, dict) else t) for t in (menu or [])]


def build_tool_advice(evidence: dict, menu) -> ToolAdvice:
    """Diagnosis-conditioned advisory shortlist + rationale. menu = view['tools'] (dicts) or
    names. Never raises; on any problem returns an inert ToolAdvice (no reorder, no shortlist)."""
    try:
        ch = get_characterization(evidence)
        st = str(ch.get("shift_type", "unknown")).lower()
        sev = str(ch.get("severity", "")).lower()
        sem = bool(ch.get("semantic_flag", False))
        prior = SHIFT_FAMILY_PRIOR.get(st, [])
        names = _names(menu)
        if not prior or not names:
            return ToolAdvice(shift_type=st)               # mixed/unknown/empty -> inert
        rank = {}
        for i, n in enumerate(names):
            fam = tool_family(n)
            if fam in prior:
                rank[n] = (prior.index(fam), i)
        shortlist = sorted(rank, key=lambda n: rank[n])[:5]
        expect = (st not in ("modality", "sequence")) and not sem
        first = shortlist[0] if shortlist else "bn_reset"
        if expect:
            why = (f"{st} shift ({sev}): normalization-family TTA usually addresses this — "
                   f"try {first} first.")
        else:
            why = (f"{st} shift ({sev}, semantic): label-free TTA is often INSUFFICIENT — "
                   f"try {first} once, then prefer requesting a few labels over exhausting "
                   f"every label-free tool.")
        return ToolAdvice(shift_type=st, shortlist=shortlist, rationale=why,
                          expect_tta_sufficient=expect, family_rank=rank)
    except Exception:
        return ToolAdvice()


def reorder_menu(menu, advice) -> list:
    """STABLE permutation of the tool-name list: the family-mapped recovery tools are placed
    into THEIR OWN slots in diagnosis-priority order; every other tool (terminal, diagnostic,
    finetune, request_labels, recommend, postproc, geometry) keeps its position. Asserts
    len+set preserved. Empty advice / no mapped tools -> identity."""
    names = list(menu) if not (menu and isinstance(menu[0], dict)) else _names(menu)
    rank = getattr(advice, "family_rank", None) or {}
    slots = [i for i, n in enumerate(names) if n in rank]
    if not slots:
        return names
    ordered = sorted((names[i] for i in slots), key=lambda n: rank[n])
    out = list(names)
    for pos, tool in zip(slots, ordered):
        out[pos] = tool
    assert len(out) == len(names) and set(out) == set(names), "reorder must be a permutation"
    return out
