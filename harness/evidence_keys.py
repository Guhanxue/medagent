"""Defensive, single-source accessors for harness evidence keys (design §A.2).

Every component (decision spine, scaling-law, shift-analysis, tool-coupling, lessons,
delivery) reads the diagnosis through these helpers so no one hardcodes a wrong path or
crashes on the `{'error':...}` / `{'raw':...}` failure shapes that analyze_domain_shift can
emit. Pure functions, no imports beyond stdlib.
"""


def get_characterization(evidence) -> dict:
    """evidence['domain_shift']['characterization'] (or the dict itself); {} on error/raw."""
    ds = (evidence or {}).get("domain_shift") or {}
    if not isinstance(ds, dict):
        return {}
    ch = ds.get("characterization", ds)
    if not isinstance(ch, dict) or "error" in ch or "raw" in ch:
        return {}
    return ch


def get_model_assessment(evidence) -> dict:
    mi = (evidence or {}).get("model_info") or {}
    if not isinstance(mi, dict):
        return {}
    a = mi.get("assessment", mi)
    return a if isinstance(a, dict) else {}


def get_introspection(evidence) -> dict:
    mi = (evidence or {}).get("model_info") or {}
    if not isinstance(mi, dict):
        return {}
    intro = mi.get("introspection") or {}
    return intro if isinstance(intro, dict) else {}


def norm_token(assessment) -> str:
    """Tokenize the free-text normalization string to one of
    {batchnorm, instancenorm, layernorm, groupnorm, unknown} via lowercase substring."""
    s = str((assessment or {}).get("normalization", "")).lower()
    for stem, tok in (("batch", "batchnorm"), ("instance", "instancenorm"),
                      ("layer", "layernorm"), ("group", "groupnorm")):
        if stem in s:
            return tok
    return "unknown"


def metric_keys(record_or_evidence):
    """Return (true_key, labeled_key, source_key) generically for seg (Dice) vs cls (AUROC),
    so mining/scaling/delivery never hardcode 'true_dice'. Detects AUROC from the record's
    own keys or a 'metric' field; defaults to Dice (segmentation)."""
    r = record_or_evidence or {}
    keys = set(r.keys()) if isinstance(r, dict) else set()
    metric = str(r.get("metric", "")).upper() if isinstance(r, dict) else ""
    is_auroc = (metric == "AUROC" or "true_auroc" in keys or "source_auroc" in keys
                or "labeled_auroc" in keys or "shipped_auroc" in keys)
    if is_auroc:
        return ("true_auroc", "labeled_auroc", "source_auroc")
    return ("true_dice", "labeled_dice", "source_dice")
