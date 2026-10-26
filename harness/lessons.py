"""Component E — lessons store (design §6). Accumulate TRANSFERABLE, CATEGORICAL lessons across
runs and surface relevant ones as ADVISORY evidence — never brewed per-dataset rules.

A lesson speaks about a tool FAMILY on a shift CLASS: condition = categorical tokens only
(task_type, shift_type, severity, normalization) — NO dataset names, NO numbers-as-rules. The
only numbers are evidence_summary stats (n_runs, median true delta) used to set confidence.

Honest limitation (design §6.3): with the current tiny corpus the fingerprint barely
generalizes (normalization is ~constant; shift_type×severity is the only discriminative axis,
and some tasks collide). So: retrieval is GATED to confidence ≥ moderate (≥2 runs, consistent
sign), and conflicting-sign lessons collapse to weak (MIXED) and are NOT surfaced. Value grows
with more datasets.

Pure stdlib + the §A.2 accessors + tool_family. Never raises on bad input.
"""
from __future__ import annotations
import json
import os
import statistics

try:                                  # robust to repo-root vs harness/-on-path
    from harness.evidence_keys import get_characterization, get_model_assessment, norm_token
    from harness.tool_advisor import tool_family
except ImportError:
    from evidence_keys import get_characterization, get_model_assessment, norm_token
    from tool_advisor import tool_family

_CONF_ORDER = {"weak": 0, "moderate": 1, "strong": 2}
_HELP_EPS = 0.02            # |median true delta| below this = no lesson (noise)
_RETRIEVE_GATE = 1         # confidence >= moderate to surface


# ── fingerprint (retrieval key) ──────────────────────────────────────────────────────
def fingerprint_of(evidence, task_type):
    """Categorical retrieval key. None until BOTH diagnostics ran (characterization AND model
    assessment present) and None on error/raw shapes — so a histogram-blind run never matches."""
    ch = get_characterization(evidence)
    asmt = get_model_assessment(evidence)
    if not ch or not asmt:
        return None
    return {"task_type": task_type, "shift_type": str(ch.get("shift_type", "unknown")).lower(),
            "severity": str(ch.get("severity", "unknown")).lower(),
            "normalization": norm_token(asmt)}


def _match(cond, fp):
    """Categorical match. normalization weight 0 (near-constant today) — matched on
    task_type × shift_type × severity only until non-BN models exist (design §6.3)."""
    return (cond.get("task_type") == fp.get("task_type")
            and cond.get("shift_type") == fp.get("shift_type")
            and cond.get("severity") == fp.get("severity"))


def lessons_evidence(store, fp, k=4):
    """Retrieve up to k lessons matching fp, GATED to confidence ≥ moderate. Empty -> {} (inert
    §A.4). Each surfaced lesson is tagged advisory:true; a note reminds the agent these are hints
    to be verified by THIS run's own VLM/labeled evidence, not rules."""
    if not fp or not store:
        return {}
    cands = [l for l in store
             if _match(l.get("condition", {}), fp)
             and _CONF_ORDER.get(l.get("confidence"), 0) >= _RETRIEVE_GATE]
    cands.sort(key=lambda l: (-_CONF_ORDER.get(l.get("confidence"), 0),
                              -abs(l.get("evidence_summary", {}).get("median_true_delta", 0.0))))
    cands = cands[:k]
    if not cands:
        return {}
    return {"retrieved_lessons": [{"claim": l["claim"], "kind": l["kind"],
                                   "confidence": l["confidence"], "advisory": True} for l in cands],
            "retrieved_lessons_note": ("hints from past runs, NOT rules; verify with this run's "
                                       "own VLM/labeled evidence before relying on them")}


# ── mining (post-hoc, from a saved per-task rec) ─────────────────────────────────────
def _true_metric(audit_entry):
    if not isinstance(audit_entry, dict):
        return None
    v = audit_entry.get("true_dice", audit_entry.get("true_auroc"))
    return float(v) if isinstance(v, (int, float)) else None


def _task_type_of(rec):
    if rec.get("task_type"):
        return str(rec["task_type"])
    return "classification" if str(rec.get("metric", "")).upper() == "AUROC" else "segmentation"


def mine_lessons(rec, task_type=None):
    """Emit categorical lessons from one saved rec. REFUSES (returns []) when the characterization
    is missing/error/raw (histogram-blind guard) — a run we couldn't diagnose teaches nothing
    transferable. Collapses tool names to FAMILIES; a per-family median true delta decides
    method_helped vs method_hurt."""
    aa = rec.get("agent_analysis", {}) or {}
    ev = {"domain_shift": aa.get("domain_shift"), "model_info": aa.get("model_info")}
    ch = get_characterization(ev)
    if not ch:
        return []
    asmt = get_model_assessment(ev)
    tt = task_type or _task_type_of(rec)
    cond = {"task_type": tt, "shift_type": str(ch.get("shift_type", "unknown")).lower(),
            "severity": str(ch.get("severity", "unknown")).lower(),
            "normalization": norm_token(asmt)}
    audit = rec.get("oracle_audit", {}) or {}
    src_true = _true_metric(audit.get("source", {}))
    if src_true is None:
        return []
    fam_deltas = {}
    for name, a in audit.items():
        if name == "source":
            continue
        t = _true_metric(a)
        if t is None:
            continue
        fam = tool_family(name)
        if fam == "other":            # postproc/geometry/finetune — not a shift-family lesson
            continue
        fam_deltas.setdefault(fam, []).append(round(t - src_true, 4))
    out = []
    for fam, deltas in sorted(fam_deltas.items()):
        md = statistics.median(deltas)
        if md > _HELP_EPS:
            kind, verb = "method_helped", "helped"
        elif md < -_HELP_EPS:
            kind, verb = "method_hurt", "hurt"
        else:
            continue
        out.append({
            # id keyed on family ONLY (not kind) so conflicting-sign runs ACCUMULATE into one
            # lesson and collapse to MIXED (design §6.3), instead of forming two rival lessons.
            "id": f"{tt}|{cond['shift_type']}|{cond['severity']}|{fam}",
            "condition": cond, "kind": kind,
            "claim": f"{fam} family {verb} on {cond['shift_type']}/{cond['severity']} {tt} "
                     f"(median true delta {md:+.2f})",
            "evidence_summary": {"n_runs": 1, "median_true_delta": round(md, 3),
                                 "deltas": deltas, "mixed": False},
            "confidence": "weak", "provenance": [str(rec.get("task", "?"))],
        })
    return out


def _confidence(n_runs, mixed):
    if mixed:
        return "weak"                 # conflicting outcomes never reach the retrieval gate
    return "strong" if n_runs >= 3 else ("moderate" if n_runs >= 2 else "weak")


def merge_lesson(by_id, lesson):
    """Accumulate a mined lesson into the store (keyed by id). Tracks all deltas → recomputes
    median; conflicting sign across runs → MIXED → weak (suppressed). Bumps confidence by n_runs."""
    ex = by_id.get(lesson["id"])
    if ex is None:
        by_id[lesson["id"]] = json.loads(json.dumps(lesson))    # deep copy
        return by_id[lesson["id"]]
    ds = list(ex["evidence_summary"].get("deltas", [])) + list(lesson["evidence_summary"].get("deltas", []))
    n = ex["evidence_summary"].get("n_runs", 1) + lesson["evidence_summary"].get("n_runs", 1)
    signs = {(1 if d > _HELP_EPS else (-1 if d < -_HELP_EPS else 0)) for d in ds}
    mixed = (1 in signs and -1 in signs)
    md = statistics.median(ds)
    ex["evidence_summary"] = {"n_runs": n, "median_true_delta": round(md, 3),
                              "deltas": ds, "mixed": mixed}
    ex["confidence"] = _confidence(n, mixed)
    ex["kind"] = "mixed" if mixed else ("method_helped" if md > _HELP_EPS else "method_hurt")
    ex["provenance"] = list(dict.fromkeys(ex.get("provenance", []) + lesson.get("provenance", [])))[:12]
    # rebuild the claim's "<family> family <verb> on <shift>/<sev> <task>" head from the merged kind
    head = ex["claim"].split(" (median")[0].split(" (MIXED")[0]
    if mixed:
        # swap the verb to a neutral MIXED phrasing
        for v in (" helped ", " hurt "):
            head = head.replace(v, " had MIXED outcomes ")
        ex["claim"] = head + " (MIXED — conflicting outcomes across runs)"
    else:
        ex["claim"] = f"{head} (median true delta {md:+.2f}, n={n})"
    return ex


def _default_store_path():
    root = os.environ.get("MEDADAPT_ROOT") or os.environ.get("PROJECT_ROOT") or "."
    return os.path.join(root, "experiments", "lessons", "lessons.jsonl")


def maybe_retrieve(shift_ev, model_ev, task_type, store_path=None):
    """Gated retrieve for the adapter hook. Returns {} (byte-inert) unless LESSONS is enabled
    AND both diagnostics are present AND a fingerprint resolves AND a >=moderate lesson matches.
    Never raises — lessons are advisory."""
    if os.environ.get("LESSONS", "0") == "0":
        return {}
    try:
        fp = fingerprint_of({"domain_shift": shift_ev, "model_info": model_ev}, task_type)
        if not fp:
            return {}
        return lessons_evidence(load_store(store_path or _default_store_path()), fp)
    except Exception:
        return {}


# ── store I/O + the runner-side update ───────────────────────────────────────────────
def load_store(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return out


def update_store_with_rec(path, rec, task_type=None):
    """Idempotent-by-id mine+merge of ONE rec into the on-disk store. Pure post-hoc; never raises
    (lessons are advisory — a mining failure must not break a finished run)."""
    try:
        store = load_store(path)
        by_id = {l["id"]: l for l in store if isinstance(l, dict) and "id" in l}
        for lesson in mine_lessons(rec, task_type=task_type):
            merge_lesson(by_id, lesson)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        merged = list(by_id.values())
        with open(path, "w") as f:
            for l in merged:
                f.write(json.dumps(l) + "\n")
        md = os.path.splitext(path)[0] + ".md"
        with open(md, "w") as f:
            f.write("# MedAdaptAgent lessons (categorical, advisory)\n\n")
            for l in sorted(merged, key=lambda x: -_CONF_ORDER.get(x.get("confidence"), 0)):
                f.write(f"- **[{l.get('confidence')}]** {l.get('claim')}  "
                        f"_(n={l['evidence_summary'].get('n_runs')}, {l.get('kind')})_\n")
        return merged
    except Exception:
        return load_store(path)
