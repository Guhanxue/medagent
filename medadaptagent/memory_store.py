"""Persistent memory store for the Experiment Manager.

After each Manager run, append a fingerprinted record of (Agent 1
diagnosis, Agent 2 assessment, tools delivered, verdict, delta) to a
JSON file. Before each new run, query the store for K most-similar
past records and pass them to Agent 3 as a prior.

This is the "learn from past runs" piece of the AutoResearch sandbox.
Two datasets that look similar through Agents 1+2 should see each
other's outcomes as evidence — without hardcoding any per-dataset
rules.

Storage format: a flat list of records in `memory.json`. Each record
is small (<1KB) so the file stays human-readable; we don't expect
more than O(100) records before the next architectural review.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


DEFAULT_MEMORY_PATH = "experiments/agent_sandbox/memory.json"


@dataclass
class MemoryRecord:
    """One row in the memory store."""
    timestamp: float
    dataset: str  # for debugging only — not used for matching
    arch: str
    fingerprint: Dict[str, Any]
    tools_winner: List[str]
    verdict: str
    delta: Optional[float]
    source_metric: Optional[float] = None  # generic (Dice for seg, AUROC/Acc for cls)
    delivered_metric: Optional[float] = None
    metric_name: Optional[str] = None  # "dice" | "accuracy" | "auroc"
    n_iterations: int = 0
    recommendation: str = ""
    # Legacy fields kept for back-compat with old memory.json entries
    source_dice: Optional[float] = None
    delivered_dice: Optional[float] = None


def build_fingerprint(domain_analysis: Dict, model_assessment: Dict,
                        model_info: Dict,
                        task_kind: Optional[str] = None,
                        metric_name: Optional[str] = None
                        ) -> Dict[str, Any]:
    """Compact summary of (task type, Agent 1, Agent 2, model arch)
    suitable for similarity matching. Stable across runs of the same
    dataset.

    `task_kind` is "seg" or "cls" — used to prevent cross-pollination
    between segmentation and classification memory entries.
    `metric_name` is the metric the run optimized (auroc / accuracy /
    dice) — useful to compare a new AUROC run against past AUROC runs
    rather than mixed."""
    da = domain_analysis or {}
    ma = model_assessment or {}
    mi = model_info or {}

    # Pull the categorical / scalar fields we want to match on.
    modality = str(da.get("modality", "unknown"))[:32]
    shift_level = da.get("shift_level")
    if not isinstance(shift_level, (int, float)):
        shift_level = None
    severity = ma.get("severity_score")
    if not isinstance(severity, (int, float)):
        severity = None
    sq = ma.get("source_quality_score")
    if not isinstance(sq, (int, float)):
        sq = None
    failure_type = str(ma.get("failure_type", "unknown"))[:48]

    # Dominant shift factor (e.g., "field-of-view shift", "intensity",
    # "cross-modality") — Agent 1's specific diagnosis. Critical for
    # matching similar adaptation strategies.
    dominant_shift = str(da.get("dominant_shift_factor", "unknown"))[:64].lower()

    # Foreground prior range — captures "small lesion" vs "large organ"
    expected_min, expected_max = None, None
    priors = (da.get("foreground_priors") or {})
    ea = priors.get("expected_area_fraction")
    if isinstance(ea, (list, tuple)) and len(ea) >= 2:
        expected_min, expected_max = ea[0], ea[1]

    norm_type = str(mi.get("norm_type", "unknown"))[:24]

    return {
        "task_kind": task_kind or "unknown",
        "metric_name": metric_name or "unknown",
        "modality": modality,
        "shift_level": shift_level,
        "severity": severity,
        "source_quality": sq,
        "failure_type": failure_type,
        "dominant_shift": dominant_shift,
        "expected_min": expected_min,
        "expected_max": expected_max,
        "norm_type": norm_type,
    }


def fingerprint_distance(a: Dict, b: Dict) -> float:
    """Lower = more similar. Scaled so a sane top-K query produces
    meaningful neighbors:

      categorical mismatches (modality, failure_type, norm_type) cost
        a lot (each adds 10);
      shift_level / severity difference adds the absolute difference
        (each in [0,1]);
      source_quality difference adds half-weight;
      structure-size mismatch (expected_min) adds 0.5x its log-ratio.
    """
    d = 0.0
    # task_kind mismatch is a hard filter — seg memory shouldn't match
    # cls runs regardless of other signals (or vice versa).
    if a.get("task_kind") and b.get("task_kind") \
        and a.get("task_kind") != b.get("task_kind"):
        d += 100.0  # effectively excluded
    for cat_key, weight in [("modality", 10), ("failure_type", 10),
                              ("dominant_shift", 8), ("norm_type", 5)]:
        if a.get(cat_key) != b.get(cat_key):
            d += weight
    for num_key, weight in [("shift_level", 1.0), ("severity", 1.0),
                              ("source_quality", 0.5)]:
        va = a.get(num_key); vb = b.get(num_key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d += weight * abs(float(va) - float(vb))
    # Structure size — log-scale because 1% vs 50% is way more
    # meaningful than 25% vs 26%.
    va = a.get("expected_min"); vb = b.get("expected_min")
    if (isinstance(va, (int, float)) and isinstance(vb, (int, float))
            and va > 0 and vb > 0):
        import math
        d += 0.5 * abs(math.log(va) - math.log(vb))
    return d


def load_memory(path: str = DEFAULT_MEMORY_PATH) -> List[MemoryRecord]:
    """Load all records from disk. Returns [] if file doesn't exist."""
    if not os.path.exists(path):
        return []
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for r in raw:
        try:
            out.append(MemoryRecord(**r))
        except TypeError:
            continue
    return out


def save_memory(records: List[MemoryRecord],
                path: str = DEFAULT_MEMORY_PATH) -> None:
    """Persist records to disk. Atomic write via temp+rename."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    os.replace(tmp, path)


def append_record(record: MemoryRecord,
                    path: str = DEFAULT_MEMORY_PATH) -> None:
    """Append one record to the store. Idempotent on (dataset, arch)
    in the sense that we keep all entries — useful for tracking
    run-to-run variance."""
    records = load_memory(path)
    records.append(record)
    save_memory(records, path)


def query_similar(fingerprint: Dict, current_dataset: Optional[str] = None,
                    k: int = 3, path: str = DEFAULT_MEMORY_PATH,
                    min_iter_count: int = 1) -> List[MemoryRecord]:
    """Return up to k records most similar to `fingerprint`, sorted
    by ascending distance. Excludes records for the same dataset to
    avoid trivial self-matches. Records with `n_iterations < min_iter_count`
    are filtered (avoids matching empty/failed runs)."""
    records = load_memory(path)
    candidates = [
        (fingerprint_distance(fingerprint, r.fingerprint), r)
        for r in records
        if (r.dataset != current_dataset and r.n_iterations >= min_iter_count)
    ]
    candidates.sort(key=lambda x: x[0])
    return [r for _, r in candidates[:k]]


def render_memory_context(records: List[MemoryRecord]) -> str:
    """Render top-K matches as a markdown block Agent 3 can read."""
    if not records:
        return ""
    lines = [
        "## Memory: outcomes on similar past datasets",
        "",
        "These are the closest matches in the persistent memory store, "
        "by Agent 1/2 fingerprint similarity. Use as evidence-weighted "
        "prior — NOT a hardcoded rule. Datasets that looked similar "
        "had these outcomes:",
        "",
    ]
    for r in records:
        delta_s = (f"delta={r.delta:+.4f}" if r.delta is not None
                   else "delta=n/a (no labels)")
        lines.append(
            f"  - past dataset={r.dataset} (arch={r.arch}, "
            f"modality={r.fingerprint.get('modality')}, "
            f"failure={r.fingerprint.get('failure_type')}, "
            f"severity={r.fingerprint.get('severity')}): "
            f"delivered tools={r.tools_winner}, verdict={r.verdict}, "
            f"{delta_s}, recommendation={r.recommendation}")
    lines.append("")
    return "\n".join(lines)


def make_record(dataset: str, arch: str, fingerprint: Dict,
                  manager_result: Dict) -> MemoryRecord:
    """Convert a Manager's return dict into a memory record. Captures
    both the new `*_metric` names and the legacy `*_dice` aliases for
    back-compat with old memory.json viewers."""
    sm = manager_result.get("source_metric",
                                manager_result.get("source_dice"))
    dm = manager_result.get("delivered_metric",
                                manager_result.get("delivered_dice"))
    return MemoryRecord(
        timestamp=time.time(),
        dataset=dataset,
        arch=arch,
        fingerprint=fingerprint,
        tools_winner=list(manager_result.get("tools_applied") or []),
        verdict=str(manager_result.get("verdict", "")),
        delta=manager_result.get("delta"),
        source_metric=sm,
        delivered_metric=dm,
        metric_name=manager_result.get("metric_name"),
        source_dice=sm,   # legacy alias
        delivered_dice=dm,  # legacy alias
        n_iterations=int(manager_result.get("n_iterations", 0)),
        recommendation=str(manager_result.get("recommendation", "")),
    )
