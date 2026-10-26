"""Shared analysis tools for the harness — the Agent-1 (domain shift) and Agent-2
(model info / provenance) capabilities, as TOOLS the lead agent can call.

- analyze_domain_shift: MEASURES source-vs-target image statistics PER AXIS (intensity,
  contrast, histogram shape, dynamic range) and has the LLM characterize the shift fusing
  those numbers with SEMANTIC reasoning over the task description + model provenance. It
  decides shift_type (intensity / contrast / modality / sequence / resolution / FOV / mixed),
  a per-axis severity, a STRICT semantic_flag (modality/sequence changes only), and a
  reconciled overall severity. Discovered, not declared.
- inspect_model: introspects the checkpoint (architecture, normalization type, in/out,
  #params) and has the LLM read provenance + adaptation implications (which TTA applies).

Both degrade gracefully to the numeric/introspected facts if no LLM is available; the
LLM-less domain-shift path is tagged trust='numeric_derived_low' and NEVER auto-escalates to
'severe' (so a numeric fallback cannot trigger smart-escalation on its own).

Design: docs/HARNESS_DESIGN_DETAIL.md §4 (Component C). Histogram-blindness fix — the legacy
single free `severity` (which laundered T1->FLAIR to "moderate") is REPLACED by
severity = MAX over per-axis severity, escalated to >= severe IFF semantic_flag. mild<moderate
<severe is an ORDERING, not a task constant; the reconciliation is the ONLY new numeric op in
the LLM path.
"""
from __future__ import annotations
import json
import re
import numpy as np

# Severity is an ORDERING (not a task constant). Used only to MAX per-axis labels and to
# escalate on a semantic (modality/sequence) change. No quality threshold lives here.
_SEV_ORDER = {"mild": 0, "moderate": 1, "severe": 2}
_SEV_NAME = {0: "mild", 1: "moderate", 2: "severe"}

# The four measurable image-statistic axes the shift is decomposed over.
_AXES = ("intensity", "contrast", "histogram", "dynamic_range")


def _image_stats(imgs, n=48):
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs)), dtype=int)
    chunks = []
    for i in idx:
        a = np.asarray(imgs[int(i)], dtype=np.float32)
        if a.size and a.max() > 1.5:      # uint8-ish -> [0,1]
            a = a / 255.0
        chunks.append(a.ravel())
    v = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    hist, _ = np.histogram(v, bins=16, range=(0.0, 1.0), density=True)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
            "p1": round(float(np.percentile(v, 1)), 4),
            "p50": round(float(np.percentile(v, 50)), 4),
            "p99": round(float(np.percentile(v, 99)), 4),
            "hist16": [round(float(x), 3) for x in hist]}


def _axis_shift(ss, ts):
    """Decompose the source->target shift into per-axis magnitudes.

    Returns (per_axis_norm, per_axis_raw):
      per_axis_norm: each axis in [0,1] (normalization scales documented below) — fed to the
                     LLM as soft context and used (only) for the dominant-axis tie-break and the
                     LLM-less fallback bands. NOT a decision gate in the LLM path.
      per_axis_raw:  the raw measured deltas/ratios (kept for transparency in the report).
    The normalization scales are conservative "this-much-is-a-large-shift" anchors, not task
    constants: a 0.20 mean shift, a 2x/0x std ratio, a TV-distance of ~1.0, a 2x dynamic-range
    ratio each map to magnitude 1.0.
    """
    eps = 1e-6
    intensity = abs(ts["mean"] - ss["mean"])                          # mean delta on [0,1]
    contrast = abs((ts["std"] + eps) / (ss["std"] + eps) - 1.0)       # std-ratio deviation
    histogram = float(np.abs(np.array(ts["hist16"]) - np.array(ss["hist16"])).sum() * (1.0 / 16))
    s_rng = (ss["p99"] - ss["p1"]); t_rng = (ts["p99"] - ts["p1"])
    dyn = abs((t_rng + eps) / (s_rng + eps) - 1.0)                    # dynamic-range ratio dev
    raw = {"intensity": round(intensity, 4), "contrast": round(contrast, 4),
           "histogram": round(histogram, 4), "dynamic_range": round(dyn, 4)}
    norm = {"intensity": min(1.0, intensity / 0.20),
            "contrast": min(1.0, contrast / 1.0),
            "histogram": min(1.0, histogram / 1.0),
            "dynamic_range": min(1.0, dyn / 1.0)}
    return {a: round(float(norm[a]), 3) for a in _AXES}, raw


def _shared_introspection(model):
    """SINGLE source of model introspection — shared by inspect_model and analyze_domain_shift
    so the two capabilities can never disagree on a mixed-norm model."""
    import torch.nn as nn
    bn = sum(1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    inn = sum(1 for m in model.modules() if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)))
    ln = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    gn = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    in_ch = out_cls = None
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)) and in_ch is None:
            in_ch = m.in_channels
    for m in model.modules():
        if isinstance(m, nn.Linear):
            out_cls = m.out_features
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            out_cls = m.out_channels
    return {"class": model.__class__.__name__, "batchnorm": bn, "instancenorm": inn,
            "layernorm": ln, "groupnorm": gn, "in_channels": in_ch, "out_classes": out_cls,
            "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}


def _parse_json(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {"raw": (txt or "")[:500]}
    except Exception:
        return {"raw": (txt or "")[:500]}


def _max_sev(axis_severity):
    """MAX over per-axis severity labels (the ONLY new numeric op). Empty -> neutral moderate."""
    vals = [_SEV_ORDER.get(str(v).lower(), 1) for v in (axis_severity or {}).values()]
    return _SEV_NAME[max(vals)] if vals else "moderate"


def _reconcile_severity(axis_severity, semantic_flag):
    """Legacy severity = MAX over axes, escalated to >= severe iff a semantic (modality/
    sequence) change. Can only ESCALATE — kills the T1->FLAIR "moderate" laundering."""
    base = _max_sev(axis_severity)
    return "severe" if semantic_flag else base


def _dominant_axis(axis_severity, per_axis):
    """Worst axis by severity ordering, tie-broken by raw normalized magnitude."""
    if axis_severity:
        return max(axis_severity,
                   key=lambda a: (_SEV_ORDER.get(str(axis_severity[a]).lower(), 1),
                                  per_axis.get(a, 0.0)))
    return max(per_axis, key=per_axis.get) if per_axis else "intensity"


def _band_fallback(mag):
    """LLM-less degradation bands. Documented NEW numbers, low-trust, and capped at 'moderate'
    so a numeric fallback can NEVER auto-escalate to severe -> recommend-alternative (§4.4)."""
    return "moderate" if mag >= 0.5 else "mild"


def _finalize_characterization(ch, per_axis):
    """Reconcile the LLM's per-axis judgment into the legacy+new characterization, legacy keys
    FIRST (600-char prompt-eviction stability)."""
    raw_as = ch.get("axis_severity")
    axis_sev = {}
    if isinstance(raw_as, dict):
        axis_sev = {str(k): str(v).lower() for k, v in raw_as.items()
                    if str(v).lower() in _SEV_ORDER}
    semantic_flag = bool(ch.get("semantic_flag", False))
    severity = _reconcile_severity(axis_sev, semantic_flag)
    dominant = _dominant_axis(axis_sev, per_axis)
    kd = ch.get("key_differences", [])
    if isinstance(kd, str):
        kd = [kd]
    kd = [str(x)[:80] for x in (kd or [])][:4]
    return {                                                  # legacy keys FIRST
        "shift_type": str(ch.get("shift_type", "mixed")).lower(),
        "severity": severity,
        "key_differences": kd,
        "training_free_outlook": str(ch.get("training_free_outlook", ""))[:160],
        "axis_severity": axis_sev,                            # NEW keys LAST
        "dominant_axis": dominant,
        "semantic_flag": semantic_flag,
        "why": str(ch.get("why", ""))[:200],
        "trust": "llm",
    }


def _fallback_characterization(per_axis, per_axis_raw, ch):
    """LLM unavailable / unparseable -> derive from numeric axes only. NEVER fires semantic_flag
    and NEVER reports 'severe' (capped bands), so smart-escalation cannot trip on numbers
    alone. Tagged trust='numeric_derived_low'."""
    axis_sev = {a: _band_fallback(per_axis.get(a, 0.0)) for a in _AXES}
    dominant = _dominant_axis(axis_sev, per_axis)
    out = {
        "shift_type": "mixed",
        "severity": _max_sev(axis_sev),                       # <= moderate by construction
        "key_differences": [f"{dominant} dominant (numeric-only, no LLM)"],
        "training_free_outlook": "LLM unavailable; numeric-derived estimate only.",
        "axis_severity": axis_sev,
        "dominant_axis": dominant,
        "semantic_flag": False,
        "why": "numeric-derived (no LLM semantic reasoning)",
        "trust": "numeric_derived_low",
    }
    err = None
    if isinstance(ch, dict):
        err = ch.get("error") or ch.get("raw")
    if err:
        out["_llm_error"] = str(err)[:160]
    return out


def analyze_domain_shift(source_imgs, target_imgs, llm, task_desc="", model=None):
    """Agent-1 v2: measure source<->target shift PER AXIS, LLM fuses the numbers with SEMANTIC
    reasoning over the deployment description + (optional) model provenance.

    `model` (the source checkpoint) is OPTIONAL provenance: if given, its architecture/norm/IO
    inform whether a modality/sequence change is plausible. Same `domain_shift` evidence key.
    """
    ss = _image_stats(source_imgs); ts = _image_stats(target_imgs)
    per_axis, per_axis_raw = _axis_shift(ss, ts)
    numeric = {
        "source_stats": ss, "target_stats": ts,
        "intensity_delta": per_axis_raw["intensity"],
        "contrast_ratio": round(ts["std"] / (ss["std"] + 1e-6), 3),
        "hist_L1": per_axis_raw["histogram"],
        "per_axis": per_axis, "per_axis_raw": per_axis_raw,
    }
    provenance = {}
    if model is not None:
        try:
            provenance = _shared_introspection(model)
        except Exception as e:
            provenance = {"error": str(e)[:160]}

    characterization = None
    if llm is not None:
        prov_line = (f"SOURCE model provenance (for plausibility): class={provenance.get('class')}, "
                     f"in_channels={provenance.get('in_channels')}, "
                     f"norm(bn/in/ln/gn)={provenance.get('batchnorm')}/{provenance.get('instancenorm')}/"
                     f"{provenance.get('layernorm')}/{provenance.get('groupnorm')}.\n"
                     if provenance and "error" not in provenance else "")
        prompt = (
            f"You are diagnosing a DOMAIN SHIFT for a medical-image model deployed at a new site.\n"
            f"Deployment context (read the source->target arrow for MODALITY/SEQUENCE cues): {task_desc}.\n"
            f"{prov_line}"
            f"SOURCE (training-domain) image stats: {ss}\n"
            f"TARGET (new-site) image stats: {ts}\n"
            f"Per-axis shift magnitudes in [0,1] (1.0 = large): {per_axis}.\n\n"
            f"Reason about WHAT changed, then respond ONLY as JSON with these keys:\n"
            f'  "shift_type": one of intensity | contrast | modality | sequence | resolution | FOV | mixed '
            f"(use 'modality' for CT<->MR / MR<->US etc.; 'sequence' for a different MR weighting "
            f"T1<->T2<->FLAIR<->DWI).\n"
            f'  "axis_severity": an object grading EACH of {list(_AXES)} as mild | moderate | severe. '
            f"Reserve 'severe' for a difference large enough to break the model on that axis ALONE; "
            f"same-modality scanner / vendor / site / field-strength variation is at most 'moderate' on every axis.\n"
            f'  "semantic_flag": true ONLY IF the source->target change alters tissue-contrast SEMANTICS '
            f"— i.e. a different imaging MODALITY (CT<->MR, MR<->US) or a different MR SEQUENCE/weighting "
            f"(T1<->T2<->FLAIR<->DWI). Set it FALSE for same-modality, same-sequence variation such as a "
            f"different scanner vendor, site, field strength, reconstruction, or plain intensity/contrast drift "
            f"— those are NOT semantic changes.\n"
            f'  "key_differences": list of short strings.\n'
            f'  "training_free_outlook": can BN-recalibration / entropy-min plausibly fix it? one line.\n'
            f'  "why": one short sentence justifying shift_type + semantic_flag.')
        try:
            parsed = _parse_json(llm.chat(prompt))
        except Exception as e:
            parsed = {"error": str(e)[:200]}
        if isinstance(parsed, dict) and "error" not in parsed and "raw" not in parsed:
            characterization = _finalize_characterization(parsed, per_axis)
        else:
            characterization = _fallback_characterization(per_axis, per_axis_raw, parsed)
    else:
        characterization = _fallback_characterization(per_axis, per_axis_raw, None)

    # characterization FIRST so its legacy keys lead the serialized domain_shift blob (the
    # 600-char LATEST-EVIDENCE dump truncates; load-bearing keys must survive). numeric +
    # provenance follow.
    return {"characterization": characterization, "numeric": numeric, "provenance": provenance}


def inspect_model(model, llm, task_desc=""):
    """Agent-2 + S1: introspect architecture/normalization/IO, LLM reads provenance."""
    intros = _shared_introspection(model)
    out = {"introspection": intros}
    if llm is not None:
        prompt = (
            f"Introspected model: {intros}. Task descriptor: {task_desc}.\n"
            f"Respond ONLY as JSON: "
            f'"model_family" (UNet | ResNet | ViT | other), '
            f'"normalization" (which test-time adaptation applies — BatchNorm => bn_reset/tent OK; '
            f"InstanceNorm/LayerNorm/GroupNorm => BN-based TTA is a NO-OP), "
            f'"likely_training_domain" (best guess of what it was trained on), '
            f'"adaptation_implications" (one line on which adaptations are worth trying).')
        try:
            out["assessment"] = _parse_json(llm.chat(prompt))
        except Exception as e:
            out["assessment"] = {"error": str(e)[:200]}
    return out
