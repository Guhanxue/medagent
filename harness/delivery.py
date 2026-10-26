"""Component F — delivery: the honest report (F1) + (F2, later) the runnable artifact.

build_report ENRICHES the agent's already-honesty-gated terminal report with structured,
clinician-facing fields (what shifted, what was tried, what's delivered, how trustworthy,
what to do next). It PRESERVES the committed trust_level / confidence / summary — it never
flips them (so amos's label_verified escalate stays label_verified). Pure + deterministic:
reads only state.evidence / state fields already in scope at the core terminal hook; never
raises (the core hook also wraps it in try/except and falls back to the raw report).

Design: docs/HARNESS_DESIGN_DETAIL.md §7.
"""
from __future__ import annotations

try:                                  # robust to repo-root vs harness/-on-path
    from harness.evidence_keys import (get_characterization, get_model_assessment,
                                       get_introspection)
except ImportError:
    from evidence_keys import get_characterization, get_model_assessment, get_introspection


def _r(x, n=4):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _decision_path(decision, state):
    """Coarse A/B/C/D label for the report (A=deploy/keep good source, B=label-free recover,
    C=few-shot finetune, D=escalate). Derived from what actually happened, not declared."""
    used = int(getattr(state, "k_labels_used", 0) or 0)
    if decision == "escalate":
        return "D (escalate — could not deliver a verified improvement within budget)"
    if decision in ("keep_source",):
        return "A (keep source — no adaptation safely beat it)"
    # ship
    if used > 0:
        return "C (few-shot: labels verified / fine-tuned the delivered model)"
    return "B (label-free adaptation, delivered as an estimate)"


def build_report(*, decision, task, state, chosen_name, chosen_metric, base_report=None):
    """Enrich `base_report` (the agent's trust_level/confidence/summary) with the structured
    delivery report. Returns a NEW dict; never mutates inputs; never raises."""
    ev = getattr(state, "evidence", None) or {}
    base = dict(base_report or {})
    ch = get_characterization(ev)
    asmt = get_model_assessment(ev)
    intro = get_introspection(ev)
    ld = ev.get("labeled_dice") if isinstance(ev.get("labeled_dice"), dict) else {}
    used = int(getattr(state, "k_labels_used", 0) or 0)

    verified = bool(used) and chosen_name in (ld or {})
    delivered_metric = (ld or {}).get(chosen_name)
    if delivered_metric is None:
        delivered_metric = chosen_metric
    sota = ev.get("expected_metric")
    usable = ev.get("usable_floor")
    reaches = (None if (_r(delivered_metric) is None or usable is None)
               else bool(_r(delivered_metric) >= float(usable)))

    flags = []
    tl = base.get("trust_level", "label_free_estimate")
    if tl == "label_free_estimate":
        flags.append("delivered metric is a LABEL-FREE estimate — NOT guaranteed >= source; "
                     "provide a few labels to verify")
    rec = ev.get("recommendation") if isinstance(ev.get("recommendation"), dict) else None
    if decision == "escalate" and not rec:
        flags.append("no grounded alternative model was found to recommend")
    if reaches is False:
        flags.append(f"delivered {task.metric_name} below the looked-up usable floor "
                     f"({_r(delivered_metric)} < {usable})")
    if ch.get("trust") == "numeric_derived_low":
        flags.append("shift diagnosis is numeric-only (LLM unavailable) — low confidence")

    bud = getattr(state, "budget_spent", None)
    budget = bud.as_dict() if hasattr(bud, "as_dict") else {}

    return {
        # --- preserved, committed, honesty-gated fields ---
        "trust_level": tl,
        "confidence": base.get("confidence"),
        "summary": base.get("summary", ""),
        # --- enrichment ---
        "decision": decision,
        "delivered_model": chosen_name,
        "metric_name": task.metric_name,
        "diagnosed_shift": {
            "shift_type": ch.get("shift_type"), "severity": ch.get("severity"),
            "semantic_flag": ch.get("semantic_flag"), "dominant_axis": ch.get("dominant_axis"),
            "key_differences": ch.get("key_differences"),
            "numeric": (ev.get("domain_shift", {}) or {}).get("numeric", {}).get("per_axis")
            if isinstance(ev.get("domain_shift"), dict) else None,
            "model": {"normalization": asmt.get("normalization"),
                      "family": asmt.get("model_family"), "class": intro.get("class"),
                      "in_channels": intro.get("in_channels")},
            "failure_mode": ev.get("failure_mode"),
        },
        "action_taken": {
            "decision_path": _decision_path(decision, state),
            "candidates_tried": [c.get("method") for c in (getattr(state, "candidates_log", []) or [])],
            "chosen": chosen_name, "k_labels_used": used,
            "budget_spent": budget, "steps": int(getattr(state, "step", 0) or 0),
        },
        "expected_performance": {
            "delivered_metric": _r(delivered_metric),
            "verification": "label_verified" if verified else "label_free_proxy",
            "field_sota": sota, "usable_floor": usable,
            "sota_rationale": ev.get("perf_rationale"), "reaches_usable": reaches,
        },
        "scaling_plan": ev.get("scaling_law") if isinstance(ev.get("scaling_law"), dict) else None,
        "recommendations": rec,
        "retrieved_lessons": ev.get("retrieved_lessons"),
        "honesty_flags": flags,
    }


def render_report_md(report) -> str:
    """Render the enriched report as a clinician-facing markdown brief. Defensive — never raises."""
    if not isinstance(report, dict):
        return "# Deployment report\n\n(no report)\n"
    g = report.get
    ds = g("diagnosed_shift", {}) or {}
    at = g("action_taken", {}) or {}
    ep = g("expected_performance", {}) or {}
    L = []
    L.append(f"# MedAdaptAgent deployment report — {g('decision', '?').upper()}")
    L.append(f"\n**Delivered:** `{g('delivered_model', '?')}`  ·  "
             f"**trust:** `{g('trust_level', '?')}`  ·  "
             f"**confidence:** {g('confidence')}")
    if g("summary"):
        L.append(f"\n> {g('summary')}")
    L.append("\n## Diagnosed shift")
    L.append(f"- type **{ds.get('shift_type')}** / severity **{ds.get('severity')}**"
             f"  (semantic: {ds.get('semantic_flag')}, dominant axis: {ds.get('dominant_axis')})")
    if ds.get("key_differences"):
        L.append(f"- key differences: {ds.get('key_differences')}")
    m = ds.get("model", {}) or {}
    L.append(f"- source model: {m.get('class')} / {m.get('family')} / norm={m.get('normalization')}")
    L.append("\n## What was done")
    L.append(f"- path: {at.get('decision_path')}")
    L.append(f"- candidates tried: {at.get('candidates_tried')}")
    L.append(f"- chosen: **{at.get('chosen')}**  ·  labels used: {at.get('k_labels_used')}  "
             f"·  steps: {at.get('steps')}")
    L.append("\n## Expected performance")
    L.append(f"- delivered {g('metric_name', 'metric')}: **{ep.get('delivered_metric')}** "
             f"({ep.get('verification')})")
    L.append(f"- field SOTA ≈ {ep.get('field_sota')}  ·  usable floor ≈ {ep.get('usable_floor')}  "
             f"·  reaches usable: {ep.get('reaches_usable')}")
    if ep.get("sota_rationale"):
        L.append(f"- basis: {ep.get('sota_rationale')}")
    sp = g("scaling_plan")
    if isinstance(sp, dict) and sp:
        fit = sp.get("fit", {}) if isinstance(sp.get("fit"), dict) else {}
        L.append("\n## Scaling law (how many labels)")
        L.append(f"- {fit.get('form')}: {fit.get('formula', '')}")
        if sp.get("needed_for_usable") is not None:
            L.append(f"- annotate ≈{sp.get('needed_for_usable')} labels to reach the usable floor")
    rec = g("recommendations")
    if isinstance(rec, dict) and rec:
        L.append("\n## Recommended alternative")
        L.append(f"- verdict: {rec.get('verdict')}")
        if rec.get("recommended_models"):
            L.append(f"- models: {rec.get('recommended_models')}")
        if rec.get("finetune_labels_estimate") is not None:
            L.append(f"- est. labels to fine-tune an alternative: {rec.get('finetune_labels_estimate')}")
    rl = g("retrieved_lessons")
    if rl:
        L.append("\n## Prior lessons (advisory)")
        for l in (rl if isinstance(rl, list) else []):
            L.append(f"- [{l.get('confidence')}] {l.get('claim')}")
    flags = g("honesty_flags") or []
    if flags:
        L.append("\n## ⚠ Honesty flags")
        for f in flags:
            L.append(f"- {f}")
    L.append("")
    return "\n".join(L)


# ============================ F2 — runnable delivery artifact ============================
import os
import json
import pickle


class ArtifactManifest:
    def __init__(self, dir, manifest):
        self.dir = dir
        self.manifest = manifest


def _env_txt():
    import sys
    out = [f"python={sys.version.split()[0]}"]
    for mod in ("torch", "numpy", "PIL"):
        try:
            m = __import__(mod)
            out.append(f"{mod}={getattr(m, '__version__', '?')}")
        except Exception:
            pass
    return "\n".join(out) + "\n"


def _picklable_notes(notes):
    """Keep only picklable entries of ctx.notes (numpy arrays ARE picklable; closures are not)."""
    out = {}
    for k, v in (notes or {}).items():
        try:
            pickle.dumps(v)
            out[k] = v
        except Exception:
            out[k] = f"<unpicklable {type(v).__name__}>"
    return out


_PREDICT_TEMPLATE = '''#!/usr/bin/env python
"""Auto-generated deployable predictor (MedAdaptAgent).

Loads the delivered model and returns per-image class masks (segmentation), replaying the
EXACT preprocessing the harness selected. See report.md for WHAT was delivered and WHY, and
the honesty flags for whether it is label-verified or a label-free estimate.

  python predict.py <input.npy> [out.npy]      # input.npy: HxW or N x HxW
Set MEDADAPT_ROOT to the project root if this artifact is moved.
"""
import sys, os, json
import numpy as np
import torch
ART = os.path.dirname(os.path.abspath(__file__))
PR = os.environ.get("MEDADAPT_ROOT", "__PR__")
sys.path.insert(0, PR)
from scripts._source_loader import load_source_model
from scripts.run_baseline_tta import _to_tensor
CFG = json.load(open(os.path.join(ART, "pipeline.json")))


def load_model(device="cuda:0"):
    """Rebuild a 2D architecture from delivered weights; 3D may need its source bundle."""
    if CFG.get("task_type") == "classification":
        from harness.cls_adapter import load_cls_model
        model = load_cls_model(os.path.join(ART, "model.pt"), CFG["arch"], CFG["nc"], device)
    else:
        checkpoint = os.path.join(ART, "model.pt") if CFG.get("dims", "2D").lower() == "2d" else CFG["ckpt"]
        model = load_source_model(checkpoint, CFG["nc"], device,
                                  arch=CFG.get("arch", "unet"), project_root=PR)
    model.load_state_dict(torch.load(os.path.join(ART, "model.pt"), map_location=device))
    model.eval()
    return model


def _replay_ctx(model, device):
    """Restore the fitted inference functions without re-running adaptation."""
    import cloudpickle
    from medadaptagent.tools import ToolContext
    with open(os.path.join(ART, "inference_state.pkl"), "rb") as handle:
        functions = cloudpickle.load(handle)
    ctx = ToolContext(source_model=model, current_model=model, target_images=[],
                      device=device, num_classes=CFG["nc"])
    ctx.preprocess_fn = functions["preprocess_fn"]
    ctx.postprocess_fn = functions["postprocess_fn"]
    ctx.inference_fn = functions["inference_fn"]
    ctx.tools_applied = CFG.get("tools_applied", [])
    return ctx


def predict(images, device="cuda:0"):
    """Return segmentation masks or classification probability vectors for image arrays."""
    model = load_model(device)
    ctx = _replay_ctx(model, device) if CFG.get("kind") == "ctx" else None
    if CFG.get("task_type") == "classification":
        from harness.cls_adapter import _cls_tensor
        if ctx is not None:
            from harness.registry_bridge import _ctx_probs
            return list(_ctx_probs(ctx, images, _cls_tensor))
        with torch.no_grad():
            return [model(_cls_tensor(im, device)).softmax(1)[0].cpu().numpy() for im in images]
    if ctx is not None:
        from harness.registry_bridge import run_inference
        return [np.asarray(run_inference(ctx, im, _to_tensor)) for im in images]
    with torch.no_grad():
        return [model(_to_tensor(im, device)).argmax(1).cpu().numpy()[0] for im in images]


if __name__ == "__main__":
    arr = np.load(sys.argv[1], allow_pickle=False)
    imgs = list(arr) if getattr(arr, "ndim", 2) > 2 else [arr]
    masks = predict(imgs, os.environ.get("DEVICE", "cuda:0"))
    dest = sys.argv[2] if len(sys.argv) > 2 else "predicted_masks.npy"
    np.save(dest, np.stack(masks))
    print("wrote", dest, np.stack(masks).shape)
'''


def _predict_py(project_root):
    return _PREDICT_TEMPLATE.replace("__PR__", str(project_root))


def package_artifact(*, out, task, cfg, dest_dir, to_tensor=None, project_root):
    """Write a runnable delivery folder for `out` (a core.Outcome) under
    <dest_dir>/<cfg name>/: model.pt (delivered weights) + predict.py + pipeline.json +
    report.json/md + structures.json + environment.txt (+ ctx_state.pkl for ctx pipelines).

    WEIGHTS deliveries (bn_reset/tent/finetune/source) round-trip exactly via model.pt; CTX
    (preprocessing-pipeline) deliveries also persist a binary sidecar of ctx.notes + the
    ordered tool list so predict.py can REPLAY them. Never raises catastrophically — returns a
    manifest with `weights_saved` False if the state_dict could not be written.
    """
    import torch
    try:
        from harness.registry_bridge import CtxCandidate
    except ImportError:
        from registry_bridge import CtxCandidate

    adir = os.path.join(dest_dir, cfg["name"])
    os.makedirs(adir, exist_ok=True)
    cand = getattr(out, "delivered", None)
    arch = cfg.get("arch", "unet")
    is_ctx = isinstance(cand, CtxCandidate)
    tools_applied = []
    if is_ctx:
        model = cand.ctx.current_model
        tools_applied = list(cand.ctx.tools_applied)
        import cloudpickle
        with open(os.path.join(adir, "inference_state.pkl"), "wb") as handle:
            cloudpickle.dump({name: getattr(cand.ctx, name) for name in
                              ("preprocess_fn", "postprocess_fn", "inference_fn")}, handle)
    else:
        model = cand

    weights_saved = False
    try:
        if model is not None:
            torch.save(model.state_dict(), os.path.join(adir, "model.pt"))
            weights_saved = True
    except Exception:
        raise RuntimeError("Could not save delivered model weights") from None
    if not weights_saved:
        raise RuntimeError("No model weights available for delivery")

    report = getattr(out, "report", {}) or {}
    manifest = {"task": cfg["name"], "decision": getattr(out, "decision", "?"),
                "delivered": report.get("delivered_model"),
                "kind": "ctx" if is_ctx else "weights", "arch": arch, "nc": cfg["nc"],
                "ckpt": cfg.get("ckpt"), "metric": task.metric_name, "task_type": task.type, "dims": task.dims,
                "structures": cfg.get("struct", {}), "tools_applied": tools_applied,
                "weights_saved": weights_saved}
    json.dump(manifest, open(os.path.join(adir, "pipeline.json"), "w"), indent=2, default=str)
    json.dump(report, open(os.path.join(adir, "report.json"), "w"), indent=2, default=str)
    open(os.path.join(adir, "report.md"), "w").write(render_report_md(report))
    json.dump(cfg.get("struct", {}), open(os.path.join(adir, "structures.json"), "w"), indent=2, default=str)
    open(os.path.join(adir, "environment.txt"), "w").write(_env_txt())
    open(os.path.join(adir, "predict.py"), "w").write(_predict_py(project_root))
    return ArtifactManifest(adir, manifest)
