"""Standalone tests for the harness core — no torch, no LLM, no GPU.

Proves (a) the four guardrail invariants fire (I1 safety, I2 budget, I3 honesty,
I4 trace), and (b) different TASKS assemble DIFFERENT toolboxes + DIFFERENT failure
analysis (the 'otherwise you are not smart' requirement)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core import (Budget, Tool, SafetyVerdict, Harness, build_harness, DECISIONS)

PASS = []; FAIL = []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name)


# ------------------------------------------------------------- common (task-blind) tools
def _deploy(st, **kw):   return {"deployment": "deployed"}
def _labels(st, K=0, **kw): return {"k_labels_used": K}
COMMON = [
    Tool("load_and_deploy", "load model + run on local data", _deploy, Budget(compute_minutes=1)),
    Tool("request_labels", "ask clinician for K labels", _labels, Budget(k_labels=1)),
    Tool("ship", "deliver adapted model", lambda st, **k: {}, terminal=True),
    Tool("keep_source", "keep source model", lambda st, **k: {}, terminal=True),
    Tool("escalate", "hand back, needs help", lambda st, **k: {}, terminal=True),
]

# --------------------------------------------------------------- two DIFFERENT task brains
class SegTask:
    type, dims, metric_name = "segmentation", "2D", "Dice"
    def toolbox(self):
        # seg-specific: mask-morphology diagnosis + entropy/BN TTA (tent APPLIES)
        return [
            Tool("diagnose_masks", "over/under-seg + boundary analysis",
                 lambda st, **k: {"evidence": {"failure": "under_seg"}}, Budget(compute_minutes=1)),
            Tool("tent", "entropy-min TTA (valid for seg)",
                 lambda st, **k: {"candidate": "seg_tent", "metric_estimate": 0.62,
                                  "evidence": {"vlm_ok": True}}, Budget(compute_minutes=2, n_trials=1)),
            Tool("bn_reset", "recalibrate BN stats",
                 lambda st, **k: {"candidate": "seg_bn", "metric_estimate": 0.70,
                                  "evidence": {"vlm_ok": True}}, Budget(compute_minutes=1, n_trials=1)),
        ]
    def safety_check(self, cand, src, ev):
        if ev.get("broken"): return SafetyVerdict(False, "VLM unanimous broken")
        if not ev.get("vlm_ok"): return SafetyVerdict(False, "no positive quality evidence")
        return SafetyVerdict(True)

class RegressionTask:
    type, dims, metric_name = "regression", "3D", "MAE"
    def toolbox(self):
        # regression-specific: residual/uncertainty diagnosis + recalibration.
        # NOTE: NO tent — entropy has no meaning for a scalar output.
        return [
            Tool("diagnose_residuals", "residual bias + uncertainty/ensemble-spread",
                 lambda st, **k: {"evidence": {"failure": "prediction_shift"}}, Budget(compute_minutes=1)),
            Tool("recalibrate", "shift/scale recalibration to target",
                 lambda st, **k: {"candidate": "reg_recal", "metric_estimate": 0.55,
                                  "evidence": {"uncertainty_ok": True}}, Budget(compute_minutes=2, n_trials=1)),
        ]
    def safety_check(self, cand, src, ev):
        if not ev.get("uncertainty_ok"): return SafetyVerdict(False, "uncertainty spike / shift>cap")
        return SafetyVerdict(True)


def policy_from(fn): return fn   # a policy is just view->action

# =========================================================================== TESTS
print("\n[T0] tasks assemble DIFFERENT toolboxes + different diagnosis")
seg_h = build_harness(lambda v: None, SegTask(), COMMON, Budget(compute_minutes=10, n_trials=5))
reg_h = build_harness(lambda v: None, RegressionTask(), COMMON, Budget(compute_minutes=10, n_trials=5))
seg_names, reg_names = set(seg_h.tools), set(reg_h.tools)
check("seg toolbox has tent + mask diagnosis", {"tent", "diagnose_masks"} <= seg_names)
check("regression toolbox has recalibrate + residual diagnosis", {"recalibrate", "diagnose_residuals"} <= reg_names)
check("tent is ABSENT from regression (entropy n/a)", "tent" not in reg_names)
check("diagnosis tool differs per task", "diagnose_masks" in seg_names and "diagnose_masks" not in reg_names)

print("\n[T1] happy path: deploy -> diagnose -> bn_reset -> ship (I1 pass, I3 ok, I4 trace)")
plan = iter([
    {"tool": "load_and_deploy", "args": {}},
    {"tool": "diagnose_masks", "args": {}},
    {"tool": "bn_reset", "args": {}},
    {"tool": "ship", "args": {"candidate": "seg_bn",
        "report": {"trust_level": "label_free_estimate", "confidence": 0.8,
                   "summary": "bn_reset recovered under-seg"}}},
])
out = build_harness(lambda v: next(plan), SegTask(), COMMON,
                    Budget(compute_minutes=10, n_trials=5)).run("adapt", "src")
check("decision == ship", out.decision == "ship")
check("delivered the adapted candidate", out.delivered == "seg_bn")
check("trace recorded every step (I4)", len(out.trace) >= 4)

print("\n[T2] I1 safety veto: ship a 'broken' candidate -> vetoed -> agent escalates")
def pol_veto(v):
    if v["guardrail_feedback"] and "VETOED" in v["guardrail_feedback"]:
        return {"tool": "escalate", "args": {"report": {"trust_level": "label_free_estimate",
                "confidence": 0.2, "summary": "could not reach safe quality"}}}
    if not v["deployed"]: return {"tool": "load_and_deploy", "args": {}}
    return {"tool": "ship", "args": {"candidate": "bad",
            "report": {"trust_level": "label_free_estimate", "confidence": 0.9, "summary": "x"}}}
# seg safety_check fails because evidence has no vlm_ok
out = build_harness(pol_veto, SegTask(), COMMON, Budget(compute_minutes=10, n_trials=5)).run("adapt", "src")
check("ship vetoed then decision == escalate", out.decision == "escalate")
check("safety_veto in trace", any(e["kind"] == "safety_veto" for e in out.trace))

print("\n[T3] I3 honesty: ship 'label_verified' with 0 labels -> rejected -> fix -> ship")
state = {"tries": 0}
def pol_honesty(v):
    if not v["deployed"]: return {"tool": "load_and_deploy", "args": {}}
    if v["guardrail_feedback"] and "honesty" in v["guardrail_feedback"]:
        return {"tool": "ship", "args": {"candidate": "seg_bn",
                "report": {"trust_level": "label_free_estimate", "confidence": 0.7, "summary": "ok"}}}
    # first run bn_reset to have evidence vlm_ok, then ship with a FALSE 'label_verified'
    if v["best_metric_estimate"] is None:
        return {"tool": "bn_reset", "args": {}}
    return {"tool": "ship", "args": {"candidate": "seg_bn",
            "report": {"trust_level": "label_verified", "confidence": 0.95, "summary": "claims verified"}}}
out = build_harness(pol_honesty, SegTask(), COMMON, Budget(compute_minutes=10, n_trials=5)).run("adapt", "src")
check("dishonest ship rejected then honest ship succeeds", out.decision == "ship")
check("honesty_rejected in trace", any(e["kind"] == "honesty_rejected" for e in out.trace))
check("final report is label_free_estimate", out.report["trust_level"] == "label_free_estimate")

print("\n[T4] I2 budget: over-budget adaptation refused -> agent escalates")
def pol_budget(v):
    if not v["deployed"]: return {"tool": "load_and_deploy", "args": {}}
    if v["guardrail_feedback"] and "budget" in v["guardrail_feedback"].lower():
        return {"tool": "escalate", "args": {"report": {"trust_level": "label_free_estimate",
                "confidence": 0.1, "summary": "out of budget"}}}
    return {"tool": "bn_reset", "args": {}}   # costs n_trials=1; cap has n_trials=0
out = build_harness(pol_budget, SegTask(), COMMON,
                    Budget(compute_minutes=10, n_trials=0)).run("adapt", "src")
check("over-budget tool refused then escalate", out.decision == "escalate")
check("budget_refused in trace", any(e["kind"] == "budget_refused" for e in out.trace))

print("\n[T5] backstop: a policy that never terminates -> forced safe escalate (max_steps)")
out = build_harness(lambda v: {"tool": "diagnose_masks", "args": {}}, SegTask(), COMMON,
                    Budget(compute_minutes=999, n_trials=999), max_steps=6).run("adapt", "src")
check("forced terminal == escalate", out.decision == "escalate")
check("never crashes / always returns an Outcome", out.decision in DECISIONS)

print("\n[T6] regression end-to-end on its OWN toolbox (no seg tools used)")
rplan = iter([
    {"tool": "load_and_deploy", "args": {}},
    {"tool": "diagnose_residuals", "args": {}},
    {"tool": "recalibrate", "args": {}},
    {"tool": "ship", "args": {"candidate": "reg_recal",
        "report": {"trust_level": "label_free_estimate", "confidence": 0.6, "summary": "recalibrated"}}},
])
out = build_harness(lambda v: next(rplan), RegressionTask(), COMMON,
                    Budget(compute_minutes=10, n_trials=5)).run("adapt reg", "src")
check("regression ships via its own toolbox", out.decision == "ship" and out.delivered == "reg_recal")

print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
sys.exit(1 if FAIL else 0)
