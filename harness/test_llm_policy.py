"""Mock-LLM tests for the agent brain (llm_policy) — no key, no network.

Proves: robust JSON parsing, the prompt carries the evidence table + guardrail
feedback, and a mock 'LLM' that only reads the prompt string can drive the harness
end to end AND recover when the harness vetoes its action."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core import Budget, Tool, SafetyVerdict, build_harness
from llm_policy import build_prompt, parse_action, make_llm_policy

PASS = []; FAIL = []
def check(n, c): (PASS if c else FAIL).append(n); print(("  PASS " if c else "  FAIL ") + n)

VALID = {"load_and_deploy", "bn_reset", "ship", "keep_source", "escalate"}

print("[P1] parse_action robustness")
check("bare json", parse_action('{"tool":"ship","args":{"x":1}}', VALID)["tool"] == "ship")
check("prose + json", parse_action('I will deploy.\n{"tool":"load_and_deploy","args":{}}', VALID)["tool"] == "load_and_deploy")
check("fenced json", parse_action('```json\n{"tool":"bn_reset","args":{}}\n```', VALID)["tool"] == "bn_reset")
check("unknown tool -> None (policy falls back)", parse_action('{"tool":"nope","args":{}}', VALID) is None)
check("garbage -> None (policy falls back)", parse_action('no json here', VALID) is None)
check("last action wins", parse_action('{"tool":"ship"} then {"tool":"keep_source","args":{}}', VALID)["tool"] == "keep_source")

print("\n[P2] prompt carries the decision context")
view = {"goal": "deploy well", "task": {"type": "segmentation", "dims": "2D", "metric": "Dice"},
        "step": 3, "deployed": True, "best_metric_estimate": 0.7, "k_labels_used": 0,
        "budget_remaining": {"n_trials": 2}, "evidence": {"failure_mode": "under_seg"},
        "candidates_tried": [{"method": "bn_reset", "metric_estimate": 0.7, "vlm_prefer": 0.3}],
        "tools": [{"name": "ship", "description": "deliver", "terminal": True}],
        "guardrail_feedback": "SHIP VETOED (safety): worse than source"}
p = build_prompt(view)
check("prompt shows candidate table w/ vlm_prefer", "bn_reset" in p and "vlm_prefer_over_source=0.3" in p)
check("prompt surfaces guardrail feedback", "VETOED" in p and "do not repeat" in p)
check("prompt lists the tool", "ship: deliver" in p)


# ----- a stub seg task + tools so a mock LLM can drive a real harness run -----
def _deploy(st, **k): return {"deployment": "d"}
class StubSeg:
    type, dims, metric_name = "segmentation", "2D", "Dice"
    def toolbox(self):
        return [
            Tool("load_and_deploy", "run model on target", _deploy, Budget(compute_minutes=1)),
            Tool("bn_reset", "recalibrate BN",
                 lambda st, **k: {"candidate": "bn", "metric_estimate": 0.72,
                                  "evidence": {"vlm_ok": True, "last_method": "bn_reset"}},
                 Budget(n_trials=1)),
        ]
    def safety_check(self, c, s, ev):
        return SafetyVerdict(bool(ev.get("vlm_ok")), "" if ev.get("vlm_ok") else "no quality evidence")
COMMON = [Tool("ship", "deliver", lambda st, **k: {}, terminal=True),
          Tool("keep_source", "keep source", lambda st, **k: {}, terminal=True),
          Tool("escalate", "hand back", lambda st, **k: {}, terminal=True)]


def mock_llm_good(prompt: str) -> str:
    """A 'reasoning' mock that only reads the prompt string the policy built."""
    if "GUARDRAIL FEEDBACK" in prompt and "VETOED" in prompt:
        return '{"tool":"keep_source","args":{"report":{"trust_level":"label_free_estimate","confidence":0.5,"summary":"kept source after veto"}}}'
    if "deployed=False" in prompt:
        return 'Let me deploy first.\n{"tool":"load_and_deploy","args":{}}'
    if "(none yet)" in prompt:
        return '{"tool":"bn_reset","args":{}}'
    return '{"tool":"ship","args":{"candidate":"bn","report":{"trust_level":"label_free_estimate","confidence":0.72,"summary":"shipped bn_reset"}}}'

print("\n[P3] mock LLM drives a full run: deploy -> bn_reset -> ship")
out = build_harness(make_llm_policy(mock_llm_good), StubSeg(), COMMON,
                    Budget(compute_minutes=10, n_trials=5)).run("deploy well", "src")
check("agent ships", out.decision == "ship")
check("trajectory = deploy + bn_reset (2 tool steps) + terminal",
      sum(1 for e in out.trace if e["kind"] == "tool") == 2
      and any(e["kind"] == "terminal" for e in out.trace))


def mock_llm_unsafe(prompt: str) -> str:
    # always tries to ship without quality evidence -> safety veto -> must recover
    if "GUARDRAIL FEEDBACK" in prompt and "VETOED" in prompt:
        return '{"tool":"escalate","args":{"report":{"trust_level":"label_free_estimate","confidence":0.1,"summary":"cannot ship safely"}}}'
    if "deployed=False" in prompt:
        return '{"tool":"load_and_deploy","args":{}}'
    return '{"tool":"ship","args":{"candidate":"x","report":{"trust_level":"label_free_estimate","confidence":0.9,"summary":"premature"}}}'

print("\n[P4] mock LLM ships unsafe -> harness vetoes -> agent reads feedback -> escalates")
out = build_harness(make_llm_policy(mock_llm_unsafe), StubSeg(), COMMON,
                    Budget(compute_minutes=10, n_trials=5)).run("deploy", "src")
check("recovers to escalate (no infinite loop)", out.decision == "escalate")
check("safety_veto recorded", any(e["kind"] == "safety_veto" for e in out.trace))

print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
sys.exit(1 if FAIL else 0)
