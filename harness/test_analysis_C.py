"""Component C unit tests (multi-axis + semantic shift analysis). Pure — fake LLM, no network.
Validates: legacy keys present (LLM + fallback), MAX-rule + semantic escalation, fallback never
auto-severe / never semantic, characterization-first ordering, model_search reads it w/o
KeyError, _shared_introspection shared with inspect_model."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # harness/ on path
import analysis as A

P = []; F = []
def chk(n, c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ") + n)

class FakeLLM:
    def __init__(self, payload): self.payload = payload
    def chat(self, prompt): return json.dumps(self.payload)

LEGACY = ["shift_type", "severity", "key_differences", "training_free_outlook"]
_SEVS = ("mild", "moderate", "severe")
# two tiny image sets with a real intensity/contrast gap
import numpy as np
SRC = [np.full((16, 16), 0.30, np.float32) + 0.02 * np.random.RandomState(i).randn(16, 16) for i in range(8)]
TGT = [np.full((16, 16), 0.62, np.float32) + 0.10 * np.random.RandomState(i).randn(16, 16) for i in range(8)]

print("[legacy keys] present + FIRST, both LLM and fallback paths")
llm = FakeLLM({"shift_type": "sequence",
               "axis_severity": {"intensity": "moderate", "contrast": "moderate",
                                 "histogram": "moderate", "dynamic_range": "mild"},
               "semantic_flag": True, "key_differences": ["T1 vs FLAIR contrast"],
               "training_free_outlook": "BN reset unlikely to fix a sequence change",
               "why": "T1->FLAIR is a different MR weighting"})
o = A.analyze_domain_shift(SRC, TGT, llm, task_desc="T1->FLAIR brain tumor")
ch = o["characterization"]
chk("top-level dict serializes characterization FIRST", list(o.keys())[0] == "characterization")
chk("characterization legacy keys present + ordered first",
    list(ch.keys())[:4] == LEGACY and all(k in ch for k in LEGACY))
chk("numeric carries per_axis + per_axis_raw", "per_axis" in o["numeric"] and "per_axis_raw" in o["numeric"])

print("[semantic escalation] semantic_flag=True -> severity forced to severe (kills laundering)")
chk("sequence + semantic_flag True -> severe", ch["severity"] == "severe" and ch["semantic_flag"] is True)
chk("shift_type preserved as 'sequence'", ch["shift_type"] == "sequence")
chk("trust='llm' on the LLM path", ch.get("trust") == "llm")

print("[MAX rule] non-semantic -> severity = MAX over axes, no escalation")
llm2 = FakeLLM({"shift_type": "contrast",
                "axis_severity": {"intensity": "moderate", "contrast": "moderate",
                                  "histogram": "moderate", "dynamic_range": "mild"},
                "semantic_flag": False, "why": "scanner contrast drift"})
c2 = A.analyze_domain_shift(SRC, TGT, llm2)["characterization"]
chk("all-moderate + semantic False -> moderate (not severe)", c2["severity"] == "moderate")
# a single severe axis must surface even when non-semantic (no laundering DOWN either)
llm3 = FakeLLM({"shift_type": "intensity",
                "axis_severity": {"intensity": "severe", "contrast": "mild",
                                  "histogram": "mild", "dynamic_range": "mild"},
                "semantic_flag": False})
c3 = A.analyze_domain_shift(SRC, TGT, llm3)["characterization"]
chk("one severe axis -> MAX surfaces severe", c3["severity"] == "severe")
chk("dominant_axis = the severe one", c3["dominant_axis"] == "intensity")

print("[fallback] no LLM -> numeric-derived, never semantic, never auto-severe, tagged low-trust")
fb = A.analyze_domain_shift(SRC, TGT, None)["characterization"]
chk("fallback has all legacy keys", all(k in fb for k in LEGACY))
chk("fallback semantic_flag == False ALWAYS", fb["semantic_flag"] is False)
chk("fallback severity <= moderate (never severe)", fb["severity"] in ("mild", "moderate"))
chk("fallback trust = numeric_derived_low", fb.get("trust") == "numeric_derived_low")
# malformed LLM output also routes to fallback + records the error
bad = A.analyze_domain_shift(SRC, TGT, FakeLLM("not json at all"))["characterization"]
chk("malformed LLM -> fallback (low-trust, not severe)",
    bad.get("trust") == "numeric_derived_low" and bad["severity"] in ("mild", "moderate"))

print("[model_search] reads finalized characterization without KeyError")
try:
    from evidence_keys import get_characterization
    ev = {"domain_shift": o}
    sc = get_characterization(ev)
    ok = sc.get("shift_type") == "sequence" and sc.get("severity") == "severe"
    # mirror model_search's own access pattern
    sc2 = o.get("characterization", o)
    ok = ok and sc2.get("shift_type") and sc2.get("severity") and sc2.get("key_differences") is not None
    chk("get_characterization + model_search access pattern both work", ok)
except Exception as e:
    chk("get_characterization + model_search access pattern both work", False); print("   RAISED:", e)

print("[introspection] _shared_introspection shared by analyze_domain_shift + inspect_model")
try:
    import torch, torch.nn as nn
    net = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                        nn.Conv2d(4, 2, 1))
    prov = A.analyze_domain_shift(SRC, TGT, None, model=net)["provenance"]
    intro = A.inspect_model(net, None)["introspection"]
    chk("provenance == inspect_model introspection (same helper)",
        prov.get("batchnorm") == intro.get("batchnorm") == 1
        and prov.get("class") == intro.get("class")
        and prov.get("in_channels") == intro.get("in_channels") == 1)
    chk("provenance present only when model given", A.analyze_domain_shift(SRC, TGT, None)["provenance"] == {})
except Exception as e:
    chk("provenance == inspect_model introspection (same helper)", False); print("   RAISED:", e)

print("[robustness] never raises on degenerate axis_severity shapes")
ok = True
for payload in [{"shift_type": "mixed"},                                   # no axis_severity
                {"axis_severity": "notadict", "semantic_flag": True},      # wrong type
                {"axis_severity": {"intensity": "bogus"}},                 # bad value
                {"axis_severity": {}, "semantic_flag": False}]:            # empty
    try:
        r = A.analyze_domain_shift(SRC, TGT, FakeLLM(payload))["characterization"]
        assert r["severity"] in _SEVS and all(k in r for k in LEGACY)
    except Exception as e:
        ok = False; print("   RAISED:", e)
chk("degenerate axis_severity never raises + always emits legacy keys", ok)

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
