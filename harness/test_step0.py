"""Step 0 (shared primitives) — inertness + correctness. No GPU, no network."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import evidence_keys as ek
from harness.llm_policy import _advisory_section, build_prompt
from harness.seg_adapter import SegTask
from harness.cls_adapter import ClsTask

P=[]; F=[]
def chk(n,c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ")+n)

print("[1] safety_margin: single source of truth, seg==cls, flat 0.05 noise floor")
import numpy as np
ok=True
for sd in [None,0.0,0.0015,0.04,0.3,0.5,0.77,0.95,1.0]:
    s=SegTask.safety_margin(None, sd); c=ClsTask.safety_margin(None, sd)
    old=0.05  # flat noise floor
    ok = ok and abs(s-c)<1e-12 and abs(s-old)<1e-12
chk("safety_margin seg==cls==flat 0.05 over grid", ok)

print("[2] evidence_keys accessors are crash-proof on failure shapes")
chk("characterization {} on missing", ek.get_characterization({})=={})
chk("characterization {} on error shape", ek.get_characterization({"domain_shift":{"characterization":{"error":"x"}}})=={})
chk("characterization {} on raw shape", ek.get_characterization({"domain_shift":{"raw":"x"}})=={})
good={"domain_shift":{"characterization":{"shift_type":"intensity","severity":"moderate"}}}
chk("characterization returns dict on good", ek.get_characterization(good).get("shift_type")=="intensity")
chk("model_assessment {} on missing", ek.get_model_assessment({})=={})
chk("introspection {} on missing", ek.get_introspection({})=={})
chk("norm_token batchnorm", ek.norm_token({"normalization":"BatchNorm present (18 layers)"})=="batchnorm")
chk("norm_token instancenorm", ek.norm_token({"normalization":"InstanceNorm 2D"})=="instancenorm")
chk("norm_token unknown", ek.norm_token({"normalization":"something"})=="unknown")
chk("metric_keys seg default", ek.metric_keys({"source_dice":0.5})==("true_dice","labeled_dice","source_dice"))
chk("metric_keys cls via auroc key", ek.metric_keys({"true_auroc":0.9})==("true_auroc","labeled_auroc","source_auroc"))
chk("metric_keys cls via metric field", ek.metric_keys({"metric":"AUROC"})==("true_auroc","labeled_auroc","source_auroc"))

print("[3] §A.4 advisory section is INERT until C/D/E add keys")
chk("advisory '' on empty evidence", _advisory_section({})=="")
chk("advisory '' on legacy domain_shift (no 'why')", _advisory_section(good)=="")
chk("advisory non-empty when tool_advice present", _advisory_section({"tool_advice":"try clahe"})!="")
chk("advisory renders 'why' when shift v2 adds it",
    "rationale" in _advisory_section({"domain_shift":{"characterization":{"why":"T1->FLAIR sequence change"}}}))

print("[4] build_prompt has NO [ADVISORY] block for a legacy view (byte-inert)")
view={"goal":"deploy well","task":{"type":"segmentation","dims":"2D","metric":"Dice"},
      "step":3,"deployed":True,"best_metric_estimate":0.7,"k_labels_used":0,
      "budget_remaining":{"n_trials":2},"evidence":{"failure_mode":"under_seg","source_stats":{}},
      "candidates_tried":[{"method":"bn_reset","metric_estimate":0.7,"vlm_prefer":0.3}],
      "tools":[{"name":"ship","description":"deliver","terminal":True},
               {"name":"bn_reset","description":"recalibrate BN"}],
      "guardrail_feedback":None}
p=build_prompt(view)
chk("no [ADVISORY in legacy prompt", "[ADVISORY" not in p)
view2=dict(view); view2["evidence"]=dict(view["evidence"]); view2["evidence"]["tool_advice"]="prefer clahe, gamma"
chk("[ADVISORY appears once a new key exists", "[ADVISORY" in build_prompt(view2))

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
