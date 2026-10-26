"""Component F1 unit tests for the honest report. Pure — fake task/state, no network/GPU."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delivery import build_report, render_report_md

P = []; F = []
def chk(n, c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ") + n)

class Task:
    type = "segmentation"; dims = "2d"; metric_name = "dice"
class Budget:
    def as_dict(self): return {"k_labels": 6, "compute_minutes": 12}
class State:
    def __init__(self, ev, used=0):
        self.evidence = ev; self.k_labels_used = used; self.best_metric = 0.5
        self.candidates_log = [{"method": "bn_reset"}, {"method": "tent"}]
        self.named_candidates = {"bn_reset": object()}
        self.budget_spent = Budget(); self.step = 7

EV = {"domain_shift": {"characterization": {"shift_type": "modality", "severity": "severe",
                                            "semantic_flag": True, "dominant_axis": "histogram",
                                            "key_differences": ["CT vs MR"]},
                       "numeric": {"per_axis": {"histogram": 0.9}}},
      "model_info": {"assessment": {"normalization": "BatchNorm", "model_family": "UNet"},
                     "introspection": {"class": "UNet2D", "in_channels": 1}},
      "labeled_dice": {"source": 0.12, "bn_reset": 0.39}, "best_by_labels": "bn_reset",
      "expected_metric": 0.85, "usable_floor": 0.65, "perf_rationale": "liver MR seg lit",
      "failure_mode": "boundary drift"}

print("[ship label_verified] enrich + preserve trust; verification=label_verified")
r = build_report(decision="ship", task=Task(), state=State(EV, used=6), chosen_name="bn_reset",
                 chosen_metric=0.39, base_report={"trust_level": "label_verified", "confidence": 0.8,
                                                  "summary": "ship bn_reset"})
chk("trust_level PRESERVED (not flipped)", r["trust_level"] == "label_verified")
chk("summary + confidence preserved", r["summary"] == "ship bn_reset" and r["confidence"] == 0.8)
chk("diagnosed_shift carries shift_type/severity/semantic", r["diagnosed_shift"]["shift_type"] == "modality"
    and r["diagnosed_shift"]["semantic_flag"] is True)
chk("action_taken lists candidates + path B/C/D", r["action_taken"]["candidates_tried"] == ["bn_reset", "tent"]
    and r["action_taken"]["decision_path"].startswith("C"))
chk("expected_performance verified + delivered from labeled_dice",
    r["expected_performance"]["verification"] == "label_verified"
    and r["expected_performance"]["delivered_metric"] == 0.39)
chk("reaches_usable computed (0.39 < 0.65 -> False)", r["expected_performance"]["reaches_usable"] is False)
chk("below-usable honesty flag present", any("usable floor" in f for f in r["honesty_flags"]))

print("[ship K=0 estimate] verification=proxy + UNVERIFIED honesty flag")
r0 = build_report(decision="ship", task=Task(), state=State(dict(EV, labeled_dice={}), used=0),
                  chosen_name="bn_reset", chosen_metric=0.39,
                  base_report={"trust_level": "label_free_estimate", "confidence": 0.3, "summary": "est"})
chk("verification = label_free_proxy", r0["expected_performance"]["verification"] == "label_free_proxy")
chk("LABEL-FREE honesty flag present", any("LABEL-FREE" in f for f in r0["honesty_flags"]))

print("[escalate no rec] honesty_flags non-empty (riga case)")
re_ = build_report(decision="escalate", task=Task(), state=State(EV, used=6), chosen_name="source",
                   chosen_metric=None, base_report={"trust_level": "label_verified", "confidence": 0.3,
                                                    "summary": "escalate"})
chk("escalate trust preserved (amos: label_verified stays)", re_["trust_level"] == "label_verified")
chk("escalate + no recommendation -> 'no grounded alternative' flag",
    any("no grounded alternative" in f for f in re_["honesty_flags"]))
chk("decision_path D for escalate", re_["action_taken"]["decision_path"].startswith("D"))

print("[scaling_plan + recommendations passthrough]")
ev2 = dict(EV, scaling_law={"fit": {"form": "saturating_exp", "formula": "y=..."}, "needed_for_usable": 5},
           recommendation={"verdict": "CT->MR unfixable", "finetune_labels_estimate": 40})
r2 = build_report(decision="escalate", task=Task(), state=State(ev2, used=6), chosen_name="source",
                  chosen_metric=None, base_report={"trust_level": "label_verified", "confidence": 0.3, "summary": "x"})
chk("scaling_plan passed through verbatim", r2["scaling_plan"]["needed_for_usable"] == 5)
chk("recommendations passed through; no 'no alternative' flag",
    r2["recommendations"]["finetune_labels_estimate"] == 40
    and not any("no grounded alternative" in f for f in r2["honesty_flags"]))

print("[render_md] produces sections, never raises")
md = render_report_md(r)
chk("md has core sections", all(h in md for h in ("Diagnosed shift", "What was done",
                                                  "Expected performance", "deployment report")))
chk("md renders honesty flags section", "Honesty flags" in md)
chk("md on scaling+rec report", "Scaling law" in render_report_md(r2) and "Recommended alternative" in render_report_md(r2))
ok = True
for bad in [{}, None, {"decision": "ship"}]:
    try: render_report_md(bad)
    except Exception as e: ok = False; print("   RAISED:", e)
chk("render never raises on junk", ok)

print("[robustness] build_report never raises on empty evidence/state")
ok = True
try:
    build_report(decision="ship", task=Task(), state=State({}, used=0), chosen_name="adapted",
                 chosen_metric=None, base_report=None)
except Exception as e:
    ok = False; print("   RAISED:", e)
chk("build_report tolerates empty evidence + no base_report", ok)

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
