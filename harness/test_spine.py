"""Component A decision-spine unit tests (P5). Pure: imports only llm_policy._labeled_directive."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # harness/ on path
from llm_policy import _labeled_directive, build_prompt

P=[]; F=[]
def chk(n,c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ")+n)

def V(ld=None, best=None, sl=None, rec=None, tools=None, sota=None, usable=None, margin=0.05,
      budget=None):
    ev={}
    if ld is not None: ev["labeled_dice"]=ld
    if best is not None: ev["best_by_labels"]=best
    if sl is not None: ev["scaling_law"]=sl
    if rec is not None: ev["recommendation"]=rec
    if sota is not None: ev["expected_metric"]=sota
    if usable is not None: ev["usable_floor"]=usable
    if margin is not None: ev["safety_margin"]=margin
    return {"evidence":ev, "tools": tools or [], "budget_remaining": budget or {"compute_minutes":50}}

FT=[{"name":"finetune_head"},{"name":"finetune_partial"},{"name":"finetune_lora"},{"name":"recommend_alternative"}]

print("[B0] verified gain -> ship directive, grounded, NO magic thresholds")
d=_labeled_directive(V(ld={"source":0.28,"finetune_partial":0.68}, best="finetune_partial",
                       tools=FT, sota=0.85, usable=0.65))
chk("B0 says LABEL-VERIFIED GAIN + ship + candidate", "LABEL-VERIFIED GAIN" in d and "ship" in d and "finetune_partial" in d)
chk("B0 grounds in looked-up sota/usable", "0.85" in d and "0.65" in d)

print("[B1] no verified gain, compute remains -> next ladder rung")
d=_labeled_directive(V(ld={"source":0.40,"finetune_head":0.42}, best="finetune_head", tools=FT))
chk("B1 directs finetune_partial (next untried rung)", "finetune_partial" in d and "call it now" in d)

print("[B-SCALE] scaling_eval available + no scaling_law yet -> route to scaling_eval, NOT ladder")
d=_labeled_directive(V(ld={"source":0.41,"bn_reset":0.34}, best="source",
                       tools=FT+[{"name":"scaling_eval"}]))
chk("B-SCALE directs scaling_eval (honest CV), not manual ladder",
    "call scaling_eval now" in d and "CROSS-VALIDATION" in d and "call finetune_head" not in d)
d2=_labeled_directive(V(ld={"source":0.28,"finetune_partial":0.68}, best="finetune_partial",
                        tools=FT+[{"name":"scaling_eval"}], sota=0.85, usable=0.65,
                        sl={"fit":{"form":"saturating_exp","formula":"y=..."},"measured":[],"needed_for_usable":5}))
chk("after scaling_law present -> B0 ships (no re-route)", "LABEL-VERIFIED GAIN" in d2 and "ship" in d2)

print("[B2] no gain, ladder exhausted -> adequacy/escalation judgment")
exhausted={"source":0.1,"finetune_head":0.1,"finetune_partial":0.1,"finetune_lora":0.1}
d=_labeled_directive(V(ld=exhausted, best="source", tools=[{"name":"recommend_alternative"}], sota=0.85, usable=0.65))
chk("B2 no-rec -> call recommend_alternative", "ladder exhausted" in d and "recommend_alternative" in d)
d=_labeled_directive(V(ld=exhausted, best="source", rec={"verdict":"CT->MRI unfixable","finetune_labels_estimate":40},
                       tools=[{"name":"recommend_alternative"}], sota=0.85, usable=0.65))
chk("B2 with-rec -> escalate + inline recommendation", "escalate" in d and "finetune_labels_estimate" in d)

print("[B-SL] scaling-law consumed via needed_for_usable")
sl={"fit":{"form":"saturating_exp","formula":"y=0.85-0.5*exp(-0.3n)"},"measured":[{"k":5,"mean":0.7}],
    "needed_for_usable":5}
d=_labeled_directive(V(ld={"source":0.28,"finetune_partial":0.68}, best="finetune_partial",
                       tools=FT, sota=0.85, usable=0.65, sl=sl))
chk("B-SL renders curve form + annotate N + verbatim", "SCALING LAW" in d and "saturating_exp" in d
    and "annotate ≈5" in d and "Hand this curve" in d)

print("[fallback] lookups absent -> degrade gracefully, still emits a branch")
d=_labeled_directive(V(ld=exhausted, best="source", tools=[{"name":"recommend_alternative"}]))
chk("fallback mentions no looked-up target", "no looked-up target available" in d and "ladder exhausted" in d)

print("[robustness] never raises on malformed inputs")
ok=True
for bad in [V(ld={"source":0.3}, best="ghost", tools=FT),                       # best not in ld
            V(ld={"source":0.3,"x":0.4}, best="x", sl={"fit":"notadict"}, tools=FT),  # malformed sl
            V(ld={}, tools=FT), {"evidence":{},"tools":[]},                       # empty
            {"evidence":{"labeled_dice":{"source":0.5}}}]:                        # no tools/budget keys
    try: _labeled_directive(bad)
    except Exception as e: ok=False; print("   RAISED:", e)
chk("malformed/empty inputs never raise", ok)
chk("no labeled_dice -> empty directive", _labeled_directive({"evidence":{},"tools":[]})=="")

print("[no-magic] directive contains none of the deleted hardcoded thresholds")
alld=" ".join(_labeled_directive(v) for v in [
    V(ld={"source":0.28,"finetune_partial":0.68}, best="finetune_partial", tools=FT, sota=0.85, usable=0.65),
    V(ld={"source":0.40,"finetune_head":0.42}, best="finetune_head", tools=FT),
    _ if False else V(ld=exhausted, best="source", tools=FT)])
chk("no 'usable_bar' / '0.30' / '0.5*usable' magic in any directive",
    "usable_bar" not in alld and "0.30" not in alld and "0.5 *" not in alld and "0.5*" not in alld)

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
