"""Unit tests for the budget-aware ship-policy additions:
  - _verify_nudge: silent at TRUE 0-label budget; fires when labels affordable + unused.
  - Harness._disclaim: stamps K=0 (label_free_estimate) ships UNVERIFIED; leaves verified ones."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_policy import _verify_nudge
from core import Harness

P = []; F = []
def chk(n, c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ") + n)

def V(k_rem, used=0, deployed=True, cands=(("bn_reset",),)):
    return {"deployed": deployed, "k_labels_used": used,
            "budget_remaining": {"k_labels": k_rem},
            "candidates_tried": [{"method": c[0]} for c in cands]}

print("[verify_nudge] silent at TRUE 0-label budget (permissive K=0 preserved)")
chk("k_labels budget 0 -> no nudge", _verify_nudge(V(0)) == "")
chk("not deployed -> no nudge", _verify_nudge(V(12, deployed=False)) == "")
chk("labels already used -> no nudge", _verify_nudge(V(12, used=6)) == "")
chk("no candidates yet -> no nudge", _verify_nudge(V(12, cands=())) == "")

print("[verify_nudge] fires when labels AFFORDABLE + unused + candidates exist")
nud = _verify_nudge(V(12))
chk("fires with k>0, used=0, candidates present", "VERIFY-BEFORE-SHIP" in nud and "12 labels" in nud)
chk("nudge names request_labels + label_verified preference",
    "request_labels" in nud and "label_verified" in nud)
chk("nudge preserves K=0 permissive framing", "TRUE 0-label budget" in nud)

print("[_disclaim] K=0 label_free_estimate ship -> UNVERIFIED stamped")
r = Harness._disclaim({"trust_level": "label_free_estimate", "confidence": 0.3, "summary": "ship bn_reset"})
chk("adds UNVERIFIED to summary", "UNVERIFIED" in r["summary"] and "ship bn_reset" in r["summary"])
chk("sets verified_not_worse_than_source=False", r["verified_not_worse_than_source"] is False)
chk("sets guarantee=none", r.get("guarantee") == "none")
r2 = Harness._disclaim(r)                         # idempotent — no double-stamp
chk("idempotent (no double UNVERIFIED)", r2["summary"].count("UNVERIFIED") == 1)

print("[_disclaim] label_verified ship left untouched")
rv = {"trust_level": "label_verified", "confidence": 0.85, "summary": "verified ship"}
out = Harness._disclaim(rv)
chk("verified ship summary unchanged", out["summary"] == "verified ship" and "verified_not_worse_than_source" not in out)
chk("non-dict tolerated", Harness._disclaim(None) is None)

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
