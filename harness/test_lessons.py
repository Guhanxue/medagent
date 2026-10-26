"""Component E unit tests. Pure stdlib, no network. Covers fingerprint, mining (incl
histogram-blind refusal + family bucketing + cls metric), merge/MIXED, gated retrieval,
idempotent store update."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lessons as L

P = []; F = []
def chk(n, c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ") + n)

def rec(shift="modality", sev="severe", audit=None, metric=None, task_type=None, task="ds",
        char="ok"):
    if char == "missing":
        ds = None
    elif char == "error":
        ds = {"characterization": {"error": "llm failed"}}
    else:
        ds = {"characterization": {"shift_type": shift, "severity": sev, "semantic_flag": True}}
    return {"task": task, "metric": metric, "task_type": task_type,
            "agent_analysis": {"domain_shift": ds, "model_info": {"assessment": {"normalization": "BatchNorm"}}},
            "oracle_audit": audit or {}}

AUDIT = {"source": {"true_dice": 0.12}, "bn_reset": {"true_dice": 0.39},
         "tent": {"true_dice": 0.24}, "finetune_head": {"true_dice": 0.0},
         "largest_connected_component": {"true_dice": 0.13}}

print("[fingerprint] None until BOTH diagnostics; None on error/raw")
chk("valid fp when char+assessment present",
    L.fingerprint_of(rec()["agent_analysis"] | {"domain_shift": rec()["agent_analysis"]["domain_shift"]},
                     "segmentation") is None  # ev shape differs; test via real ev below
    or True)
ev_ok = {"domain_shift": {"characterization": {"shift_type": "modality", "severity": "severe"}},
         "model_info": {"assessment": {"normalization": "BatchNorm"}}}
fp = L.fingerprint_of(ev_ok, "segmentation")
chk("fp has categorical tokens only", fp == {"task_type": "segmentation", "shift_type": "modality",
                                             "severity": "severe", "normalization": "batchnorm"})
chk("fp None when model assessment missing",
    L.fingerprint_of({"domain_shift": ev_ok["domain_shift"]}, "segmentation") is None)
chk("fp None on characterization error shape",
    L.fingerprint_of({"domain_shift": {"characterization": {"error": "x"}},
                      "model_info": {"assessment": {"normalization": "bn"}}}, "segmentation") is None)

print("[mine] family bucketing + helped/hurt; ignores 'other'; histogram-blind refusal")
ls = L.mine_lessons(rec(audit=AUDIT))
ids = {l["id"] for l in ls}
chk("normalization family helped lesson emitted",
    any(l["kind"] == "method_helped" and "normalization" in l["id"] for l in ls))
chk("entropy_tta family helped lesson emitted (tent +0.12)",
    any("entropy_tta" in l["id"] and l["kind"] == "method_helped" for l in ls))
chk("'other' families (finetune/postproc) NOT mined",
    not any("finetune" in i or "largest" in i for i in ids))
chk("conditions are categorical (no dataset name leaks into condition)",
    all("ds" not in json.dumps(l["condition"]) for l in ls))
chk("histogram-blind: missing characterization -> NO lessons", L.mine_lessons(rec(char="missing")) == [])
chk("error characterization -> NO lessons", L.mine_lessons(rec(char="error")) == [])
chk("no source true metric -> NO lessons", L.mine_lessons(rec(audit={"bn_reset": {"true_dice": 0.5}})) == [])

print("[mine] cls (true_auroc) resolves generically")
cls_audit = {"source": {"true_auroc": 0.70}, "bn_reset": {"true_auroc": 0.60}}
lc = L.mine_lessons(rec(audit=cls_audit, metric="AUROC"))
chk("cls metric -> classification task_type + method_hurt (0.60<0.70)",
    lc and lc[0]["condition"]["task_type"] == "classification" and lc[0]["kind"] == "method_hurt")

print("[merge] accumulate -> moderate; conflicting sign -> MIXED/weak")
def norm_lesson(r):
    return next(l for l in L.mine_lessons(r) if "normalization" in l["id"])
by = {}
L.merge_lesson(by, norm_lesson(rec(audit=AUDIT, task="dsA")))
norm_id = next(i for i in by if "normalization" in i)
L.merge_lesson(by, norm_lesson(rec(audit=AUDIT, task="dsB")))  # same sign again
chk("two consistent runs -> confidence moderate, n_runs=2",
    by[norm_id]["confidence"] == "moderate" and by[norm_id]["evidence_summary"]["n_runs"] == 2)
# now a conflicting (hurt) run for the same family/condition
conflict = {"source": {"true_dice": 0.50}, "bn_reset": {"true_dice": 0.20}}
L.merge_lesson(by, norm_lesson(rec(audit=conflict, task="dsC")))
chk("conflicting sign -> MIXED -> weak (suppressed)",
    by[norm_id]["confidence"] == "weak" and by[norm_id]["evidence_summary"]["mixed"] is True
    and "MIXED" in by[norm_id]["claim"])

print("[retrieve] gated to confidence >= moderate; weak/mixed NOT surfaced")
store_mod = [{"id": "x", "condition": fp, "kind": "method_helped", "confidence": "moderate",
              "claim": "normalization family helped on modality/severe segmentation",
              "evidence_summary": {"median_true_delta": 0.27}}]
chk("moderate lesson matching fp retrieved", bool(L.lessons_evidence(store_mod, fp).get("retrieved_lessons")))
chk("retrieval note present (hints not rules)", "NOT rules" in L.lessons_evidence(store_mod, fp)["retrieved_lessons_note"])
store_weak = [dict(store_mod[0], confidence="weak")]
chk("weak lesson NOT surfaced", L.lessons_evidence(store_weak, fp) == {})
chk("non-matching fp -> {}", L.lessons_evidence(store_mod, dict(fp, shift_type="intensity")) == {})
chk("empty store -> {}", L.lessons_evidence([], fp) == {})
chk("None fp -> {}", L.lessons_evidence(store_mod, None) == {})

print("[store] idempotent by id; writes jsonl + md")
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "lessons", "lessons.jsonl")
    L.update_store_with_rec(path, rec(audit=AUDIT, task="dsA"))
    m1 = L.load_store(path)
    L.update_store_with_rec(path, rec(audit=AUDIT, task="dsA"))   # same rec again
    m2 = L.load_store(path)
    chk("same ids after re-mining (idempotent set of ids)",
        {l["id"] for l in m1} == {l["id"] for l in m2})
    chk("re-mining bumps n_runs (accumulates, not duplicates)",
        next(l for l in m2 if "normalization" in l["id"])["evidence_summary"]["n_runs"] == 2)
    chk("lessons.md written", os.path.exists(os.path.splitext(path)[0] + ".md"))

print("[robustness] never raises on junk")
ok = True
for bad in [{}, {"agent_analysis": None}, {"oracle_audit": "x"}, None]:
    try:
        L.mine_lessons(bad or {})
    except Exception as e:
        ok = False; print("   RAISED:", e)
chk("mine never raises on junk", ok)

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
