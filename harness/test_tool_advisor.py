"""Component D unit tests. Pure — no network. Covers tool_advisor (families, priors, reorder
is a stable permutation, never raises) + the TOOL_ADVICE gating in llm_policy.build_prompt
(OFF = byte-inert; ON = reordered menu + §A.4 advisory, no leak into the 600-char dump)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tool_advisor as TA

P = []; F = []
def chk(n, c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ") + n)

MENU = ["analyze_domain_shift", "inspect_model", "bn_reset", "tent", "sar", "eata",
        "gamma_correct", "histogram_match", "spectral", "fda_warmup",
        "weight_average_source_adapted", "largest_connected_component",
        "finetune_head", "request_labels", "ship", "keep_source", "escalate"]

def ev_with(shift_type, severity="moderate", semantic=False):
    return {"domain_shift": {"characterization": {
        "shift_type": shift_type, "severity": severity, "semantic_flag": semantic,
        "axis_severity": {}, "dominant_axis": "intensity"}}}

print("[families] mechanism mapping correct")
chk("bn_reset/gamma/clahe -> normalization",
    all(TA.tool_family(t) == TA.NORMALIZATION for t in ("bn_reset", "gamma_correct", "clahe", "histogram_match")))
chk("tent/sar/eata -> entropy_tta",
    all(TA.tool_family(t) == TA.ENTROPY_TTA for t in ("tent", "sar", "eata")))
chk("spectral/fda_warmup -> spectral",
    TA.tool_family("spectral") == TA.SPECTRAL and TA.tool_family("fda_warmup") == TA.SPECTRAL)
chk("postproc/finetune/unknown -> other",
    all(TA.tool_family(t) == TA.OTHER for t in ("largest_connected_component", "finetune_head", "ship", "zzz")))

print("[priors] keyed on real enum; NORMALIZATION leads every shift; mixed/unknown empty")
chk("every prior leads with normalization",
    all(v[0] == TA.NORMALIZATION for k, v in TA.SHIFT_FAMILY_PRIOR.items() if v))
chk("mixed + unknown -> empty (no reorder)",
    TA.SHIFT_FAMILY_PRIOR["mixed"] == [] and TA.SHIFT_FAMILY_PRIOR["unknown"] == [])

print("[advice] intensity -> tta sufficient; modality -> NOT sufficient (the real signal)")
a_int = TA.build_tool_advice(ev_with("intensity"), MENU)
chk("intensity: bn_reset first in shortlist", a_int.shortlist and a_int.shortlist[0] == "bn_reset")
chk("intensity: expect_tta_sufficient True", a_int.expect_tta_sufficient is True)
a_mod = TA.build_tool_advice(ev_with("modality", "severe", True), MENU)
chk("modality: bn_reset STILL first (anti-chaos-regression)", a_mod.shortlist[0] == "bn_reset")
chk("modality: expect_tta_sufficient False", a_mod.expect_tta_sufficient is False)
chk("modality rationale says label-free INSUFFICIENT + prefer labels",
    "INSUFFICIENT" in a_mod.rationale and "labels" in a_mod.rationale)
a_mix = TA.build_tool_advice(ev_with("mixed"), MENU)
chk("mixed -> inert advice (no family_rank, no shortlist)", not a_mix.family_rank and not a_mix.shortlist)

print("[reorder] STABLE PERMUTATION; non-family tools fixed; bn_reset stays first")
order = TA.reorder_menu(MENU, a_mod)
chk("reorder preserves len + set (permutation)", len(order) == len(MENU) and set(order) == set(MENU))
chk("terminals/diagnostics keep their slots",
    order.index("ship") == MENU.index("ship") and order.index("analyze_domain_shift") == 0
    and order.index("finetune_head") == MENU.index("finetune_head"))
# the first family-mapped slot in MENU is bn_reset's index; NORMALIZATION leads -> bn_reset there
first_fam_slot = min(i for i, t in enumerate(MENU) if TA.tool_family(t) in TA.SHIFT_FAMILY_PRIOR["modality"])
chk("bn_reset occupies the first recovery slot", order[first_fam_slot] == "bn_reset")
chk("mixed/empty advice -> identity reorder",
    TA.reorder_menu(MENU, a_mix) == MENU and TA.reorder_menu(MENU, TA.ToolAdvice()) == MENU)

print("[robustness] never raises")
ok = True
for bad_ev, bad_menu in [({}, MENU), ({"domain_shift": "oops"}, MENU),
                         (ev_with("intensity"), []), (ev_with("intensity"), [{"name": "bn_reset"}]),
                         (None, None)]:
    try:
        adv = TA.build_tool_advice(bad_ev, bad_menu)
        TA.reorder_menu(bad_menu or [], adv)
    except Exception as e:
        ok = False; print("   RAISED:", e)
chk("malformed evidence/menu never raise", ok)

print("[gating] build_prompt: OFF = byte-inert; ON = reordered + advisory, no 600-char leak")
import importlib
import llm_policy
importlib.reload(llm_policy)
VIEW = {"goal": "deploy", "task": {"type": "segmentation", "dims": "2d", "metric": "dice"},
        "step": 3, "deployed": True, "best_metric_estimate": 0.5, "k_labels_used": 0,
        "budget_remaining": {"compute_minutes": 40}, "candidates_tried": [],
        "guardrail_feedback": None,
        "evidence": ev_with("modality", "severe", True),
        "tools": [{"name": n, "description": n + " desc"} for n in MENU]}
os.environ.pop("TOOL_ADVICE", None)
p_off = llm_policy.build_prompt(VIEW)
os.environ["TOOL_ADVICE"] = "1"
p_on = llm_policy.build_prompt(VIEW)
os.environ.pop("TOOL_ADVICE", None)
chk("OFF: no ADVISORY tool-suggestion line", "suggested tools" not in p_off)
chk("ON: ADVISORY shortlist rendered", "suggested tools" in p_on and "bn_reset" in p_on)
# the 600-char LATEST EVIDENCE dump must NOT carry tool_advice (it lives in §A.4)
dump_on = p_on.split("LATEST EVIDENCE:")[1].split("TOOLS:")[0]
chk("ON: tool_advice NOT in the 600-char evidence dump", "tool_advice" not in dump_on and "suggest" not in dump_on)
# menu reorder is presentation-only: same set of tool lines OFF vs ON
def tool_lines(p):
    seg = p.split("TOOLS:\n")[1].split("\n\n")[0]
    return sorted(l.strip() for l in seg.splitlines() if l.strip().startswith("- "))
chk("ON menu = same tool set as OFF (permutation, nothing dropped/added)",
    tool_lines(p_on) == tool_lines(p_off))

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
