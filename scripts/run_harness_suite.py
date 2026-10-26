#!/usr/bin/env python
"""Run the deployment harness across EVERY 2D segmentation task we have, and save
the ENTIRE decision logic per task — to show the agent makes a SMART, task-specific
choice (keep-source on mild traps, recover on recoverable shifts, request labels when
uncertain, escalate when hopeless), not one-size-fits-all.

Saves experiments/harness_decisions/<task>.json (full trajectory + decision + report
+ labeled_dice + hidden-oracle audit) and a suite_summary.json + printed table.
Real LLM brain + pairwise VLM + K-shot (uses the explicitly configured LLM backend).
"""

from pathlib import Path
import os, sys, json
import numpy as np
PROJECT_ROOT = os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, PROJECT_ROOT)
DEVICE = "cuda:0"

from scripts._source_loader import load_source_model
from scripts.run_ablations import load_npz_dataset
from scripts.run_baseline_tta import _avg_dice, _to_tensor   # noqa: F401 (adapter uses it)
from harness.core import Tool, Budget, build_harness
from harness.seg_adapter import SegTask
from harness.llm_policy import make_llm_policy

# Distinct deployment tasks spanning the decision regimes (one canonical target each).
TASKS = [
    dict(name="prostate_RUNMC_to_HK", shift="mild cross-site prostate MRI (source usually best)",
         ckpt="checkpoints/unet_prostate_runmc_best.pt", nc=2, fg_prior=0.05,
         src="data/prostate/source", tgt="data/prostate/target_HK",
         struct={"background": 0, "prostate": 1}),
    dict(name="riga_BinRushed_to_Magrabia", shift="cross-scanner fundus disc/cup",
         ckpt="checkpoints/unet_riga_binrushed_best.pt", nc=3, fg_prior=0.15,
         src="data/riga_plus/source", tgt="data/riga_plus/target_Magrabia",
         struct={"background": 0, "disc": 1, "cup": 2}),
    dict(name="chaos_CT_to_MRI_T2SPIR", shift="CT->MRI liver (strong, RECOVERABLE)",
         ckpt="checkpoints/unet_chaos_ct_best.pt", nc=2, fg_prior=0.12,
         src="data/chaos/source", tgt="data/chaos/target",
         struct={"background": 0, "liver": 1}),
    dict(name="amos_CT_to_MRI_multiorgan", shift="CT->MRI 4-organ (SEVERE / baseline-limited)",
         ckpt="checkpoints/unet_amos_ct_best.pt", nc=5, fg_prior=0.10,
         src="data/amos/source", tgt="data/amos/target",
         struct={"background": 0, "liver": 1, "spleen": 2, "right_kidney": 3, "left_kidney": 4}),
    dict(name="brats_T1_to_FLAIR_tumor", shift="T1->FLAIR brain TUMOR (small lesion)",
         ckpt="checkpoints/unet_brats_t1_best.pt", nc=2, fg_prior=0.02,
         src="data/brats/source", tgt="data/brats/target_FLAIR",
         struct={"background": 0, "tumor": 1}),
    dict(name="mmwhs_CT_whole_heart", shift="cardiac CT whole-heart cross-domain",
         ckpt="checkpoints/unet_mmwhs_ct_best.pt", nc=5, fg_prior=0.10,
         src="data/mmwhs/source", tgt="data/mmwhs/target",
         struct={"background": 0, "MYO": 1, "LA": 2, "LV": 3, "RV": 4}),
    dict(name="fundus_REFUGE", shift="fundus disc/cup cross-domain",
         ckpt="checkpoints/unet_refuge_best.pt", nc=3, fg_prior=0.10,
         src="data/fundus/source", tgt="data/fundus/target",
         struct={"background": 0, "disc": 1, "cup": 2}),
    dict(name="fundus_roi_REFUGE", shift="fundus ROI-cropped disc/cup",
         ckpt="checkpoints/fundus_roi/unet_refuge_best.pt", nc=3, fg_prior=0.30,
         src="data/fundus_roi/source", tgt="data/fundus_roi/target",
         struct={"background": 0, "disc": 1, "cup": 2}),
    dict(name="mnms_Siemens_to_GE", shift="cardiac MRI cross-vendor (Siemens->GE)",
         ckpt="checkpoints/unet_mnms_siemens_best.pt", nc=4, fg_prior=0.05,
         src="data/mnms/vendor_Siemens", tgt="data/mnms/vendor_GE",
         struct={"background": 0, "LV": 1, "MYO": 2, "RV": 3}),
    dict(name="mnms_field_1.5T_to_3T", shift="cardiac MRI cross-field-strength (1.5T->3T)",
         ckpt="checkpoints/unet_mnms_15T_best.pt", nc=4, fg_prior=0.05,
         src="data/mnms_field/field_1.5T", tgt="data/mnms_field/field_3T",
         struct={"background": 0, "LV": 1, "MYO": 2, "RV": 3}),
    # SegFormer (MiT transformer encoder, LayerNorm + 1 decode-head BatchNorm)
    # — the FIRST 2D transformer SEGMENTER in the harness. SAME prostate
    # RUNMC->HK shift/target as prostate_RUNMC_to_HK (a UNet) so the agent's
    # decision is a clean CNN-vs-transformer cross-arch comparison on an
    # identical task. arch="segformer" routes the loader to SegFormer2D.
    dict(name="prostate_RUNMC_to_HK_segformer",
         shift="mild cross-site prostate MRI (SegFormer transformer source)",
         ckpt="checkpoints/segformer_prostate_runmc_best.pt", nc=2, fg_prior=0.05,
         src="data/prostate/source", tgt="data/prostate/target_HK",
         struct={"background": 0, "prostate": 1}, arch="segformer"),
]


def make_vlm(task, llm, timgs, nc, struct):
    from medadaptagent.graph.agents.execution_engineer import _build_vlm_task_ctx
    from medadaptagent.tasks import get_task
    ctx = _build_vlm_task_ctx({"structures": struct, "foreground_priors": {}}, {})
    ctx["num_classes_hint"] = nc
    anchors = [timgs[i] for i in np.linspace(0, len(timgs) - 1, min(6, len(timgs)), dtype=int)]
    task.set_vlm(llm, ctx, get_task("segmentation"), anchors)


def run_one(cfg, llm):
    base = load_source_model(os.path.join(PROJECT_ROOT, cfg["ckpt"]), cfg["nc"],
                             DEVICE, arch=cfg.get("arch", "unet"), project_root=None)
    timgs, tlbls = load_npz_dataset(os.path.join(PROJECT_ROOT, cfg["tgt"]), 50)
    try:
        simgs, _ = load_npz_dataset(os.path.join(PROJECT_ROOT, cfg["src"]), 40)
    except Exception:
        simgs = timgs
    task_desc = (f"{cfg['shift']}; segmenting {list(cfg['struct'].keys())[1:]} "
                 f"(deploying {os.path.basename(cfg['ckpt'])} on {cfg['tgt']})")
    task = SegTask(base, simgs, timgs, cfg["nc"], DEVICE, fg_prior=cfg["fg_prior"],
                   tlbls=tlbls, task_desc=task_desc)
    make_vlm(task, llm, timgs, cfg["nc"], cfg["struct"])
    common = [Tool("ship", "deliver the chosen model", lambda st, **k: {}, terminal=True),
              Tool("keep_source", "keep the source model", lambda st, **k: {}, terminal=True),
              Tool("escalate", "hand back to clinician", lambda st, **k: {}, terminal=True)]
    budget = Budget(compute_minutes=int(os.environ.get("COMPUTE_MIN", "60")),
                    n_trials=4, llm_usd=5.0, k_labels=int(os.environ.get("KLAB", "12")))
    goal = (f"Deploy this {cfg['struct']} segmenter at a new site ({cfg['shift']}) without "
            f"shipping a worse model; adapt within budget if it underperforms.")
    # no_progress_cap raised: on a degenerate regime NO adaptation improves the label-free
    # quality, so a tight cap force-escalates BEFORE the agent can request_labels ->
    # recommend_alternative. Give it room to complete the smart-escalation flow.
    h = build_harness(make_llm_policy(lambda p: llm.chat(p)), task, common, budget,
                      max_steps=24, no_progress_cap=10)
    out = h.run(goal=goal, source_model=base)

    # Final evaluation excludes every target slice revealed for training or selection.
    audit_imgs, audit_lbls = list(timgs), list(tlbls)
    if hasattr(task, "final_audit_data"):
        audit_imgs, audit_lbls = task.final_audit_data()
    elif getattr(task, "_labels_revealed", False):
        used_ids = {id(x) for x in list(task._train[0]) + list(task._hold[0])}
        untouched = [(x, y) for x, y in zip(timgs, tlbls) if id(x) not in used_ids]
        if not untouched:
            raise RuntimeError("no untouched target slices remain for final evaluation")
        audit_imgs, audit_lbls = map(list, zip(*untouched))

    # hidden-oracle audit (true Dice the agent never saw) — ctx-aware so bridged
    # preprocessing/postprocessing/TTA candidates are measured through their full chain.
    from harness.registry_bridge import seg_dice_of
    audit = {}
    for b in task.built:
        d = float(seg_dice_of(b["model"], audit_imgs, audit_lbls, cfg["nc"], DEVICE))
        audit[b["name"]] = {"label_free_quality": round(b["quality"], 4),
                            "prediction_stats": b.get("stats"),
                            "vlm_prefer": (None if b.get("vlm_prefer") is None else round(b["vlm_prefer"], 3)),
                            "true_dice": round(d, 4), "shipped": (b["model"] is out.delivered)}
    src_d = audit.get("source", {}).get("true_dice", 0.0)
    ship_d = next((a["true_dice"] for a in audit.values() if a["shipped"]), src_d)
    oracle_d = max((a["true_dice"] for a in audit.values()), default=src_d)
    rec = {"task": cfg["name"], "shift": cfg["shift"], "ckpt": cfg["ckpt"], "tgt": cfg["tgt"],
           "nc": cfg["nc"], "structures": cfg["struct"],
           "decision": out.decision, "report": out.report, "steps": out.steps,
           "budget_spent": out.budget_spent.as_dict(),
           "labeled_dice": task._labeled_dice,
           "trajectory": out.trace,                       # <-- the ENTIRE decision logic
           "oracle_audit": audit,
           "source_dice": round(src_d, 4), "shipped_dice": round(ship_d, 4),
           "delta_vs_source": round(ship_d - src_d, 4), "oracle_dice": round(oracle_d, 4),
           "safety_held": ship_d >= src_d - 1e-6, "shipped_is_oracle": abs(ship_d - oracle_d) < 1e-6,
           "agent_analysis": {k: out.evidence.get(k) for k in
                              ("domain_shift", "model_info", "failure_mode", "recommendation",
                               "retrieved_lessons", "zero_shot_comparison")}}
    rec.update({"audit_population": "target_slices_never_revealed_to_system",
                "n_target_total": len(timgs),
                "n_target_excluded_from_test": len(timgs) - len(audit_imgs),
                "n_target_untouched": len(audit_imgs)})
    if os.environ.get("PACKAGE", "0") != "0":
        try:
            from harness.delivery import package_artifact
            cfg.setdefault("arch", "unet")
            man = package_artifact(out=out, task=task, cfg=cfg,
                                   dest_dir=os.path.join(PROJECT_ROOT, "experiments", "artifacts"),
                                   project_root=PROJECT_ROOT)
            print("  packaged artifact ->", man.dir)
        except Exception as e:
            print("  package_artifact failed:", e)
    return rec


def main():
    from medadaptagent.knowledge.llm_client import create_llm_client
    model = os.environ.get("MEDADAPT_LLM_MODEL", os.environ.get("MODEL"))
    print(f"== LLM brain + VLM model: {model} ==")
    llm = create_llm_client(model=model)
    _tag = os.environ.get("OUTDIR_TAG", "")
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "harness_decisions" + (f"_{_tag}" if _tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    tasks = [c for c in TASKS if not only or c["name"] in only]
    summary = []
    for cfg in tasks:
        print(f"\n{'='*70}\n### TASK: {cfg['name']}  ({cfg['shift']})\n{'='*70}")
        try:
            rec = run_one(cfg, llm)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {cfg['name']} FAILED: {e}")
            summary.append({"task": cfg["name"], "decision": "ERROR", "error": str(e)[:200]})
            continue
        with open(os.path.join(out_dir, cfg["name"] + ".json"), "w") as f:
            json.dump(rec, f, indent=2, default=str)
        if os.environ.get("LESSONS", "0") != "0":
            from harness.lessons import update_store_with_rec
            update_store_with_rec(os.path.join(PROJECT_ROOT, "experiments", "lessons", "lessons.jsonl"),
                                  rec, task_type="segmentation")
        traj = " -> ".join(e.get("tool", e.get("kind")) for e in rec["trajectory"]
                           if e["kind"] in ("tool", "terminal", "safety_veto"))
        print(f"  decision={rec['decision']}  source={rec['source_dice']} "
              f"shipped={rec['shipped_dice']} (delta {rec['delta_vs_source']:+}) "
              f"oracle={rec['oracle_dice']} safe={rec['safety_held']}")
        print(f"  trajectory: {traj}")
        summary.append({k: rec[k] for k in ("task", "shift", "decision", "source_dice",
                        "shipped_dice", "delta_vs_source", "oracle_dice", "safety_held",
                        "shipped_is_oracle", "steps")})
    with open(os.path.join(out_dir, "suite_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\n{'='*70}\n SUITE SUMMARY — smart choice per task?\n{'='*70}")
    print(f"{'task':32s} {'decision':11s} {'src':>6s} {'ship':>6s} {'delta':>7s} {'safe':>5s}")
    for s in summary:
        if s["decision"] == "ERROR":
            print(f"{s['task']:32s} ERROR"); continue
        print(f"{s['task']:32s} {s['decision']:11s} {s['source_dice']:6.3f} "
              f"{s['shipped_dice']:6.3f} {s['delta_vs_source']:+7.3f} {str(s['safety_held']):>5s}")
    print(f"\nFull per-task decision logic saved -> {out_dir}/<task>.json")


if __name__ == "__main__":
    main()
