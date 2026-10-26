#!/usr/bin/env python
"""Run the deployment harness across CLASSIFICATION tasks — proving the same harness +
LLM brain + K-shot + smart-escalation loop works for cls (AUROC), not just seg (Dice).
Saves full decision logic per task -> experiments/harness_decisions_cls/<task>.json.
"""

from pathlib import Path
import os, sys, json
import numpy as np
PROJECT_ROOT = os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, PROJECT_ROOT)
DEVICE = "cuda:0"

from harness.core import Tool, Budget, build_harness
from harness.cls_adapter import ClsTask, load_cls_model, load_cls_npz, _metric
from harness.llm_policy import make_llm_policy

TASKS = [
    dict(name="ham2isic_resnet18", shift="dermoscopy melanoma, HAM10000->ISIC2019 (cross-source)",
         ckpt="checkpoints/cls/ham10000_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/ham10000/src.npz", tgt="data/cls/isic2019/tgt.npz",
         classes="malignant vs benign skin lesion"),
    dict(name="eyepacs2aptos_resnet18", shift="DR fundus, EyePACS->APTOS (cross-camera)",
         ckpt="checkpoints/cls/eyepacs_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/eyepacs/src.npz", tgt="data/cls/aptos/tgt.npz",
         classes="referable vs non-referable diabetic retinopathy"),
    dict(name="aptos2eyepacs_resnet18", shift="DR fundus, APTOS->EyePACS (cross-camera)",
         ckpt="checkpoints/cls/aptos_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/aptos/src.npz", tgt="data/cls/eyepacs/tgt.npz",
         classes="referable vs non-referable diabetic retinopathy"),
    dict(name="camelyon17_resnet18",
         shift="histopathology H&E whole-slide tumor patches, Camelyon17-WILDS "
               "cross-hospital STAIN shift (train hosp{0,3,4} -> OOD test hospital 2, "
               "distinct staining protocol / scanner)",
         ckpt="checkpoints/cls/camelyon17_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/camelyon17/src.npz", tgt="data/cls/camelyon17/tgt.npz",
         classes="tumor vs normal lymph-node tissue patch"),
    # M4: same Camelyon17 cross-hospital STAIN shift, but a TRANSFORMER source
    # (timm vit_small_patch16_224, LayerNorm). bn_reset/tent are no-ops here;
    # the norm-agnostic norm_stat_tent (LayerNorm affines) is the lever to test.
    dict(name="camelyon17_vit_small",
         shift="histopathology H&E whole-slide tumor patches, Camelyon17-WILDS "
               "cross-hospital STAIN shift (train hosp{0,3,4} -> OOD test hospital 2, "
               "distinct staining protocol / scanner) — TRANSFORMER source (ViT/LayerNorm)",
         ckpt="checkpoints/cls/camelyon17_vit_small_best.pt", arch="vit_small", nc=2,
         src="data/cls/camelyon17/src.npz", tgt="data/cls/camelyon17/tgt.npz",
         classes="tumor vs normal lymph-node tissue patch"),
    # M5 (2026-06-23): SECOND transformer source on the SAME Camelyon17 cross-hospital
    # STAIN shift, but a hierarchical-window Swin Transformer (timm swin_tiny_patch4_window7_224,
    # LayerNorm only, ZERO BatchNorm). bn_reset/tent are STRUCTURAL no-ops here exactly as on
    # the ViT; the norm-agnostic transformer levers (norm_stat_tent_safe / t3a / norm_stat_tent)
    # are what the agent must reach. Proves the harness loads + characterizes a 2nd transformer.
    dict(name="camelyon17_swin_tiny",
         shift="histopathology H&E whole-slide tumor patches, Camelyon17-WILDS "
               "cross-hospital STAIN shift (train hosp{0,3,4} -> OOD test hospital 2, "
               "distinct staining protocol / scanner) — TRANSFORMER source (Swin-Tiny/LayerNorm)",
         ckpt="checkpoints/cls/camelyon17_swin_tiny_best.pt", arch="swin_tiny", nc=2,
         src="data/cls/camelyon17/src.npz", tgt="data/cls/camelyon17/tgt.npz",
         classes="tumor vs normal lymph-node tissue patch"),
    # ---- 10-task expansion (2026-06-17): 3 source-reusing real-world targets ----
    dict(name="aptos2messidor2_resnet18", shift="DR fundus, APTOS->Messidor-2 (France, TOPCON camera; strong cross-camera/grader break, source AUROC 0.985->0.730)",
         ckpt="checkpoints/cls/aptos_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/aptos/src.npz", tgt="data/cls/messidor2/tgt.npz",
         classes="referable vs non-referable diabetic retinopathy"),
    dict(name="aptos2ddr_resnet18", shift="DR fundus, APTOS->DDR (China, ~12.5k; cross-population break, source AUROC 0.985->0.827)",
         ckpt="checkpoints/cls/aptos_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/aptos/src.npz", tgt="data/cls/ddr/tgt.npz",
         classes="referable vs non-referable diabetic retinopathy"),
    dict(name="ham2padufes_resnet18", shift="dermatology cross-MODALITY, HAM10000 dermoscopy->PAD-UFES-20 smartphone clinical photos (large modality shift)",
         ckpt="checkpoints/cls/ham10000_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/ham10000/src.npz", tgt="data/cls/padufes/tgt.npz",
         classes="malignant melanoma vs benign/other skin lesion"),
    # ---- pathmnist (cross-source colon histopath) + chest (NIH->CheXpert cross-institution) ----
    dict(name="pathmnist_resnet18", shift="colon histopathology 9-class, NCT-CRC-HE-100K (German hospital) -> CRC-VAL-HE-7K (Mainz) cross-source",
         ckpt="checkpoints/cls/pathmnist_resnet18_best.pt", arch="resnet18", nc=9,
         src="data/cls/pathmnist/src.npz", tgt="data/cls/pathmnist/tgt.npz",
         classes="9-class colorectal tissue type"),
    dict(name="nih2chexpert_resnet18", shift="chest X-ray Cardiomegaly, NIH ChestX-ray14 (US) -> CheXpert (Stanford, US) cross-institution",
         ckpt="checkpoints/cls/nih_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/nih/src.npz", tgt="data/cls/chexpert/tgt.npz",
         classes="cardiomegaly present vs absent"),
    # ---- MEDICAL FOUNDATION TRANSFORMER source (2026-06-24): same NIH->CheXpert
    # cross-institution cardiomegaly shift, but the source is RAD-DINO
    # (microsoft/rad-dino, a DINOv2 ViT-B self-supervised on chest radiographs) —
    # a genuine medical foundation transformer vs the ImageNet timm ViT/Swin. Pure
    # LayerNorm (25 LN, ZERO BatchNorm), so bn_reset/tent are STRUCTURAL no-ops;
    # the norm-agnostic transformer levers are the adaptation path. Proves the
    # harness loads + characterizes + adapts a real foundation-model source on the
    # medical-foundation-on-medical-data match (CXR backbone on CXR task).
    dict(name="nih2chexpert_raddino", shift="chest X-ray Cardiomegaly, NIH ChestX-ray14 (US) -> CheXpert (Stanford, US) cross-institution — MEDICAL FOUNDATION TRANSFORMER source (RAD-DINO, DINOv2 ViT-B self-supervised on chest radiographs; LayerNorm)",
         ckpt="checkpoints/cls/nih_raddino_best.pt", arch="rad_dino", nc=2,
         src="data/cls/nih/src.npz", tgt="data/cls/chexpert/tgt.npz",
         classes="cardiomegaly present vs absent"),
    dict(name="breakhis_40to400_resnet18", shift="breast histopathology benign/malignant, BreakHis 40X->400X cross-MAGNIFICATION (objective-lens acquisition shift, patient-disjoint); source AUROC drops ~1.00->0.80",
         ckpt="checkpoints/cls/breakhis_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/breakhis/src.npz", tgt="data/cls/breakhis/tgt.npz",
         classes="malignant vs benign breast tumor tissue"),
    # ---- CRC tumor-vs-normal cross-CENTER/cross-COUNTRY STAIN shift (2026-06-21) ----
    # NCT-CRC-HE-100K (Germany, Macenko-norm) -> Chaoyang (Beijing, China; different
    # scanner + stain protocol). In-domain AUROC 0.999, on-target 0.727 (gap -0.272,
    # LARGER than Camelyon17). RECOVERABLE: bn_reset +0.026, finetune_K12 +0.065.
    dict(name="crc_nct2chaoyang_resnet18", shift="colorectal histopathology tumor-vs-normal, "
         "NCT-CRC-HE-100K (Heidelberg/Mannheim Germany, Macenko-normalized) -> Chaoyang "
         "(Beijing Chaoyang Hospital, China) cross-country/cross-center/cross-scanner STAIN "
         "shift; source AUROC drops 0.999->0.727 (gap 0.27); bn_reset +0.03, K12 finetune +0.07",
         ckpt="checkpoints/cls/crc_chaoyang_resnet18_best.pt", arch="resnet18", nc=2,
         src="data/cls/crc_chaoyang/src.npz", tgt="data/cls/crc_chaoyang/tgt.npz",
         classes="adenocarcinoma vs normal colorectal mucosa"),
]


def run_one(cfg, llm):
    base = load_cls_model(os.path.join(PROJECT_ROOT, cfg["ckpt"]), cfg["arch"], cfg["nc"], DEVICE)
    timgs, tlabs = load_cls_npz(os.path.join(PROJECT_ROOT, cfg["tgt"]))
    try:
        simgs, _ = load_cls_npz(os.path.join(PROJECT_ROOT, cfg["src"]))
    except Exception:
        simgs = timgs
    if tlabs is None:
        raise RuntimeError("target npz has no labels — cannot score")
    # EXPLORE_EVAL_N: in-loop candidate scoring uses a small stratified subset for SPEED; the
    # final audit below re-scores on the FULL timgs/tlabs (the honest delivered number).
    explore_eval_n = int(os.environ.get("EXPLORE_EVAL_N", "4000") or 0)
    task = ClsTask(base, simgs, timgs, cfg["nc"], DEVICE, tlabs=tlabs, metric="auroc",
                   task_desc=f"{cfg['shift']}; classifying {cfg['classes']}",
                   explore_eval_n=explore_eval_n)
    task.set_llm(llm)
    common = [Tool("ship", "deliver the chosen model", lambda st, **k: {}, terminal=True),
              Tool("keep_source", "keep the source model", lambda st, **k: {}, terminal=True),
              Tool("escalate", "hand back to clinician", lambda st, **k: {}, terminal=True)]
    # EXPLORATION-BUDGET FIX (2026-06, task #cls-cam17): the cls LABEL-FREE pool is LARGE
    # (bn_reset + tent + ~9 input-space color/stain tools = ~11 candidates, each costing one
    # n_trials). The prior n_trials=4 was the BINDING constraint — the agent could only build 4
    # candidates, then every further build was budget_refused (it looped on a refused tool to
    # max_steps and escalated, MISSING the +0.08 stain recovery). Raise n_trials >= pool size so
    # the WHOLE label-free pool can be built AND ranked, and raise max_steps / no_progress_cap to
    # >= 2x pool + diagnostics + request_labels + ship so building + ranking the full pool fits.
    # All env-overridable (per-regime tuning from the slurm). Mirrors the seg selection fix (#157).
    n_trials = int(os.environ.get("N_TRIALS", "14"))
    max_steps = int(os.environ.get("MAX_STEPS", "40"))
    no_progress_cap = int(os.environ.get("NO_PROGRESS_CAP", "20"))
    budget = Budget(compute_minutes=int(os.environ.get("COMPUTE_MIN","60")), n_trials=n_trials,
                    llm_usd=5.0, k_labels=int(os.environ.get("KLAB","12")))
    goal = (f"Deploy this {cfg['classes']} classifier at a new site ({cfg['shift']}) without "
            f"shipping a worse model (metric=AUROC); adapt within budget if it underperforms.")
    h = build_harness(make_llm_policy(lambda p: llm.chat(p)), task, common, budget,
                      max_steps=max_steps, no_progress_cap=no_progress_cap)
    out = h.run(goal=goal, source_model=base)

    from harness.registry_bridge import cls_metric_of
    audit = {}
    for b in task.built:
        a = float(cls_metric_of(b["model"], timgs, tlabs, cfg["nc"], DEVICE,
                                metric_kind="auroc", metric_fn=_metric))
        audit[b["name"]] = {"label_free_quality": round(b["quality"], 4),
                            "true_auroc": round(a, 4), "shipped": (b["model"] is out.delivered)}
    src_a = audit.get("source", {}).get("true_auroc", 0.5)
    ship_a = next((v["true_auroc"] for v in audit.values() if v["shipped"]), src_a)
    oracle_a = max((v["true_auroc"] for v in audit.values()), default=src_a)
    return {"task": cfg["name"], "shift": cfg["shift"], "metric": "AUROC", "nc": cfg["nc"],
            "decision": out.decision, "report": out.report, "steps": out.steps,
            "budget_spent": out.budget_spent.as_dict(), "labeled_auroc": task._labeled_dice,
            "trajectory": out.trace, "oracle_audit": audit,
            "source_auroc": round(src_a, 4), "shipped_auroc": round(ship_a, 4),
            "delta_vs_source": round(ship_a - src_a, 4), "oracle_auroc": round(oracle_a, 4),
            "safety_held": ship_a >= src_a - 1e-6, "shipped_is_oracle": abs(ship_a - oracle_a) < 1e-6,
            "agent_analysis": {k: out.evidence.get(k) for k in
                               ("domain_shift", "model_info", "failure_mode", "recommendation", "retrieved_lessons")}}


def main():
    from medadaptagent.knowledge.llm_client import create_llm_client
    model = os.environ.get("MEDADAPT_LLM_MODEL", os.environ.get("MODEL"))
    print(f"== cls suite | model {model} ==")
    llm = create_llm_client(model=model)
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    tasks = [c for c in TASKS if not only or c["name"] in only]
    _tag = os.environ.get("OUTDIR_TAG", "")
    out_dir = os.path.join(PROJECT_ROOT, "experiments",
                           "harness_decisions_cls" + (f"_{_tag}" if _tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for cfg in tasks:
        print(f"\n{'='*70}\n### CLS TASK: {cfg['name']} ({cfg['shift']})\n{'='*70}")
        try:
            rec = run_one(cfg, llm)
        except Exception as e:
            import traceback; traceback.print_exc()
            summary.append({"task": cfg["name"], "decision": "ERROR", "error": str(e)[:200]}); continue
        json.dump(rec, open(os.path.join(out_dir, cfg["name"] + ".json"), "w"), indent=2, default=str)
        if os.environ.get("LESSONS", "0") != "0":
            from harness.lessons import update_store_with_rec
            update_store_with_rec(os.path.join(PROJECT_ROOT, "experiments", "lessons", "lessons.jsonl"),
                                  rec, task_type="classification")
        traj = " -> ".join(e.get("tool", e.get("kind")) for e in rec["trajectory"]
                           if e["kind"] in ("tool", "terminal", "safety_veto"))
        print(f"  decision={rec['decision']} source_AUROC={rec['source_auroc']} "
              f"shipped={rec['shipped_auroc']} (d{rec['delta_vs_source']:+}) oracle={rec['oracle_auroc']} "
              f"safe={rec['safety_held']}")
        print(f"  trajectory: {traj}")
        summary.append({k: rec[k] for k in ("task", "shift", "decision", "source_auroc",
                        "shipped_auroc", "delta_vs_source", "oracle_auroc", "safety_held", "steps")})
    json.dump(summary, open(os.path.join(out_dir, "suite_summary.json"), "w"), indent=2, default=str)
    print(f"\n\n{'='*70}\n CLS SUITE SUMMARY — smart choice per task?\n{'='*70}")
    print(f"{'task':28s} {'decision':11s} {'srcAUC':>7s} {'shipAUC':>7s} {'delta':>7s} {'safe':>5s}")
    for s in summary:
        if s["decision"] == "ERROR":
            print(f"{s['task']:28s} ERROR {s.get('error','')[:50]}"); continue
        print(f"{s['task']:28s} {s['decision']:11s} {s['source_auroc']:7.3f} "
              f"{s['shipped_auroc']:7.3f} {s['delta_vs_source']:+7.3f} {str(s['safety_held']):>5s}")
    print(f"\nFull per-task decision logic -> {out_dir}/<task>.json")


if __name__ == "__main__":
    main()
