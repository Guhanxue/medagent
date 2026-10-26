#!/usr/bin/env python
"""Run the deployment harness across our 3D volumetric segmentation tasks, saving the
ENTIRE decision logic per task — the 3D sibling of scripts/run_harness_suite.py.

Mirrors the 2D runner: load source model + target/source volumes, build SegTask3D,
wire the pairwise VLM, run the SAME core.py harness with ship/keep_source/escalate
terminals, then compute the hidden-oracle 3D Dice per built candidate. Honors the same
env knobs (ONLY, KLAB, NO_FINETUNE, COMPUTE_MIN, SCALING_EVAL) so the budget regimes
match the 2D path one-for-one.

Saves experiments/harness_decisions_3d/<task>.json (full trajectory + decision + report
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
def _load_3d_volumes(img_dir, lbl_dir, max_n):
    """Inlined from the legacy sandbox (whose module-top import is broken). Loads up to
    max_n matching nii.gz volumes+labels as calibration.Volume (numpy + .affine)."""
    import nibabel as nib
    from medadaptagent.calibration import Volume
    names = sorted(f for f in os.listdir(img_dir) if f.endswith(".nii.gz") and not f.startswith("._"))
    names = names[:max_n] if max_n else names
    imgs, lbls = [], []
    for nm in names:
        inii = nib.as_closest_canonical(nib.load(os.path.join(img_dir, nm))); lnii = nib.as_closest_canonical(nib.load(os.path.join(lbl_dir, nm)))  # RAS reorient
        imgs.append(Volume.from_array(inii.get_fdata().astype(np.float32), inii.affine.astype(np.float32)))
        lbls.append(Volume.from_array(lnii.get_fdata().astype(np.int64), lnii.affine.astype(np.float32)))
    return imgs, lbls
from scripts.run_agent4_only import THREED_DATASETS
from harness.core import Tool, Budget, build_harness
from harness.seg3d_adapter import SegTask3D
from harness.llm_policy import make_llm_policy
from medadaptagent.tasks import get_task

# Which 3D datasets to run + their expected fg VOXEL fraction (spleen-in-abdomen is tiny).
# fg_prior lives HERE (per-dataset config), never as a magic constant inside the adapter.
# Default a_min/a_max omitted => read PER-DATASET from the source bundle's inference.json
# ScaleIntensityRanged below (window_for_dataset). Generic [-1024,1024] is WRONG for these
# bundles (spleen bundle = -57/164) and produced 0 Dice even on the model's own data.
TASKS_3D = [
    dict(name="totalseg_spleen_nc", fg_prior=0.006,
         shift="MSD-spleen UNet on TotalSegmentator NON-contrast abdominal CT (contrast shift)"),
    dict(name="msd_spleen", fg_prior=0.006,
         shift="MSD-spleen UNet on its own MSD spleen CT (no-shift baseline; keep source)"),
    dict(name="totalseg_spleen_ce", fg_prior=0.006,
         shift="MSD-spleen UNet on TotalSegmentator contrast-enhanced abdominal CT"),
    dict(name="synapse_spleen", fg_prior=0.006,
         shift="MSD-spleen UNet on BTCV/Synapse abdominal CT (cross-cohort)"),
    # --- SwinUNETR (BTCV 14-class TRANSFORMER segmenter) tasks ---------------
    # The swin bundle is loaded class-extracted to nc=2 (bg+spleen, BTCV ch 1).
    # window_for_dataset() reads ScaleIntensityRange -175/250 from the swin
    # bundle's own inference.json -> correct transformer preprocessing.
    dict(name="swinunetr_btcv_spleen_ce", fg_prior=0.006,
         shift="SwinUNETR (BTCV transformer) on TotalSegmentator CE abdominal CT "
               "(in-distribution-ish GATE / no-shift reference)"),
    dict(name="swinunetr_btcv_spleen_nc", fg_prior=0.006,
         shift="SwinUNETR (BTCV transformer, trained on CE CT) on TotalSegmentator "
               "NON-contrast abdominal CT (contrast->non-contrast shift)"),
    # --- UNETR (a SECOND 3D TRANSFORMER segmenter: pure-ViT global attention, vs
    # SwinUNETR's WINDOWED attention) ---------------------------------------
    # UNETR norms = ViT-encoder LayerNorm + InstanceNorm decoder, NO BatchNorm ->
    # bn_reset/tent are structural no-ops (toolbox() auto-detects 0 BN and emits
    # the "NO-OP (NO BatchNorm)" branch). norm_stat_tent / input-space TTA /
    # finetune are the real levers. window -57/164 read from the unetr bundle's
    # inference.json (UNETR trained from scratch on MSD spleen, out_channels=2).
    dict(name="unetr_btcv_spleen_ce", fg_prior=0.006,
         shift="UNETR (pure-ViT-encoder transformer) on TotalSegmentator CE abdominal "
               "CT (in-distribution-ish GATE / no-shift reference)"),
    dict(name="unetr_btcv_spleen_nc", fg_prior=0.006,
         shift="UNETR (pure-ViT-encoder transformer) on TotalSegmentator NON-contrast "
               "abdominal CT (contrast->non-contrast shift)"),
]


def window_for_dataset(spec):
    """Per-dataset intensity window + voxel spacing, read from the SOURCE bundle's
    configs/inference.json (ScaleIntensityRanged + Spacingd). Explicit a_min/a_max/pixdim on
    the spec override the bundle. USE_SPACING=1 enables the Spacingd resample (intensity-only
    by default — the dominant factor)."""
    a_min = spec.get("a_min"); a_max = spec.get("a_max"); pixdim = spec.get("pixdim")
    use_spacing = os.environ.get("USE_SPACING", "0") != "0"
    ds = THREED_DATASETS[spec["name"]]
    bundle_dir = os.path.dirname(os.path.dirname(os.path.join(PROJECT_ROOT, ds["ckpt"])))
    cfg_path = os.path.join(bundle_dir, "configs", "inference.json")
    if os.path.exists(cfg_path):
        try:
            import json as _json
            cfg = _json.load(open(cfg_path))
            for t in cfg.get("preprocessing", {}).get("transforms", []):
                tgt = str(t.get("_target_", ""))
                if "ScaleIntensityRange" in tgt:
                    if a_min is None:
                        a_min = t.get("a_min")
                    if a_max is None:
                        a_max = t.get("a_max")
                if "Spacing" in tgt and use_spacing and pixdim is None:
                    pd = t.get("pixdim")
                    if isinstance(pd, (list, tuple)):
                        pixdim = list(pd)
        except Exception:
            pass
    if a_min is None or a_max is None:        # fallback: generic CT soft-tissue-ish window
        a_min, a_max = -57.0, 164.0
    return float(a_min), float(a_max), (None if not use_spacing else pixdim)


def make_vlm(task, llm, timgs, nc, struct, fg_prior):
    from medadaptagent.graph.agents.execution_engineer import _build_vlm_task_ctx
    ctx = _build_vlm_task_ctx({"structures": struct, "foreground_priors": {}}, {})
    ctx["num_classes_hint"] = nc
    anchors = [timgs[i] for i in np.linspace(0, len(timgs) - 1, min(6, len(timgs)), dtype=int)]
    task.set_vlm(llm, ctx, get_task("segmentation_3d"), anchors)


def run_one(spec, llm):
    ds = THREED_DATASETS[spec["name"]]
    struct = ds["cfg"]["structures"]
    nc = ds["nc"]
    base = load_source_model(ds["ckpt"], nc, DEVICE, arch=ds["arch"], project_root=PROJECT_ROOT)
    max_t = ds.get("max_target", 6)
    timgs, tlbls = _load_3d_volumes(os.path.join(PROJECT_ROOT, ds["tgt"]),
                                    os.path.join(PROJECT_ROOT, ds["tgt_labels"]), max_t)
    try:
        simgs, _ = _load_3d_volumes(os.path.join(PROJECT_ROOT, ds["src"]),
                                    os.path.join(PROJECT_ROOT, ds["src_labels"]),
                                    min(4, max_t))
    except Exception:
        simgs = timgs
    task_desc = (f"{spec['shift']}; segmenting {list(struct.keys())[1:]} (3D volumetric, "
                 f"deploying {os.path.basename(ds['ckpt'])} on {ds['tgt']})")
    a_min, a_max, pixdim = window_for_dataset(spec)
    print(f"  [preproc] window=[{a_min},{a_max}] pixdim={pixdim} "
          f"(from {os.path.basename(os.path.dirname(os.path.dirname(ds['ckpt'])))} bundle)",
          flush=True)
    task = SegTask3D(base, simgs, timgs, nc, DEVICE, fg_prior=spec["fg_prior"],
                     tlbls=tlbls, task_desc=task_desc,
                     a_min=a_min, a_max=a_max, pixdim=pixdim)
    make_vlm(task, llm, timgs, nc, struct, spec["fg_prior"])

    common = [Tool("ship", "deliver the chosen model", lambda st, **k: {}, terminal=True),
              Tool("keep_source", "keep the source model", lambda st, **k: {}, terminal=True),
              Tool("escalate", "hand back to clinician", lambda st, **k: {}, terminal=True)]
    klab = int(os.environ.get("KLAB", "12"))
    budget = Budget(compute_minutes=int(os.environ.get("COMPUTE_MIN", "60")),
                    n_trials=4, llm_usd=5.0, k_labels=klab)
    nf = os.environ.get("NO_FINETUNE", "0") != "0"
    goal = (f"Deploy this {struct} 3D segmenter at a new site ({spec['shift']}) without "
            f"shipping a worse model; adapt within budget if it underperforms."
            + (" Labels may be used to SELECT among candidates, but do NOT fine-tune "
               "(finetune_* tools are disabled)." if nf else ""))
    h = build_harness(make_llm_policy(lambda p: llm.chat(p)), task, common, budget,
                      max_steps=24, no_progress_cap=10)
    out = h.run(goal=goal, source_model=base)

    # hidden-oracle audit (true 3D Dice the agent never saw)
    audit = {}
    for b in task.built:
        d = task._dice_of_3d(b["model"], timgs, tlbls)
        audit[b["name"]] = {"label_free_quality": round(b["quality"], 4),
                            "vlm_prefer": (None if b.get("vlm_prefer") is None else round(b["vlm_prefer"], 3)),
                            "true_dice": round(d, 4), "shipped": (b["model"] is out.delivered)}
    src_d = audit.get("source", {}).get("true_dice", 0.0)
    ship_d = next((a["true_dice"] for a in audit.values() if a["shipped"]), src_d)
    oracle_d = max((a["true_dice"] for a in audit.values()), default=src_d)
    rec = {"task": spec["name"], "shift": spec["shift"], "ckpt": ds["ckpt"], "tgt": ds["tgt"],
           "nc": nc, "structures": struct, "fg_prior": spec["fg_prior"],
           "regime": ("K0" if klab == 0 else ("K%d_NO_FINETUNE" % klab if nf else "K%d_finetune" % klab)),
           "decision": out.decision, "report": out.report, "steps": out.steps,
           "budget_spent": out.budget_spent.as_dict(),
           "labeled_dice": task._labeled_dice,
           "trajectory": out.trace,
           "oracle_audit": audit,
           "source_dice": round(src_d, 4), "shipped_dice": round(ship_d, 4),
           "delta_vs_source": round(ship_d - src_d, 4), "oracle_dice": round(oracle_d, 4),
           "safety_held": ship_d >= src_d - 1e-6, "shipped_is_oracle": abs(ship_d - oracle_d) < 1e-6,
           "agent_analysis": {k: out.evidence.get(k) for k in
                              ("domain_shift", "model_info", "failure_mode", "recommendation",
                               "retrieved_lessons")}}
    return rec


def main():
    from medadaptagent.knowledge.llm_client import create_llm_client
    model = os.environ.get("MEDADAPT_LLM_MODEL", os.environ.get("MODEL"))
    print(f"== LLM brain + VLM model: {model} ==", flush=True)
    llm = create_llm_client(model=model)
    # OUT_DIR env override (additive, backward-compatible): lets a run write to a
    # caller-owned dir when the default harness_decisions_3d is owned by another user
    # (e.g. jbao) and not group-writable for overwrite.
    out_dir = os.environ.get("OUT_DIR") or os.path.join(
        PROJECT_ROOT, "experiments", "harness_decisions_3d")
    os.makedirs(out_dir, exist_ok=True)
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    tasks = [c for c in TASKS_3D if not only or c["name"] in only]
    summary = []
    for spec in tasks:
        print(f"\n{'='*70}\n### 3D TASK: {spec['name']}  ({spec['shift']})\n{'='*70}", flush=True)
        try:
            rec = run_one(spec, llm)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {spec['name']} FAILED: {e}", flush=True)
            summary.append({"task": spec["name"], "decision": "ERROR", "error": str(e)[:200]})
            continue
        with open(os.path.join(out_dir, spec["name"] + ".json"), "w") as f:
            json.dump(rec, f, indent=2, default=str)
        traj = " -> ".join(e.get("tool", e.get("kind")) for e in rec["trajectory"]
                           if e["kind"] in ("tool", "terminal", "safety_veto", "forced_terminal"))
        print(f"  regime={rec['regime']} decision={rec['decision']}  source={rec['source_dice']} "
              f"shipped={rec['shipped_dice']} (delta {rec['delta_vs_source']:+}) "
              f"oracle={rec['oracle_dice']} safe={rec['safety_held']}", flush=True)
        print(f"  trajectory: {traj}", flush=True)
        summary.append({k: rec[k] for k in ("task", "shift", "regime", "decision", "source_dice",
                        "shipped_dice", "delta_vs_source", "oracle_dice", "safety_held",
                        "shipped_is_oracle", "steps")})
    with open(os.path.join(out_dir, "suite_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\n{'='*70}\n 3D SUITE SUMMARY — smart choice per task?\n{'='*70}", flush=True)
    print(f"{'task':24s} {'regime':16s} {'decision':11s} {'src':>6s} {'ship':>6s} {'delta':>7s} {'safe':>5s}")
    for s in summary:
        if s["decision"] == "ERROR":
            print(f"{s['task']:24s} ERROR"); continue
        print(f"{s['task']:24s} {s['regime']:16s} {s['decision']:11s} {s['source_dice']:6.3f} "
              f"{s['shipped_dice']:6.3f} {s['delta_vs_source']:+7.3f} {str(s['safety_held']):>5s}")
    print(f"\nFull per-task 3D decision logic saved -> {out_dir}/<task>.json")


if __name__ == "__main__":
    main()
