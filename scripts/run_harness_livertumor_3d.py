#!/usr/bin/env python
"""Run the deployment harness on the 3D liver-TUMOR recovery task — the lesion
sibling of scripts/run_harness_suite_3d.py.

Loads the 3-class (bg/liver/TUMOR) MONAI SegResNet source model + the LiTS
thick-slice TARGET (n=47, domain_split.json["target_eval"]), applies the low-
dose+recon corruption to the TARGET images ONLY (clean source reference), builds
SegTask3DLiverTumor (decision metric = TUMOR class-2 Dice), wires the pairwise
VLM, runs the SAME core.py harness, then reports the hidden-oracle 3D tumor Dice.

Honors the SAME env knobs as the 3D suite runner: KLAB, NO_FINETUNE, COMPUTE_MIN,
SCALING_EVAL, MODEL — so the budget regimes match the rest of the matrix.

Budget matrix (run one cell per sbatch via env):
  KLAB=0                          -> K=0 label-free
  KLAB=12 NO_FINETUNE=1           -> K=12 selection-only
  KLAB=12                         -> K=12 few-shot finetune (recovery cell)
  KLAB=12 SCALING_EVAL=1          -> K=12 finetune + CV scaling gate

Saves experiments/harness_decisions_livertumor_3d/liver_tumor_<regime>.json.
"""

from pathlib import Path
import os, sys, json
import numpy as np
PROJECT_ROOT = os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, PROJECT_ROOT)
DEVICE = "cuda:0"

from scripts._source_loader import load_source_model
from harness.core import Tool, Budget, build_harness
from harness.seg3d_livertumor_adapter import SegTask3DLiverTumor, corrupt_target_volumes
from harness.llm_policy import make_llm_policy
from medadaptagent.tasks import get_task

DATA = os.path.join(PROJECT_ROOT, "data/3d/msd_liver/Task03_Liver")
IMG = os.path.join(DATA, "imagesTr")
LBL = os.path.join(DATA, "labelsTr")
SPLIT = os.path.join(PROJECT_ROOT, "data/3d/msd_liver/domain_split.json")
CKPT_REL = "checkpoints/segresnet_liver_source_best.pt"
NC = 3                       # 0=bg, 1=liver, 2=tumor
TARGET_CLASS = 2             # tumor
# expected TARGET tumor VOXEL fraction (tiny — lesion in a CT volume). Lives HERE
# (task config), never a magic constant inside the adapter.
TUMOR_FG_PRIOR = 0.0015
STRUCT = {"background": 0, "liver": 1, "tumor": 2}
SHIFT = ("3-class liver/TUMOR SegResNet on LiTS thick-slice CT under a low-dose + "
         "recon-kernel acquisition shift (sigma_hu=40, blur_sigma=0.7); the TUMOR "
         "(class 2) Dice degrades from source 0.531 to ~0.377 — recover the lesion.")


def _load_vol(path, dtype):
    import nibabel as nib
    from medadaptagent.calibration import Volume
    nii = nib.as_closest_canonical(nib.load(path))     # RAS reorient
    return Volume.from_array(nii.get_fdata().astype(dtype), nii.affine.astype(np.float32))


def _load_split_volumes(files, want_labels=True):
    imgs, lbls = [], []
    for nm in files:
        imgs.append(_load_vol(os.path.join(IMG, nm), np.float32))
        if want_labels:
            lbls.append(_load_vol(os.path.join(LBL, nm), np.int64))
    return imgs, (lbls if want_labels else None)


def make_vlm(task, llm, timgs):
    from medadaptagent.graph.agents.execution_engineer import _build_vlm_task_ctx
    ctx = _build_vlm_task_ctx({"structures": STRUCT, "foreground_priors": {}}, {})
    ctx["num_classes_hint"] = NC
    anchors = [timgs[i] for i in np.linspace(0, len(timgs) - 1, min(6, len(timgs)), dtype=int)]
    task.set_vlm(llm, ctx, get_task("segmentation_3d"), anchors)


def run_one(llm):
    with open(SPLIT) as f:
        split = json.load(f)
    tgt_files = split["target_eval"]                  # n=47 LiTS thick-slice
    src_files = split["source_holdout"][:4]           # clean source reference (shift analysis)

    base = load_source_model(CKPT_REL, NC, DEVICE, arch="monai_segresnet3d",
                             project_root=PROJECT_ROOT)

    timgs_clean, tlbls = _load_split_volumes(tgt_files, want_labels=True)
    # >>> THE SHIFT: corrupt the TARGET images only (labels untouched, source clean).
    timgs = corrupt_target_volumes(timgs_clean, sigma_hu=40.0, blur_sigma=0.7, seed=1234)
    # free the clean copies immediately (47 full LiTS volumes -> host-RAM OOM guard):
    # only the corrupted target images + labels are needed downstream.
    del timgs_clean
    import gc; gc.collect()
    print(f"  [shift] corrupted {len(timgs)} target images (low-dose sigma_hu=40, blur=0.7)",
          flush=True)
    simgs, _ = _load_split_volumes(src_files, want_labels=False)

    task = SegTask3DLiverTumor(base, simgs, timgs, NC, DEVICE, fg_prior=TUMOR_FG_PRIOR,
                               tlbls=tlbls, task_desc=SHIFT, a_min=-21.0, a_max=189.0,
                               pixdim=(1.5, 1.5, 2.0), target_class=TARGET_CLASS)
    make_vlm(task, llm, timgs)

    common = [Tool("ship", "deliver the chosen model", lambda st, **k: {}, terminal=True),
              Tool("keep_source", "keep the source model", lambda st, **k: {}, terminal=True),
              Tool("escalate", "hand back to clinician", lambda st, **k: {}, terminal=True)]
    klab = int(os.environ.get("KLAB", "12"))
    budget = Budget(compute_minutes=int(os.environ.get("COMPUTE_MIN", "60")),
                    n_trials=4, llm_usd=5.0, k_labels=klab)
    nf = os.environ.get("NO_FINETUNE", "0") != "0"
    goal = (f"Deploy this liver/TUMOR 3D segmenter at a new low-dose site ({SHIFT}) without "
            f"shipping a worse model; the DECISION metric is TUMOR (class 2) Dice. Adapt "
            f"within budget if the lesion underperforms."
            + (" Labels may be used to SELECT among candidates, but do NOT fine-tune "
               "(finetune_* tools are disabled)." if nf else ""))
    h = build_harness(make_llm_policy(lambda p: llm.chat(p)), task, common, budget,
                      max_steps=24, no_progress_cap=10)
    out = h.run(goal=goal, source_model=base)

    # ------------------------------------------------------------------ DISJOINT held-out oracle
    # IN-SAMPLE-CONTAMINATION FIX (2026-06-11): the hidden-oracle audit MUST score the
    # candidates ONLY on the TARGET volumes that were NOT used to FINE-TUNE — exactly like the
    # spleen/foundation adapters' disjoint held-out (run_foundation_spleen FIRST n_ft fine-tune /
    # NEXT n_eval disjoint; phaseB_selection rank_names vs held_names). The OLD code scored every
    # candidate on ALL 47 target volumes, but at K=24 the finetune trained on ~16 of those 47, so
    # the headline delta/oracle were IN-SAMPLE inflated (finetune_head 0.6122 / +0.141 instead of
    # the honest held-out ~0.54 / ~+0.07 already in labeled_tumor_dice).
    #
    # The harness handed ALL `timgs`/`tlbls` to the task; the adapter's _ensure_labels carved out
    # the finetune TRAIN split into task._train[0] (the SAME Volume objects from `timgs`). The
    # honest oracle set is therefore every (img,lbl) pair whose IMAGE is NOT in that train split.
    # With no labels revealed (K=0 / NO_FINETUNE) nothing was trained on the target, so the FULL
    # target set is already a valid held-out and we keep all 47.
    train_imgs = list((getattr(task, "_train", None) or ([], []))[0])
    train_ids = {id(v) for v in train_imgs}                 # object identity (same Volumes as timgs)
    held_pairs = [(im, lb) for im, lb in zip(timgs, tlbls) if id(im) not in train_ids]
    if not held_pairs:                                      # degenerate guard (should not happen)
        held_pairs = list(zip(timgs, tlbls))
    himgs = [p[0] for p in held_pairs]
    hlbls = [p[1] for p in held_pairs]
    n_trained = len(train_imgs)
    print(f"  [oracle] DISJOINT held-out: {len(himgs)}/{len(timgs)} target volumes "
          f"({n_trained} excluded — used to fine-tune); scoring all candidates on the held-out only",
          flush=True)

    # hidden-oracle audit (true 3D TUMOR Dice the agent never saw) + liver sanity — HELD-OUT ONLY.
    audit = {}
    for b in task.built:
        d = task._dice_of_3d(b["model"], himgs, hlbls)              # TUMOR Dice (held-out)
        ld = task.liver_dice_of(b["model"], himgs, hlbls)           # liver sanity (held-out)
        audit[b["name"]] = {"label_free_quality": round(b["quality"], 4),
                            "vlm_prefer": (None if b.get("vlm_prefer") is None
                                           else round(b["vlm_prefer"], 3)),
                            "true_tumor_dice": round(d, 4),
                            "true_liver_dice": round(ld, 4),
                            "shipped": (b["model"] is out.delivered)}
    src_d = audit.get("source", {}).get("true_tumor_dice", 0.0)
    src_liver = audit.get("source", {}).get("true_liver_dice", 0.0)
    ship_d = next((a["true_tumor_dice"] for a in audit.values() if a["shipped"]), src_d)
    ship_liver = next((a["true_liver_dice"] for a in audit.values() if a["shipped"]), src_liver)
    oracle_d = max((a["true_tumor_dice"] for a in audit.values()), default=src_d)
    rec = {"task": "liver_tumor", "shift": SHIFT, "ckpt": CKPT_REL,
           "nc": NC, "structures": STRUCT, "target_class": TARGET_CLASS,
           "fg_prior": TUMOR_FG_PRIOR, "n_target": len(timgs),
           # honest disjoint-held-out accounting: how many target vols the oracle scored on,
           # and how many were EXCLUDED because the finetune trained on them.
           "n_oracle_heldout": len(himgs), "n_finetune_excluded": n_trained,
           "oracle_is_disjoint_heldout": True,
           "regime": ("K0" if klab == 0 else ("K%d_NO_FINETUNE" % klab if nf
                       else ("K%d_finetune_scaling" % klab
                             if os.environ.get("SCALING_EVAL", "0") == "1"
                             else "K%d_finetune" % klab))),
           "decision": out.decision, "report": out.report, "steps": out.steps,
           "budget_spent": out.budget_spent.as_dict(),
           "labeled_tumor_dice": task._labeled_dice,
           "trajectory": out.trace, "oracle_audit": audit,
           "source_tumor_dice": round(src_d, 4), "shipped_tumor_dice": round(ship_d, 4),
           "delta_tumor_vs_source": round(ship_d - src_d, 4),
           "oracle_tumor_dice": round(oracle_d, 4),
           "source_liver_dice": round(src_liver, 4), "shipped_liver_dice": round(ship_liver, 4),
           "safety_held": ship_d >= src_d - 1e-6,
           "shipped_is_oracle": abs(ship_d - oracle_d) < 1e-6,
           "agent_analysis": {k: out.evidence.get(k) for k in
                              ("domain_shift", "model_info", "failure_mode",
                               "recommendation", "retrieved_lessons")}}
    return rec


def main():
    from medadaptagent.knowledge.llm_client import create_llm_client
    model = os.environ.get("MEDADAPT_LLM_MODEL", os.environ.get("MODEL"))
    print(f"== LLM brain + VLM model: {model} ==", flush=True)
    llm = create_llm_client(model=model)
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "harness_decisions_livertumor_3d")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}\n### 3D TASK: liver_tumor ({SHIFT})\n{'='*70}", flush=True)
    rec = run_one(llm)
    fname = "liver_tumor_%s.json" % rec["regime"]
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    traj = " -> ".join(e.get("tool", e.get("kind")) for e in rec["trajectory"]
                       if e["kind"] in ("tool", "terminal", "safety_veto", "forced_terminal"))
    print(f"\n  regime={rec['regime']} decision={rec['decision']}  "
          f"src_tumor={rec['source_tumor_dice']} shipped_tumor={rec['shipped_tumor_dice']} "
          f"(delta {rec['delta_tumor_vs_source']:+}) oracle_tumor={rec['oracle_tumor_dice']} "
          f"safe={rec['safety_held']}", flush=True)
    print(f"  liver (sanity): src={rec['source_liver_dice']} shipped={rec['shipped_liver_dice']}",
          flush=True)
    print(f"  trajectory: {traj}", flush=True)
    print(f"\nSaved -> {os.path.join(out_dir, fname)}", flush=True)


if __name__ == "__main__":
    main()
