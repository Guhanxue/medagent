#!/usr/bin/env python
"""Run the deployment harness on the 4-channel-MRI / 3-region BraTS 3D segmentation task,
saving the ENTIRE decision logic per regime — the cross-MODALITY 3D sibling of
scripts/run_harness_suite_3d.py.

THE SHIFT (constructed, since the MSD data IS the bundle's training distribution): the
TARGET is the same volumes with the FLAIR channel ZEROED ("deployment site doesn't acquire
FLAIR"). FLAIR is the dominant cue for Whole Tumor, so dropping it degrades source Dice but
is recoverable by fine-tuning the model to work without it. SOURCE (full 4-ch) = no-shift
control. simgs = full-4ch volumes; timgs = FLAIR-zeroed volumes; tlbls = the region labels.

Mirrors the 1-ch CT runner: load the bundle's 4-ch SegResNet via load_source_model, build
SegTask3DMRI, wire an MRI-aware pairwise VLM, run the SAME unchanged core.py harness, then
compute the hidden-oracle region Dice per built candidate. Honors the same env knobs
(ONLY/KLAB/NO_FINETUNE/COMPUTE_MIN/SCALING_EVAL). Writes to a jbao-writable NEW dir:
experiments/harness_decisions_3d_mri/<task>.json + suite_summary.json.

CORRECTNESS env: set CORRECTNESS_ONLY=1 to ONLY run the pure-torch correctness check
(full-4ch source Dice >= 0.75 + FLAIR-dropped source Dice) and exit — no Azure key needed.
"""

from pathlib import Path
import os, sys, json
import numpy as np

PROJECT_ROOT = os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, PROJECT_ROOT)
DEVICE = "cuda:0" if os.environ.get("FORCE_CPU", "0") == "0" else "cpu"

from scripts._source_loader import load_source_model
from harness.seg3d_mri_adapter import SegTask3DMRI, _REGION_SETS

# ----------------------------------------------------------------------------- CONFIG
# Per-task config lives HERE (never magic constants in the adapter). One BraTS task with
# a FLAIR-dropout shift. nc = 3 REGIONS (sigmoid). fg_prior ~ tumor voxel fraction.
# msd_channel_order: MSD Task01 stores [FLAIR, T1w, T1gd, T2w]; the bundle expects
# [T1c, T1, T2, FLAIR] -> reorder = [2, 1, 3, 0]. flair_bundle_idx = 3 (zeroed in target).
TASKS_3D_MRI = [
    dict(name="brats_mri", nc=3, fg_prior=0.02,
         data="data/brats/raw/Task01_BrainTumour",
         msd_channel_order=[2, 1, 3, 0], flair_bundle_idx=3, n_vol=8,
         ckpt="data/monai_bundles/brats_mri_segmentation/models/model.pt",
         roi=[224, 224, 144],
         structures={"background": 0, "TC": 1, "WT": 2, "ET": 3},
         shift="BraTS 4-ch MRI SegResNet [T1c,T1,T2,FLAIR] deployed where FLAIR is NOT "
               "acquired (FLAIR-dropout) — cross-modality/missing-channel shift; FLAIR is "
               "the dominant Whole-Tumor cue, so source Dice degrades but is recoverable"),
]


# ----------------------------------------------------------------------------- DATA LOADER
def _load_brats_volumes(data_dir, names, ch_order):
    """Load BraTS MSD volumes as numpy [4, D, H, W] (channels-first, REORDERED to the
    bundle's [T1c, T1, T2, FLAIR] order) + integer labels [D, H, W] {0,1,2,3}. Skips the
    AppleDouble ._* files. ch_order maps bundle channel i -> MSD source channel ch_order[i]."""
    import nibabel as nib
    imgs, lbls = [], []
    for nm in names:
        im = nib.load(os.path.join(data_dir, "imagesTr", nm)).get_fdata().astype(np.float32)
        lb = nib.load(os.path.join(data_dir, "labelsTr", nm)).get_fdata().astype(np.int64)
        # im is [D, H, W, 4] (channels-last MSD) -> [4, D, H, W] reordered to bundle order
        im = np.transpose(im, (3, 0, 1, 2))                 # [4, D, H, W] in MSD order
        im = im[ch_order]                                   # -> bundle order [T1c,T1,T2,FLAIR]
        imgs.append(np.ascontiguousarray(im))
        lbls.append(np.ascontiguousarray(lb))
    return imgs, lbls


def _flair_dropout(vols, flair_idx):
    """Return copies with the FLAIR channel zeroed (the constructed deployment shift)."""
    out = []
    for v in vols:
        w = v.copy()
        w[flair_idx] = 0.0
        out.append(np.ascontiguousarray(w))
    return out


def _list_names(data_dir, n):
    img_dir = os.path.join(data_dir, "imagesTr")
    names = sorted(f for f in os.listdir(img_dir)
                   if f.endswith(".nii.gz") and not f.startswith("._"))
    return names[:n]


# ----------------------------------------------------------------------------- MRI VLM TASK
def _make_mri_vlm_task():
    """An MRI-aware SegmentationTask3D for the pairwise-VLM render: 4-channel volumes
    [4,D,H,W], sigmoid->0.5 region masks. Subclass keeps core.py/seg_adapter.py unchanged;
    only render_pair_for_vlm / render_prediction_panel are overridden for the brats output.
    A display channel (T1c=0) provides the grayscale background per view."""
    from medadaptagent.tasks.segmentation_3d import (
        SegmentationTask3D, _b64_png, _overlay_mask_on_slice, SEG3D_VLM_PROMPT,
        _sliding_window_predict)
    import torch

    DISPLAY_CH = 0          # T1c for the grayscale background
    ROI = (224, 224, 144)

    def _region_argmax_from_probs(probs):
        """3-channel sigmoid probs [3,D,H,W] -> a single integer overlay mask [D,H,W] with
        1=TC, 2=WT-only(edema), 3=ET, for display. Painted in WT < TC < ET precedence so the
        most specific region wins visually."""
        m = np.zeros(probs.shape[1:], dtype=np.uint8)
        wt = probs[1] >= 0.5; tc = probs[0] >= 0.5; et = probs[2] >= 0.5
        m[wt] = 2; m[tc] = 1; m[et] = 3
        return m

    def _znorm_disp(vol):
        ch = np.asarray(vol)[DISPLAY_CH].astype(np.float32)
        lo, hi = float(ch.min()), float(ch.max())
        return (ch - lo) / (hi - lo + 1e-6)

    class _MRIVLMTask(SegmentationTask3D):
        def _probs(self, model, vol):
            x = torch.from_numpy(self._znorm4(vol)).unsqueeze(0).to(self._dev)
            logits = _sliding_window_predict(model, x, patch=ROI, batch_size=1, overlap=0.5)
            return torch.sigmoid(logits)[0].detach().cpu().numpy()

        def _znorm4(self, vol):
            arr = np.asarray(vol).astype(np.float32)
            out = np.empty_like(arr)
            for c in range(arr.shape[0]):
                mask = arr[c] != 0
                if mask.any():
                    out[c] = np.where(mask, (arr[c] - arr[c][mask].mean())
                                      / (arr[c][mask].std() + 1e-8), 0.0)
                else:
                    out[c] = 0.0
            return out

        def render_pair_for_vlm(self, model_source, model_current, anchor_imgs,
                                num_classes, device, size=256):
            self._dev = device
            images, labels = [], []
            for i, vol in enumerate(anchor_imgs):
                src = _region_argmax_from_probs(self._probs(model_source, vol))
                cur = _region_argmax_from_probs(self._probs(model_current, vol))
                disp = _znorm_disp(vol)
                D, H, W = disp.shape
                views = [("axial", D // 2, lambda a: (disp[a], src[a], cur[a])),
                         ("coronal", H // 2, lambda a: (disp[:, a, :], src[:, a, :], cur[:, a, :])),
                         ("sagittal", W // 2, lambda a: (disp[:, :, a], src[:, :, a], cur[:, :, a]))]
                for vname, idx, getter in views:
                    raw, s_m, c_m = getter(idx)
                    raw_n = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
                    images.append(_b64_png(raw_n, size=size))
                    labels.append(f"Sample {i+1} RAW {vname} mid-slice")
                    images.append(_b64_png(_overlay_mask_on_slice(raw, s_m, 4), size=size))
                    labels.append(f"Sample {i+1} SOURCE {vname} mask overlay")
                    images.append(_b64_png(_overlay_mask_on_slice(raw, c_m, 4), size=size))
                    labels.append(f"Sample {i+1} CURRENT {vname} mask overlay")
            return images, labels

        def vlm_prompt_template(self):
            return SEG3D_VLM_PROMPT

    return _MRIVLMTask()


def make_vlm(task, llm, timgs, nc, struct, fg_prior):
    from medadaptagent.graph.agents.execution_engineer import _build_vlm_task_ctx
    ctx = _build_vlm_task_ctx({"structures": struct, "foreground_priors": {}}, {})
    ctx["num_classes_hint"] = 4         # 4 display labels (bg + TC/WT/ET) for the overlay
    anchors = [timgs[i] for i in np.linspace(0, len(timgs) - 1, min(6, len(timgs)), dtype=int)]
    task.set_vlm(llm, ctx, _make_mri_vlm_task(), anchors)


# ----------------------------------------------------------------------------- CORRECTNESS
def correctness_check(spec):
    """Pure-torch (no Azure key) correctness gate: full-4ch source mean-region Dice (must
    reach >= 0.75) + per-region + FLAIR-dropped source Dice (the shift magnitude)."""
    data_dir = os.path.join(PROJECT_ROOT, spec["data"])
    names = _list_names(data_dir, spec["n_vol"])
    print(f"  loading {len(names)} BraTS volumes: {names}", flush=True)
    full, lbls = _load_brats_volumes(data_dir, names, spec["msd_channel_order"])
    dropped = _flair_dropout(full, spec["flair_bundle_idx"])
    base = load_source_model(spec["ckpt"], spec["nc"], DEVICE, arch="monai_bundle",
                             project_root=PROJECT_ROOT)
    task = SegTask3DMRI(base, full, full, spec["nc"], DEVICE, fg_prior=spec["fg_prior"],
                        tlbls=lbls, task_desc=spec["shift"], roi=tuple(spec["roi"]))
    full_per = task._dice_3d_per_region(base, full, lbls)
    drop_per = task._dice_3d_per_region(base, dropped, lbls)
    print(f"\n  == CORRECTNESS (pure torch, no key) ==", flush=True)
    print(f"  FULL-4ch source Dice: mean={full_per['mean']}  "
          f"TC={full_per['TC']} WT={full_per['WT']} ET={full_per['ET']}", flush=True)
    print(f"  FLAIR-DROPPED source Dice: mean={drop_per['mean']}  "
          f"TC={drop_per['TC']} WT={drop_per['WT']} ET={drop_per['ET']}", flush=True)
    gate = full_per["mean"] >= 0.75
    print(f"  GATE full>=0.75: {'PASS' if gate else 'FAIL'}  "
          f"shift magnitude (full-dropped) = {round(full_per['mean'] - drop_per['mean'], 4)}",
          flush=True)
    return {"full": full_per, "dropped": drop_per, "gate_pass": gate}


# ----------------------------------------------------------------------------- HARNESS RUN
def run_one(spec, llm):
    from harness.core import Tool, Budget, build_harness
    from harness.llm_policy import make_llm_policy
    data_dir = os.path.join(PROJECT_ROOT, spec["data"])
    names = _list_names(data_dir, spec["n_vol"])
    full, lbls = _load_brats_volumes(data_dir, names, spec["msd_channel_order"])
    dropped = _flair_dropout(full, spec["flair_bundle_idx"])   # TARGET = FLAIR-zeroed
    simgs = full[:min(4, len(full))]                            # SOURCE control = full 4-ch
    base = load_source_model(spec["ckpt"], spec["nc"], DEVICE, arch="monai_bundle",
                             project_root=PROJECT_ROOT)
    struct = spec["structures"]; nc = spec["nc"]
    task_desc = (f"{spec['shift']}; segmenting BraTS regions {[k for k in struct if k!='background']} "
                 f"(3D volumetric, 4-ch MRI in / 3-ch sigmoid out)")
    task = SegTask3DMRI(base, simgs, dropped, nc, DEVICE, fg_prior=spec["fg_prior"],
                        tlbls=lbls, task_desc=task_desc, roi=tuple(spec["roi"]))
    make_vlm(task, llm, dropped, nc, struct, spec["fg_prior"])

    common = [Tool("ship", "deliver the chosen model", lambda st, **k: {}, terminal=True),
              Tool("keep_source", "keep the source model", lambda st, **k: {}, terminal=True),
              Tool("escalate", "hand back to clinician", lambda st, **k: {}, terminal=True)]
    klab = int(os.environ.get("KLAB", "12"))
    budget = Budget(compute_minutes=int(os.environ.get("COMPUTE_MIN", "60")),
                    n_trials=4, llm_usd=5.0, k_labels=klab)
    nf = os.environ.get("NO_FINETUNE", "0") != "0"
    goal = (f"Deploy this BraTS 3D segmenter at a new site ({spec['shift']}) without shipping "
            f"a worse model; adapt within budget if it underperforms."
            + (" Labels may be used to SELECT among candidates, but do NOT fine-tune "
               "(finetune_* tools are disabled)." if nf else ""))
    h = build_harness(make_llm_policy(lambda p: llm.chat(p)), task, common, budget,
                      max_steps=24, no_progress_cap=10)
    out = h.run(goal=goal, source_model=base)

    # hidden-oracle audit (true region Dice on the FLAIR-dropout target the agent never saw)
    audit = {}
    for b in task.built:
        d = task._dice_3d_mri(b["model"], dropped, lbls)
        audit[b["name"]] = {"label_free_quality": round(b["quality"], 4),
                            "vlm_prefer": (None if b.get("vlm_prefer") is None else round(b["vlm_prefer"], 3)),
                            "true_dice": round(d, 4), "shipped": (b["model"] is out.delivered)}
    src_d = audit.get("source", {}).get("true_dice", 0.0)
    ship_d = next((a["true_dice"] for a in audit.values() if a["shipped"]), src_d)
    oracle_d = max((a["true_dice"] for a in audit.values()), default=src_d)
    # full-4ch reference Dice (the no-shift ceiling) — reported, not part of the decision
    full_ref = task._dice_3d_mri(base, full, lbls)
    rec = {"task": spec["name"], "shift": spec["shift"], "ckpt": spec["ckpt"],
           "nc": nc, "structures": struct, "fg_prior": spec["fg_prior"],
           "regime": ("K0" if klab == 0 else ("K%d_NO_FINETUNE" % klab if nf else "K%d_finetune" % klab)),
           "decision": out.decision, "report": out.report, "steps": out.steps,
           "budget_spent": out.budget_spent.as_dict(),
           "labeled_dice": task._labeled_dice, "trajectory": out.trace,
           "oracle_audit": audit,
           "full4ch_source_dice": round(full_ref, 4),
           "source_dice": round(src_d, 4), "shipped_dice": round(ship_d, 4),
           "delta_vs_source": round(ship_d - src_d, 4), "oracle_dice": round(oracle_d, 4),
           "safety_held": ship_d >= src_d - 1e-6, "shipped_is_oracle": abs(ship_d - oracle_d) < 1e-6,
           "agent_analysis": {k: out.evidence.get(k) for k in
                              ("domain_shift", "model_info", "failure_mode", "recommendation",
                               "retrieved_lessons")}}
    return rec


def main():
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "harness_decisions_3d_mri")
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    tasks = [c for c in TASKS_3D_MRI if not only or c["name"] in only]

    if os.environ.get("CORRECTNESS_ONLY", "0") != "0":
        os.makedirs(out_dir, exist_ok=True)
        res = {}
        for spec in tasks:
            print(f"\n{'='*70}\n### CORRECTNESS: {spec['name']}\n{'='*70}", flush=True)
            res[spec["name"]] = correctness_check(spec)
        with open(os.path.join(out_dir, "correctness.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nSaved -> {out_dir}/correctness.json", flush=True)
        return

    from medadaptagent.knowledge.llm_client import create_llm_client
    model = os.environ.get("MEDADAPT_LLM_MODEL", os.environ.get("MODEL"))
    print(f"== LLM brain + VLM model: {model} ==", flush=True)
    llm = create_llm_client(model=model)
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for spec in tasks:
        print(f"\n{'='*70}\n### 3D MRI TASK: {spec['name']}  ({spec['shift']})\n{'='*70}", flush=True)
        try:
            rec = run_one(spec, llm)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {spec['name']} FAILED: {e}", flush=True)
            summary.append({"task": spec["name"], "decision": "ERROR", "error": str(e)[:200]})
            continue
        # Regime-suffixed per-task JSON so the 3 budget cells (separate process invocations)
        # each retain their full trajectory instead of overwriting one another. Also keep the
        # plain <task>.json as the most-recent for convenience.
        for fn in (f"{spec['name']}_{rec['regime']}.json", spec["name"] + ".json"):
            with open(os.path.join(out_dir, fn), "w") as f:
                json.dump(rec, f, indent=2, default=str)
        traj = " -> ".join(e.get("tool", e.get("kind")) for e in rec["trajectory"]
                           if e["kind"] in ("tool", "terminal", "safety_veto", "forced_terminal"))
        print(f"  regime={rec['regime']} decision={rec['decision']}  "
              f"full4ch={rec['full4ch_source_dice']} source(FLAIRdrop)={rec['source_dice']} "
              f"shipped={rec['shipped_dice']} (delta {rec['delta_vs_source']:+}) "
              f"oracle={rec['oracle_dice']} safe={rec['safety_held']}", flush=True)
        print(f"  trajectory: {traj}", flush=True)
        summary.append({k: rec[k] for k in ("task", "shift", "regime", "decision",
                        "full4ch_source_dice", "source_dice", "shipped_dice", "delta_vs_source",
                        "oracle_dice", "safety_held", "shipped_is_oracle", "steps")})
    # append-merge so the 3 budget cells (separate process invocations) accumulate
    sum_path = os.path.join(out_dir, "suite_summary.json")
    prior = []
    if os.path.exists(sum_path):
        try:
            prior = json.load(open(sum_path))
        except Exception:
            prior = []
    with open(sum_path, "w") as f:
        json.dump(prior + summary, f, indent=2, default=str)

    print(f"\n\n{'='*70}\n 3D MRI SUITE — regime decisions\n{'='*70}", flush=True)
    print(f"{'task':12s} {'regime':16s} {'decision':11s} {'full':>6s} {'src':>6s} {'ship':>6s} {'delta':>7s} {'safe':>5s}")
    for s in summary:
        if s["decision"] == "ERROR":
            print(f"{s['task']:12s} ERROR"); continue
        print(f"{s['task']:12s} {s['regime']:16s} {s['decision']:11s} "
              f"{s['full4ch_source_dice']:6.3f} {s['source_dice']:6.3f} "
              f"{s['shipped_dice']:6.3f} {s['delta_vs_source']:+7.3f} {str(s['safety_held']):>5s}")
    print(f"\nFull per-regime decision logic saved -> {out_dir}/<task>.json")


if __name__ == "__main__":
    main()
