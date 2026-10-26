#!/usr/bin/env python
"""End-to-end external-model deployment experiment for the paper.

Three independently published MONAI bundles and an additional CT-to-MRI case.
Uses the shape-preserving inference fix for short-axis volumes. Local labels are exposed only through a fixed calibration split;
all reported deployment Dice values use a disjoint hidden test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(os.environ.get("MEDADAPT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = "cuda:0"

from harness.core import Budget, Tool, build_harness
from scripts.public_repair_support import install_api_meter, export_and_verify_eager
from harness.llm_policy import make_llm_policy
from harness.seg3d_adapter import SegTask3D, _resample_to_spacing
from scripts._source_loader import load_source_model


def _check_image_label_grid(image, label, name):
    """Allow sub-voxel header roundoff without resampling images or labels."""
    if image.shape != label.shape:
        raise ValueError(f"Image/label shape mismatch: {name}")
    corners = np.array(np.meshgrid(*[(0, n - 1) for n in image.shape],
                                   indexing="ij")).reshape(3, -1)
    corners = np.vstack((corners, np.ones((1, corners.shape[1]))))
    displacement = np.linalg.solve(image.affine, label.affine) @ corners - corners
    max_voxels = float(np.linalg.norm(displacement[:3], axis=0).max())
    if not np.isfinite(max_voxels) or max_voxels > 0.1:
        raise ValueError(f"Image/label grid mismatch: {name} ({max_voxels:.6f} voxels)")
    if max_voxels > 1e-4:
        print(f"Header tolerance: {name}, max grid difference {max_voxels:.6f} voxels", flush=True)
    return max_voxels


def _load_3d_volumes(img_dir, lbl_dir, max_n, names=None, label_id=None):
    """Load matched raw NIfTI cases in RAS; optionally extract one anatomy."""
    import nibabel as nib
    from medadaptagent.calibration import Volume
    selected = names if names is not None else names_in(Path(img_dir), max_n)
    imgs, lbls = [], []
    for name in selected:
        ip, lp = Path(img_dir) / name, Path(lbl_dir) / name
        if not ip.is_file() or not lp.is_file():
            raise FileNotFoundError(f"Missing paired image/label: {name}")
        image = nib.as_closest_canonical(nib.load(ip))
        label = nib.as_closest_canonical(nib.load(lp))
        _check_image_label_grid(image, label, name)
        mask = label.get_fdata().astype(np.int64)
        if label_id is not None:
            mask = (mask == label_id).astype(np.int64)
        imgs.append(Volume.from_array(image.get_fdata().astype(np.float32),
                                      image.affine.astype(np.float32)))
        lbls.append(Volume.from_array(mask, label.affine.astype(np.float32)))
    return imgs, lbls


MODELS = {
    "spleen_unet": {
        "bundle": "spleen_ct_segmentation",
        "version": "0.6.1",
        "architecture": "3D U-Net",
        "source_domain": "MSD Task09 Spleen contrast-enhanced CT",
        "target_domain": "TotalSegmentator non-contrast CT",
        "ckpt": "data/monai_bundles/spleen_ct_segmentation/models/model.pt",
        "arch": "monai_unet3d",
        "target": "data/3d/totalseg_spleen_nc/imagesTs",
        "target_labels": "data/3d/totalseg_spleen_nc/labelsTs",
        "source_ref": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "source_ref_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "a_min": -57.0,
        "a_max": 164.0,
        "pixdim": [1.5, 1.5, 2.0],
        "official_url": "https://huggingface.co/MONAI/spleen_ct_segmentation",
    },
    "swinunetr_btcv": {
        "bundle": "swin_unetr_btcv_segmentation",
        "version": "0.5.8",
        "architecture": "Swin UNETR",
        "source_domain": "BTCV contrast-enhanced abdominal CT",
        "target_domain": "TotalSegmentator non-contrast CT",
        "ckpt": "data/monai_bundles/swin_unetr_btcv_segmentation/models/model.pt",
        "arch": "monai_bundle",
        "target": "data/3d/totalseg_spleen_nc/imagesTs",
        "target_labels": "data/3d/totalseg_spleen_nc/labelsTs",
        "source_ref": "data/3d/synapse_spleen/imagesTs",
        "source_ref_labels": "data/3d/synapse_spleen/labelsTs",
        "a_min": -175.0,
        "a_max": 250.0,
        "pixdim": [1.5, 1.5, 2.0],
        "official_url": "https://huggingface.co/MONAI/swin_unetr_btcv_segmentation",
    },
    "wholebody_segresnet": {
        "bundle": "wholeBody_ct_segmentation",
        "version": "0.2.7",
        "architecture": "SegResNet (105-class; spleen channel extracted)",
        "source_domain": "TotalSegmentator whole-body CT",
        "target_domain": "MSD Task09 Spleen CT",
        "ckpt": "data/monai_bundles/wholeBody_ct_segmentation/models/model.pt",
        "arch": "monai_bundle",
        "target": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "target_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "source_ref": "data/3d/totalseg_spleen_ce/imagesTs",
        "source_ref_labels": "data/3d/totalseg_spleen_ce/labelsTs",
        "preprocessing": "wholebody",
        "pixdim": [1.5, 1.5, 1.5],
        "official_url": "https://huggingface.co/MONAI/wholeBody_ct_segmentation",
    },
}


MODELS["spleen_unet_amos_mri"] = {
    **MODELS["spleen_unet"],
    "target_domain": "AMOS multicenter abdominal MRI (CT-to-MRI modality shift)",
    "target": "data/amos/raw/amos22/imagesTr",
    "target_labels": "data/amos/raw/amos22/labelsTr",
    "target_modality": "MRI",
    "target_label_id": 1,
}

MODELS["swinunetr_amos_mri"] = {
    **MODELS["swinunetr_btcv"],
    "target_domain": "AMOS multicenter abdominal MRI (CT-to-MRI modality shift)",
    "target": "data/amos/raw/amos22/imagesTr",
    "target_labels": "data/amos/raw/amos22/labelsTr",
    "target_modality": "MRI",
    "target_label_id": 1,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def names_in(path: Path, max_n: int | None = None, modality=None) -> list[str]:
    out = sorted(p.name for p in path.iterdir()
                 if p.name.endswith(".nii.gz") and not p.name.startswith("._"))
    if modality == "MRI":
        out = [name for name in out if re.fullmatch(r"amos_\d+\.nii\.gz", name)
               and 501 <= int(name.split("_")[1].split(".")[0]) <= 600]
    return out[:max_n] if max_n else out


def fixed_split(n: int, k: int = 5, n_test: int = 10):
    if k not in (5, 20) or n_test != 10:
        raise ValueError("This comparison supports K=5/20 with the same ten test cases")
    if n < k + n_test:
        raise RuntimeError(f"Need at least {k+n_test} target cases; found {n}")
    chosen = np.linspace(0, n - 1, 5 + n_test, dtype=int).tolist()
    if len(set(chosen)) != len(chosen):
        raise RuntimeError("Deterministic split produced duplicate indices")
    calibration = chosen[::3][:5]
    hidden_test = [i for i in chosen if i not in set(calibration)][:n_test]
    # Preserve all K=5 roles: prefix ordering keeps the same train/selection cases.
    calibration += [i for i in range(n)
                    if i not in set(calibration + hidden_test)][:k - 5]
    assert len(calibration) == k and len(hidden_test) == n_test
    assert not set(calibration) & set(hidden_test)
    return calibration, hidden_test


def wholebody_to_tensor(pixdim):
    """Reproduce Orientation->Spacing->Normalize(nonzero)->Scale[-1,1]."""
    def fn(img, size, device):
        import torch
        arr = np.asarray(img).astype(np.float32)
        affine = getattr(img, "affine", None)
        if affine is not None:
            arr = _resample_to_spacing(arr, affine, pixdim, order=1)
        nz = arr != 0
        if nz.any():
            mean = float(arr[nz].mean())
            std = float(arr[nz].std())
            arr = arr.copy()
            arr[nz] = (arr[nz] - mean) / max(std, 1e-8)
        lo, hi = float(arr.min()), float(arr.max())
        arr = 2.0 * (arr - lo) / max(hi - lo, 1e-8) - 1.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    fn._window = "NormalizeIntensity(nonzero)+ScaleIntensity[-1,1]"
    fn._pixdim = pixdim
    return fn


class FixedSplitSegTask3D(SegTask3D):
    """Expose only fixed calibration labels and rank every candidate identically."""
    def __init__(self, *args, calibration_indices, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibration_indices = list(calibration_indices)
        self.calibration_train_indices = []
        self.calibration_holdout_indices = []

    def toolbox(self):
        tools = super().toolbox()
        for tool in tools:
            if tool.name == "request_labels":
                tool.description = tool.description.replace(
                    "K (3-12)", f"K (up to {len(self.calibration_indices)})")
        return tools

    def _ensure_labels(self, K=5):
        if self._labels_revealed:
            return self._train
        K = min(int(K), len(self.calibration_indices))
        idx = self.calibration_indices[:K]
        hold_pos = set(range(0, len(idx), 3)) if len(idx) >= 2 else set()
        self.calibration_holdout_indices = [idx[p] for p in range(len(idx)) if p in hold_pos]
        self.calibration_train_indices = [idx[p] for p in range(len(idx)) if p not in hold_pos]
        if not self.calibration_train_indices:
            self.calibration_train_indices = list(self.calibration_holdout_indices)
        self._train = ([self.timgs[i] for i in self.calibration_train_indices],
                       [self.tlbls[i] for i in self.calibration_train_indices])
        self._hold = ([self.timgs[i] for i in self.calibration_holdout_indices],
                      [self.tlbls[i] for i in self.calibration_holdout_indices])
        self.labeled_k = len(idx)
        self._labels_revealed = True
        return self._train

    def _rank_on_holdout(self):
        h_imgs, h_lbls = self._hold
        if not h_imgs:
            raise RuntimeError("Calibration holdout is empty")
        self._labeled_dice = {
            b["name"]: self._dice_of_3d(b["model"], h_imgs, h_lbls)
            for b in self.built
        }
        return self._labeled_dice


def evaluate_per_case(task, model, indices):
    values = []
    for i in indices:
        d = task._dice_of_3d(model, [task.timgs[i]], [task.tlbls[i]])
        values.append({"index": i, "dice": d})
    return values


def mean_dice(rows):
    return round(float(np.mean([r["dice"] for r in rows])), 4) if rows else None


def enable_vlm(task, llm, timgs):
    """Use the same model-specific preprocessing for VLM masks and metric inference."""
    from medadaptagent.graph.agents.execution_engineer import _build_vlm_task_ctx
    ctx = _build_vlm_task_ctx(
        {"structures": {"background": 0, "spleen": 1}, "foreground_priors": {}}, {})
    ctx["num_classes_hint"] = 2
    anchors = [timgs[i] for i in np.linspace(0, len(timgs) - 1,
                                             min(6, len(timgs)), dtype=int)]
    task.set_vlm(llm, ctx, task._t3d(), anchors)


def require_target_labels(policy, k_labels):
    """Obtain the fixed labels for delivery/tuning while preserving error escalation."""
    def wrapped(view):
        action = policy(view)
        name = (action or {}).get("tool", "")
        requires_labels = name in ("ship", "keep_source") \
            or name.startswith("finetune_") or name.startswith("probe_")
        if requires_labels \
                and int(view.get("k_labels_used", 0)) < k_labels:
            return {"tool": "request_labels", "args": {"K": k_labels}}
        if (action or {}).get("tool") == "request_labels":
            return {"tool": "request_labels", "args": {"K": k_labels}}
        return action

    return wrapped


def run(model_id: str, out_root: Path, max_target: int, llm_model: str, k_labels: int = 5,
        planner_backend: str | None = None, planner_model: str | None = None):
    started = time.time()
    endpoint_config = PROJECT_ROOT / "runtime_config.json"
    if endpoint_config.exists():
        config = json.loads(endpoint_config.read_text())
        os.environ.setdefault("AZURE_OPENAI_ENDPOINT", config["azure_endpoint"])
        if config.get("azure_api_version"):
            os.environ.setdefault("AZURE_API_VERSION", config["azure_api_version"])
    install_api_meter(Path(os.environ["MEDADAPT_API_LEDGER"]), cap_usd=10.0)
    import torch
    np.random.seed(20260902)
    torch.manual_seed(20260902)
    torch.cuda.manual_seed_all(20260902)
    torch.set_num_threads(4)
    cfg = MODELS[model_id]
    out_dir = out_root / (model_id if k_labels == 5 else f"{model_id}_K{k_labels}")
    if (out_dir / "decision_chain.json").exists():
        raise FileExistsError(f"Completed result exists; choose a new output root: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {model_id}: {cfg['source_domain']} -> {cfg['target_domain']}", flush=True)

    ckpt = PROJECT_ROOT / cfg["ckpt"]
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    target_names = names_in(PROJECT_ROOT / cfg["target"], max_target, cfg.get("target_modality"))
    timgs, tlbls = _load_3d_volumes(str(PROJECT_ROOT / cfg["target"]),
                                    str(PROJECT_ROOT / cfg["target_labels"]), max_target,
                                    names=target_names, label_id=cfg.get("target_label_id"))
    if len(timgs) != len(target_names):
        raise RuntimeError("Target name/volume count mismatch")
    calibration, hidden_test = fixed_split(len(timgs), k=k_labels, n_test=10)
    simgs, _ = _load_3d_volumes(str(PROJECT_ROOT / cfg["source_ref"]),
                                 str(PROJECT_ROOT / cfg["source_ref_labels"]), 4)

    base = load_source_model(cfg["ckpt"], 2, DEVICE, arch=cfg["arch"],
                             project_root=str(PROJECT_ROOT))
    desc = (f"Public {cfg['architecture']} model released as MONAI/{cfg['bundle']}; "
            f"source domain: {cfg['source_domain']}; deployment domain: "
            f"{cfg['target_domain']}; target task: 3D spleen segmentation.")
    task = FixedSplitSegTask3D(
        base, simgs, timgs, 2, DEVICE, fg_prior=0.006, tlbls=tlbls,
        task_desc=desc, a_min=cfg.get("a_min", -57.0),
        a_max=cfg.get("a_max", 164.0), pixdim=cfg.get("pixdim"),
        calibration_indices=calibration,
    )
    if cfg.get("preprocessing") == "wholebody":
        task._to_tensor = wholebody_to_tensor(cfg["pixdim"])

    from medadaptagent.knowledge.llm_client import create_llm_client
    llm = create_llm_client(backend="azure", model=llm_model)
    enable_vlm(task, llm, timgs)
    planner = (create_llm_client(backend=planner_backend, model=planner_model)
               if planner_backend else llm)

    common = [
        Tool("ship", "deliver the chosen model", lambda st, **kw: {}, terminal=True),
        Tool("keep_source", "keep the unmodified public model", lambda st, **kw: {}, terminal=True),
        Tool("escalate", "refuse automated deployment and hand back for expert review",
             lambda st, **kw: {}, terminal=True),
    ]
    contract = (
        "Deploy this publicly released spleen segmentation model on the local dataset. "
        f"{k_labels} labeled local volumes are provided and one GPU is available. Automatically analyze the "
        "model and site shift, restrict the action space to compatible adjustments, "
        f"and request and use all {k_labels} labels before the terminal decision. Return a deployable model only when calibration evidence supports it; "
        "otherwise refuse automated deployment safely. The objective is improvement over the source; "
        "no absolute performance floor is requested. Literature performance estimates are context, "
        "not additional release conditions. Retain the source when it is the best supported option."
    )
    os.environ["KLAB"] = str(k_labels)
    os.environ["NO_FINETUNE"] = "0"
    budget = Budget(compute_minutes=75, n_trials=4, llm_usd=5.0, k_labels=k_labels)

    def observed_chat(prompt):
        # Same request as llm.chat; save response diagnostics without credentials.
        response = planner.cached_complete(messages=[{"role": "user", "content": prompt}],
                                       max_tokens=2000, temperature=0, seed=42, system=None)
        choice = response.choices[0]
        record = {"prompt": prompt, "reply": choice.message.content or "",
                  "finish_reason": choice.finish_reason,
                  "usage": response.usage.model_dump() if response.usage else None}
        with (out_dir / "policy_responses.jsonl").open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        return record["reply"]

    brain = make_llm_policy(observed_chat)

    contract_policy = require_target_labels(brain, k_labels)

    harness = build_harness(contract_policy, task, common,
                            budget, max_steps=24, no_progress_cap=10, finalize_on_no_progress=True)
    n_selection = len(range(0, k_labels, 3))
    print(f"Starting harness: K={k_labels}, {k_labels - n_selection} training and "
          f"{n_selection} selection volumes; test labels hidden", flush=True)
    outcome = harness.run(goal=contract, source_model=base)
    print(f"Harness finished: decision={outcome.decision}; evaluating hidden test", flush=True)

    # Persist the terminal decision before any packaging or hidden-test operation.
    selected_name = next((b["name"] for b in task.built if b["model"] is outcome.delivered), None)
    (out_dir / "terminal_decision.json").write_text(json.dumps({
        "decision": outcome.decision, "selected_action": selected_name,
        "report": outcome.report, "decision_chain": outcome.trace,
        "calibration_holdout_dice": task._labeled_dice,
        "target_case_ids": target_names, "calibration_indices": calibration,
        "train_indices": task.calibration_train_indices,
        "selection_indices": task.calibration_holdout_indices,
        "hidden_test_indices": hidden_test,
        "source_checkpoint_sha256": sha256(ckpt),
        "budget_spent": outcome.budget_spent.as_dict(),
    }, indent=2, default=str))
    if outcome.delivered is not None:
        torch.save({k:v.detach().cpu() for k,v in outcome.delivered.state_dict().items()},
                   out_dir / "selected_state_dict.pt")
    source_rows = evaluate_per_case(task, base, hidden_test)
    candidate_audit = {}
    delivered_rows = None
    delivered_dice = None
    deployment_artifact = None
    if outcome.decision != "escalate" and outcome.delivered is not None:
        deployment_artifact, delivered_rows = export_and_verify_eager(
            task, outcome.delivered, hidden_test, out_dir,
            metadata={"model_id": model_id, "selected_action": selected_name,
                      "source_checkpoint_sha256": sha256(ckpt), "k_labels": k_labels})
        delivered_dice = mean_dice(delivered_rows)

    split = {
        "selection_rule": ("K=5 uses five interleaved calibration cases from 15 evenly spaced cases; "
                           "the other ten are fixed hidden test. K=20 appends non-test cases in index "
                           "order. Every third calibration position is selection; the rest train."),
        "k_labels": k_labels,
        "calibration_indices": calibration,
        "calibration_case_ids": [target_names[i] for i in calibration],
        "calibration_train_indices": task.calibration_train_indices,
        "calibration_holdout_indices": task.calibration_holdout_indices,
        "hidden_test_indices": hidden_test,
        "hidden_test_case_ids": [target_names[i] for i in hidden_test],
        "hidden_test_labels_visible_to_controller": False,
    }
    provenance = {
        "model_id": model_id,
        "bundle": cfg["bundle"],
        "bundle_version": cfg["version"],
        "architecture": cfg["architecture"],
        "official_url": cfg["official_url"],
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "source_domain": cfg["source_domain"],
        "target_domain": cfg["target_domain"],
    }
    selected_name = next((b["name"] for b in task.built
                          if b["model"] is outcome.delivered), None)
    result = {
        "evaluation_version": "20261025_public_model_deployment",
        "planner": {"backend": planner_backend or "azure", "model": planner_model or llm_model},
        "analysis_and_critique_model": llm_model,
        "random_seed": 20260902,
        "target_label_mapping": cfg.get("target_label_id", "native_binary"),
        "contract": contract,
        "provenance": provenance,
        "split": split,
        "decision": outcome.decision,
        "selected_action": selected_name,
        "source_hidden_test_dice": mean_dice(source_rows),
        "delivered_hidden_test_dice": delivered_dice,
        "delta_vs_direct_deployment": (None if delivered_dice is None else
                                       round(delivered_dice - mean_dice(source_rows), 4)),
        "source_hidden_test_per_case": source_rows,
        "delivered_hidden_test_per_case": delivered_rows,
        "deployment_artifact": deployment_artifact,
        "calibration_holdout_dice": task._labeled_dice,
        "candidate_hidden_test_audit": candidate_audit,
        "budget_spent": outcome.budget_spent.as_dict(),
        "steps": outcome.steps,
        "runtime_seconds": round(time.time() - started, 1),
        "report": outcome.report,
        "decision_chain": outcome.trace,
        "agent_analysis": {k: outcome.evidence.get(k) for k in
                           ("domain_shift", "model_info", "failure_mode",
                            "recommendation", "retrieved_lessons")},
    }
    (out_dir / "contract.txt").write_text(contract + "\n")
    (out_dir / "split_manifest.json").write_text(json.dumps(split, indent=2))
    (out_dir / "decision_chain.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "candidate_test_audit.json").write_text(
        json.dumps(candidate_audit, indent=2, default=str))
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out_dir / "deployment_report.md").write_text(
        f"# External-model deployment: {model_id}\n\n"
        f"- Decision: **{outcome.decision}**\n"
        f"- Selected action: **{selected_name or 'none'}**\n"
        f"- Direct-deployment hidden-test Dice: **{result['source_hidden_test_dice']:.4f}**\n"
        + (f"- Delivered hidden-test Dice: **{delivered_dice:.4f}**\n"
           f"- Change: **{result['delta_vs_direct_deployment']:+.4f}**\n"
           if delivered_dice is not None else
           "- Delivered model: **none; inspect terminal report for reason**\n")
        + f"- Labels used: **{task.labeled_k}** ({len(task.calibration_train_indices)} train, "
        f"{len(task.calibration_holdout_indices)} internal holdout)\n"
        f"- Hidden test: **10 cases**\n"
        f"- Runtime: **{result['runtime_seconds']:.1f} s**\n\n"
        f"{outcome.report}\n")
    print(json.dumps({k: result[k] for k in
                      ("decision", "selected_action", "source_hidden_test_dice",
                       "delivered_hidden_test_dice", "delta_vs_direct_deployment",
                       "runtime_seconds")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", choices=sorted(MODELS))
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "experiments" /
                                                   "public_model_repair_20261025"))
    parser.add_argument("--max-target", type=int, default=30)
    parser.add_argument("--k-labels", type=int, choices=(5, 20), default=5)
    parser.add_argument("--llm-model", default=os.environ.get("MODEL", "gpt-5.2-2025-12-11"))
    parser.add_argument("--planner-backend", choices=("azure", "local", "openai"))
    parser.add_argument("--planner-model")
    args = parser.parse_args()
    if bool(args.planner_backend) != bool(args.planner_model):
        parser.error("Set both --planner-backend and --planner-model")
    run(args.model_id, Path(args.out_root), args.max_target, args.llm_model, args.k_labels,
        args.planner_backend, args.planner_model)


if __name__ == "__main__":
    main()
