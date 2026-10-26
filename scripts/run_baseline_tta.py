#!/usr/bin/env python
"""
Run published TTA baselines (Tent, SAR, BN-reset-only) on a target dataset.
Bypasses the MedAdaptAgent reasoning chain entirely — applies the baseline
uniformly regardless of domain/modality/shift severity. Evaluates adapted
model vs source model on ground-truth target labels.

Purpose: establish the per-dataset ceiling of each published method so we
can compare vs. our pipeline's delivered Dice. If a baseline beats our
pipeline on a dataset, our reasoning layer is adding noise rather than
value — worth investigating.

Implementations follow the published formulations:
  - Tent (Wang et al., ICLR 2021): per-image entropy minimization on
    BN/IN/LN affine params; batch-stat forward pass.
  - SAR (Niu et al., ICLR 2023): Tent + sharpness-aware step (rho=0.05).
  - BN-reset only: reset running stats on target (no gradient updates).

Usage:
    python scripts/run_baseline_tta.py --dataset prostate --method tent --device cuda:0
    python scripts/run_baseline_tta.py --arch unet --method tent --device cuda:0     # all 8
    python scripts/run_baseline_tta.py --arch unet --method sar  --device cuda:1
"""

import os, sys, json, time, copy
import numpy as np
from PIL import Image as PILImage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_ablations import load_npz_dataset
from scripts._source_loader import load_source_model
from scripts.run_agent4_only import (
    DATASETS, STUNET_DATASETS, NNUNET_DATASETS, SEGFORMER_DATASETS)
from scripts import modern_tta_baselines as _mtta
from medadaptagent.graph.norm_utils import (
    iter_norm_modules, iter_norm_affine_params,
    reset_norm_running_stats, disable_running_stats, has_running_stats,
)


def _to_tensor(img, device, size=256):
    pil = PILImage.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)
                              if img.max() <= 1.0 else img.astype(np.uint8))
    pil = pil.resize((size, size), PILImage.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        t = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
    else:
        t = torch.from_numpy(arr).permute(2, 0, 1)
    return t.unsqueeze(0).to(device)


def _avg_dice(model, images, labels, num_classes, device, size=256):
    """Mean Dice over FG classes, matching compare_source_vs_adapted.py."""
    model.eval()
    per_class = {c: [] for c in range(1, num_classes)}
    with torch.no_grad():
        for img, lbl in zip(images, labels):
            if lbl is None:
                continue
            t = _to_tensor(img, device, size)
            pred = model(t).argmax(dim=1).cpu().numpy()[0]
            lbl_pil = PILImage.fromarray(lbl.astype(np.int32).astype(np.uint8))
            lbl_r = np.array(lbl_pil.resize((size, size), PILImage.NEAREST))
            for c in range(1, num_classes):
                if num_classes == 3 and c == 1:
                    p, g = (pred >= 1).astype(float), (lbl_r >= 1).astype(float)
                else:
                    p, g = (pred == c).astype(float), (lbl_r == c).astype(float)
                if g.sum() > 0:
                    d = float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))
                    per_class[c].append(d)
    means = [float(np.mean(v)) for v in per_class.values() if v]
    return float(np.mean(means)) if means else 0.0


# ============================================================
# Published TTA baselines (self-contained implementations)
# ============================================================

def _reliable_threshold(model, target_images, device, ent_margin,
                          num_classes, min_admit_frac=0.5):
    """Percentile-floored reliability threshold for SAR/EATA-style filters.

    The published absolute cutoff ent_margin*log(C) is calibrated for
    single-label CLASSIFICATION entropy. On dense SEGMENTATION the
    per-image MEAN pixel-entropy almost always exceeds it, so the filter
    rejects EVERY image and the method becomes a silent no-op — which is
    exactly why sar/eata/sfda collapsed to byte-identical Dice. Fix:
    admit AT LEAST min_admit_frac of the most-reliable images by taking
    the larger of the absolute cutoff and the min_admit_frac quantile.
    Returns (threshold, n_admitted, n_total)."""
    model.eval()
    ents = []
    with torch.no_grad():
        for img in target_images:
            logits = model(_to_tensor(img, device))
            probs = F.softmax(logits, dim=1)
            ents.append(-(probs * torch.log(probs + 1e-10)
                          ).sum(dim=1).mean().item())
    ents_arr = np.array(ents)
    abs_cut = ent_margin * float(np.log(max(num_classes, 2)))
    pct_cut = float(np.quantile(ents_arr, min_admit_frac)) if len(ents_arr) else abs_cut
    thr = max(abs_cut, pct_cut)
    n_admit = int((ents_arr <= thr + 1e-9).sum())
    return thr, n_admit, len(ents_arr)


def _frac_changed(src_model, adapted_model, images, device, n=25):
    """Fraction of argmax pixels that changed between source and adapted.
    A no-op adaptation (the sar/eata/sfda collapse) yields ~0.0 here, so
    this is logged per dataset to catch silent no-ops in the future."""
    src_model.eval(); adapted_model.eval()
    chg = tot = 0
    with torch.no_grad():
        for img in images[:n]:
            t = _to_tensor(img, device)
            a = src_model(t).argmax(dim=1)
            b = adapted_model(t).argmax(dim=1)
            chg += int((a != b).sum().item()); tot += a.numel()
    return chg / max(tot, 1)


def _git_sha():
    """Short git SHA of the repo, for stamping into result JSONs so any
    number is traceable to the exact code that produced it."""
    try:
        import subprocess
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = os.path.join(PROJECT_ROOT, ".gitconfig.shared")
        return subprocess.run(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, env=env).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def baseline_bn_reset(model, target_images, device):
    """BN-reset only — reset running stats, forward target batch to
    re-estimate, return. No gradient updates."""
    model.train()
    reset_norm_running_stats(model)
    # Forward pass to accumulate fresh stats
    with torch.no_grad():
        for img in target_images[:200]:
            t = _to_tensor(img, device)
            try:
                model(t)
            except Exception:
                pass
    model.eval()
    return model


def baseline_tent(model, target_images, device, lr=1e-3, steps_per_image=1,
                   epochs=1):
    """Tent (Wang et al. ICLR 2021): per-image entropy minimization on
    norm-affine params. Uses batch statistics (no running stats) during
    adaptation."""
    disable_running_stats(model)  # batch-stat mode for BN / IN-with-tracking
    # Collect norm-affine params BEFORE freezing. iter_norm_affine_params
    # skips any param whose requires_grad is already False, so freezing
    # first yields an EMPTY list -> the optimizer gets nothing and the
    # method silently no-ops (this is the real cause of the tent/sar/eata/
    # sfda byte-identical collapse: all four fell through to BN-stat-only).
    params = list(iter_norm_affine_params(model))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    if not params:
        # Pure LN/GN without affine? Nothing to update.
        return model
    opt = torch.optim.Adam(params, lr=lr)
    model.train()  # train mode so batch stats are used
    for _ in range(epochs):
        for img in target_images:
            t = _to_tensor(img, device)
            for _s in range(steps_per_image):
                opt.zero_grad()
                logits = model(t)
                probs = F.softmax(logits, dim=1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                ent.backward()
                opt.step()
    model.eval()
    return model


def baseline_sar(model, target_images, device, lr=5e-3, rho=0.05,
                  ent_margin=0.4, steps_per_image=1, epochs=1):
    """SAR (Niu et al. ICLR 2023): sharpness-aware Tent. Skip high-entropy
    samples via `ent_margin × log(C)` where C is number of classes."""
    disable_running_stats(model)
    # Collect affine params BEFORE freezing (see baseline_tent note): the
    # freeze-then-collect ordering returned an empty param list and made
    # this method a silent no-op identical to bare BN recalibration.
    params = list(iter_norm_affine_params(model))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    if not params:
        return model
    # Estimate num_classes from model output once
    model.train()
    with torch.no_grad():
        sample_logits = model(_to_tensor(target_images[0], device))
        num_classes = sample_logits.shape[1]
    threshold, _na, _nt = _reliable_threshold(
        model, target_images, device, ent_margin, num_classes)
    print(f"  [sar] reliability filter admits {_na}/{_nt} samples "
          f"(thr={threshold:.3f})")

    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    for _ in range(epochs):
        for img in target_images:
            t = _to_tensor(img, device)
            for _s in range(steps_per_image):
                # First forward for entropy filter
                opt.zero_grad()
                logits1 = model(t)
                probs1 = F.softmax(logits1, dim=1)
                ent1 = -(probs1 * torch.log(probs1 + 1e-10)).sum(dim=1).mean()
                if ent1.item() > threshold:
                    continue  # skip high-entropy sample (SAR's reliability filter)

                # Sharpness-aware step
                ent1.backward()
                with torch.no_grad():
                    grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params
                                                if p.grad is not None) + 1e-12)
                    scale = rho / (grad_norm + 1e-12)
                    # Perturb
                    for p in params:
                        if p.grad is not None:
                            e_w = p.grad * scale
                            p.add_(e_w)
                            p._sar_perturb = e_w
                # Second forward/backward at perturbed weights
                opt.zero_grad()
                logits2 = model(t)
                probs2 = F.softmax(logits2, dim=1)
                ent2 = -(probs2 * torch.log(probs2 + 1e-10)).sum(dim=1).mean()
                ent2.backward()
                # Undo perturbation before the step
                with torch.no_grad():
                    for p in params:
                        if hasattr(p, "_sar_perturb"):
                            p.sub_(p._sar_perturb)
                            del p._sar_perturb
                opt.step()
    model.eval()
    return model


def baseline_eata(model, target_images, device, lr=1e-3, ent_margin=0.4,
                    anchor_beta=2000.0, cosine_thresh=0.05, epochs=1):
    """EATA (Niu et al., ICML 2022): Tent + reliable-sample filter
    (entropy < ent_margin * log(C)) + redundant-sample filter (cosine
    similarity to running mean-prob > cosine_thresh → skip) +
    anchor-weight regularization (beta * ||theta - theta_src||^2 on the
    adapted params). The anchor term prevents catastrophic forgetting on
    the source task.
    """
    disable_running_stats(model)
    # Collect affine params BEFORE freezing (see baseline_tent note): the
    # freeze-then-collect ordering returned an empty param list and made
    # this method a silent no-op identical to bare BN recalibration.
    params = list(iter_norm_affine_params(model))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    if not params:
        return model
    anchor = [p.detach().clone() for p in params]
    model.train()
    with torch.no_grad():
        logits0 = model(_to_tensor(target_images[0], device))
        num_classes = logits0.shape[1]
    threshold, _na, _nt = _reliable_threshold(
        model, target_images, device, ent_margin, num_classes)
    print(f"  [eata] reliability filter admits {_na}/{_nt} samples "
          f"(thr={threshold:.3f})")

    opt = torch.optim.Adam(params, lr=lr)
    running_mean_prob = None
    for _ in range(epochs):
        for img in target_images:
            t = _to_tensor(img, device)
            opt.zero_grad()
            logits = model(t)
            probs = F.softmax(logits, dim=1)
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            # Reliable-sample filter
            if ent.item() > threshold:
                continue
            # Redundant-sample filter (batch-mean prob cosine vs running)
            mean_prob = probs.mean(dim=[2, 3]).flatten().detach()
            if running_mean_prob is not None:
                cos = F.cosine_similarity(
                    running_mean_prob, mean_prob, dim=0).item()
                if cos > 1.0 - cosine_thresh:  # too similar to prior
                    continue
                running_mean_prob = 0.9 * running_mean_prob + 0.1 * mean_prob
            else:
                running_mean_prob = mean_prob
            # Anchor regularization
            anchor_loss = sum(
                ((p - a) ** 2).sum() for p, a in zip(params, anchor))
            loss = ent + (1.0 / anchor_beta) * anchor_loss
            loss.backward()
            opt.step()
    model.eval()
    return model


def baseline_sfda(model, target_images, device, lr=5e-4, epochs=1,
                    ent_weight=1.0, div_weight=1.0, im_margin=0.4):
    """SFDA (Liang et al., SHOT ICML 2020; backbone for SFDA++ family):
    information-maximization (IM) loss = entropy − diversity. Per-image
    entropy minimized, batch-marginal distribution diversity maximized.
    No teacher/student, no pseudo-labels — the cleanest SFDA baseline.
    The full SFDA++ (Yang et al. MICCAI 2023) adds dual-path contrastive
    learning, which requires source-feature bank + curriculum — out of
    scope for this label-free sanity baseline. Docs/README should note
    this as "SFDA (IM loss)" not "SFDA++".
    """
    disable_running_stats(model)
    # Collect affine params BEFORE freezing (see baseline_tent note): the
    # freeze-then-collect ordering returned an empty param list and made
    # this method a silent no-op identical to bare BN recalibration.
    params = list(iter_norm_affine_params(model))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    if not params:
        return model
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(epochs):
        for img in target_images:
            t = _to_tensor(img, device)
            opt.zero_grad()
            logits = model(t)
            probs = F.softmax(logits, dim=1)
            # Per-pixel entropy (minimize): confident predictions
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            # Batch-marginal diversity (maximize): avoid class collapse
            mean_prob = probs.mean(dim=[0, 2, 3])
            div = -(mean_prob * torch.log(mean_prob + 1e-10)).sum()
            loss = ent_weight * ent - div_weight * div
            loss.backward()
            opt.step()
    model.eval()
    return model


# Give the modern-TTA module the host tensorizer so its eval grid matches
# _avg_dice exactly, keeping all numbers comparable to baselines_unet.csv.
_mtta.set_tensorizer(_to_tensor)

METHODS = {
    "bn_reset": baseline_bn_reset,
    "tent": baseline_tent,
    "sar": baseline_sar,
    "eata": baseline_eata,
    "sfda": baseline_sfda,
    "memo": _mtta.baseline_memo,
    "cotta": _mtta.baseline_cotta,
    "t3a": _mtta.baseline_t3a,
    "vptta": _mtta.baseline_vptta,
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--method", required=True, choices=list(METHODS.keys()))
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--arch", default="unet",
                     choices=["unet", "stunet", "nnunet", "segformer", "all"])
    ap.add_argument("--out", default=None,
                     help="Optional JSON output path. Default: "
                          "experiments/baselines/{method}_{arch}.json")
    ap.add_argument("--epochs", type=int, default=5,
                     help="Adaptation epochs over the target set. Default "
                          "5 (was 1): a single norm-affine epoch was too "
                          "weak to change argmax masks, collapsing "
                          "sar/eata/sfda to byte-identical Dice.")
    args = ap.parse_args()

    # Build active registry
    if args.arch == "unet":
        active = dict(DATASETS)
    elif args.arch == "stunet":
        active = dict(STUNET_DATASETS)
    elif args.arch == "nnunet":
        active = dict(NNUNET_DATASETS)
    elif args.arch == "segformer":
        active = dict(SEGFORMER_DATASETS)
    else:
        active = {**DATASETS, **STUNET_DATASETS, **NNUNET_DATASETS, **SEGFORMER_DATASETS}

    names = list(active.keys()) if args.dataset == "all" else [args.dataset]

    out_path = args.out or os.path.join(
        PROJECT_ROOT, "experiments", "baselines",
        f"{args.method}_{args.arch}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = {}
    for name in names:
        ds = active[name]
        arch = ds.get("arch", "unet")
        print(f"\n=== {name} (arch={arch}, method={args.method}) ===")
        try:
            torch.manual_seed(42); np.random.seed(42)

            # Load source + target
            src = load_source_model(
                os.path.join(PROJECT_ROOT, ds["ckpt"]), ds["nc"], args.device,
                arch=arch, project_root=None)
            mt = ds.get("max_target", 50)
            tgt_images, tgt_labels = load_npz_dataset(
                os.path.join(PROJECT_ROOT, ds["tgt"]), mt)
            # Diagnostic: prove adaptation has something to update + data to
            # update on. n_affine==0 or n_tgt==0 => silent no-op (the old
            # sar/eata/sfda collapse manifested as near-zero elapsed_s).
            _nparam = len(list(iter_norm_affine_params(src)))
            print(f"  [diag] n_affine_params={_nparam}  n_tgt_images="
                  f"{len(tgt_images)}  epochs={args.epochs}")

            # Source Dice on target (before adaptation)
            src_dice = _avg_dice(src, tgt_images, tgt_labels, ds["nc"], args.device)

            # Adapt with chosen baseline. Episodic methods (MEMO, VPTTA)
            # register a per-image eval predictor so their per-image
            # adaptation is reflected in the reported Dice; other methods
            # leave it None and are evaluated on the returned model.
            _mtta.EVAL_PREDICTORS.pop(args.method, None)
            adapted = copy.deepcopy(src)
            t0 = time.time()
            import inspect as _inspect
            _sig = _inspect.signature(METHODS[args.method])
            _extra = {"epochs": args.epochs} if "epochs" in _sig.parameters else {}
            METHODS[args.method](adapted, tgt_images, args.device, **_extra)
            elapsed = time.time() - t0
            # No-op diagnostic: fraction of argmax pixels changed vs source.
            # ~0.0 = the adaptation did nothing (the sar/eata/sfda collapse).
            frac_changed = _frac_changed(src, adapted, tgt_images, args.device)

            predictor = _mtta.EVAL_PREDICTORS.get(args.method)
            if predictor is not None:
                ada_dice = _mtta.avg_dice_via_predictor(
                    predictor, tgt_images, tgt_labels, ds["nc"])
            else:
                ada_dice = _avg_dice(adapted, tgt_images, tgt_labels,
                                     ds["nc"], args.device)
            delta = ada_dice - src_dice
            print(f"  src={src_dice:.4f}  {args.method}={ada_dice:.4f}  "
                  f"delta={delta:+.4f}  frac_changed={frac_changed:.4f}  "
                  f"({elapsed:.0f}s)")

            results[name] = {
                "arch": arch,
                "method": args.method,
                "epochs": args.epochs,
                "src_dice": src_dice,
                "adapted_dice": ada_dice,
                "delta": delta,
                "frac_changed": frac_changed,
                "elapsed_s": elapsed,
                "git_sha": _git_sha(),
            }
            del src, adapted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'dataset':24s}  {'src':>7s}  {args.method:>7s}  {'delta':>8s}")
    print("-" * 70)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:24s}  error: {r['error'][:40]}")
        else:
            print(f"{name:24s}  {r['src_dice']:>7.4f}  "
                  f"{r['adapted_dice']:>7.4f}  {r['delta']:>+8.4f}")
    print("=" * 70)
    print(f"Saved: {out_path}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
