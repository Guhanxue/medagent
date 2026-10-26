#!/usr/bin/env python
"""Step 4 bet test: does a DIVERSE toolbox widen the agentic-selection gap?

Adds two genuinely-orthogonal (non-entropy) candidates to {source,
bn_reset, tent}:
  proto_align  -- gradient adapt with a FEATURE-MOMENT-ALIGNMENT loss
                  (CORAL to frozen SOURCE penultimate-feature moments),
                  NOT entropy -> should not collapse onto the tent argmax.
  spectral_match -- input-space amplitude-spectrum match toward the source
                  mean amplitude (frequency family), training-free.
For each dataset evaluate all 5 candidates' full-target Dice + the
combined VLM+FM critic, then test the central redesign bet:
  (a) does the oracle-best ROTATE across more families?
  (b) does VLM+FM-selection beat best-fixed-policy by MORE than the
      3-tool baseline (+0.033)?
"""

from pathlib import Path
import os, sys, copy
import numpy as np
PROJECT_ROOT = os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, PROJECT_ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.run_ablations import load_npz_dataset
from scripts._source_loader import load_source_model
from scripts.run_baseline_tta import (_to_tensor, _avg_dice,
                                         baseline_bn_reset, baseline_tent)
from scripts.run_agent4_only import DATASETS
from medadaptagent.graph.norm_utils import (iter_norm_affine_params,
                                              disable_running_stats)
from medadaptagent.graph.agents.execution_engineer import (
    _vlm_iteration_vote, _build_vlm_task_ctx)
from medadaptagent.tasks import get_task
from medadaptagent.knowledge.llm_client import create_llm_client
from medadaptagent.critics import fm_critic

DSETS = ["chaos", "prostate", "mnms", "fundus_roi", "riga_plus",
          "brats", "mmwhs", "mnms_field"]
DEVICE = "cuda:0"


# ---------- penultimate-feature hook ----------
def _final_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


# ---------- proto_align (CORAL to source feature moments) ----------
def _feature_moments(model, images, device, store):
    model.eval(); mus, stds = [], []
    with torch.no_grad():
        for img in images[:40]:
            model(_to_tensor(img, device))
            f = store["x"]                       # [1,C,H,W]
            mus.append(f.mean(dim=[0, 2, 3]))
            stds.append(f.std(dim=[0, 2, 3]))
    return torch.stack(mus).mean(0), torch.stack(stds).mean(0)


def proto_align(model, target_images, source_images, device, epochs=5, lr=1e-3):
    store = {}
    conv = _final_conv(model)
    h = conv.register_forward_pre_hook(lambda m, inp: store.__setitem__("x", inp[0]))
    mu_s, std_s = _feature_moments(model, source_images, device, store)
    disable_running_stats(model)
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(iter_norm_affine_params(model))
    for p in params:
        p.requires_grad_(True)
    if params:
        opt = torch.optim.Adam(params, lr=lr)
        model.train()
        for _ in range(epochs):
            for img in target_images:
                opt.zero_grad()
                model(_to_tensor(img, device))
                f = store["x"]
                mu_t = f.mean(dim=[0, 2, 3]); std_t = f.std(dim=[0, 2, 3])
                loss = ((mu_t - mu_s) ** 2).mean() + ((std_t - std_s) ** 2).mean()
                loss.backward(); opt.step()
        model.eval()
    h.remove()
    return model


# ---------- spectral_amplitude_match (input-space) ----------
def _mean_amp(images, size=256):
    acc = None; n = 0
    for img in images[:40]:
        a = np.asarray(img, np.float32)
        if a.ndim == 3: a = a.mean(-1)
        a = np.array(_to_tensor(img, "cpu")[0, 0])  # resized grid
        amp = np.abs(np.fft.fftshift(np.fft.fft2(a)))
        acc = amp if acc is None else acc + amp; n += 1
    return acc / max(n, 1)


class SpectralWrap(nn.Module):
    """Wrap a seg model so inputs get amplitude-matched to source first."""
    def __init__(self, model, src_amp, beta=0.5):
        super().__init__(); self.m = model
        self.register_buffer("src_amp", torch.from_numpy(src_amp).float())
        self.beta = beta
    def forward(self, x):
        # x: [1,3,H,W]; match per-channel amplitude toward src_amp
        out = []
        for c in range(x.shape[1]):
            X = torch.fft.fftshift(torch.fft.fft2(x[:, c]))
            amp = X.abs(); ph = torch.angle(X)
            sa = self.src_amp.to(x.device)
            new_amp = (1 - self.beta) * amp + self.beta * sa
            Xn = new_amp * torch.exp(1j * ph)
            out.append(torch.fft.ifft2(torch.fft.ifftshift(Xn)).real)
        return self.m(torch.stack(out, dim=1))


def main():
    llm = create_llm_client(backend="azure", model="gpt-4.1-2025-04-14")
    nz = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    table = {}  # ds -> {cand: (dice, vlm, fm)}
    for name in DSETS:
        ds = DATASETS[name]; nc = ds["nc"]; cfg = ds.get("cfg", {})
        task = get_task("segmentation")
        ctx = _build_vlm_task_ctx({"structures": cfg.get("structures", {}),
                                    "foreground_priors": {}}, {})
        ctx["num_classes_hint"] = nc
        base = load_source_model(os.path.join(PROJECT_ROOT, ds["ckpt"]), nc,
                                   DEVICE, arch=ds.get("arch", "unet"), project_root=None)
        timgs, tlbls = load_npz_dataset(os.path.join(PROJECT_ROOT, ds["tgt"]),
                                          ds.get("max_target", 50))
        # source images (for proto_align + spectral)
        try:
            simgs, _ = load_npz_dataset(os.path.join(PROJECT_ROOT, ds["src"]), 40)
        except Exception:
            simgs = timgs  # degrade
        anchors = [timgs[i] for i in np.linspace(0, len(timgs) - 1,
                                                   min(6, len(timgs)), dtype=int)]
        cands = {"source": base}
        a = copy.deepcopy(base); baseline_bn_reset(a, timgs, DEVICE); cands["bn_reset"] = a
        b = copy.deepcopy(base); baseline_tent(b, timgs, DEVICE, epochs=5); cands["tent"] = b
        pa = copy.deepcopy(base); proto_align(pa, timgs, simgs, DEVICE); cands["proto_align"] = pa
        try:
            sm = SpectralWrap(copy.deepcopy(base), _mean_amp(simgs)).to(DEVICE).eval()
            cands["spectral_match"] = sm
        except Exception as e:
            print(f"  [spectral skip {name}] {e}")
        table[name] = {}
        for cn, mdl in cands.items():
            dice = _avg_dice(mdl, timgs, tlbls, nc, DEVICE)
            _, _, _, ab = _vlm_iteration_vote(current_model=mdl, source_model=base,
                anchor_imgs=anchors, device=DEVICE, llm_client=llm, task_ctx=ctx, k=6, task=task)
            vlm = float(np.mean([1.0 if v == "usable" else 0.0 for v in ab])) if ab else 0.0
            fm, _ = fm_critic.fm_agreement(mdl, timgs, _to_tensor, nc, DEVICE, n_anchors=10)
            table[name][cn] = (dice, vlm, fm)
            print(f"{name:11s} {cn:14s} Dice={dice:.4f} VLM={vlm:.3f} FM={fm:.3f}")
        del base, a, b, pa
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # bet evaluation
    print("\n=== oracle-best candidate per dataset (rotation) ===")
    from collections import Counter
    best = {n: max(d.items(), key=lambda kv: kv[1][0])[0] for n, d in table.items()}
    for n, c in best.items(): print(f"  {n:11s} -> {c}")
    print("  rotation:", dict(Counter(best.values())), "->", len(set(best.values())), "distinct winners")
    # combined selector
    def sel(d, key):
        return max(d.items(), key=key)[1][0]
    fixed_tent = np.mean([table[n]["tent"][0] for n in DSETS])
    oracle = np.mean([max(table[n].values(), key=lambda v: v[0])[0] for n in DSETS])
    # need normalized vlm/fm per dataset for combo selection
    means = {"VLM+FM": [], "FM": [], "VLM": []}
    for n, d in table.items():
        vmax = max(v[1] for v in d.values()) or 1; fmax = max(v[2] for v in d.values()) or 1
        means["VLM+FM"].append(sel(d, lambda kv: 0.6 * kv[1][1] / vmax + 0.4 * kv[1][2] / fmax))
        means["FM"].append(sel(d, lambda kv: kv[1][2]))
        means["VLM"].append(sel(d, lambda kv: kv[1][1]))
    print("\n=== mean delivered Dice (5-candidate toolbox, n=8) ===")
    print(f"  VLM+FM-select : {np.mean(means['VLM+FM']):.4f}")
    print(f"  FM-select     : {np.mean(means['FM']):.4f}")
    print(f"  VLM-select    : {np.mean(means['VLM']):.4f}")
    print(f"  fixed: tent   : {fixed_tent:.4f}")
    print(f"  oracle        : {oracle:.4f}")
    print(f"\nBET: VLM+FM-select - fixed_tent = {np.mean(means['VLM+FM'])-fixed_tent:+.4f}  "
          f"(3-tool baseline was +0.033)")


if __name__ == "__main__":
    main()
