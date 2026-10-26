#!/usr/bin/env python
"""Modern TTA baselines (CoTTA 2022, T3A 2021, MEMO 2022, VPTTA 2024) for the
MedAdaptAgent auto-adapt scoreboard. Public entry points; helpers + MEMO/CoTTA
live in scripts._mtta_core. Self-contained, label-free, dataset-agnostic. Uses
the host 2D _to_tensor (via set_tensorizer); NOT 3D-safe. Episodic methods
(MEMO, VPTTA) register a per-image closure in EVAL_PREDICTORS[name]; the host
scores them via avg_dice_via_predictor so reported Dice reflects real per-image
adaptation, not the persisted source."""

import numpy as np
import torch
import torch.nn.functional as F

from scripts._mtta_core import (
    set_tensorizer, EVAL_PREDICTORS, _to_tensor,
    _model_num_classes, _find_final_classifier,
    baseline_memo, baseline_cotta, avg_dice_via_predictor,
)

__all__ = [
    "set_tensorizer", "EVAL_PREDICTORS",
    "baseline_memo", "baseline_cotta", "baseline_t3a", "baseline_vptta",
    "avg_dice_via_predictor",
]


def baseline_t3a(model, target_images, device, n_support=20,
                 conf_filter_frac=0.5):
    # T3A (Iwasawa 2021): training-free, backbone frozen. NOTE/seg-variant: a
    # 2D-UNet has no global penultimate vector, so T3A runs per-pixel:
    # penultimate features = input to the final 1x1 conv (forward pre-hook,
    # arch-agnostic). Per class, prototype = confidence-weighted mean of
    # penultimate vectors over top conf_filter_frac confident pixels, seeded
    # from the source classifier rows (T3A support init). The head forward is
    # swapped to cosine-to-prototype. cls archs reduce to literal Linear-T3A.
    num_classes = _model_num_classes(model, target_images, device)
    head = _find_final_classifier(model, num_classes)
    if head is None:
        print("  [t3a] NOTE: no 1x1 classification head found; T3A no-op "
              "on this arch (returning source).")
        EVAL_PREDICTORS["t3a"] = None
        model.eval()
        return model
    holder = {}

    def _pre(module, inp):
        holder["x"] = inp[0].detach()

    handle = head.register_forward_pre_hook(_pre)
    w = head.weight.detach()
    feat_dim = w.shape[1]
    seed = F.normalize(w.reshape(num_classes, feat_dim), dim=1)
    psum = torch.zeros(num_classes, feat_dim, device=device)
    pcnt = torch.zeros(num_classes, device=device)
    support = target_images[:max(1, n_support)] if n_support else target_images
    model.eval()
    with torch.no_grad():
        for img in support:
            logits = model(_to_tensor(img, device))
            f = holder["x"]
            assert f.shape[0] == 1, "T3A per-pixel reshape assumes batch==1"
            prob = F.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
            fc = conf.reshape(-1)
            fp = pred.reshape(-1)
            ff = f.reshape(feat_dim, -1).t()
            if conf_filter_frac < 1.0:
                k = max(1, int(fc.numel() * conf_filter_frac))
                keep = torch.topk(fc, k).indices
                fc, fp, ff = fc[keep], fp[keep], ff[keep]
            ff = F.normalize(ff, dim=1)
            for c in range(num_classes):
                m = (fp == c)
                if m.any():
                    psum[c] += (ff[m] * fc[m].unsqueeze(1)).sum(0)
                    pcnt[c] += fc[m].sum()
    handle.remove()
    protos = seed.clone()
    for c in range(num_classes):
        if pcnt[c] > 0:
            protos[c] = F.normalize(psum[c] / pcnt[c], dim=0)
    _install_prototype_head(head, protos)
    EVAL_PREDICTORS["t3a"] = None
    model.eval()
    return model


def _install_prototype_head(head, protos):
    # Cosine-to-prototype final conv (unit prototype weights + per-pixel feature
    # normalization, temperature-scaled). Keeps model(x) -> logits intact.
    feat_dim = protos.shape[1]
    extra = head.weight.ndim - 2
    proto_w = protos.reshape(protos.shape[0], feat_dim, *([1] * extra))
    with torch.no_grad():
        head.weight.copy_(proto_w.to(head.weight.dtype))
        if head.bias is not None:
            head.bias.zero_()
    _orig = head.forward
    temperature = 10.0

    def _cos(x):
        return _orig(F.normalize(x, dim=1)) * temperature

    head.forward = _cos


def baseline_vptta(model, target_images, device, lr=0.05, prompt_alpha=0.01,
                   warm_n=5):
    # VPTTA (Chen 2024): per-image low-frequency Fourier-amplitude visual prompt
    # warmed up by entropy on the FROZEN model (backbone never updated).
    # NOTE/SIMPLIFICATION: published VPTTA also keeps a memory bank and
    # warm-STARTS each prompt from nearest neighbors; external_baselines/VPTTA
    # implements that but is bolted to a specific ResUNet + fundus/polyp
    # dataloaders and is not wireable into this dataset-agnostic 2D-UNet path,
    # so we use the prompt-warmup variant WITHOUT the memory-bank warm-start.
    # Episodic: returns frozen source, registers EVAL_PREDICTORS['vptta'].
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    def _apply(t, prompt, pad, ps):
        _, _, H, W = t.shape
        fft = torch.fft.fft2(t, dim=(-2, -1))
        amp, pha = torch.abs(fft), torch.angle(fft)
        amp = torch.fft.fftshift(amp)
        full = F.pad(prompt, [pad, W - pad - ps, pad, H - pad - ps],
                     mode="constant", value=1.0)
        amp = torch.fft.ifftshift(amp * full)
        recon = torch.fft.ifft2(
            torch.complex(torch.cos(pha) * amp, torch.sin(pha) * amp),
            dim=(-2, -1), s=[H, W]).real
        return recon

    def _adapt_one(img, size=256):
        t = _to_tensor(img, device, size)
        _, c2, h2, _ = t.shape
        ps = max(1, int(h2 * prompt_alpha))
        pad = (h2 - ps) // 2
        prompt = torch.ones((1, c2, ps, ps), device=device, requires_grad=True)
        opt = torch.optim.Adam([prompt], lr=lr)
        for _s in range(warm_n):
            opt.zero_grad()
            probs = F.softmax(model(_apply(t, prompt, pad, ps)), dim=1)
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            ent.backward()
            opt.step()
        with torch.no_grad():
            warped = _apply(t, prompt.detach(), pad, ps)
            pred = model(warped).argmax(dim=1).cpu().numpy()[0]
        return pred.astype(np.uint8)

    EVAL_PREDICTORS["vptta"] = _adapt_one
    model.eval()
    return model
