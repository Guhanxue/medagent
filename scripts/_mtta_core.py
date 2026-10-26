#!/usr/bin/env python
"""Shared state + helpers + MEMO/CoTTA for the modern-TTA scoreboard
baselines. Public entry points and T3A/VPTTA live in
scripts/modern_tta_baselines.py. Self-contained, label-free,
dataset-agnostic; uses the host 2D _to_tensor (via set_tensorizer)."""

import copy
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F

from medadaptagent.graph.norm_utils import (
    iter_norm_affine_params, disable_running_stats,
)

_TO_TENSOR = None
EVAL_PREDICTORS = {}


def set_tensorizer(fn):
    global _TO_TENSOR
    _TO_TENSOR = fn


def _to_tensor(img, device, size=256):
    if _TO_TENSOR is None:
        raise RuntimeError("set_tensorizer(host_to_tensor) must be called first.")
    return _TO_TENSOR(img, device, size)


def _model_num_classes(model, target_images, device):
    was = model.training
    model.eval()
    with torch.no_grad():
        logits = model(_to_tensor(target_images[0], device))
    if was:
        model.train()
    return int(logits.shape[1])


def _find_final_classifier(model, num_classes):
    # Arch-agnostic seg head: last conv, out_channels==num_classes, singleton
    # kernel (UNet/nnUNet out_conv, STUNet seg_outputs). No name/dataset assume.
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    last = None
    for mod in model.modules():
        if isinstance(mod, conv_types) and mod.out_channels == num_classes:
            ks = mod.kernel_size
            ks = ks if isinstance(ks, tuple) else (ks,)
            if all(k == 1 for k in ks):
                last = mod
    return last


def _augment(t, n_aug):
    views = [
        (t, False), (t.flip(-1), True),
        (torch.clamp(t * 1.1, 0, 1), False),
        (torch.clamp(t * 0.9, 0, 1), False),
        (torch.clamp((t - 0.5) * 1.15 + 0.5, 0, 1), False),
        (torch.clamp((t - 0.5) * 0.85 + 0.5, 0, 1), False),
        (torch.clamp(t + 0.02 * torch.randn_like(t), 0, 1), False),
        (torch.clamp(t.pow(1.1), 0, 1), False),
    ]
    return views[:max(2, n_aug)]


def baseline_memo(model, target_images, device, lr=1e-4, n_aug=8):
    # MEMO (Zhang 2022): episodic single-sample marginal-entropy. NOTE: updates
    # the norm-affine set (norm-agnostic) not ALL params -- the only deviation
    # from the paper, for parity with the other affine-only rows. Returns
    # source, registers EVAL_PREDICTORS['memo'].
    disable_running_stats(model)
    src_state = copy.deepcopy(model.state_dict())
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(iter_norm_affine_params(model))
    for p in params:
        p.requires_grad_(True)
    if not params:
        EVAL_PREDICTORS["memo"] = None
        return model

    def _adapt_one(img, size=256):
        model.load_state_dict(src_state)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)
        t = _to_tensor(img, device, size)
        model.train()
        opt.zero_grad()
        probs = []
        for aug_t, hflip in _augment(t, n_aug):
            p = F.softmax(model(aug_t), dim=1)
            probs.append(p.flip(-1) if hflip else p)
        p_bar = torch.stack(probs, dim=0).mean(dim=0)
        ent = -(p_bar * torch.log(p_bar + 1e-10)).sum(dim=1).mean()
        ent.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(_to_tensor(img, device, size)).argmax(dim=1).cpu().numpy()[0]
        return pred.astype(np.uint8)

    EVAL_PREDICTORS["memo"] = _adapt_one
    model.load_state_dict(src_state)
    model.eval()
    return model


def baseline_cotta(model, target_images, device, lr=1e-3, alpha=0.999,
                   restore_prob=0.01, aug_conf_thresh=0.1, n_aug=4, epochs=1):
    # CoTTA (Wang 2022): EMA weight-averaged teacher; uncertain teacher
    # (entropy>thr) -> aug-averaged pseudo-label, else single. Student
    # (norm-affine, batch-stat) trained by soft CE; stochastic restore of a
    # random restore_prob fraction of affine weights each step. NON-episodic.
    num_classes = _model_num_classes(model, target_images, device)
    conf_thr = aug_conf_thresh * float(np.log(max(num_classes, 2)))
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    src_named = {k: v.detach().clone() for k, v in model.state_dict().items()}
    disable_running_stats(model)
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(iter_norm_affine_params(model))
    for p in params:
        p.requires_grad_(True)
    if not params:
        EVAL_PREDICTORS["cotta"] = None
        return model
    opt = torch.optim.Adam(params, lr=lr)

    def _augs(t):
        v = [t, t.flip(-1), torch.clamp(t * 1.1, 0, 1),
             torch.clamp((t - 0.5) * 1.15 + 0.5, 0, 1)]
        f = [False, True, False, False]
        return list(zip(v, f))[:max(1, n_aug)]

    model.train()
    for _ in range(epochs):
        for img in target_images:
            t = _to_tensor(img, device)
            with torch.no_grad():
                tp = F.softmax(teacher(t), dim=1)
                ent = -(tp * torch.log(tp + 1e-10)).sum(dim=1).mean()
                if ent.item() > conf_thr:
                    acc, n = tp.clone(), 1
                    for v, hflip in _augs(t)[1:]:
                        vp = F.softmax(teacher(v), dim=1)
                        acc = acc + (vp.flip(-1) if hflip else vp)
                        n += 1
                    pseudo = (acc / n).detach()
                else:
                    pseudo = tp.detach()
            opt.zero_grad()
            s_logp = F.log_softmax(model(t), dim=1)
            loss = -(pseudo * s_logp).sum(dim=1).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                s_sd, t_sd = model.state_dict(), teacher.state_dict()
                for k in t_sd.keys():
                    if k not in s_sd:
                        continue  # NOTE: student loses running_mean/var under disable_running_stats; EMA covers affine+conv only (faithful seg variant)
                    if t_sd[k].dtype.is_floating_point:
                        t_sd[k].mul_(alpha).add_(s_sd[k].detach(), alpha=1 - alpha)
                    else:
                        t_sd[k].copy_(s_sd[k])
            if restore_prob > 0:
                _stochastic_restore(model, src_named, restore_prob)
    EVAL_PREDICTORS["cotta"] = None
    model.eval()
    return model


def _stochastic_restore(model, src_state, restore_prob):
    affine_ids = {id(p) for p in iter_norm_affine_params(model)}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if id(p) in affine_ids and name in src_state:
                mask = (torch.rand_like(p) < restore_prob).float()
                p.mul_(1.0 - mask).add_(src_state[name].to(p.device) * mask)


def avg_dice_via_predictor(predictor, images, labels, num_classes, size=256):
    # Mirrors the host _avg_dice for an episodic per-image predictor (parity).
    per_class = {c: [] for c in range(1, num_classes)}
    for img, lbl in zip(images, labels):
        if lbl is None:
            continue
        pred = predictor(img)
        lbl_pil = PILImage.fromarray(lbl.astype(np.int32).astype(np.uint8))
        lbl_r = np.array(lbl_pil.resize((size, size), PILImage.NEAREST))
        for c in range(1, num_classes):
            if num_classes == 3 and c == 1:
                p, g = (pred >= 1).astype(float), (lbl_r >= 1).astype(float)
            else:
                p, g = (pred == c).astype(float), (lbl_r == c).astype(float)
            if g.sum() > 0:
                per_class[c].append(float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8)))
    means = [float(np.mean(v)) for v in per_class.values() if v]
    return float(np.mean(means)) if means else 0.0
