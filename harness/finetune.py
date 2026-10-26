"""Few-shot FINE-TUNING tools for the deployment harness — the adaptation family that
actually CONSUMES K labeled target examples to TRAIN, rather than only ranking label-free
candidates. Four capacity tiers (head < partial < lora < full) so the agent can match the
label budget: head-only for 3-5 labels, full only when the budget is large.

Task-agnostic: the adapter passes its own `to_tensor` callable and the `task` flag
('seg'|'cls'), so this module imports nothing from the adapters (no circular import).
BN layers are kept in eval() during training — at batch=1 the running stats would be
corrupted; we adapt the affine/conv weights, not the BN statistics.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("head", "partial", "lora", "full")


# --------------------------------------------------------------------------- #
#  parameter selection per capacity tier
# --------------------------------------------------------------------------- #
def _param_leaf_modules(model):
    """Leaf modules that directly own parameters, in registration (~forward) order:
    encoder -> bottleneck -> decoder -> head, so the TAIL is the decoder/head."""
    return [(n, m) for n, m in model.named_modules()
            if list(m.parameters(recurse=False))]


def _head_module(model, nc):
    """The final task head: last Conv2d with out_channels==nc (seg) or last Linear with
    out_features==nc (cls). Robust across UNet/STU-Net/nnU-Net and resnet/vit."""
    head = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == nc:
            head = (n, m)
        elif isinstance(m, nn.Linear) and m.out_features == nc:
            head = (n, m)
    return head


def set_trainable(model, mode, nc, partial_frac=0.4):
    """Freeze everything, then unfreeze the params appropriate to `mode`. Returns the
    number of trainable params (0 means 'nothing to train' — caller should no-op)."""
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "head":
        h = _head_module(model, nc)
        if h:
            for p in h[1].parameters():
                p.requires_grad_(True)
    elif mode == "partial":
        mods = _param_leaf_modules(model)
        k = max(1, int(len(mods) * partial_frac))
        for _, m in mods[-k:]:
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)
        h = _head_module(model, nc)            # always include the head
        if h:
            for p in h[1].parameters():
                p.requires_grad_(True)
    elif mode == "early":
        # SURGICAL fine-tuning (Lee et al. 2023): for INPUT/appearance-level shifts, adapting the
        # EARLIEST layers outperforms head-only tuning (the shift enters at the input). Unfreeze
        # the first ~10% of leaf modules + the head; freeze the middle.
        mods = _param_leaf_modules(model)
        k = max(1, int(len(mods) * 0.10))
        for _, m in mods[:k]:
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)
        h = _head_module(model, nc)
        if h:
            for p in h[1].parameters():
                p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
#  minimal LoRA (low-rank adapters on conv / linear), base frozen
# --------------------------------------------------------------------------- #
class LoRAConv2d(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Conv2d(base.in_channels, r, base.kernel_size, base.stride,
                           base.padding, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)              # start as identity (delta=0)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def _set_submodule(model, dotted, new):
    parent = model
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def inject_lora(model, r=4, alpha=8, tail_frac=0.5):
    """Wrap the Conv2d / Linear modules in the TAIL `tail_frac` of the network with LoRA
    adapters; freeze the base. Returns #trainable params (the LoRA A/B weights)."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    k = max(1, int(len(targets) * tail_frac))
    for n, m in targets[-k:]:
        wrap = LoRAConv2d(m, r, alpha) if isinstance(m, nn.Conv2d) else LoRALinear(m, r, alpha)
        _set_submodule(model, n, wrap)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
#  losses
# --------------------------------------------------------------------------- #
def _seg_loss(logit, lbl):                 # logit [B,nc,H,W], lbl [B,H,W] long
    ce = F.cross_entropy(logit, lbl)
    p = F.softmax(logit, 1)
    oh = F.one_hot(lbl, logit.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (p * oh).sum(dims)
    denom = p.sum(dims) + oh.sum(dims)
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return ce + dice


# --------------------------------------------------------------------------- #
#  the one entry point the adapters call
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  per-dataset/per-budget recipe + collapse-safe entry point
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  per-dataset/per-budget recipe + collapse-safe entry point
# --------------------------------------------------------------------------- #
def _recipe_for(n, mode):
    """Derive (lr, steps) from the LABEL BUDGET n and capacity tier — the lesson from the
    messidor2 collapse: a fixed steps=80/lr=1e-3 overfits a handful of noisy labels. Few
    labels -> gentle (low lr, few steps); larger budgets -> stronger. High-capacity tiers
    (full/partial) get an lr cut to avoid collapse on the noisiest targets."""
    if   n >= 24: lr = 8e-4
    elif n >= 12: lr = 5e-4
    elif n >= 6:  lr = 3e-4
    else:         lr = 2e-4
    if mode in ("full", "partial"):
        lr *= 0.5
    steps = int(min(200, max(30, n * 8)))
    return lr, steps


def _one_loss(logit, lbl, task, nc, device):
    if task == "seg":
        y = torch.as_tensor(np.asarray(lbl), device=device, dtype=torch.long).unsqueeze(0)
        if y.shape[-2:] != logit.shape[-2:]:
            y = F.interpolate(y.unsqueeze(1).float(), size=logit.shape[-2:],
                              mode="nearest").squeeze(1).long()
        return _seg_loss(logit, y)
    y = torch.as_tensor([int(lbl)], device=device, dtype=torch.long)
    if logit.dim() > 2:
        logit = logit.flatten(1)[:, :nc]
    return F.cross_entropy(logit, y)


def finetune(model, imgs, lbls, *, task, nc, device, to_tensor,
             mode="head", steps=None, lr=None, seed=0, val_frac=0.3):
    """Fine-tune a DEEPCOPY of `model` on (imgs, lbls) under `mode` with a PER-BUDGET recipe
    (`_recipe_for`: lr/steps from the label count + capacity tier). When the budget is large
    enough for a meaningful internal val (N >= 13 -> val >= ~4), hold out `val_frac`, track
    val-loss, restore the BEST checkpoint, early-stop on plateau, and if NO step beats the
    source's own val-loss return the SOURCE weights (collapse guard). At SMALL N a 2-sample
    val cannot see a real recovery, so we DON'T steal training data or guard — we train on all
    labels with the gentle recipe and return the final model (preserves few-shot recoveries).
    BN kept in eval() (batch=1 stats unreliable)."""
    if mode not in MODES:
        mode = "head"
    m = copy.deepcopy(model).to(device)
    n_train = inject_lora(m) if mode == "lora" else set_trainable(m, mode, nc)
    m = m.to(device)
    if n_train == 0:
        m.eval()
        return m, {"mode": mode, "trainable_params": 0, "note": "no trainable params; no-op"}
    N = len(imgs)
    r_lr, r_steps = _recipe_for(N, mode)
    lr = r_lr if lr is None else lr
    steps = r_steps if steps is None else steps

    use_val = N >= 13                          # need >= ~4 val samples to be meaningful
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    n_val = max(4, int(round(N * val_frac))) if use_val else 0
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:] if n_val and len(perm) - n_val >= 1 else perm

    def _val_loss():
        if not val_idx:
            return None
        was = m.training; m.eval(); tot = 0.0
        with torch.no_grad():
            for i in val_idx:
                tot += float(_one_loss(m(to_tensor(imgs[i], device)), lbls[i], task, nc, device))
        if was: m.train()
        return tot / len(val_idx)

    def _set_train():
        m.train()
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.eval()

    src_val = _val_loss() if use_val else None
    best_val = src_val
    best_state = copy.deepcopy(m.state_dict()) if use_val else None
    best_step = 0
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    eval_every = max(5, steps // 10)
    patience, bad = 3, 0
    losses = []
    for step in range(steps):
        _set_train()
        i = tr_idx[step % len(tr_idx)]
        loss = _one_loss(m(to_tensor(imgs[i], device)), lbls[i], task, nc, device)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach().cpu()))
        if use_val and (step + 1) % eval_every == 0:
            v = _val_loss()
            if v is not None and (best_val is None or v < best_val - 1e-4):
                best_val, best_step, bad = v, step + 1, 0
                best_state = copy.deepcopy(m.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
    if use_val:
        m.load_state_dict(best_state)
    m.eval()
    guard = bool(use_val and src_val is not None and best_step == 0)
    return m, {"mode": mode, "trainable_params": int(n_train), "steps": steps, "lr": round(lr, 6),
               "loss_first": round(losses[0], 4) if losses else None,
               "loss_last": round(losses[-1], 4) if losses else None,
               "used_val": use_val, "n_val": len(val_idx),
               "src_val_loss": None if src_val is None else round(src_val, 4),
               "best_val_loss": None if best_val is None else round(best_val, 4),
               "picked_step": best_step, "recipe": f"lr={lr:.1e},steps={steps}", "collapse_guard": guard}
