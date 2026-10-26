"""
Norm-agnostic helpers used by every agent so the pipeline supports BN / IN /
LN / GN models through a single dispatch, not per-architecture branches.

Contract (2026-04-20): adding a new architecture is purely additive. No agent
code may contain `isinstance(x, BatchNorm*)` — everything goes through here.

Norm-type taxonomy:
    "batchnorm"     → nn.BatchNorm{1,2,3}d           (has running_mean / running_var)
    "instancenorm"  → nn.InstanceNorm{1,2,3}d        (affine only unless track_running_stats)
    "layernorm"     → nn.LayerNorm                   (affine only)
    "groupnorm"     → nn.GroupNorm                   (affine only)

Capabilities (what a TTA mechanism can rely on, derived from norm type):
    "tent_affine"        — can update `weight` and `bias` of norm layers via entropy-min.
                            Present on ALL norm types.
    "bn_running_stats"   — can reset / replace running statistics (batch-stat
                            recalibration). Present only on BatchNorm, and on
                            InstanceNorm when `track_running_stats=True`.

Every TTA mechanism in `empirical_protocols.py` declares the capabilities it
needs; `get_available_protocols(capabilities)` filters the registry.
"""

from typing import Iterator, List, Tuple, Set

import torch.nn as nn


_BATCHNORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_INSTANCENORMS = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
_LAYERNORMS = (nn.LayerNorm,)
_GROUPNORMS = (nn.GroupNorm,)


def norm_type_of(module: nn.Module) -> str:
    """Return the norm-type tag for a module, or "" if it's not a norm layer."""
    if isinstance(module, _BATCHNORMS):
        return "batchnorm"
    if isinstance(module, _INSTANCENORMS):
        return "instancenorm"
    if isinstance(module, _LAYERNORMS):
        return "layernorm"
    if isinstance(module, _GROUPNORMS):
        return "groupnorm"
    return ""


def iter_norm_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module, str]]:
    """Yield (name, module, norm_type) for every norm layer in the model."""
    for name, mod in model.named_modules():
        nt = norm_type_of(mod)
        if nt:
            yield name, mod, nt


def iter_norm_affine_params(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield the learnable affine params (weight + bias) of every norm layer.
    This is the parameter set Tent-family methods update."""
    for _, mod, _ in iter_norm_modules(model):
        if getattr(mod, "weight", None) is not None and mod.weight.requires_grad is not False:
            yield mod.weight
        if getattr(mod, "bias", None) is not None and mod.bias.requires_grad is not False:
            yield mod.bias


def has_running_stats(module: nn.Module) -> bool:
    """True iff the norm layer tracks running_mean / running_var (eligible
    for BN-style stat resetting)."""
    if isinstance(module, _BATCHNORMS):
        return True
    if isinstance(module, _INSTANCENORMS):
        return bool(getattr(module, "track_running_stats", False))
    return False


def reset_norm_running_stats(model: nn.Module) -> int:
    """Reset running_mean/var to identity on every norm layer that has them.
    Returns the number of layers actually reset (0 for IN/LN/GN models)."""
    n_reset = 0
    for _, mod, _ in iter_norm_modules(model):
        if has_running_stats(mod):
            if hasattr(mod, "running_mean") and mod.running_mean is not None:
                mod.running_mean.zero_()
            if hasattr(mod, "running_var") and mod.running_var is not None:
                mod.running_var.fill_(1.0)
            if hasattr(mod, "num_batches_tracked") and mod.num_batches_tracked is not None:
                mod.num_batches_tracked.zero_()
            n_reset += 1
    return n_reset


def disable_running_stats(model: nn.Module) -> int:
    """Switch BN/IN-with-tracking layers to use batch statistics in their
    forward pass (Tent-style). Sets `track_running_stats = False` and clears
    the buffers. Returns the number of layers switched. No-op on LN/GN."""
    n = 0
    for _, mod, _ in iter_norm_modules(model):
        if has_running_stats(mod):
            mod.track_running_stats = False
            mod.running_mean = None
            mod.running_var = None
            n += 1
    return n


def restore_running_stats_from(source_model: nn.Module,
                                 adapted_model: nn.Module) -> int:
    """After `disable_running_stats` has been called on adapted_model, copy
    the source_model's running stats back into adapted_model's norm layers
    so the state_dict is a complete, loadable checkpoint.

    Only copies to layers that have had their stats dropped. No-op for LN/GN
    (which never have running stats) and for models whose BN/IN still track.
    Returns the number of layers whose stats were restored.
    """
    import torch
    n = 0
    src_modules = dict(source_model.named_modules())
    for name, mod, nt in iter_norm_modules(adapted_model):
        if nt not in ("batchnorm", "instancenorm"):
            continue
        sm = src_modules.get(name)
        if sm is None:
            continue
        if mod.running_mean is None and getattr(sm, "running_mean", None) is not None:
            mod.track_running_stats = True
            mod.register_buffer("running_mean", sm.running_mean.clone())
            mod.register_buffer("running_var", sm.running_var.clone())
            mod.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long,
                              device=sm.running_mean.device),
            )
            n += 1
    return n


def detect_norm_type(model: nn.Module) -> str:
    """Return the dominant norm type in the model. Useful for logging and for
    gating protocol capabilities. If no norm layers found, returns "none"."""
    counts: dict = {}
    for _, _, nt in iter_norm_modules(model):
        counts[nt] = counts.get(nt, 0) + 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def model_capabilities(model: nn.Module) -> Set[str]:
    """Which TTA capabilities are supported by this model? Protocol filtering
    in `empirical_protocols.get_available_protocols` reads this."""
    caps: Set[str] = set()
    for _, mod, _ in iter_norm_modules(model):
        caps.add("tent_affine")
        if has_running_stats(mod):
            caps.add("bn_running_stats")
    return caps


def count_norm_layers(model: nn.Module) -> dict:
    """Diagnostic helper: how many layers of each norm type. Used in model
    info / Agent 2 prompts."""
    counts = {"batchnorm": 0, "instancenorm": 0, "layernorm": 0, "groupnorm": 0}
    for _, _, nt in iter_norm_modules(model):
        counts[nt] = counts.get(nt, 0) + 1
    return counts
