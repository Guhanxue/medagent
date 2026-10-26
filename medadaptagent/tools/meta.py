"""Meta-tools: noop, chain, weight averaging."""
from __future__ import annotations
import copy
from typing import List
import torch
from .base import Tool, ToolContext, register, apply_tool


def _fn_noop(ctx: ToolContext) -> ToolContext:
    """Do nothing. Used when Agent 3 decides the source model is already
    fine, or when no label-free signal suggests adaptation."""
    ctx.notes.setdefault("noop", {})["applied"] = True
    return ctx


def _fn_chain(ctx: ToolContext) -> ToolContext:
    """Apply a list of tool names in sequence. The list is read from
    ctx.notes['chain']['tools'] (list of strings). Chain is flat; no
    parallel branching. Per-tool kwargs can be passed via
    ctx.notes['chain']['per_tool_kwargs'] = {tool_name: {...}}."""
    notes = ctx.notes.get("chain", {})
    tools: List[str] = notes.get("tools", [])
    per_kw = notes.get("per_tool_kwargs", {}) or {}
    for name in tools:
        if name == "chain":
            continue  # prevent infinite recursion
        kw = per_kw.get(name, {})
        if kw:
            ctx.notes.setdefault(name, {}).update(kw)
        ctx = apply_tool(name, ctx)
    return ctx


def _fn_weight_average(ctx: ToolContext) -> ToolContext:
    """0.5 * source + 0.5 * current model weights (layer-wise).
    Safe regularizer to pull an over-adapted model back toward source.
    Alpha tunable via ctx.notes['weight_average']['alpha'] (weight on
    current_model, 0..1). Default 0.5."""
    notes = ctx.notes.get("weight_average", {})
    alpha = float(notes.get("alpha", 0.5))
    src_state = ctx.source_model.state_dict()
    cur_state = ctx.current_model.state_dict()
    merged = {}
    for k in cur_state:
        if k not in src_state or cur_state[k].dtype != src_state[k].dtype:
            merged[k] = cur_state[k]
            continue
        if not torch.is_floating_point(cur_state[k]):
            merged[k] = cur_state[k]
            continue
        merged[k] = alpha * cur_state[k] + (1.0 - alpha) * src_state[k].to(
            cur_state[k].device)
    ctx.current_model.load_state_dict(merged, strict=False)
    ctx.notes.setdefault("weight_average", {})["alpha_used"] = alpha
    return ctx


register(Tool(
    name="noop", fn=_fn_noop, category="meta", side_effects=[],
    profile=(
        "MECHANISM: identity — returns the source model unchanged. Use as the safe baseline candidate when adaptation is risky. Do nothing. Return the source model unchanged. Use when source "
        "is already at-or-above baseline, or when all label-free signals "
        "suggest adaptation is risky. Always safe."),
))
register(Tool(
    name="chain", fn=_fn_chain, category="meta", side_effects=[],
    profile=(
        "MECHANISM: sequential application of multiple tools' mechanisms; left to right; the chain is implicit in the tools list. Apply a list of tools in sequence (e.g., histogram_match then "
        "tent). Tool list is passed via ctx.notes['chain']['tools']. "
        "Useful for preprocessing + adaptation combinations."),
))
register(Tool(
    name="weight_average_source_adapted", fn=_fn_weight_average,
    category="meta", side_effects=["model_weights"],
    profile=(
        "MECHANISM: linear interpolation between source and adapted weights at alpha — anchors a possibly-overadapted model back toward source. Blend 0.5*source + 0.5*current weights. Safe regularizer when "
        "an adaptation step may have over-shot. Acts as a soft anchor "
        "toward source. Alpha tunable."),
))
