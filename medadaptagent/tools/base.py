"""Tool base classes for the MedAdaptAgent toolbox.

A Tool is a small, composable operation the pipeline can apply to a
(source_model, target_images) context. Tools mutate a ToolContext object
in-place or return a new one. All tools share the same signature so
Agent 3 can reason over a uniform set of options.

Categories:
  preprocessing   — intensity/histogram/bias transforms on target images
  spatial         — crop/resize/flip/reorient on target images
  tta_aug         — inference-time augmentation (flip TTA, multi-scale)
  stats_adapt     — non-gradient model adaptation (BN reset)
  gradient_adapt  — gradient-based TTA (Tent, SAR, EATA, SFDA, …)
  frequency       — Fourier / FDA methods
  postprocessing  — mask cleanup (largest CC, hole fill, …)
  meta            — noop, chain(t1, t2), etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from medadaptagent.tasks.base import TaskAdapter


@dataclass
class ToolContext:
    """Mutable state passed between tools.

    Each tool reads and/or mutates some of these fields. The pipeline
    runs forward inference through `apply_inference(ctx, img)` which
    composes `preprocess_fn -> inference_fn -> postprocess_fn`.

    Fields:
      source_model     — original source checkpoint, never modified
      current_model    — model used for forward (may have adapted weights)
      target_images    — list of H×W or H×W×C numpy images
      device           — e.g. "cuda:0"
      num_classes      — predicted classes including background
      preprocess_fn    — applied per-image before model forward
      postprocess_fn   — applied per-prediction mask after argmax
      inference_fn     — custom (model, img_tensor) -> logits (None = default)
      tools_applied    — running list of tool names applied
      notes            — free-form per-tool log entries
      fewshot_indices  — Optional. Indices into target_images that have
                          GT labels. Set when --n_labels K > 0. Used by
                          semi-supervised tool variants (tent_ss / sar_ss /
                          eata_ss) for the supervised CE term.
      fewshot_labels   — Optional. List of np.ndarray label maps aligned
                          with fewshot_indices.
    """
    source_model: nn.Module
    current_model: nn.Module
    target_images: List[np.ndarray]
    device: str
    num_classes: int
    preprocess_fn: Callable[[np.ndarray], np.ndarray] = field(
        default_factory=lambda: (lambda x: x))
    postprocess_fn: Callable[[np.ndarray], np.ndarray] = field(
        default_factory=lambda: (lambda x: x))
    inference_fn: Optional[Callable] = None
    tools_applied: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)
    fewshot_indices: Optional[List[int]] = None
    fewshot_labels: Optional[List[np.ndarray]] = None
    task: Optional["TaskAdapter"] = None  # set by run_execution_engineer
                                            # Default at use-site: seg


@dataclass
class Tool:
    """Tool descriptor.

    - fn        : callable that takes a ToolContext and returns one
    - category  : grouping label for Agent 3
    - profile   : short human-readable "when it helps / when it hurts"
                  text shown to Agent 3 in its prompt. Keep to 1-3 lines.
    - requires  : capability tags (e.g. 'bn_running_stats', 'norm_affine')
    - side_effects : what this tool mutates on the ctx, for audit
    """
    name: str
    fn: Callable[[ToolContext], ToolContext]
    category: str
    profile: str
    requires: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    # Tools whose implementation assumes 2D inputs (PIL, cv2.CLAHE,
    # 2D FFT, axis-0/1 cropping, etc.) declare supports_3d=False.
    # Capability filter in the agent excludes them on 3D tasks.
    supports_3d: bool = True


TOOL_REGISTRY: Dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Register a tool into the global registry. Idempotent."""
    TOOL_REGISTRY[tool.name] = tool
    return tool


def get(name: str) -> Tool:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"unknown tool '{name}'. Known: "
                       f"{sorted(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[name]


def apply_tool(name: str, ctx: ToolContext, **kwargs) -> ToolContext:
    """Dispatch by name. Extra kwargs are passed through via ctx.notes
    under the tool's name (so tools can read per-invocation overrides).
    """
    tool = get(name)
    if kwargs:
        ctx.notes.setdefault(name, {}).update(kwargs)
    ctx = tool.fn(ctx)
    ctx.tools_applied.append(name)
    return ctx


def available_tools(capabilities: Optional[List[str]] = None) -> List[Tool]:
    """Return tools whose `requires` are satisfied by `capabilities`.
    If capabilities is None, return all tools (unfiltered).

    When `volumetric_predictions` is in capabilities (the 3D-task tag),
    additionally exclude tools whose `supports_3d` is False — their
    implementations assume 2D inputs (PIL, cv2.CLAHE, 2D FFT) and
    would crash or silently corrupt 3D volumes."""
    if capabilities is None:
        return list(TOOL_REGISTRY.values())
    caps = set(capabilities)
    is_3d = "volumetric_predictions" in caps
    out = []
    for t in TOOL_REGISTRY.values():
        if not all(r in caps for r in t.requires):
            continue
        if is_3d and not getattr(t, "supports_3d", True):
            continue
        out.append(t)
    return out


# --------------------------------------------------------------
# Inference helper — runs ctx's preprocess → model → postprocess
# --------------------------------------------------------------

def _default_inference(model: nn.Module, img_tensor: torch.Tensor) -> torch.Tensor:
    return model(img_tensor)


def run_inference(ctx: ToolContext, img: np.ndarray,
                    to_tensor_fn: Callable) -> np.ndarray:
    """Run the full preprocess → model → postprocess chain on one image.
    `to_tensor_fn(img, device) -> 4D tensor` is supplied by caller so we
    don't need to know the exact resize/normalize convention here.
    Returns a 2D uint8 argmax mask.
    """
    pre = ctx.preprocess_fn(img)
    t = to_tensor_fn(pre, ctx.device)
    inf = ctx.inference_fn or _default_inference
    with torch.no_grad():
        logits = inf(ctx.current_model, t)
        pred = logits.argmax(dim=1).cpu().numpy()[0]
    return ctx.postprocess_fn(pred)
