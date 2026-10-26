"""MedAdaptAgent tool registry — unified toolbox for Agent 3/4.

A ToolContext flows through a sequence of tools; each tool mutates one
of: target_images preprocessing, model weights, inference forward,
prediction postprocessing. Agent 3 reasons over `render_tool_menu()`
and picks one tool (or a chain).

Public API:
  - ToolContext, Tool, TOOL_REGISTRY, register, get, apply_tool,
    available_tools, run_inference   (from base)
  - render_tool_menu, render_capability_note  (from profiles)
"""
from .base import (
    ToolContext,
    Tool,
    TOOL_REGISTRY,
    register,
    get,
    apply_tool,
    available_tools,
    run_inference,
)

# Side-effect imports: registering each tool into TOOL_REGISTRY
from . import preprocessing  # noqa: F401
from . import spatial  # noqa: F401
from . import tta_aug  # noqa: F401
from . import adapt  # noqa: F401
from . import frequency  # noqa: F401
from . import postprocessing  # noqa: F401
from . import meta  # noqa: F401

from .profiles import render_tool_menu, render_capability_note  # noqa: F401

# Phase D4: tool composition layer (atoms -> ordered pipeline). Additive
# on top of the registry; does NOT register anything new.
from . import composition  # noqa: F401
from .composition import (  # noqa: F401
    Atom,
    normalize_composition,
    composition_to_legacy,
    validate_composition,
    render_composition,
)


__all__ = [
    "ToolContext", "Tool", "TOOL_REGISTRY", "register", "get",
    "apply_tool", "available_tools", "run_inference",
    "render_tool_menu", "render_capability_note",
    "composition", "Atom", "normalize_composition",
    "composition_to_legacy", "validate_composition", "render_composition",
]
