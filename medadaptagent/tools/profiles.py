"""Profile rendering for Agent 3.

Turns the TOOL_REGISTRY into a markdown-table prompt block that Agent 3
can read. Keeps the descriptions short and grouped by category so the
LLM doesn't drown in detail.
"""
from __future__ import annotations
from typing import List, Optional
from .base import TOOL_REGISTRY, Tool


CATEGORY_ORDER = [
    "preprocessing", "spatial", "tta_aug", "stats_adapt",
    "gradient_adapt", "frequency", "postprocessing", "meta",
]


def _render_tool_line(t: Tool) -> str:
    name = f"`{t.name}`"
    return f"- {name}: {t.profile}"


def render_tool_menu(capabilities: Optional[List[str]] = None) -> str:
    """Return a markdown-formatted menu of available tools grouped by
    category. Goes through `available_tools()` so 3D-only filtering
    (supports_3d=False excluded on volumetric tasks) is honored."""
    from .base import available_tools
    visible = available_tools(capabilities)
    visible_names = {t.name for t in visible}
    lines = []
    for cat in CATEGORY_ORDER:
        cat_tools = [t for t in TOOL_REGISTRY.values()
                      if t.category == cat and t.name in visible_names]
        if not cat_tools:
            continue
        lines.append(f"### {cat}")
        for t in cat_tools:
            lines.append(_render_tool_line(t))
        lines.append("")
    return "\n".join(lines).rstrip()


def render_capability_note(capabilities: Optional[List[str]] = None) -> str:
    if not capabilities:
        return ""
    return (
        "Note: this model supports the following capabilities: "
        + ", ".join(sorted(capabilities))
        + ". Tools requiring unavailable capabilities have been omitted.")
