"""Active-labeling subpackage for MedAdaptAgent.

Provides acquisition scorers that let the agentic system choose WHICH
target samples to request labels for, instead of drawing them passively.
The scorers are task-agnostic: they read only the model + TaskAdapter +
Agent-1 priors that are already in scope for any task in the registry.
"""
from .acquisition import (  # noqa: F401
    score_target_samples,
    select_active_indices,
)
