"""
Adapter Bank: Registry for managing PEFT adapters.
Handles registration, injection/removal, config-driven activation,
and parameter isolation (only adapter params are trainable).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from .style_adapter import StyleAdapter
from .boundary_adapter import BoundaryAdapter
from .scale_adapter import ScaleAdapter
from .semantic_adapter import SemanticAdapter


@dataclass
class AdapterConfig:
    """Configuration for an adapter instance."""
    name: str
    enabled: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    # Adapter-specific hyperparameters
    kwargs: dict = field(default_factory=dict)


ADAPTER_CLASSES = {
    "style": StyleAdapter,
    "boundary": BoundaryAdapter,
    "scale": ScaleAdapter,
    "semantic": SemanticAdapter,
}

# Failure type → recommended adapter mapping
FAILURE_TO_ADAPTER = {
    "miss": ["semantic"],
    "hallucination": ["style"],
    "boundary_error": ["boundary", "style"],
    "class_confusion": ["semantic", "scale"],
}


class AdapterBank(nn.Module):
    """Central registry and manager for all PEFT adapters.

    Manages creation, injection, removal, and optimization of adapters.
    Only adapter parameters are trainable; the base model stays frozen.
    """

    def __init__(self, base_model_dim: int = 256, text_dim: int = 512,
                 num_classes: int = 4):
        super().__init__()
        self.base_model_dim = base_model_dim
        self.text_dim = text_dim
        self.num_classes = num_classes

        self.adapters = nn.ModuleDict()
        self.adapter_configs: Dict[str, AdapterConfig] = {}
        self.active_adapters: Set[str] = set()

    def register_adapter(self, name: str, config: AdapterConfig = None) -> nn.Module:
        """Register and create an adapter by name."""
        if name not in ADAPTER_CLASSES:
            raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTER_CLASSES.keys())}")

        if config is None:
            config = AdapterConfig(name=name)
        self.adapter_configs[name] = config

        if name == "semantic":
            kwargs = {
                "vis_dim": self.base_model_dim,
                "text_dim": self.text_dim,
                "n_classes": self.num_classes,
            }
        else:
            kwargs = {"dim": self.base_model_dim}
        kwargs.update(config.kwargs)

        adapter = ADAPTER_CLASSES[name](**kwargs)
        self.adapters[name] = adapter

        if config.enabled:
            self.active_adapters.add(name)

        return adapter

    def register_all(self, configs: Dict[str, AdapterConfig] = None):
        """Register all available adapters."""
        for name in ADAPTER_CLASSES:
            cfg = configs.get(name) if configs else None
            self.register_adapter(name, cfg)

    def activate(self, names: List[str]):
        """Activate specific adapters (deactivate others)."""
        self.active_adapters = set()
        for name in names:
            if name not in self.adapters:
                self.register_adapter(name)
            self.active_adapters.add(name)
            self.adapter_configs[name].enabled = True

    def deactivate(self, names: List[str]):
        """Deactivate specific adapters."""
        for name in names:
            self.active_adapters.discard(name)
            if name in self.adapter_configs:
                self.adapter_configs[name].enabled = False

    def deactivate_all(self):
        """Deactivate all adapters."""
        self.active_adapters.clear()
        for config in self.adapter_configs.values():
            config.enabled = False

    def get_active_adapters(self) -> Dict[str, nn.Module]:
        """Return dict of currently active adapter modules."""
        return {
            name: self.adapters[name]
            for name in self.active_adapters
            if name in self.adapters
        }

    def get_adapter(self, name: str) -> Optional[nn.Module]:
        """Get adapter by name."""
        return self.adapters[name] if name in self.adapters else None

    def is_active(self, name: str) -> bool:
        """Check if an adapter is active."""
        return name in self.active_adapters

    def get_trainable_params(self) -> List[Dict]:
        """Get optimizer parameter groups for active adapters."""
        param_groups = []
        for name in self.active_adapters:
            if name in self.adapters:
                config = self.adapter_configs[name]
                param_groups.append({
                    "params": list(self.adapters[name].parameters()),
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "name": name,
                })
        return param_groups

    def total_trainable_params(self) -> int:
        """Total trainable parameters across all registered adapters."""
        total = 0
        for adapter in self.adapters.values():
            total += sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        return total

    def active_trainable_params(self) -> int:
        """Trainable parameters for currently active adapters only."""
        total = 0
        for name in self.active_adapters:
            if name in self.adapters:
                total += sum(
                    p.numel() for p in self.adapters[name].parameters()
                    if p.requires_grad
                )
        return total

    def adapters_from_failure_report(self, failure_distribution: Dict[str, float],
                                     threshold: float = 0.15) -> List[str]:
        """Select adapters based on failure type distribution.

        Args:
            failure_distribution: Dict mapping failure types to proportions.
                e.g., {"boundary_error": 0.4, "miss": 0.3, "hallucination": 0.2, "class_confusion": 0.1}
            threshold: Minimum proportion to activate the corresponding adapter.

        Returns:
            List of adapter names to activate.
        """
        selected = set()
        for failure_type, proportion in failure_distribution.items():
            if proportion >= threshold and failure_type in FAILURE_TO_ADAPTER:
                for adapter_name in FAILURE_TO_ADAPTER[failure_type]:
                    selected.add(adapter_name)
        return list(selected)

    def summary(self) -> str:
        """Print summary of all adapters and their status."""
        lines = ["AdapterBank Summary:"]
        lines.append(f"  Total registered: {len(self.adapters)}")
        lines.append(f"  Active: {len(self.active_adapters)}")
        lines.append(f"  Total params: {self.total_trainable_params():,}")
        lines.append(f"  Active params: {self.active_trainable_params():,}")
        lines.append("")
        for name, adapter in self.adapters.items():
            status = "ACTIVE" if name in self.active_adapters else "inactive"
            params = sum(p.numel() for p in adapter.parameters())
            lines.append(f"  [{status}] {name}: {params:,} params")
        return "\n".join(lines)

    def state_dict_adapters_only(self) -> Dict:
        """Save only adapter states (not base model)."""
        return {
            "adapters": {name: adapter.state_dict() for name, adapter in self.adapters.items()},
            "active": list(self.active_adapters),
            "configs": {name: vars(cfg) for name, cfg in self.adapter_configs.items()},
        }

    def load_state_dict_adapters_only(self, state: Dict):
        """Load adapter states."""
        for name, adapter_state in state["adapters"].items():
            if name in self.adapters:
                self.adapters[name].load_state_dict(adapter_state)
        self.active_adapters = set(state.get("active", []))
