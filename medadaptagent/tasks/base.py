"""TaskAdapter — the per-task interface that the agentic pipeline
dispatches to for metric, loss, rendering, and capability tagging.

The agent layer (Agent 1-5, tournament, replan, memory) stays
task-agnostic. Task-specific logic lives in concrete subclasses
(SegmentationTask, ClassificationTask, ...) that implement this ABC.

Adding a new task type to the framework = adding one subclass +
registering it. No agent code changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn


class TaskAdapter(ABC):
    """Per-task interface. One instance per dataset / per pipeline run."""

    def __init__(self):
        # Optional source-model preprocessing (set by calibration probe).
        # When set, to_tensor() routes through it instead of the task's
        # legacy default. Lets one TaskAdapter handle any source model
        # convention discovered at runtime.
        self._source_pp_name = None  # canonical name string
        self._source_pp_fn = None    # (img, size, device) -> tensor
        self._source_pp_size = None  # detected input resolution

    def set_source_preprocessing(self, name, fn, size):
        """Install the calibration probe's verdict. Called once at the
        start of each pipeline run, before any tool / VLM call."""
        self._source_pp_name = name
        self._source_pp_fn = fn
        self._source_pp_size = int(size)

    def source_preprocessing_info(self):
        """Read-only view of the configured preprocessing."""
        return {
            "name": self._source_pp_name,
            "size": self._source_pp_size,
            "configured": self._source_pp_fn is not None,
        }

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable string id, e.g., 'segmentation', 'classification'."""

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Human-readable metric name, e.g., 'dice', 'accuracy'."""

    # --------------------------------------------------------------
    # Forward pass + prediction
    # --------------------------------------------------------------

    @abstractmethod
    def forward_pred(self, model: nn.Module, image_tensor: torch.Tensor
                       ) -> Tuple[torch.Tensor, np.ndarray]:
        """Run model on a 4D image tensor (B=1).

        Returns (logits, pred_index). For segmentation, pred is [H, W].
        For classification, pred is a Python int wrapped as 0-d array.
        """

    # --------------------------------------------------------------
    # Metric on a single labeled sample
    # --------------------------------------------------------------

    @abstractmethod
    def sample_metric(self, model: nn.Module, image: np.ndarray,
                        label: Any, num_classes: int, device: str,
                        size: int = 256) -> Optional[float]:
        """Compute the task metric on ONE labeled sample.

        Returns None if the sample doesn't contribute (e.g., no
        foreground present for seg). Caller averages.
        """

    def avg_metric(self, model: nn.Module, images: List[np.ndarray],
                     labels: List[Any], num_classes: int, device: str,
                     size: int = 256) -> float:
        """Average sample_metric across paired images+labels. Skips
        samples that return None."""
        scores = []
        for img, lbl in zip(images, labels):
            if lbl is None:
                continue
            s = self.sample_metric(model, img, lbl, num_classes, device, size)
            if s is not None:
                scores.append(s)
        return float(np.mean(scores)) if scores else 0.0

    # --------------------------------------------------------------
    # Cross-entropy on K labeled samples (used by *_ss semi-supervised
    # tool variants). Must use ctx.preprocess_fn so upstream
    # preprocessing also shapes the supervised input.
    # --------------------------------------------------------------

    @abstractmethod
    def ce_loss_on_labels(self, model: nn.Module,
                              images: List[np.ndarray],
                              labels: List[Any],
                              indices: List[int],
                              preprocess_fn,
                              num_classes: int,
                              device: str,
                              size: int = 256) -> torch.Tensor:
        """Mean CE on the K labeled samples. Returns 0 if no labeled
        samples are valid."""

    # --------------------------------------------------------------
    # Catastrophic-collapse / FG-ratio style sanity signal. For seg
    # this is fg_ratio; for cls it's ill-defined — return None to
    # skip the gate, OR return a confidence proxy.
    # --------------------------------------------------------------

    @abstractmethod
    def coverage_signal(self, model: nn.Module,
                            images: List[np.ndarray],
                            device: str,
                            max_samples: int = 200) -> Optional[float]:
        """Optional sanity signal in [0,1] — fg ratio for seg, None or
        mean-confidence for cls. The catastrophic-collapse gate uses
        this to detect "model predicts nothing" or "model predicts
        everywhere"; cls tasks may simply return None to skip."""

    # --------------------------------------------------------------
    # VLM rendering. Returns (images_b64, labels) — same contract as
    # build_multimodal_message expects.
    # --------------------------------------------------------------

    @abstractmethod
    def render_pair_for_vlm(self, model_source: nn.Module,
                                model_current: nn.Module,
                                anchor_imgs: List[np.ndarray],
                                num_classes: int,
                                device: str,
                                size: int = 256) -> Tuple[List[str], List[str]]:
        """For each anchor image, render a triplet: raw + source-pred-vis +
        current-pred-vis. Returns (b64_images, labels) flattened."""

    @abstractmethod
    def vlm_prompt_template(self) -> str:
        """Per-iter VLM prompt for this task. Must produce JSON with
        per_sample (better/similar/worse), per_sample_absolute
        (usable/broken), note."""

    # --------------------------------------------------------------
    # Capability tags this task offers (filtered into Agent 3's tool
    # menu). E.g., seg offers `pixel_predictions` so postprocessing
    # tools that operate on masks are available; cls does not.
    # --------------------------------------------------------------

    @abstractmethod
    def task_capabilities(self) -> List[str]:
        """List of capability tags, e.g., ['pixel_predictions'] for seg
        or ['scalar_predictions'] for classification."""

    @abstractmethod
    def to_tensor(self, image: np.ndarray, device: str,
                    size: int = 256) -> torch.Tensor:
        """Convert an HWC numpy image (uint8 or float [0,1]) to a 4D
        tensor [1,3,H,W] on the given device, applying any task-specific
        normalization. For seg this is just [0,1] scaling. For cls this
        applies the ImageNet mean/std normalization the source classifier
        was trained with."""


# --------------------------------------------------------------
# Registry — adding a new task is "register(name, cls)"
# --------------------------------------------------------------

class TaskRegistry:
    _registry: Dict[str, Type[TaskAdapter]] = {}

    @classmethod
    def register(cls, name: str, klass: Type[TaskAdapter]) -> None:
        cls._registry[name] = klass

    @classmethod
    def get(cls, name: str) -> Type[TaskAdapter]:
        if name not in cls._registry:
            raise KeyError(
                f"unknown task '{name}'. registered: "
                f"{sorted(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def make(cls, name: str) -> TaskAdapter:
        return cls.get(name)()

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._registry)
