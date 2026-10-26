"""Task abstraction package.

Adding a new task type to MedAdaptAgent = subclass TaskAdapter,
implement the methods, and register. Agents (1-5) and the tournament
machinery dispatch through this interface, so they stay task-agnostic.

Usage:
    from medadaptagent.tasks import get_task
    task = get_task("segmentation")  # or "classification"
    dice = task.avg_metric(model, images, labels, num_classes, device)
    rendered = task.render_pair_for_vlm(...)
"""
from .base import TaskAdapter, TaskRegistry
from .segmentation import SegmentationTask  # registers on import
from .classification import ClassificationTask
from .segmentation_3d import SegmentationTask3D  # registers on import
from .classification_3d import Classification3DTask  # registers on import


def get_task(name: str) -> TaskAdapter:
    """Make a fresh task adapter by name. Default: 'segmentation'."""
    return TaskRegistry.make(name)


def available_tasks():
    return TaskRegistry.names()


__all__ = ["TaskAdapter", "TaskRegistry", "SegmentationTask",
            "ClassificationTask", "SegmentationTask3D",
            "Classification3DTask", "get_task", "available_tasks"]
