"""
Visualization utilities for MedAdaptAgent.
Generates figures for paper and debugging.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn


def plot_segmentation_comparison(image: np.ndarray, predictions: Dict[str, np.ndarray],
                                  ground_truth: np.ndarray = None,
                                  title: str = "", save_path: str = None):
    """Plot image with multiple segmentation overlays for comparison.

    Args:
        image: (H, W) or (H, W, 3) input image.
        predictions: Dict of method_name → mask.
        ground_truth: Optional ground truth mask.
        title: Figure title.
        save_path: Path to save figure.
    """
    n_cols = len(predictions) + (2 if ground_truth is not None else 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    if n_cols == 1:
        axes = [axes]

    # Original image
    if image.ndim == 2:
        axes[0].imshow(image, cmap="gray")
    else:
        axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Ground truth
    col = 1
    if ground_truth is not None:
        axes[col].imshow(image if image.ndim == 3 else np.stack([image]*3, -1))
        axes[col].imshow(ground_truth, alpha=0.4, cmap="jet")
        axes[col].set_title("Ground Truth")
        axes[col].axis("off")
        col += 1

    # Predictions
    for name, pred in predictions.items():
        axes[col].imshow(image if image.ndim == 3 else np.stack([image]*3, -1))
        axes[col].imshow(pred, alpha=0.4, cmap="jet")
        axes[col].set_title(name)
        axes[col].axis("off")
        col += 1

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.close()
    return fig


def plot_adaptation_trajectory(q_scores: List[float], dice_scores: List[float] = None,
                                adapter_selections: List[List[str]] = None,
                                save_path: str = None):
    """Plot Q(x) score trajectory across adaptation iterations.

    Args:
        q_scores: Q-score per iteration.
        dice_scores: Optional Dice per iteration (if labels available).
        adapter_selections: List of active adapters per iteration.
        save_path: Path to save figure.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    iterations = range(1, len(q_scores) + 1)
    ax1.plot(iterations, q_scores, "b-o", label="Q(x) Score", linewidth=2)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Q(x) Score", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")

    if dice_scores:
        ax2 = ax1.twinx()
        ax2.plot(iterations, dice_scores, "r-s", label="Dice", linewidth=2)
        ax2.set_ylabel("Dice Score", color="r", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="r")

    if adapter_selections:
        for i, adapters in enumerate(adapter_selections):
            ax1.annotate(
                ", ".join(a[:3] for a in adapters),
                xy=(i + 1, q_scores[i]),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=7,
                ha="center",
                rotation=30,
            )

    ax1.set_title("Adaptation Trajectory", fontsize=14)
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.95))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_failure_distribution(type_distributions: List[Dict[str, float]],
                               iteration_labels: List[str] = None,
                               save_path: str = None):
    """Plot failure type distribution across iterations as stacked bar chart."""
    n = len(type_distributions)
    if iteration_labels is None:
        iteration_labels = [f"Iter {i+1}" for i in range(n)]

    types = ["miss", "hallucination", "boundary_error", "class_confusion"]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n)
    width = 0.6
    bottom = np.zeros(n)

    for ftype, color in zip(types, colors):
        values = [d.get(ftype, 0) for d in type_distributions]
        ax.bar(x, values, width, bottom=bottom, label=ftype.replace("_", " ").title(),
               color=color)
        bottom += np.array(values)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Proportion")
    ax.set_title("Failure Type Distribution Across Iterations")
    ax.set_xticks(x)
    ax.set_xticklabels(iteration_labels)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_adapter_selection_heatmap(adapter_selections: List[List[str]],
                                    dataset_names: List[str] = None,
                                    save_path: str = None):
    """Plot adapter selection heatmap across iterations and datasets."""
    all_adapters = ["style", "boundary", "scale", "semantic"]

    if dataset_names is None:
        dataset_names = [f"Iter {i+1}" for i in range(len(adapter_selections))]

    matrix = np.zeros((len(adapter_selections), len(all_adapters)))
    for i, selected in enumerate(adapter_selections):
        for j, adapter in enumerate(all_adapters):
            matrix[i, j] = 1 if adapter in selected else 0

    fig, ax = plt.subplots(figsize=(6, max(3, len(adapter_selections) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(all_adapters)))
    ax.set_xticklabels([a.capitalize() for a in all_adapters])
    ax.set_yticks(range(len(dataset_names)))
    ax.set_yticklabels(dataset_names)
    ax.set_title("Adapter Selection Pattern")

    # Add text annotations
    for i in range(len(adapter_selections)):
        for j in range(len(all_adapters)):
            ax.text(j, i, "ON" if matrix[i, j] else "",
                    ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_entropy_map(image: np.ndarray, entropy_map: np.ndarray,
                     prediction: np.ndarray = None,
                     title: str = "Entropy Map", save_path: str = None):
    """Plot entropy map overlay on image."""
    n_cols = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Image
    if image.ndim == 2:
        axes[0].imshow(image, cmap="gray")
    else:
        axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Entropy map
    im = axes[1].imshow(entropy_map, cmap="hot")
    axes[1].set_title("Entropy")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    if prediction is not None:
        axes[2].imshow(image if image.ndim == 3 else np.stack([image]*3, -1))
        axes[2].imshow(prediction, alpha=0.4, cmap="jet")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


# ================================================================
# Prediction saving for qualitative figures
# ================================================================

def save_predictions(
    model: nn.Module,
    images: List[np.ndarray],
    labels: List[np.ndarray],
    output_dir: str,
    device: str = "cuda",
    image_size: int = 256,
    tag: str = "adapted",
    max_samples: int = None,
):
    """Run model on images and save predictions + GT as npz files.

    Saves to output_dir/{tag}_predictions/:
      - {idx:04d}.npz with keys: image, prediction, ground_truth, dice_per_pixel

    Args:
        model: Segmentation model.
        images: List of input images (H, W, 3) or (H, W).
        labels: List of ground truth label maps.
        output_dir: Directory to save results.
        device: Torch device.
        image_size: Input size for model.
        tag: Prefix for output subfolder (e.g., "source_only", "adapted").
        max_samples: Limit number of saved samples.
    """
    from PIL import Image as PILImage

    save_dir = os.path.join(output_dir, f"{tag}_predictions")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    n = len(images) if max_samples is None else min(max_samples, len(images))

    with torch.no_grad():
        for idx in range(n):
            img = images[idx]
            lbl = labels[idx] if idx < len(labels) else None

            # Prepare tensor
            if img.ndim == 2:
                tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
            elif img.ndim == 3 and img.shape[2] == 3:
                tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            else:
                tensor = torch.from_numpy(img.astype(np.float32))
            tensor = tensor.unsqueeze(0).to(device)

            pred = model(tensor).argmax(dim=1).cpu().numpy()[0]

            save_data = {"image": img, "prediction": pred}
            if lbl is not None:
                lbl_pil = PILImage.fromarray(lbl.astype(np.uint8))
                lbl_r = np.array(lbl_pil.resize((image_size, image_size), PILImage.NEAREST))
                save_data["ground_truth"] = lbl_r

            np.savez_compressed(os.path.join(save_dir, f"{idx:04d}.npz"), **save_data)

    print(f"  Saved {n} predictions to {save_dir}")
    return save_dir


def generate_qualitative_grid(
    images: List[np.ndarray],
    predictions_dict: Dict[str, List[np.ndarray]],
    ground_truths: List[np.ndarray] = None,
    sample_indices: List[int] = None,
    save_path: str = None,
    title: str = "",
    num_classes: int = None,
):
    """Generate a grid of qualitative results for paper figures.

    Rows = samples, Columns = [Input, GT, Method1, Method2, ...]

    Args:
        images: Input images.
        predictions_dict: {method_name: list of prediction masks}.
        ground_truths: Optional GT masks.
        sample_indices: Which samples to show (default: first 4).
        save_path: Path to save figure.
        title: Figure title.
        num_classes: Number of classes for colormap normalization.
    """
    if sample_indices is None:
        sample_indices = list(range(min(4, len(images))))

    methods = list(predictions_dict.keys())
    n_rows = len(sample_indices)
    n_cols = 1 + (1 if ground_truths else 0) + len(methods)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    vmax = num_classes if num_classes else None

    for row, idx in enumerate(sample_indices):
        col = 0
        img = images[idx]

        # Input
        if img.ndim == 2:
            axes[row, col].imshow(img, cmap="gray")
        else:
            img_show = img
            if img.max() <= 1.0:
                img_show = (img * 255).astype(np.uint8)
            axes[row, col].imshow(img_show)
        if row == 0:
            axes[row, col].set_title("Input", fontsize=10)
        axes[row, col].axis("off")
        col += 1

        # GT
        if ground_truths is not None:
            bg = np.stack([img] * 3, -1) if img.ndim == 2 else img.copy()
            if bg.max() <= 1.0:
                bg = (bg * 255).astype(np.uint8)
            axes[row, col].imshow(bg)
            axes[row, col].imshow(ground_truths[idx], alpha=0.5, cmap="jet",
                                   vmin=0, vmax=vmax)
            if row == 0:
                axes[row, col].set_title("Ground Truth", fontsize=10)
            axes[row, col].axis("off")
            col += 1

        # Predictions
        for m_idx, method in enumerate(methods):
            bg = np.stack([img] * 3, -1) if img.ndim == 2 else img.copy()
            if bg.max() <= 1.0:
                bg = (bg * 255).astype(np.uint8)
            axes[row, col].imshow(bg)
            axes[row, col].imshow(predictions_dict[method][idx], alpha=0.5,
                                   cmap="jet", vmin=0, vmax=vmax)
            if row == 0:
                axes[row, col].set_title(method, fontsize=10)
            axes[row, col].axis("off")
            col += 1

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved qualitative grid to {save_path}")
        plt.close()
    return fig
