"""
Unified data loader for MMWHS, CrossMoDA, and Fundus datasets.
Handles 2D/3D loading, normalization, and slice extraction.
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class MedicalImageDataset(Dataset):
    """Base dataset for loading preprocessed 2D slices (as .npz files)."""

    def __init__(self, data_dir: str, split: str = "target",
                 transform=None, with_labels: bool = False):
        """
        Args:
            data_dir: Path to preprocessed dataset directory.
            split: 'source' or 'target'.
            transform: Optional transforms.
            with_labels: If True, also load ground-truth labels.
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.with_labels = with_labels

        self.image_files = sorted(glob.glob(str(self.data_dir / "images" / "*.npz")))
        if not self.image_files:
            self.image_files = sorted(glob.glob(str(self.data_dir / "images" / "*.npy")))
        if not self.image_files:
            self.image_files = sorted(glob.glob(str(self.data_dir / "images" / "*.png")))

        self.label_dir = self.data_dir / "labels"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        ext = os.path.splitext(img_path)[1]

        if ext == ".npz":
            data = np.load(img_path)
            image = data["image"].astype(np.float32)
        elif ext == ".npy":
            image = np.load(img_path).astype(np.float32)
        elif ext == ".png":
            from PIL import Image
            image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported format: {ext}")

        sample = {"image": image, "path": img_path, "idx": idx}

        if self.with_labels and self.label_dir.exists():
            label_name = os.path.basename(img_path)
            label_path = self.label_dir / label_name
            if label_path.exists():
                if ext == ".npz":
                    sample["label"] = np.load(str(label_path))["label"].astype(np.int64)
                elif ext == ".npy":
                    sample["label"] = np.load(str(label_path)).astype(np.int64)
                elif ext == ".png":
                    from PIL import Image
                    sample["label"] = np.array(Image.open(str(label_path))).astype(np.int64)

        if self.transform:
            sample = self.transform(sample)

        return sample


class MMWHSDataset(MedicalImageDataset):
    """MMWHS cardiac CT→MRI dataset.

    4 structures: LV myocardium, LV blood cavity, RV blood cavity, left atrium.
    Source: CT, Target: MRI.
    """
    NUM_CLASSES = 4
    CLASS_NAMES = [
        "LV_myocardium",
        "LV_blood_cavity",
        "RV_blood_cavity",
        "left_atrium",
    ]
    LABEL_MAP = {205: 1, 500: 2, 600: 3, 420: 4}  # Original → sequential

    def __init__(self, data_dir: str, split: str = "target", **kwargs):
        super().__init__(os.path.join(data_dir, "mmwhs"), split, **kwargs)


class CrossMoDADataset(MedicalImageDataset):
    """CrossMoDA vestibular schwannoma dataset.

    2 structures: VS tumor, cochlea.
    Source: ceT1, Target: hrT2.
    """
    NUM_CLASSES = 2
    CLASS_NAMES = ["VS_tumor", "cochlea"]

    def __init__(self, data_dir: str, split: str = "target", **kwargs):
        super().__init__(os.path.join(data_dir, "crossmoda"), split, **kwargs)


class FundusDataset(MedicalImageDataset):
    """Fundus dataset: REFUGE (source) → RIM-ONE (target).

    2 structures: optic disc, optic cup.
    Already 2D RGB images.
    """
    NUM_CLASSES = 2
    CLASS_NAMES = ["optic_disc", "optic_cup"]

    def __init__(self, data_dir: str, split: str = "target", **kwargs):
        super().__init__(os.path.join(data_dir, "fundus"), split, **kwargs)


DATASET_REGISTRY = {
    "mmwhs": MMWHSDataset,
    "crossmoda": CrossMoDADataset,
    "fundus": FundusDataset,
}


def get_dataset(name: str, data_dir: str, split: str = "target", **kwargs) -> MedicalImageDataset:
    """Get dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](data_dir, split=split, **kwargs)


def get_dataloader(name: str, data_dir: str, split: str = "target",
                   batch_size: int = 4, num_workers: int = 4,
                   shuffle: bool = False, **kwargs):
    """Get DataLoader by dataset name."""
    dataset = get_dataset(name, data_dir, split, **kwargs)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, pin_memory=True,
    )


# =============================================
# Preprocessing utilities
# =============================================

def normalize_volume(volume: np.ndarray, clip_range: Tuple[float, float] = None) -> np.ndarray:
    """Normalize a 3D volume to [0, 1]."""
    if clip_range:
        volume = np.clip(volume, clip_range[0], clip_range[1])
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)
    return volume.astype(np.float32)


def extract_2d_slices(volume: np.ndarray, label: np.ndarray = None,
                      axis: int = 2, skip_empty: bool = True,
                      min_foreground_ratio: float = 0.001):
    """Extract 2D slices from a 3D volume along the given axis.

    Args:
        volume: 3D volume (D, H, W).
        label: Optional 3D label map.
        axis: Slicing axis (0=axial for z-first, 2=axial for x,y,z ordering).
        skip_empty: Skip slices with no foreground in label.
        min_foreground_ratio: Minimum fraction of foreground pixels to keep slice.

    Returns:
        List of (image_slice, label_slice) tuples.
    """
    slices = []
    n_slices = volume.shape[axis]

    for i in range(n_slices):
        if axis == 0:
            img_slice = volume[i]
            lbl_slice = label[i] if label is not None else None
        elif axis == 1:
            img_slice = volume[:, i]
            lbl_slice = label[:, i] if label is not None else None
        else:
            img_slice = volume[:, :, i]
            lbl_slice = label[:, :, i] if label is not None else None

        if skip_empty and lbl_slice is not None:
            fg_ratio = (lbl_slice > 0).sum() / lbl_slice.size
            if fg_ratio < min_foreground_ratio:
                continue

        slices.append((img_slice, lbl_slice))

    return slices


def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                 is_label: bool = False) -> np.ndarray:
    """Resize image or label to target size."""
    from PIL import Image as PILImage
    if is_label:
        pil_img = PILImage.fromarray(image.astype(np.uint8))
        pil_img = pil_img.resize(target_size, PILImage.NEAREST)
    else:
        if image.ndim == 2:
            pil_img = PILImage.fromarray((image * 255).astype(np.uint8))
        else:
            pil_img = PILImage.fromarray((image * 255).astype(np.uint8))
        pil_img = pil_img.resize(target_size, PILImage.BILINEAR)
    return np.array(pil_img).astype(np.float32) / 255.0 if not is_label else np.array(pil_img)


def prepare_for_sam2(image: np.ndarray, target_size: int = 1024) -> np.ndarray:
    """Prepare a 2D image for SAM2 input.

    SAM2 expects (3, H, W) float32 tensor, typically 1024x1024.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    h, w = image.shape[:2]
    if h != target_size or w != target_size:
        image = resize_image(image, (target_size, target_size))

    if image.max() <= 1.0:
        image = image  # already [0, 1]
    else:
        image = image / 255.0

    # HWC → CHW
    image = image.transpose(2, 0, 1)
    return image.astype(np.float32)
