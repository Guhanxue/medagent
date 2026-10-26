"""
Flexible data ingestion for MedAdaptAgent.

Supports loading data from:
  - Preprocessed .npz files (our format)
  - Raw NIfTI volumes (.nii, .nii.gz)
  - DICOM directories
  - PNG/JPEG image folders
  - NumPy arrays directly

No forced preprocessing format — adapts to whatever the user has.

Usage:
    from medadaptagent.utils.data_ingestion import load_medical_images

    # Raw NIfTI
    images, labels = load_medical_images("data/amos/imagesTr/",
                                          label_dir="data/amos/labelsTr/",
                                          modality="ct")

    # Preprocessed NPZ
    images, labels = load_medical_images("data/fundus/target/")

    # Single volume
    images, labels = load_medical_images("data/brain.nii.gz")
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_medical_images(
    data_path: str,
    label_dir: str = None,
    modality: str = "auto",
    image_size: int = 256,
    max_images: int = 200,
    slice_axis: int = 2,
    min_fg_ratio: float = 0.001,
    to_rgb: bool = True,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """Load medical images from any supported format.

    Args:
        data_path: Path to images. Can be:
            - Directory of .npz files (preprocessed)
            - Directory of .nii.gz files (raw NIfTI)
            - Directory of .png/.jpg files
            - Single .nii.gz file
            - Directory with images/ subdirectory
        label_dir: Optional path to labels. Auto-detected if None.
        modality: "ct", "mri", or "auto" for normalization.
        image_size: Resize target (H, W).
        max_images: Maximum number of 2D images to return.
        slice_axis: Axis for 3D volume slicing (0, 1, or 2).
        min_fg_ratio: Min foreground ratio for slice selection.
        to_rgb: Convert grayscale to 3-channel RGB.

    Returns:
        (images, labels): Lists of (H,W,3) float32 arrays and optional labels.
    """
    data_path = str(data_path)

    # Auto-detect format
    if os.path.isfile(data_path):
        if data_path.endswith(('.nii', '.nii.gz')):
            return _load_nifti_volume(data_path, label_dir, modality,
                                      image_size, max_images, slice_axis,
                                      min_fg_ratio, to_rgb)
        elif data_path.endswith('.npz'):
            return _load_single_npz(data_path, to_rgb)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Path not found: {data_path}")

    # Check for images/ subdirectory (our preprocessed format)
    img_subdir = os.path.join(data_path, "images")
    if os.path.isdir(img_subdir):
        lbl_subdir = os.path.join(data_path, "labels")
        return _load_npz_dir(img_subdir,
                             lbl_subdir if os.path.isdir(lbl_subdir) else None,
                             max_images, to_rgb)

    # Check for NIfTI files
    nii_files = sorted(glob.glob(os.path.join(data_path, "*.nii.gz")))
    nii_files += sorted(glob.glob(os.path.join(data_path, "*.nii")))
    if nii_files:
        return _load_nifti_dir(data_path, label_dir, modality,
                               image_size, max_images, slice_axis,
                               min_fg_ratio, to_rgb)

    # Check for NPZ files directly in directory
    npz_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
    if npz_files:
        return _load_npz_dir(data_path, label_dir, max_images, to_rgb)

    # Check for PNG/JPG
    img_files = sorted(glob.glob(os.path.join(data_path, "*.png")))
    img_files += sorted(glob.glob(os.path.join(data_path, "*.jpg")))
    if img_files:
        return _load_image_dir(data_path, label_dir, image_size, max_images, to_rgb)

    raise FileNotFoundError(
        f"No supported files found in {data_path}. "
        "Supported: .nii.gz, .npz, .png, .jpg"
    )


def _normalize(volume: np.ndarray, modality: str) -> np.ndarray:
    """Normalize a volume based on modality."""
    if modality == "ct":
        volume = np.clip(volume, -200, 400)
        volume = (volume - (-200)) / 600.0
    else:  # mri or auto
        fg = volume[volume > 0]
        if len(fg) > 0:
            p1, p99 = np.percentile(fg, [1, 99])
            volume = np.clip((volume - p1) / (p99 - p1 + 1e-8), 0, 1)
        else:
            volume = np.clip(volume, 0, 1)
    return volume.astype(np.float32)


def _detect_modality(volume: np.ndarray) -> str:
    """Auto-detect CT vs MRI from intensity distribution."""
    vals = volume[volume != 0]
    if len(vals) == 0:
        return "mri"
    if vals.min() < -100 or vals.max() > 1000:
        return "ct"
    return "mri"


def _resize_2d(img: np.ndarray, size: int, is_label: bool = False) -> np.ndarray:
    """Resize a 2D image/label."""
    from PIL import Image as PILImage
    if is_label:
        pil = PILImage.fromarray(img.astype(np.uint8))
        pil = pil.resize((size, size), PILImage.NEAREST)
        return np.array(pil)
    else:
        if img.ndim == 3:
            pil = PILImage.fromarray((img * 255).astype(np.uint8))
        else:
            pil = PILImage.fromarray((img * 255).astype(np.uint8))
        pil = pil.resize((size, size), PILImage.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is (H, W, 3)."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img


# ================================================================
# Format-specific loaders
# ================================================================

def _load_npz_dir(img_dir: str, lbl_dir: str = None,
                  max_images: int = 200, to_rgb: bool = True
                  ) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """Load preprocessed .npz files."""
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])
    files = files[:max_images]

    images, labels = [], []
    for f in files:
        data = np.load(os.path.join(img_dir, f))
        img = data["image"].astype(np.float32)
        if to_rgb:
            img = _to_rgb(img)
        images.append(img)

        if lbl_dir:
            lbl_path = os.path.join(lbl_dir, f)
            if os.path.exists(lbl_path):
                labels.append(np.load(lbl_path)["label"])
            else:
                labels.append(None)

    return images, labels if labels else None


def _load_single_npz(path: str, to_rgb: bool = True
                     ) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """Load a single .npz file."""
    data = np.load(path)
    img = data["image"].astype(np.float32)
    if to_rgb:
        img = _to_rgb(img)
    lbl = data.get("label")
    return [img], [lbl] if lbl is not None else None


def _load_nifti_volume(
    path: str, label_path: str = None, modality: str = "auto",
    image_size: int = 256, max_slices: int = 200, axis: int = 2,
    min_fg: float = 0.001, to_rgb: bool = True,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """Load a single NIfTI volume, extract 2D slices."""
    import nibabel as nib

    vol = nib.load(path).get_fdata().astype(np.float32)
    if modality == "auto":
        modality = _detect_modality(vol)
    vol = _normalize(vol, modality)

    lbl_vol = None
    if label_path and os.path.exists(label_path):
        lbl_vol = nib.load(label_path).get_fdata().astype(np.int32)

    images, labels = [], []
    n = vol.shape[axis]
    for i in range(n):
        if axis == 2:
            s = vol[:, :, i]
            l = lbl_vol[:, :, i] if lbl_vol is not None else None
        elif axis == 1:
            s = vol[:, i, :]
            l = lbl_vol[:, i, :] if lbl_vol is not None else None
        else:
            s = vol[i, :, :]
            l = lbl_vol[i, :, :] if lbl_vol is not None else None

        # Filter empty slices
        if l is not None:
            if (l > 0).sum() / l.size < min_fg:
                continue
        elif s.mean() < 0.02:
            continue

        s = _resize_2d(s, image_size)
        if to_rgb:
            s = _to_rgb(s)
        images.append(s)

        if l is not None:
            labels.append(_resize_2d(l, image_size, is_label=True))

        if len(images) >= max_slices:
            break

    return images, labels if labels else None


def _load_nifti_dir(
    data_dir: str, label_dir: str = None, modality: str = "auto",
    image_size: int = 256, max_images: int = 200, axis: int = 2,
    min_fg: float = 0.001, to_rgb: bool = True,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """Load all NIfTI volumes from a directory, extract 2D slices."""
    nii_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    nii_files += sorted(glob.glob(os.path.join(data_dir, "*.nii")))

    # Try to find label directory
    if label_dir is None:
        for candidate in ["labels", "labelsTr", "masks", "annotations"]:
            cand_dir = os.path.join(os.path.dirname(data_dir), candidate)
            if os.path.isdir(cand_dir):
                label_dir = cand_dir
                break

    all_images, all_labels = [], []
    for nii_path in nii_files:
        if len(all_images) >= max_images:
            break

        # Find matching label
        lbl_path = None
        if label_dir:
            name = os.path.basename(nii_path)
            lbl_path = os.path.join(label_dir, name)
            if not os.path.exists(lbl_path):
                # Try common naming conventions
                for suffix in ["_seg", "_label", "_Label", "_mask"]:
                    alt = name.replace(".nii", f"{suffix}.nii")
                    alt_path = os.path.join(label_dir, alt)
                    if os.path.exists(alt_path):
                        lbl_path = alt_path
                        break
                else:
                    lbl_path = None

        remaining = max_images - len(all_images)
        imgs, lbls = _load_nifti_volume(
            nii_path, lbl_path, modality, image_size,
            remaining, axis, min_fg, to_rgb,
        )
        all_images.extend(imgs)
        if lbls:
            all_labels.extend(lbls)

    return all_images, all_labels if all_labels else None


def _load_image_dir(
    data_dir: str, label_dir: str = None,
    image_size: int = 256, max_images: int = 200, to_rgb: bool = True,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """Load PNG/JPG images from a directory."""
    from PIL import Image as PILImage

    files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    files += sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    files = files[:max_images]

    images, labels = [], []
    for f in files:
        pil = PILImage.open(f).resize((image_size, image_size), PILImage.BILINEAR)
        img = np.array(pil).astype(np.float32) / 255.0
        if to_rgb:
            img = _to_rgb(img)
        images.append(img)

        if label_dir:
            name = os.path.splitext(os.path.basename(f))[0]
            for ext in [".png", ".npy", ".npz"]:
                lbl_path = os.path.join(label_dir, name + ext)
                if os.path.exists(lbl_path):
                    if ext == ".npy":
                        labels.append(np.load(lbl_path))
                    elif ext == ".npz":
                        labels.append(np.load(lbl_path)["label"])
                    else:
                        lbl = np.array(PILImage.open(lbl_path).resize(
                            (image_size, image_size), PILImage.NEAREST))
                        labels.append(lbl)
                    break

    return images, labels if labels else None
