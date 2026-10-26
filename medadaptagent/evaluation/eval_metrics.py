"""
Evaluation metrics for medical image segmentation.
Dice, HD95, ASD for both 2D and 3D segmentation tasks.
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum())


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray, voxel_spacing=None) -> float:
    """Compute 95th percentile Hausdorff distance."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf

    if voxel_spacing is None:
        voxel_spacing = np.ones(pred.ndim)
    voxel_spacing = np.array(voxel_spacing)

    pred_border = pred ^ binary_erosion(pred)
    gt_border = gt ^ binary_erosion(gt)

    pred_coords = np.argwhere(pred_border) * voxel_spacing
    gt_coords = np.argwhere(gt_border) * voxel_spacing

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.inf

    dt_gt = distance_transform_edt(~gt, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred, sampling=voxel_spacing)

    dist_pred_to_gt = dt_gt[pred_border]
    dist_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return np.percentile(all_distances, 95)


def average_surface_distance(pred: np.ndarray, gt: np.ndarray, voxel_spacing=None) -> float:
    """Compute Average Surface Distance (ASD)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf

    if voxel_spacing is None:
        voxel_spacing = np.ones(pred.ndim)

    pred_border = pred ^ binary_erosion(pred)
    gt_border = gt ^ binary_erosion(gt)

    dt_gt = distance_transform_edt(~gt, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred, sampling=voxel_spacing)

    dist_pred_to_gt = dt_gt[pred_border]
    dist_gt_to_pred = dt_pred[gt_border]

    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return np.inf

    return (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2.0


def compute_metrics_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int,
                               voxel_spacing=None, class_names=None):
    """Compute per-class Dice, HD95, ASD.

    Args:
        pred: Integer label map (H, W) or (D, H, W).
        gt: Integer label map (same shape).
        num_classes: Number of foreground classes (labels 1..num_classes).
        voxel_spacing: Physical spacing per axis.
        class_names: Optional list of class names.

    Returns:
        Dict with per-class and mean metrics.
    """
    results = {}
    dice_scores = []
    hd95_scores = []
    asd_scores = []

    for c in range(1, num_classes + 1):
        pred_c = (pred == c)
        gt_c = (gt == c)
        d = dice_coefficient(pred_c, gt_c)
        h = hausdorff_distance_95(pred_c, gt_c, voxel_spacing)
        a = average_surface_distance(pred_c, gt_c, voxel_spacing)

        name = class_names[c - 1] if class_names else f"class_{c}"
        results[name] = {"dice": d, "hd95": h, "asd": a}
        dice_scores.append(d)
        if np.isfinite(h):
            hd95_scores.append(h)
        if np.isfinite(a):
            asd_scores.append(a)

    results["mean"] = {
        "dice": np.mean(dice_scores) if dice_scores else 0.0,
        "hd95": np.mean(hd95_scores) if hd95_scores else np.inf,
        "asd": np.mean(asd_scores) if asd_scores else np.inf,
    }
    return results


def compute_metrics_2d_batch(preds: list, gts: list, num_classes: int, class_names=None):
    """Compute metrics over a batch of 2D slices (e.g., fundus images)."""
    all_results = []
    for pred, gt in zip(preds, gts):
        r = compute_metrics_per_class(pred, gt, num_classes, class_names=class_names)
        all_results.append(r)

    aggregated = {}
    if not all_results:
        return aggregated

    keys = all_results[0].keys()
    for key in keys:
        dice_list = [r[key]["dice"] for r in all_results]
        hd95_list = [r[key]["hd95"] for r in all_results if np.isfinite(r[key]["hd95"])]
        asd_list = [r[key]["asd"] for r in all_results if np.isfinite(r[key]["asd"])]
        aggregated[key] = {
            "dice_mean": np.mean(dice_list),
            "dice_std": np.std(dice_list),
            "hd95_mean": np.mean(hd95_list) if hd95_list else np.inf,
            "hd95_std": np.std(hd95_list) if hd95_list else 0.0,
            "asd_mean": np.mean(asd_list) if asd_list else np.inf,
            "asd_std": np.std(asd_list) if asd_list else 0.0,
        }
    return aggregated


def compute_metrics_3d_volume(pred_slices: list, gt_slices: list, num_classes: int,
                               voxel_spacing=None, class_names=None):
    """Stack 2D slices into 3D volume and compute volumetric metrics."""
    pred_vol = np.stack(pred_slices, axis=0)
    gt_vol = np.stack(gt_slices, axis=0)
    return compute_metrics_per_class(pred_vol, gt_vol, num_classes,
                                      voxel_spacing=voxel_spacing,
                                      class_names=class_names)
