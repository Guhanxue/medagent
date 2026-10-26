"""
Data-side adaptation tools for bridging domain gaps.

Instead of (only) adjusting the model to the data, these tools adjust
the data to better match what the model expects. The agent can use
either approach or both in combination.

Tools:
1. Histogram Matching: Match target intensity distribution to source
2. Auto Scale Detection + Correction: Detect and fix FOV/scale mismatch
3. Intensity Normalization: Standardize intensity across domains
4. Stain/Color Normalization: For histopathology or color-sensitive tasks
5. Resolution Harmonization: Match spatial resolution between domains
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import ndimage


# ============================================================
# 1. Histogram Matching
# ============================================================

def histogram_match_channel(source: np.ndarray,
                             reference: np.ndarray,
                             bins: int = 256) -> np.ndarray:
    """Match histogram of source to reference for a single channel.

    Args:
        source: (H, W) float32 [0, 1] source channel.
        reference: (H, W) float32 [0, 1] reference channel.
        bins: Number of histogram bins.

    Returns:
        matched: (H, W) float32 [0, 1] histogram-matched source.
    """
    src_vals = (source * (bins - 1)).astype(np.int32).ravel()
    ref_vals = (reference * (bins - 1)).astype(np.int32).ravel()

    # Compute CDFs
    src_hist, _ = np.histogram(src_vals, bins=bins, range=(0, bins - 1))
    ref_hist, _ = np.histogram(ref_vals, bins=bins, range=(0, bins - 1))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # Build mapping: for each source bin, find closest reference bin by CDF
    mapping = np.zeros(bins, dtype=np.int32)
    for s in range(bins):
        idx = np.argmin(np.abs(src_cdf[s] - ref_cdf))
        mapping[s] = idx

    matched = mapping[src_vals].reshape(source.shape)
    return (matched / (bins - 1)).astype(np.float32)


def histogram_match(target_img: np.ndarray,
                     source_stats: Dict,
                     per_channel: bool = True) -> np.ndarray:
    """Match target image histogram to source distribution.

    Args:
        target_img: (H, W, C) float32 [0,1] target image.
        source_stats: Dict with 'reference_image' (H,W,C) or
            'mean_histogram' per channel.
        per_channel: Match each channel independently.

    Returns:
        matched: (H, W, C) float32 [0,1] matched image.
    """
    ref = source_stats.get('reference_image')
    if ref is None:
        return target_img

    matched = np.zeros_like(target_img)
    if per_channel:
        for c in range(target_img.shape[2]):
            matched[:, :, c] = histogram_match_channel(
                target_img[:, :, c], ref[:, :, c])
    else:
        # Match on luminance only
        tgt_gray = np.mean(target_img, axis=2)
        ref_gray = np.mean(ref, axis=2)
        scale = histogram_match_channel(tgt_gray, ref_gray)
        ratio = np.where(tgt_gray > 1e-6, scale / (tgt_gray + 1e-6), 1.0)
        for c in range(target_img.shape[2]):
            matched[:, :, c] = np.clip(target_img[:, :, c] * ratio, 0, 1)

    return matched.astype(np.float32)


def histogram_match_batch(target_images: List[np.ndarray],
                           source_images: List[np.ndarray],
                           n_references: int = 10) -> List[np.ndarray]:
    """Match target image histograms to source distribution.

    Uses a random subset of source images as reference pool, and computes
    a "mean reference" by averaging several source images.

    Args:
        target_images: List of target images.
        source_images: List of source images (reference pool).
        n_references: Number of source images to average for reference.

    Returns:
        List of histogram-matched target images.
    """
    # Build reference from random source images
    indices = np.random.choice(len(source_images),
                                min(n_references, len(source_images)),
                                replace=False)
    ref = np.mean([source_images[i] for i in indices], axis=0)
    stats = {'reference_image': ref.astype(np.float32)}

    return [histogram_match(img, stats) for img in target_images]


# ============================================================
# 2. Auto Scale Detection + Correction
# ============================================================

def detect_foreground_scale(images: List[np.ndarray],
                             labels: List[np.ndarray] = None,
                             model=None,
                             image_size: int = 256,
                             device: str = 'cpu') -> Dict:
    """Detect the scale of foreground objects in a set of images.

    Uses either ground truth labels (if available) or model predictions
    to estimate the typical foreground area fraction and bounding box size.

    Args:
        images: List of images (H, W, C) float32 [0,1].
        labels: Optional ground truth labels.
        model: Optional model for prediction-based detection.
        image_size: Model input size.
        device: Torch device.

    Returns:
        Dict with 'mean_fg_fraction', 'mean_bbox_size', 'fg_center',
        'fg_fractions' per image.
    """
    import torch
    from PIL import Image as PILImage

    fg_fractions = []
    bbox_sizes = []
    centers = []

    for i, img in enumerate(images):
        if labels is not None and i < len(labels):
            mask = labels[i] > 0
        elif model is not None:
            # Use model predictions
            pil = PILImage.fromarray((img * 255).astype(np.uint8))
            pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
            tensor = torch.from_numpy(
                np.array(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                pred = model(tensor.to(device)).argmax(dim=1).cpu().numpy()[0]
            mask = pred > 0
        else:
            # Fallback: simple Otsu-like thresholding
            gray = np.mean(img, axis=2)
            thresh = np.mean(gray) + 0.5 * np.std(gray)
            mask = gray > thresh

        fg_frac = mask.sum() / mask.size
        fg_fractions.append(fg_frac)

        if mask.sum() > 0:
            ys, xs = np.where(mask)
            h = ys.max() - ys.min()
            w = xs.max() - xs.min()
            bbox_sizes.append((h / mask.shape[0], w / mask.shape[1]))
            centers.append((ys.mean() / mask.shape[0], xs.mean() / mask.shape[1]))
        else:
            bbox_sizes.append((0, 0))
            centers.append((0.5, 0.5))

    return {
        'mean_fg_fraction': float(np.mean(fg_fractions)),
        'std_fg_fraction': float(np.std(fg_fractions)),
        'fg_fractions': fg_fractions,
        'mean_bbox_size': (float(np.mean([b[0] for b in bbox_sizes])),
                           float(np.mean([b[1] for b in bbox_sizes]))),
        'mean_center': (float(np.mean([c[0] for c in centers])),
                        float(np.mean([c[1] for c in centers]))),
    }


def compute_scale_mismatch(source_scale: Dict, target_scale: Dict) -> Dict:
    """Compute scale mismatch between source and target.

    Returns:
        Dict with 'scale_ratio', 'needs_correction', 'recommended_action'.
    """
    src_fg = source_scale['mean_fg_fraction']
    tgt_fg = target_scale['mean_fg_fraction']

    if src_fg < 1e-6 or tgt_fg < 1e-6:
        return {
            'scale_ratio': float('inf'),
            'needs_correction': True,
            'recommended_action': 'Cannot estimate scale — model may be failing',
        }

    ratio = tgt_fg / src_fg

    if ratio > 3.0:
        action = f'Target FG is {ratio:.1f}x larger than source. ' \
                 f'Recommend: zoom/crop source images to match target scale, ' \
                 f'or resize target to match source FOV.'
    elif ratio < 0.33:
        action = f'Target FG is {ratio:.1f}x smaller than source. ' \
                 f'Recommend: crop target around ROI to match source scale.'
    else:
        action = f'Scale ratio {ratio:.1f}x is within acceptable range.'

    return {
        'scale_ratio': float(ratio),
        'needs_correction': ratio > 3.0 or ratio < 0.33,
        'recommended_action': action,
        'source_fg_fraction': float(src_fg),
        'target_fg_fraction': float(tgt_fg),
    }


def auto_crop_to_foreground(image: np.ndarray,
                              mask: np.ndarray = None,
                              padding_factor: float = 2.0,
                              target_size: Tuple[int, int] = (512, 512)
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Automatically crop image around foreground region.

    If no mask is provided, uses intensity-based foreground detection.

    Args:
        image: (H, W, C) float32 [0,1].
        mask: (H, W) binary foreground mask. Optional.
        padding_factor: Crop size = padding_factor * foreground bbox.
        target_size: (H, W) output size after crop.

    Returns:
        cropped_image: (target_H, target_W, C) float32.
        crop_bbox: (y1, y2, x1, x2) the crop coordinates used.
    """
    from PIL import Image as PILImage

    h, w = image.shape[:2]

    if mask is None:
        # Auto-detect foreground by intensity
        gray = np.mean(image, axis=2)
        # Remove black borders (common in fundus/retinal images)
        border_mask = gray > 0.05
        if border_mask.sum() < 100:
            border_mask = np.ones_like(gray, dtype=bool)
        mask = border_mask

    ys, xs = np.where(mask)
    if len(ys) == 0:
        # No foreground — return center crop
        s = min(h, w)
        cy, cx = h // 2, w // 2
        y1, y2 = cy - s // 2, cy + s // 2
        x1, x2 = cx - s // 2, cx + s // 2
    else:
        cy, cx = ys.mean(), xs.mean()
        fg_h = ys.max() - ys.min()
        fg_w = xs.max() - xs.min()
        size = max(fg_h, fg_w) * padding_factor
        half = size / 2

        y1 = max(0, int(cy - half))
        y2 = min(h, int(cy + half))
        x1 = max(0, int(cx - half))
        x2 = min(w, int(cx + half))

        # Make square
        crop_h, crop_w = y2 - y1, x2 - x1
        if crop_h > crop_w:
            diff = crop_h - crop_w
            x1 = max(0, x1 - diff // 2)
            x2 = min(w, x1 + crop_h)
            x1 = max(0, x2 - crop_h)
        elif crop_w > crop_h:
            diff = crop_w - crop_h
            y1 = max(0, y1 - diff // 2)
            y2 = min(h, y1 + crop_w)
            y1 = max(0, y2 - crop_w)

    cropped = image[y1:y2, x1:x2]
    pil = PILImage.fromarray((cropped * 255).astype(np.uint8))
    pil = pil.resize((target_size[1], target_size[0]), PILImage.BILINEAR)
    result = np.array(pil).astype(np.float32) / 255.0

    return result, np.array([y1, y2, x1, x2])


def auto_rescale_batch(images: List[np.ndarray],
                        source_scale: Dict,
                        target_scale: Dict,
                        labels: List[np.ndarray] = None
                        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Automatically rescale images to match source foreground scale.

    If target FG is much larger than source, zoom out (pad + resize).
    If target FG is much smaller, zoom in (crop around FG + resize).

    Args:
        images: Target images to rescale.
        source_scale: Scale info from source (from detect_foreground_scale).
        target_scale: Scale info from target.
        labels: Optional labels to transform alongside images.

    Returns:
        rescaled_images, rescaled_labels (or None).
    """
    from PIL import Image as PILImage

    ratio = target_scale['mean_fg_fraction'] / max(source_scale['mean_fg_fraction'], 1e-6)

    if 0.5 <= ratio <= 2.0:
        # Scale is close enough — no correction needed
        return images, labels

    rescaled_imgs = []
    rescaled_lbls = [] if labels is not None else None
    h, w = images[0].shape[:2]

    if ratio > 2.0:
        # Target FG is larger — zoom out by padding to simulate wider FOV
        # Pad the image to make foreground smaller relative to image
        pad = int(max(h, w) * (ratio - 1) / 2)
        for i, img in enumerate(images):
            padded = np.zeros((h + 2 * pad, w + 2 * pad, img.shape[2]),
                               dtype=np.float32)
            padded[pad:pad + h, pad:pad + w] = img
            # Resize back to original size
            pil = PILImage.fromarray((padded * 255).astype(np.uint8))
            pil = pil.resize((w, h), PILImage.BILINEAR)
            rescaled_imgs.append(np.array(pil).astype(np.float32) / 255.0)

            if labels is not None and i < len(labels):
                lbl_padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
                lbl_padded[pad:pad + h, pad:pad + w] = labels[i]
                lbl_pil = PILImage.fromarray(lbl_padded)
                lbl_pil = lbl_pil.resize((w, h), PILImage.NEAREST)
                rescaled_lbls.append(np.array(lbl_pil))

    else:
        # Target FG is smaller — crop around foreground
        for i, img in enumerate(images):
            lbl = labels[i] if labels is not None and i < len(labels) else None
            fg_mask = lbl > 0 if lbl is not None else None
            cropped, bbox = auto_crop_to_foreground(
                img, mask=fg_mask, padding_factor=2.0, target_size=(h, w))
            rescaled_imgs.append(cropped)

            if lbl is not None:
                y1, y2, x1, x2 = bbox
                lbl_crop = lbl[y1:y2, x1:x2]
                lbl_pil = PILImage.fromarray(lbl_crop)
                lbl_pil = lbl_pil.resize((w, h), PILImage.NEAREST)
                rescaled_lbls.append(np.array(lbl_pil))

    return rescaled_imgs, rescaled_lbls


# ============================================================
# 3. Intensity Normalization
# ============================================================

def zscore_normalize(image: np.ndarray,
                      mean: np.ndarray = None,
                      std: np.ndarray = None) -> np.ndarray:
    """Z-score normalize image per channel.

    Args:
        image: (H, W, C) float32.
        mean: Per-channel mean. If None, computed from image.
        std: Per-channel std. If None, computed from image.

    Returns:
        normalized: (H, W, C) float32.
    """
    if mean is None:
        mean = image.mean(axis=(0, 1))
    if std is None:
        std = image.std(axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)
    return ((image - mean) / std).astype(np.float32)


def percentile_normalize(image: np.ndarray,
                           lower: float = 1.0,
                           upper: float = 99.0) -> np.ndarray:
    """Percentile-based intensity normalization.

    Clips to [lower_pct, upper_pct] range and scales to [0, 1].
    Robust to outlier intensities (common in MRI).

    Args:
        image: (H, W, C) or (H, W) float32.
        lower: Lower percentile for clipping.
        upper: Upper percentile for clipping.

    Returns:
        normalized: same shape, float32 [0, 1].
    """
    lo = np.percentile(image, lower)
    hi = np.percentile(image, upper)
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    clipped = np.clip(image, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def compute_domain_intensity_stats(images: List[np.ndarray]) -> Dict:
    """Compute intensity statistics for a set of images.

    Returns:
        Dict with per-channel mean, std, percentiles.
    """
    all_means = []
    all_stds = []
    for img in images:
        all_means.append(img.mean(axis=(0, 1)))
        all_stds.append(img.std(axis=(0, 1)))

    means = np.mean(all_means, axis=0)
    stds = np.mean(all_stds, axis=0)

    # Sample pixels for percentile computation
    n_sample = min(len(images), 50)
    sample_pixels = np.concatenate([
        images[i].reshape(-1, images[i].shape[-1]) if images[i].ndim == 3
        else images[i].ravel()[:, None]
        for i in range(n_sample)
    ], axis=0)

    percentiles = {}
    for p in [1, 5, 25, 50, 75, 95, 99]:
        percentiles[p] = np.percentile(sample_pixels, p, axis=0).tolist()

    return {
        'mean': means.tolist(),
        'std': stds.tolist(),
        'percentiles': percentiles,
    }


def match_intensity_stats(target_images: List[np.ndarray],
                            source_stats: Dict,
                            target_stats: Dict) -> List[np.ndarray]:
    """Match target intensity statistics to source.

    Simple linear transform: rescale target mean/std to match source.

    Args:
        target_images: List of target images.
        source_stats: Source intensity stats (from compute_domain_intensity_stats).
        target_stats: Target intensity stats.

    Returns:
        List of intensity-matched target images.
    """
    src_mean = np.array(source_stats['mean'])
    src_std = np.array(source_stats['std'])
    tgt_mean = np.array(target_stats['mean'])
    tgt_std = np.array(target_stats['std'])

    # Avoid division by zero
    tgt_std = np.where(tgt_std < 1e-6, 1.0, tgt_std)

    matched = []
    for img in target_images:
        normalized = (img - tgt_mean) / tgt_std * src_std + src_mean
        matched.append(np.clip(normalized, 0, 1).astype(np.float32))

    return matched


# ============================================================
# 4. Resolution Harmonization
# ============================================================

def estimate_effective_resolution(image: np.ndarray) -> float:
    """Estimate effective spatial resolution by measuring high-frequency content.

    Higher values = more detail = higher effective resolution.

    Args:
        image: (H, W, C) or (H, W) float32.

    Returns:
        resolution_score: float (higher = more detail).
    """
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Laplacian variance as sharpness proxy
    laplacian = ndimage.laplace(gray)
    return float(np.var(laplacian))


def match_resolution(target_img: np.ndarray,
                      source_resolution: float,
                      target_resolution: float,
                      max_sigma: float = 3.0) -> np.ndarray:
    """Match effective resolution by blurring or sharpening.

    If target is sharper than source (higher resolution), blur it.
    If target is blurrier, mild sharpening is applied.

    Args:
        target_img: (H, W, C) float32 [0,1].
        source_resolution: Source resolution score.
        target_resolution: Target resolution score.
        max_sigma: Maximum blur sigma.

    Returns:
        matched: (H, W, C) float32 [0,1].
    """
    ratio = target_resolution / max(source_resolution, 1e-8)

    if ratio > 1.5:
        # Target is sharper — blur to match source
        sigma = min(max_sigma, (ratio - 1.0) * 1.5)
        matched = np.zeros_like(target_img)
        for c in range(target_img.shape[2]):
            matched[:, :, c] = ndimage.gaussian_filter(target_img[:, :, c], sigma)
        return matched.astype(np.float32)
    elif ratio < 0.67:
        # Target is blurrier — mild unsharp masking
        blurred = np.zeros_like(target_img)
        for c in range(target_img.shape[2]):
            blurred[:, :, c] = ndimage.gaussian_filter(target_img[:, :, c], 1.0)
        sharpened = target_img + 0.5 * (target_img - blurred)
        return np.clip(sharpened, 0, 1).astype(np.float32)
    else:
        return target_img


# ============================================================
# 5. Combined Data Adaptation Pipeline
# ============================================================

class DataAdapter:
    """Orchestrates data-side adaptation based on shift analysis.

    The agent calls analyze() to detect what data transformations are needed,
    then adapt() to apply them. This can be used standalone or in combination
    with model-side adaptation.
    """

    def __init__(self):
        self.analysis = {}
        self.transformations_applied = []

    def analyze(self,
                source_images: List[np.ndarray],
                target_images: List[np.ndarray],
                source_labels: List[np.ndarray] = None,
                model=None,
                device: str = 'cpu',
                image_size: int = 256) -> Dict:
        """Analyze what data-side adaptations are needed.

        Returns:
            Dict with recommended transformations and their parameters.
        """
        recommendations = []

        # 1. Intensity analysis
        src_stats = compute_domain_intensity_stats(source_images)
        tgt_stats = compute_domain_intensity_stats(target_images)

        mean_diff = np.abs(
            np.array(src_stats['mean']) - np.array(tgt_stats['mean'])
        ).mean()
        std_ratio = np.mean(
            np.array(tgt_stats['std']) / np.maximum(np.array(src_stats['std']), 1e-6)
        )

        if mean_diff > 0.1 or abs(std_ratio - 1.0) > 0.3:
            recommendations.append({
                'tool': 'histogram_matching',
                'priority': 2,
                'reason': f'Intensity gap: mean_diff={mean_diff:.3f}, '
                          f'std_ratio={std_ratio:.2f}',
            })

        # 2. Scale analysis
        src_scale = detect_foreground_scale(
            source_images[:50], source_labels[:50] if source_labels else None,
            model, image_size, device)
        tgt_scale = detect_foreground_scale(
            target_images[:50], None, model, image_size, device)
        scale_info = compute_scale_mismatch(src_scale, tgt_scale)

        if scale_info['needs_correction']:
            recommendations.append({
                'tool': 'auto_rescale',
                'priority': 1,  # highest priority — fix scale first
                'reason': scale_info['recommended_action'],
                'scale_ratio': scale_info['scale_ratio'],
            })

        # 3. Resolution analysis
        src_res = np.mean([estimate_effective_resolution(img)
                           for img in source_images[:20]])
        tgt_res = np.mean([estimate_effective_resolution(img)
                           for img in target_images[:20]])
        res_ratio = tgt_res / max(src_res, 1e-8)

        if res_ratio > 1.5 or res_ratio < 0.67:
            recommendations.append({
                'tool': 'resolution_matching',
                'priority': 3,
                'reason': f'Resolution mismatch: ratio={res_ratio:.2f}',
            })

        # Sort by priority (lowest number = highest priority)
        recommendations.sort(key=lambda x: x['priority'])

        self.analysis = {
            'source_intensity': src_stats,
            'target_intensity': tgt_stats,
            'source_scale': src_scale,
            'target_scale': tgt_scale,
            'scale_mismatch': scale_info,
            'source_resolution': float(src_res),
            'target_resolution': float(tgt_res),
            'recommendations': recommendations,
        }
        return self.analysis

    def adapt(self,
              target_images: List[np.ndarray],
              source_images: List[np.ndarray],
              target_labels: List[np.ndarray] = None,
              apply_tools: List[str] = None) -> Tuple[List[np.ndarray],
                                                        Optional[List[np.ndarray]]]:
        """Apply recommended data transformations to target images.

        Args:
            target_images: Images to transform.
            source_images: Source images (reference for matching).
            target_labels: Optional labels to transform alongside.
            apply_tools: Specific tools to apply. If None, applies all
                recommended tools from analyze().

        Returns:
            adapted_images, adapted_labels
        """
        if not self.analysis:
            raise RuntimeError("Call analyze() before adapt()")

        if apply_tools is None:
            apply_tools = [r['tool'] for r in self.analysis['recommendations']]

        adapted = list(target_images)
        labels = list(target_labels) if target_labels is not None else None
        self.transformations_applied = []

        for tool_name in apply_tools:
            if tool_name == 'auto_rescale':
                adapted, labels = auto_rescale_batch(
                    adapted,
                    self.analysis['source_scale'],
                    self.analysis['target_scale'],
                    labels,
                )
                self.transformations_applied.append('auto_rescale')

            elif tool_name == 'histogram_matching':
                adapted = histogram_match_batch(adapted, source_images)
                self.transformations_applied.append('histogram_matching')

            elif tool_name == 'intensity_matching':
                adapted = match_intensity_stats(
                    adapted,
                    self.analysis['source_intensity'],
                    self.analysis['target_intensity'],
                )
                self.transformations_applied.append('intensity_matching')

            elif tool_name == 'resolution_matching':
                src_res = self.analysis['source_resolution']
                tgt_res = self.analysis['target_resolution']
                adapted = [match_resolution(img, src_res, tgt_res)
                           for img in adapted]
                self.transformations_applied.append('resolution_matching')

        return adapted, labels

    def get_summary(self) -> str:
        """Get human-readable summary of analysis and adaptations."""
        if not self.analysis:
            return "No analysis performed yet."

        lines = ["Data Adaptation Analysis:"]
        sm = self.analysis.get('scale_mismatch', {})
        if sm:
            lines.append(f"  Scale ratio: {sm.get('scale_ratio', '?'):.1f}x "
                         f"(source FG={sm.get('source_fg_fraction', 0)*100:.1f}%, "
                         f"target FG={sm.get('target_fg_fraction', 0)*100:.1f}%)")

        lines.append(f"  Resolution: source={self.analysis.get('source_resolution', 0):.4f}, "
                     f"target={self.analysis.get('target_resolution', 0):.4f}")

        recs = self.analysis.get('recommendations', [])
        if recs:
            lines.append(f"  Recommendations ({len(recs)}):")
            for r in recs:
                lines.append(f"    [{r['priority']}] {r['tool']}: {r['reason']}")
        else:
            lines.append("  No data-side corrections needed.")

        if self.transformations_applied:
            lines.append(f"  Applied: {', '.join(self.transformations_applied)}")

        return '\n'.join(lines)
