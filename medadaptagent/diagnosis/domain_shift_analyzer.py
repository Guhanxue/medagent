"""
Domain Shift Analyzer: Statistical verification of domain shift.

Step 2 of the MedAdaptAgent pipeline:
  1. RAG says "X is different between source and target"
  2. This module VERIFIES it with actual data statistics
  3. Produces a structured shift report for the adaptation strategy

Computes:
  - Intensity distribution shift (per-channel histograms, KL divergence)
  - Frequency spectrum shift (FFT amplitude comparison)
  - Scale/size shift (foreground area statistics)
  - Feature-space distance (BiomedCLIP embeddings)
  - Model failure pattern analysis (entropy, confidence, prediction distribution)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ShiftReport:
    """Structured report of verified domain shift characteristics."""
    # Intensity shift
    intensity_shift: float = 0.0  # KL divergence of intensity distributions
    channel_shifts: Dict[str, float] = field(default_factory=dict)  # per-channel mean diff

    # Frequency shift
    low_freq_shift: float = 0.0   # style/appearance shift
    high_freq_shift: float = 0.0  # texture/edge shift

    # Scale shift
    scale_ratio: float = 1.0  # target_foreground_area / source_foreground_area

    # Feature shift
    feature_distance: float = 0.0  # cosine distance in BiomedCLIP space

    # Model failure analysis
    failure_type: str = ""  # "miss", "hallucination", "boundary", "confusion"
    mean_confidence: float = 0.0
    mean_entropy: float = 0.0
    prediction_distribution: Dict[str, float] = field(default_factory=dict)

    # Verified shift types (from statistical tests)
    verified_shifts: List[str] = field(default_factory=list)

    # Raw statistics for LLM reasoning
    source_stats: Dict = field(default_factory=dict)
    target_stats: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["Domain Shift Analysis Report:"]
        lines.append(f"  Intensity shift (KL): {self.intensity_shift:.4f}")
        for ch, diff in self.channel_shifts.items():
            lines.append(f"    {ch}: {diff:+.4f}")
        lines.append(f"  Low-freq shift (style): {self.low_freq_shift:.4f}")
        lines.append(f"  High-freq shift (texture): {self.high_freq_shift:.4f}")
        lines.append(f"  Scale ratio: {self.scale_ratio:.2f}")
        lines.append(f"  Feature distance: {self.feature_distance:.4f}")
        lines.append(f"  Failure type: {self.failure_type}")
        lines.append(f"  Mean confidence: {self.mean_confidence:.4f}")
        lines.append(f"  Mean entropy: {self.mean_entropy:.4f}")
        lines.append(f"  Verified shifts: {self.verified_shifts}")
        return "\n".join(lines)


class DomainShiftAnalyzer:
    """Analyze and verify domain shift between source and target distributions."""

    def __init__(self, device="cuda"):
        self.device = device

    def compute_image_statistics(self, images: List[np.ndarray]) -> Dict:
        """Compute comprehensive statistics for a set of images.

        Args:
            images: List of (H, W, 3) float32 [0,1] images.
        """
        all_means = []
        all_stds = []
        channel_means = {"R": [], "G": [], "B": []}
        histograms = {"R": np.zeros(256), "G": np.zeros(256), "B": np.zeros(256)}

        for img in images:
            all_means.append(img.mean())
            all_stds.append(img.std())
            # Handle grayscale (H,W) or single-channel images
            if img.ndim == 2:
                for name in ["R", "G", "B"]:
                    channel_means[name].append(img.mean())
                    hist, _ = np.histogram(
                        (img * 255).astype(np.uint8), bins=256, range=(0, 255)
                    )
                    histograms[name] += hist
            else:
                for c, name in enumerate(["R", "G", "B"]):
                    channel_means[name].append(img[:, :, c].mean())
                    hist, _ = np.histogram(
                        (img[:, :, c] * 255).astype(np.uint8), bins=256, range=(0, 255)
                    )
                    histograms[name] += hist

        # Normalize histograms
        for name in histograms:
            histograms[name] = histograms[name] / (histograms[name].sum() + 1e-8)

        return {
            "mean_intensity": float(np.mean(all_means)),
            "std_intensity": float(np.mean(all_stds)),
            "channel_means": {k: float(np.mean(v)) for k, v in channel_means.items()},
            "channel_stds": {k: float(np.std(v)) for k, v in channel_means.items()},
            "histograms": histograms,
        }

    def compute_frequency_statistics(self, images: List[np.ndarray]) -> Dict:
        """Analyze frequency spectrum of images."""
        low_freq_energy = []
        high_freq_energy = []

        for img in images:
            # Convert to grayscale for frequency analysis
            gray = np.mean(img, axis=-1) if img.ndim == 3 else img
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            h, w = gray.shape
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 8  # low frequency = center

            # Low frequency (center of spectrum) = style/appearance
            y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
            low_mask = (x * x + y * y) <= radius * radius
            low_freq_energy.append(float(magnitude[low_mask].mean()))

            # High frequency (periphery) = texture/edges
            high_mask = ~low_mask
            high_freq_energy.append(float(magnitude[high_mask].mean()))

        return {
            "low_freq_mean": float(np.mean(low_freq_energy)),
            "high_freq_mean": float(np.mean(high_freq_energy)),
        }

    def compute_foreground_statistics(self, labels: List[np.ndarray],
                                       class_names: List[str] = None,
                                       num_classes: int = None) -> Dict:
        """Analyze foreground region sizes.

        Args:
            labels: List of (H, W) label maps.
            class_names: List of class names (index 0 = background).
            num_classes: Number of classes. Inferred from labels if not given.
        """
        if num_classes is None:
            num_classes = max(int(lbl.max()) for lbl in labels) + 1
        if class_names is None:
            class_names = ["background"] + [f"class_{i}" for i in range(1, num_classes)]

        fg_areas = []
        per_class_areas = {name: [] for name in class_names[1:]}

        for lbl in labels:
            total = lbl.size
            fg_areas.append((lbl > 0).sum() / total)
            for idx, name in enumerate(class_names[1:], start=1):
                per_class_areas[name].append((lbl == idx).sum() / total)

        result = {
            "fg_area_mean": float(np.mean(fg_areas)),
            "fg_area_std": float(np.std(fg_areas)),
        }
        for name in class_names[1:]:
            result[f"{name}_area_mean"] = float(np.mean(per_class_areas[name]))
        return result

    def analyze_model_failures(self, model: torch.nn.Module,
                                target_images: List[np.ndarray],
                                target_labels: List[np.ndarray] = None,
                                image_size: int = 256,
                                num_classes: int = None,
                                class_names: List[str] = None) -> Dict:
        """Analyze how the source model fails on target data.

        Args:
            model: Source model (eval mode, on device).
            target_images: List of (H,W,3) float32 images.
            target_labels: Optional GT labels (for failure classification).
            image_size: Model input size.
            num_classes: Number of classes. If None, inferred from model output.
            class_names: List of class names (index 0 = background).
        """
        from PIL import Image as PILImage

        model.eval()
        all_entropy = []
        all_confidence = []
        total_pixels = 0
        failure_types = {"miss": 0, "hallucination": 0, "boundary": 0, "confusion": 0}

        # Infer num_classes from first forward pass if not provided
        class_counts_initialized = False
        class_counts = {}

        with torch.no_grad():
            for i, img in enumerate(target_images):
                # Resize and prepare
                pil = PILImage.fromarray((img * 255).astype(np.uint8))
                pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
                arr = np.array(pil).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    tensor = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1).unsqueeze(0).to(self.device)
                else:
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)

                # Initialize class_counts on first pass
                if not class_counts_initialized:
                    nc = num_classes or logits.shape[1]
                    if class_names is None:
                        class_names = ["background"] + [f"class_{i}" for i in range(1, nc)]
                    class_counts = {name: 0 for name in class_names}
                    class_counts_initialized = True

                # Confidence and entropy
                max_prob = probs.max(dim=1).values
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)

                all_confidence.append(float(max_prob.mean()))
                all_entropy.append(float(entropy.mean()))

                # Class distribution
                pred_np = pred.cpu().numpy()[0]
                n_pixels = pred_np.size
                total_pixels += n_pixels
                for idx, name in enumerate(class_names):
                    class_counts[name] += int((pred_np == idx).sum())

                # Failure classification (if GT available)
                if target_labels is not None:
                    lbl = target_labels[i]
                    if lbl.shape != pred_np.shape:
                        lbl_pil = PILImage.fromarray(lbl)
                        lbl_pil = lbl_pil.resize(
                            (image_size, image_size), PILImage.NEAREST
                        )
                        lbl = np.array(lbl_pil)

                    gt_fg = (lbl > 0)
                    pred_fg = (pred_np > 0)

                    if gt_fg.any() and not pred_fg.any():
                        failure_types["miss"] += 1
                    elif not gt_fg.any() and pred_fg.any():
                        failure_types["hallucination"] += 1
                    elif gt_fg.any() and pred_fg.any():
                        intersection = (gt_fg & pred_fg).sum()
                        union = (gt_fg | pred_fg).sum()
                        iou = intersection / max(union, 1)
                        if iou > 0.3:
                            failure_types["boundary"] += 1
                        else:
                            failure_types["confusion"] += 1

        pred_dist = {k: v / max(total_pixels, 1) for k, v in class_counts.items()}

        # Determine primary failure type: check if all foreground classes are near zero
        fg_total = sum(v for k, v in pred_dist.items() if k != "background")
        if fg_total < 0.01:
            primary_failure = "miss"
        elif failure_types:
            primary_failure = max(failure_types, key=failure_types.get)
        else:
            primary_failure = "unknown"

        return {
            "mean_confidence": float(np.mean(all_confidence)),
            "mean_entropy": float(np.mean(all_entropy)),
            "prediction_distribution": pred_dist,
            "failure_types": failure_types,
            "primary_failure": primary_failure,
        }

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence between two distributions."""
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def analyze(self, source_images: List[np.ndarray],
                target_images: List[np.ndarray],
                source_labels: List[np.ndarray] = None,
                target_labels: List[np.ndarray] = None,
                model: torch.nn.Module = None,
                image_size: int = 256,
                num_classes: int = None,
                class_names: List[str] = None) -> ShiftReport:
        """Full domain shift analysis.

        Args:
            source_images: List of source domain images.
            target_images: List of target domain images.
            source_labels: Optional source labels (for scale analysis).
            target_labels: Optional target labels (for failure classification).
            model: Optional source model (for failure analysis).
            image_size: Model input size.
            num_classes: Number of segmentation classes.
            class_names: List of class names (index 0 = background).

        Returns:
            ShiftReport with verified shift characteristics.
        """
        report = ShiftReport()

        # 1. Intensity statistics
        print("  Analyzing intensity distributions...")
        src_stats = self.compute_image_statistics(source_images)
        tgt_stats = self.compute_image_statistics(target_images)
        report.source_stats = src_stats
        report.target_stats = tgt_stats

        # KL divergence per channel
        kl_total = 0
        for ch in ["R", "G", "B"]:
            kl = self.kl_divergence(src_stats["histograms"][ch],
                                     tgt_stats["histograms"][ch])
            kl_total += kl
            diff = tgt_stats["channel_means"][ch] - src_stats["channel_means"][ch]
            report.channel_shifts[ch] = diff

        report.intensity_shift = kl_total / 3

        # Verify intensity shift
        if report.intensity_shift > 0.1:
            report.verified_shifts.append("intensity_shift")

        # 2. Frequency analysis
        print("  Analyzing frequency spectra...")
        src_freq = self.compute_frequency_statistics(source_images)
        tgt_freq = self.compute_frequency_statistics(target_images)

        report.low_freq_shift = abs(
            tgt_freq["low_freq_mean"] - src_freq["low_freq_mean"]
        ) / max(src_freq["low_freq_mean"], 1e-8)
        report.high_freq_shift = abs(
            tgt_freq["high_freq_mean"] - src_freq["high_freq_mean"]
        ) / max(src_freq["high_freq_mean"], 1e-8)

        if report.low_freq_shift > 0.2:
            report.verified_shifts.append("style_shift")
        if report.high_freq_shift > 0.2:
            report.verified_shifts.append("texture_shift")

        # 3. Scale/size analysis
        if source_labels is not None and target_labels is not None:
            print("  Analyzing foreground scale...")
            src_fg = self.compute_foreground_statistics(
                source_labels, class_names=class_names, num_classes=num_classes)
            tgt_fg = self.compute_foreground_statistics(
                target_labels, class_names=class_names, num_classes=num_classes)
            report.source_stats["foreground"] = src_fg
            report.target_stats["foreground"] = tgt_fg

            if src_fg["fg_area_mean"] > 0:
                report.scale_ratio = tgt_fg["fg_area_mean"] / src_fg["fg_area_mean"]
                if abs(report.scale_ratio - 1.0) > 0.5:
                    report.verified_shifts.append("scale_shift")

        # 4. Model failure analysis
        if model is not None:
            print("  Analyzing model failures on target...")
            failure_info = self.analyze_model_failures(
                model, target_images, target_labels, image_size,
                num_classes=num_classes, class_names=class_names,
            )
            report.mean_confidence = failure_info["mean_confidence"]
            report.mean_entropy = failure_info["mean_entropy"]
            report.prediction_distribution = failure_info["prediction_distribution"]
            report.failure_type = failure_info["primary_failure"]

            if report.failure_type == "miss":
                report.verified_shifts.append("complete_failure")

        return report
