"""
Failure Classifier: Classify detected failures into Type A/B/C/D.
Maps failure types to recommended adapters.

Type A (Miss): Model completely misses a structure
Type B (Hallucination): False positive predictions
Type C (Boundary Error): Correct detection but poor boundaries
Type D (Class Confusion): Misclassification between similar structures
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from scipy.ndimage import binary_erosion, binary_dilation, label as scipy_label


@dataclass
class FailureRegion:
    """A detected failure region."""
    failure_type: str  # "miss", "hallucination", "boundary_error", "class_confusion"
    location: tuple  # (y_center, x_center)
    severity: float  # 0-1 scale
    area_fraction: float  # fraction of image
    details: str = ""


@dataclass
class FailureReport:
    """Structured failure report for a set of images."""
    regions: List[FailureRegion] = field(default_factory=list)
    type_distribution: Dict[str, float] = field(default_factory=dict)
    overall_severity: float = 0.0
    recommended_adapters: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "type_distribution": self.type_distribution,
            "overall_severity": self.overall_severity,
            "recommended_adapters": self.recommended_adapters,
            "num_regions": len(self.regions),
            "summary": self.summary,
            "regions": [
                {
                    "type": r.failure_type,
                    "location": r.location,
                    "severity": r.severity,
                    "area_fraction": r.area_fraction,
                    "details": r.details,
                }
                for r in self.regions
            ],
        }


class FailureClassifier:
    """Classify failure regions into taxonomy types.

    Uses heuristic rules based on entropy, confidence, and prediction patterns.
    The LLM diagnostician can override or refine these classifications.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.entropy_threshold = self.config.get("entropy_threshold", 0.5)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)
        self.boundary_entropy_threshold = self.config.get("boundary_entropy_threshold", 0.7)
        self.min_region_area = self.config.get("min_region_area", 100)  # pixels

    def classify(self, prediction: np.ndarray,
                 entropy_map: np.ndarray,
                 confidence_map: np.ndarray,
                 tta_variance: np.ndarray,
                 expected_regions: List[Dict] = None) -> FailureReport:
        """Classify failures in a single prediction.

        Args:
            prediction: Binary or multi-class prediction mask.
            entropy_map: Pixel-wise entropy.
            confidence_map: Pixel-wise confidence.
            tta_variance: TTA variance map.
            expected_regions: Optional expected anatomy (from domain profile).

        Returns:
            FailureReport with classified regions.
        """
        regions = []

        # Type A: Miss — expected region with near-zero prediction
        miss_regions = self._detect_misses(prediction, entropy_map, expected_regions)
        regions.extend(miss_regions)

        # Type B: Hallucination — prediction in unexpected location
        halluc_regions = self._detect_hallucinations(
            prediction, confidence_map, expected_regions
        )
        regions.extend(halluc_regions)

        # Type C: Boundary error — high entropy at prediction boundaries
        boundary_regions = self._detect_boundary_errors(
            prediction, entropy_map, tta_variance
        )
        regions.extend(boundary_regions)

        # Type D: Class confusion — high entropy in predicted regions
        confusion_regions = self._detect_class_confusion(
            prediction, entropy_map, confidence_map
        )
        regions.extend(confusion_regions)

        # Compute type distribution
        type_counts = {"miss": 0, "hallucination": 0, "boundary_error": 0, "class_confusion": 0}
        for r in regions:
            type_counts[r.failure_type] += r.severity

        total = sum(type_counts.values())
        if total > 0:
            type_dist = {k: v / total for k, v in type_counts.items()}
        else:
            type_dist = {k: 0.0 for k in type_counts}

        # Map to adapters
        adapter_map = {
            "miss": ["semantic"],
            "hallucination": ["style"],
            "boundary_error": ["boundary", "style"],
            "class_confusion": ["semantic", "scale"],
        }
        recommended = set()
        for ftype, proportion in type_dist.items():
            if proportion > 0.15:
                for adapter in adapter_map.get(ftype, []):
                    recommended.add(adapter)

        report = FailureReport(
            regions=regions,
            type_distribution=type_dist,
            overall_severity=float(np.mean([r.severity for r in regions])) if regions else 0.0,
            recommended_adapters=list(recommended),
            summary=self._generate_summary(type_dist, regions),
        )
        return report

    def _detect_misses(self, prediction, entropy_map,
                       expected_regions=None) -> List[FailureRegion]:
        """Detect Type A: regions where model produced no prediction but should have."""
        regions = []
        if expected_regions is None:
            # Use entropy to find high-uncertainty areas with no prediction
            no_pred = prediction == 0
            high_entropy = entropy_map > self.entropy_threshold
            miss_mask = no_pred & high_entropy

            labeled, n_components = scipy_label(miss_mask)
            for comp in range(1, n_components + 1):
                component = labeled == comp
                area = component.sum()
                if area < self.min_region_area:
                    continue
                ys, xs = np.where(component)
                regions.append(FailureRegion(
                    failure_type="miss",
                    location=(int(ys.mean()), int(xs.mean())),
                    severity=float(entropy_map[component].mean()),
                    area_fraction=area / prediction.size,
                    details=f"High-entropy region with no prediction, area={area}px",
                ))
        else:
            for expected in expected_regions:
                y, x, r = expected.get("y", 0), expected.get("x", 0), expected.get("radius", 50)
                mask = np.zeros_like(prediction, dtype=bool)
                yy, xx = np.ogrid[:prediction.shape[0], :prediction.shape[1]]
                mask[(yy - y) ** 2 + (xx - x) ** 2 <= r ** 2] = True
                pred_in_region = prediction[mask].sum()
                if pred_in_region < 0.1 * mask.sum():
                    regions.append(FailureRegion(
                        failure_type="miss",
                        location=(y, x),
                        severity=0.8,
                        area_fraction=mask.sum() / prediction.size,
                        details=f"Expected region at ({y},{x}) has <10% prediction coverage",
                    ))
        return regions

    def _detect_hallucinations(self, prediction, confidence_map,
                                expected_regions=None) -> List[FailureRegion]:
        """Detect Type B: predictions in unexpected locations."""
        regions = []
        if prediction.sum() == 0:
            return regions

        pred_mask = prediction > 0
        # Look for high-confidence but spatially isolated predictions
        labeled, n_components = scipy_label(pred_mask)
        component_sizes = []
        for comp in range(1, n_components + 1):
            component_sizes.append((labeled == comp).sum())

        if not component_sizes:
            return regions

        median_size = np.median(component_sizes)

        for comp in range(1, n_components + 1):
            component = labeled == comp
            size = component.sum()
            # Small isolated components are likely hallucinations
            if size < 0.2 * median_size and size > 10:
                ys, xs = np.where(component)
                regions.append(FailureRegion(
                    failure_type="hallucination",
                    location=(int(ys.mean()), int(xs.mean())),
                    severity=float(confidence_map[component].mean()),
                    area_fraction=size / prediction.size,
                    details=f"Small isolated prediction ({size}px vs median {median_size:.0f}px)",
                ))

        return regions

    def _detect_boundary_errors(self, prediction, entropy_map,
                                 tta_variance) -> List[FailureRegion]:
        """Detect Type C: high uncertainty at prediction boundaries."""
        regions = []
        pred_mask = prediction > 0
        if pred_mask.sum() == 0:
            return regions

        # Extract boundary pixels
        boundary = pred_mask ^ binary_erosion(pred_mask)
        if boundary.sum() < 5:
            return regions

        boundary_entropy = entropy_map[boundary].mean()
        boundary_tta_var = tta_variance[boundary].mean()

        if boundary_entropy > self.boundary_entropy_threshold or boundary_tta_var > 0.2:
            # Find worst boundary segments
            high_entropy_boundary = boundary & (entropy_map > self.boundary_entropy_threshold)
            if high_entropy_boundary.sum() > 0:
                ys, xs = np.where(high_entropy_boundary)
                regions.append(FailureRegion(
                    failure_type="boundary_error",
                    location=(int(ys.mean()), int(xs.mean())),
                    severity=float(boundary_entropy),
                    area_fraction=float(high_entropy_boundary.sum()) / prediction.size,
                    details=f"High boundary entropy={boundary_entropy:.3f}, TTA_var={boundary_tta_var:.3f}",
                ))

        return regions

    def _detect_class_confusion(self, prediction, entropy_map,
                                 confidence_map) -> List[FailureRegion]:
        """Detect Type D: class confusion in predicted regions."""
        regions = []
        pred_mask = prediction > 0
        if pred_mask.sum() == 0:
            return regions

        # Interior pixels (not boundary) with high entropy = class confusion
        interior = binary_erosion(pred_mask, iterations=2)
        if interior.sum() < self.min_region_area:
            return regions

        interior_entropy = entropy_map[interior]
        high_entropy_interior = interior & (entropy_map > self.entropy_threshold)

        if high_entropy_interior.sum() > 0.1 * interior.sum():
            ys, xs = np.where(high_entropy_interior)
            regions.append(FailureRegion(
                failure_type="class_confusion",
                location=(int(ys.mean()), int(xs.mean())),
                severity=float(interior_entropy.mean()),
                area_fraction=float(high_entropy_interior.sum()) / prediction.size,
                details=f"Interior confusion: {high_entropy_interior.sum()}/{interior.sum()} pixels high-entropy",
            ))

        return regions

    def _generate_summary(self, type_dist: Dict[str, float],
                          regions: List[FailureRegion]) -> str:
        """Generate human-readable failure summary."""
        if not regions:
            return "No significant failures detected."

        dominant = max(type_dist, key=type_dist.get)
        lines = [f"Detected {len(regions)} failure regions."]
        lines.append(f"Dominant failure: {dominant} ({type_dist[dominant]:.0%})")
        for ftype, prop in sorted(type_dist.items(), key=lambda x: -x[1]):
            if prop > 0.05:
                lines.append(f"  {ftype}: {prop:.0%}")
        return " ".join(lines)


def classify_batch(predictions: List[np.ndarray],
                   entropy_maps: List[np.ndarray],
                   confidence_maps: List[np.ndarray],
                   tta_variances: List[np.ndarray],
                   config: dict = None) -> FailureReport:
    """Classify failures across a batch and aggregate."""
    classifier = FailureClassifier(config)
    all_regions = []
    for pred, ent, conf, tta in zip(predictions, entropy_maps, confidence_maps, tta_variances):
        report = classifier.classify(pred, ent, conf, tta)
        all_regions.extend(report.regions)

    # Aggregate type distribution
    type_counts = {"miss": 0, "hallucination": 0, "boundary_error": 0, "class_confusion": 0}
    for r in all_regions:
        type_counts[r.failure_type] += r.severity
    total = sum(type_counts.values())
    type_dist = {k: v / total if total > 0 else 0 for k, v in type_counts.items()}

    recommended = set()
    adapter_map = {
        "miss": ["semantic"],
        "hallucination": ["style"],
        "boundary_error": ["boundary", "style"],
        "class_confusion": ["semantic", "scale"],
    }
    for ftype, prop in type_dist.items():
        if prop > 0.15:
            for a in adapter_map.get(ftype, []):
                recommended.add(a)

    return FailureReport(
        regions=all_regions,
        type_distribution=type_dist,
        overall_severity=np.mean([r.severity for r in all_regions]) if all_regions else 0.0,
        recommended_adapters=list(recommended),
        summary=f"Batch of {len(predictions)} images: {len(all_regions)} failure regions",
    )
