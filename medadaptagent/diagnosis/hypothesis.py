"""
Structured hypothesis and diagnosis dataclasses for CGFD framework.

ClinicalHypothesis: A falsifiable prediction about domain adaptation failure.
VerifiedDiagnosis: Collection of verified/refuted hypotheses with measured values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ClinicalHypothesis:
    """A single falsifiable hypothesis about a domain adaptation failure.

    Generated in Stage 1 from clinical domain knowledge. Verified in Stage 2
    against information-theoretic measures.

    Attributes:
        hypothesis_id: Unique ID (e.g., "H1_intensity_shift").
        clinical_cause: WHY the failure happens (clinical reason).
        expected_failure_type: What kind of failure: miss, boundary, confusion,
            hallucination, or scale_mismatch.
        affected_classes: Which segmentation classes are affected.
        measurable_prediction: Which CGFD measure to check and what to expect.
        measure_type: Which measure verifies this: "ccped", "lddp", or "acs".
        severity_estimate: Expected severity: "low", "medium", "high".
        verified: Whether this hypothesis was confirmed by measurement.
        verification_status: "confirmed", "refuted", or "inconclusive".
        measured_value: The actual measured value from verification.
        expected_range: (min, max) range for the measured value to confirm.
    """
    hypothesis_id: str = ""
    clinical_cause: str = ""
    expected_failure_type: str = ""  # miss, boundary, confusion, hallucination
    affected_classes: List[str] = field(default_factory=list)
    measurable_prediction: str = ""
    measure_type: str = ""  # ccped, lddp, acs
    severity_estimate: str = "medium"

    # Verification results (filled in Stage 2)
    verified: bool = False
    verification_status: str = "pending"  # confirmed, refuted, inconclusive
    measured_value: float = 0.0
    expected_range: Tuple[float, float] = (0.0, 1.0)
    verification_detail: str = ""

    def to_dict(self) -> Dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "clinical_cause": self.clinical_cause,
            "expected_failure_type": self.expected_failure_type,
            "affected_classes": self.affected_classes,
            "measurable_prediction": self.measurable_prediction,
            "measure_type": self.measure_type,
            "severity_estimate": self.severity_estimate,
            "verification_status": self.verification_status,
            "measured_value": self.measured_value,
            "expected_range": list(self.expected_range),
            "verification_detail": self.verification_detail,
        }


@dataclass
class CCPEDResult:
    """Class-Conditional Predictive Entropy Decomposition result for one class.

    Attributes:
        class_name: Name of the segmentation class.
        h_interior: Mean entropy in interior pixels of predicted class region.
        h_boundary: Mean entropy in boundary pixels of predicted class region.
        h_absent: Mean entropy where class is expected but not predicted.
        n_interior: Number of interior pixels.
        n_boundary: Number of boundary pixels.
        n_absent: Number of absent-region pixels.
        failure_type: Inferred failure type from entropy pattern.
    """
    class_name: str = ""
    h_interior: float = 0.0
    h_boundary: float = 0.0
    h_absent: float = 0.0
    n_interior: int = 0
    n_boundary: int = 0
    n_absent: int = 0
    failure_type: str = ""  # miss, boundary, confusion, healthy

    def to_dict(self) -> Dict:
        return {
            "class_name": self.class_name,
            "h_interior": self.h_interior,
            "h_boundary": self.h_boundary,
            "h_absent": self.h_absent,
            "n_interior": self.n_interior,
            "n_boundary": self.n_boundary,
            "n_absent": self.n_absent,
            "failure_type": self.failure_type,
        }


@dataclass
class LDDPResult:
    """Layer-wise Domain Discrepancy Profile result.

    Attributes:
        layer_names: Ordered list of layer/stage names.
        mmd_values: MMD² value at each layer.
        peak_layer: Layer with highest discrepancy.
        profile_type: Interpretation: "early" (appearance shift),
            "deep" (semantic shift), "uniform" (catastrophic), "minimal".
        recommended_adaptation_depth: How deep to adapt (layer index).
    """
    layer_names: List[str] = field(default_factory=list)
    mmd_values: List[float] = field(default_factory=list)
    peak_layer: str = ""
    profile_type: str = ""
    recommended_adaptation_depth: int = -1

    def to_dict(self) -> Dict:
        return {
            "layer_mmd": dict(zip(self.layer_names, self.mmd_values)),
            "peak_layer": self.peak_layer,
            "profile_type": self.profile_type,
            "recommended_adaptation_depth": self.recommended_adaptation_depth,
        }


@dataclass
class ACSResult:
    """Anatomical Consistency Score result.

    Attributes:
        total_score: Composite ACS (0-1, higher is better).
        per_rule_scores: ACS for each containment/spatial rule.
        per_class_scores: Per-class shape/size consistency.
        violations: List of specific anatomical violations detected.
    """
    total_score: float = 0.0
    per_rule_scores: Dict[str, float] = field(default_factory=dict)
    per_class_scores: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_score": self.total_score,
            "per_rule_scores": self.per_rule_scores,
            "per_class_scores": self.per_class_scores,
            "violations": self.violations,
        }


@dataclass
class VerifiedDiagnosis:
    """Complete diagnosis from CGFD Stage 2.

    Combines hypothesis verification with raw measure results.

    Attributes:
        hypotheses: List of verified ClinicalHypothesis objects.
        ccped_results: Per-class CCPED results.
        lddp_result: Layer-wise domain discrepancy profile.
        acs_result: Anatomical consistency score.
        confirmed_hypotheses: Subset that were confirmed.
        refuted_hypotheses: Subset that were refuted.
        new_findings: Unexpected findings not predicted by any hypothesis.
        overall_severity: Combined severity score (0-1).
        recommended_strategy: Mapping from diagnosis to adaptation actions.
    """
    hypotheses: List[ClinicalHypothesis] = field(default_factory=list)
    ccped_results: Dict[str, CCPEDResult] = field(default_factory=dict)
    lddp_result: Optional[LDDPResult] = None
    acs_result: Optional[ACSResult] = None

    confirmed_hypotheses: List[str] = field(default_factory=list)
    refuted_hypotheses: List[str] = field(default_factory=list)
    new_findings: List[str] = field(default_factory=list)
    overall_severity: float = 0.0

    recommended_strategy: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable diagnosis summary."""
        lines = [
            "=" * 60,
            "CGFD Verified Diagnosis",
            "=" * 60,
        ]

        # Hypothesis results
        lines.append(f"\nHypotheses: {len(self.confirmed_hypotheses)} confirmed, "
                     f"{len(self.refuted_hypotheses)} refuted")
        for h in self.hypotheses:
            status = h.verification_status.upper()
            lines.append(f"  [{status}] {h.hypothesis_id}: {h.clinical_cause}")
            if h.verification_detail:
                lines.append(f"           {h.verification_detail}")

        # CCPED
        if self.ccped_results:
            lines.append("\nClass-Conditional Entropy Decomposition:")
            for name, r in self.ccped_results.items():
                lines.append(
                    f"  {name}: H_int={r.h_interior:.3f}, "
                    f"H_bnd={r.h_boundary:.3f}, H_abs={r.h_absent:.3f} "
                    f"-> {r.failure_type}"
                )

        # LDDP
        if self.lddp_result:
            lines.append(f"\nDomain Discrepancy Profile: {self.lddp_result.profile_type}")
            lines.append(f"  Peak layer: {self.lddp_result.peak_layer}")
            for name, val in zip(self.lddp_result.layer_names,
                                 self.lddp_result.mmd_values):
                bar = "#" * int(val * 50)
                lines.append(f"  {name:>15s}: {val:.4f} |{bar}")

        # ACS
        if self.acs_result:
            lines.append(f"\nAnatomical Consistency Score: {self.acs_result.total_score:.3f}")
            for rule, score in self.acs_result.per_rule_scores.items():
                lines.append(f"  {rule}: {score:.3f}")
            if self.acs_result.violations:
                lines.append("  Violations:")
                for v in self.acs_result.violations:
                    lines.append(f"    - {v}")

        # New findings
        if self.new_findings:
            lines.append("\nUnexpected findings:")
            for f in self.new_findings:
                lines.append(f"  - {f}")

        lines.append(f"\nOverall severity: {self.overall_severity:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "ccped": {k: v.to_dict() for k, v in self.ccped_results.items()},
            "lddp": self.lddp_result.to_dict() if self.lddp_result else {},
            "acs": self.acs_result.to_dict() if self.acs_result else {},
            "confirmed": self.confirmed_hypotheses,
            "refuted": self.refuted_hypotheses,
            "new_findings": self.new_findings,
            "overall_severity": self.overall_severity,
            "recommended_strategy": self.recommended_strategy,
        }
