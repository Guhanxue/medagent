"""
Clinically-Grounded Failure Decomposition (CGFD) Framework.

Three-stage pipeline:
  Stage 1: Clinical Hypothesis Generation
    - LLM or rule-based generation of falsifiable predictions about failures
  Stage 2: Information-Theoretic Verification
    - Verify hypotheses with CCPED, LDDP, ACS measures
  Stage 3: Hypothesis-Verified Adaptation Loop
    - Re-verify after adaptation to track which failures are being resolved
    - Use ACS as label-free stopping criterion

Fully generalizable: parameterized only by TaskConfig.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from PIL import Image as PILImage

from medadaptagent.pipeline_config import TaskConfig
from medadaptagent.diagnosis.hypothesis import (
    ClinicalHypothesis, CCPEDResult, LDDPResult, ACSResult, VerifiedDiagnosis,
)
from medadaptagent.diagnosis.measures import (
    compute_ccped_batch, compute_lddp, compute_acs,
)


class CGFD:
    """Clinically-Grounded Failure Decomposition orchestrator.

    Implements the three-stage hypothesis → verify → adapt-and-track loop.
    Task-agnostic: all task specifics come from TaskConfig.

    Usage:
        cgfd = CGFD(config, llm_client=None)

        # Stage 1: Generate hypotheses
        hypotheses = cgfd.generate_hypotheses(domain_knowledge)

        # Stage 2: Verify with measures
        diagnosis = cgfd.verify(model, source_images, target_images)

        # Stage 3: Track during adaptation
        for iteration in range(max_iter):
            adapt(model, ...)
            update = cgfd.re_verify(model, target_images, previous=diagnosis)
            if update.acs_converged:
                break
    """

    def __init__(self, config: TaskConfig, llm_client=None,
                 image_size: int = None, preprocess_fn=None):
        """
        Args:
            config: TaskConfig defining structures, priors, rules.
            llm_client: Optional anthropic.Anthropic for LLM hypothesis generation.
            image_size: Model input size. If None, uses config.image_size.
            preprocess_fn: Optional function(image) -> tensor for model input.
        """
        self.config = config
        self.llm_client = llm_client
        self.image_size = image_size or config.image_size
        self.preprocess_fn = preprocess_fn

        # Tracking across iterations
        self.diagnosis_history: List[VerifiedDiagnosis] = []
        self.acs_history: List[float] = []

    # ==========================================================
    # Stage 1: Clinical Hypothesis Generation
    # ==========================================================

    def generate_hypotheses(
        self, domain_knowledge: Dict = None,
    ) -> List[ClinicalHypothesis]:
        """Generate falsifiable clinical hypotheses about expected failures.

        Args:
            domain_knowledge: Optional domain knowledge from RAG/LLM step 1.

        Returns:
            List of ClinicalHypothesis objects.
        """
        if self.llm_client:
            hypotheses = self._llm_hypotheses(domain_knowledge)
        else:
            hypotheses = self._rule_based_hypotheses(domain_knowledge)

        print(f"\n  CGFD Stage 1: Generated {len(hypotheses)} hypotheses")
        for h in hypotheses:
            print(f"    [{h.hypothesis_id}] {h.clinical_cause}")
            print(f"      Expected: {h.expected_failure_type} "
                  f"on {h.affected_classes}")

        return hypotheses

    def _llm_hypotheses(self, domain_knowledge: Dict = None,
                        ) -> List[ClinicalHypothesis]:
        """Use LLM to generate clinical hypotheses."""
        prompt = self._build_hypothesis_prompt(domain_knowledge)

        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return [self._dict_to_hypothesis(d) for d in parsed]
        except Exception as e:
            print(f"  LLM hypothesis generation failed ({e}), "
                  f"falling back to rule-based.")

        return self._rule_based_hypotheses(domain_knowledge)

    def _build_hypothesis_prompt(self, domain_knowledge: Dict = None) -> str:
        """Build prompt for LLM hypothesis generation."""
        config = self.config
        fg_classes = list(config.fg_classes.keys())
        class_desc = json.dumps(config.class_descriptions, indent=2)
        priors = json.dumps(config.anatomy_priors, indent=2)
        rules = str(config.containment_rules)

        knowledge_str = ""
        if domain_knowledge:
            knowledge_str = f"\n## Domain Knowledge\n{json.dumps(domain_knowledge, indent=2)}\n"

        return f"""You are a medical image analysis expert. Generate FALSIFIABLE hypotheses about why a segmentation model will fail when transferred from source to target domain.

## Task
{config.get_task_description()}

## Source: {config.source_domain}
## Target: {config.target_domain}

## Classes to segment: {fg_classes}
## Class descriptions: {class_desc}
## Anatomy priors: {priors}
## Structural rules: {rules}
{knowledge_str}
## Available verification measures
1. CCPED (Class-Conditional Predictive Entropy Decomposition):
   - H_interior(class): entropy inside predicted region
   - H_boundary(class): entropy at boundaries
   - H_absent(class): entropy where class expected but missing
   → Detects: miss, boundary failure, confusion

2. LDDP (Layer-wise Domain Discrepancy Profile):
   - MMD at each network layer (early=appearance, deep=semantic)
   → Detects: where in the network the shift breaks

3. ACS (Anatomical Consistency Score):
   - Containment, centroid alignment, shape regularity
   → Detects: structural violations in predictions

## Instructions
Generate 3-6 hypotheses. Each must be:
- Clinically grounded (explain WHY from imaging physics/biology)
- Falsifiable (specify which measure should show what)
- Class-specific (which structures are affected)

Return a JSON array:
```json
[
  {{
    "hypothesis_id": "H1_...",
    "clinical_cause": "why this failure happens (clinical reason)",
    "expected_failure_type": "miss|boundary|confusion|hallucination",
    "affected_classes": ["class1", "class2"],
    "measurable_prediction": "what CCPED/LDDP/ACS should show",
    "measure_type": "ccped|lddp|acs",
    "severity_estimate": "low|medium|high"
  }}
]
```

Be specific. Reference actual imaging properties of {config.source_domain} vs {config.target_domain}."""

    def _dict_to_hypothesis(self, d: Dict) -> ClinicalHypothesis:
        """Convert parsed dict to ClinicalHypothesis."""
        return ClinicalHypothesis(
            hypothesis_id=d.get("hypothesis_id", "H_unknown"),
            clinical_cause=d.get("clinical_cause", ""),
            expected_failure_type=d.get("expected_failure_type", "unknown"),
            affected_classes=d.get("affected_classes", []),
            measurable_prediction=d.get("measurable_prediction", ""),
            measure_type=d.get("measure_type", "ccped"),
            severity_estimate=d.get("severity_estimate", "medium"),
        )

    def _rule_based_hypotheses(
        self, domain_knowledge: Dict = None,
    ) -> List[ClinicalHypothesis]:
        """Generate hypotheses from TaskConfig without LLM.

        Works for ANY task by reasoning from config fields.
        """
        config = self.config
        fg_classes = list(config.fg_classes.keys())
        hypotheses = []
        h_idx = 1

        # Hypothesis: Intensity/appearance shift
        hypotheses.append(ClinicalHypothesis(
            hypothesis_id=f"H{h_idx}_intensity_shift",
            clinical_cause=(
                f"Imaging device/protocol difference between "
                f"{config.source_domain} and {config.target_domain} "
                f"causes intensity distribution mismatch"
            ),
            expected_failure_type="miss",
            affected_classes=fg_classes,
            measurable_prediction=(
                "LDDP should show high MMD at early layers (appearance shift). "
                "CCPED should show high H_interior or complete miss."
            ),
            measure_type="lddp",
            severity_estimate="high",
        ))
        h_idx += 1

        # Hypothesis: Per-class miss for small structures
        for cls_name in fg_classes:
            prior = config.anatomy_priors.get(cls_name, {})
            area_range = prior.get("area_range", (0.0, 1.0))

            if area_range[1] < 0.15:  # small structure
                hypotheses.append(ClinicalHypothesis(
                    hypothesis_id=f"H{h_idx}_{cls_name}_miss",
                    clinical_cause=(
                        f"{cls_name} is a small structure "
                        f"(expected area {area_range[0]*100:.1f}-{area_range[1]*100:.1f}%). "
                        f"Domain shift may cause complete miss."
                    ),
                    expected_failure_type="miss",
                    affected_classes=[cls_name],
                    measurable_prediction=(
                        f"CCPED: H_absent({cls_name}) should be high. "
                        f"ACS: {cls_name} area should be below expected range."
                    ),
                    measure_type="ccped",
                    severity_estimate="high",
                ))
                h_idx += 1

        # Hypothesis: Containment violations
        for outer, relation, inner in config.containment_rules:
            hypotheses.append(ClinicalHypothesis(
                hypothesis_id=f"H{h_idx}_{inner}_containment",
                clinical_cause=(
                    f"If {outer} prediction degrades, {inner} "
                    f"(which should be {relation} {outer}) will also fail. "
                    f"Structural relationship may be violated."
                ),
                expected_failure_type="confusion",
                affected_classes=[outer, inner],
                measurable_prediction=(
                    f"ACS containment score for '{outer} {relation} {inner}' "
                    f"should be low. {inner} may appear outside {outer}."
                ),
                measure_type="acs",
                severity_estimate="medium",
            ))
            h_idx += 1

        # Hypothesis: Scale mismatch (from domain knowledge if available)
        if domain_knowledge:
            shifts = domain_knowledge.get("shift_types", [])
            for shift in shifts:
                if any(kw in str(shift).lower() for kw in
                       ["scale", "size", "fov", "resolution", "zoom"]):
                    hypotheses.append(ClinicalHypothesis(
                        hypothesis_id=f"H{h_idx}_scale_shift",
                        clinical_cause=f"Scale/FOV difference: {shift}",
                        expected_failure_type="boundary",
                        affected_classes=fg_classes,
                        measurable_prediction=(
                            "CCPED: boundary entropy should be high for all classes. "
                            "ACS: area fractions may be outside expected range."
                        ),
                        measure_type="ccped",
                        severity_estimate="medium",
                    ))
                    h_idx += 1
                    break

        # Hypothesis: Boundary degradation (always relevant)
        hypotheses.append(ClinicalHypothesis(
            hypothesis_id=f"H{h_idx}_boundary_degradation",
            clinical_cause=(
                "Domain shift typically degrades boundary precision "
                "even when structures are detected, due to texture/contrast changes."
            ),
            expected_failure_type="boundary",
            affected_classes=fg_classes,
            measurable_prediction=(
                "CCPED: H_boundary should be significantly higher than H_interior "
                "for detected classes."
            ),
            measure_type="ccped",
            severity_estimate="medium",
        ))

        return hypotheses

    # ==========================================================
    # Stage 2: Information-Theoretic Verification
    # ==========================================================

    def verify(
        self,
        model: nn.Module,
        source_images: List[np.ndarray],
        target_images: List[np.ndarray],
        hypotheses: List[ClinicalHypothesis] = None,
    ) -> VerifiedDiagnosis:
        """Verify hypotheses with CCPED, LDDP, ACS measures.

        Args:
            model: Source model (eval mode).
            source_images: Source domain images.
            target_images: Target domain images.
            hypotheses: Hypotheses to verify. If None, generates them.

        Returns:
            VerifiedDiagnosis with all measures and verified hypotheses.
        """
        print("\n  CGFD Stage 2: Information-Theoretic Verification")

        if hypotheses is None:
            hypotheses = self.generate_hypotheses()

        config = self.config
        device = next(model.parameters()).device

        # 1. Compute CCPED
        print("    Computing CCPED (entropy decomposition)...")
        ccped_results = compute_ccped_batch(
            model, target_images,
            class_names=config.structures,
            anatomy_priors=config.anatomy_priors,
            image_size=self.image_size,
            preprocess_fn=self.preprocess_fn,
        )

        # 2. Compute LDDP
        print("    Computing LDDP (layer-wise domain discrepancy)...")
        lddp_result = compute_lddp(
            model, source_images, target_images,
            image_size=self.image_size,
            preprocess_fn=self.preprocess_fn,
        )

        # 3. Compute ACS (need predictions first)
        print("    Computing ACS (anatomical consistency)...")
        predictions = self._get_predictions(model, target_images, device)
        acs_result = compute_acs(
            predictions,
            class_names=config.structures,
            containment_rules=config.containment_rules,
            anatomy_priors=config.anatomy_priors,
        )

        # 4. Verify each hypothesis
        print("    Verifying hypotheses...")
        confirmed = []
        refuted = []
        for h in hypotheses:
            self._verify_hypothesis(h, ccped_results, lddp_result, acs_result)
            if h.verification_status == "confirmed":
                confirmed.append(h.hypothesis_id)
            elif h.verification_status == "refuted":
                refuted.append(h.hypothesis_id)

        # 5. Check for unexpected findings
        new_findings = self._detect_new_findings(
            ccped_results, lddp_result, acs_result, hypotheses)

        # 6. Compute overall severity
        severity = self._compute_severity(
            ccped_results, lddp_result, acs_result)

        # 7. Build recommended strategy
        strategy = self._build_strategy(
            ccped_results, lddp_result, acs_result, hypotheses)

        diagnosis = VerifiedDiagnosis(
            hypotheses=hypotheses,
            ccped_results=ccped_results,
            lddp_result=lddp_result,
            acs_result=acs_result,
            confirmed_hypotheses=confirmed,
            refuted_hypotheses=refuted,
            new_findings=new_findings,
            overall_severity=severity,
            recommended_strategy=strategy,
        )

        self.diagnosis_history.append(diagnosis)
        self.acs_history.append(acs_result.total_score)

        print(diagnosis.summary())
        return diagnosis

    def _get_predictions(
        self, model: nn.Module, images: List[np.ndarray],
        device: torch.device, max_images: int = 50,
    ) -> List[np.ndarray]:
        """Get model predictions on target images."""
        model.eval()
        predictions = []

        with torch.no_grad():
            for img in images[:max_images]:
                if self.preprocess_fn is not None:
                    tensor = self.preprocess_fn(img).to(device)
                else:
                    pil = PILImage.fromarray((img * 255).astype(np.uint8))
                    pil = pil.resize(
                        (self.image_size, self.image_size), PILImage.BILINEAR)
                    tensor = torch.from_numpy(
                        np.array(pil).astype(np.float32) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0).to(device)

                logits = model(tensor)
                pred = logits.argmax(dim=1)[0].cpu().numpy()
                predictions.append(pred)

        return predictions

    def _verify_hypothesis(
        self,
        hypothesis: ClinicalHypothesis,
        ccped: Dict[str, CCPEDResult],
        lddp: LDDPResult,
        acs: ACSResult,
    ):
        """Verify a single hypothesis against measured values."""
        h = hypothesis

        if h.measure_type == "ccped":
            self._verify_ccped_hypothesis(h, ccped)
        elif h.measure_type == "lddp":
            self._verify_lddp_hypothesis(h, lddp)
        elif h.measure_type == "acs":
            self._verify_acs_hypothesis(h, acs)
        else:
            h.verification_status = "inconclusive"
            h.verification_detail = f"Unknown measure type: {h.measure_type}"

    def _verify_ccped_hypothesis(
        self, h: ClinicalHypothesis, ccped: Dict[str, CCPEDResult],
    ):
        """Verify a CCPED-related hypothesis."""
        affected = h.affected_classes
        failure_type = h.expected_failure_type

        matching_failures = 0
        total_checked = 0

        for cls_name in affected:
            if cls_name not in ccped:
                continue
            total_checked += 1
            result = ccped[cls_name]

            if failure_type == "miss" and result.failure_type in ("miss", "partial_miss"):
                matching_failures += 1
                h.measured_value = result.h_absent
            elif failure_type == "boundary" and result.failure_type == "boundary":
                matching_failures += 1
                h.measured_value = result.h_boundary
            elif failure_type == "confusion" and result.failure_type == "confusion":
                matching_failures += 1
                h.measured_value = result.h_interior
            elif failure_type == result.failure_type:
                matching_failures += 1

        if total_checked == 0:
            h.verification_status = "inconclusive"
            h.verification_detail = "No affected classes found in CCPED results"
            return

        ratio = matching_failures / total_checked
        if ratio >= 0.5:
            h.verification_status = "confirmed"
            h.verified = True
            h.verification_detail = (
                f"{matching_failures}/{total_checked} classes show "
                f"expected {failure_type} pattern"
            )
        elif ratio > 0:
            h.verification_status = "inconclusive"
            h.verification_detail = (
                f"Partial match: {matching_failures}/{total_checked} classes"
            )
        else:
            h.verification_status = "refuted"
            actual_types = [ccped[c].failure_type for c in affected if c in ccped]
            h.verification_detail = (
                f"Expected {failure_type}, observed: {actual_types}"
            )

    def _verify_lddp_hypothesis(
        self, h: ClinicalHypothesis, lddp: LDDPResult,
    ):
        """Verify an LDDP-related hypothesis."""
        if not lddp.mmd_values:
            h.verification_status = "inconclusive"
            h.verification_detail = "No LDDP data available"
            return

        max_mmd = max(lddp.mmd_values)
        h.measured_value = max_mmd

        # Check if profile matches expected pattern
        if "early" in h.measurable_prediction.lower():
            if lddp.profile_type in ("early_shift", "uniform_shift"):
                h.verification_status = "confirmed"
                h.verified = True
                h.verification_detail = (
                    f"Profile: {lddp.profile_type}, peak at {lddp.peak_layer}, "
                    f"max MMD={max_mmd:.4f}"
                )
            else:
                h.verification_status = "refuted"
                h.verification_detail = (
                    f"Expected early shift, got {lddp.profile_type}"
                )
        elif "deep" in h.measurable_prediction.lower():
            if lddp.profile_type in ("deep_shift", "uniform_shift"):
                h.verification_status = "confirmed"
                h.verified = True
            else:
                h.verification_status = "refuted"
        else:
            # General shift hypothesis — confirmed if any significant MMD
            if max_mmd > 0.01:
                h.verification_status = "confirmed"
                h.verified = True
                h.verification_detail = (
                    f"Significant domain discrepancy detected: "
                    f"max MMD={max_mmd:.4f} at {lddp.peak_layer}"
                )
            else:
                h.verification_status = "refuted"
                h.verification_detail = "Minimal domain discrepancy"

    def _verify_acs_hypothesis(
        self, h: ClinicalHypothesis, acs: ACSResult,
    ):
        """Verify an ACS-related hypothesis."""
        h.measured_value = acs.total_score

        # Check specific rule violations
        relevant_violations = [
            v for v in acs.violations
            if any(cls in v for cls in h.affected_classes)
        ]

        if relevant_violations:
            h.verification_status = "confirmed"
            h.verified = True
            h.verification_detail = f"Violations found: {relevant_violations}"
        elif acs.total_score < 0.5:
            h.verification_status = "confirmed"
            h.verified = True
            h.verification_detail = f"Low ACS ({acs.total_score:.3f}) suggests structural issues"
        else:
            h.verification_status = "refuted"
            h.verification_detail = f"ACS={acs.total_score:.3f}, no relevant violations"

    def _detect_new_findings(
        self,
        ccped: Dict[str, CCPEDResult],
        lddp: LDDPResult,
        acs: ACSResult,
        hypotheses: List[ClinicalHypothesis],
    ) -> List[str]:
        """Detect findings not predicted by any hypothesis."""
        findings = []
        hypothesized_classes = set()
        for h in hypotheses:
            hypothesized_classes.update(h.affected_classes)

        # Check for unexpected class failures
        for name, result in ccped.items():
            if name not in hypothesized_classes and result.failure_type != "healthy":
                findings.append(
                    f"Unexpected {result.failure_type} failure for class '{name}' "
                    f"(not predicted by any hypothesis)"
                )

        # Check for unexpected LDDP pattern
        if lddp and lddp.profile_type == "deep_shift":
            has_deep_hypothesis = any(
                "deep" in h.measurable_prediction.lower() or
                "semantic" in h.measurable_prediction.lower()
                for h in hypotheses
            )
            if not has_deep_hypothesis:
                findings.append(
                    f"Unexpected deep semantic shift detected "
                    f"(peak at {lddp.peak_layer})"
                )

        # Check for unexpected ACS violations
        if acs:
            for violation in acs.violations:
                # Check if any hypothesis predicted this
                predicted = any(
                    any(cls in violation for cls in h.affected_classes)
                    for h in hypotheses if h.measure_type == "acs"
                )
                if not predicted:
                    findings.append(f"Unexpected: {violation}")

        return findings

    def _compute_severity(
        self,
        ccped: Dict[str, CCPEDResult],
        lddp: LDDPResult,
        acs: ACSResult,
    ) -> float:
        """Compute overall severity score (0-1)."""
        scores = []

        # CCPED: fraction of classes with non-healthy failures
        if ccped:
            n_failing = sum(1 for r in ccped.values()
                           if r.failure_type != "healthy")
            scores.append(n_failing / max(len(ccped), 1))

        # LDDP: max normalized MMD
        if lddp and lddp.mmd_values:
            max_mmd = max(lddp.mmd_values)
            # Normalize: 0.1 = moderate, 0.5+ = severe
            scores.append(min(max_mmd * 2, 1.0))

        # ACS: inverted (low ACS = high severity)
        if acs:
            scores.append(1 - acs.total_score)

        return float(np.mean(scores)) if scores else 0.5

    def _build_strategy(
        self,
        ccped: Dict[str, CCPEDResult],
        lddp: LDDPResult,
        acs: ACSResult,
        hypotheses: List[ClinicalHypothesis],
    ) -> Dict:
        """Build recommended adaptation strategy from diagnosis.

        Maps CGFD findings → specific adaptation mechanisms and weights.
        """
        strategy = {
            "loss_weights": {
                "entropy": 0.0,
                "consistency": 0.0,
                "diversity": 0.0,
                "pseudo_label": 0.0,
                "anatomy": 0.0,
            },
            "trainable_scope": "bn+output",
            "use_anatomy_refinement": False,
            "reasoning": [],
        }

        # CCPED-driven decisions
        has_miss = any(r.failure_type in ("miss", "partial_miss")
                      for r in ccped.values())
        has_boundary = any(r.failure_type == "boundary" for r in ccped.values())
        has_confusion = any(r.failure_type == "confusion" for r in ccped.values())

        if has_miss:
            strategy["loss_weights"]["diversity"] = 1.0
            strategy["loss_weights"]["entropy"] = 0.5
            strategy["reasoning"].append(
                "Diversity loss activated: class miss detected, prevent collapse"
            )

        if has_boundary:
            strategy["loss_weights"]["consistency"] = 1.0
            strategy["loss_weights"]["entropy"] = 0.8
            strategy["reasoning"].append(
                "Consistency + entropy: boundary failures need stable predictions"
            )

        if has_confusion:
            strategy["loss_weights"]["entropy"] = 1.0
            strategy["loss_weights"]["pseudo_label"] = 0.5
            strategy["reasoning"].append(
                "Entropy + pseudo-label: confusion needs sharper predictions"
            )

        # LDDP-driven decisions
        if lddp:
            if lddp.profile_type == "early_shift":
                strategy["trainable_scope"] = "bn+output"
                strategy["reasoning"].append(
                    "Early shift: BN reset + output head adaptation sufficient"
                )
            elif lddp.profile_type == "deep_shift":
                strategy["trainable_scope"] = "bn+all_decoder"
                strategy["reasoning"].append(
                    "Deep shift: need decoder adaptation for semantic alignment"
                )
            elif lddp.profile_type == "uniform_shift":
                strategy["trainable_scope"] = "bn+all_decoder"
                strategy["loss_weights"]["consistency"] = max(
                    strategy["loss_weights"]["consistency"], 1.0)
                strategy["reasoning"].append(
                    "Uniform shift: comprehensive adaptation needed"
                )

        # ACS-driven decisions
        if acs and acs.violations:
            strategy["use_anatomy_refinement"] = True
            strategy["loss_weights"]["anatomy"] = 1.0
            strategy["reasoning"].append(
                f"Anatomy refinement activated: {len(acs.violations)} "
                f"structural violations detected"
            )

        # Default: ensure at least some adaptation
        if all(v == 0 for v in strategy["loss_weights"].values()):
            strategy["loss_weights"]["entropy"] = 0.5
            strategy["loss_weights"]["consistency"] = 0.5
            strategy["reasoning"].append(
                "Default: mild entropy + consistency for general improvement"
            )

        return strategy

    # ==========================================================
    # Stage 3: Re-verification (Track Adaptation Progress)
    # ==========================================================

    def re_verify(
        self,
        model: nn.Module,
        target_images: List[np.ndarray],
        source_images: List[np.ndarray] = None,
        previous: VerifiedDiagnosis = None,
    ) -> Dict:
        """Re-verify after adaptation to track which failures are resolved.

        Computes delta-CCPED, delta-LDDP, delta-ACS compared to previous.
        Uses ACS monotonicity as convergence / over-adaptation signal.

        Args:
            model: Adapted model.
            target_images: Target domain images.
            source_images: Source images (for LDDP, optional on re-verify).
            previous: Previous VerifiedDiagnosis for delta computation.

        Returns:
            Dict with:
                - current_acs: Current ACS score
                - acs_delta: Change from previous
                - acs_converged: Whether ACS has plateaued or declined
                - resolved_failures: Which class failures improved
                - remaining_failures: Which failures persist
                - updated_strategy: Optional strategy update
        """
        print("\n    CGFD Re-verification...")
        config = self.config
        device = next(model.parameters()).device

        # Compute current CCPED
        ccped_results = compute_ccped_batch(
            model, target_images,
            class_names=config.structures,
            anatomy_priors=config.anatomy_priors,
            image_size=self.image_size,
            preprocess_fn=self.preprocess_fn,
        )

        # Compute current ACS
        predictions = self._get_predictions(model, target_images, device)
        acs_result = compute_acs(
            predictions,
            class_names=config.structures,
            containment_rules=config.containment_rules,
            anatomy_priors=config.anatomy_priors,
        )

        current_acs = acs_result.total_score
        self.acs_history.append(current_acs)

        # Delta computation
        acs_delta = 0.0
        acs_converged = False
        resolved = []
        remaining = []

        if previous:
            prev_acs = previous.acs_result.total_score if previous.acs_result else 0.0
            acs_delta = current_acs - prev_acs

            # Convergence: ACS improvement < 0.01 or ACS decreased
            if len(self.acs_history) >= 2:
                recent_improvement = self.acs_history[-1] - self.acs_history[-2]
                if recent_improvement < 0.01:
                    acs_converged = True

            # Track class-level resolution
            for name, result in ccped_results.items():
                prev_result = previous.ccped_results.get(name)
                if prev_result:
                    if (prev_result.failure_type != "healthy" and
                            result.failure_type == "healthy"):
                        resolved.append(name)
                    elif result.failure_type != "healthy":
                        remaining.append(
                            f"{name}: {result.failure_type}")

        # Print summary
        print(f"      ACS: {current_acs:.3f} (delta={acs_delta:+.3f})")
        print(f"      Converged: {acs_converged}")
        if resolved:
            print(f"      Resolved: {resolved}")
        if remaining:
            print(f"      Remaining: {remaining}")

        # Build update recommendation
        updated_strategy = None
        if remaining and not acs_converged:
            updated_strategy = self._build_strategy(
                ccped_results, None, acs_result, [])

        return {
            "current_acs": current_acs,
            "acs_delta": acs_delta,
            "acs_converged": acs_converged,
            "ccped_results": ccped_results,
            "acs_result": acs_result,
            "resolved_failures": resolved,
            "remaining_failures": remaining,
            "updated_strategy": updated_strategy,
            "acs_history": list(self.acs_history),
        }

    def get_acs_curve(self) -> List[float]:
        """Return the ACS trajectory over adaptation iterations."""
        return list(self.acs_history)
