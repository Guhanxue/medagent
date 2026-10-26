"""
MedAdaptAgent: 3-Step Agentic Domain Adaptation Pipeline.

Fully task-agnostic. To apply to a new dataset:
  1. Create a TaskConfig (see pipeline_config.py)
  2. Pass it to MedAdaptAgent
  3. Call agent.run()

Pipeline:
  Step 1: RAG-based Domain Knowledge
    - Query knowledge base about known domain shift characteristics
    - Produce structured explanation of WHY shift exists

  Step 2: Statistical Verification
    - Compute image/frequency/scale statistics comparing source vs target
    - Run source model on target, analyze failure patterns
    - Verify RAG predictions with actual data

  Step 3: Targeted Adaptation
    - Based on verified diagnosis, design specific adaptation strategy
    - BN stats reset + self-supervised gradient updates
    - Anatomy-aware pseudo-label refinement with structural constraints
    - Per-class quality monitoring with early stopping
"""

import os
import json
import time
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from medadaptagent.diagnosis.domain_shift_analyzer import DomainShiftAnalyzer, ShiftReport
from medadaptagent.diagnosis.cgfd import CGFD
from medadaptagent.pipeline_config import (
    TaskConfig, STEP1_RAG_PROMPT, STEP2_VERIFY_PROMPT, STEP3_ADAPT_PROMPT,
)


@dataclass
class AdaptationPlan:
    """Strategy designed by the agent based on diagnosis."""
    learning_rate: float = 1e-4
    num_steps: int = 200
    loss_weights: Dict[str, float] = field(default_factory=dict)
    trainable_scope: str = "bn+output"
    use_anatomy_refinement: bool = False
    rationale: str = ""
    # Stage 2: pseudo-label refinement (no BN reset)
    stage2_iterations: int = 0
    stage2_lr: float = 5e-5
    stage2_num_steps: int = 300
    stage2_loss_weights: Dict[str, float] = field(default_factory=dict)
    # Pseudo-label denoising strategy (agent-selected)
    # Options: "none", "prototype", "vlm_proxy", "both"
    #   - "prototype": feature-based denoising (PrototypeDenoiser)
    #   - "vlm_proxy": VLM class-presence gating (ProDe-inspired)
    #   - "both": prototype + VLM proxy cascade
    denoising_method: str = "none"
    # Feature augmentation regularization (SF(DA)²-inspired)
    use_feature_augment: bool = False


class MedAdaptAgent:
    """Task-agnostic agentic system for domain adaptation.

    Pipeline:
      1. Understand WHY domain shift exists (RAG + LLM)
      2. VERIFY shift with statistics and model failure analysis
      3. ADAPT model with targeted, self-supervised corrections

    Usage:
        from medadaptagent.pipeline_config import FUNDUS_CONFIG
        agent = MedAdaptAgent(model, src_imgs, tgt_imgs, config=FUNDUS_CONFIG)
        results = agent.run()
    """

    def __init__(self, source_model: nn.Module,
                 source_images: List[np.ndarray],
                 target_images: List[np.ndarray],
                 config: TaskConfig,
                 source_labels: List[np.ndarray] = None,
                 target_labels: List[np.ndarray] = None,
                 llm_client=None):
        """
        Args:
            source_model: Model trained on source domain.
            source_images: List of (H,W,3) float32 [0,1] source images.
            target_images: List of (H,W,3) float32 [0,1] target images.
            config: TaskConfig defining the task (structures, priors, etc.).
            source_labels: Optional source labels for scale analysis.
            target_labels: Optional target GT (development evaluation only).
            llm_client: Optional anthropic.Anthropic client for LLM reasoning.
        """
        self.source_model = source_model
        self.source_images = source_images
        self.target_images = target_images
        self.config = config
        self.source_labels = source_labels
        self.target_labels = target_labels
        self.llm_client = llm_client
        self.device = config.device
        self.image_size = config.image_size

        self.analyzer = DomainShiftAnalyzer(device=self.device)
        self.cgfd = CGFD(config=config, llm_client=llm_client,
                         image_size=config.image_size)
        self.shift_report = None
        self.adaptation_plan = None
        self.adapted_model = None
        self.domain_knowledge = {}
        self.cgfd_diagnosis = None  # VerifiedDiagnosis from CGFD
        self.history = []

        # Extract model provenance if available (external models)
        self.model_provenance = getattr(source_model, 'provenance', None)
        if self.model_provenance is not None:
            # Feed provenance into domain knowledge for CGFD
            self.domain_knowledge.update(
                self.model_provenance.to_domain_knowledge()
            )

        # Logs for reproducibility -- every decision and its reason
        self.pipeline_log = {
            "config": {
                "task_name": config.task_name,
                "source_domain": config.source_domain,
                "target_domain": config.target_domain,
                "structures": config.structures,
                "anatomy_priors": config.anatomy_priors,
                "containment_rules": config.containment_rules,
            },
            "model_provenance": (self.model_provenance.summary()
                                if self.model_provenance else "self-trained"),
            "steps": [],
        }

    def _log_step(self, step_name: str, data: Dict):
        """Append a step record to the pipeline log."""
        self.pipeline_log["steps"].append({
            "step": step_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **data,
        })

    # ==========================================================
    # STEP 1: RAG-Based Domain Knowledge
    # ==========================================================
    def step1_domain_knowledge(self, knowledge_base=None) -> Dict:
        """Query knowledge base for domain shift explanations.

        If llm_client is available, sends STEP1_RAG_PROMPT to LLM.
        Otherwise, uses rule-based generation from TaskConfig.
        """
        print("\n" + "=" * 60)
        print("STEP 1: RAG-Based Domain Knowledge")
        print("=" * 60)

        if self.llm_client:
            knowledge = self._llm_domain_knowledge()
        else:
            knowledge = self._rule_based_domain_knowledge()

        print(f"  Task: {self.config.get_task_description()}")
        print(f"  Known shift types: {knowledge.get('shift_types', [])}")
        print(f"  Expected challenges: {knowledge.get('challenges', [])}")
        print(f"  Literature strategies: {knowledge.get('strategies', [])}")
        print(f"  Anatomy priors: {json.dumps(knowledge.get('anatomy_priors', {}), indent=4)}")

        self.domain_knowledge = knowledge
        self._log_step("step1_rag_knowledge", {
            "method": "llm" if self.llm_client else "rule_based",
            "knowledge": knowledge,
        })
        return knowledge

    def _llm_domain_knowledge(self) -> Dict:
        """Use LLM for Step 1 domain knowledge generation."""
        prompt = STEP1_RAG_PROMPT.format(
            task_description=self.config.get_task_description(),
            source_domain=self.config.source_domain,
            target_domain=self.config.target_domain,
            class_descriptions=json.dumps(self.config.class_descriptions, indent=2),
        )
        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                # Merge LLM anatomy priors with config priors (config takes priority)
                if "anatomy_priors" not in result:
                    result["anatomy_priors"] = {}
                for cls, prior in self.config.anatomy_priors.items():
                    if cls not in result["anatomy_priors"]:
                        result["anatomy_priors"][cls] = prior
                return result
        except Exception as e:
            print(f"  LLM call failed ({e}), falling back to rule-based.")
        return self._rule_based_domain_knowledge()

    def _rule_based_domain_knowledge(self) -> Dict:
        """Generate domain knowledge from TaskConfig without LLM.

        Works for ANY task -- builds knowledge from the config's structured fields.
        """
        cfg = self.config
        shift_types = []
        challenges = []
        strategies = [
            "BatchNorm statistics adaptation (correct distribution shift)",
            "Entropy minimization (encourage confident target predictions)",
            "Consistency regularization (enforce augmentation invariance)",
            "Pseudo-label self-training with confidence filtering",
        ]

        # Infer shift types from domain descriptions
        src_low = cfg.source_domain.lower()
        tgt_low = cfg.target_domain.lower()

        if any(w in src_low or w in tgt_low for w in ["ct", "mri", "t1", "t2"]):
            shift_types.append("cross-modality (different imaging physics)")
            challenges.append("Fundamentally different tissue contrast mechanisms")
            strategies.append("Frequency-domain style transfer")
        if any(w in src_low or w in tgt_low for w in ["device", "camera", "scanner", "hospital"]):
            shift_types.append("cross-device (different acquisition hardware)")
            challenges.append("Varying resolution, noise patterns, and calibration")
        if any(w in src_low or w in tgt_low for w in ["fundus", "retina", "eye"]):
            shift_types.append("intensity shift (exposure/gain differences)")
            challenges.append("Color distribution varies across fundus cameras")

        # Generic shifts that apply to most medical imaging DA tasks
        shift_types.extend([
            "intensity/contrast distribution shift",
            "spatial resolution or field-of-view difference",
        ])
        challenges.extend([
            "Source-trained BN statistics mismatched to target distribution",
            "Model may predict all background due to distribution mismatch",
        ])

        return {
            "source_domain": cfg.source_domain,
            "target_domain": cfg.target_domain,
            "task": cfg.get_task_description(),
            "shift_types": shift_types,
            "challenges": challenges,
            "strategies": strategies,
            "anatomy_priors": cfg.anatomy_priors,
            "containment_rules": cfg.containment_rules,
        }

    # ==========================================================
    # STEP 2: Statistical Verification
    # ==========================================================
    def step2_verify_shift(self) -> ShiftReport:
        """Verify domain shift with actual data statistics."""
        print("\n" + "=" * 60)
        print("STEP 2: Statistical Verification of Domain Shift")
        print("=" * 60)

        class_names = list(self.config.structures.keys())
        self.shift_report = self.analyzer.analyze(
            source_images=self.source_images,
            target_images=self.target_images,
            source_labels=self.source_labels,
            target_labels=self.target_labels,
            model=self.source_model,
            image_size=self.image_size,
            num_classes=self.config.num_classes,
            class_names=class_names,
        )

        print("\n" + self.shift_report.summary())
        verification = self._verify_rag_predictions()

        # CGFD Stage 1+2: Generate hypotheses and verify with
        # information-theoretic measures (CCPED, LDDP, ACS)
        print("\n" + "=" * 60)
        print("CGFD: Clinically-Grounded Failure Decomposition")
        print("=" * 60)
        hypotheses = self.cgfd.generate_hypotheses(self.domain_knowledge)
        self.cgfd_diagnosis = self.cgfd.verify(
            model=self.source_model,
            source_images=self.source_images,
            target_images=self.target_images,
            hypotheses=hypotheses,
        )

        self._log_step("step2_statistical_verification", {
            "shift_report": self.shift_report.summary(),
            "verified_shifts": self.shift_report.verified_shifts,
            "rag_verification": verification,
            "cgfd_diagnosis": self.cgfd_diagnosis.to_dict(),
        })
        return self.shift_report

    def _verify_rag_predictions(self) -> Dict:
        """Cross-reference RAG knowledge with statistical findings."""
        print("\n  Verification against RAG predictions:")
        rag_shifts = self.domain_knowledge.get("shift_types", [])
        verified = self.shift_report.verified_shifts

        # Keyword map: RAG terms -> statistical shift names
        keyword_map = {
            "intensity": "intensity_shift",
            "color": "intensity_shift",
            "contrast": "intensity_shift",
            "exposure": "intensity_shift",
            "field-of-view": "scale_shift",
            "resolution": "scale_shift",
            "scale": "scale_shift",
            "device": ["style_shift", "texture_shift"],
            "camera": ["style_shift", "texture_shift"],
            "scanner": ["style_shift", "texture_shift"],
            "modality": ["style_shift", "texture_shift", "intensity_shift"],
            "style": "style_shift",
            "texture": "texture_shift",
            "frequency": "style_shift",
        }

        confirmed, unverified = [], []
        for rag_pred in rag_shifts:
            matched = False
            for keyword, shift_names in keyword_map.items():
                if keyword in rag_pred.lower():
                    if isinstance(shift_names, str):
                        shift_names = [shift_names]
                    if any(s in verified for s in shift_names):
                        print(f"    [CONFIRMED] {rag_pred}")
                        confirmed.append(rag_pred)
                        matched = True
                        break
            if not matched:
                print(f"    [UNVERIFIED] {rag_pred}")
                unverified.append(rag_pred)

        new_findings = []
        for finding in verified:
            predicted = any(
                any(k in rp.lower() for k in finding.replace("_", " ").split())
                for rp in rag_shifts
            )
            if not predicted:
                print(f"    [NEW FINDING] {finding} (not predicted by RAG)")
                new_findings.append(finding)

        return {"confirmed": confirmed, "unverified": unverified,
                "new_findings": new_findings}

    # ==========================================================
    # STEP 3: Targeted Adaptation
    # ==========================================================
    def step3_adapt(self, max_iterations: int = None) -> nn.Module:
        """Design and execute targeted adaptation strategy.

        Two-stage approach:
          Stage 1: BN reset + entropy/consistency/diversity → coarse recovery
          Stage 2: No BN reset + pseudo-labels + anatomy → fine-class refinement
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations

        print("\n" + "=" * 60)
        print("STEP 3: Targeted Adaptation")
        print("=" * 60)

        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        self.adaptation_plan = self._design_adaptation_plan()
        plan = self.adaptation_plan
        print(f"\n  Adaptation Plan:")
        print(f"    Rationale: {plan.rationale}")
        print(f"    Stage 1: lr={plan.learning_rate}, steps={plan.num_steps}, "
              f"iters={max_iterations}")
        print(f"    Stage 1 losses: {plan.loss_weights}")
        print(f"    Trainable scope: {plan.trainable_scope}")
        print(f"    Anatomy post-processing: {plan.use_anatomy_refinement}")

        self.adapted_model = copy.deepcopy(self.source_model)
        self.adapted_model.to(self.device)

        self._log_step("step3_adaptation_plan", {
            "learning_rate": plan.learning_rate,
            "num_steps": plan.num_steps,
            "loss_weights": plan.loss_weights,
            "trainable_scope": plan.trainable_scope,
            "use_anatomy_refinement": plan.use_anatomy_refinement,
            "rationale": plan.rationale,
        })

        # =============================================
        # Stage 1: BN recovery (BN reset every iter)
        # =============================================
        print("\n  === STAGE 1: BN Recovery + Entropy ===")
        best_q = 0.0
        best_state = copy.deepcopy(self.adapted_model.state_dict())

        for iteration in range(1, max_iterations + 1):
            print(f"\n  --- Stage 1, Iteration {iteration}/{max_iterations} ---")
            metrics = self._adapt_one_iteration(iteration)
            self.history.append({"iteration": iteration, **metrics})
            self._print_iteration_results(metrics)

            self._log_step(f"stage1_iteration_{iteration}", {
                **metrics, **self.history[-1],
            })

            # Track best model in Stage 1
            q_curr = self.history[-1]["quality_score"]
            if q_curr > best_q:
                best_q = q_curr
                best_state = copy.deepcopy(self.adapted_model.state_dict())

            # CGFD re-verification: track which failures are being resolved
            if self.cgfd_diagnosis is not None and iteration % 2 == 0:
                cgfd_update = self.cgfd.re_verify(
                    model=self.adapted_model,
                    target_images=self.target_images,
                    previous=self.cgfd_diagnosis,
                )
                self.history[-1]["cgfd_acs"] = cgfd_update["current_acs"]
                self.history[-1]["cgfd_acs_delta"] = cgfd_update["acs_delta"]

                if cgfd_update["acs_converged"] and iteration > 1:
                    print(f"    CGFD: ACS converged "
                          f"(score={cgfd_update['current_acs']:.3f}), "
                          f"stopping adaptation.")
                    break

            # Early stopping on quality degradation
            if len(self.history) >= 2:
                q_prev = self.history[-2]["quality_score"]
                if q_curr < q_prev - 0.02:
                    print(f"    Quality degraded ({q_prev:.4f} -> {q_curr:.4f}), "
                          f"reverting to best ({best_q:.4f}) and stopping Stage 1.")
                    self.adapted_model.load_state_dict(best_state)
                    break

        # =============================================
        # Stage 2: Anatomy-aware inference post-processing
        # =============================================
        # Instead of gradient-based refinement (conflicts with BN reset),
        # apply anatomy priors as inference-time post-processing
        if plan.use_anatomy_refinement and self.config.containment_rules:
            print(f"\n  === STAGE 2: Anatomy Post-Processing (no gradient updates) ===")
            self._evaluate_anatomy_postprocessing()

        return self.adapted_model

    def _evaluate_anatomy_postprocessing(self):
        """Evaluate the effect of anatomy post-processing on predictions.

        This applies containment rules and area constraints to correct
        class confusion (e.g., cup over-predicted as disc).
        """
        if self.target_labels is None:
            print("    No target labels for post-processing evaluation.")
            return

        from PIL import Image as PILImage
        model = self.adapted_model
        model.eval()
        cfg = self.config

        raw_dices = {name: [] for name in cfg.fg_classes}
        post_dices = {name: [] for name in cfg.fg_classes}

        with torch.no_grad():
            for img, lbl in zip(self.target_images, self.target_labels):
                tensor = self._img_to_tensor(img).to(self.device)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1).cpu().numpy()[0]
                probs_np = probs.cpu().numpy()[0]

                # Apply anatomy post-processing
                refined = self._postprocess_prediction(pred, probs_np)

                lbl_pil = PILImage.fromarray(lbl)
                lbl_pil = lbl_pil.resize((self.image_size, self.image_size),
                                          PILImage.NEAREST)
                lbl_r = np.array(lbl_pil)

                for cls_name, cls_idx in cfg.fg_classes.items():
                    if cls_idx == 1 and cfg.num_classes == 3:
                        raw_c = (pred >= 1).astype(float)
                        ref_c = (refined >= 1).astype(float)
                        gt_c = (lbl_r >= 1).astype(float)
                    else:
                        raw_c = (pred == cls_idx).astype(float)
                        ref_c = (refined == cls_idx).astype(float)
                        gt_c = (lbl_r == cls_idx).astype(float)

                    if gt_c.sum() > 0:
                        raw_dice = float(2 * (raw_c * gt_c).sum() /
                                         (raw_c.sum() + gt_c.sum() + 1e-8))
                        ref_dice = float(2 * (ref_c * gt_c).sum() /
                                         (ref_c.sum() + gt_c.sum() + 1e-8))
                        raw_dices[cls_name].append(raw_dice)
                        post_dices[cls_name].append(ref_dice)

        print("    Raw vs Post-processed Dice:")
        for cls_name in cfg.fg_classes:
            raw = np.mean(raw_dices[cls_name]) if raw_dices[cls_name] else 0
            post = np.mean(post_dices[cls_name]) if post_dices[cls_name] else 0
            diff = post - raw
            print(f"    {cls_name}: {raw:.4f} -> {post:.4f} ({diff:+.4f})")

        # Store for final results
        self.postprocess_results = {
            f"{name}_dice_raw": float(np.mean(raw_dices[name]))
            if raw_dices[name] else 0 for name in cfg.fg_classes
        }
        self.postprocess_results.update({
            f"{name}_dice_post": float(np.mean(post_dices[name]))
            if post_dices[name] else 0 for name in cfg.fg_classes
        })

    def _postprocess_prediction(self, pred: np.ndarray,
                                 probs: np.ndarray) -> np.ndarray:
        """Apply anatomy-aware post-processing to a single prediction.

        Corrects common errors:
        1. Cup over-prediction: if cup area > expected, shrink to central region
        2. Cup missing: recover from probability within disc
        3. Containment: ensure inner class is within outer class
        """
        cfg = self.config
        refined = pred.copy()

        # Step 1: Connected component filtering per class
        for cls_name, cls_idx in cfg.fg_classes.items():
            mask = (refined == cls_idx)
            if mask.sum() == 0:
                continue
            labeled, n_components = ndimage.label(mask)
            if n_components > 1:
                sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
                largest = np.argmax(sizes) + 1
                refined[mask & (labeled != largest)] = 0

        # Step 2: Apply containment rules
        for outer_name, relation, inner_name in cfg.containment_rules:
            outer_idx = cfg.structures.get(outer_name)
            inner_idx = cfg.structures.get(inner_name)
            if outer_idx is None or inner_idx is None:
                continue
            if relation != "contains":
                continue

            # The combined outer+inner region
            combined_region = (refined == outer_idx) | (refined == inner_idx)
            inner_mask = (refined == inner_idx)
            outer_only = combined_region & ~inner_mask

            # Step 2a: Inner pixels outside combined region -> reassign to background
            # (inner_mask is a subset of combined_region by construction,
            #  so this handles edge cases from prior refinement steps)
            isolated_inner = inner_mask & ~(ndimage.binary_dilation(outer_only, iterations=2) | outer_only)
            if isolated_inner.any():
                refined[isolated_inner] = 0

            # Step 2b: Fix inner class over-prediction
            # If inner area is unreasonably large relative to outer area
            inner_prior = cfg.anatomy_priors.get(inner_name, {})
            outer_prior = cfg.anatomy_priors.get(outer_name, {})
            inner_max_area = inner_prior.get("area_range", (0.005, 0.25))[1]

            inner_frac = inner_mask.sum() / max(pred.size, 1)
            combined_area = combined_region.sum()

            if combined_area > 0:
                inner_ratio = inner_mask.sum() / combined_area
                # Cup should typically be 20-70% of disc area (CDR)
                # If inner > 70% of combined, it's over-predicted
                if inner_ratio > 0.7:
                    # Shrink inner: keep only central part
                    ys, xs = np.where(combined_region)
                    cy, cx = ys.mean(), xs.mean()
                    dist = np.sqrt((np.arange(pred.shape[0])[:, None] - cy)**2 +
                                   (np.arange(pred.shape[1])[None, :] - cx)**2)
                    # Target inner to be ~40% of combined area
                    target_inner_pixels = int(0.4 * combined_area)
                    # Within combined region, keep the closest N pixels as inner
                    dist_in_region = np.full_like(dist, np.inf)
                    dist_in_region[combined_region] = dist[combined_region]
                    flat_dist = dist_in_region.ravel()
                    if target_inner_pixels > 0 and target_inner_pixels < flat_dist.size:
                        threshold = np.partition(flat_dist, target_inner_pixels
                                                 )[target_inner_pixels]
                        new_inner = combined_region & (dist <= threshold)
                        new_outer = combined_region & ~new_inner
                        refined[new_inner] = inner_idx
                        refined[new_outer] = outer_idx

                # Step 2c: Inner class missing or too small
                elif inner_ratio < 0.15 and combined_area > 100:
                    # Recover inner from central region of combined
                    ys, xs = np.where(combined_region)
                    cy, cx = ys.mean(), xs.mean()
                    dist = np.sqrt((np.arange(pred.shape[0])[:, None] - cy)**2 +
                                   (np.arange(pred.shape[1])[None, :] - cx)**2)
                    # Target ~35% of combined area
                    target_inner_pixels = int(0.35 * combined_area)
                    dist_in_region = np.full_like(dist, np.inf)
                    dist_in_region[combined_region] = dist[combined_region]
                    flat_dist = dist_in_region.ravel()
                    if target_inner_pixels > 0 and target_inner_pixels < flat_dist.size:
                        threshold = np.partition(flat_dist, target_inner_pixels
                                                 )[target_inner_pixels]
                        new_inner = combined_region & (dist <= threshold)
                        refined[new_inner] = inner_idx
                        refined[combined_region & ~new_inner] = outer_idx

        return refined

    def _print_iteration_results(self, metrics: Dict):
        """Print iteration results including GT eval if available."""
        print(f"    Loss: {metrics['loss']:.4f}")
        print(f"    Quality score: {metrics['quality_score']:.4f}")

        for cls_name, cls_q in metrics.get("per_class_quality", {}).items():
            print(f"    Q({cls_name}): {cls_q:.4f}")

        if self.target_labels is not None:
            eval_metrics = self._evaluate_on_target()
            self.history[-1].update(eval_metrics)
            for cls_name in self.config.fg_classes:
                key = f"{cls_name}_dice"
                print(f"    [GT] {cls_name}: {eval_metrics.get(key, 0):.4f}")

    def _design_adaptation_plan(self) -> AdaptationPlan:
        """Design adaptation strategy based on verified shift report.

        Decision logic (task-agnostic):
          - complete_failure -> aggressive BN reset + diversity
          - intensity/style shift -> BN adapt + consistency
          - scale shift -> consistency
          - has containment_rules -> anatomy refinement
          - always: entropy minimization
        """
        plan = AdaptationPlan()
        report = self.shift_report
        cfg = self.config
        rationale_parts = []

        # Base losses
        plan.loss_weights = {
            "entropy": 0.3,
            "consistency": 0.2,
            "diversity": 0.2,
            "bn_adapt": 0.3,
            "pseudo_label": 0.0,
            "anatomy": 0.0,
        }

        # Check for complete model failure
        is_total_failure = (
            report.mean_confidence > 0.99 and
            report.failure_type in ("miss", "confusion") and
            report.prediction_distribution.get("background", 0) > 0.99
        )

        if "complete_failure" in report.verified_shifts or is_total_failure:
            plan.learning_rate = 1e-3
            plan.num_steps = 500
            plan.loss_weights["diversity"] = 0.3
            plan.loss_weights["bn_adapt"] = 0.4
            plan.trainable_scope = "bn+decoder1"
            rationale_parts.append(
                "Model completely fails on target (predicts all background with "
                f"{report.mean_confidence:.1%} confidence). "
                "Aggressive BN reset + diversity regularization to break collapse."
            )

        if "intensity_shift" in report.verified_shifts:
            plan.loss_weights["bn_adapt"] = max(plan.loss_weights["bn_adapt"], 0.4)
            ch = report.channel_shifts
            rationale_parts.append(
                f"Intensity shift confirmed (R={ch.get('R',0):+.3f}, "
                f"G={ch.get('G',0):+.3f}, B={ch.get('B',0):+.3f}). "
                "BN stats correction critical."
            )

        if "style_shift" in report.verified_shifts:
            plan.loss_weights["consistency"] = max(plan.loss_weights["consistency"], 0.3)
            rationale_parts.append(
                f"Style shift confirmed (low_freq={report.low_freq_shift:.3f}). "
                "Consistency regularization strengthened."
            )

        if "scale_shift" in report.verified_shifts:
            plan.loss_weights["consistency"] = max(plan.loss_weights["consistency"], 0.3)
            rationale_parts.append(
                f"Scale shift confirmed (ratio={report.scale_ratio:.2f}). "
                "Multi-scale consistency added."
            )

        # Enable anatomy-aware inference post-processing if containment rules exist
        if cfg.containment_rules or cfg.anatomy_priors:
            plan.use_anatomy_refinement = True
            rationale_parts.append(
                f"Anatomy post-processing: {len(cfg.containment_rules)} containment "
                f"rules, {len(cfg.anatomy_priors)} shape priors applied at inference "
                f"time to correct class confusion (e.g., cup/disc boundary)."
            )

        # ============================================================
        # Pseudo-label denoising strategy selection
        # ============================================================
        # Agent selects denoising method based on diagnosis signals:
        #
        # halluc_ratio (from failure report):
        #   < 0.2  → predictions clean → "none" (no denoising needed)
        #   0.2-0.5 → moderate noise → "prototype" (feature-based filtering)
        #             If class_descriptions available → "vlm_proxy" (ProDe-style)
        #   > 0.5  → heavy noise → "both" (cascade: VLM gates + prototype filters)
        #            But if halluc > 0.7, even VLM denoising may not help much
        #
        # VLM denoising is only used when class_descriptions are available
        # in the TaskConfig (the agent needs text descriptions for CLIP).
        halluc_ratio = 0.0
        if hasattr(self, 'state') and self.state.failure_report:
            halluc_ratio = self.state.failure_report.type_distribution.get(
                "hallucination", 0.0)
        elif hasattr(self, 'failure_report') and self.failure_report:
            fr = self.failure_report
            if hasattr(fr, 'type_distribution') and fr.type_distribution:
                halluc_ratio = fr.type_distribution.get("hallucination", 0.0)

        has_class_descriptions = bool(
            getattr(cfg, 'class_descriptions', None) or
            getattr(cfg, 'class_names', None))

        if halluc_ratio > 0.5 and has_class_descriptions:
            plan.denoising_method = "both"
            rationale_parts.append(
                f"Denoising: VLM proxy + prototype cascade "
                f"(halluc {halluc_ratio:.0%}: heavy noise, "
                f"VLM gates class presence, prototype filters spatially)")
        elif halluc_ratio > 0.2 and has_class_descriptions:
            plan.denoising_method = "vlm_proxy"
            rationale_parts.append(
                f"Denoising: VLM proxy (ProDe-inspired, "
                f"halluc {halluc_ratio:.0%}: moderate noise, "
                f"CLIP gates class presence per image)")
        elif halluc_ratio > 0.2:
            plan.denoising_method = "prototype"
            rationale_parts.append(
                f"Denoising: prototype-based "
                f"(halluc {halluc_ratio:.0%}: feature similarity filtering)")
        else:
            plan.denoising_method = "none"

        # Feature augmentation regularization (SF(DA)²-inspired)
        # Use when: feature distance is high but predictions aren't
        # completely wrong — helps learn domain-invariant features.
        if halluc_ratio > 0.2 and halluc_ratio < 0.7:
            plan.use_feature_augment = True
            rationale_parts.append(
                "Feature augment: spectral perturbation regularization "
                "(SF(DA)²-inspired, encourages domain-invariant features)")

        if not rationale_parts:
            rationale_parts.append("Default strategy with entropy + consistency.")

        # Integrate CGFD-recommended strategy if available
        if self.cgfd_diagnosis and self.cgfd_diagnosis.recommended_strategy:
            cgfd_strategy = self.cgfd_diagnosis.recommended_strategy
            cgfd_weights = cgfd_strategy.get("loss_weights", {})

            # CGFD overrides loss weights where it has stronger signal
            for key, val in cgfd_weights.items():
                if key in plan.loss_weights and val > plan.loss_weights[key]:
                    plan.loss_weights[key] = val

            # CGFD may recommend deeper adaptation scope
            cgfd_scope = cgfd_strategy.get("trainable_scope", "")
            scope_depth = {"bn_only": 0, "bn+output": 1, "bn+decoder1": 2,
                          "bn+all_decoder": 3}
            if scope_depth.get(cgfd_scope, 0) > scope_depth.get(plan.trainable_scope, 0):
                plan.trainable_scope = cgfd_scope

            if cgfd_strategy.get("use_anatomy_refinement"):
                plan.use_anatomy_refinement = True

            cgfd_reasoning = cgfd_strategy.get("reasoning", [])
            if cgfd_reasoning:
                rationale_parts.append(
                    "CGFD: " + "; ".join(cgfd_reasoning))

        plan.rationale = " ".join(rationale_parts)
        return plan

    # ==========================================================
    # Norm utilities (task- and architecture-agnostic — delegate to norm_utils)
    # ==========================================================
    def _get_norm_param_names(self, model):
        """Get parameter names for all normalization layers (BN, IN, GN, LN)."""
        from medadaptagent.graph.norm_utils import iter_norm_modules
        names = set()
        for name, mod, _ in iter_norm_modules(model):
            for pname, _p in mod.named_parameters():
                names.add(f"{name}.{pname}")
        return names

    def _reset_norm_stats(self, model):
        """Reset running statistics on norm layers that have them (BN, and IN
        with track_running_stats=True). No-op on LN/GN which have no running
        stats — their affine params are handled via gradient updates instead.
        """
        from medadaptagent.graph.norm_utils import (
            iter_norm_modules, reset_norm_running_stats, count_norm_layers,
        )
        reset_norm_running_stats(model)
        counts = count_norm_layers(model)
        bn_count = counts.get("batchnorm", 0)
        in_count = counts.get("instancenorm", 0)
        ln_count = counts.get("layernorm", 0)

        # Architecture tag inferred from dominant norm type (kept for any
        # downstream code that reads self._architecture_type).
        if bn_count == 0 and in_count == 0 and ln_count > 0:
            self._architecture_type = "transformer"
        elif in_count > 0 and bn_count == 0:
            self._architecture_type = "instancenorm_net"
        else:
            self._architecture_type = "batchnorm_net"

        print(f"    Norm layers: {bn_count} BN, {in_count} IN, {ln_count} LN "
              f"-> arch type: {self._architecture_type}")

    def _select_trainable_params(self, model, scope: str) -> List[nn.Parameter]:
        """Select trainable parameters based on scope string.

        Architecture-agnostic scopes:
          "bn_only"        -> only norm layer params
          "bn+output"      -> norm + output/head layer
          "bn+decoder1"    -> norm + output + last decoder block
          "bn+all_decoder" -> norm + all decoder blocks

        Automatically detects architecture patterns:
          - Conv nets (UNet): dec*, up*, out_conv, final
          - ViT (SAM2, MedSAM2): blocks, attn, mlp, norm, mask_decoder
          - Swin UNETR: layers, downsample, encoder*, decoder*
          - nnUNet (STU-Net): seg_outputs, conv_blocks_context/localization
        """
        norm_param_names = self._get_norm_param_names(model)
        trainable = []

        # Detect architecture type from parameter names
        all_names = [n for n, _ in model.named_parameters()]
        name_str = " ".join(all_names)

        is_vit = any(k in name_str for k in ["blocks.", "attn.", "patch_embed",
                                               "mask_decoder", "image_encoder"])
        is_swin = any(k in name_str for k in ["layers.", "downsample.",
                                                "swinViT", "encoder1"])
        is_nnunet = any(k in name_str for k in ["conv_blocks_context",
                                                  "conv_blocks_localization",
                                                  "seg_outputs"])

        for name, param in model.named_parameters():
            param.requires_grad = False
            should_train = False

            # Norm params always trainable
            if name in norm_param_names:
                should_train = True

            # Output/head layer
            if scope in ("bn+output", "bn+decoder1", "bn+all_decoder"):
                # UNet patterns
                if "out_conv" in name or "final" in name or "head" in name:
                    should_train = True
                # ViT/SAM patterns
                if "output_hypernetworks" in name or "iou_prediction" in name:
                    should_train = True
                if "mask_decoder.output" in name:
                    should_train = True
                # Swin UNETR patterns
                if "out." in name or "conv_final" in name:
                    should_train = True
                # nnUNet patterns
                if "seg_outputs" in name:
                    should_train = True

            # Last decoder block
            if scope in ("bn+decoder1", "bn+all_decoder"):
                # UNet
                if "dec1" in name or "up1" in name or "decoder.0" in name:
                    should_train = True
                # ViT: last transformer block
                if is_vit:
                    if "mask_decoder" in name:
                        should_train = True
                # Swin UNETR: last decoder stage
                if is_swin:
                    if "decoder1" in name or "decoder5" in name:
                        should_train = True
                # nnUNet: last localization block
                if is_nnunet:
                    if "conv_blocks_localization.0" in name:
                        should_train = True

            # All decoder blocks
            if scope == "bn+all_decoder":
                # UNet
                if "dec" in name or "up" in name or "decoder" in name:
                    should_train = True
                # ViT: mask decoder + late image encoder blocks
                if is_vit:
                    if "mask_decoder" in name:
                        should_train = True
                    # Last few transformer blocks in image encoder
                    if "image_encoder.blocks." in name:
                        try:
                            block_idx = int(name.split("image_encoder.blocks.")[1].split(".")[0])
                            # Train last 4 blocks
                            total_blocks = max(
                                int(n.split("image_encoder.blocks.")[1].split(".")[0])
                                for n in all_names if "image_encoder.blocks." in n
                            ) + 1
                            if block_idx >= total_blocks - 4:
                                should_train = True
                        except (ValueError, IndexError):
                            pass
                # Swin UNETR: all decoder stages
                if is_swin:
                    if any(k in name for k in ["decoder", "out."]):
                        should_train = True
                # nnUNet: all localization blocks
                if is_nnunet:
                    if "conv_blocks_localization" in name or "tu" in name:
                        should_train = True

            if should_train:
                param.requires_grad = True
                trainable.append(param)

        return trainable

    # ==========================================================
    # Anatomy-aware pseudo-label refinement (task-agnostic)
    # ==========================================================
    def _refine_pseudo_labels(self, pred: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Refine pseudo-labels using anatomy priors and containment rules.

        Args:
            pred: (H, W) predicted label map.
            probs: (C, H, W) softmax probabilities.

        Returns:
            Refined (H, W) label map.
        """
        cfg = self.config
        refined = pred.copy()

        # 1. Per-class area constraints: remove implausibly small/large regions
        for cls_name, cls_idx in cfg.fg_classes.items():
            prior = cfg.anatomy_priors.get(cls_name, {})
            area_range = prior.get("area_range", (0.001, 0.9))
            min_area, max_area = area_range

            mask = (refined == cls_idx)
            area_frac = mask.sum() / mask.size

            if area_frac < min_area:
                # Too small -- probably noise. Remove this class.
                refined[mask] = 0
            elif area_frac > max_area:
                # Too large -- keep only high-confidence pixels
                cls_conf = probs[cls_idx]
                threshold = np.percentile(cls_conf[mask], 100 * (1 - max_area / area_frac))
                refined[mask & (cls_conf < threshold)] = 0

        # 2. Connected component filtering: keep largest component per class
        for cls_name, cls_idx in cfg.fg_classes.items():
            mask = (refined == cls_idx)
            if mask.sum() == 0:
                continue
            labeled, n_components = ndimage.label(mask)
            if n_components > 1:
                sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
                largest = np.argmax(sizes) + 1
                refined[mask & (labeled != largest)] = 0

        # 3. Containment rules: enforce structural relationships
        for outer_name, relation, inner_name in cfg.containment_rules:
            outer_idx = cfg.structures.get(outer_name)
            inner_idx = cfg.structures.get(inner_name)
            if outer_idx is None or inner_idx is None:
                continue

            if relation == "contains":
                # The "outer" region is defined as pixels predicted as either
                # outer or inner class (since inner is a sub-region of outer)
                outer_region = (refined == outer_idx) | (refined == inner_idx)
                inner_mask = (refined == inner_idx)
                outer_only = outer_region & ~inner_mask

                # Remove inner pixels that are spatially isolated from outer
                isolated_inner = inner_mask & ~(ndimage.binary_dilation(outer_only, iterations=2) | outer_only)
                if isolated_inner.any():
                    refined[isolated_inner] = outer_idx

                # Recover inner class if collapsed or too small
                inner_area = inner_mask.sum()
                inner_prior = cfg.anatomy_priors.get(inner_name, {})
                inner_min = inner_prior.get("area_range", (0.005, 0.25))[0]
                inner_frac = inner_area / max(pred.size, 1)

                if inner_frac < inner_min and outer_region.sum() > 0:
                    inner_prob = probs[inner_idx]

                    # Strategy 1: probability-based recovery
                    # Use adaptive threshold: top N% of inner_prob within outer
                    inner_within_outer = inner_prob * outer_region.astype(float)
                    if inner_within_outer.max() > 0.05:
                        # Target: inner class should be ~30-50% of outer area
                        target_inner_pixels = int(0.3 * outer_region.sum())
                        target_inner_pixels = max(target_inner_pixels, 10)
                        flat_probs = inner_within_outer.ravel()
                        threshold = np.partition(flat_probs, -target_inner_pixels
                                                 )[-target_inner_pixels]
                        threshold = max(threshold, 0.05)  # minimum threshold
                        candidate = outer_region & (inner_prob >= threshold)
                        if candidate.sum() > 0:
                            refined[candidate] = inner_idx

                    # Strategy 2: spatial centroid prior (inner is typically central)
                    if (refined == inner_idx).sum() == 0 and outer_region.sum() > 100:
                        ys, xs = np.where(outer_region)
                        cy, cx = ys.mean(), xs.mean()
                        # Distance from centroid
                        dist = np.sqrt((np.arange(pred.shape[0])[:, None] - cy)**2 +
                                       (np.arange(pred.shape[1])[None, :] - cx)**2)
                        max_dist = np.sqrt(outer_region.sum() / np.pi) * 0.6
                        central = outer_region & (dist < max_dist)
                        if central.sum() > 0:
                            refined[central] = inner_idx

        return refined

    # ==========================================================
    # Adaptation iteration
    # ==========================================================
    def _adapt_one_iteration(self, iteration: int, **kwargs) -> Dict:
        """Run one adaptation iteration.

        BN reset every iter + entropy/consistency/diversity
          -> breaks class collapse, recovers coarse segmentation
        Anatomy refinement applied as post-processing after all iterations.
        """
        # Per-iteration seed for reproducibility
        torch.manual_seed(42 + iteration)
        np.random.seed(42 + iteration)

        model = self.adapted_model
        plan = self.adaptation_plan
        target_tensors = self._prepare_target_batch()
        loss_weights = plan.loss_weights
        lr = plan.learning_rate
        num_steps = plan.num_steps

        # Normalization statistics adaptation
        if loss_weights.get("bn_adapt", 0) > 0:
            print("    Norm reset + target statistics estimation...")
            self._reset_norm_stats(model)

            arch_type = getattr(self, '_architecture_type', 'batchnorm_net')

            if arch_type in ("batchnorm_net", "instancenorm_net"):
                # BN/IN: run forward passes to estimate running stats on target
                model.train()
                with torch.no_grad():
                    for _ in range(5):  # 5 passes for stable stats
                        indices = np.random.permutation(len(target_tensors))
                        for start in range(0, len(indices), 16):
                            batch_idx = indices[start:start + 16]
                            batch = torch.stack(
                                [target_tensors[i] for i in batch_idx]
                            ).to(self.device)
                            model(batch)
            else:
                # Transformer (LayerNorm only): no running stats to estimate.
                # LayerNorm params will be adapted via gradient updates below.
                print("    Transformer arch: skipping BN estimation "
                      "(LayerNorm adapted via gradient updates)")

        # Check current prediction quality
        model.eval()
        with torch.no_grad():
            sample = torch.stack(target_tensors[:16]).to(self.device)
            sample_pred = model(sample).argmax(dim=1).cpu().numpy()
            fg_ratio = (sample_pred > 0).mean()
            print(f"    Predictions: {fg_ratio*100:.1f}% foreground")
            for cls_name, cls_idx in self.config.fg_classes.items():
                cls_ratio = (sample_pred == cls_idx).mean()
                print(f"      {cls_name} (class {cls_idx}): {cls_ratio*100:.1f}%")

        # Self-supervised gradient updates
        print(f"    Gradient updates (lr={lr}, steps={num_steps})...")
        model.train()
        trainable_params = self._select_trainable_params(model, plan.trainable_scope)

        if iteration == 1:
            n_trainable = sum(p.numel() for p in trainable_params)
            n_total = sum(p.numel() for p in model.parameters())
            print(f"    Trainable: {n_trainable:,} / {n_total:,} "
                  f"({100*n_trainable/n_total:.2f}%)")

        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        total_loss = 0
        n_steps = 0

        for step in range(num_steps):
            idx = np.random.choice(len(target_tensors),
                                   size=min(16, len(target_tensors)), replace=False)
            batch = torch.stack([target_tensors[i] for i in idx]).to(self.device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Entropy minimization
            w_ent = loss_weights.get("entropy", 0)
            if w_ent > 0:
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
                loss = loss + w_ent * entropy.mean()

            # Consistency regularization
            w_con = loss_weights.get("consistency", 0)
            if w_con > 0:
                batch_flip = batch.flip(-1)
                probs_flip = F.softmax(model(batch_flip), dim=1).flip(-1)
                loss = loss + w_con * F.mse_loss(probs, probs_flip.detach())

            # Diversity regularization (prevents class collapse)
            w_div = loss_weights.get("diversity", 0)
            if w_div > 0:
                mean_probs = probs.mean(dim=(0, 2, 3))
                uniform = torch.ones_like(mean_probs) / mean_probs.shape[0]
                loss = loss + w_div * F.kl_div(
                    mean_probs.log(), uniform, reduction="batchmean")

            # Pseudo-label supervision (weight controlled by plan)
            w_pl = loss_weights.get("pseudo_label", 0)
            if w_pl > 0 and hasattr(self, '_pseudo_labels') and self._pseudo_labels:
                pl_batch = torch.stack(
                    [self._pseudo_labels[i] for i in idx]
                ).to(self.device)
                max_prob = probs.max(dim=1).values
                confident = (max_prob > 0.5).float()
                ce = F.cross_entropy(logits, pl_batch, reduction="none")
                loss = loss + w_pl * (ce * confident).mean()

            # Anatomy containment loss (weight controlled by plan)
            w_anat = loss_weights.get("anatomy", 0)
            if w_anat > 0 and self.config.containment_rules:
                anat_loss = self._anatomy_containment_loss(probs)
                loss = loss + w_anat * anat_loss

            # Feature augmentation regularization (SF(DA)²-inspired)
            if plan.use_feature_augment and not hasattr(self, '_feat_augmenter'):
                from medadaptagent.adaptation.advanced_tta import FeatureAugmentRegularizer
                self._feat_augmenter = FeatureAugmentRegularizer(
                    perturbation_strength=0.1, low_freq_ratio=0.1)
            if plan.use_feature_augment and hasattr(self, '_feat_augmenter'):
                feat_aug_loss = self._feat_augmenter(model, batch)
                loss = loss + 0.1 * feat_aug_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1

            if (step + 1) % 50 == 0:
                print(f"    Step {step+1}/{num_steps}, loss={loss.item():.4f}")

        quality, per_class_q = self._compute_quality_score()
        return {"loss": total_loss / max(n_steps, 1),
                "quality_score": quality,
                "per_class_quality": per_class_q}

    def _generate_pseudo_labels(self, model, target_tensors) -> List[torch.Tensor]:
        """Generate anatomy-refined pseudo-labels for all target images."""
        model.eval()
        pseudo_labels = []
        n_refined = 0

        with torch.no_grad():
            for tensor in target_tensors:
                inp = tensor.unsqueeze(0).to(self.device)
                probs = F.softmax(model(inp), dim=1)
                pred = probs.argmax(dim=1).cpu().numpy()[0]
                probs_np = probs.cpu().numpy()[0]

                refined = self._refine_pseudo_labels(pred, probs_np)
                if not np.array_equal(pred, refined):
                    n_refined += 1
                pseudo_labels.append(torch.from_numpy(refined).long())

        print(f"    Refined {n_refined}/{len(target_tensors)} pseudo-labels "
              f"using anatomy priors")
        return pseudo_labels

    def _get_anatomy_class_distribution(self) -> torch.Tensor:
        """Build expected class distribution from anatomy priors.

        Instead of uniform [1/C, 1/C, ...], uses the midpoint of each class's
        expected area range from anatomy priors. Background gets the remainder.
        """
        cfg = self.config
        n_classes = cfg.num_classes
        dist = torch.zeros(n_classes)

        # Sum up expected foreground areas
        fg_total = 0.0
        for cls_name, cls_idx in cfg.fg_classes.items():
            prior = cfg.anatomy_priors.get(cls_name, {})
            area_range = prior.get("area_range", (0.01, 0.3))
            # Use geometric mean of range for expected area
            expected = (area_range[0] * area_range[1]) ** 0.5
            dist[cls_idx] = expected
            fg_total += expected

        # Background gets whatever is left (clamped to at least 0.3)
        dist[0] = max(1.0 - fg_total, 0.3)
        # Normalize to sum to 1
        dist = dist / dist.sum()
        return dist

    def _anatomy_containment_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """Soft containment loss: inner class probability should not exceed
        outer class probability outside the outer region.

        E.g., cup probability should be 0 wherever disc probability is 0.
        """
        loss = torch.tensor(0.0, device=self.device)
        n_rules = 0

        for outer_name, relation, inner_name in self.config.containment_rules:
            outer_idx = self.config.structures.get(outer_name)
            inner_idx = self.config.structures.get(inner_name)
            if outer_idx is None or inner_idx is None:
                continue
            if relation != "contains":
                continue

            # P(inner) should be 0 wherever P(outer) + P(inner) is low
            outer_prob = probs[:, outer_idx]
            inner_prob = probs[:, inner_idx]
            combined = outer_prob + inner_prob

            # Penalize inner probability outside the combined region
            violation = inner_prob * (1.0 - combined).clamp(min=0)
            loss = loss + violation.mean()
            n_rules += 1

        return loss / max(n_rules, 1)

    # ==========================================================
    # Data preparation (task-agnostic)
    # ==========================================================
    def _prepare_target_batch(self) -> List[torch.Tensor]:
        from PIL import Image as PILImage
        tensors = []
        for img in self.target_images:
            pil = PILImage.fromarray((img * 255).astype(np.uint8))
            pil = pil.resize((self.image_size, self.image_size), PILImage.BILINEAR)
            arr = np.array(pil).astype(np.float32) / 255.0
            # Handle grayscale (H,W) -> (3,H,W)
            if arr.ndim == 2:
                tensor = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
            else:
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(tensor)
        return tensors

    def _img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        from PIL import Image as PILImage
        pil = PILImage.fromarray((img * 255).astype(np.uint8))
        pil = pil.resize((self.image_size, self.image_size), PILImage.BILINEAR)
        arr = np.array(pil).astype(np.float32) / 255.0
        # Handle grayscale (H,W) -> (3,H,W)
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
        else:
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor.unsqueeze(0)

    # ==========================================================
    # Quality score (task-agnostic, annotation-free)
    # ==========================================================
    def _compute_quality_score(self) -> Tuple[float, Dict[str, float]]:
        """Annotation-free quality score using TTA consistency + anatomy priors.

        Returns: (overall_score, per_class_scores)
        """
        model = self.adapted_model
        model.eval()
        cfg = self.config

        per_class_consistencies = {name: [] for name in cfg.fg_classes}
        per_class_plausibility = {name: [] for name in cfg.fg_classes}
        confidences = []

        with torch.no_grad():
            for img in self.target_images[:50]:
                tensor = self._img_to_tensor(img).to(self.device)
                probs = F.softmax(model(tensor), dim=1)
                pred = probs.argmax(dim=1)
                confidences.append(float(probs.max(dim=1).values.mean()))

                # TTA consistency
                probs_flip = F.softmax(model(tensor.flip(-1)), dim=1).flip(-1)
                pred_flip = probs_flip.argmax(dim=1)

                for cls_name, cls_idx in cfg.fg_classes.items():
                    p1 = (pred == cls_idx).float()
                    p2 = (pred_flip == cls_idx).float()
                    if p1.sum() > 0 or p2.sum() > 0:
                        iou = (p1 * p2).sum() / ((p1 + p2 - p1 * p2).sum() + 1e-8)
                        per_class_consistencies[cls_name].append(float(iou))

                # Anatomical plausibility per class
                pred_np = pred.cpu().numpy()[0]
                for cls_name, cls_idx in cfg.fg_classes.items():
                    prior = cfg.anatomy_priors.get(cls_name, {})
                    area_range = prior.get("area_range", (0.001, 0.9))
                    area_frac = (pred_np == cls_idx).sum() / pred_np.size
                    plausible = area_range[0] <= area_frac <= area_range[1]
                    per_class_plausibility[cls_name].append(float(plausible))

                # Check containment rules
                for outer_name, relation, inner_name in cfg.containment_rules:
                    outer_idx = cfg.structures.get(outer_name)
                    inner_idx = cfg.structures.get(inner_name)
                    if outer_idx is None or inner_idx is None:
                        continue
                    if relation == "contains":
                        outer_only = (pred_np == outer_idx)
                        inner_mask = (pred_np == inner_idx)
                        if inner_mask.sum() > 0 and outer_only.sum() > 0:
                            # Check inner and outer are in same connected component
                            combined = outer_only | inner_mask
                            labeled, _ = ndimage.label(combined)
                            # For each inner pixel, check it shares a component with outer
                            inner_labels = set(np.unique(labeled[inner_mask])) - {0}
                            outer_labels = set(np.unique(labeled[outer_only])) - {0}
                            shared = inner_labels & outer_labels
                            # Fraction of inner in shared components
                            if inner_labels:
                                shared_mask = np.isin(labeled, list(shared)) & inner_mask
                                contained = shared_mask.sum() / inner_mask.sum()
                            else:
                                contained = 0.0
                            per_class_plausibility[inner_name][-1] *= float(contained)
                        elif inner_mask.sum() > 0 and outer_only.sum() == 0:
                            # Inner exists but outer doesn't — violation
                            per_class_plausibility[inner_name][-1] *= 0.0

        # Aggregate per-class scores
        per_class_q = {}
        for cls_name in cfg.fg_classes:
            cons = np.mean(per_class_consistencies[cls_name]) if per_class_consistencies[cls_name] else 0.0
            plaus = np.mean(per_class_plausibility[cls_name]) if per_class_plausibility[cls_name] else 0.0
            per_class_q[cls_name] = float(0.5 * cons + 0.5 * plaus)

        overall_consistency = np.mean(
            [v for vals in per_class_consistencies.values() for v in vals]
        ) if any(per_class_consistencies.values()) else 0.0
        overall_plausibility = np.mean(
            [v for vals in per_class_plausibility.values() for v in vals]
        ) if any(per_class_plausibility.values()) else 0.0
        confidence = np.mean(confidences)

        overall = float(0.4 * overall_consistency + 0.3 * confidence + 0.3 * overall_plausibility)
        return overall, per_class_q

    # ==========================================================
    # GT Evaluation (development only, task-agnostic)
    # ==========================================================
    def _evaluate_on_target(self) -> Dict:
        """Evaluate on target GT. NOT used in adaptation, only for development."""
        from PIL import Image as PILImage
        model = self.adapted_model
        model.eval()
        cfg = self.config

        per_class_dices = {name: [] for name in cfg.fg_classes}

        with torch.no_grad():
            for img, lbl in zip(self.target_images, self.target_labels):
                tensor = self._img_to_tensor(img).to(self.device)
                pred = model(tensor).argmax(dim=1).cpu().numpy()[0]

                lbl_pil = PILImage.fromarray(lbl)
                lbl_pil = lbl_pil.resize((self.image_size, self.image_size),
                                          PILImage.NEAREST)
                lbl_r = np.array(lbl_pil)

                for cls_name, cls_idx in cfg.fg_classes.items():
                    # For hierarchical classes (disc includes cup), handle merging
                    if cls_idx == 1 and cfg.num_classes == 3:
                        pred_c = (pred >= 1).astype(float)
                        gt_c = (lbl_r >= 1).astype(float)
                    else:
                        pred_c = (pred == cls_idx).astype(float)
                        gt_c = (lbl_r == cls_idx).astype(float)

                    if gt_c.sum() > 0:
                        dice = float(2 * (pred_c * gt_c).sum() /
                                     (pred_c.sum() + gt_c.sum() + 1e-8))
                        per_class_dices[cls_name].append(dice)

        result = {}
        all_dices = []
        for cls_name, dices in per_class_dices.items():
            mean_dice = float(np.mean(dices)) if dices else 0.0
            result[f"{cls_name}_dice"] = mean_dice
            all_dices.append(mean_dice)
        result["avg_dice"] = float(np.mean(all_dices)) if all_dices else 0.0
        return result

    # ==========================================================
    # Run full pipeline
    # ==========================================================
    def run(self, max_iterations: int = None) -> Dict:
        """Execute the full 3-step MedAdaptAgent pipeline."""
        t0 = time.time()

        knowledge = self.step1_domain_knowledge()
        shift_report = self.step2_verify_shift()
        adapted_model = self.step3_adapt(max_iterations=max_iterations)

        total_time = time.time() - t0

        print("\n" + "=" * 60)
        print("MedAdaptAgent Pipeline Complete")
        print("=" * 60)
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Verified shifts: {shift_report.verified_shifts}")
        if self.history:
            print(f"  Quality: {self.history[0]['quality_score']:.4f} -> "
                  f"{self.history[-1]['quality_score']:.4f}")
            for cls_name in self.config.fg_classes:
                key = f"{cls_name}_dice"
                if key in self.history[-1]:
                    print(f"  Final {cls_name} Dice: {self.history[-1][key]:.4f}")

        # Print CGFD summary
        if self.cgfd_diagnosis:
            acs_curve = self.cgfd.get_acs_curve()
            if len(acs_curve) >= 2:
                print(f"  ACS trajectory: {acs_curve[0]:.3f} -> {acs_curve[-1]:.3f}")
            print(f"  CGFD confirmed: {self.cgfd_diagnosis.confirmed_hypotheses}")
            print(f"  CGFD severity: {self.cgfd_diagnosis.overall_severity:.2f}")

        results = {
            "knowledge": knowledge,
            "shift_report": shift_report.summary(),
            "plan": self.adaptation_plan.rationale,
            "history": self.history,
            "total_time": total_time,
            "pipeline_log": self.pipeline_log,
            "cgfd_diagnosis": (self.cgfd_diagnosis.to_dict()
                              if self.cgfd_diagnosis else None),
            "acs_curve": self.cgfd.get_acs_curve(),
        }

        return results

    def save_pipeline_log(self, path: str):
        """Save the full pipeline log for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.pipeline_log, f, indent=2, default=str)
        print(f"Pipeline log saved to {path}")
