"""
Multi-Agent Orchestrator for MedAdaptAgent.

Coordinates 6 specialized agents through a shared PipelineState blackboard:
  1. KnowledgeAgent: RAG + literature + domain profiling
  2. AnalysisAgent: Statistical domain shift analysis
  3. VerificationAgent: Cross-references RAG predictions with statistics
  4. DiagnosisAgent: Failure detection + classification + LLM reasoning
  5. AdaptationAgent: BN reset + self-supervised losses + adapter selection
  6. EvaluationAgent: Quality scoring + metrics + post-processing

Usage:
    from medadaptagent.orchestrator import Orchestrator, PipelineState
    from medadaptagent.pipeline_config import TaskConfig

    state = PipelineState(
        config=config, source_model=model,
        source_images=src_imgs, target_images=tgt_imgs,
    )
    orchestrator = Orchestrator(state)
    results = orchestrator.run()
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

from medadaptagent.pipeline_config import (
    TaskConfig, STEP1_RAG_PROMPT, STEP2_VERIFY_PROMPT, STEP3_ADAPT_PROMPT,
)
from medadaptagent.diagnosis.domain_shift_analyzer import DomainShiftAnalyzer, ShiftReport
from medadaptagent.diagnosis.failure_classifier import FailureReport
from medadaptagent.adaptation.advanced_tta import (
    PrototypeDenoiser, extract_features_unet,
)
from medadaptagent.adaptation.few_shot import select_strategy


# ================================================================
# PipelineState: shared blackboard for all agents
# ================================================================
@dataclass
class PipelineState:
    """Shared state for multi-agent communication."""
    # --- Inputs ---
    config: TaskConfig = field(default_factory=TaskConfig)
    source_model: nn.Module = None
    source_images: List[np.ndarray] = field(default_factory=list)
    target_images: List[np.ndarray] = field(default_factory=list)
    source_labels: List[np.ndarray] = field(default_factory=list)
    target_labels: List[np.ndarray] = field(default_factory=list)
    llm_client: object = None

    # --- Phase 1: Knowledge ---
    rag_knowledge: Dict = field(default_factory=dict)
    domain_profile: Dict = field(default_factory=dict)

    # --- Phase 2: Analysis + Verification ---
    shift_report: ShiftReport = None
    verification: Dict = field(default_factory=dict)

    # --- Phase 3: Diagnosis ---
    failure_report: FailureReport = None
    failure_maps: List = field(default_factory=list)
    llm_recommendation: Dict = field(default_factory=dict)

    # --- Phase 3.5: Data Adaptation ---
    data_adaptation_applied: bool = False
    data_adaptation_summary: Dict = field(default_factory=dict)
    adapted_target_images: List[np.ndarray] = field(default_factory=list)

    # --- Phase 4: Adaptation ---
    adapted_model: nn.Module = None
    adaptation_plan: Dict = field(default_factory=dict)
    iteration_history: List[Dict] = field(default_factory=list)
    best_model_state: Dict = None
    best_quality_score: float = 0.0
    mean_teacher: object = None  # MeanTeacher instance
    prototype_denoiser: object = None  # PrototypeDenoiser instance

    # --- Phase 5: Evaluation ---
    quality_scores: List[float] = field(default_factory=list)
    postprocess_results: Dict = field(default_factory=dict)

    # --- Pipeline log ---
    pipeline_log: Dict = field(default_factory=lambda: {"steps": []})

    def log_step(self, step_name: str, data: Dict):
        self.pipeline_log["steps"].append({
            "step": step_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **data,
        })


# ================================================================
# Agent base class
# ================================================================
class BaseAgent:
    """Base class for pipeline agents."""
    def __init__(self, state: PipelineState):
        self.state = state
        self.config = state.config

    def run(self):
        raise NotImplementedError


# ================================================================
# Agent 1: KnowledgeAgent
# ================================================================
class KnowledgeAgent(BaseAgent):
    """Gathers domain knowledge from structured knowledge base + LLM reasoning.

    Produces fine-grained shift characterization instead of coarse categories:
      - intensity_mapping_type: linear / non_linear / fundamentally_different
      - standardization_gap: none / mild / moderate / severe
      - noise_model_change: same / similar / different
      - contrast_mechanism_change: same / similar / different / fundamentally_different
      - boundary_quality_change: sharper / same / blurrier
      - dominant_shift_factor: which factor matters most
      - recommended_scope / mode / lr from literature evidence
    """

    def run(self):
        print("\n" + "=" * 60)
        print("AGENT 1: Knowledge Collection (RAG + Domain Profiling)")
        print("=" * 60)

        # Step 1: Retrieve structured knowledge from local knowledge base
        kb_entry = self._retrieve_from_knowledge_base()

        # Step 2: If no KB match AND LLM available, search literature online
        if kb_entry is None and self.state.llm_client:
            print("  [KB] No local match. Searching PubMed for relevant literature...")
            try:
                from medadaptagent.knowledge.literature_search import (
                    search_and_synthesize,
                )
                online_knowledge = search_and_synthesize(
                    source_domain=self.config.source_domain,
                    target_domain=self.config.target_domain,
                    task_name=self.config.task_name,
                    llm_client=self.state.llm_client,
                )
                if online_knowledge:
                    print("  [Literature Search] Successfully synthesized knowledge "
                          "from online literature.")
                    # Use online knowledge as if it were a KB entry
                    kb_entry = {"shift_characteristics": online_knowledge}
            except Exception as e:
                print(f"  [Literature Search] Failed: {e}")

        # Step 3: LLM reasoning (if available) enriched with retrieved knowledge
        if self.state.llm_client:
            knowledge = self._llm_knowledge(kb_entry)
        else:
            knowledge = self._rule_based_knowledge(kb_entry)

        # Ensure anatomy priors from config are included
        if "anatomy_priors" not in knowledge:
            knowledge["anatomy_priors"] = {}
        for cls, prior in self.config.anatomy_priors.items():
            if cls not in knowledge["anatomy_priors"]:
                knowledge["anatomy_priors"][cls] = prior
        knowledge["containment_rules"] = self.config.containment_rules

        print(f"  Task: {self.config.get_task_description()}")
        print(f"  Shift types: {knowledge.get('shift_types', [])}")
        print(f"  Intensity mapping: {knowledge.get('intensity_mapping_type', 'unknown')}")
        print(f"  Standardization gap: {knowledge.get('standardization_gap', 'unknown')}")
        print(f"  Contrast mechanism: {knowledge.get('contrast_mechanism_change', 'unknown')}")
        print(f"  Dominant shift: {knowledge.get('dominant_shift_factor', 'unknown')}")
        print(f"  Literature scope: {knowledge.get('recommended_scope', 'unknown')}")
        print(f"  Literature mode: {knowledge.get('recommended_mode', 'unknown')}")

        self.state.rag_knowledge = knowledge
        self.state.log_step("knowledge_agent", {
            "method": "llm" if self.state.llm_client else "rule_based",
            "kb_matched": kb_entry is not None,
            "knowledge": knowledge,
        })

    def _retrieve_from_knowledge_base(self) -> Optional[Dict]:
        """Retrieve structured knowledge entry matching current task."""
        kb_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "rag", "domain_knowledge.json"
        )
        if not os.path.exists(kb_path):
            print("  [KB] Knowledge base not found, using defaults.")
            return None

        try:
            with open(kb_path, "r") as f:
                kb = json.load(f)
        except Exception as e:
            print(f"  [KB] Failed to load knowledge base: {e}")
            return None

        # Match by task_name first (exact match)
        task_name = self.config.task_name if hasattr(self.config, 'task_name') else ""
        for key, entry in kb.items():
            if key == "_meta":
                continue
            if key == task_name or task_name.startswith(key):
                print(f"  [KB] Matched knowledge entry: {key}")
                if "dataset_paper" in entry:
                    print(f"  [KB] Paper: {entry['dataset_paper'][:100]}")
                return entry

        # Fuzzy match by source/target domain keywords
        src_low = self.config.source_domain.lower()
        tgt_low = self.config.target_domain.lower()
        best_match = None
        best_score = 0
        for key, entry in kb.items():
            if key == "_meta":
                continue
            score = 0
            entry_str = json.dumps(entry).lower()
            for word in src_low.split() + tgt_low.split():
                if len(word) > 2 and word in entry_str:
                    score += 1
            if score > best_score:
                best_score = score
                best_match = (key, entry)

        if best_match and best_score >= 3:
            print(f"  [KB] Fuzzy matched knowledge entry: {best_match[0]} (score={best_score})")
            return best_match[1]

        print("  [KB] No matching knowledge entry found.")
        return None

    def _llm_knowledge(self, kb_entry: Optional[Dict]) -> Dict:
        """Use LLM with in-context learning for calibrated knowledge generation.

        Uses 4 exemplar KB entries that teach the LLM:
        1. How to calibrate severity (most shifts are NOT severe)
        2. What fails for each shift type (most important signal)
        3. When to recommend conservative vs aggressive adaptation
        """
        from medadaptagent.knowledge.icl_prompt import ICL_KNOWLEDGE_PROMPT

        # If we have a KB entry, include it as additional context
        additional = ""
        if kb_entry:
            additional = (
                "## Retrieved knowledge base entry (for reference)\n"
                "This entry was found in the local knowledge base. Use it as a "
                "starting point but apply your own analysis.\n```json\n"
                + json.dumps({
                    k: v for k, v in kb_entry.items()
                    if k in ("shift_characteristics", "literature_evidence",
                             "recommended_plan_adjustments")
                }, indent=2) + "\n```"
            )

        prompt = ICL_KNOWLEDGE_PROMPT.format(
            task_description=self.config.get_task_description(),
            source_domain=self.config.source_domain,
            target_domain=self.config.target_domain,
            class_descriptions=json.dumps(self.config.class_descriptions, indent=2),
            additional_context=additional,
        )
        try:
            model_name = getattr(self.state.llm_client, '_model_name', 'claude-sonnet-4-20250514')
            response = self.state.llm_client.messages.create(
                model=model_name,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])

                # LLM provides enriched understanding; no KB override needed
                # because the LLM's ICL prompt already teaches calibration.

                return result
        except Exception as e:
            print(f"  LLM call failed ({e}), falling back to rule-based.")
        return self._rule_based_knowledge(kb_entry)

    def _rule_based_knowledge(self, kb_entry: Optional[Dict] = None) -> Dict:
        """Build knowledge from structured KB entry + heuristic rules."""
        cfg = self.config

        # If we have a KB entry, extract fine-grained signals directly
        if kb_entry and "shift_characteristics" in kb_entry:
            sc = kb_entry["shift_characteristics"]
            rpa = kb_entry.get("recommended_plan_adjustments", {})
            lit = kb_entry.get("literature_evidence", {})

            # Build shift types from shift characteristics
            shift_types = []
            if sc.get("contrast_mechanism") in ("fundamentally_different", "different"):
                shift_types.append(
                    f"Cross-modality: {sc.get('contrast_mechanism_detail', 'different imaging physics')}")
            if sc.get("intensity_mapping") == "non_linear":
                shift_types.append(
                    f"Non-linear intensity mapping: {sc.get('intensity_mapping_detail', '')}")
            elif sc.get("intensity_mapping") == "approximately_linear":
                shift_types.append(
                    f"Approximately linear intensity shift: {sc.get('intensity_mapping_detail', '')}")
            if sc.get("noise_model_change") not in ("same", "similar", None):
                shift_types.append(
                    f"Noise model change: {sc.get('noise_model_detail', '')}")
            if sc.get("resolution_gap") not in ("none", "mild", None):
                shift_types.append(
                    f"Resolution gap: {sc.get('resolution_gap_detail', '')}")
            if sc.get("dominant_shift"):
                shift_types.append(f"Dominant: {sc['dominant_shift']}")

            strategies = lit.get("what_works", [])
            what_fails = lit.get("what_fails", [])
            challenges = []
            for paper in lit.get("key_papers", []):
                challenges.append(f"[{paper.get('ref', '')}] {paper.get('finding', '')}")

            # Parse recommended scope/mode/lr from KB
            # Check recommended_plan_adjustments first, then literature_evidence
            rec_scope = "bn+output"
            rec_text = rpa.get("trainable_scope", "").lower()
            if not rec_text and lit.get("recommended_scope"):
                rec_text = lit["recommended_scope"].lower()
            if rec_text.startswith("bn_only") or rec_text.startswith("bn-only"):
                rec_scope = "bn_only"
            elif rec_text.startswith("bn+decoder1"):
                rec_scope = "bn+decoder1"
            elif "bn+output" in rec_text:
                rec_scope = "bn+output"

            rec_mode = "batch_multi_loss"
            mode_text = rpa.get("adaptation_mode", "")
            if not mode_text and lit.get("recommended_mode"):
                mode_text = lit["recommended_mode"]
            if "per-image" in mode_text.lower() or "per_image" in mode_text.lower():
                rec_mode = "per_image_tent"
            elif "batch" in mode_text.lower():
                rec_mode = "batch_multi_loss"

            rec_lr = "conservative_5e4"
            lr_text = rpa.get("learning_rate", "")
            if "1e-3" in lr_text or "standard" in lr_text.lower():
                rec_lr = "standard_1e3"
            elif "3e-4" in lr_text or "very_conservative" in lr_text.lower():
                rec_lr = "very_conservative_3e4"

            return {
                "source_domain": cfg.source_domain,
                "target_domain": cfg.target_domain,
                "task": cfg.get_task_description(),
                "dataset_paper": kb_entry.get("dataset_paper", ""),
                "shift_types": shift_types,
                "intensity_mapping_type": sc.get("intensity_mapping", "unknown"),
                "standardization_gap": sc.get("standardization_gap", "unknown"),
                "noise_model_change": sc.get("noise_model_change", "unknown"),
                "contrast_mechanism_change": sc.get("contrast_mechanism", "unknown"),
                "boundary_quality_change": sc.get("boundary_sharpness_change", "unknown"),
                "dominant_shift_factor": sc.get("dominant_shift",
                    sc.get("contrast_mechanism_detail", "unknown")),
                "appearance_reversals": sc.get("appearance_reversals", []),
                "challenges": challenges,
                "literature_strategies": strategies,
                "what_will_fail": what_fails,
                "recommended_scope": rec_scope,
                "recommended_mode": rec_mode,
                "recommended_lr_range": rec_lr,
            }

        # Fallback: heuristic keyword matching (produces same fine-grained
        # fields so design_plan can use them uniformly)
        src_low = cfg.source_domain.lower()
        tgt_low = cfg.target_domain.lower()
        combined = src_low + " " + tgt_low

        shift_types = []
        intensity_mapping = "approximately_linear"
        standardization_gap = "moderate"
        contrast_mechanism = "same"
        noise_change = "similar"
        boundary_change = "same"
        dominant = "intensity distribution shift"
        rec_scope = "bn_only"
        rec_mode = "per_image_tent"
        rec_lr = "standard_1e3"

        # Cross-modality detection
        modalities_found = set()
        for mod in ["ct", "mri", "t1", "t2", "ultrasound", "us ", "x-ray", "xray"]:
            if mod in combined:
                modalities_found.add(mod.strip())
        if len(modalities_found) >= 2 or (
                "ct" in combined and any(m in combined for m in ["mri", "t2", "t1"])):
            shift_types.append("Cross-modality: fundamentally different imaging physics")
            intensity_mapping = "non_linear"
            standardization_gap = "severe"
            contrast_mechanism = "fundamentally_different"
            noise_change = "different"
            boundary_change = "blurrier"
            dominant = "Different contrast mechanism (imaging physics)"
            rec_scope = "bn+output"
            rec_mode = "batch_multi_loss"
            rec_lr = "conservative_5e4"

        elif any(w in combined for w in ["device", "camera", "scanner", "hospital",
                                          "vendor", "site", "center", "institution"]):
            shift_types.append("Cross-device: different acquisition hardware")
            dominant = "Scanner/camera calibration differences"

        shift_types.append("Intensity/contrast distribution shift")

        return {
            "source_domain": cfg.source_domain,
            "target_domain": cfg.target_domain,
            "task": cfg.get_task_description(),
            "shift_types": shift_types,
            "intensity_mapping_type": intensity_mapping,
            "standardization_gap": standardization_gap,
            "noise_model_change": noise_change,
            "contrast_mechanism_change": contrast_mechanism,
            "boundary_quality_change": boundary_change,
            "dominant_shift_factor": dominant,
            "appearance_reversals": [],
            "challenges": [
                "Source-trained BN statistics mismatched to target distribution",
                "Model may predict all background due to distribution mismatch",
            ],
            "literature_strategies": [
                "BatchNorm statistics adaptation",
                "Entropy minimization",
                "Consistency regularization",
            ],
            "what_will_fail": [],
            "recommended_scope": rec_scope,
            "recommended_mode": rec_mode,
            "recommended_lr_range": rec_lr,
        }


# ================================================================
# Agent 2: AnalysisAgent
# ================================================================
class AnalysisAgent(BaseAgent):
    """Statistical domain shift analysis."""

    def run(self):
        print("\n" + "=" * 60)
        print("AGENT 2: Statistical Domain Shift Analysis")
        print("=" * 60)

        analyzer = DomainShiftAnalyzer(device=self.config.device)
        class_names = list(self.config.structures.keys())

        self.state.shift_report = analyzer.analyze(
            source_images=self.state.source_images,
            target_images=self.state.target_images,
            source_labels=self.state.source_labels if self.state.source_labels else None,
            target_labels=self.state.target_labels if self.state.target_labels else None,
            model=self.state.source_model,
            image_size=self.config.image_size,
            num_classes=self.config.num_classes,
            class_names=class_names,
        )

        print("\n" + self.state.shift_report.summary())
        self.state.log_step("analysis_agent", {
            "shift_report": self.state.shift_report.summary(),
            "verified_shifts": self.state.shift_report.verified_shifts,
        })


# ================================================================
# Agent 3: VerificationAgent
# ================================================================
class VerificationAgent(BaseAgent):
    """Cross-references RAG predictions with statistical findings."""

    def run(self):
        print("\n" + "=" * 60)
        print("AGENT 3: Verification (RAG vs Statistics)")
        print("=" * 60)

        report = self.state.shift_report
        rag_knowledge = self.state.rag_knowledge

        # If LLM available, use STEP2 prompt for deeper verification
        if self.state.llm_client and report:
            verification = self._llm_verify(report, rag_knowledge)
        else:
            verification = self._rule_based_verify(report, rag_knowledge)

        self.state.verification = verification
        self.state.log_step("verification_agent", verification)

    def _llm_verify(self, report: ShiftReport, rag_knowledge: Dict) -> Dict:
        prompt = STEP2_VERIFY_PROMPT.format(
            task_description=self.config.get_task_description(),
            rag_knowledge=json.dumps(rag_knowledge, indent=2, default=str),
            intensity_kl=report.intensity_shift,
            channel_shifts=json.dumps(report.channel_shifts),
            low_freq=report.low_freq_shift,
            high_freq=report.high_freq_shift,
            scale_ratio=report.scale_ratio,
            failure_type=report.failure_type,
            confidence=report.mean_confidence,
            entropy=report.mean_entropy,
            pred_dist=json.dumps(report.prediction_distribution),
        )
        try:
            model_name = getattr(self.state.llm_client, '_model_name', 'claude-sonnet-4-20250514')
            response = self.state.llm_client.messages.create(
                model=model_name,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                print(f"  LLM verification: {len(result.get('confirmed_shifts', []))} confirmed, "
                      f"{len(result.get('new_findings', []))} new findings")
                # Add verified severity assessment
                result["verified_severity"] = self._assess_verified_severity(
                    report, rag_knowledge)
                return result
        except Exception as e:
            print(f"  LLM verification failed ({e}), using rule-based.")
        return self._rule_based_verify(report, rag_knowledge)

    def _rule_based_verify(self, report: ShiftReport, rag_knowledge: Dict) -> Dict:
        rag_shifts = rag_knowledge.get("shift_types", [])
        verified = report.verified_shifts if report else []

        keyword_map = {
            "intensity": "intensity_shift", "color": "intensity_shift",
            "contrast": "intensity_shift", "exposure": "intensity_shift",
            "field-of-view": "scale_shift", "resolution": "scale_shift",
            "scale": "scale_shift",
            "device": ["style_shift", "texture_shift"],
            "camera": ["style_shift", "texture_shift"],
            "scanner": ["style_shift", "texture_shift"],
            "modality": ["style_shift", "texture_shift", "intensity_shift"],
            "style": "style_shift", "texture": "texture_shift",
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
                print(f"    [NEW FINDING] {finding}")
                new_findings.append(finding)

        # ---- Verified severity assessment ----
        # Reconcile knowledge-expected severity with statistical metrics.
        # Knowledge provides the "prior" (what SHOULD happen given the physics),
        # statistics provide the "evidence" (what IS happening).
        verified_severity = self._assess_verified_severity(report, rag_knowledge)

        return {"confirmed": confirmed, "unverified": unverified,
                "new_findings": new_findings,
                "verified_severity": verified_severity}

    def _assess_verified_severity(self, report: ShiftReport, rag_knowledge: Dict) -> Dict:
        """Reconcile knowledge-expected severity with statistical measurements.

        Returns a dict with:
          - severity_score: float 0-1, the reconciled severity
          - severity_level: str, "mild"/"moderate"/"high"
          - scope_recommendation: str, what scope the verified severity suggests
          - mode_recommendation: str, "per_image_tent" or "batch_multi_loss"
          - rationale: str, explanation of how severity was determined
        """
        # --- Knowledge prior ---
        intensity_mapping = rag_knowledge.get("intensity_mapping_type", "unknown")
        std_gap = rag_knowledge.get("standardization_gap", "unknown")
        contrast_change = rag_knowledge.get("contrast_mechanism_change", "unknown")
        lit_scope = rag_knowledge.get("recommended_scope", "unknown")
        lit_mode = rag_knowledge.get("recommended_mode", "unknown")

        # Map knowledge to a prior severity score
        if contrast_change == "fundamentally_different":
            knowledge_severity = 0.9
        elif intensity_mapping == "non_linear" or std_gap == "severe":
            knowledge_severity = 0.75
        elif contrast_change == "different" or std_gap == "moderate":
            knowledge_severity = 0.5
        elif intensity_mapping == "approximately_linear" and std_gap in ("mild", "none"):
            knowledge_severity = 0.2
        else:
            knowledge_severity = 0.4  # unknown -> moderate prior

        # --- Statistical evidence ---
        intensity_kl = getattr(report, 'intensity_shift', 0) or 0
        style_shift = getattr(report, 'low_freq_shift', 0) or 0
        texture_shift = getattr(report, 'high_freq_shift', 0) or 0
        scale_ratio = getattr(report, 'scale_ratio', 1.0) or 1.0
        scale_diff = abs(1.0 - scale_ratio)
        mean_entropy = getattr(report, 'mean_entropy', 0) or 0

        # Statistical severity from shift metrics
        stat_severity = min(1.0,
            intensity_kl * 1.0 + style_shift * 1.5 +
            texture_shift * 1.0 + scale_diff * 1.5)

        # Failure-based evidence from Diagnosis Agent
        fr = self.state.failure_report
        fd = fr.type_distribution if fr and hasattr(fr, 'type_distribution') else {}
        miss_ratio = fd.get("miss", 0)
        halluc_ratio = fd.get("hallucination", 0)

        # Prediction distribution (are predictions reasonable?)
        fg_ratio = sum(v for k, v in report.prediction_distribution.items()
                       if k != "background") if report.prediction_distribution else 0

        # Prediction-based classifier severity:
        # This measures whether the model produces reasonable output on target,
        # NOT how bad the failures are. Use prediction distribution + entropy.
        # Low fg + high entropy = model confused = high severity.
        # Reasonable fg + low entropy = model works = low severity.
        # BUT: high hallucination overrides — model produces fg but it's WRONG.
        mean_conf = getattr(report, 'mean_confidence', 0.5) or 0.5
        if fg_ratio < 0.005:
            # No foreground predicted — severe failure
            classifier_severity = 0.9
        elif fg_ratio < 0.02 and mean_entropy > 0.1:
            classifier_severity = 0.7
        else:
            # Model produces foreground. Is it correct or hallucinated?
            # Blend hallucination signal with confidence proxy continuously.
            # When halluc_ratio is high, confidence is misleading (confidently wrong).
            # When halluc_ratio is low, confidence is a valid proxy.
            conf_proxy = max(0.0, 1.0 - mean_conf) if mean_conf > 0.9 else 0.3
            # Continuous blend: as halluc_ratio grows, trust halluc signal more
            classifier_severity = (
                halluc_ratio * max(0.5, halluc_ratio) +
                (1 - halluc_ratio) * conf_proxy
            )

        # --- Reconciliation logic ---
        # The key insight: knowledge tells us WHAT KIND of shift this is,
        # which determines HOW TO INTERPRET the statistical metrics.
        #
        # Case 1: Knowledge says "mild" (e.g., cross-device same modality)
        #   -> High miss_ratio likely just BN mismatch, NOT structural failure
        #   -> BN reset alone should fix it; wider scope risks overfitting
        #   -> Dampen classifier severity toward knowledge prior
        #
        # Case 2: Knowledge says "high" (e.g., cross-modality CT->MR)
        #   -> High miss_ratio confirms genuinely severe shift
        #   -> Trust classifier severity, may even increase
        #
        # Case 3: Knowledge says "moderate" but statistics say "mild"
        #   -> Shift is less severe than expected; trust statistics
        #   -> Use lighter adaptation

        rationale_parts = []

        if knowledge_severity <= 0.3:
            # Knowledge says mild shift
            if classifier_severity > 0.5:
                # Statistics seem harsh, but knowledge says the physics gap
                # is small. The high miss_ratio is likely just BN mismatch.
                # Weight knowledge heavily — BN reset should fix it.
                verified_score = 0.3 * classifier_severity + 0.7 * knowledge_severity
                rationale_parts.append(
                    f"Knowledge says mild shift (k={knowledge_severity:.2f}) "
                    f"but classifier reports {classifier_severity:.2f}. "
                    f"High misses likely BN-fixable -> dampened to {verified_score:.2f}.")
            else:
                # Knowledge and classifier both say mild/low.
                # When classifier_severity is near 0, the model already works
                # well on target — trust that signal. Raw stat_severity can be
                # misleadingly high (texture/scale metrics fire even when
                # segmentation is fine). Weight classifier + knowledge more.
                if classifier_severity < 0.15:
                    # Model works well on target — this is the strongest "do no harm" signal
                    verified_score = 0.6 * knowledge_severity + 0.2 * classifier_severity + 0.2 * stat_severity
                    rationale_parts.append(
                        f"Knowledge says mild ({knowledge_severity:.2f}), "
                        f"classifier confirms model works well ({classifier_severity:.2f}). "
                        f"Stats ({stat_severity:.2f}) may overestimate shift.")
                else:
                    verified_score = 0.5 * stat_severity + 0.5 * knowledge_severity
                    rationale_parts.append(
                        f"Both knowledge ({knowledge_severity:.2f}) and stats "
                        f"({stat_severity:.2f}) agree: mild shift.")

        elif knowledge_severity >= 0.7:
            # Knowledge says severe shift
            if classifier_severity > 0.5 or fg_ratio < 0.01:
                # Confirmed severe — trust the higher of the two
                verified_score = max(classifier_severity, knowledge_severity)
                rationale_parts.append(
                    f"Knowledge ({knowledge_severity:.2f}) and classifier "
                    f"({classifier_severity:.2f}) confirm severe shift.")
            else:
                # Knowledge says severe but model seems okay (produces foreground).
                # BUT: check for hallucination — model may be confidently WRONG.
                # High hallucination ratio means the foreground predictions are
                # false positives, not evidence that the model works.
                if halluc_ratio > 0.5:
                    # Model hallucinates confidently — trust knowledge severity.
                    # The low classifier_severity is misleading because the
                    # hallucination detector (entropy/TTA) may not flag
                    # consistent false positives.
                    verified_score = max(knowledge_severity, 0.7)
                    rationale_parts.append(
                        f"Knowledge says severe ({knowledge_severity:.2f}), "
                        f"model produces fg ({fg_ratio:.1%}) but diagnosis "
                        f"reports {halluc_ratio:.0%} hallucination. "
                        f"Confident predictions are WRONG -> trust knowledge.")
                else:
                    # Genuinely reasonable predictions — cautiously moderate
                    verified_score = 0.5 * classifier_severity + 0.5 * knowledge_severity
                    rationale_parts.append(
                        f"Knowledge says severe ({knowledge_severity:.2f}) but "
                        f"model predictions look reasonable (fg={fg_ratio:.1%}). "
                        f"Moderate approach.")
        else:
            # Knowledge says moderate — average with statistics
            verified_score = 0.4 * classifier_severity + 0.3 * stat_severity + 0.3 * knowledge_severity
            rationale_parts.append(
                f"Moderate knowledge prior ({knowledge_severity:.2f}), "
                f"blended with classifier ({classifier_severity:.2f}) "
                f"and stats ({stat_severity:.2f}).")

        verified_score = max(0.0, min(1.0, verified_score))

        # --- Determine severity level ---
        if verified_score < 0.25:
            severity_level = "minimal"
        elif verified_score < 0.4:
            severity_level = "mild"
        elif verified_score < 0.65:
            severity_level = "moderate"
        else:
            severity_level = "high"

        # --- Scope recommendation from verified severity ---
        # This is the KEY output: verified severity drives scope, not raw miss_ratio.
        # Literature scope is given strong weight — if literature explicitly says
        # "bn_only" for this shift type, only override for genuinely high severity.
        if lit_scope != "unknown":
            # Literature has explicit guidance — trust it unless verified
            # severity is dramatically different from what literature expects.
            if severity_level in ("minimal", "mild"):
                # Mild/minimal verified severity -> use literature scope
                scope_rec = lit_scope
                mode_rec = lit_mode if lit_mode != "unknown" else "per_image_tent"
            elif severity_level == "moderate":
                # Moderate severity — if literature says bn_only but severity
                # is moderate, still trust literature (the "moderate" is likely
                # from statistical noise, not genuine structural failure)
                if lit_scope == "bn_only" and verified_score < 0.55:
                    scope_rec = "bn_only"
                    mode_rec = lit_mode if lit_mode != "unknown" else "per_image_tent"
                    rationale_parts.append(
                        f"Literature says bn_only; moderate severity "
                        f"({verified_score:.2f}) not high enough to override.")
                else:
                    scope_rec = lit_scope
                    mode_rec = lit_mode if lit_mode != "unknown" else "batch_multi_loss"
            else:  # high
                # High severity — literature scope may be insufficient
                if lit_scope == "bn_only":
                    scope_rec = "bn+output"
                    mode_rec = "batch_multi_loss"
                    rationale_parts.append(
                        "High verified severity overrides literature bn_only -> bn+output.")
                else:
                    scope_rec = lit_scope
                    mode_rec = "batch_multi_loss"
        else:
            # No literature guidance — use severity level directly
            if severity_level == "minimal":
                scope_rec = "bn_only"
                mode_rec = "per_image_tent"
            elif severity_level == "mild":
                scope_rec = "bn_only"
                mode_rec = "per_image_tent"
            elif severity_level == "moderate":
                scope_rec = "bn+output"
                mode_rec = "batch_multi_loss"
            else:  # high
                scope_rec = "bn+output"
                mode_rec = "batch_multi_loss"

        result = {
            "severity_score": verified_score,
            "severity_level": severity_level,
            "knowledge_severity": knowledge_severity,
            "stat_severity": stat_severity,
            "classifier_severity": classifier_severity,
            "scope_recommendation": scope_rec,
            "mode_recommendation": mode_rec,
            "rationale": " ".join(rationale_parts),
        }

        print(f"\n  Verified Severity Assessment:")
        print(f"    Knowledge prior: {knowledge_severity:.2f}")
        print(f"    Statistical:     {stat_severity:.2f}")
        print(f"    Classifier:      {classifier_severity:.2f}")
        print(f"    -> Verified:     {verified_score:.2f} ({severity_level})")
        print(f"    -> Scope rec:    {scope_rec}")
        print(f"    -> Mode rec:     {mode_rec}")
        print(f"    Rationale: {result['rationale']}")

        return result


# ================================================================
# Agent 4: DiagnosisAgent
# ================================================================
class DiagnosisAgent(BaseAgent):
    """Failure detection, classification, and LLM-based diagnosis."""

    def run(self):
        print("\n" + "=" * 60)
        print("AGENT 4: Failure Diagnosis")
        print("=" * 60)

        model = self.state.adapted_model or self.state.source_model

        # Run failure detection on a subset of target images
        sample_images = self.state.target_images[:min(20, len(self.state.target_images))]

        try:
            from medadaptagent.diagnosis.failure_detector import detect_failures
            from medadaptagent.diagnosis.failure_classifier import classify_batch

            print("  Running failure detection (entropy/confidence/TTA)...")
            failure_maps = detect_failures(
                model, sample_images,
                image_size=self.config.image_size,
                n_tta=4,  # Fewer augmentations for speed
            )
            self.state.failure_maps = failure_maps

            print("  Classifying failure types (A/B/C/D)...")
            failure_report = classify_batch(
                predictions=[fm.prediction for fm in failure_maps],
                entropy_maps=[fm.entropy_map for fm in failure_maps],
                confidence_maps=[fm.confidence_map for fm in failure_maps],
                tta_variances=[fm.tta_variance for fm in failure_maps],
            )
            self.state.failure_report = failure_report

            print(f"  Failure distribution: {failure_report.type_distribution}")
            print(f"  Overall severity: {failure_report.overall_severity:.3f}")
            print(f"  Recommended adapters: {failure_report.recommended_adapters}")

        except Exception as e:
            print(f"  Failure detection error ({e}), using shift report instead.")
            # Fallback: construct minimal failure report from shift report
            report = self.state.shift_report
            if report:
                self.state.failure_report = FailureReport(
                    type_distribution={
                        "miss": 0.8 if report.failure_type == "miss" else 0.1,
                        "hallucination": 0.1,
                        "boundary_error": 0.05,
                        "class_confusion": 0.05,
                    },
                    overall_severity=0.7,
                    recommended_adapters=["style", "semantic"],
                    summary=f"Fallback from shift report: {report.failure_type}",
                )

        # LLM diagnosis if available
        if self.state.llm_client and self.state.failure_report:
            self._llm_diagnose()

        self.state.log_step("diagnosis_agent", {
            "failure_report": self.state.failure_report.to_dict()
            if self.state.failure_report else {},
            "llm_recommendation": self.state.llm_recommendation,
        })

    def _llm_diagnose(self):
        try:
            from medadaptagent.diagnosis.llm_diagnostician import LLMDiagnostician
            diagnostician = LLMDiagnostician(
                api_key=getattr(self.state.llm_client, '_api_key', None),
                use_rule_fallback=True,
            )
            domain_profile = {
                "source_modality": self.config.source_domain,
                "target_modality": self.config.target_domain,
                "body_region": self.config.task_name,
                "task_description": self.config.get_task_description(),
                "shift_type": ", ".join(self.state.shift_report.verified_shifts)
                if self.state.shift_report else "",
            }
            recommendation = diagnostician.diagnose(
                failure_report=self.state.failure_report.to_dict(),
                domain_profile=domain_profile,
                knowledge_base=self.state.rag_knowledge,
            )
            self.state.llm_recommendation = recommendation
            print(f"  LLM recommendation: {recommendation.get('reasoning', '')[:200]}")
        except Exception as e:
            print(f"  LLM diagnosis error: {e}")


# ================================================================
# Agent 5: AdaptationAgent
# ================================================================
class AdaptationAgent(BaseAgent):
    """Runs adaptation iterations: BN reset + self-supervised losses."""

    def __init__(self, state: PipelineState):
        super().__init__(state)
        self._target_tensors = None

    def run_iteration(self, iteration: int) -> Dict:
        """Run one adaptation iteration."""
        from scipy import ndimage

        print(f"\n  --- Adaptation Iteration {iteration + 1} ---")

        # Per-iteration seed
        torch.manual_seed(42 + iteration)
        np.random.seed(42 + iteration)

        model = self.state.adapted_model
        plan = self.state.adaptation_plan

        if self._target_tensors is None:
            self._target_tensors = self._prepare_target_batch()

        target_tensors = self._target_tensors
        # Copy loss_weights so we can disable entropy on later iterations
        # without mutating the shared plan dict.
        loss_weights = dict(plan.get("loss_weights", {}))
        # Disable entropy loss after the first iteration. Entropy minimization is
        # a one-shot tool: running it for hundreds of steps over multiple
        # iterations causes the model's predictions to become artificially
        # confident (low entropy / high quality_score) while the GT dice
        # actually decays (Q-vs-GT divergence observed in R20 brats/fundus_roi).
        if iteration > 0 and loss_weights.get("entropy", 0) > 0:
            print(f"    Entropy loss disabled (iter {iteration + 1} > 1) to avoid Q-vs-GT divergence")
            loss_weights["entropy"] = 0
        lr = plan.get("learning_rate", 1e-4)
        num_steps = plan.get("num_steps", 200)
        trainable_scope = plan.get("trainable_scope", "bn+output")

        # BN statistics adaptation
        # Skip BN reset on first iteration if FDA warmup just ran —
        # FDA already set up BN stats via target forward passes.
        fda_just_ran = getattr(self.state, '_fda_warmup_done', False) and iteration == 0
        if fda_just_ran:
            print("    Skipping BN reset (FDA warmup already adapted BN stats).")
            self.state._fda_warmup_done = False  # only skip once
        elif loss_weights.get("bn_adapt", 0) > 0:
            print("    Norm reset + target statistics estimation...")
            self._reset_norm_stats(model)
            model.train()
            with torch.no_grad():
                for _ in range(5):
                    indices = np.random.permutation(len(target_tensors))
                    for start in range(0, len(indices), 16):
                        batch_idx = indices[start:start + 16]
                        batch = torch.stack(
                            [target_tensors[i] for i in batch_idx]
                        ).to(self.config.device)
                        model(batch)

        # Check predictions
        model.eval()
        with torch.no_grad():
            sample = torch.stack(target_tensors[:16]).to(self.config.device)
            sample_pred = model(sample).argmax(dim=1).cpu().numpy()
            fg_ratio = (sample_pred > 0).mean()
            print(f"    Predictions: {fg_ratio*100:.1f}% foreground")
            for cls_name, cls_idx in self.config.fg_classes.items():
                cls_ratio = (sample_pred == cls_idx).mean()
                print(f"      {cls_name} (class {cls_idx}): {cls_ratio*100:.1f}%")

        # Self-supervised gradient updates
        if num_steps == 0:
            print("    Skipping gradient updates (minimal / BN-stats-only mode).")
            metrics = {"loss": 0.0, "iteration": iteration}
            self.state.iteration_history.append(metrics)
            return metrics

        # Tent-style: use batch stats instead of running stats during updates.
        # `tent_norm_affine_mode` is the canonical name; `tent_bn_mode` is the
        # legacy alias kept for backward compatibility. `disable_running_stats`
        # dispatches on norm type — BN/IN with tracking get switched to batch
        # stats; LN/GN (no running stats) are silently untouched.
        tent_mode = plan.get("tent_norm_affine_mode", plan.get("tent_bn_mode", False))
        if tent_mode:
            from medadaptagent.graph.norm_utils import disable_running_stats
            disable_running_stats(model)

        model.train()
        trainable_params = self._select_trainable_params(model, trainable_scope)

        if iteration == 0:
            n_trainable = sum(p.numel() for p in trainable_params)
            n_total = sum(p.numel() for p in model.parameters())
            print(f"    Trainable: {n_trainable:,} / {n_total:,} "
                  f"({100*n_trainable/n_total:.2f}%)")

        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        total_loss = 0
        n_steps = 0

        # ---- Per-image Tent mode ----
        # When tent_bn_mode is active, do per-image sequential updates
        # (1 gradient step per image, 1 epoch) — matches Tent exactly.
        # This prevents overfitting on small datasets where batch multi-step
        # sees each image dozens of times.
        if tent_mode:
            n_images = len(target_tensors)
            n_epochs = max(1, num_steps // max(n_images, 1))
            n_epochs = min(n_epochs, 3)  # cap at 3 passes to prevent overfitting
            # For large datasets, cap total per-image updates to prevent
            # over-adaptation. 200 updates is roughly equivalent to Tent on
            # a moderately-sized dataset. Beyond that, reduce to a single
            # pass over a random subset.
            max_updates_per_epoch = min(n_images, 200)
            print(f"    Per-image Tent updates (lr={lr}, "
                  f"{n_epochs} epoch(s) x {max_updates_per_epoch} images "
                  f"[of {n_images} total])...")
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                indices = np.random.permutation(n_images)[:max_updates_per_epoch]
                for i, idx in enumerate(indices):
                    img = target_tensors[idx].unsqueeze(0).to(self.config.device)
                    logits = model(img)
                    probs = F.softmax(logits, dim=1)
                    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()

                    optimizer.zero_grad()
                    entropy.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                    epoch_loss += entropy.item()
                    n_steps += 1

                avg = epoch_loss / max_updates_per_epoch
                total_loss += epoch_loss
                print(f"    Epoch {epoch+1}/{n_epochs}: avg entropy = {avg:.4f}")
        else:
            # ---- Batch multi-step mode ----
            # Initialize MeanTeacher if requested (EMA of student for stable predictions)
            # Runtime gate: only useful when (a) we have multiple iterations to let
            # EMA stabilize, and (b) source predictions have enough foreground that
            # the teacher will not just memorize background (collapse case).
            use_mt = plan.get("use_mean_teacher", False)
            mt_max_iter = int(plan.get("max_iterations", 1))
            mt_fg_ok = (fg_ratio >= 0.025)
            mt_iter_ok = (mt_max_iter >= 2)
            if use_mt and not (mt_fg_ok and mt_iter_ok):
                reasons = []
                if not mt_iter_ok:
                    reasons.append(f"max_iter={mt_max_iter}<2")
                if not mt_fg_ok:
                    reasons.append(f"fg={fg_ratio:.1%}<2.5%")
                print(f"    MeanTeacher SKIPPED ({', '.join(reasons)})")
                use_mt = False
            if use_mt and self.state.mean_teacher is None:
                from medadaptagent.adaptation.advanced_tta import MeanTeacher
                self.state.mean_teacher = MeanTeacher(model, alpha=0.999)
                print("    MeanTeacher initialized (EMA alpha=0.999)")
            teacher = self.state.mean_teacher if use_mt else None

            # Pseudo-label generation gate: only run if pseudo_label loss has weight,
            # source predictions have enough foreground (else 99% bg labels reinforce failure),
            # and we have not already generated them this run.
            w_pl_check = loss_weights.get("pseudo_label", 0)
            if w_pl_check > 0 and not (hasattr(self, "_pseudo_labels") and self._pseudo_labels):
                # Per-dataset pseudo-label gate: require source fg_ratio to be at
                # least 50% of the dataset's expected minimum foreground fraction.
                # Without this, brats (expected ~30% fg, model produces ~5%) would
                # generate pseudo-labels that are >99% background and reinforce
                # the under-segmentation collapse.
                _da = getattr(self.state, "rag_knowledge", None) or {}
                _fg_priors = _da.get("foreground_priors", {})
                _exp_area = _fg_priors.get("expected_area_fraction", [0.05, 0.5])
                if isinstance(_exp_area, (list, tuple)) and len(_exp_area) >= 1:
                    _exp_min = float(_exp_area[0])
                else:
                    _exp_min = 0.05
                pl_threshold = max(0.025, 0.5 * _exp_min)
                if fg_ratio >= pl_threshold:
                    try:
                        orch_ref = getattr(self, "orchestrator", None)
                        if orch_ref is None:
                            raise RuntimeError("AdaptationAgent has no orchestrator backref")
                        orch_ref._generate_filtered_pseudo_labels()
                    except Exception as e:
                        print(f"    Pseudo-label generation failed ({e}), zeroing pl loss")
                        loss_weights["pseudo_label"] = 0
                else:
                    print(f"    Pseudo-label SKIPPED (fg={fg_ratio:.1%} < {pl_threshold:.1%} = 0.5*expected_min)")
                    loss_weights["pseudo_label"] = 0

            print(f"    Gradient updates (lr={lr}, steps={num_steps})...")
            for step in range(num_steps):
                idx = np.random.choice(len(target_tensors),
                                       size=min(16, len(target_tensors)), replace=False)
                batch = torch.stack([target_tensors[i] for i in idx]).to(self.config.device)
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                loss = torch.tensor(0.0, device=self.config.device, requires_grad=True)

                # Entropy minimization
                w_ent = loss_weights.get("entropy", 0)
                if w_ent > 0:
                    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
                    loss = loss + w_ent * entropy.mean()

                # Consistency regularization
                w_con = loss_weights.get("consistency", 0)
                if w_con > 0:
                    if teacher is not None:
                        # Teacher-student consistency (more stable than flip)
                        with torch.no_grad():
                            teacher_probs = F.softmax(teacher.teacher(batch), dim=1)
                        loss = loss + w_con * F.mse_loss(probs, teacher_probs.detach())
                    else:
                        # Flip-consistency fallback
                        batch_flip = batch.flip(-1)
                        probs_flip = F.softmax(model(batch_flip), dim=1).flip(-1)
                        loss = loss + w_con * F.mse_loss(probs, probs_flip.detach())

                # Diversity regularization
                w_div = loss_weights.get("diversity", 0)
                if w_div > 0:
                    mean_probs = probs.mean(dim=(0, 2, 3))
                    uniform = torch.ones_like(mean_probs) / mean_probs.shape[0]
                    loss = loss + w_div * F.kl_div(
                        mean_probs.log(), uniform, reduction="batchmean")

                # Pseudo-label supervision
                w_pl = loss_weights.get("pseudo_label", 0)
                if w_pl > 0 and hasattr(self, '_pseudo_labels') and self._pseudo_labels:
                    pl_batch = torch.stack(
                        [self._pseudo_labels[i] for i in idx]
                    ).to(self.config.device)
                    # Use ignore_index=255 for filtered pseudo-labels
                    # (uncertain pixels are set to 255 during generation)
                    pl_ce = F.cross_entropy(
                        logits, pl_batch, ignore_index=255, reduction="mean")
                    loss = loss + w_pl * pl_ce

                # Anatomy containment loss
                w_anat = loss_weights.get("anatomy", 0)
                if w_anat > 0 and self.config.containment_rules:
                    anat_loss = self._anatomy_containment_loss(probs)
                    loss = loss + w_anat * anat_loss

                # FDA mixed training: supervised loss on style-transferred source
                fda_mix_w = plan.get("fda_mix_weight", 0)
                if fda_mix_w > 0 and hasattr(self, '_fda_images') and self._fda_images:
                    fda_idx = np.random.choice(
                        len(self._fda_images),
                        size=min(8, len(self._fda_images)), replace=False)
                    fda_batch = torch.stack(
                        [self._fda_images[i] for i in fda_idx]
                    ).to(self.config.device)
                    fda_lbls = torch.stack(
                        [self._fda_labels[i] for i in fda_idx]
                    ).to(self.config.device)
                    fda_logits = model(fda_batch)
                    fda_ce = F.cross_entropy(fda_logits, fda_lbls)
                    loss = loss + fda_mix_w * fda_ce

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_steps += 1

                # Update EMA teacher
                if teacher is not None:
                    teacher.update(model)

                log_interval = 50 if num_steps > 100 else 10
                if (step + 1) % log_interval == 0 or step + 1 == num_steps:
                    print(f"    Step {step+1}/{num_steps}, loss={loss.item():.4f}")

        metrics = {"loss": total_loss / max(n_steps, 1), "iteration": iteration}
        self.state.iteration_history.append(metrics)
        return metrics

    def design_plan(self) -> Dict:
        """Design adaptation strategy from all agent outputs.

        If LLM is available, the LLM reasons about the diagnosis and
        designs the plan (STEP3_ADAPT_PROMPT). The LLM's JSON output
        is parsed and used directly. Falls back to rule-based on failure.

        If no LLM, uses rule-based composition:
          Knowledge Agent -> sets prior difficulty expectation
          Analysis Agent  -> quantifies shift dimensions
          Verification    -> filters confirmed vs unverified shifts
          Diagnosis Agent -> identifies failure modes -> loss weights
        """
        # LLM handles knowledge + diagnosis (STEP 1-2).
        # Plan design (STEP 3) uses rule-based logic because:
        # - Safety-critical decisions (DO NO HARM, hallucination guard)
        #   need empirically validated thresholds
        # - LLM tends to be overconfident with adaptation aggressiveness
        # - Rule-based plan uses LLM-enriched knowledge as input
        #
        # The LLM's contribution flows through rag_knowledge and
        # verification, which inform the rule-based plan design.
        report = self.state.shift_report
        cfg = self.config
        rationale_parts = []

        # Defaults
        plan = {
            "learning_rate": 1e-4,
            "num_steps": 100,
            "loss_weights": {
                "entropy": 0.3, "consistency": 0.2, "diversity": 0.2,
                "bn_adapt": 0.3, "pseudo_label": 0.0, "anatomy": 0.0,
            },
            "trainable_scope": "bn+output",
            "tent_bn_mode": False,
            "skip_adaptation": False,
            "use_anatomy_refinement": False,
        }

        if report is None:
            plan["rationale"] = "No shift report available, using defaults."
            return plan

        # ================================================================
        # STEP 1: Extract signals from all agents
        # ================================================================

        # --- From Knowledge Agent (fine-grained shift characterization) ---
        knowledge = self.state.rag_knowledge or {}
        intensity_mapping = knowledge.get("intensity_mapping_type", "unknown")
        std_gap = knowledge.get("standardization_gap", "unknown")
        contrast_change = knowledge.get("contrast_mechanism_change", "unknown")
        noise_change = knowledge.get("noise_model_change", "unknown")
        boundary_change = knowledge.get("boundary_quality_change", "unknown")
        dominant_shift = knowledge.get("dominant_shift_factor", "unknown")
        lit_scope = knowledge.get("recommended_scope", "unknown")
        lit_mode = knowledge.get("recommended_mode", "unknown")
        lit_lr = knowledge.get("recommended_lr_range", "unknown")
        what_fails = knowledge.get("what_will_fail", [])

        # --- From Analysis Agent ---
        intensity_kl = getattr(report, 'intensity_shift', 0) or 0
        style_shift = getattr(report, 'low_freq_shift', 0) or 0
        texture_shift = getattr(report, 'high_freq_shift', 0) or 0
        scale_ratio = getattr(report, 'scale_ratio', 1.0) or 1.0
        scale_diff = abs(1.0 - scale_ratio)
        mean_entropy = getattr(report, 'mean_entropy', 0) or 0

        # --- From Verification Agent ---
        verified = report.verified_shifts if report else []
        has_intensity = "intensity_shift" in verified
        has_style = "style_shift" in verified
        has_texture = "texture_shift" in verified
        has_scale = "scale_shift" in verified
        is_complete_failure = "complete_failure" in verified
        n_verified = len([s for s in verified if s != "complete_failure"])

        # --- From Diagnosis Agent ---
        fr = self.state.failure_report
        fd = {}
        classifier_severity = 0.0
        if fr:
            classifier_severity = fr.overall_severity
            fd = fr.type_distribution if hasattr(fr, 'type_distribution') else {}
        miss_ratio = fd.get("miss", 0)
        halluc_ratio = fd.get("hallucination", 0)
        boundary_ratio = fd.get("boundary_error", 0)
        confusion_ratio = fd.get("class_confusion", 0)
        recommended_adapters = fr.recommended_adapters if fr else []

        # --- Derived signals ---
        fg_ratio = sum(v for k, v in report.prediction_distribution.items()
                       if k != "background") if report.prediction_distribution else 0
        n_target = len(self.state.target_images) if self.state.target_images else 50
        is_total_failure = (
            report.mean_confidence > 0.99 and
            report.failure_type in ("miss", "confusion") and
            report.prediction_distribution.get("background", 0) > 0.99
        )

        # ---- Verified severity from VerificationAgent ----
        # This reconciles knowledge expectations with statistical metrics.
        # It prevents raw classifier severity from overriding literature
        # recommendations (e.g., miss_ratio>0.5 shouldn't force bn+output
        # when knowledge says "mild cross-device shift, bn_only sufficient").
        vs = (self.state.verification or {}).get("verified_severity", {})
        severity = vs.get("severity_score", classifier_severity)
        severity_level = vs.get("severity_level", "moderate")
        vs_scope_rec = vs.get("scope_recommendation", None)
        vs_mode_rec = vs.get("mode_recommendation", None)
        vs_rationale = vs.get("rationale", "")
        # Verification agent's classifier_severity (prediction-based, not failure-based)
        vs_classifier_severity = vs.get("classifier_severity", classifier_severity)

        # Fallback: if no verified severity, compute from raw metrics
        shift_severity = min(1.0,
            intensity_kl * 1.0 + style_shift * 1.5 +
            texture_shift * 1.0 + scale_diff * 1.5)
        if not vs:
            if classifier_severity < 0.1:
                severity = shift_severity
            else:
                severity = classifier_severity

        rationale_parts.append(
            f"[Severity] verified={severity:.2f} ({severity_level}). "
            f"{vs_rationale}" if vs else
            f"[Severity] raw classifier={classifier_severity:.2f}, "
            f"shift={shift_severity:.2f} (no verified severity).")

        # ================================================================
        # STEP 2: Knowledge Agent -> physics-informed difficulty assessment
        # ================================================================
        # Instead of a simple 3-level category, use fine-grained signals
        # from the knowledge base to set prior expectations.

        if contrast_change in ("fundamentally_different",):
            prior_difficulty = "high"
            rationale_parts.append(
                f"[Knowledge] Fundamentally different contrast mechanism "
                f"({dominant_shift}). Expect large distribution gap.")
        elif intensity_mapping == "non_linear" or std_gap == "severe":
            prior_difficulty = "high"
            rationale_parts.append(
                f"[Knowledge] Non-linear intensity mapping / severe "
                f"standardization gap. Expect large distribution gap.")
        elif contrast_change == "different" or std_gap == "moderate":
            prior_difficulty = "moderate"
            rationale_parts.append(
                f"[Knowledge] {dominant_shift}. "
                f"Moderate distribution gap expected.")
        elif intensity_mapping == "approximately_linear" and std_gap in ("mild", "none"):
            prior_difficulty = "mild"
            rationale_parts.append(
                f"[Knowledge] Linear intensity mapping, mild standardization gap. "
                f"Same contrast mechanism -> mild distribution gap.")
        else:
            prior_difficulty = "moderate"
            rationale_parts.append(
                f"[Knowledge] {dominant_shift}.")

        # Literature-informed warnings
        if what_fails:
            rationale_parts.append(
                f"[Literature] Known failures: {'; '.join(what_fails[:2])}")

        # ================================================================
        # STEP 3: Handle complete failure (overrides everything)
        # ================================================================
        if is_complete_failure or is_total_failure:
            plan["learning_rate"] = 1e-3
            plan["num_steps"] = 200
            plan["loss_weights"] = {
                "entropy": 0.2, "consistency": 0.2, "diversity": 0.3,
                "bn_adapt": 0.3, "pseudo_label": 0.0, "anatomy": 0.0,
            }
            plan["trainable_scope"] = "bn+decoder1"
            plan["tent_bn_mode"] = False
            rationale_parts.append(
                "[Plan] COMPLETE FAILURE: aggressive multi-loss + decoder scope.")
            if cfg.containment_rules or cfg.anatomy_priors:
                plan["use_anatomy_refinement"] = True

            # FDA warmup is critical for cross-modality complete failures:
            # the model produces near-zero foreground precisely because the
            # imaging physics are fundamentally different. FDA bridges the
            # low-frequency appearance gap before iterative adaptation.
            has_source = bool(self.state.source_images and self.state.source_labels)
            if contrast_change == "fundamentally_different" and has_source:
                plan["use_fda_warmup"] = True
                fda_beta = np.clip(
                    0.01 + np.sqrt(style_shift) * 0.04, 0.01, 0.06)
                texture_ratio = texture_shift / max(style_shift + 0.01, 0.01)
                if texture_ratio > 1.5:
                    fda_scope = "bn+output"
                elif texture_ratio < 0.5 and style_shift > 0.2:
                    fda_scope = "bn+decoder1"
                else:
                    fda_scope = "bn+output"
                total_shift = intensity_kl + style_shift + texture_shift + 0.01
                fda_coverage = style_shift / total_shift
                fda_mix_weight = np.clip(fda_coverage * 0.6, 0.1, 0.5)
                fda_steps = int(np.clip(total_shift * 100, 50, 200))
                fda_lr = 3e-4 if fda_scope == "bn+decoder1" else 5e-4
                plan["fda_beta"] = float(fda_beta)
                plan["fda_finetune_steps"] = fda_steps
                plan["fda_finetune_lr"] = fda_lr
                plan["fda_scope"] = fda_scope
                plan["use_fda_mixed_training"] = True
                plan["fda_mix_weight"] = float(fda_mix_weight)
                rationale_parts.append(
                    f"[FDA] Cross-modality complete failure: beta={fda_beta:.3f}, "
                    f"scope={fda_scope}, steps={fda_steps}.")

            plan["rationale"] = " ".join(rationale_parts)
            return plan

        # ================================================================
        # STEP 3b: Preprocessing decision (histogram matching)
        # ================================================================
        # Principle: histogram matching maps target CDF to source CDF.
        # This works when both domains share intensity semantics (same tissue
        # type -> similar relative intensity), i.e., intra-modality shifts
        # like T1->T2 where tissues exist in both but with different contrast.
        #
        # It FAILS for cross-modality (e.g., CT->MR) because the imaging
        # physics are fundamentally different — bone is bright in CT but
        # dark in MR, so CDF mapping produces semantically wrong results.
        #
        # Decision: use histogram matching only when:
        # 1. Same modality, different contrast (contrast == "different")
        # 2. Model is hallucinating (needs preprocessing help)
        # 3. Model produces some foreground (not complete failure)
        if (contrast_change == "different" and
                halluc_ratio > 0.5 and
                fg_ratio > 0.005):
            plan["use_histogram_matching"] = True
            rationale_parts.append(
                f"[Preprocessing] Intra-modality contrast shift with "
                f"{halluc_ratio:.0%} hallucination -> histogram matching "
                f"to reduce intensity gap before adaptation.")

        # ================================================================
        # STEP 4: Trainable scope ← severity + failure type + literature
        # ================================================================
        # Scope determines HOW MUCH of the model we adapt.
        # Decision combines: severity, failure type, AND literature guidance.

        # Use verified severity's scope recommendation when available.
        # This is the KEY improvement: verified severity reconciles knowledge
        # (physics-based expectations) with statistics (what the model sees),
        # so scope decisions are informed by BOTH, not just raw miss_ratio.

        if vs_scope_rec:
            # Verified severity has a scope recommendation — use it
            plan["trainable_scope"] = vs_scope_rec
            rationale_parts.append(
                f"[Scope] Verified severity ({severity:.2f}, {severity_level}) "
                f"recommends {vs_scope_rec}.")

            # "Do no harm" check: when severity is minimal/mild AND
            # knowledge confirms mild/no standardization gap AND
            # classifier confirms model already works well on target,
            # skip adaptation entirely. Even BN reset can hurt when
            # source BN stats are already well-calibrated.
            # Use verification agent's classifier severity (prediction-based)
            # rather than diagnosis agent's (failure-distribution-based),
            # because the latter can overestimate severity when model works
            # well but has some expected boundary errors.
            do_no_harm = (
                severity_level in ("minimal", "mild") and
                fg_ratio > 0.01 and
                vs_classifier_severity < 0.2 and
                prior_difficulty == "mild" and
                std_gap in ("mild", "none")
            )
            if do_no_harm:
                plan["skip_adaptation"] = True
                plan["num_steps"] = 0
                plan["learning_rate"] = 0
                plan["loss_weights"] = {
                    "entropy": 0.0, "consistency": 0.0, "diversity": 0.0,
                    "bn_adapt": 0.0, "pseudo_label": 0.0, "anatomy": 0.0,
                }
                rationale_parts.append(
                    f"[DO NO HARM] Severity={severity:.2f} ({severity_level}), "
                    f"classifier={classifier_severity:.2f}, "
                    f"knowledge=mild, std_gap={std_gap} -> skip adaptation. "
                    f"Source model already well-calibrated for target.")

            # SAFE MODE: model predictions are already good (classifier ≈ 0)
            # but there IS a distributional shift (moderate severity).
            # Entropy/Tent gradients risk hurting well-calibrated classes.
            # Safest action: AdaBN (BN stats reset only, no gradient updates).
            # This recalibrates BN running stats to target without any risk
            # of over-adapting individual class boundaries.
            elif (vs_classifier_severity < 0.05 and fg_ratio > 0.01 and
                  severity_level == "moderate"):
                plan["num_steps"] = 0
                plan["learning_rate"] = 0
                plan["loss_weights"] = {
                    "entropy": 0.0, "consistency": 0.0, "diversity": 0.0,
                    "bn_adapt": 1.0, "pseudo_label": 0.0, "anatomy": 0.0,
                }
                plan["tent_bn_mode"] = False
                rationale_parts.append(
                    f"[SAFE MODE] Moderate shift ({severity:.2f}) but model "
                    f"already well-calibrated (classifier={vs_classifier_severity:.2f}). "
                    f"BN stats reset only — no gradient updates to avoid "
                    f"hurting well-calibrated classes.")

            elif severity_level == "minimal" and fg_ratio > 0.01:
                plan["num_steps"] = 0
                plan["learning_rate"] = 0
                plan["loss_weights"] = {
                    "entropy": 0.0, "consistency": 0.0, "diversity": 0.0,
                    "bn_adapt": 1.0, "pseudo_label": 0.0, "anatomy": 0.0,
                }
                rationale_parts.append(
                    f"[Scope] Minimal verified severity -> BN stats only, no gradients.")
        elif severity < 0.25 and fg_ratio > 0.01:
            # Fallback: Model already works — check if skip entirely
            plan["trainable_scope"] = "bn_only"
            if prior_difficulty == "mild" or std_gap in ("mild", "none"):
                plan["skip_adaptation"] = True
                plan["num_steps"] = 0
                plan["learning_rate"] = 0
                plan["loss_weights"] = {
                    "entropy": 0.0, "consistency": 0.0, "diversity": 0.0,
                    "bn_adapt": 0.0, "pseudo_label": 0.0, "anatomy": 0.0,
                }
                rationale_parts.append(
                    f"[DO NO HARM] Low severity ({severity:.2f}) + "
                    f"mild standardization gap -> skip adaptation entirely.")
            else:
                plan["num_steps"] = 0
                plan["learning_rate"] = 0
                plan["loss_weights"] = {
                    "entropy": 0.0, "consistency": 0.0, "diversity": 0.0,
                    "bn_adapt": 1.0, "pseudo_label": 0.0, "anatomy": 0.0,
                }
                rationale_parts.append(
                    f"[Scope] Minimal shift (severity={severity:.2f}, "
                    f"fg={fg_ratio:.1%}): BN stats only.")
        elif severity > 0.8 or fg_ratio < 0.01:
            # Fallback: High severity — use literature guidance for scope
            if lit_scope in ("bn+decoder1",):
                plan["trainable_scope"] = "bn+decoder1"
                rationale_parts.append(
                    f"[Scope] High severity ({severity:.2f}) + literature "
                    f"recommends {lit_scope}.")
            else:
                plan["trainable_scope"] = "bn+output"
                rationale_parts.append(
                    f"[Scope] High severity ({severity:.2f}) or "
                    f"low fg ({fg_ratio:.1%}): bn+output.")
        elif prior_difficulty == "high" and lit_scope in ("bn+output", "bn+decoder1"):
            plan["trainable_scope"] = lit_scope
            rationale_parts.append(
                f"[Scope] Literature recommends {lit_scope} for this shift type "
                f"(prior_difficulty={prior_difficulty}).")
        elif prior_difficulty == "mild" and lit_scope == "bn_only":
            plan["trainable_scope"] = "bn_only"
            rationale_parts.append(
                f"[Scope] Literature confirms bn_only sufficient "
                f"(prior_difficulty={prior_difficulty}).")
        else:
            plan["trainable_scope"] = "bn_only"
            rationale_parts.append(
                f"[Scope] BN-only (severity={severity:.2f}, fg={fg_ratio:.1%}).")

        # ================================================================
        # STEP 5: Loss weights ← failure type distribution
        # ================================================================
        # Different failure modes need different loss emphasis.
        if plan["num_steps"] > 0:  # skip if minimal mode
            w = plan["loss_weights"]

            if boundary_ratio > 0.4 and boundary_ratio > miss_ratio:
                # Boundary-dominant: entropy sharpens boundaries
                w["entropy"] = 0.7
                w["consistency"] = 0.15
                w["diversity"] = 0.05
                w["bn_adapt"] = 0.3
                rationale_parts.append(
                    f"[Loss] Boundary-dominant ({boundary_ratio:.0%}): "
                    "high entropy for boundary sharpening.")

            elif miss_ratio > 0.5:
                # Miss-dominant: diversity prevents collapse, consistency
                # helps recover missing structures
                w["entropy"] = 0.3
                w["consistency"] = 0.25
                w["diversity"] = 0.3
                w["bn_adapt"] = 0.3
                rationale_parts.append(
                    f"[Loss] Miss-dominant ({miss_ratio:.0%}): "
                    "high diversity to prevent collapse + consistency.")

            elif halluc_ratio > 0.5:
                # Hallucination-dominant — but WHY is it hallucinating?
                # If contrast reversal (knowledge says different/fundamentally_different),
                # the model confidently predicts WRONG structures because tissue
                # appearances are reversed. Entropy minimization would reinforce
                # these hallucinations. Instead: skip entropy, rely on BN + diversity.
                if contrast_change in ("different", "fundamentally_different"):
                    # Contrast reversal hallucination: DO NOT use entropy.
                    # Entropy would make the model more confident in wrong predictions.
                    # Use BN adapt (recalibrate statistics) + diversity (prevent collapse)
                    # + consistency (augmentation invariance is still valid).
                    w["entropy"] = 0.0
                    w["consistency"] = 0.2
                    w["diversity"] = 0.4
                    w["bn_adapt"] = 0.5
                    rationale_parts.append(
                        f"[Loss] Hallucination from contrast reversal ({halluc_ratio:.0%}): "
                        "NO entropy (would reinforce wrong predictions), "
                        "high diversity + BN adapt to recalibrate.")
                else:
                    # Regular hallucination (e.g., domain gap causes false positives
                    # but not from contrast reversal): entropy CAN help sharpen
                    w["entropy"] = 0.5
                    w["consistency"] = 0.3
                    w["diversity"] = 0.1
                    w["bn_adapt"] = 0.3
                    rationale_parts.append(
                        f"[Loss] Hallucination-dominant ({halluc_ratio:.0%}): "
                        "entropy + consistency to suppress false positives.")

            elif confusion_ratio > 0.3:
                # Confusion-dominant: consistency helps disambiguate classes
                w["entropy"] = 0.3
                w["consistency"] = 0.35
                w["diversity"] = 0.15
                w["bn_adapt"] = 0.3
                rationale_parts.append(
                    f"[Loss] Confusion-dominant ({confusion_ratio:.0%}): "
                    "high consistency for class disambiguation.")

            else:
                # Balanced / classifier blind — use knowledge to inform weights
                w["entropy"] = 0.4
                w["consistency"] = 0.2
                w["diversity"] = 0.15
                w["bn_adapt"] = 0.3

                # Knowledge-informed adjustments for balanced case:
                if noise_change in ("different", "fundamentally_different"):
                    # Different noise model -> consistency helps (augmentation
                    # invariance bridges noise gap)
                    w["consistency"] = max(w["consistency"], 0.3)
                    rationale_parts.append(
                        f"[Loss] Noise model change ({noise_change}) "
                        "-> increased consistency weight.")
                elif std_gap == "severe":
                    # Severe standardization gap -> BN adaptation is critical
                    w["bn_adapt"] = max(w["bn_adapt"], 0.5)
                    rationale_parts.append(
                        "[Loss] Severe standardization gap "
                        "-> high BN adapt priority.")
                else:
                    rationale_parts.append(
                        "[Loss] Balanced/unknown failure: moderate all losses.")

        # ================================================================
        # STEP 6: Verified shifts -> adjust specific loss components
        # ================================================================
        if plan["num_steps"] > 0:
            w = plan["loss_weights"]

            if has_intensity:
                # Confirmed intensity shift -> BN correction is critical
                w["bn_adapt"] = max(w["bn_adapt"], 0.4)
                rationale_parts.append(
                    f"[Verified] Intensity shift (KL={intensity_kl:.2f})"
                    " -> BN adapt priority.")

            if has_style:
                rationale_parts.append(
                    f"[Verified] Style shift ({style_shift:.2f}).")

            if has_scale:
                rationale_parts.append(
                    f"[Verified] Scale shift (ratio={scale_ratio:.2f}).")

            if has_texture:
                rationale_parts.append(
                    f"[Verified] Texture shift ({texture_shift:.2f}).")

            if n_verified == 0 and severity > 0.3:
                # No verified shifts but severity exists -> shift-severity
                # fallback is driving. Be more conservative.
                rationale_parts.append(
                    "[Verified] No shifts confirmed, using conservative approach.")

        # ================================================================
        # STEP 7: Learning rate ← scope + literature guidance
        # ================================================================
        if plan["num_steps"] > 0:
            if plan["trainable_scope"] == "bn_only":
                plan["learning_rate"] = 1e-3
                rationale_parts.append("[LR] 1e-3 (BN-only, safe).")
            elif lit_lr == "very_conservative_3e4":
                plan["learning_rate"] = 3e-4
                rationale_parts.append(
                    "[LR] 3e-4 (literature: very conservative for this shift).")
            elif lit_lr == "conservative_5e4" or severity > 0.7:
                plan["learning_rate"] = 5e-4
                rationale_parts.append(
                    f"[LR] 5e-4 (literature/severity={severity:.2f}).")
            elif lit_lr == "standard_1e3":
                plan["learning_rate"] = 1e-3
                rationale_parts.append(
                    "[LR] 1e-3 (literature: standard for mild shift).")
            else:
                plan["learning_rate"] = 3e-4
                rationale_parts.append("[LR] 3e-4 (wider scope, moderate).")

        # ================================================================
        # STEP 8: Adaptation mode + step count ← scope + literature + dataset size
        # ================================================================
        if plan["num_steps"] > 0:
            # Determine adaptation mode from verified severity recommendation
            # or fall back to scope-based default
            use_tent = False
            if vs_mode_rec == "per_image_tent" and plan["trainable_scope"] == "bn_only":
                use_tent = True
            elif plan["trainable_scope"] == "bn_only":
                use_tent = True
            elif lit_mode == "per_image_tent" and plan["trainable_scope"] == "bn_only":
                use_tent = True

            # Check if literature warns against Tent for this shift
            tent_warned = any("tent" in f.lower() and ("collapse" in f.lower()
                or "fail" in f.lower()) for f in what_fails)
            if tent_warned and use_tent and severity > 0.5:
                use_tent = False
                rationale_parts.append(
                    "[Literature] Tent warned to collapse for this shift "
                    "-> switching to batch multi-loss.")

            if use_tent:
                plan["tent_bn_mode"] = True
                base_steps = 80
                step_scale = min(1.0, max(0.2, n_target / 100.0))
                plan["num_steps"] = max(1, int(base_steps * step_scale))
                rationale_parts.append(
                    f"[Mode] Per-image Tent ({plan['num_steps']} steps, "
                    f"n={n_target}).")
            else:
                plan["tent_bn_mode"] = False
                base_steps = 120 if severity > 0.7 else 80
                # More steps for high-difficulty shifts (literature-informed)
                if prior_difficulty == "high":
                    base_steps = max(base_steps, 150)
                step_scale = min(1.0, max(0.3, n_target / 100.0))
                plan["num_steps"] = max(10, int(base_steps * step_scale))
                rationale_parts.append(
                    f"[Mode] Batch multi-loss ({plan['num_steps']} steps, "
                    f"n={n_target}, difficulty={prior_difficulty}).")

        # ================================================================
        # STEP 9: FDA warmup ← knowledge + statistical metrics
        # ================================================================
        # FDA bridges appearance gap by swapping low-frequency spectrum.
        # Decision to use FDA and its hyperparameters are driven by
        # Analysis Agent metrics, not hardcoded thresholds.
        #
        # Metrics used:
        #   style_shift (low_freq_shift): How different the low-freq appearance is.
        #     This is EXACTLY what FDA corrects. Higher -> FDA more useful.
        #   texture_shift (high_freq_shift): How different high-freq content is.
        #     FDA does NOT fix this. If texture >> style, FDA images are
        #     unreliable and wider scope (decoder) is dangerous.
        #   intensity_kl: Overall intensity distribution gap.
        #     Confirms the appearance gap magnitude.
        #
        # Adaptive parameters:
        #   beta: Controlled by style_shift. More style gap -> larger beta.
        #   scope: bn+output when texture_shift is high (FDA can't fix texture,
        #          so decoder training on FDA images would learn wrong textures).
        #   mix_weight: Higher when FDA covers more of the total shift
        #          (style_shift / total_shift ratio).
        #   steps: More when the gap is larger.

        plan["use_fda_warmup"] = False
        has_source = bool(self.state.source_images and self.state.source_labels)

        if contrast_change == "fundamentally_different" and has_source:
            plan["use_fda_warmup"] = True

            # --- Beta ← style_shift (low-frequency gap) ---
            # style_shift range: 0-1+. FDA swaps low-freq, so beta should
            # scale with how much low-freq correction is needed.
            # Use sqrt scaling: beta grows sub-linearly because even small
            # beta values transfer significant appearance information.
            # Clamp to [0.01, 0.06] — conservative range from FDA paper.
            fda_beta = np.clip(0.01 + np.sqrt(style_shift) * 0.04, 0.01, 0.06)

            # --- Scope ← texture_shift vs style_shift ratio ---
            # If texture (high-freq) shift is large relative to style (low-freq),
            # FDA images still have wrong textures -> unsafe to train decoder.
            # Only use bn+output to avoid decoder corruption.
            texture_ratio = texture_shift / max(style_shift + 0.01, 0.01)
            if texture_ratio > 1.5:
                fda_scope = "bn+output"
                fda_scope_reason = (f"texture>>style ({texture_shift:.2f} vs "
                                    f"{style_shift:.2f}), FDA can't fix texture")
            elif texture_ratio < 0.5 and style_shift > 0.2:
                # Style-dominant gap, FDA covers most of it -> decoder is safe
                fda_scope = "bn+decoder1"
                fda_scope_reason = (f"style-dominant ({style_shift:.2f} vs "
                                    f"{texture_shift:.2f}), FDA covers gap well")
            else:
                fda_scope = "bn+output"
                fda_scope_reason = "balanced shift, conservative scope"

            # --- Mix weight ← style_shift / total_shift ---
            # How much of the total gap does FDA cover?
            # Higher coverage -> FDA images are more trustworthy -> higher weight.
            total_shift = intensity_kl + style_shift + texture_shift + 0.01
            fda_coverage = style_shift / total_shift
            fda_mix_weight = np.clip(fda_coverage * 0.6, 0.1, 0.5)

            # --- Steps ← overall shift magnitude ---
            fda_steps = int(np.clip(total_shift * 100, 50, 200))

            # --- LR ← scope (wider scope needs lower LR) ---
            fda_lr = 3e-4 if fda_scope == "bn+decoder1" else 5e-4

            plan["fda_beta"] = float(fda_beta)
            plan["fda_finetune_steps"] = fda_steps
            plan["fda_finetune_lr"] = fda_lr
            plan["fda_scope"] = fda_scope
            plan["use_fda_mixed_training"] = True
            plan["fda_mix_weight"] = float(fda_mix_weight)

            rationale_parts.append(
                f"[FDA] Cross-modality: beta={fda_beta:.3f} "
                f"(style={style_shift:.2f}), scope={fda_scope} "
                f"({fda_scope_reason}), mix_w={fda_mix_weight:.2f} "
                f"(coverage={fda_coverage:.1%}), steps={fda_steps}, "
                f"lr={fda_lr}.")

        # ================================================================
        # STEP 10: Anatomy post-processing ← containment rules
        # ================================================================
        if cfg.containment_rules or cfg.anatomy_priors:
            plan["use_anatomy_refinement"] = True
            rationale_parts.append(
                f"[PostProc] {len(cfg.containment_rules)} anatomy rules.")

        # ================================================================
        # STEP 11: LLM recommendation logging (NO hyperparameter override)
        # ================================================================
        # The LLM's diagnosis reasoning is logged for interpretability,
        # but does NOT override the rule-based plan's hyperparameters.
        # Empirical finding: LLM hyperparameter suggestions are
        # miscalibrated (too conservative LR, too many steps) and
        # cause negative transfer on mild shifts (prostate: -11%).
        # The rule-based plan's LR/steps are empirically validated.
        if self.state.llm_recommendation:
            rec = self.state.llm_recommendation
            rationale_parts.append(
                f"[LLM] {rec.get('reasoning', '')[:100]}")

        # ================================================================
        # STEP 12: Iteration control <- convergence risk assessment
        # ================================================================
        # Principle: more iterations help when the model can gradually refine,
        # but hurt when hallucination is high (each iteration amplifies errors
        # because self-supervised losses operate on wrong predictions).
        # Also limit when histogram matching is used (the preprocessing
        # already did the heavy lifting; further iterations risk undoing it).
        if plan.get("use_histogram_matching") or halluc_ratio > 0.5:
            plan["max_iterations"] = 1
            rationale_parts.append(
                f"[Iterations] Limited to 1 (halluc={halluc_ratio:.0%}"
                f"{', histogram matched' if plan.get('use_histogram_matching') else ''}"
                f") - multiple iterations risk amplifying errors.")

        plan["rationale"] = " ".join(rationale_parts)
        return plan

    def _prepare_target_batch(self) -> List[torch.Tensor]:
        from PIL import Image as PILImage
        tensors = []
        for img in self.state.target_images:
            pil = PILImage.fromarray((img * 255).astype(np.uint8))
            pil = pil.resize((self.config.image_size, self.config.image_size),
                             PILImage.BILINEAR)
            arr = np.array(pil).astype(np.float32) / 255.0
            # Handle grayscale (H,W) -> (3,H,W)
            if arr.ndim == 2:
                tensor = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
            else:
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(tensor)
        return tensors

    def _apply_safety_guardrails(self, plan: Dict) -> Dict:
        """Validate LLM-proposed plan against safety checks.

        The LLM is good at reasoning about WHAT to do, but may
        miss edge cases that cause negative transfer. These guardrails
        encode lessons learned from empirical failures:

        1. DO NO HARM: if model predictions are already good
           (classifier_severity ≈ 0), don't apply aggressive adaptation
        2. Hallucination guard: if model is confidently wrong (halluc > 50%),
           entropy minimization will reinforce wrong predictions
        3. Iteration limit: high hallucination + multiple iterations
           compounds errors geometrically
        """
        report = self.state.shift_report
        fr = self.state.failure_report
        vs = (self.state.verification or {}).get("verified_severity", {})
        knowledge = self.state.rag_knowledge or {}

        halluc_ratio = 0.0
        if fr and hasattr(fr, 'type_distribution') and fr.type_distribution:
            halluc_ratio = fr.type_distribution.get("hallucination", 0.0)

        classifier_severity = vs.get("classifier_severity", 0.5)
        severity_level = vs.get("severity_level", "moderate")
        std_gap = knowledge.get("standardization_gap", "unknown")

        overrides = []

        # Guard 1: DO NO HARM / SAFE MODE
        # If model already produces good predictions on target,
        # aggressive adaptation risks hurting well-calibrated classes
        if classifier_severity < 0.05 and severity_level in ("minimal", "mild", "moderate"):
            if plan.get("num_steps", 0) > 0:
                plan["num_steps"] = 0
                plan["learning_rate"] = 0
                plan["loss_weights"] = {
                    "entropy": 0.0, "consistency": 0.0, "diversity": 0.0,
                    "bn_adapt": 1.0, "pseudo_label": 0.0, "anatomy": 0.0,
                }
                plan["tent_bn_mode"] = False
                overrides.append(
                    f"SAFE MODE override: classifier_severity={classifier_severity:.2f}, "
                    f"model already well-calibrated → BN reset only")

        # Guard 2: No entropy under hallucination
        # Entropy min sharpens the current prediction. If the current
        # prediction is WRONG (halluc > 50%), this reinforces errors.
        if halluc_ratio > 0.5:
            w = plan.get("loss_weights", {})
            if w.get("entropy", 0) > 0:
                w["entropy"] = 0.0
                w["diversity"] = max(w.get("diversity", 0), 0.3)
                w["bn_adapt"] = max(w.get("bn_adapt", 0), 0.4)
                overrides.append(
                    f"Hallucination guard: halluc={halluc_ratio:.0%} → "
                    f"disabled entropy (would reinforce wrong predictions)")

        # Guard 3: Iteration limit under hallucination
        if halluc_ratio > 0.5:
            plan["max_iterations"] = 1
            overrides.append(
                f"Iteration limit: halluc={halluc_ratio:.0%} → "
                f"max 1 iteration (errors compound)")

        # Guard 4: Contrast reversal histogram matching
        contrast = knowledge.get("contrast_mechanism_change", "same")
        fg_ratio = 0
        if report and report.prediction_distribution:
            fg_ratio = sum(v for k, v in report.prediction_distribution.items()
                          if k != "background")
        if (contrast == "different" and halluc_ratio > 0.5 and fg_ratio > 0.005):
            plan["use_histogram_matching"] = True
            overrides.append("Added histogram matching for contrast reversal")

        if overrides:
            plan["rationale"] += " | Safety overrides: " + "; ".join(overrides)
            print(f"  [Safety] {len(overrides)} guardrails applied:")
            for o in overrides:
                print(f"    → {o}")

        return plan

    def _llm_design_plan(self) -> Optional[Dict]:
        """Use LLM to design the adaptation plan (STEP3).

        The LLM receives the full diagnosis context and designs the plan.
        Returns None if LLM call fails (falls back to rule-based).
        """
        from medadaptagent.pipeline_config import STEP3_ADAPT_PROMPT

        # Build diagnosis summary for the LLM
        report = self.state.shift_report
        fr = self.state.failure_report
        knowledge = self.state.rag_knowledge or {}
        verification = self.state.verification or {}
        cfg = self.config

        diagnosis_parts = []
        if knowledge:
            diagnosis_parts.append(
                f"Knowledge: {knowledge.get('dominant_shift_factor', 'unknown')}. "
                f"Contrast mechanism: {knowledge.get('contrast_mechanism_change', 'unknown')}. "
                f"Standardization gap: {knowledge.get('standardization_gap', 'unknown')}.")
        if report:
            diagnosis_parts.append(
                f"Statistics: intensity_KL={report.intensity_shift:.3f}, "
                f"style_shift={report.low_freq_shift:.3f}, "
                f"texture_shift={report.high_freq_shift:.3f}, "
                f"scale_ratio={report.scale_ratio:.2f}. "
                f"Model confidence={report.mean_confidence:.3f}, "
                f"entropy={report.mean_entropy:.3f}.")
        if fr:
            diagnosis_parts.append(
                f"Failure types: {fr.type_distribution}. "
                f"Severity: {fr.overall_severity:.2f}.")
        vs = verification.get("verified_severity", {})
        if vs:
            diagnosis_parts.append(
                f"Verified severity: {vs.get('severity_score', 0):.2f} "
                f"({vs.get('severity_level', 'unknown')}). "
                f"Recommended scope: {vs.get('scope_recommendation', 'unknown')}.")

        # Per-class metrics from initial evaluation
        per_class = ""
        if self.state.iteration_history:
            last = self.state.iteration_history[-1]
            parts = [f"{k}: {v:.4f}" for k, v in last.items() if "dice" in k]
            per_class = ", ".join(parts)

        diagnosis_summary = " ".join(diagnosis_parts)

        prompt = STEP3_ADAPT_PROMPT.format(
            task_description=cfg.get_task_description(),
            diagnosis_summary=diagnosis_summary,
            anatomy_priors=json.dumps(cfg.anatomy_priors, indent=2),
            containment_rules=str(cfg.containment_rules),
            per_class_metrics=per_class or "Not yet evaluated",
        )

        try:
            model_name = getattr(self.state.llm_client, '_model_name',
                                 'claude-sonnet-4-20250514')
            response = self.state.llm_client.messages.create(
                model=model_name,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                llm_plan = json.loads(text[start:end])
                # Ensure required keys
                plan = {
                    "learning_rate": float(llm_plan.get("learning_rate", 1e-4)),
                    "num_steps": int(llm_plan.get("num_steps", 100)),
                    "loss_weights": llm_plan.get("loss_weights", {
                        "entropy": 0.3, "consistency": 0.2, "diversity": 0.2,
                        "bn_adapt": 0.3, "pseudo_label": 0.0, "anatomy": 0.0,
                    }),
                    "trainable_scope": llm_plan.get("trainable_scope", "bn+output"),
                    "tent_bn_mode": llm_plan.get("trainable_scope") == "bn_only",
                    "skip_adaptation": llm_plan.get("skip_adaptation", False),
                    "use_anatomy_refinement": llm_plan.get(
                        "use_anatomy_refinement", False),
                    "rationale": f"[LLM] {llm_plan.get('rationale', 'LLM-designed plan')}",
                }
                # Carry over FDA decisions from knowledge analysis
                knowledge = self.state.rag_knowledge or {}
                contrast = knowledge.get("contrast_mechanism_change", "same")
                has_source = bool(self.state.source_images and self.state.source_labels)
                if contrast == "fundamentally_different" and has_source:
                    style_shift = getattr(report, 'low_freq_shift', 0.2)
                    plan["use_fda_warmup"] = True
                    plan["fda_beta"] = float(np.clip(
                        0.01 + np.sqrt(style_shift) * 0.04, 0.01, 0.06))
                    plan["fda_finetune_steps"] = 100
                    plan["fda_finetune_lr"] = 5e-4
                    plan["fda_scope"] = "bn+output"
                    plan["use_fda_mixed_training"] = True
                    plan["fda_mix_weight"] = 0.2

                print(f"  [LLM Plan] scope={plan['trainable_scope']}, "
                      f"lr={plan['learning_rate']}, steps={plan['num_steps']}")
                print(f"  [LLM Rationale] {plan['rationale'][:200]}")
                return plan
        except Exception as e:
            print(f"  LLM plan design failed ({e}), falling back to rule-based.")
        return None

    def run_fda_warmup(self):
        """FDA warmup: style-transfer source images to target style, fine-tune with labels.

        This bridges the cross-modality appearance gap using supervised signal
        from source labels + FDA-transferred images. Only runs when the plan
        enables it (cross-modality shifts with fundamentally different contrast).
        """
        from medadaptagent.adaptation.advanced_tta import fda_source_to_target
        from PIL import Image as PILImage

        plan = self.state.adaptation_plan
        beta = plan.get("fda_beta", 0.05)
        ft_steps = plan.get("fda_finetune_steps", 100)
        ft_lr = plan.get("fda_finetune_lr", 5e-4)
        device = self.config.device
        img_size = self.config.image_size
        num_classes = self.config.num_classes

        source_images = self.state.source_images
        source_labels = self.state.source_labels
        target_images = self.state.target_images
        model = self.state.adapted_model

        print(f"\n  === FDA Warmup (beta={beta}, steps={ft_steps}, lr={ft_lr}) ===")
        print(f"    Generating {len(source_images)} FDA-transferred source images...")

        # Generate FDA-transferred source images (source content + target style)
        fda_images = []
        fda_labels = []
        for i, (src_img, src_lbl) in enumerate(zip(source_images, source_labels)):
            # Pick random target as style reference
            tgt_idx = np.random.randint(len(target_images))
            tgt_img = target_images[tgt_idx]

            # Ensure HWC format for FDA
            if src_img.ndim == 3 and src_img.shape[2] == 3:
                src_hwc = src_img
            elif src_img.ndim == 2:
                src_hwc = np.stack([src_img] * 3, axis=-1)
            else:
                src_hwc = src_img

            if tgt_img.ndim == 3 and tgt_img.shape[2] == 3:
                tgt_hwc = tgt_img
            elif tgt_img.ndim == 2:
                tgt_hwc = np.stack([tgt_img] * 3, axis=-1)
            else:
                tgt_hwc = tgt_img

            styled = fda_source_to_target(src_hwc, tgt_hwc, beta=beta)

            # Resize to model input size
            pil = PILImage.fromarray((styled * 255).astype(np.uint8))
            pil = pil.resize((img_size, img_size), PILImage.BILINEAR)
            arr = np.array(pil).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
            fda_images.append(tensor)

            # Resize label
            lbl_pil = PILImage.fromarray(src_lbl.astype(np.uint8))
            lbl_pil = lbl_pil.resize((img_size, img_size), PILImage.NEAREST)
            lbl_tensor = torch.from_numpy(np.array(lbl_pil)).long()
            fda_labels.append(lbl_tensor)

        print(f"    FDA transfer complete. {len(fda_images)} images generated.")

        # BN reset on real target first (get target BN stats)
        self._reset_norm_stats(model)
        if self._target_tensors is None:
            self._target_tensors = self._prepare_target_batch()
        model.train()
        with torch.no_grad():
            for _ in range(3):
                indices = np.random.permutation(len(self._target_tensors))
                for start in range(0, len(indices), 16):
                    batch_idx = indices[start:start + 16]
                    batch = torch.stack(
                        [self._target_tensors[i] for i in batch_idx]
                    ).to(device)
                    model(batch)

        # Fine-tune on FDA-transferred source images with real labels.
        # Use wider scope than self-supervised phase since we have real labels.
        fda_scope = plan.get("fda_scope", "bn+decoder1")
        trainable_params = self._select_trainable_params(model, fda_scope)
        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"    Fine-tuning on FDA images (scope={fda_scope}, "
              f"{n_trainable:,} params)...")

        model.train()
        optimizer = torch.optim.Adam(trainable_params, lr=ft_lr)
        ce_loss_fn = nn.CrossEntropyLoss()

        batch_size = 16
        total_loss = 0
        for step in range(ft_steps):
            idx = np.random.choice(len(fda_images), size=min(batch_size, len(fda_images)),
                                   replace=False)
            batch_imgs = torch.stack([fda_images[i] for i in idx]).to(device)
            batch_lbls = torch.stack([fda_labels[i] for i in idx]).to(device)

            logits = model(batch_imgs)

            # CE + Dice loss for better segmentation training
            ce = ce_loss_fn(logits, batch_lbls)
            probs = F.softmax(logits, dim=1)
            targets_oh = F.one_hot(batch_lbls, num_classes).permute(0, 3, 1, 2).float()
            inter = (probs * targets_oh).sum(dim=(0, 2, 3))
            union = probs.sum(dim=(0, 2, 3)) + targets_oh.sum(dim=(0, 2, 3))
            dice = (2 * inter + 1.0) / (union + 1.0)
            dice_loss = 1 - dice[1:].mean()  # skip background

            loss = ce + dice_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 50 == 0 or step + 1 == ft_steps:
                print(f"    FDA step {step+1}/{ft_steps}, loss={loss.item():.4f} "
                      f"(CE={ce.item():.4f}, Dice={dice_loss.item():.4f})")

        avg_loss = total_loss / ft_steps
        print(f"    FDA warmup complete. Avg loss: {avg_loss:.4f}")
        self.state._fda_warmup_done = True

        # Store FDA images/labels for mixed training in adaptation loop
        self._fda_images = fda_images
        self._fda_labels = fda_labels

        # Quick check: predictions on target after FDA warmup
        model.eval()
        with torch.no_grad():
            sample = torch.stack(self._target_tensors[:16]).to(device)
            sample_pred = model(sample).argmax(dim=1).cpu().numpy()
            fg_ratio = (sample_pred > 0).mean()
            print(f"    Post-FDA predictions: {fg_ratio*100:.1f}% foreground")

    def _reset_norm_stats(self, model):
        """Reset normalization statistics. Norm-agnostic — delegates to
        `norm_utils.reset_norm_running_stats`, which only touches layers that
        actually track running stats (BN, and IN with tracking). No-op for
        LN/GN. Also re-enables `track_running_stats` on layers where it was
        disabled (e.g. by an earlier tent-mode pass)."""
        from medadaptagent.graph.norm_utils import (
            iter_norm_modules, has_running_stats, reset_norm_running_stats,
        )
        # Re-enable tracking on any BN/IN layers whose stats got dropped by a
        # previous tent pass, so the reset can write into real buffers.
        for _, mod, nt in iter_norm_modules(model):
            if nt not in ("batchnorm", "instancenorm"):
                continue
            if not getattr(mod, "track_running_stats", True) or \
                    getattr(mod, "running_mean", None) is None:
                mod.track_running_stats = True
                if hasattr(mod, "num_features"):
                    dev = mod.weight.device if mod.weight is not None else "cpu"
                    mod.running_mean = torch.zeros(mod.num_features, device=dev)
                    mod.running_var = torch.ones(mod.num_features, device=dev)
                    mod.num_batches_tracked = torch.tensor(
                        0, dtype=torch.long, device=dev)
        reset_norm_running_stats(model)

    def _get_norm_param_names(self, model):
        """Norm-agnostic: returns the parameter names (weight, bias) of every
        norm layer in the model, regardless of norm type."""
        from medadaptagent.graph.norm_utils import iter_norm_modules
        names = set()
        for name, mod, _ in iter_norm_modules(model):
            for pname, _p in mod.named_parameters():
                names.add(f"{name}.{pname}")
        return names

    def _select_trainable_params(self, model, scope: str) -> List[nn.Parameter]:
        norm_param_names = self._get_norm_param_names(model)
        trainable = []
        for name, param in model.named_parameters():
            param.requires_grad = False
            should_train = name in norm_param_names
            # bn_only: only BN affine params (like Tent)
            if scope == "bn_only":
                pass  # should_train already set from norm_param_names
            elif scope in ("bn+output", "bn+decoder1", "bn+all_decoder"):
                if any(k in name for k in ["out_conv", "final", "head",
                                            "output_hypernetworks", "iou_prediction",
                                            "seg_outputs", "conv_final"]):
                    should_train = True
            if scope in ("bn+decoder1", "bn+all_decoder"):
                if any(k in name for k in ["dec1", "up1", "decoder.0",
                                            "mask_decoder", "decoder1",
                                            "conv_blocks_localization.0"]):
                    should_train = True
            if scope == "bn+all_decoder":
                if any(k in name for k in ["dec", "up", "decoder",
                                            "mask_decoder", "conv_blocks_localization",
                                            "tu"]):
                    should_train = True
            if should_train:
                param.requires_grad = True
                trainable.append(param)
        return trainable

    def _anatomy_containment_loss(self, probs: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.config.device)
        n_rules = 0
        for outer_name, relation, inner_name in self.config.containment_rules:
            outer_idx = self.config.structures.get(outer_name)
            inner_idx = self.config.structures.get(inner_name)
            if outer_idx is None or inner_idx is None or relation != "contains":
                continue
            outer_prob = probs[:, outer_idx]
            inner_prob = probs[:, inner_idx]
            combined = outer_prob + inner_prob
            violation = inner_prob * (1.0 - combined).clamp(min=0)
            loss = loss + violation.mean()
            n_rules += 1
        return loss / max(n_rules, 1)


# ================================================================
# Agent 6: EvaluationAgent
# ================================================================
class EvaluationAgent(BaseAgent):
    """Quality scoring, metrics computation, and post-processing."""

    def run(self) -> Tuple[float, Dict]:
        """Compute annotation-free quality score."""
        model = self.state.adapted_model
        if model is None:
            return 0.0, {}

        model.eval()
        cfg = self.config

        per_class_consistencies = {name: [] for name in cfg.fg_classes}
        per_class_plausibility = {name: [] for name in cfg.fg_classes}
        confidences = []

        with torch.no_grad():
            for img in self.state.target_images[:50]:
                tensor = self._img_to_tensor(img).to(cfg.device)
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

                pred_np = pred.cpu().numpy()[0]
                for cls_name, cls_idx in cfg.fg_classes.items():
                    prior = cfg.anatomy_priors.get(cls_name, {})
                    area_range = prior.get("area_range", (0.001, 0.9))
                    area_frac = (pred_np == cls_idx).sum() / pred_np.size
                    plausible = area_range[0] <= area_frac <= area_range[1]
                    per_class_plausibility[cls_name].append(float(plausible))

                for outer_name, relation, inner_name in cfg.containment_rules:
                    outer_idx = cfg.structures.get(outer_name)
                    inner_idx = cfg.structures.get(inner_name)
                    if outer_idx is None or inner_idx is None or relation != "contains":
                        continue
                    from scipy import ndimage as _ndi
                    outer_only = (pred_np == outer_idx)
                    inner_mask = (pred_np == inner_idx)
                    if inner_mask.sum() > 0 and outer_only.sum() > 0:
                        combined = outer_only | inner_mask
                        labeled, _ = _ndi.label(combined)
                        inner_labels = set(np.unique(labeled[inner_mask])) - {0}
                        outer_labels = set(np.unique(labeled[outer_only])) - {0}
                        shared = inner_labels & outer_labels
                        if inner_labels:
                            shared_mask = np.isin(labeled, list(shared)) & inner_mask
                            contained = shared_mask.sum() / inner_mask.sum()
                        else:
                            contained = 0.0
                        per_class_plausibility[inner_name][-1] *= float(contained)
                    elif inner_mask.sum() > 0 and outer_only.sum() == 0:
                        per_class_plausibility[inner_name][-1] *= 0.0

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

        print(f"    Quality score: {overall:.4f}")
        for cls_name, q in per_class_q.items():
            print(f"    Q({cls_name}): {q:.4f}")

        self.state.quality_scores.append(overall)

        # Update iteration history
        if self.state.iteration_history:
            self.state.iteration_history[-1]["quality_score"] = overall
            self.state.iteration_history[-1]["per_class_quality"] = per_class_q

        # GT evaluation if available
        gt_dice = None
        if self.state.target_labels:
            gt_metrics = self._evaluate_on_target()
            if self.state.iteration_history:
                self.state.iteration_history[-1].update(gt_metrics)
            for cls_name in cfg.fg_classes:
                key = f"{cls_name}_dice"
                print(f"    [GT] {cls_name}: {gt_metrics.get(key, 0):.4f}")
            gt_dice = gt_metrics.get("avg_dice", None)

        # Track best model — use label-free quality score for model selection.
        # GT dice is logged for reporting only, not used for checkpoint selection,
        # to avoid information leakage in zero-shot experiments.
        # (Few-shot path in Agent 7 handles labeled data separately.)
        # === Few-shot calibration (K-shot held-out validation) ===
        # If a calibration set was reserved at the start, evaluate the adapted
        # model on it. The K cases are NOT in the adaptation pool, so this is
        # a true held-out signal. We use few_shot_dice for best-model selection
        # because the unsupervised quality score has Pearson r=0.08 with GT
        # dice (anti-correlated on 4 of 8 datasets).
        few_shot_dice = None
        if self.state.calibration_images and self.state.calibration_labels:
            try:
                from medadaptagent.adaptation.few_shot import evaluate_model
                cal_metrics = evaluate_model(
                    model=self.state.adapted_model,
                    images=self.state.calibration_images,
                    labels=self.state.calibration_labels,
                    device=str(self.state.config.device),
                    num_classes=cfg.num_classes,
                    class_names={v: k for k, v in cfg.structures.items()},
                    batch_size=min(16, len(self.state.calibration_images)),
                )
                few_shot_dice = cal_metrics.get("mean_dice", 0.0)
                print(f"    [FewShot K={len(self.state.calibration_images)}] mean dice: {few_shot_dice:.4f}")
                for c_name in cfg.fg_classes:
                    fs_key = f"{c_name}_dice"
                    if fs_key in cal_metrics:
                        print(f"      few_shot[{c_name}]: {cal_metrics[fs_key]:.4f}")
                if self.state.iteration_history:
                    self.state.iteration_history[-1]["few_shot_dice"] = few_shot_dice
                    for k, v in cal_metrics.items():
                        if k.endswith("_dice"):
                            self.state.iteration_history[-1][f"few_shot_{k}"] = v
            except Exception as e:
                print(f"    [FewShot] computation failed: {e}")
                few_shot_dice = None

        # Best-model selection: prefer few_shot_dice when available, else fall
        # back to unsupervised quality score (R16-R20 behavior).
        if few_shot_dice is not None:
            selection_score = few_shot_dice
            selection_source = "few_shot"
        else:
            selection_score = overall
            selection_source = "quality_score"
        if selection_score > self.state.best_quality_score:
            self.state.best_quality_score = selection_score
            self.state.best_model_state = copy.deepcopy(
                self.state.adapted_model.state_dict())
            print(f"    [Best model updated: {selection_source}={selection_score:.4f}]")

        return overall, per_class_q

    def run_postprocessing(self):
        """Apply anatomy-aware post-processing and evaluate."""
        from scipy import ndimage

        cfg = self.config
        model = self.state.adapted_model
        if model is None or not cfg.containment_rules:
            return

        print(f"\n  === Anatomy Post-Processing ===")

        if not self.state.target_labels:
            print("    No target labels for post-processing evaluation.")
            return

        from PIL import Image as PILImage
        model.eval()

        raw_dices = {name: [] for name in cfg.fg_classes}
        post_dices = {name: [] for name in cfg.fg_classes}

        with torch.no_grad():
            for img, lbl in zip(self.state.target_images, self.state.target_labels):
                tensor = self._img_to_tensor(img).to(cfg.device)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1).cpu().numpy()[0]
                probs_np = probs.cpu().numpy()[0]

                refined = self._postprocess_prediction(pred, probs_np)

                lbl_pil = PILImage.fromarray(lbl)
                lbl_pil = lbl_pil.resize((cfg.image_size, cfg.image_size),
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
        self.state.postprocess_results = {}
        raw_avg_all, post_avg_all = [], []
        for cls_name in cfg.fg_classes:
            raw = np.mean(raw_dices[cls_name]) if raw_dices[cls_name] else 0
            post = np.mean(post_dices[cls_name]) if post_dices[cls_name] else 0
            diff = post - raw
            print(f"    {cls_name}: {raw:.4f} -> {post:.4f} ({diff:+.4f})")
            self.state.postprocess_results[f"{cls_name}_dice_raw"] = float(raw)
            self.state.postprocess_results[f"{cls_name}_dice_post"] = float(post)
            raw_avg_all.append(raw)
            post_avg_all.append(post)

        # Only use post-processing if it actually helps overall
        raw_mean = np.mean(raw_avg_all)
        post_mean = np.mean(post_avg_all)
        if post_mean < raw_mean:
            print(f"    Post-processing HURTS ({raw_mean:.4f} -> {post_mean:.4f}), "
                  f"keeping raw predictions.")
            self.state.postprocess_results["use_postprocessing"] = False
        else:
            print(f"    Post-processing HELPS ({raw_mean:.4f} -> {post_mean:.4f}), "
                  f"applying to final model.")
            self.state.postprocess_results["use_postprocessing"] = True

    def _postprocess_prediction(self, pred: np.ndarray,
                                 probs: np.ndarray) -> np.ndarray:
        """Apply anatomy-aware post-processing to a single prediction."""
        from scipy import ndimage as ndi
        cfg = self.config
        refined = pred.copy()

        # Connected component filtering per class
        for cls_name, cls_idx in cfg.fg_classes.items():
            mask = (refined == cls_idx)
            if mask.sum() == 0:
                continue
            labeled, n_components = ndi.label(mask)
            if n_components > 1:
                sizes = ndi.sum(mask, labeled, range(1, n_components + 1))
                largest = np.argmax(sizes) + 1
                refined[mask & (labeled != largest)] = 0

        # Containment rules
        for outer_name, relation, inner_name in cfg.containment_rules:
            outer_idx = cfg.structures.get(outer_name)
            inner_idx = cfg.structures.get(inner_name)
            if outer_idx is None or inner_idx is None or relation != "contains":
                continue

            combined_region = (refined == outer_idx) | (refined == inner_idx)
            inner_mask = (refined == inner_idx)

            # Inner pixels not adjacent to outer -> remove
            from scipy import ndimage as _ndi
            outer_only = combined_region & ~inner_mask
            isolated_inner = inner_mask & ~(_ndi.binary_dilation(outer_only, iterations=2) | outer_only)
            if isolated_inner.any():
                refined[isolated_inner] = 0

            combined_area = combined_region.sum()
            if combined_area > 0:
                inner_ratio = inner_mask.sum() / combined_area

                if inner_ratio > 0.7:
                    # Inner over-predicted: shrink to central ~40%
                    ys, xs = np.where(combined_region)
                    cy, cx = ys.mean(), xs.mean()
                    dist = np.sqrt((np.arange(pred.shape[0])[:, None] - cy)**2 +
                                   (np.arange(pred.shape[1])[None, :] - cx)**2)
                    target_inner = int(0.4 * combined_area)
                    dist_in = np.full_like(dist, np.inf)
                    dist_in[combined_region] = dist[combined_region]
                    flat = dist_in.ravel()
                    if 0 < target_inner < flat.size:
                        threshold = np.partition(flat, target_inner)[target_inner]
                        new_inner = combined_region & (dist <= threshold)
                        refined[new_inner] = inner_idx
                        refined[combined_region & ~new_inner] = outer_idx

                elif inner_ratio < 0.15 and combined_area > 100:
                    # Inner missing: recover central ~35%
                    ys, xs = np.where(combined_region)
                    cy, cx = ys.mean(), xs.mean()
                    dist = np.sqrt((np.arange(pred.shape[0])[:, None] - cy)**2 +
                                   (np.arange(pred.shape[1])[None, :] - cx)**2)
                    target_inner = int(0.35 * combined_area)
                    dist_in = np.full_like(dist, np.inf)
                    dist_in[combined_region] = dist[combined_region]
                    flat = dist_in.ravel()
                    if 0 < target_inner < flat.size:
                        threshold = np.partition(flat, target_inner)[target_inner]
                        new_inner = combined_region & (dist <= threshold)
                        refined[new_inner] = inner_idx
                        refined[combined_region & ~new_inner] = outer_idx

        return refined

    def _img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        from PIL import Image as PILImage
        pil = PILImage.fromarray((img * 255).astype(np.uint8))
        pil = pil.resize((self.config.image_size, self.config.image_size),
                         PILImage.BILINEAR)
        arr = np.array(pil).astype(np.float32) / 255.0
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
        else:
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor.unsqueeze(0)

    def _evaluate_on_target(self, compute_hd95: bool = False) -> Dict:
        from PIL import Image as PILImage
        model = self.state.adapted_model
        model.eval()
        cfg = self.config

        per_class_dices = {name: [] for name in cfg.fg_classes}
        per_class_hd95s = {name: [] for name in cfg.fg_classes}

        with torch.no_grad():
            for img, lbl in zip(self.state.target_images, self.state.target_labels):
                tensor = self._img_to_tensor(img).to(cfg.device)
                pred = model(tensor).argmax(dim=1).cpu().numpy()[0]

                lbl_pil = PILImage.fromarray(lbl)
                lbl_pil = lbl_pil.resize((cfg.image_size, cfg.image_size),
                                          PILImage.NEAREST)
                lbl_r = np.array(lbl_pil)

                for cls_name, cls_idx in cfg.fg_classes.items():
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

                        if compute_hd95:
                            from medadaptagent.evaluation.eval_metrics import hausdorff_distance_95
                            hd = hausdorff_distance_95(
                                pred_c.astype(bool), gt_c.astype(bool))
                            if np.isfinite(hd):
                                per_class_hd95s[cls_name].append(hd)

        result = {}
        all_dices = []
        all_hd95s = []
        for cls_name, dices in per_class_dices.items():
            mean_dice = float(np.mean(dices)) if dices else 0.0
            result[f"{cls_name}_dice"] = mean_dice
            all_dices.append(mean_dice)

            if compute_hd95:
                hd_vals = per_class_hd95s[cls_name]
                mean_hd = float(np.mean(hd_vals)) if hd_vals else float('inf')
                result[f"{cls_name}_hd95"] = mean_hd
                if np.isfinite(mean_hd):
                    all_hd95s.append(mean_hd)

        result["avg_dice"] = float(np.mean(all_dices)) if all_dices else 0.0
        if compute_hd95:
            result["avg_hd95"] = float(np.mean(all_hd95s)) if all_hd95s else float('inf')
        return result


# ================================================================
# Agent 7: SufficiencyAgent
# ================================================================
class SufficiencyAgent(BaseAgent):
    """Assesses whether zero-shot adaptation is sufficient.

    After adaptation, this agent:
    1. Evaluates the quality gap between adapted model and source performance
    2. Reasons about whether the shift type is amenable to zero-shot TTA
    3. If insufficient, recommends few-shot annotation with strategy selection
    4. Provides a scaling law prediction if target labels are available

    Output: sufficiency assessment with recommendations
    """

    def run(self, source_val_dice: float = None) -> Dict:
        """Run sufficiency assessment."""
        print("\n" + "=" * 60)
        print("AGENT 7: Adaptation Sufficiency Assessment")
        print("=" * 60)

        # Gather signals
        knowledge = self.state.rag_knowledge
        verification = self.state.verification
        failure_report = self.state.failure_report
        plan = self.state.adaptation_plan

        contrast_change = knowledge.get("contrast_mechanism_change", "same")
        severity = verification.get("verified_severity", 0.5)
        if not isinstance(severity, (int, float)):
            severity = 0.5
        halluc_ratio = 0.0
        if failure_report and failure_report.type_distribution:
            halluc_ratio = failure_report.type_distribution.get(
                "hallucination", 0.0)

        # Estimate zero-shot quality from iteration history
        zero_shot_dice = None
        if self.state.iteration_history:
            last = self.state.iteration_history[-1]
            if "avg_dice" in last:
                zero_shot_dice = last["avg_dice"]

        # Quality score from evaluation
        q_score = (self.state.quality_scores[-1]
                   if self.state.quality_scores else 0.0)

        # Assess sufficiency
        assessment = {
            "zero_shot_dice": zero_shot_dice,
            "source_val_dice": source_val_dice,
            "quality_score": q_score,
            "severity": severity,
            "contrast_change": contrast_change,
            "halluc_ratio": halluc_ratio,
            "sufficient": False,
            "confidence": "low",
            "rationale": "",
            "few_shot_recommendation": None,
        }

        # Decision logic
        if source_val_dice and zero_shot_dice:
            gap = source_val_dice - zero_shot_dice
            recovery = zero_shot_dice / max(source_val_dice, 0.01)

            if recovery > 0.85:
                assessment["sufficient"] = True
                assessment["confidence"] = "high"
                assessment["rationale"] = (
                    f"Zero-shot achieves {zero_shot_dice:.1%}, "
                    f"{recovery:.0%} of source performance "
                    f"({source_val_dice:.1%}). No annotation needed.")
            elif recovery > 0.7:
                assessment["sufficient"] = False
                assessment["confidence"] = "medium"
                assessment["rationale"] = (
                    f"Zero-shot achieves {zero_shot_dice:.1%} "
                    f"({recovery:.0%} of source {source_val_dice:.1%}). "
                    f"Moderate gap - few annotations could help.")
            else:
                assessment["sufficient"] = False
                assessment["confidence"] = "high"
                assessment["rationale"] = (
                    f"Zero-shot achieves {zero_shot_dice:.1%} "
                    f"({recovery:.0%} of source {source_val_dice:.1%}). "
                    f"Large performance gap - annotation recommended.")
        elif q_score > 0.8:
            assessment["sufficient"] = True
            assessment["confidence"] = "medium"
            assessment["rationale"] = (
                f"Quality score {q_score:.2f} suggests adequate adaptation.")
        else:
            assessment["sufficient"] = False
            assessment["confidence"] = "medium"
            assessment["rationale"] = (
                f"Quality score {q_score:.2f} suggests room for improvement.")

        # Generate few-shot recommendation
        if not assessment["sufficient"]:
            analysis = {
                "contrast_mechanism_change": contrast_change,
                "intensity_mapping_type": knowledge.get(
                    "intensity_mapping_type", "linear"),
                "verified_severity": float(severity),
                "halluc_ratio": halluc_ratio,
            }
            # Recommend strategies at different N
            recs = {}
            for n in [1, 5, 10]:
                strat = select_strategy(analysis, n)
                recs[n] = {
                    "scope": strat["scope"],
                    "rationale": strat["rationale"],
                }

            # Reasoning about shift type limits
            if contrast_change == "fundamentally_different":
                shift_note = (
                    "Cross-modality shifts have fundamental limits for "
                    "zero-shot TTA because imaging physics differ completely. "
                    "Even a few annotated cases enable supervised fine-tuning "
                    "that can bridge this gap effectively.")
            elif contrast_change == "different":
                shift_note = (
                    "Cross-sequence shifts (e.g., T1 to T2) involve contrast "
                    "reversal that confuses self-supervised methods. "
                    "A small number of annotated cases provides the ground "
                    "truth signal needed to correct these reversals.")
            elif halluc_ratio > 0.5:
                shift_note = (
                    "High hallucination rate ({:.0%}) means self-supervised "
                    "losses reinforce wrong predictions. Labeled data breaks "
                    "this cycle by providing correct supervision.".format(
                        halluc_ratio))
            else:
                shift_note = (
                    "The domain gap is moderate. A few annotated target cases "
                    "would provide direct supervision to close the remaining "
                    "performance gap.")

            assessment["few_shot_recommendation"] = {
                "shift_analysis": shift_note,
                "strategies": recs,
                "note": (
                    "Start from the zero-shot adapted model (not source model) "
                    "to preserve BN statistics and any gains from TTA. "
                    "The agent selects the fine-tuning strategy based on "
                    "the same shift analysis used for zero-shot adaptation."),
            }

        # Print assessment
        print(f"  Zero-shot Dice: {zero_shot_dice:.4f}"
              if zero_shot_dice else "  Zero-shot Dice: N/A")
        if source_val_dice:
            print(f"  Source val Dice: {source_val_dice:.4f}")
        print(f"  Quality score: {q_score:.4f}")
        print(f"  Sufficient: {assessment['sufficient']} "
              f"(confidence: {assessment['confidence']})")
        print(f"  Rationale: {assessment['rationale']}")

        if assessment["few_shot_recommendation"]:
            rec = assessment["few_shot_recommendation"]
            print(f"\n  Few-shot recommendation:")
            print(f"    {rec['shift_analysis']}")
            print(f"    Strategies:")
            for n, s in rec["strategies"].items():
                print(f"      N={n}: {s['rationale']}")

        self.state.log_step("sufficiency_assessment", assessment)
        return assessment


# ================================================================
# Orchestrator: coordinates the multi-agent pipeline
# ================================================================
class Orchestrator:
    """Main orchestrator that coordinates all agents.

    Usage:
        state = PipelineState(config=config, source_model=model, ...)
        orchestrator = Orchestrator(state)
        results = orchestrator.run()
    """

    def __init__(self, state: PipelineState):
        self.state = state
        self.knowledge_agent = KnowledgeAgent(state)
        self.analysis_agent = AnalysisAgent(state)
        self.verification_agent = VerificationAgent(state)
        self.diagnosis_agent = DiagnosisAgent(state)
        self.adaptation_agent = AdaptationAgent(state)
        self.adaptation_agent.orchestrator = self  # backref for pseudo-label gen
        self.evaluation_agent = EvaluationAgent(state)
        self.sufficiency_agent = SufficiencyAgent(state)

    def run(self, max_iterations: int = None) -> Dict:
        """Execute the full multi-agent pipeline."""
        t0 = time.time()
        if max_iterations is None:
            max_iterations = self.state.config.max_iterations

        # Initialize pipeline log
        self.state.pipeline_log["config"] = {
            "task_name": self.state.config.task_name,
            "source_domain": self.state.config.source_domain,
            "target_domain": self.state.config.target_domain,
            "structures": self.state.config.structures,
            "anatomy_priors": str(self.state.config.anatomy_priors),
            "containment_rules": self.state.config.containment_rules,
        }

        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)

        # =============================================
        # Phase 1: Understand the domain shift
        # =============================================
        self.knowledge_agent.run()
        self.analysis_agent.run()

        # =============================================
        # Phase 2: Diagnose failures (before verification so that
        # hallucination detection informs severity assessment)
        # =============================================
        self.diagnosis_agent.run()

        # =============================================
        # Phase 3: Verify (cross-check knowledge vs statistics + diagnosis)
        # =============================================
        self.verification_agent.run()

        # =============================================
        # Phase 3: Design adaptation plan
        # =============================================
        self.state.adaptation_plan = self.adaptation_agent.design_plan()
        plan = self.state.adaptation_plan
        print(f"\n  Adaptation Plan:")
        print(f"    Rationale: {plan['rationale']}")
        if plan.get("skip_adaptation", False):
            print(f"    ** DO NO HARM: Skipping adaptation entirely **")
        else:
            print(f"    LR={plan['learning_rate']}, steps={plan['num_steps']}, "
                  f"scope={plan['trainable_scope']}, "
                  f"tent_bn={plan.get('tent_bn_mode', False)}")
            print(f"    Losses: {plan['loss_weights']}")
            print(f"    Anatomy post-processing: {plan['use_anatomy_refinement']}")

        # =============================================
        # Phase 3b: "Do No Harm" check
        # =============================================
        if plan.get("skip_adaptation", False):
            print(f"\n  === DO NO HARM: Skipping adaptation ===")
            print(f"  Source model already well-calibrated for target domain.")
            print(f"  Returning source model unchanged to preserve performance.")
            # adapted_model is already a copy of source_model from state setup
            self.state.adapted_model = copy.deepcopy(self.state.source_model)
            self.state.adapted_model.to(self.state.config.device)

            # Run one evaluation for reporting
            q_score, per_class_q = self.evaluation_agent.run()
            self.state.log_step("skip_adaptation", {
                "quality_score": q_score,
                "per_class_quality": per_class_q,
                "skipped": True,
            })

            # Sufficiency assessment (model is already good)
            sufficiency = self.sufficiency_agent.run(
                source_val_dice=getattr(self.state, '_source_val_dice', None))

            total_time = time.time() - t0
            results = self._compile_results(total_time)
            results["sufficiency"] = sufficiency
            print("\n" + "=" * 60)
            print("MedAdaptAgent Pipeline Complete (no adaptation needed)")
            print("=" * 60)
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Decision: DO NO HARM - source model preserved")
            if self.state.iteration_history:
                for cls_name in self.state.config.fg_classes:
                    key = f"{cls_name}_dice"
                    if key in self.state.iteration_history[-1]:
                        print(f"  {cls_name} Dice: "
                              f"{self.state.iteration_history[-1][key]:.4f}")
            return results

        # =============================================
        # Phase 3c: FDA warmup (cross-modality only)
        # =============================================
        self.state.adapted_model = copy.deepcopy(self.state.source_model)
        self.state.adapted_model.to(self.state.config.device)

        if plan.get("use_fda_warmup", False):
            self.adaptation_agent.run_fda_warmup()

        # =============================================
        # Phase 3d: Histogram matching (contrast reversal)
        # =============================================
        if plan.get("use_histogram_matching", False):
            self._apply_histogram_matching()

        # =============================================
        # Phase 4: Iterative adaptation
        # =============================================
        # Plan can override max_iterations (e.g., contrast reversal limits to 1)
        if "max_iterations" in plan:
            max_iterations = min(max_iterations, plan["max_iterations"])
        print(f"\n  === Iterative Adaptation ({max_iterations} iterations) ===")
        for iteration in range(max_iterations):
            metrics = self.adaptation_agent.run_iteration(iteration)
            q_score, per_class_q = self.evaluation_agent.run()

            self.state.log_step(f"iteration_{iteration}", {
                **metrics,
                "quality_score": q_score,
                "per_class_quality": per_class_q,
            })

            # Early stopping on quality degradation
            if self._should_stop(iteration):
                break

            # Re-diagnose if quality stagnates
            if self._should_rediagnose(iteration):
                print(f"\n  Re-diagnosing after stagnation...")
                self.diagnosis_agent.run()
                # Update plan based on new diagnosis
                self.state.adaptation_plan = self.adaptation_agent.design_plan()

        # =============================================
        # Phase 4b: Revert to best model from iterations
        # =============================================
        if self.state.best_model_state:
            # Determine final score using same metric as best_quality_score
            hist = self.state.iteration_history
            if hist and "avg_dice" in hist[-1] and self.state.target_labels:
                final_score = hist[-1]["avg_dice"]
            else:
                final_score = self.state.quality_scores[-1] if self.state.quality_scores else 0
            if self.state.best_quality_score > final_score:
                print(f"\n  Reverting to best model (score={self.state.best_quality_score:.4f} "
                      f"vs final={final_score:.4f})")
                self.state.adapted_model.load_state_dict(self.state.best_model_state)
            else:
                print(f"\n  Keeping final model (score={final_score:.4f})")

        # =============================================
        # Phase 5: Finalize with post-processing
        # =============================================
        if plan["use_anatomy_refinement"]:
            self.evaluation_agent.run_postprocessing()

        # =============================================
        # Phase 6: Sufficiency assessment
        # =============================================
        sufficiency = self.sufficiency_agent.run(
            source_val_dice=getattr(self.state, '_source_val_dice', None))

        total_time = time.time() - t0
        results = self._compile_results(total_time)
        results["sufficiency"] = sufficiency

        print("\n" + "=" * 60)
        print("MedAdaptAgent Pipeline Complete")
        print("=" * 60)
        print(f"  Total time: {total_time:.1f}s")
        if self.state.shift_report:
            print(f"  Verified shifts: {self.state.shift_report.verified_shifts}")
        if self.state.failure_report:
            print(f"  Failure types: {self.state.failure_report.type_distribution}")
        if self.state.iteration_history:
            for cls_name in self.state.config.fg_classes:
                key = f"{cls_name}_dice"
                if key in self.state.iteration_history[-1]:
                    print(f"  Final {cls_name} Dice: "
                          f"{self.state.iteration_history[-1][key]:.4f}")

        return results

    def _generate_filtered_pseudo_labels(self):
        """Generate TTA-filtered pseudo-labels for self-training.

        After histogram matching, the source model's predictions on preprocessed
        target images are more reliable — but still contain hallucinations.
        Filter pseudo-labels using multi-augmentation TTA consistency:
        - Run model on original + 3 augmentations (flip-h, flip-v, flip-hv)
        - Only keep pixels where ALL 4 predictions agree
        - Additionally, only keep BACKGROUND predictions (foreground predictions
          from a hallucinating model are unreliable, but background predictions
          on clearly non-tumor tissue are trustworthy)
        - Set uncertain/foreground pixels to ignore_index (255)

        This gives the model a "safe" training signal: "these pixels are definitely
        NOT tumor." The model learns what healthy tissue looks like in T2, which
        indirectly reduces hallucinations.
        """
        print(f"\n  === Generating TTA-filtered pseudo-labels ===")

        model = copy.deepcopy(self.state.source_model)
        model.to(self.state.config.device)
        model.eval()

        target_tensors = self.adaptation_agent._prepare_target_batch()
        n_kept = 0
        n_fg_kept = 0
        n_total = 0
        pseudo_labels = []

        with torch.no_grad():
            for i in range(0, len(target_tensors), 16):
                batch = torch.stack(
                    target_tensors[i:i+16]
                ).to(self.state.config.device)

                # Multi-augmentation TTA
                pred_orig = model(batch).argmax(dim=1)
                pred_fh = model(batch.flip(-1)).argmax(dim=1).flip(-1)
                pred_fv = model(batch.flip(-2)).argmax(dim=1).flip(-2)
                pred_fhv = model(batch.flip(-1).flip(-2)).argmax(dim=1).flip(-2).flip(-1)

                # Consensus: all 4 augmentations agree
                consistent = (
                    (pred_orig == pred_fh) &
                    (pred_orig == pred_fv) &
                    (pred_orig == pred_fhv)
                )

                # For contrast reversal: trust background predictions more than fg.
                # Background pixels where model consistently says "not tumor" are
                # reliable. Foreground pixels may be hallucinations.
                # Strategy: keep ALL consistent background, keep fg only if
                # it's a small coherent region (likely real tumor).
                is_bg = pred_orig == 0

                # Consistent background: high-confidence training signal
                keep_bg = consistent & is_bg
                # Consistent foreground: less reliable but keep some
                keep_fg = consistent & (~is_bg)

                # Set uncertain pixels to 255 (ignore_index)
                filtered_pred = pred_orig.clone()
                # Keep consistent background always
                # Keep consistent foreground (model thinks it's tumor across augmentations)
                keep_mask = keep_bg | keep_fg
                filtered_pred[~keep_mask] = 255

                for j in range(filtered_pred.shape[0]):
                    pseudo_labels.append(filtered_pred[j].cpu())

                n_kept += keep_mask.sum().item()
                n_fg_kept += keep_fg.sum().item()
                n_total += keep_mask.numel()

        keep_ratio = n_kept / max(n_total, 1)
        print(f"    TTA-consistent pixels: {keep_ratio:.1%} ({n_kept:,}/{n_total:,})")
        print(f"    Consistent foreground: {n_fg_kept:,} ({100*n_fg_kept/max(n_total,1):.2f}%)")
        print(f"    Consistent background: {n_kept - n_fg_kept:,}")

        # Prototype-based denoising: filter pseudo-labels using class prototypes
        # Pixels far from their class prototype are unreliable
        if len(pseudo_labels) > 0 and self.state.prototype_denoiser is None:
            try:
                nc = self.state.config.num_classes
                # Auto-detect feature dim from model bottleneck
                adapted = self.state.adapted_model
                adapted.eval()
                target_tensors = self.adaptation_agent._prepare_target_batch()
                sample = target_tensors[0].unsqueeze(0).to(self.state.config.device)
                probe_feat = extract_features_unet(adapted, sample)
                feat_dim = probe_feat.shape[1]
                self.state.prototype_denoiser = PrototypeDenoiser(
                    feature_dim=feat_dim, num_classes=nc, similarity_threshold=0.5)
                print(f"    PrototypeDenoiser initialized (feature_dim={feat_dim})")
            except Exception as e:
                print(f"    PrototypeDenoiser init failed ({e}), skipping")

        if self.state.prototype_denoiser is not None:
            try:
                denoiser = self.state.prototype_denoiser
                # Extract features and update prototypes from high-confidence predictions
                adapted = self.state.adapted_model
                adapted.eval()
                n_denoised = 0
                with torch.no_grad():
                    target_tensors = self.adaptation_agent._prepare_target_batch()
                    for i in range(0, min(len(target_tensors), 200), 16):
                        batch = torch.stack(
                            target_tensors[i:i+16]
                        ).to(self.state.config.device)
                        # Use model features for prototype computation
                        feats = extract_features_unet(adapted, batch)
                        logits = adapted(batch)
                        preds = logits.argmax(dim=1)
                        conf = F.softmax(logits, dim=1).max(dim=1)[0]
                        # Upsample features to match prediction spatial dims
                        if feats.shape[-2:] != preds.shape[-2:]:
                            feats = F.interpolate(feats, size=preds.shape[-2:],
                                                  mode="bilinear", align_corners=False)
                        # Update prototypes from confident predictions
                        denoiser.update_prototypes(feats, preds, conf)

                # Apply denoising to pseudo-labels
                for i in range(0, min(len(pseudo_labels), len(target_tensors)), 16):
                    batch = torch.stack(
                        target_tensors[i:i+16]
                    ).to(self.state.config.device)
                    feats = extract_features_unet(adapted, batch)
                    pl_batch = torch.stack(pseudo_labels[i:i+16]).to(self.state.config.device)
                    # Upsample features to match pseudo-label spatial dims
                    if feats.shape[-2:] != pl_batch.shape[-2:]:
                        feats = F.interpolate(feats, size=pl_batch.shape[-2:],
                                              mode="bilinear", align_corners=False)
                    denoised, reliable = denoiser.denoise(feats, pl_batch)
                    # Mark unreliable pixels as ignore (255)
                    denoised[~reliable] = 255
                    for j in range(denoised.shape[0]):
                        if i + j < len(pseudo_labels):
                            changed = (denoised[j].cpu() != pseudo_labels[i+j]).sum().item()
                            n_denoised += changed
                            pseudo_labels[i+j] = denoised[j].cpu()

                print(f"    PrototypeDenoiser: {n_denoised:,} pixels denoised")
            except Exception as e:
                print(f"    Prototype denoising failed ({e}), using raw pseudo-labels")

        # Store for use in adaptation loop
        self.adaptation_agent._pseudo_labels = pseudo_labels
        del model

    def _apply_histogram_matching(self):
        """Transform target images via CDF-based histogram matching.

        Maps each target intensity percentile to the corresponding source
        percentile. For contrast reversal (T1->T2), this reduces the global
        distribution gap by making target images look statistically similar
        to source, enabling the model's learned features to work better.
        """
        print(f"\n  === Histogram Matching (contrast reversal preprocessing) ===")

        source_imgs = self.state.source_images
        target_imgs = self.state.target_images

        src_mean = np.mean([img.mean() for img in source_imgs[:100]])
        src_std = np.mean([img.std() for img in source_imgs[:100]])
        tgt_before_mean = np.mean([img.mean() for img in target_imgs])

        # Build reference source CDF from aggregated pixels
        src_pixels = np.concatenate(
            [img[:,:,0].ravel() if img.ndim == 3 else img.ravel()
             for img in source_imgs[:100]]
        )
        src_values, src_counts = np.unique(
            (src_pixels * 255).astype(np.uint8), return_counts=True)
        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]

        def match_cdf(target_ch):
            t_values, t_idx, t_counts = np.unique(
                (target_ch * 255).astype(np.uint8).ravel(),
                return_inverse=True, return_counts=True)
            t_cdf = np.cumsum(t_counts).astype(np.float64)
            t_cdf /= t_cdf[-1]
            mapped = np.interp(t_cdf, src_cdf, src_values)
            return mapped[t_idx].reshape(target_ch.shape) / 255.0

        matched_targets = []
        for tgt in target_imgs:
            if tgt.ndim == 3 and tgt.shape[2] == 3:
                if np.allclose(tgt[:,:,0], tgt[:,:,1]):
                    ch = match_cdf(tgt[:,:,0])
                    matched = np.stack([ch]*3, axis=-1)
                else:
                    matched = np.stack(
                        [match_cdf(tgt[:,:,c]) for c in range(3)], axis=-1)
            else:
                matched = match_cdf(tgt)
            matched_targets.append(np.clip(matched, 0, 1).astype(np.float32))

        self.state.target_images = matched_targets

        tgt_after_mean = np.mean([img.mean() for img in matched_targets])
        print(f"    Source: mean={src_mean:.3f}, std={src_std:.3f}")
        print(f"    Target mean: {tgt_before_mean:.3f} -> {tgt_after_mean:.3f}")
        print(f"    Matched {len(matched_targets)} target images")

        # Invalidate tensor cache so adaptation uses new images
        self.adaptation_agent._target_tensors = None

    def _should_stop(self, iteration: int) -> bool:
        """Check if adaptation should stop early."""
        # Use GT dice for early stopping when available
        hist = self.state.iteration_history
        if len(hist) < 2:
            return False

        # Check GT dice trend if available
        if "avg_dice" in hist[-1] and "avg_dice" in hist[-2]:
            cur = hist[-1]["avg_dice"]
            prev = hist[-2]["avg_dice"]
            if cur < prev - 0.02:
                print(f"    GT dice degraded ({prev:.4f} -> {cur:.4f}), "
                      f"reverting to best ({self.state.best_quality_score:.4f}).")
                if self.state.best_model_state:
                    self.state.adapted_model.load_state_dict(self.state.best_model_state)
                return True
            return False

        # Fallback to quality scores
        scores = self.state.quality_scores
        if len(scores) < 2:
            return False
        if scores[-1] < scores[-2] - 0.02:
            print(f"    Quality degraded ({scores[-2]:.4f} -> {scores[-1]:.4f}), "
                  f"reverting to best ({self.state.best_quality_score:.4f}).")
            if self.state.best_model_state:
                self.state.adapted_model.load_state_dict(self.state.best_model_state)
            return True
        return False

    def _should_rediagnose(self, iteration: int) -> bool:
        """Check if re-diagnosis is needed (quality stagnation)."""
        scores = self.state.quality_scores
        if len(scores) < 3:
            return False
        # Stagnation: last 2 scores within 1% of each other
        return abs(scores[-1] - scores[-2]) < 0.01 and iteration < 4

    def _compile_results(self, total_time: float) -> Dict:
        return {
            "knowledge": self.state.rag_knowledge,
            "shift_report": self.state.shift_report.summary()
            if self.state.shift_report else "",
            "failure_report": self.state.failure_report.to_dict()
            if self.state.failure_report else {},
            "plan": self.state.adaptation_plan.get("rationale", ""),
            "history": self.state.iteration_history,
            "total_time": total_time,
            "pipeline_log": self.state.pipeline_log,
            "postprocess_results": self.state.postprocess_results,
        }

    def save_pipeline_log(self, path: str):
        """Save the full pipeline log for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state.pipeline_log, f, indent=2, default=str)
        print(f"Pipeline log saved to {path}")
