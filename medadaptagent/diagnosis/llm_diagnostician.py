"""
LLM Diagnostician: Uses an LLM to perform causal reasoning about failures
and recommend specific adaptation strategies.

Takes structured failure reports + domain knowledge → actionable recommendations.
"""

import json
from typing import Dict, List, Optional


class LLMDiagnostician:
    """LLM-based failure diagnosis and strategy recommendation.

    Consumes quantitative diagnostics (entropy distributions, failure type
    proportions, feature statistics) and produces structured adaptation plans.
    """

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514",
                 use_rule_fallback: bool = True):
        """
        Args:
            api_key: Anthropic API key. If None, uses rule-based fallback.
            model: LLM model to use.
            use_rule_fallback: If True, fall back to rules when LLM unavailable.
        """
        self.api_key = api_key
        self.model = model
        self.use_rule_fallback = use_rule_fallback
        self.client = None

        if api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                print("Warning: anthropic package not installed. Using rule-based fallback.")

    def diagnose(self, failure_report: dict, domain_profile: dict,
                 knowledge_base: dict = None) -> Dict:
        """Generate diagnosis and adaptation recommendation.

        Args:
            failure_report: From FailureClassifier.to_dict().
            domain_profile: Domain characteristics (modality, body region, etc.).
            knowledge_base: Retrieved literature knowledge.

        Returns:
            Structured recommendation dict.
        """
        if self.client:
            return self._llm_diagnose(failure_report, domain_profile, knowledge_base)
        elif self.use_rule_fallback:
            return self._rule_based_diagnose(failure_report, domain_profile)
        else:
            raise RuntimeError("No LLM client and rule fallback disabled.")

    def _llm_diagnose(self, failure_report: dict, domain_profile: dict,
                      knowledge_base: dict = None) -> Dict:
        """Use LLM for causal reasoning about failures."""
        prompt = self._build_diagnosis_prompt(failure_report, domain_profile, knowledge_base)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            # Parse structured JSON from response
            recommendation = self._parse_llm_response(text)
            return recommendation

        except Exception as e:
            print(f"LLM diagnosis failed: {e}. Falling back to rules.")
            return self._rule_based_diagnose(failure_report, domain_profile)

    def _build_diagnosis_prompt(self, failure_report: dict, domain_profile: dict,
                                 knowledge_base: dict = None) -> str:
        """Build the diagnosis prompt for the LLM."""
        prompt = f"""You are a medical image analysis expert diagnosing why a segmentation model fails on a new domain.

## Domain Context
- Source modality: {domain_profile.get('source_modality', 'unknown')}
- Target modality: {domain_profile.get('target_modality', 'unknown')}
- Body region: {domain_profile.get('body_region', 'unknown')}
- Task: {domain_profile.get('task_description', 'segmentation')}

## Failure Analysis Results
{json.dumps(failure_report, indent=2)}

## Available Adaptation Strategies
1. **Style Adapter** (~0.1M params): Frequency-domain transform for appearance/contrast gaps. Best for: hallucinations from style mismatch, low-frequency domain differences.
2. **Boundary Adapter** (~0.3M params): Boundary-aware attention. Best for: imprecise boundaries, edge artifacts.
3. **Scale Adapter** (~0.5M params): Multi-resolution LoRA. Best for: resolution mismatches, object size differences across domains.
4. **Semantic Adapter** (~0.2M params): Text-guided prototypes. Best for: missed structures, class confusion, semantic alignment.

## Self-Supervised Objectives
- (a) Entropy minimization: reduces uncertainty on confident regions
- (b) Consistency regularization: enforces augmentation invariance
- (c) Pseudo-label self-training: trains on high-confidence predictions
- (d) Text-guided alignment: aligns visual features with BiomedCLIP text embeddings

"""
        if knowledge_base:
            prompt += f"""## Retrieved Knowledge
{json.dumps(knowledge_base, indent=2)[:3000]}

"""
        prompt += """## Task
Based on the failure analysis, provide a structured JSON recommendation:
```json
{
    "diagnosed_causes": ["list of probable causes"],
    "recommended_adapters": ["list of adapter names to activate"],
    "adapter_priority": ["ordered list, most important first"],
    "self_supervised_objectives": ["list of objectives to use"],
    "hyperparameter_suggestions": {
        "learning_rate": float,
        "entropy_threshold": float,
        "pseudo_label_threshold": float,
        "consistency_weight": float,
        "max_steps": int
    },
    "reasoning": "Brief explanation of the diagnostic reasoning",
    "confidence": float  // 0-1
}
```

Respond with ONLY the JSON block."""
        return prompt

    def _parse_llm_response(self, text: str) -> Dict:
        """Parse structured JSON from LLM response."""
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Fallback: extract key information manually
        return {
            "diagnosed_causes": ["Could not parse LLM response"],
            "recommended_adapters": ["style", "boundary"],
            "adapter_priority": ["style", "boundary"],
            "self_supervised_objectives": ["entropy_min", "consistency"],
            "hyperparameter_suggestions": {
                "learning_rate": 1e-4,
                "entropy_threshold": 0.3,
                "pseudo_label_threshold": 0.8,
                "consistency_weight": 1.0,
                "max_steps": 200,
            },
            "reasoning": f"LLM response parsing failed. Raw: {text[:200]}",
            "confidence": 0.3,
        }

    def _rule_based_diagnose(self, failure_report: dict,
                              domain_profile: dict) -> Dict:
        """Rule-based fallback when LLM is unavailable.

        Maps failure type proportions directly to adapter recommendations.
        """
        type_dist = failure_report.get("type_distribution", {})

        adapters = set()
        priorities = []
        objectives = ["entropy_min", "consistency"]

        # Rule: boundary errors → boundary + style adapters
        if type_dist.get("boundary_error", 0) > 0.3:
            adapters.update(["boundary", "style"])
            priorities.extend(["boundary", "style"])
            objectives.append("pseudo_label")

        # Rule: misses → semantic adapter
        if type_dist.get("miss", 0) > 0.2:
            adapters.add("semantic")
            priorities.append("semantic")
            objectives.append("text_alignment")

        # Rule: hallucinations → style adapter
        if type_dist.get("hallucination", 0) > 0.2:
            adapters.add("style")
            if "style" not in priorities:
                priorities.append("style")

        # Rule: class confusion → semantic + scale
        if type_dist.get("class_confusion", 0) > 0.2:
            adapters.update(["semantic", "scale"])
            priorities.extend(["semantic", "scale"])
            objectives.append("text_alignment")

        # Default: if no strong signal, use style + semantic
        if not adapters:
            adapters = {"style", "semantic"}
            priorities = ["style", "semantic"]

        # Severity-based learning rate
        severity = failure_report.get("overall_severity", 0.5)
        lr = 1e-4 if severity < 0.5 else 5e-5

        # Cross-modality needs more aggressive style adaptation
        shift_type = domain_profile.get("shift_type", "")
        if "cross-modality" in shift_type:
            adapters.add("style")
            lr = 1e-4

        causes = []
        if type_dist.get("boundary_error", 0) > 0.3:
            causes.append("Domain shift causing boundary degradation")
        if type_dist.get("miss", 0) > 0.2:
            causes.append("Target structures not recognized due to appearance change")
        if type_dist.get("hallucination", 0) > 0.2:
            causes.append("False activations from domain-specific artifacts")
        if type_dist.get("class_confusion", 0) > 0.2:
            causes.append("Class ambiguity increased by domain shift")
        if not causes:
            causes.append("General domain shift affecting model performance")

        return {
            "diagnosed_causes": causes,
            "recommended_adapters": list(adapters),
            "adapter_priority": list(dict.fromkeys(priorities)),
            "self_supervised_objectives": list(dict.fromkeys(objectives)),
            "hyperparameter_suggestions": {
                "learning_rate": lr,
                "entropy_threshold": 0.3,
                "pseudo_label_threshold": 0.8 + 0.05 * severity,
                "consistency_weight": 1.0,
                "max_steps": int(200 + 100 * severity),
            },
            "reasoning": f"Rule-based: dominant failure={max(type_dist, key=type_dist.get) if type_dist else 'none'}",
            "confidence": 0.6,
        }
