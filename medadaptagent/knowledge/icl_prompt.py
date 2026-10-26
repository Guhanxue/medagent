"""
In-Context Learning prompt for LLM knowledge generation.

Teaches the LLM to produce calibrated domain knowledge by showing
exemplar entries with explicit reasoning about:
  1. WHY each shift characteristic matters for adaptation
  2. WHAT adaptation strategies work and WHY
  3. WHAT strategies FAIL and WHY (most important!)
  4. HOW to calibrate severity (not everything is "severe")

The exemplars encode lessons from 16 iterations of empirical debugging.
"""

ICL_KNOWLEDGE_PROMPT = """You are a medical imaging domain adaptation expert. Your task is to analyze the domain shift between a source and target dataset and produce a structured knowledge entry that will guide an automated adaptation system.

## CRITICAL CALIBRATION RULES

These rules were learned from extensive empirical testing. Follow them carefully:

1. **Severity calibration**: Most cross-device/cross-site shifts are MILD to MODERATE, not severe. Only cross-modality (CT→MRI) or contrast reversal (T1→T2) are genuinely severe. Do NOT overestimate severity.

2. **"What fails" is more important than "what works"**: The system needs to know what to AVOID. Be specific about failure mechanisms.

3. **Scope recommendation**: Start conservative.
   - Same modality, mild shift → bn_only (BN statistics correction is sufficient)
   - Same modality, moderate shift → bn+output (output head remapping)
   - Cross-modality → bn+output (NOT decoder — overfitting risk on small target sets)
   - NEVER recommend "all" scope for zero-shot — only for few-shot with N≥5

4. **Entropy minimization is DANGEROUS** when the model hallucinates (predicts wrong classes confidently). It reinforces wrong predictions. Only recommend entropy when model predictions are structurally correct.

5. **BN reset is cheap and safe** — it should almost always be the first step. It handles most mild shifts entirely.

## EXEMPLAR 1: Cross-site MRI (MILD shift)

**Task**: Prostate segmentation, Site A (Siemens 3T) → Site B (GE 3T)

**Analysis**:
```json
{{
    "shift_types": ["Cross-vendor MRI: same T2w sequence, different scanner manufacturer. Intensity differences from coil sensitivity, reconstruction algorithms, and field inhomogeneity."],
    "intensity_mapping_type": "approximately_linear",
    "standardization_gap": "moderate",
    "contrast_mechanism_change": "same",
    "noise_model_change": "similar",
    "boundary_quality_change": "same",
    "dominant_shift_factor": "Scanner vendor differences in reconstruction and coil sensitivity. Same tissue contrast, different intensity distribution.",
    "recommended_scope": "bn_only",
    "recommended_mode": "per_image_tent",
    "recommended_lr_range": "standard_1e3",
    "what_will_fail": [
        "Aggressive decoder adaptation — destroys well-learned prostate features for a shift that BN reset can handle",
        "High learning rates (>1e-3) — model already works well, aggressive updates cause divergence",
        "Multiple iterations when model is already well-calibrated — each iteration adds noise with diminishing returns"
    ]
}}
```

**Key lesson**: The model already segments prostate well on source. The shift is just BN statistics mismatch. Aggressive adaptation HURTS because it undoes good features for a problem that BN reset solves.

## EXEMPLAR 2: Contrast reversal (MODERATE-SEVERE shift)

**Task**: Brain tumor segmentation, T1-weighted → T2-weighted MRI

**Analysis**:
```json
{{
    "shift_types": ["Cross-sequence MRI: T1 and T2 weightings produce opposite contrast for key tissues. CSF is dark in T1 but bright in T2. Gray-white matter contrast is reversed."],
    "intensity_mapping_type": "non_linear",
    "standardization_gap": "moderate",
    "contrast_mechanism_change": "different",
    "noise_model_change": "similar",
    "boundary_quality_change": "similar",
    "dominant_shift_factor": "Contrast reversal: tissue intensities are non-linearly remapped between T1 and T2. The model's learned intensity→class mapping is fundamentally wrong.",
    "recommended_scope": "bn+output",
    "recommended_mode": "batch_multi_loss",
    "recommended_lr_range": "conservative_5e4",
    "what_will_fail": [
        "Entropy minimization — model confidently predicts CSF as tumor (100% hallucination). Entropy sharpens these WRONG predictions, making adaptation worse than source-only.",
        "Multiple iterations — errors compound. Each iteration reinforces hallucinated predictions.",
        "Aggressive decoder adaptation — overfits to hallucinated pseudo-labels",
        "BN-only without histogram matching — insufficient for contrast reversals"
    ]
}}
```

**Key lesson**: When the model is CONFIDENTLY WRONG (high hallucination), entropy minimization is catastrophic. Histogram matching as preprocessing + conservative 1-iteration adaptation is the safest approach.

## EXEMPLAR 3: Cross-modality (SEVERE shift)

**Task**: Liver segmentation, CT → MRI

**Analysis**:
```json
{{
    "shift_types": ["Cross-modality: CT uses X-ray attenuation (Hounsfield units), MRI uses proton relaxation (T2-SPIR). Completely different imaging physics. Bone is bright in CT but dark in MRI. Soft tissue contrast patterns are fundamentally different."],
    "intensity_mapping_type": "fundamentally_different",
    "standardization_gap": "severe",
    "contrast_mechanism_change": "fundamentally_different",
    "noise_model_change": "different",
    "boundary_quality_change": "blurrier",
    "dominant_shift_factor": "Fundamentally different imaging physics. No pixel-level correspondence between CT and MRI intensity values.",
    "recommended_scope": "bn+output",
    "recommended_mode": "batch_multi_loss",
    "recommended_lr_range": "conservative_5e4",
    "what_will_fail": [
        "BN-only adaptation — corrects statistics but features are fundamentally mismatched across modalities",
        "Entropy alone — model produces near-zero foreground, entropy has no useful gradient signal",
        "Self-training without FDA — pseudo-labels from source model on MRI are extremely noisy",
        "Per-image Tent — too noisy for cross-modality, needs batch-level gradient signal"
    ]
}}
```

**Key lesson**: Cross-modality is genuinely severe. FDA (Fourier Domain Adaptation) bridges the low-frequency appearance gap. Without it, the model produces near-zero foreground on the target modality.

## EXEMPLAR 4: Multi-vendor cardiac (VERY MILD shift)

**Task**: Cardiac segmentation, Siemens MRI → Philips MRI

**Analysis**:
```json
{{
    "shift_types": ["Multi-vendor MRI: same bSSFP sequence, different scanner vendor. Minimal intensity difference, slight texture variation from reconstruction."],
    "intensity_mapping_type": "approximately_linear",
    "standardization_gap": "mild",
    "contrast_mechanism_change": "same",
    "noise_model_change": "similar",
    "boundary_quality_change": "same",
    "dominant_shift_factor": "Very mild vendor-specific differences in bSSFP reconstruction. Model likely already generalizes well.",
    "recommended_scope": "bn_only",
    "recommended_mode": "per_image_tent",
    "recommended_lr_range": "standard_1e3",
    "what_will_fail": [
        "ANY aggressive adaptation — model already works at 83% on target. Adaptation risks hurting well-calibrated classes.",
        "CoTTA/stochastic restoration — augmentation noise exceeds the actual domain gap, causes catastrophic collapse",
        "SFDA methods (DPL, ProDe) — pseudo-label noise on small shifts destroys the already-good predictions"
    ]
}}
```

**Key lesson**: When source-only performance is already high (>80%), the safest strategy is DO NO HARM — either skip adaptation entirely or apply only BN statistics reset. Every other method risks hurting what already works.

---

## YOUR TASK

Now analyze this new source→target pair. Follow the same format and calibration principles as the exemplars above.

**Task**: {task_description}
**Source**: {source_domain}
**Target**: {target_domain}
**Segmentation classes**: {class_descriptions}

{additional_context}

Return a JSON object following the exact same structure as the exemplars. Pay special attention to:
1. Calibrate severity honestly — is this really severe, or just moderate?
2. List specific failure modes in "what_will_fail"
3. Match recommended_scope to the actual severity (don't over-recommend)

```json
{{
    "shift_types": [...],
    "intensity_mapping_type": "...",
    "standardization_gap": "...",
    "contrast_mechanism_change": "...",
    "noise_model_change": "...",
    "boundary_quality_change": "...",
    "dominant_shift_factor": "...",
    "recommended_scope": "...",
    "recommended_mode": "...",
    "recommended_lr_range": "...",
    "what_will_fail": [...]
}}
```"""
