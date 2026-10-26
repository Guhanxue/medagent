"""
MedAdaptAgent Pipeline Configuration.

Defines TaskConfig (task-agnostic) and prompt templates for the 3-step pipeline.
To apply to a new dataset: create a TaskConfig and pass it to MedAdaptAgent.

Example:
    config = TaskConfig(
        task_name="cardiac_ct_to_mri",
        source_domain="CT (Hounsfield units, high bone contrast)",
        target_domain="MRI (T2-weighted, soft tissue contrast)",
        structures={"background": 0, "LV": 1, "MYO": 2, "RV": 3},
        anatomy_priors={
            "LV": {"shape": "elliptical cavity", "area_range": (0.02, 0.15)},
            "MYO": {"shape": "ring around LV", "area_range": (0.03, 0.20)},
            "RV": {"shape": "crescent adjacent to LV", "area_range": (0.02, 0.15)},
        },
        containment_rules=[("MYO", "contains", "LV")],
    )
    agent = MedAdaptAgent(model, src_imgs, tgt_imgs, config=config)
    results = agent.run()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class TaskConfig:
    """Task-agnostic configuration for MedAdaptAgent pipeline.

    This is the ONLY thing you need to change for a new dataset/task.
    """
    # --- Task identity ---
    task_name: str = ""
    source_domain: str = ""     # e.g., "REFUGE fundus (Zeiss Visucam 500)"
    target_domain: str = ""     # e.g., "RIM-ONE DL (multiple hospitals)"

    # --- Segmentation classes ---
    # Map from class name -> label index. Index 0 is always background.
    structures: Dict[str, int] = field(default_factory=lambda: {"background": 0})
    # Human-readable descriptions for LLM reasoning
    class_descriptions: Dict[str, str] = field(default_factory=dict)

    # --- Anatomy priors (optional but helps) ---
    # Per-class shape/size constraints for pseudo-label refinement
    anatomy_priors: Dict[str, Dict] = field(default_factory=dict)
    # Structural relationships: list of (classA, relation, classB)
    # Relations: "contains" (A contains B), "adjacent" (A touches B),
    #            "disjoint" (A and B don't overlap)
    containment_rules: List[Tuple[str, str, str]] = field(default_factory=list)

    # --- Adaptation settings ---
    image_size: int = 256
    max_iterations: int = 5
    device: str = "cuda"

    # --- Few-shot calibration ---
    # K target cases reserved as a held-out validation set, used to
    # compute few_shot_dice each iteration. Replaces the unreliable
    # unsupervised quality score for best-model selection.
    # Set to 0 to disable (returns to fully unsupervised mode).
    n_calibration_samples: int = 5  # was 3; bumped for lower few_shot_dice variance

    # --- Per-channel semantics for Agent 1 ---
    # Human-readable name for each input channel, in the same order they
    # appear on the C axis of the image arrays. Used by the Domain Expert
    # to label per-channel grayscale montages and the per-channel intensity
    # statistics table. Examples:
    #   ["R", "G", "B"]              for color fundus
    #   ["T1", "T1ce", "T2", "FLAIR"] for multi-modal brain MRI (BraTS-style)
    #   ["T2W", "DWI", "ADC"]        for multi-phase prostate MRI
    # If left as None, channels are labeled "channel_0", "channel_1", ...
    channel_names: Optional[List[str]] = None

    # --- Training context for Agent 2 (Model Analyst) ---
    # How was the source model trained? These fields help Agent 2 reason
    # about whether a domain shift is large RELATIVE to what training
    # already covered. A model trained with z-score normalization and
    # strong intensity augmentation is much more robust to a -50% mean
    # shift than a model trained with no normalization and no augmentation.
    #
    # training_normalization: how input intensities were normalized during
    # training. One of:
    #   "0-1"        — min-max scaled to [0, 1]
    #   "z-score"    — per-image mean=0, std=1
    #   "instance"   — per-image instance normalization layer
    #   "imagenet"   — ImageNet mean/std
    #   "none"       — raw intensities
    #   "unknown"    — not specified (Agent 2 assumes worst case)
    training_normalization: str = "unknown"
    #
    # training_augmentations: list of augmentation types used during
    # training. Used by Agent 2 to gauge robustness to specific shifts.
    # Examples:
    #   "intensity_jitter"  — robust to mild calibration shifts
    #   "gaussian_noise"    — robust to noise model changes
    #   "gamma"             — robust to non-linear intensity remapping
    #   "flip"              — robust to L/R orientation
    #   "rotate"            — robust to rotation
    #   "scale" / "zoom"    — robust to FOV changes
    #   "elastic"           — robust to geometric distortion
    #   "bias_field"        — robust to MRI coil inhomogeneity
    #   "histogram_match"   — robust to histogram drift
    # Empty list means no augmentation; None means unknown.
    training_augmentations: Optional[List[str]] = None

    @property
    def num_classes(self) -> int:
        return len(self.structures)

    @property
    def fg_classes(self) -> Dict[str, int]:
        """Foreground classes only (exclude background)."""
        return {k: v for k, v in self.structures.items() if v > 0}

    def get_task_description(self) -> str:
        classes = ", ".join(self.fg_classes.keys())
        return (f"{self.task_name}: {classes} segmentation. "
                f"Source: {self.source_domain}. Target: {self.target_domain}.")


# ================================================================
# Prompt templates for the 3-step pipeline
# These are the reusable "reasoning instructions" for each step.
# When an LLM is available, these are sent as prompts.
# When no LLM, the agent uses rule-based fallback with the same structure.
# ================================================================

STEP1_RAG_PROMPT = """You are a medical imaging domain adaptation expert. Analyze the physics-level reasons for domain shift between source and target, and recommend specific adaptation strategies based on published literature.

## Task
{task_description}

## Source domain
{source_domain}

## Target domain
{target_domain}

## Segmentation classes
{class_descriptions}

## Retrieved domain knowledge (from knowledge base)
{domain_knowledge}

## Instructions
Based on the retrieved knowledge AND your own expertise about medical imaging physics, analyze this specific source-to-target shift. Think about:

1. **Imaging physics**: How does image formation differ? (e.g., X-ray attenuation vs MRI relaxation, different camera optics, different scanner vendors)
2. **Intensity mapping**: Is the mapping between source and target intensities linear, non-linear, or completely different? (This determines whether BN recalibration alone can work)
3. **Standardization**: Does either modality have a standardized intensity scale? (CT has HU, MRI/fundus do not)
4. **Noise characteristics**: How do noise models differ? (Gaussian, Rician, Poisson - affects texture features)
5. **Resolution/scale**: Are there resolution or field-of-view differences?
6. **Boundary quality**: Will anatomical boundaries appear sharper or blurrier in the target?
7. **Training protocol**: What preprocessing was used? What might the model have never seen?

Return a JSON object with fine-grained shift characterization:
```json
{{
    "shift_types": ["list of shift types with physics-level explanations"],
    "intensity_mapping_type": "linear | non_linear | fundamentally_different",
    "standardization_gap": "none | mild | moderate | severe",
    "noise_model_change": "same | similar | different | fundamentally_different",
    "resolution_gap": "none | mild | moderate | severe",
    "contrast_mechanism_change": "same | similar | different | fundamentally_different",
    "boundary_quality_change": "sharper | same | blurrier | unpredictable",
    "dominant_shift_factor": "which single factor contributes most to the domain gap",
    "challenges": ["specific challenges for this source->target pair"],
    "literature_strategies": ["what published papers recommend for this type of shift"],
    "what_will_fail": ["approaches that are known to fail for this shift type"],
    "recommended_scope": "bn_only | bn+output | bn+decoder1",
    "recommended_mode": "per_image_tent | batch_multi_loss",
    "recommended_lr_range": "standard_1e3 | conservative_5e4 | very_conservative_3e4",
    "anatomy_priors": {{
        "class_name": {{
            "shape": "expected shape",
            "area_range": [min_fraction, max_fraction]
        }}
    }}
}}
```

Be SPECIFIC to this exact domain pair. Reference the imaging physics, not just generic labels."""


STEP2_VERIFY_PROMPT = """You are a medical image analysis expert. Statistical analysis has been performed comparing source and target domains. Interpret these findings.

## Task
{task_description}

## RAG predictions (Step 1)
{rag_knowledge}

## Statistical findings
- Intensity shift (KL divergence): {intensity_kl:.4f}
  Channel shifts: {channel_shifts}
- Frequency shift: low_freq={low_freq:.4f}, high_freq={high_freq:.4f}
- Scale ratio (target/source foreground): {scale_ratio:.2f}
- Model failure type: {failure_type}
- Model confidence on target: {confidence:.4f}
- Model entropy on target: {entropy:.4f}
- Prediction distribution: {pred_dist}

## Instructions
1. For each RAG prediction, state whether it is CONFIRMED or UNVERIFIED by the statistics.
2. Note any NEW findings not predicted by RAG.
3. Rank the verified shifts by severity.
4. Return JSON:
```json
{{
    "confirmed_shifts": ["list of confirmed shift types"],
    "unverified_shifts": ["list of unverified predictions"],
    "new_findings": ["list of unexpected findings"],
    "severity_ranking": ["most severe to least severe"],
    "interpretation": "brief interpretation of what these shifts mean for adaptation"
}}
```"""


STEP3_ADAPT_PROMPT = """You are a medical image analysis expert. Design a targeted adaptation strategy based on verified domain shift diagnosis.

## Task
{task_description}

## Verified diagnosis
{diagnosis_summary}

## Available adaptation mechanisms
1. **BN statistics reset**: Replace source BatchNorm running stats with target stats via forward pass. Corrects intensity/contrast distribution mismatch. Cost: 0 trainable params.
2. **Entropy minimization**: Encourage confident predictions on target. Reduces uncertainty. Risk: can reinforce wrong predictions.
3. **Consistency regularization**: Predictions should be invariant to augmentations (flip, rotate). Provides implicit supervision signal.
4. **Diversity regularization**: Prevent class collapse (all pixels predicted as one class). KL divergence toward uniform class distribution.
5. **Pseudo-label self-training**: Generate pseudo-labels from confident predictions, use as supervision. Can incorporate anatomy priors.
6. **Anatomy-aware refinement**: Enforce structural constraints (e.g., cup inside disc, LV inside MYO).

## Anatomy priors
{anatomy_priors}

## Containment rules
{containment_rules}

## Per-class performance after initial BN adaptation
{per_class_metrics}

## Instructions
Design the adaptation strategy. For each mechanism, decide whether to use it and set its weight.
Return JSON:
```json
{{
    "learning_rate": float,
    "num_steps": int,
    "loss_weights": {{
        "entropy": float,
        "consistency": float,
        "diversity": float,
        "pseudo_label": float,
        "anatomy": float
    }},
    "trainable_scope": "bn_only | bn+output | bn+decoder1 | bn+all_decoder",
    "use_anatomy_refinement": true/false,
    "rationale": "explanation of why these specific choices"
}}
```"""


# ================================================================
# Pre-built TaskConfigs for the 3 benchmark datasets
# Also available in tasks/{fundus,cardiac,crossmoda}/config.py
# ================================================================

FUNDUS_CONFIG = TaskConfig(
    task_name="fundus_cross_device",
    source_domain="REFUGE (Zeiss Visucam 500 + Canon CR-2, full retina FOV)",
    target_domain="RIM-ONE DL (multiple hospitals, various cameras, disc-centered crop)",
    channel_names=["R", "G", "B"],
    structures={"background": 0, "disc": 1, "cup": 2},
    class_descriptions={
        "disc": "Optic disc: bright elliptical region where optic nerve enters retina",
        "cup": "Optic cup: central depression within the optic disc, paler than disc rim",
    },
    anatomy_priors={
        "disc": {
            "shape": "roughly circular/elliptical",
            "area_range": (0.02, 0.40),
            "description": "Single connected component, 10-40% of cropped image",
        },
        "cup": {
            "shape": "roughly circular, inside disc",
            "area_range": (0.005, 0.25),
            "description": "Single connected component, CDR typically 0.3-0.7",
        },
    },
    containment_rules=[("disc", "contains", "cup")],
)

CARDIAC_CONFIG = TaskConfig(
    task_name="cardiac_cross_modality",
    source_domain="CT (Hounsfield units, high bone/calcification contrast)",
    target_domain="MRI (balanced SSFP, high soft-tissue contrast)",
    structures={"background": 0, "MYO": 1, "LA": 2, "LV": 3, "AA": 4},
    class_descriptions={
        "MYO": "Myocardium: heart muscle wall, ring-like structure",
        "LA": "Left atrium: upper-left chamber of the heart",
        "LV": "Left ventricle: lower-left pumping chamber, elliptical cavity",
        "AA": "Ascending aorta: large vessel above the heart",
    },
    anatomy_priors={
        "MYO": {"shape": "ring/annular", "area_range": (0.02, 0.15)},
        "LV": {"shape": "elliptical cavity", "area_range": (0.01, 0.10)},
        "LA": {"shape": "irregular chamber", "area_range": (0.01, 0.10)},
        "AA": {"shape": "circular cross-section", "area_range": (0.005, 0.05)},
    },
    containment_rules=[("MYO", "contains", "LV")],
)

CROSSMODA_CONFIG = TaskConfig(
    task_name="crossmoda_cross_sequence",
    source_domain="ceT1 MRI (contrast-enhanced T1, tumor enhances brightly)",
    target_domain="hrT2 MRI (high-resolution T2, tumor appears hyperintense)",
    structures={"background": 0, "VS": 1, "cochlea": 2},
    class_descriptions={
        "VS": "Vestibular schwannoma: tumor at cerebellopontine angle",
        "cochlea": "Cochlea: spiral-shaped inner ear structure",
    },
    anatomy_priors={
        "VS": {"shape": "irregular mass", "area_range": (0.001, 0.10)},
        "cochlea": {"shape": "small spiral/snail-like", "area_range": (0.0005, 0.01)},
    },
    containment_rules=[],
)

# ================================================================
# IXI Brain MRI (3 scanners, SUBTLE shift)
# ================================================================
IXI_CONFIG = TaskConfig(
    task_name="ixi_cross_scanner",
    source_domain="Hammersmith Hospital (Philips Intera 3T)",
    target_domain="Guy's Hospital (Philips Gyroscan 1.5T) or IOP (GE 1.5T)",
    structures={"background": 0, "brain": 1},
    class_descriptions={
        "brain": "Brain parenchyma: all intracranial tissue within skull",
    },
    anatomy_priors={
        "brain": {
            "shape": "large connected mass filling skull cavity",
            "area_range": (0.15, 0.75),
            "description": "Single large connected component, 15-75% of axial slice",
        },
    },
    containment_rules=[],
)

# ================================================================
# CC-359 Brain MRI (6 scanner domains, SUBTLE shift)
# ================================================================
CC359_CONFIG = TaskConfig(
    task_name="cc359_cross_scanner",
    source_domain="Siemens 3T brain MRI",
    target_domain="GE/Philips 1.5T/3T brain MRI (5 target domains)",
    structures={"background": 0, "brain": 1},
    class_descriptions={
        "brain": "Brain parenchyma: intracranial tissue segmented from skull",
    },
    anatomy_priors={
        "brain": {
            "shape": "large connected mass",
            "area_range": (0.15, 0.75),
            "description": "Single connected component, gold-standard mask",
        },
    },
    containment_rules=[],
)

# ================================================================
# AMOS CT → MRI (15 abdominal organs, MEDIUM-LARGE shift)
# ================================================================
AMOS_CONFIG = TaskConfig(
    task_name="amos_cross_modality",
    source_domain="CT (500 volumes, HU windowing, high bone contrast)",
    target_domain="MRI (100 volumes, T1/T2 sequences, high soft-tissue contrast)",
    structures={
        "background": 0, "spleen": 1, "right_kidney": 2, "left_kidney": 3,
        "gallbladder": 4, "esophagus": 5, "liver": 6, "stomach": 7,
        "aorta": 8, "IVC": 9, "pancreas": 10, "right_adrenal": 11,
        "left_adrenal": 12, "duodenum": 13, "bladder": 14, "prostate_uterus": 15,
    },
    class_descriptions={
        "spleen": "Spleen: left upper quadrant, homogeneous parenchymal organ",
        "right_kidney": "Right kidney: retroperitoneal, bean-shaped",
        "left_kidney": "Left kidney: retroperitoneal, bean-shaped",
        "gallbladder": "Gallbladder: small sac under liver",
        "esophagus": "Esophagus: tubular structure, posterior mediastinum",
        "liver": "Liver: large right upper quadrant organ",
        "stomach": "Stomach: left upper quadrant, variable shape",
        "aorta": "Aorta: large vessel, circular cross-section",
        "IVC": "Inferior vena cava: large vein, right of aorta",
        "pancreas": "Pancreas: retroperitoneal, elongated gland",
        "right_adrenal": "Right adrenal gland: small, above right kidney",
        "left_adrenal": "Left adrenal gland: small, above left kidney",
        "duodenum": "Duodenum: C-shaped, around pancreatic head",
        "bladder": "Bladder: pelvic midline, variable fill",
        "prostate_uterus": "Prostate/uterus: pelvic floor organ",
    },
    anatomy_priors={
        "liver": {"shape": "large irregular", "area_range": (0.05, 0.40)},
        "spleen": {"shape": "ovoid", "area_range": (0.01, 0.10)},
        "right_kidney": {"shape": "bean-shaped", "area_range": (0.005, 0.06)},
        "left_kidney": {"shape": "bean-shaped", "area_range": (0.005, 0.06)},
        "aorta": {"shape": "circular", "area_range": (0.001, 0.02)},
        "IVC": {"shape": "circular/elliptical", "area_range": (0.001, 0.02)},
        "pancreas": {"shape": "elongated", "area_range": (0.005, 0.05)},
        "stomach": {"shape": "irregular crescent", "area_range": (0.01, 0.15)},
        "gallbladder": {"shape": "small ovoid", "area_range": (0.001, 0.03)},
        "bladder": {"shape": "round/ovoid", "area_range": (0.005, 0.10)},
    },
    containment_rules=[],
    image_size=256,
)

# ================================================================
# Prostate Multi-Site MRI (6 sites, SUBTLE shift)
# ================================================================
PROSTATE_CONFIG = TaskConfig(
    task_name="prostate_cross_site",
    source_domain="Site A (Siemens 3T, RUNMC)",
    target_domain="Sites B-F (various vendors/field strengths)",
    structures={"background": 0, "prostate": 1, "peripheral_zone": 2},
    class_descriptions={
        "prostate": "Prostate gland: pelvic organ, roughly elliptical in axial view",
        "peripheral_zone": "Peripheral zone: outer region of prostate, higher signal on T2",
    },
    anatomy_priors={
        "prostate": {
            "shape": "roughly elliptical",
            "area_range": (0.02, 0.20),
            "description": "Single connected component, 2-20% of axial slice",
        },
        "peripheral_zone": {
            "shape": "crescent/ring around transition zone",
            "area_range": (0.005, 0.10),
        },
    },
    containment_rules=[("prostate", "contains", "peripheral_zone")],
)


# ================================================================
# CHAOS CT -> MR (Cross-modality, SEVERE shift)
# ================================================================
CHAOS_CONFIG = TaskConfig(
    task_name="chaos_cross_modality",
    source_domain="CT abdominal scans (20 patients, CHAOS challenge)",
    target_domain="MRI T2-SPIR abdominal scans (20 patients, CHAOS challenge)",
    structures={"background": 0, "liver": 1},
    class_descriptions={
        "liver": "Liver: largest solid abdominal organ, right upper quadrant",
    },
    anatomy_priors={
        "liver": {
            "shape": "large, irregular, roughly triangular in axial view",
            "area_range": (0.05, 0.50),
            "description": "Single connected component, largest abdominal organ",
        },
    },
    containment_rules=[],
    image_size=256,
    max_iterations=3,
)


# ================================================================
# BraTS T1 -> T2 (Cross-sequence, MODERATE shift)
# ================================================================
BRATS_CONFIG = TaskConfig(
    task_name="brats_t1_to_t2",
    source_domain="T1-weighted brain MRI (BraTS/MSD, 484 patients, co-registered, skull-stripped)",
    target_domain="T2-weighted brain MRI (same patients, different contrast mechanism)",
    structures={"background": 0, "tumor": 1},
    class_descriptions={
        "tumor": "Brain tumor: includes enhancing, non-enhancing, and peritumoral edema",
    },
    containment_rules=[],
    anatomy_priors={},
    image_size=256,
    max_iterations=3,
)


# ================================================================
# M&Ms Cardiac Multi-Vendor (SUBTLE shift)
# ================================================================
MNMS_CONFIG = TaskConfig(
    task_name="mnms_cross_vendor",
    source_domain="Siemens MRI scanner (vendor A)",
    target_domain="Philips/Canon/GE MRI scanners (vendors B/C/D)",
    structures={"background": 0, "LV": 1, "MYO": 2, "RV": 3},
    class_descriptions={
        "LV": "Left ventricle cavity",
        "MYO": "Myocardium (heart muscle wall)",
        "RV": "Right ventricle cavity",
    },
    anatomy_priors={
        "LV": {"area_range": (0.02, 0.25)},
        "MYO": {"area_range": (0.02, 0.25)},
        "RV": {"area_range": (0.01, 0.20)},
    },
    containment_rules=[],
)
