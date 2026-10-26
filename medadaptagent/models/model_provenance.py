"""
Model Provenance Reader — auto-extract training context from external models.

When loading someone else's pretrained model, the agent needs to know:
  - What dataset was it trained on?
  - What modality / imaging protocol?
  - What classes does it segment?
  - What preprocessing was applied?
  - What architecture / training setup?

This module reads that information from:
  1. MONAI bundle metadata (configs, docs)
  2. GitHub README files
  3. Paper abstracts (from checkpoint metadata or linked papers)
  4. Checkpoint metadata (training config stored in state dict)
  5. LLM-based parsing of unstructured documentation

The extracted ModelProvenance feeds into CGFD hypothesis generation
for better clinical reasoning about why the model might fail.

Usage:
    from medadaptagent.models.model_provenance import read_provenance

    # From MONAI bundle
    prov = read_provenance("monai://swin_unetr_btcv_segmentation")

    # From local checkpoint + README
    prov = read_provenance("/path/to/model.pt",
                            readme_path="/path/to/README.md")

    # From GitHub URL
    prov = read_provenance("https://github.com/user/repo",
                            llm_client=anthropic_client)
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class ModelProvenance:
    """Everything we know about how an external model was trained."""

    # Identity
    model_name: str = "unknown"
    architecture: str = "unknown"  # "unet", "swin_unetr", "nnunet", etc.
    source_url: str = ""           # GitHub/HuggingFace URL
    paper_title: str = ""
    paper_url: str = ""

    # Training data
    training_dataset: str = ""         # "BTCV", "TotalSegmentator", etc.
    training_modality: str = ""        # "CT", "MRI T1", "fundus", etc.
    training_classes: List[str] = field(default_factory=list)
    num_training_classes: int = 0
    training_samples: int = 0          # approximate number of training images

    # Preprocessing
    preprocessing: Dict[str, str] = field(default_factory=dict)
    # e.g., {"normalization": "HU clipping [-175, 250]",
    #        "spacing": "1.5x1.5x2.0mm", "input_size": "96x96x96"}

    # Architecture details
    input_channels: int = 1
    input_size: str = ""               # "96x96x96", "256x256", etc.
    norm_type: str = ""                # "BatchNorm", "InstanceNorm", "LayerNorm"
    total_params: int = 0

    # Performance (if reported)
    reported_performance: Dict[str, float] = field(default_factory=dict)
    # e.g., {"mean_dice": 0.82, "liver_dice": 0.96}

    # Raw text for LLM reasoning
    raw_description: str = ""

    # Confidence in extracted info
    confidence: str = "low"  # "low", "medium", "high"

    def to_domain_knowledge(self) -> Dict:
        """Convert provenance to domain knowledge dict for CGFD."""
        return {
            "source_model_info": {
                "name": self.model_name,
                "architecture": self.architecture,
                "trained_on": self.training_dataset,
                "modality": self.training_modality,
                "classes": self.training_classes,
                "preprocessing": self.preprocessing,
                "norm_type": self.norm_type,
                "reported_performance": self.reported_performance,
            },
            "expected_shift_causes": self._infer_shift_causes(),
            "raw_description": self.raw_description,
        }

    def _infer_shift_causes(self) -> List[str]:
        """Infer likely causes of domain shift from provenance."""
        causes = []
        mod = self.training_modality.lower()

        if "ct" in mod:
            causes.append("Model trained on CT - HU-based normalization and "
                         "bone/air contrast. MRI target will have completely "
                         "different intensity distribution.")
            causes.append("CT-trained BatchNorm statistics will be mismatched "
                         "for MRI intensity ranges.")
        if "mri" in mod and "t1" in mod:
            causes.append("Model trained on T1 MRI - T2 target will have "
                         "inverted contrast (CSF bright vs dark).")
        if "3t" in mod.lower() or "1.5t" in mod.lower():
            causes.append("Model trained on specific field strength - "
                         "different field strength changes SNR and contrast.")

        if self.preprocessing.get("normalization"):
            causes.append(f"Model used {self.preprocessing['normalization']} "
                         "preprocessing — target data may use different "
                         "normalization, causing distribution mismatch.")

        if self.input_channels == 1:
            causes.append("Model expects single-channel input — RGB or "
                         "multi-sequence input will need channel adaptation.")

        if not causes:
            causes.append("Unknown training details - will rely on "
                         "data-driven diagnosis (CCPED/LDDP/ACS).")

        return causes

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Model: {self.model_name} ({self.architecture})"]
        if self.training_dataset:
            lines.append(f"Trained on: {self.training_dataset} ({self.training_modality})")
        if self.training_classes:
            lines.append(f"Classes: {', '.join(self.training_classes[:10])}"
                        + (f" +{len(self.training_classes)-10} more"
                           if len(self.training_classes) > 10 else ""))
        if self.preprocessing:
            lines.append(f"Preprocessing: {self.preprocessing}")
        if self.reported_performance:
            perf = ", ".join(f"{k}={v:.3f}" for k, v in
                           list(self.reported_performance.items())[:3])
            lines.append(f"Reported: {perf}")
        lines.append(f"Confidence: {self.confidence}")
        return "\n".join(lines)


# ================================================================
# Known model databases (manually curated for reliable provenance)
# ================================================================

KNOWN_MODELS = {
    "swin_unetr_btcv_segmentation": ModelProvenance(
        model_name="SwinUNETR-BTCV",
        architecture="swin_unetr",
        source_url="https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV",
        paper_title="Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images",
        training_dataset="BTCV (Beyond the Cranial Vault)",
        training_modality="CT (abdominal, portal venous phase)",
        training_classes=["spleen", "right_kidney", "left_kidney", "gallbladder",
                         "esophagus", "liver", "stomach", "aorta", "IVC",
                         "portal_vein_splenic_vein", "pancreas",
                         "right_adrenal", "left_adrenal"],
        num_training_classes=14,
        training_samples=30,  # BTCV has 30 training volumes
        preprocessing={
            "normalization": "CT windowing, dataset-specific",
            "spacing": "resampled to 1.5x1.5x2.0mm",
            "input_size": "96x96x96 patches",
            "augmentation": "RandCropByPosNegLabel, RandFlip, RandRotate90, RandShiftIntensity",
        },
        input_channels=1,
        input_size="96x96x96",
        norm_type="LayerNorm",
        total_params=62_190_000,
        reported_performance={"mean_dice": 0.8225},
        confidence="high",
    ),

    "clip_driven_universal": ModelProvenance(
        model_name="CLIP-Driven-Universal-Model",
        architecture="swin_unetr",
        source_url="https://github.com/ljwztc/CLIP-Driven-Universal-Model",
        paper_title="CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection",
        paper_url="https://arxiv.org/abs/2301.00785",
        training_dataset="12 CT datasets (BTCV, AMOS-CT, LiTS, KiTS, Pancreas-CT, "
                        "CHAOS-CT, WORD, CT-ORG, AbdomenCT-1K, AbdomenCT-12organ, "
                        "3D-IRCADb, MSD tasks)",
        training_modality="CT (multi-site, multi-protocol abdominal)",
        training_classes=[
            "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus",
            "liver", "stomach", "aorta", "IVC", "pancreas", "right_adrenal",
            "left_adrenal", "duodenum", "bladder", "prostate_uterus",
            "hepatic_tumor", "pancreatic_tumor", "kidney_tumor", "liver_tumor",
            "colon_tumor", "lung_tumor", "kidney_cyst",
        ],
        num_training_classes=32,
        training_samples=3410,
        preprocessing={
            "normalization": "CT HU clipping [-175, 250], z-score per volume",
            "spacing": "resampled to median dataset spacing",
            "input_size": "96x96x96 patches",
        },
        input_channels=1,
        input_size="96x96x96",
        norm_type="LayerNorm",
        total_params=62_190_000,
        reported_performance={"mean_dice": 0.841, "msd_rank": 1},
        confidence="high",
    ),

    "totalsegmentator": ModelProvenance(
        model_name="TotalSegmentator",
        architecture="nnunet",
        source_url="https://github.com/wasserth/TotalSegmentator",
        paper_title="TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images",
        training_dataset="TotalSegmentator dataset (1204 CT volumes)",
        training_modality="CT (whole-body, various protocols and scanners)",
        training_classes=["spleen", "right_kidney", "left_kidney", "gallbladder",
                         "liver", "stomach", "pancreas", "right_adrenal",
                         "left_adrenal", "aorta", "IVC", "heart",
                         "right_lung", "left_lung"],  # abbreviated, 104 total
        num_training_classes=104,
        training_samples=1204,
        preprocessing={
            "normalization": "nnUNet default (per-dataset z-score)",
            "spacing": "nnUNet default (median spacing per dataset)",
            "input_size": "nnUNet patch-based",
        },
        input_channels=1,
        input_size="patch-based (nnUNet)",
        norm_type="InstanceNorm",
        total_params=58_000_000,
        reported_performance={"mean_dice": 0.943},
        confidence="high",
    ),

    "stunet": ModelProvenance(
        model_name="STU-Net",
        architecture="nnunet",
        source_url="https://github.com/Ziyan-Huang/STU-Net",
        paper_title="STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training",
        training_dataset="TotalSegmentator (1204 CT, 104 classes)",
        training_modality="CT (whole-body)",
        training_classes=["104 anatomical structures (same as TotalSegmentator)"],
        num_training_classes=104,
        training_samples=1204,
        preprocessing={
            "normalization": "nnUNet default",
            "spacing": "nnUNet default",
        },
        input_channels=1,
        norm_type="InstanceNorm",
        total_params=58_260_000,
        confidence="high",
    ),
}


# ================================================================
# Provenance extraction functions
# ================================================================

def read_provenance(source: str,
                    readme_path: str = None,
                    config_path: str = None,
                    llm_client=None) -> ModelProvenance:
    """Auto-extract model provenance from available sources.

    Tries multiple strategies in order:
    1. Check known model database
    2. Parse MONAI bundle metadata
    3. Read checkpoint metadata
    4. Parse README/documentation
    5. LLM-based extraction from unstructured text

    Args:
        source: Model source string (same format as load_external_model).
        readme_path: Optional path to README or documentation file.
        config_path: Optional path to training config file.
        llm_client: Optional anthropic.Anthropic client for LLM parsing.

    Returns:
        ModelProvenance with as much info as could be extracted.
    """
    prov = ModelProvenance()

    # Strategy 1: Known model lookup
    for key, known in KNOWN_MODELS.items():
        if key in source.lower() or (known.source_url and
                                      known.source_url in source):
            return known

    # Strategy 2: MONAI bundle metadata
    if source.startswith("monai://"):
        bundle_name = source.replace("monai://", "")
        prov = _read_monai_bundle_provenance(bundle_name)
        if prov.confidence != "low":
            return prov

    # Strategy 3: Checkpoint metadata
    if os.path.exists(source) and source.endswith(('.pt', '.pth', '.ckpt')):
        prov = _read_checkpoint_metadata(source)

    # Strategy 4: README / documentation
    if readme_path and os.path.exists(readme_path):
        doc_info = _parse_readme(readme_path)
        prov = _merge_provenance(prov, doc_info)

    # Strategy 5: Training config
    if config_path and os.path.exists(config_path):
        cfg_info = _parse_training_config(config_path)
        prov = _merge_provenance(prov, cfg_info)

    # Strategy 6: LLM-based extraction
    if llm_client and prov.raw_description:
        llm_info = _llm_extract_provenance(prov.raw_description, llm_client)
        prov = _merge_provenance(prov, llm_info)

    return prov


def _read_monai_bundle_provenance(bundle_name: str) -> ModelProvenance:
    """Read provenance from MONAI bundle metadata."""
    prov = ModelProvenance(model_name=bundle_name, confidence="medium")

    bundle_dir = os.path.expanduser(f"~/.monai/bundles/{bundle_name}")
    if not os.path.exists(bundle_dir):
        return prov

    # Read metadata.json
    meta_path = os.path.join(bundle_dir, "configs", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        prov.paper_title = meta.get("description", "")
        prov.architecture = meta.get("network_data_format", {}).get(
            "network", {}).get("name", "unknown")
        prov.raw_description = json.dumps(meta, indent=2)
        prov.confidence = "high"

    # Read training config for preprocessing details
    train_cfg = os.path.join(bundle_dir, "configs", "train.json")
    if os.path.exists(train_cfg):
        with open(train_cfg) as f:
            cfg = json.load(f)
        # Extract transforms
        transforms = cfg.get("train", {}).get("preprocessing", [])
        if transforms:
            prov.preprocessing["transforms"] = str(transforms)[:500]

    # Read docs
    for doc_name in ["README.md", "docs/README.md"]:
        doc_path = os.path.join(bundle_dir, doc_name)
        if os.path.exists(doc_path):
            with open(doc_path) as f:
                prov.raw_description += "\n\n" + f.read()[:5000]
            break

    return prov


def _read_checkpoint_metadata(ckpt_path: str) -> ModelProvenance:
    """Extract metadata stored in checkpoint dict."""
    import torch
    prov = ModelProvenance(confidence="low")

    try:
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return prov

    if not isinstance(state, dict):
        return prov

    # Common metadata keys
    if "config" in state:
        cfg = state["config"]
        if isinstance(cfg, dict):
            prov.training_dataset = cfg.get("dataset", cfg.get("data", ""))
            prov.input_size = str(cfg.get("input_size", cfg.get("img_size", "")))
            prov.preprocessing["from_config"] = str(cfg)[:500]
            prov.confidence = "medium"

    if "hyper_parameters" in state:
        hp = state["hyper_parameters"]
        if isinstance(hp, dict):
            prov.preprocessing["lr"] = str(hp.get("lr", ""))
            prov.training_samples = hp.get("num_samples", 0)

    if "epoch" in state:
        prov.preprocessing["trained_epochs"] = str(state["epoch"])

    # Infer architecture from key patterns
    keys = list(state.get("model_state_dict", state.get("state_dict", state)).keys())
    if any("swinViT" in k for k in keys[:100]):
        prov.architecture = "swin_unetr"
    elif any("seg_layers" in k for k in keys[:100]):
        prov.architecture = "nnunet"
    elif any("enc1" in k or "encoder1" in k for k in keys[:100]):
        prov.architecture = "unet"

    # Count params
    sd = state.get("model_state_dict", state.get("state_dict", state))
    if isinstance(sd, dict):
        prov.total_params = sum(v.numel() for v in sd.values()
                                if hasattr(v, 'numel'))

    return prov


def _parse_readme(readme_path: str) -> ModelProvenance:
    """Extract training info from README/documentation."""
    prov = ModelProvenance(confidence="low")

    with open(readme_path) as f:
        text = f.read()

    prov.raw_description = text[:5000]

    # Extract dataset mentions
    dataset_patterns = [
        r'(?:trained|training)\s+(?:on|with|using)\s+([A-Z][\w\-]+(?:\s+[\w\-]+){0,3})',
        r'dataset[:\s]+([A-Z][\w\-]+)',
        r'(BTCV|AMOS|TotalSegmentator|MMWHS|CrossMoDA|BraTS|LiTS|KiTS|ACDC|M&Ms)',
    ]
    for pattern in dataset_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            prov.training_dataset = matches[0] if isinstance(matches[0], str) else matches[0][0]
            prov.confidence = "medium"
            break

    # Extract modality
    modality_patterns = [
        r'(CT|MRI|T1|T2|FLAIR|fundus|retinal|X-ray|ultrasound)',
    ]
    for pattern in modality_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            prov.training_modality = matches[0]
            break

    # Extract architecture
    arch_patterns = [
        r'(UNet|U-Net|nnUNet|SwinUNETR|Swin UNETR|SegResNet|TransUNet|UNETR)',
    ]
    for pattern in arch_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            prov.architecture = matches[0].lower().replace(" ", "_").replace("-", "")
            break

    # Extract performance numbers
    dice_patterns = [
        r'[Dd]ice[:\s]+(\d+\.?\d*)\s*%?',
        r'(\d+\.?\d*)\s*%?\s*[Dd]ice',
        r'mean\s+[Dd]ice[:\s]+(\d+\.?\d*)',
    ]
    for pattern in dice_patterns:
        matches = re.findall(pattern, text)
        if matches:
            val = float(matches[0])
            if val > 1:
                val /= 100
            prov.reported_performance["mean_dice"] = val
            break

    return prov


def _parse_training_config(config_path: str) -> ModelProvenance:
    """Parse a training config file (JSON, YAML, or Python)."""
    prov = ModelProvenance(confidence="low")

    ext = os.path.splitext(config_path)[1]

    try:
        if ext in ('.json',):
            with open(config_path) as f:
                cfg = json.load(f)
        elif ext in ('.yaml', '.yml'):
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            with open(config_path) as f:
                prov.raw_description = f.read()[:3000]
            return prov

        # Extract common fields
        prov.training_dataset = str(cfg.get("dataset", cfg.get("data", {}).get("name", "")))
        prov.input_size = str(cfg.get("input_size", cfg.get("img_size", "")))

        if "num_classes" in cfg:
            prov.num_training_classes = cfg["num_classes"]
        if "in_channels" in cfg:
            prov.input_channels = cfg["in_channels"]

        prov.preprocessing["from_config"] = json.dumps(cfg, indent=2, default=str)[:1000]
        prov.confidence = "medium"

    except Exception:
        pass

    return prov


def _llm_extract_provenance(text: str, llm_client) -> ModelProvenance:
    """Use LLM to extract structured training info from unstructured text."""
    prov = ModelProvenance(confidence="low")

    prompt = f"""Extract the following information from this model documentation.
Return a JSON object with these fields (use null if not found):

- model_name: string
- architecture: string (unet, swin_unetr, nnunet, etc.)
- training_dataset: string
- training_modality: string (CT, MRI, etc.)
- training_classes: list of strings
- num_training_classes: int
- input_channels: int
- input_size: string
- normalization: string (how inputs are normalized)
- reported_dice: float (0-1 scale)

Documentation:
{text[:3000]}

Return ONLY the JSON object, no other text."""

    try:
        response = llm_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text.strip()

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())

            prov.model_name = data.get("model_name") or "unknown"
            prov.architecture = data.get("architecture") or "unknown"
            prov.training_dataset = data.get("training_dataset") or ""
            prov.training_modality = data.get("training_modality") or ""
            prov.training_classes = data.get("training_classes") or []
            prov.num_training_classes = data.get("num_training_classes") or 0
            prov.input_channels = data.get("input_channels") or 1
            prov.input_size = data.get("input_size") or ""
            if data.get("normalization"):
                prov.preprocessing["normalization"] = data["normalization"]
            if data.get("reported_dice"):
                prov.reported_performance["mean_dice"] = float(data["reported_dice"])
            prov.confidence = "medium"

    except Exception as e:
        print(f"LLM provenance extraction failed: {e}")

    return prov


def _merge_provenance(base: ModelProvenance,
                      update: ModelProvenance) -> ModelProvenance:
    """Merge two provenance objects, preferring non-empty fields from update."""
    for field_name in [
        "model_name", "architecture", "source_url", "paper_title", "paper_url",
        "training_dataset", "training_modality", "input_size", "norm_type",
    ]:
        base_val = getattr(base, field_name)
        update_val = getattr(update, field_name)
        if update_val and (not base_val or base_val == "unknown"):
            setattr(base, field_name, update_val)

    if update.training_classes and not base.training_classes:
        base.training_classes = update.training_classes
    if update.num_training_classes and not base.num_training_classes:
        base.num_training_classes = update.num_training_classes
    if update.total_params and not base.total_params:
        base.total_params = update.total_params
    if update.training_samples and not base.training_samples:
        base.training_samples = update.training_samples

    base.preprocessing.update(update.preprocessing)
    base.reported_performance.update(update.reported_performance)

    if update.raw_description:
        base.raw_description += "\n" + update.raw_description

    # Upgrade confidence
    conf_order = {"low": 0, "medium": 1, "high": 2}
    if conf_order.get(update.confidence, 0) > conf_order.get(base.confidence, 0):
        base.confidence = update.confidence

    return base
