#!/usr/bin/env python
"""
Run ONLY Agent 4 (Execution Engineer) on all 8 datasets.

For each dataset:
  1. Load Agent 1 JSON (used as `knowledge` arg)
  2. Load Agent 3 plan JSON
  3. Load source model + source/target data + labels
  4. Run run_execution_engineer(...)
  5. Save iteration_history + quality_scores + final metrics to
     experiments/agent4_analyses/{ds}.json

Unlike Agents 1-3, this one actually runs adaptation (gradient steps +
BN reset + FDA + evaluation). Expect a few minutes per dataset.

Usage:
    export AZURE_OPENAI_API_KEY=...  # not strictly needed for Agent 4
    python scripts/run_agent4_only.py --device cuda:0
    python scripts/run_agent4_only.py --device cuda:0 --dataset prostate
"""

import os, sys, json, time, numpy as np, torch
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from medadaptagent.graph.agents.execution_engineer import run_execution_engineer
from medadaptagent.knowledge.llm_client import create_llm_client
from medadaptagent.models.unet import UNet
from medadaptagent.pipeline_config import TaskConfig
from scripts.run_ablations import load_npz_dataset
from scripts._source_loader import load_source_model


DATASETS = {
    "riga_plus": {
        "src": "data/riga_plus/source", "tgt": "data/riga_plus/target_MESSIDOR_Base2",
        "ckpt": "checkpoints/unet_riga_binrushed_best.pt", "nc": 3,
        "cfg": {"task_name": "riga_plus_cross_device",
                "source_domain": "RIGA BinRushed", "target_domain": "MESSIDOR Base2",
                "structures": {"background": 0, "disc": 1, "cup": 2}},
    },
    "prostate": {
        "src": "data/prostate/source", "tgt": "data/prostate/target_BIDMC",
        "ckpt": "checkpoints/unet_prostate_runmc_best.pt", "nc": 2,
        "cfg": {"task_name": "prostate_cross_site",
                "source_domain": "RUNMC", "target_domain": "BIDMC",
                "structures": {"background": 0, "prostate": 1}},
    },
    "chaos": {
        "src": "data/chaos/source", "tgt": "data/chaos/target",
        "ckpt": "checkpoints/unet_chaos_ct_best.pt", "nc": 2,
        "cfg": {"task_name": "chaos_cross_modality",
                "source_domain": "CT abdominal", "target_domain": "MRI T2-SPIR",
                "structures": {"background": 0, "liver": 1},
                # Phase S1 provenance hints — agent searches these first
                "provenance_search": ("CHAOS challenge combined CT MRI "
                                        "abdominal organ segmentation"),
                "paper_url": "https://arxiv.org/abs/2001.06535"},
    },
    "fundus_roi": {
        "src": "data/fundus_roi/source", "tgt": "data/fundus_roi/target",
        "ckpt": "checkpoints/fundus_roi/unet_refuge_best.pt", "nc": 3,
        "cfg": {"task_name": "fundus_cross_device",
                "source_domain": "REFUGE", "target_domain": "RIM-ONE DL",
                "structures": {"background": 0, "disc": 1, "cup": 2}},
    },
    "mnms": {
        "src": "data/mnms/vendor_Siemens", "tgt": "data/mnms/vendor_Philips",
        "ckpt": "checkpoints/unet_mnms_siemens_best.pt", "nc": 4,
        "cfg": {"task_name": "mnms_cross_vendor",
                "source_domain": "Siemens bSSFP", "target_domain": "Philips bSSFP",
                "structures": {"background": 0, "LV": 1, "MYO": 2, "RV": 3}},
    },
    "brats": {
        "src": "data/brats/source", "tgt": "data/brats/target",
        "ckpt": "checkpoints/unet_brats_t1_best.pt", "nc": 2,
        "max_target": 200,
        "cfg": {"task_name": "brats_t1_to_t2",
                "source_domain": "T1-weighted brain MRI", "target_domain": "T2-weighted brain MRI",
                "structures": {"background": 0, "tumor": 1}},
    },
    "mnms_field": {
        "src": "data/mnms_field/field_1.5T", "tgt": "data/mnms_field/field_3T",
        "ckpt": "checkpoints/unet_mnms_15T_best.pt", "nc": 4,
        "cfg": {"task_name": "mnms_cross_field",
                "source_domain": "1.5T bSSFP", "target_domain": "3T bSSFP",
                "structures": {"background": 0, "LV": 1, "MYO": 2, "RV": 3}},
    },
    "mmwhs": {
        "src": "data/mmwhs/source", "tgt": "data/mmwhs/target",
        "ckpt": "checkpoints/unet_mmwhs_ct_best.pt", "nc": 5,
        "cfg": {"task_name": "mmwhs_cross_modality",
                "source_domain": "CT cardiac", "target_domain": "MRI cardiac bSSFP",
                "structures": {"background": 0, "MYO": 1, "LA": 2, "LV": 3, "RV": 4}},
    },
}


# STU-Net (InstanceNorm 2D) source models — same target domains, different
# architecture. Every entry carries `arch="stunet"` so the shared loader
# dispatches to STUNetLite. mnms / mnms_field have no source-side training
# data so no STU-Net equivalent is provided.
STUNET_DATASETS = {
    "riga_plus_stunet": {
        **DATASETS["riga_plus"],
        "ckpt": "checkpoints/stunet_riga_binrushed_best.pt", "arch": "stunet",
    },
    "prostate_stunet": {
        **DATASETS["prostate"],
        "ckpt": "checkpoints/stunet_prostate_runmc_best.pt", "arch": "stunet",
    },
    "chaos_stunet": {
        **DATASETS["chaos"],
        "ckpt": "checkpoints/stunet_chaos_ct_best.pt", "arch": "stunet",
    },
    "fundus_roi_stunet": {
        **DATASETS["fundus_roi"],
        "ckpt": "checkpoints/stunet_refuge_best.pt", "arch": "stunet",
    },
    "brats_stunet": {
        **DATASETS["brats"],
        "ckpt": "checkpoints/stunet_brats_t1_best.pt", "arch": "stunet",
    },
    "mmwhs_stunet": {
        **DATASETS["mmwhs"],
        "ckpt": "checkpoints/stunet_mmwhs_ct_best.pt", "arch": "stunet",
    },
    "mnms_stunet": {
        **DATASETS["mnms"],
        "ckpt": "checkpoints/stunet_mnms_siemens_best.pt", "arch": "stunet",
    },
    "mnms_field_stunet": {
        **DATASETS["mnms_field"],
        "ckpt": "checkpoints/stunet_mnms_15T_best.pt", "arch": "stunet",
    },
}


# Vanilla nnU-Net 2D (InstanceNorm + LeakyReLU, UNet topology, no residuals).
# Same 8 target domains as DATASETS / STUNET_DATASETS; only ckpt + arch change.
# 3D segmentation datasets — MONAI Bundle path. Volumes loaded as
# nibabel-backed `Volume` objects (numpy + affine), preprocessing
# chain resolved via the Tier-1 sidecar at <ckpt_dir>/model.meta.json.
THREED_DATASETS = {
    "msd_spleen": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "tgt_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "ckpt": "data/monai_bundles/spleen_ct_segmentation/models/model.pt",
        "arch": "monai_unet3d", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "msd_spleen_baseline",
                "source_domain": "MSD Task09_Spleen",
                "target_domain": "MSD Task09_Spleen",
                "structures": {"background": 0, "spleen": 1}},
    },
    # Contrast-vs-non-contrast shift target. Pre-processed by
    # scripts/preprocess_totalseg_spleen.py (TotalSegmentator v2 abdominal
    # cases split by aorta HU; spleen mask binarized; MSD-style layout).
    "totalseg_spleen_nc": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/totalseg_spleen_nc/imagesTs",
        "tgt_labels": "data/3d/totalseg_spleen_nc/labelsTs",
        "ckpt": "data/monai_bundles/spleen_ct_segmentation/models/model.pt",
        "arch": "monai_unet3d", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "totalseg_spleen_nc_contrast_shift",
                "source_domain": "MSD Task09_Spleen (PVP CT)",
                "target_domain": "TotalSegmentator non-contrast abdominal CT",
                "structures": {"background": 0, "spleen": 1}},
    },
    "totalseg_spleen_ce": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/totalseg_spleen_ce/imagesTs",
        "tgt_labels": "data/3d/totalseg_spleen_ce/labelsTs",
        "ckpt": "data/monai_bundles/spleen_ct_segmentation/models/model.pt",
        "arch": "monai_unet3d", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "totalseg_spleen_ce_cohort_shift",
                "source_domain": "MSD Task09_Spleen",
                "target_domain": "TotalSegmentator contrast-enhanced abdominal CT",
                "structures": {"background": 0, "spleen": 1}},
    },
    # Cross-cohort domain shift target. Pre-processed by
    # scripts/preprocess_synapse_spleen.py (binarized to spleen-only,
    # filenames normalized). Sandbox sees MSD-style layout.
    "synapse_spleen": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/synapse_spleen/imagesTs",
        "tgt_labels": "data/3d/synapse_spleen/labelsTs",
        "ckpt": "data/monai_bundles/spleen_ct_segmentation/models/model.pt",
        "arch": "monai_unet3d", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "synapse_spleen_cross_cohort",
                "source_domain": "MSD Task09_Spleen",
                "target_domain": "BTCV/Synapse abdominal CT",
                "structures": {"background": 0, "spleen": 1}},
    },
    # Bundle-agnostic loader smoke target: MONAI wholeBody_ct_segmentation
    # (SegResNet, 105-class TotalSeg-style) adapted to MSD-spleen. Class
    # index 1 (spleen) extracted via sidecar's class_extract field.
    # Demonstrates the chain pattern generalizes across architectures
    # (UNet → SegResNet) without sandbox code changes.
    "wholebody_msd_spleen": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "tgt_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "ckpt": "data/monai_bundles/wholeBody_ct_segmentation/models/model.pt",
        "arch": "monai_bundle", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "wholebody_msd_spleen_bundle_agnostic",
                "source_domain": "MONAI wholeBody (TotalSegmentator v2)",
                "target_domain": "MSD Task09_Spleen",
                "structures": {"background": 0, "spleen": 1}},
    },
    # Pancreas (DiNTS NAS arch). Source = MSD Task07 pancreas (bundle
    # default); target = MSD spleen smoke. Pancreas model on spleen
    # scans is a wrong-organ stress test — Agent 1/2 should flag the
    # mismatch via near-zero foreground predictions.
    "pancreas_dints_msd_spleen": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "tgt_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "ckpt": "data/monai_bundles/pancreas_ct_dints_segmentation/models/model.pt",
        "arch": "monai_bundle", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "pancreas_dints_msd_spleen_organ_shift",
                "source_domain": "MONAI pancreas_ct_dints (DiNTS NAS)",
                "target_domain": "MSD Task09_Spleen (organ mismatch)",
                "structures": {"background": 0, "pancreas": 1}},
    },
    # Multi-organ SwinUNETR on BTCV spleen channel. Source = BTCV;
    # target = Synapse (multi-organ, spleen extracted). Real
    # cross-arch test: SwinUNETR (transformer) vs the UNet baselines
    # on the same spleen task as synapse_spleen above.
    "btcv_swin_synapse_spleen": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/synapse_spleen/imagesTs",
        "tgt_labels": "data/3d/synapse_spleen/labelsTs",
        "ckpt": "data/monai_bundles/swin_unetr_btcv_segmentation/models/model.pt",
        "arch": "monai_bundle", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "btcv_swin_synapse_spleen_cross_arch",
                "source_domain": "MONAI swin_unetr_btcv (SwinUNETR transformer)",
                "target_domain": "Synapse BTCV abdominal CT (spleen extracted)",
                "structures": {"background": 0, "spleen": 1}},
    },
    # SwinUNETR (BTCV 14-class transformer) on TotalSegmentator NON-contrast
    # abdominal CT -> contrast->non-contrast shift. Spleen = BTCV channel 1,
    # exposed via the bundle sidecar class_extract=[1] (nc=2: bg+spleen).
    # SAME target as totalseg_spleen_nc (UNet baseline) -> cross-arch shift test
    # for a transformer SEGMENTER. Window/spacing read from the swin bundle's
    # inference.json (ScaleIntensityRange -175/250, pixdim 1.5/1.5/2.0).
    "swinunetr_btcv_spleen_nc": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/totalseg_spleen_nc/imagesTs",
        "tgt_labels": "data/3d/totalseg_spleen_nc/labelsTs",
        "ckpt": "data/monai_bundles/swin_unetr_btcv_segmentation/models/model.pt",
        "arch": "monai_bundle", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "swinunetr_btcv_spleen_nc_contrast_shift",
                "source_domain": "MONAI swin_unetr_btcv (SwinUNETR transformer, BTCV CE CT)",
                "target_domain": "TotalSegmentator NON-contrast abdominal CT (spleen)",
                "structures": {"background": 0, "spleen": 1}},
    },
    # SwinUNETR contrast-enhanced GATE: same transformer on TotalSegmentator
    # contrast-enhanced abdominal CT (in-distribution-ish) -> the correctness
    # gate + the no-shift reference for the nc drop.
    "swinunetr_btcv_spleen_ce": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/totalseg_spleen_ce/imagesTs",
        "tgt_labels": "data/3d/totalseg_spleen_ce/labelsTs",
        "ckpt": "data/monai_bundles/swin_unetr_btcv_segmentation/models/model.pt",
        "arch": "monai_bundle", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "swinunetr_btcv_spleen_ce_gate",
                "source_domain": "MONAI swin_unetr_btcv (SwinUNETR transformer, BTCV CE CT)",
                "target_domain": "TotalSegmentator contrast-enhanced abdominal CT (spleen)",
                "structures": {"background": 0, "spleen": 1}},
    },
    # UNETR (BTCV-style PURE-ViT-encoder transformer SEGMENTER) on TotalSegmentator
    # NON-contrast abdominal CT -> contrast->non-contrast shift. SAME target as
    # swinunetr_btcv_spleen_nc + totalseg_spleen_nc -> a SECOND 3D transformer
    # segmenter (UNETR global ViT attention vs SwinUNETR WINDOWED attention vs
    # UNet conv). UNETR norms = ViT-encoder LayerNorm + InstanceNorm decoder, NO
    # BatchNorm -> bn_reset/tent are structural NO-OPS (toolbox() auto-detects 0
    # BN). Trained from scratch on MSD spleen (out_channels=2 directly -> no
    # class_extract). Window -57/164 read from the unetr bundle's inference.json.
    "unetr_btcv_spleen_nc": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/totalseg_spleen_nc/imagesTs",
        "tgt_labels": "data/3d/totalseg_spleen_nc/labelsTs",
        "ckpt": "data/monai_bundles/unetr_btcv_segmentation/models/model.pt",
        "arch": "monai_bundle", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "unetr_btcv_spleen_nc_contrast_shift",
                "source_domain": "MONAI UNETR (pure-ViT-encoder transformer, MSD spleen CT)",
                "target_domain": "TotalSegmentator NON-contrast abdominal CT (spleen)",
                "structures": {"background": 0, "spleen": 1}},
    },
    # UNETR contrast-enhanced GATE: same UNETR transformer on TotalSegmentator
    # contrast-enhanced abdominal CT (in-distribution-ish) -> correctness gate +
    # the no-shift reference for the nc drop. Mirrors swinunetr_btcv_spleen_ce.
    "unetr_btcv_spleen_ce": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/totalseg_spleen_ce/imagesTs",
        "tgt_labels": "data/3d/totalseg_spleen_ce/labelsTs",
        "ckpt": "data/monai_bundles/unetr_btcv_segmentation/models/model.pt",
        "arch": "monai_bundle", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "unetr_btcv_spleen_ce_gate",
                "source_domain": "MONAI UNETR (pure-ViT-encoder transformer, MSD spleen CT)",
                "target_domain": "TotalSegmentator contrast-enhanced abdominal CT (spleen)",
                "structures": {"background": 0, "spleen": 1}},
    },
    # Multi-organ 8-class UNet on MSD spleen target. Spleen channel
    # (4) extracted via sidecar — same target as msd_spleen baseline,
    # different source bundle.
    "multi_organ_msd_spleen": {
        "src": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "src_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "tgt": "data/3d/msd_spleen/Task09_Spleen/imagesTr",
        "tgt_labels": "data/3d/msd_spleen/Task09_Spleen/labelsTr",
        "ckpt": "data/monai_bundles/multi_organ_segmentation/models/model.pt",
        "arch": "monai_bundle", "nc": 2, "task": "segmentation_3d",
        "max_target": 6,
        "cfg": {"task_name": "multi_organ_msd_spleen_class_extract",
                "source_domain": "MONAI multi_organ_segmentation (8-class)",
                "target_domain": "MSD Task09_Spleen",
                "structures": {"background": 0, "spleen": 1}},
    },
}


NNUNET_DATASETS = {
    "riga_plus_nnunet":  {**DATASETS["riga_plus"],  "ckpt": "checkpoints/nnunet_riga_binrushed_best.pt", "arch": "nnunet"},
    "prostate_nnunet":   {**DATASETS["prostate"],   "ckpt": "checkpoints/nnunet_prostate_runmc_best.pt", "arch": "nnunet"},
    "chaos_nnunet":      {**DATASETS["chaos"],      "ckpt": "checkpoints/nnunet_chaos_ct_best.pt",        "arch": "nnunet"},
    "fundus_roi_nnunet": {**DATASETS["fundus_roi"], "ckpt": "checkpoints/nnunet_refuge_best.pt",           "arch": "nnunet"},
    "brats_nnunet":      {**DATASETS["brats"],      "ckpt": "checkpoints/nnunet_brats_t1_best.pt",         "arch": "nnunet"},
    "mnms_field_nnunet": {**DATASETS["mnms_field"], "ckpt": "checkpoints/nnunet_mnms_15T_best.pt",         "arch": "nnunet"},
    "mnms_nnunet":       {**DATASETS["mnms"],       "ckpt": "checkpoints/nnunet_mnms_siemens_best.pt",     "arch": "nnunet"},
    "mmwhs_nnunet":      {**DATASETS["mmwhs"],      "ckpt": "checkpoints/nnunet_mmwhs_ct_best.pt",          "arch": "nnunet"},
}


# SegFormer (MiT transformer encoder, LayerNorm; all-MLP decode head w/ a
# single BatchNorm2d) — the FIRST 2D transformer SEGMENTER. arch="segformer"
# routes through scripts/_source_loader -> SegFormer2D, whose forward upsamples
# HF's H/4 logits to full res so the UNet seg contract holds. Same prostate
# RUNMC->{BIDMC,HK} domains as the UNet/STU-Net/nnU-Net entries -> a clean
# cross-arch (CNN-BN/IN vs transformer-LN) comparison on an identical shift.
SEGFORMER_DATASETS = {
    "prostate_segformer": {
        **DATASETS["prostate"],
        "ckpt": "checkpoints/segformer_prostate_runmc_best.pt", "arch": "segformer",
    },
}


def _base_ds_name(ds_name):
    """Strip arch suffix so STU-Net entries reuse the base dataset's
    Agent 1 analysis (domain-shift analysis is arch-independent)."""
    for suffix in ("_stunet", "_nnunet", "_segformer"):
        if ds_name.endswith(suffix):
            return ds_name[: -len(suffix)]
    return ds_name


def _load_json(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    with open(path) as f:
        return json.load(f)


def _load_json_with_base_fallback(dir_path, ds_name, label):
    """Try ds_name.json first; fall back to _base_ds_name(ds_name).json."""
    p_exact = os.path.join(dir_path, f"{ds_name}.json")
    if os.path.exists(p_exact):
        return _load_json(p_exact, label)
    base = _base_ds_name(ds_name)
    p_base = os.path.join(dir_path, f"{base}.json")
    if os.path.exists(p_base):
        return _load_json(p_base, label)
    raise FileNotFoundError(
        f"Missing {label}: {p_exact} (also checked {p_base})")


def _load_source_model(ckpt_rel, nc, device, arch="unet"):
    """Thin wrapper around `scripts._source_loader.load_source_model` that
    also passes PROJECT_ROOT so relative checkpoint paths work. Backward
    compat: `arch` defaults to "unet" so existing call sites are unchanged.
    """
    return load_source_model(ckpt_rel, nc, device,
                              arch=arch, project_root=PROJECT_ROOT)


def _seed(seed: int = 42):
    """Seed all RNGs. Default 42 preserves prior single-seed behavior;
    pass `seed=N` from the CLI to drive multi-seed sweeps (Phase E1)."""
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _serialize_results(exec_result):
    """Strip non-serializable fields (adapted_model, pipeline_state) and
    keep only metrics / history / execution_report for the JSON."""
    out = {
        "skipped": exec_result.get("skipped", False),
        "did_revert": exec_result.get("did_revert", False),
        "iteration_history": [],
        "quality_scores": exec_result.get("quality_scores", []),
        "execution_report": exec_result.get("execution_report", {}),
    }
    for h in exec_result.get("iteration_history", []):
        clean = {}
        for k, v in h.items():
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                clean[k] = v
        out["iteration_history"].append(clean)
    # Pure-TTA: no GT dice on the adaptation set. Record the final VLM score
    # and label-free quality_score (for trajectory inspection). The post-hoc
    # sanity audit script computes GT Dice separately.
    if out["iteration_history"]:
        final = out["iteration_history"][-1]
        out["final_vlm_score"] = final.get("vlm_score")
        out["final_quality_score"] = final.get("quality_score")
        out["final_method"] = final.get("method")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--llm-model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--arch", default="unet",
                        choices=["unet", "stunet", "nnunet", "segformer", "all"],
                        help="Which source-model family to run: 'unet' (BN UNet), "
                             "'stunet' (IN + residual), 'nnunet' (vanilla IN UNet), "
                             "'segformer' (MiT transformer, LayerNorm), or 'all'.")
    args = parser.parse_args()

    llm = create_llm_client(backend="azure", model=args.llm_model)

    # Assemble the active DATASETS view based on --arch, then pick which
    # entries to run. `--dataset all` means "every entry in the active view".
    if args.arch == "unet":
        active = dict(DATASETS)
    elif args.arch == "stunet":
        active = dict(STUNET_DATASETS)
    elif args.arch == "nnunet":
        active = dict(NNUNET_DATASETS)
    elif args.arch == "segformer":
        active = dict(SEGFORMER_DATASETS)
    else:  # "all"
        active = {**DATASETS, **STUNET_DATASETS, **NNUNET_DATASETS, **SEGFORMER_DATASETS}
    DATASETS.update(STUNET_DATASETS)
    DATASETS.update(NNUNET_DATASETS)
    DATASETS.update(SEGFORMER_DATASETS)

    datasets = list(active.keys()) if args.dataset == "all" else [args.dataset]
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "agent4_analyses")
    os.makedirs(out_dir, exist_ok=True)

    a1_dir = os.path.join(PROJECT_ROOT, "experiments", "agent1_analyses")
    a3_dir = os.path.join(PROJECT_ROOT, "experiments", "agent3_analyses")

    all_results = {}
    for ds_name in datasets:
        ds = DATASETS[ds_name]
        print(f"\n{'#' * 70}")
        print(f"### Agent 4 inspection: {ds_name}")
        print(f"{'#' * 70}")

        try:
            _seed()

            # 1. Load Agent 1 + Agent 3 outputs
            domain_analysis = _load_json_with_base_fallback(
                a1_dir, ds_name, "Agent 1")
            plan = _load_json(
                os.path.join(a3_dir, f"{ds_name}.json"), "Agent 3")
            print(f"  Plan: protocol={plan.get('protocol')} "
                  f"scope={plan.get('trainable_scope')} "
                  f"steps={plan.get('num_steps')} iter={plan.get('max_iterations')}")

            # 2. Load source model
            model = _load_source_model(ds["ckpt"], ds["nc"], args.device,
                                         arch=ds.get("arch", "unet"))

            # 3. Build TaskConfig
            cfg = ds.get("cfg", {}).copy()
            cfg["image_size"] = 256
            cfg["max_iterations"] = plan.get("max_iterations", 1)
            tc = TaskConfig(**cfg)

            # 4. Load source + target data. Target labels are DROPPED here
            # to enforce pure-TTA behavior in Agent 4 (they are only used in
            # the separate post-hoc sanity audit).
            mt = ds.get("max_target", 50)
            src, src_lbl = load_npz_dataset(os.path.join(PROJECT_ROOT, ds["src"]), 100)
            tgt, _ = load_npz_dataset(os.path.join(PROJECT_ROOT, ds["tgt"]), mt)
            print(f"  Data: {len(src)} source ({len(src_lbl or [])} lbls) + "
                  f"{len(tgt)} target (labels withheld — pure TTA)")

            # Load Agent 2 for the VLM task context (failure_type, severity)
            a2 = _load_json(os.path.join(PROJECT_ROOT, "experiments",
                                           "agent2_analyses", f"{ds_name}.json"),
                             "Agent 2")

            # 5. Run Agent 4
            t0 = time.time()
            exec_result = run_execution_engineer(
                plan=plan,
                source_model=model,
                source_images=src,
                target_images=tgt,
                source_labels=src_lbl or [],
                task_config=tc,
                knowledge=domain_analysis,
                device=args.device,
                llm_client=llm,
                model_assessment=a2,
            )
            elapsed = time.time() - t0

            # 6. Serialize + save
            result = _serialize_results(exec_result)
            result["elapsed_seconds"] = round(elapsed, 1)
            result["plan_used"] = {
                "protocol": plan.get("protocol"),
                "trainable_scope": plan.get("trainable_scope"),
                "num_steps": plan.get("num_steps"),
                "max_iterations": plan.get("max_iterations"),
                "learning_rate": plan.get("learning_rate"),
                "loss_weights": plan.get("loss_weights"),
            }

            all_results[ds_name] = result
            with open(os.path.join(out_dir, f"{ds_name}.json"), "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Save adapted model state_dict for Agent 5's consumption.
            # Tent / SAR paths disable BN track_running_stats, which drops
            # running_mean/running_var from state_dict. Restore them from the
            # source model before saving so the state_dict is a complete
            # checkpoint loadable by a fresh model.
            adapted_model = exec_result.get("adapted_model")
            if adapted_model is not None:
                import torch.nn as _nn
                for (_, sm), (_, am) in zip(model.named_modules(),
                                              adapted_model.named_modules()):
                    if isinstance(am, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)):
                        if am.running_mean is None and sm.running_mean is not None:
                            am.running_mean = sm.running_mean.clone()
                        if am.running_var is None and sm.running_var is not None:
                            am.running_var = sm.running_var.clone()
                        if not am.track_running_stats:
                            am.track_running_stats = True
                            if sm.running_mean is not None:
                                am.register_buffer("running_mean", sm.running_mean.clone())
                                am.register_buffer("running_var", sm.running_var.clone())
                                am.register_buffer("num_batches_tracked",
                                                    torch.tensor(0, dtype=torch.long))
                ckpt_path = os.path.join(out_dir, f"{ds_name}_adapted.pt")
                torch.save(adapted_model.state_dict(), ckpt_path)
                print(f"    saved adapted model: {ckpt_path}")

            # Console summary
            print(f"\n>>> {ds_name} ({elapsed:.0f}s)")
            fvs = result.get('final_vlm_score')
            fvs_s = f'{fvs:+.2f}' if isinstance(fvs,(int,float)) else fvs
            print(f"    final_vlm_score: {fvs_s}")
            print(f"    did_revert     : {result.get('did_revert')}")
            print(f"    skipped        : {result['skipped']}")
            er = result.get('execution_report') or {}
            if er:
                print(f"    status         : {er.get('status','?')}")
                print(f"    stopped_reason : {er.get('stopped_reason','?')}")
                print(f"    best_iter_idx  : {er.get('best_iteration_index','?')}")
                if er.get('collapse_events'):
                    print(f"    collapses      : {er['collapse_events']}")
            for i, h in enumerate(result['iteration_history']):
                vs = h.get('vlm_score', '?')
                qs = h.get('quality_score', '?')
                vs_s = f'{vs:+.2f}' if isinstance(vs,(int,float)) else vs
                qs_s = f'{qs:.3f}' if isinstance(qs,(int,float)) else qs
                votes = h.get('vlm_votes')
                v_str = ('/'.join(votes) if votes else '—')
                print(f"    iter {i}: vlm_score={vs_s} ({v_str}) "
                      f"quality_score={qs_s} method={h.get('method','?')}")

            # Free GPU memory
            del model, exec_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n>>> {ds_name}: FAILED -- {e}")
            import traceback; traceback.print_exc()
            all_results[ds_name] = {"error": str(e)}

    summary_file = os.path.join(out_dir, "all_results.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll Agent 4 results saved to {out_dir}/")


if __name__ == "__main__":
    main()
