#!/usr/bin/env python
"""
Run ablation studies for MedAdaptAgent using the Orchestrator pipeline.

A1: w/o RAG knowledge base
A2: w/o failure diagnosis (blind adapter activation)
A3: w/o iterative refinement (single-pass)
A4: w/o LLM orchestration (rule-based)
A5: w/o self-evaluation Q(x)
A6: w/o post-processing (anatomy refinement)
A7: BN reset only (no self-supervised losses)

Usage:
    python scripts/run_ablations.py --dataset fundus_roi --device cuda:0 --ablation all
    python scripts/run_ablations.py --dataset fundus_roi --device cuda:0 --ablation A1
"""

import os
import sys
import json
import copy
import argparse
import logging
import time
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medadaptagent.orchestrator import Orchestrator, PipelineState
from medadaptagent.pipeline_config import (
    TaskConfig, FUNDUS_CONFIG, CARDIAC_CONFIG, CROSSMODA_CONFIG,
    CHAOS_CONFIG, BRATS_CONFIG, PROSTATE_CONFIG, MNMS_CONFIG,
)
from medadaptagent.models.unet import UNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Ablations")


# ================================================================
# Dataset configs (matching run_baselines.py)
# ================================================================

DATASET_CONFIGS = {
    "fundus_roi": {
        "source_dir": "data/fundus_roi/source",
        "target_dir": "data/fundus_roi/target",
        "checkpoint": "checkpoints/fundus_roi/unet_refuge_best.pt",
        "num_classes": 3,
        "image_size": 256,
        "task_config": FUNDUS_CONFIG,
    },
    "fundus": {
        "source_dir": "data/fundus/source",
        "target_dir": "data/fundus/target",
        "checkpoint": "checkpoints/unet_refuge_best.pt",
        "num_classes": 3,
        "image_size": 256,
        "task_config": FUNDUS_CONFIG,
    },
    "chaos": {
        "source_dir": "data/chaos/source",
        "target_dir": "data/chaos/target",
        "checkpoint": "checkpoints/unet_chaos_ct_best.pt",
        "num_classes": 2,
        "image_size": 256,
        "task_config": CHAOS_CONFIG,
    },
    "brats": {
        "source_dir": "data/brats/source",
        "target_dir": "data/brats/target",
        "checkpoint": "checkpoints/unet_brats_t1_best.pt",
        "num_classes": 2,
        "image_size": 256,
        "task_config": BRATS_CONFIG,
    },
    "prostate": {
        "source_dir": "data/prostate/source",
        "target_dir": "data/prostate/target_BIDMC",
        "checkpoint": "checkpoints/unet_prostate_runmc_best.pt",
        "num_classes": 2,
        "image_size": 256,
        # Binary prostate segmentation (model is 2-class, not 3)
        "task_config": TaskConfig(
            task_name="prostate_cross_site",
            source_domain="Site A (Siemens 3T, RUNMC)",
            target_domain="Site B (BIDMC)",
            structures={"background": 0, "prostate": 1},
            class_descriptions={
                "prostate": "Prostate gland: roughly elliptical in axial view",
            },
            anatomy_priors={
                "prostate": {
                    "shape": "roughly elliptical",
                    "area_range": (0.02, 0.20),
                },
            },
            containment_rules=[],
        ),
    },
    "mnms": {
        "source_dir": "data/mnms/source",
        "target_dir": "data/mnms/target",
        "checkpoint": "checkpoints/mnms/unet_siemens_best.pt",
        "num_classes": 4,
        "image_size": 256,
        "task_config": MNMS_CONFIG,
    },
    "mmwhs": {
        "source_dir": "data/mmwhs/source",
        "target_dir": "data/mmwhs/target",
        "checkpoint": "checkpoints/unet_mmwhs_ct_best.pt",
        "num_classes": 5,
        "image_size": 256,
        "task_config": CARDIAC_CONFIG,
    },
    "crossmoda": {
        "source_dir": "data/crossmoda/source",
        "target_dir": "data/crossmoda/target",
        "checkpoint": "checkpoints/crossmoda/unet_cet1_best.pt",
        "num_classes": 3,
        "image_size": 256,
        "task_config": CROSSMODA_CONFIG,
    },
}


ABLATIONS = {
    "A0": {
        "name": "Full pipeline (control)",
        "description": "Full MedAdaptAgent pipeline — baseline for ablation comparison",
    },
    "A1": {
        "name": "w/o RAG knowledge base",
        "description": "Remove RAG knowledge base, use generic adaptation",
    },
    "A2": {
        "name": "w/o failure diagnosis",
        "description": "Skip diagnosis agent, activate all adapters blindly",
    },
    "A3": {
        "name": "w/o iterative refinement",
        "description": "Single-pass adaptation (1 iteration)",
    },
    "A4": {
        "name": "w/o LLM orchestration",
        "description": "Rule-based strategy selection (no LLM client)",
    },
    "A5": {
        "name": "w/o self-evaluation Q(x)",
        "description": "No convergence checking via quality score",
    },
    "A6": {
        "name": "w/o post-processing",
        "description": "No anatomy-aware post-processing",
    },
    "A7": {
        "name": "BN reset only",
        "description": "BN adaptation only, no self-supervised losses",
    },
}


def load_npz_dataset(data_dir, max_samples=None):
    """Load preprocessed .npz images and labels.

    Handles both (H, W, C) and (C, H, W) image formats,
    returning consistently as (H, W, C) for the pipeline.
    """
    img_dir = os.path.join(data_dir, "images")
    lbl_dir = os.path.join(data_dir, "labels")

    if os.path.isdir(img_dir):
        files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])
    else:
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        img_dir = data_dir
        lbl_dir = data_dir

    if max_samples:
        files = files[:max_samples]

    images, labels = [], []
    for f in files:
        if img_dir == data_dir:
            data = np.load(os.path.join(data_dir, f))
            img = data["image"].astype(np.float32)
            lbl = data.get("label")
        else:
            img = np.load(os.path.join(img_dir, f))["image"].astype(np.float32)
            lbl_path = os.path.join(lbl_dir, f)
            if os.path.exists(lbl_path):
                lbl = np.load(lbl_path)["label"]
            else:
                lbl = None

        # Normalize to (H, W, C) — pipeline expects channels-last
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
            img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)  # (H, W) -> (H, W, 3)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)  # (H, W, 1) -> (H, W, 3)

        images.append(img)
        labels.append(lbl)

    return images, labels


def run_ablation(ablation_id, config, source_model, source_images, target_images,
                 source_labels, target_labels, device, max_iterations=3, llm_client=None):
    """Run a single ablation experiment using the Orchestrator."""
    ablation = ABLATIONS[ablation_id]
    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION {ablation_id}: {ablation['name']}")
    logger.info(f"{'='*60}")

    # Deep copy config for this ablation
    cfg = copy.deepcopy(config)
    cfg.device = str(device)
    cfg.max_iterations = max_iterations

    # Apply ablation-specific modifications
    if ablation_id == "A3":
        cfg.max_iterations = 1
    elif ablation_id == "A6":
        cfg.containment_rules = []
        cfg.anatomy_priors = {}
    elif ablation_id == "A7":
        cfg.max_iterations = 1

    # Build pipeline state
    state = PipelineState(
        config=cfg,
        source_model=copy.deepcopy(source_model),
        source_images=source_images,
        target_images=target_images,
        source_labels=source_labels,
        target_labels=target_labels,
        llm_client=None if ablation_id == "A4" else llm_client,
    )

    orchestrator = Orchestrator(state)

    # ---- A1: w/o RAG knowledge base ----
    # Must monkeypatch KnowledgeAgent to prevent KB lookup entirely.
    # Clearing config fields alone is insufficient because task_name
    # still matches the KB entry.
    if ablation_id == "A1":
        original_knowledge_run = orchestrator.knowledge_agent.run
        def noop_knowledge():
            logger.info("  [A1] Skipping RAG knowledge, using empty defaults")
            # Set generic "unknown" knowledge — forces plan to rely purely
            # on statistical analysis without any domain-specific guidance.
            state.rag_knowledge = {
                "shift_types": [],
                "intensity_mapping_type": "unknown",
                "standardization_gap": "unknown",
                "contrast_mechanism_change": "unknown",
                "noise_model_change": "unknown",
                "boundary_quality_change": "unknown",
                "dominant_shift_factor": "unknown",
                "recommended_scope": "unknown",
                "recommended_mode": "unknown",
                "recommended_lr_range": "unknown",
                "what_will_fail": [],
                "prior_difficulty": "unknown",
                "anatomy_priors": {},
                "containment_rules": [],
            }
        orchestrator.knowledge_agent.run = noop_knowledge

    # ---- A2: w/o failure diagnosis ----
    # Skip diagnosis agent AND clear diagnosis-dependent fields so
    # the plan uses default (uniform) loss weights instead of
    # failure-type-specific ones. Also clear diagnosis info from
    # verification to prevent indirect influence on severity assessment.
    if ablation_id == "A2":
        original_diag = orchestrator.diagnosis_agent.run
        def noop_diagnosis():
            logger.info("  [A2] Skipping diagnosis agent")
            state.failure_report = None
        orchestrator.diagnosis_agent.run = noop_diagnosis
        # Also patch verification to not use diagnosis signals
        original_verify = orchestrator.verification_agent.run
        def verify_without_diagnosis():
            original_verify()
            # After verification, clear diagnosis-dependent severity fields
            # so plan uses pure knowledge + stats without diagnosis signals.
            vs = state.verification.get("verified_severity", {})
            if vs:
                # Recompute classifier_severity without failure report info:
                # just use prediction quality (fg_ratio + confidence)
                report = state.shift_report
                if report:
                    fg_ratio = sum(v for k, v in report.prediction_distribution.items()
                                   if k != "background") if report.prediction_distribution else 0
                    mean_conf = getattr(report, 'mean_confidence', 0.5) or 0.5
                    if fg_ratio < 0.005:
                        raw_severity = 0.9
                    elif fg_ratio < 0.02:
                        raw_severity = 0.7
                    else:
                        raw_severity = max(0.0, 1.0 - mean_conf) if mean_conf > 0.9 else 0.3
                    vs["classifier_severity"] = raw_severity
        orchestrator.verification_agent.run = verify_without_diagnosis

    # ---- A5: w/o self-evaluation Q(x) ----
    # Disable ALL quality-score-driven decisions:
    # 1. Early stopping (quality degradation check)
    # 2. Best model reversion (tracking best quality)
    # 3. Quality-based rediagnosis (stagnation detection)
    # Without Q(x), the agent always uses the final model from
    # the last iteration, even if an earlier iteration was better.
    if ablation_id == "A5":
        orchestrator._should_stop = lambda iteration: False
        orchestrator._should_rediagnose = lambda iteration: False
        # Prevent best model tracking by making update a no-op
        original_eval_run = orchestrator.evaluation_agent.run
        def eval_without_tracking():
            result = original_eval_run()
            # Clear best model state so reversion never happens
            state.best_model_state = None
            state.best_quality_score = 0.0
            return result
        orchestrator.evaluation_agent.run = eval_without_tracking

    # ---- A7: BN reset only ----
    # Enable BN adaptation but disable all self-supervised losses.
    if ablation_id == "A7":
        original_design = orchestrator.adaptation_agent.design_plan
        def bn_only_plan():
            return {
                "learning_rate": 0.0,
                "num_steps": 0,
                "loss_weights": {"bn_adapt": 1.0},
                "trainable_scope": "bn_only",
                "use_anatomy_refinement": False,
                "rationale": "[A7 ablation] BN reset only, no self-supervised losses",
            }
        orchestrator.adaptation_agent.design_plan = bn_only_plan

    t0 = time.time()
    results = orchestrator.run(max_iterations=cfg.max_iterations)
    elapsed = time.time() - t0

    # Extract per-class dice from iteration history
    final_metrics = {}
    if state.iteration_history:
        last = state.iteration_history[-1]
        final_metrics = {k: v for k, v in last.items() if "dice" in k or "quality" in k}

    # Check postprocessed results
    if state.postprocess_results:
        for k, v in state.postprocess_results.items():
            if "dice" in k:
                final_metrics[f"{k}_post"] = v

    # Extract adaptation plan details for auditability
    plan_summary = {}
    if state.adaptation_plan:
        plan_summary = {
            "trainable_scope": state.adaptation_plan.get("trainable_scope", "unknown"),
            "num_steps": state.adaptation_plan.get("num_steps", 0),
            "learning_rate": state.adaptation_plan.get("learning_rate", 0),
            "loss_weights": state.adaptation_plan.get("loss_weights", {}),
            "skip_adaptation": state.adaptation_plan.get("skip_adaptation", False),
            "rationale": state.adaptation_plan.get("rationale", ""),
        }

    summary = {
        "ablation_id": ablation_id,
        "ablation_name": ablation["name"],
        "final_metrics": final_metrics,
        "plan": plan_summary,
        "quality_scores": [float(q) for q in state.quality_scores],
        "time_seconds": elapsed,
        "num_iterations": len(state.iteration_history),
    }

    logger.info(f"  Ablation {ablation_id} complete in {elapsed:.1f}s")
    logger.info(f"  Metrics: {final_metrics}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run MedAdaptAgent ablation studies")
    parser.add_argument("--dataset", default="fundus_roi",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ablation", default="all",
                        help="Ablation ID (A1-A7) or 'all'")
    parser.add_argument("--max_source", type=int, default=100)
    parser.add_argument("--max_target", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Max iterations (must match full pipeline for fair comparison)")
    parser.add_argument("--llm_api_key", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = DATASET_CONFIGS[args.dataset]

    # Load model
    ckpt_path = os.path.join(str(PROJECT_ROOT), cfg["checkpoint"])
    model = UNet(in_channels=3, num_classes=cfg["num_classes"], base_ch=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded from {ckpt_path}")

    # Load data
    source_dir = os.path.join(str(PROJECT_ROOT), cfg["source_dir"])
    target_dir = os.path.join(str(PROJECT_ROOT), cfg["target_dir"])
    source_images, source_labels = load_npz_dataset(source_dir, args.max_source)
    target_images, target_labels = load_npz_dataset(target_dir, args.max_target)
    logger.info(f"Source: {len(source_images)} images, Target: {len(target_images)} images")

    # LLM client
    llm_client = None
    if args.llm_api_key:
        try:
            import anthropic
            llm_client = anthropic.Anthropic(api_key=args.llm_api_key)
        except ImportError:
            pass

    # Determine which ablations to run
    if args.ablation == "all":
        ablation_ids = list(ABLATIONS.keys())
    else:
        ablation_ids = [a.strip() for a in args.ablation.split(",")]

    task_config = cfg["task_config"]

    # Run ablations
    all_results = {}
    for aid in ablation_ids:
        if aid not in ABLATIONS:
            logger.warning(f"Unknown ablation: {aid}, skipping")
            continue
        all_results[aid] = run_ablation(
            aid, task_config, model,
            source_images, target_images,
            source_labels, target_labels,
            device, args.max_iterations, llm_client,
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    for aid, r in all_results.items():
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in r["final_metrics"].items()
                                if isinstance(v, float))
        logger.info(f"  {aid} ({r['ablation_name']:30s}): {metrics_str}")

    # Save
    output_dir = os.path.join(str(PROJECT_ROOT), "experiments", "ablations", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ablation_results.json")

    # Merge with existing
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
    existing.update(all_results)

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
