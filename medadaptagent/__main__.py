"""
MedAdaptAgent CLI: one-command model adaptation.

Usage:
    # Basic: provide model + target data
    medadapt --model checkpoints/unet_ct.pt --target data/mri_target/

    # With source data (enables FDA for cross-modality)
    medadapt --model checkpoints/unet_ct.pt --target data/mri_target/ --source data/ct_source/

    # Specify task info
    medadapt --model checkpoints/unet_ct.pt --target data/mri_target/ \
        --task "liver_segmentation" --source-domain "CT abdominal" --target-domain "MRI T2-SPIR"

    # With annotations (few-shot)
    medadapt --model checkpoints/unet_ct.pt --target data/mri_target/ --n-shots 5

    # Choose LLM backend
    medadapt --model checkpoints/unet_ct.pt --target data/mri_target/ --llm azure --llm-model gpt-5.2-2025-12-11

    # Save output
    medadapt --model checkpoints/unet_ct.pt --target data/mri_target/ --output adapted_output/
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

from medadaptagent.graph.pipeline import run_pipeline
from medadaptagent.pipeline_config import TaskConfig
from medadaptagent.knowledge.llm_client import create_llm_client


def _is_monai_bundle(path: str) -> bool:
    """A MONAI bundle is a directory containing configs/inference.{json,yaml}
    and models/model.pt."""
    if not os.path.isdir(path):
        return False
    cfg = os.path.join(path, "configs")
    if not os.path.isdir(cfg):
        return False
    has_cfg = any(os.path.exists(os.path.join(cfg, f))
                    for f in ("inference.json", "inference.yaml"))
    has_ckpt = os.path.exists(os.path.join(path, "models", "model.pt"))
    return has_cfg and has_ckpt


def _load_monai_bundle(bundle_dir: str, device: str):
    """Bundle-agnostic path: read sidecar for num_classes + chain, load
    network via ConfigParser. Returns (model, info, task_adapter, chain)
    ready for the adaptation pipeline."""
    import json as _json
    from scripts._source_loader import load_source_model
    from medadaptagent.calibration import (
        _build_monai_chain_fn, build_label_chain_fn,
    )
    from medadaptagent.tasks import get_task

    sidecar_path = os.path.join(bundle_dir, "models", "model.meta.json")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(
            f"MONAI bundle missing model.meta.json sidecar at "
            f"{sidecar_path}. Write one declaring n_classes + "
            f"preprocessing_chain + label_chain (see "
            f"data/monai_bundles/*/models/model.meta.json for the "
            f"schema).")
    sidecar = _json.load(open(sidecar_path))
    n_classes = int(sidecar.get("n_classes", 2))
    print(f"  MONAI bundle: {bundle_dir}")
    print(f"  Sidecar: n_classes={n_classes}, "
           f"chain={len(sidecar.get('preprocessing_chain', []))} transforms")

    ckpt_path = os.path.join(bundle_dir, "models", "model.pt")
    model = load_source_model(
        ckpt_rel=ckpt_path, num_classes=n_classes, device=device,
        arch="monai_bundle")

    task = get_task("segmentation_3d")
    if "preprocessing_chain" in sidecar:
        pre_fn = _build_monai_chain_fn(sidecar["preprocessing_chain"])
        size = int(sidecar.get("input_size", 96))
        task.set_source_preprocessing("monai_chain", pre_fn, size)
    lbl_fn = (build_label_chain_fn(sidecar["label_chain"])
                if "label_chain" in sidecar else None)

    info = {
        "architecture": "monai_bundle",
        "in_channels": 1,
        "num_classes": n_classes,
        "norm_type": "auto",
        "total_params": sum(p.numel() for p in model.parameters()),
        "modality": sidecar.get("modality"),
    }
    print(f"  Loaded {info['total_params']/1e6:.1f}M params via ConfigParser")
    return model, info, task, lbl_fn


def auto_detect_model(checkpoint_path: str, device: str):
    """Auto-detect model architecture from checkpoint.

    Examines the state_dict keys to determine:
    - Architecture (UNet, SwinUNETR, SegResNet, nnUNet)
    - Number of input channels
    - Number of output classes
    - Normalization type (BatchNorm, InstanceNorm, LayerNorm)
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif isinstance(ckpt, dict) and any(k.startswith("encoder") or k.startswith("inc") for k in ckpt):
        state_dict = ckpt
        metadata = {}
    else:
        state_dict = ckpt
        metadata = {}

    # Detect num_classes from output layer
    num_classes = None
    for key in reversed(list(state_dict.keys())):
        if "out" in key and "weight" in key and state_dict[key].dim() >= 2:
            num_classes = state_dict[key].shape[0]
            break
    if num_classes is None:
        # Last conv layer
        for key in reversed(list(state_dict.keys())):
            if "weight" in key and state_dict[key].dim() == 4:
                num_classes = state_dict[key].shape[0]
                break

    # Detect in_channels from first conv
    in_channels = 3  # default
    for key in list(state_dict.keys()):
        if "weight" in key and state_dict[key].dim() == 4:
            in_channels = state_dict[key].shape[1]
            break

    # Detect architecture
    keys_str = " ".join(state_dict.keys())
    if "inc." in keys_str and "down" in keys_str and "up" in keys_str:
        arch = "unet"
    elif "swinViT" in keys_str or "swin" in keys_str:
        arch = "swin_unetr"
    elif "encoder" in keys_str and "decoder" in keys_str:
        arch = "encoder_decoder"
    else:
        arch = "unet"  # default

    # Detect norm type
    has_bn = any("running_mean" in k for k in state_dict)
    has_ln = any("layernorm" in k.lower() or "layer_norm" in k.lower() for k in state_dict)
    norm_type = "batchnorm" if has_bn else ("layernorm" if has_ln else "instancenorm")

    # Load model
    if arch == "unet":
        from medadaptagent.models.unet import UNet
        model = UNet(in_channels=in_channels, num_classes=num_classes, base_ch=64)
    else:
        from medadaptagent.models.unet import UNet
        model = UNet(in_channels=in_channels, num_classes=num_classes, base_ch=64)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"  Warning: state_dict mismatch ({e}). Trying to load with strict=False.")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    info = {
        "architecture": arch,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "norm_type": norm_type,
        "total_params": sum(p.numel() for p in model.parameters()),
        "source_val_dice": metadata.get("best_dice"),
    }

    print(f"  Architecture: {arch}")
    print(f"  Classes: {num_classes}, Channels: {in_channels}")
    print(f"  Norm: {norm_type}, Params: {info['total_params']/1e6:.1f}M")

    return model, info


def _load_nifti_volumes(img_dir: str, lbl_dir: str, max_samples=None):
    """Load 3D .nii.gz volumes from imagesTs/labelsTs (MSD layout).
    Returns lists of Volume objects (numpy + .affine) for the adaptation
    pipeline."""
    import nibabel as nib
    from medadaptagent.calibration import Volume
    files = sorted(f for f in os.listdir(img_dir)
                     if f.endswith((".nii", ".nii.gz")))
    if max_samples:
        files = files[:max_samples]
    images, labels = [], []
    for f in files:
        ni = nib.load(os.path.join(img_dir, f))
        images.append(Volume.from_array(
            ni.get_fdata().astype(np.float32),
            ni.affine.astype(np.float32)))
        lbl_p = os.path.join(lbl_dir, f)
        if os.path.exists(lbl_p):
            nl = nib.load(lbl_p)
            labels.append(Volume.from_array(
                nl.get_fdata().astype(np.int64),
                nl.affine.astype(np.float32)))
        else:
            labels.append(None)
    return images, labels


def load_data(data_dir: str, max_samples: int = None):
    """Load images and labels from a directory.

    Supports:
    - 3D MSD layout: imagesTs/ + labelsTs/ (.nii.gz) — for MONAI bundles
    - 2D images/ + labels/ subdirectories (separate .npz)
    - combined .npz files with 'image' + 'label' keys
    - .png/.jpg images (labels optional)
    """
    # 3D imagesTs/labelsTs layout (MSD-style, used by many 3D datasets)
    imgs_ts = os.path.join(data_dir, "imagesTs")
    lbls_ts = os.path.join(data_dir, "labelsTs")
    if os.path.isdir(imgs_ts):
        return _load_nifti_volumes(imgs_ts, lbls_ts, max_samples)
    # Top-level .nii.gz volumes
    if any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(data_dir)):
        sibling_lbl = data_dir.replace("imagesTs", "labelsTs")
        if os.path.isdir(sibling_lbl) and sibling_lbl != data_dir:
            return _load_nifti_volumes(data_dir, sibling_lbl, max_samples)
        return _load_nifti_volumes(data_dir, data_dir, max_samples)
    img_dir = os.path.join(data_dir, "images")
    lbl_dir = os.path.join(data_dir, "labels")

    if os.path.isdir(img_dir):
        # Separate images/labels dirs
        files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])
        if max_samples:
            files = files[:max_samples]
        images, labels = [], []
        for f in files:
            img = np.load(os.path.join(img_dir, f))["image"]
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
                img = img.transpose(1, 2, 0)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            images.append(img.astype(np.float32))
            lbl_path = os.path.join(lbl_dir, f)
            labels.append(np.load(lbl_path)["label"] if os.path.exists(lbl_path) else None)
    else:
        # Combined npz or flat directory
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        if max_samples:
            files = files[:max_samples]
        images, labels = [], []
        for f in files:
            data = np.load(os.path.join(data_dir, f))
            img = data["image"].astype(np.float32)
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
                img = img.transpose(1, 2, 0)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            images.append(img)
            labels.append(data["label"] if "label" in data else None)

    return images, labels


def main():
    parser = argparse.ArgumentParser(
        prog="medadapt",
        description="MedAdaptAgent: One-command medical image segmentation model adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  medadapt --model model.pt --target target_data/
  medadapt --model model.pt --target target_data/ --source source_data/
  medadapt --model model.pt --target target_data/ --output adapted/
  medadapt --model model.pt --target target_data/ --llm azure --llm-model gpt-5.2-2025-12-11
        """)

    # Required
    parser.add_argument("--model", required=True,
                        help="Path to source model checkpoint (.pt)")
    parser.add_argument("--target", required=True,
                        help="Path to target data directory")

    # Optional data
    parser.add_argument("--source", default=None,
                        help="Path to source data (enables FDA for cross-modality)")
    parser.add_argument("--n-shots", type=int, default=0,
                        help="Number of annotated target volumes for few-shot")

    # Task info (agent auto-fills if not provided)
    parser.add_argument("--task", default="",
                        help="Task name (e.g., 'liver_segmentation')")
    parser.add_argument("--source-domain", default="",
                        help="Source domain description (agent infers if empty)")
    parser.add_argument("--target-domain", default="",
                        help="Target domain description (agent infers if empty)")

    # LLM
    parser.add_argument("--llm", default=None,
                        choices=["azure", "anthropic", "openai"],
                        help="LLM backend (auto-detects if not set)")
    parser.add_argument("--llm-model", default=None,
                        help="LLM model name")

    # Output
    parser.add_argument("--output", default=None,
                        help="Directory to save adapted model + report")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device")

    # Advanced
    parser.add_argument("--max-source", type=int, default=100,
                        help="Max source images to load")
    parser.add_argument("--max-target", type=int, default=None,
                        help="Max target images to load")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max adaptation iterations")

    args = parser.parse_args()

    print("=" * 60)
    print("MedAdaptAgent")
    print("=" * 60)

    # Step 1: Auto-detect model
    print("\n[1/4] Loading model...")
    device = torch.device(args.device)
    bundle_extras = None
    if _is_monai_bundle(args.model):
        model, model_info, task_adapter, label_chain_fn = _load_monai_bundle(
            args.model, str(device))
        bundle_extras = {"task": task_adapter, "label_chain": label_chain_fn}
    else:
        model, model_info = auto_detect_model(args.model, device)

    # Step 2: Load data
    print("\n[2/4] Loading data...")
    target_images, target_labels = load_data(args.target, args.max_target)
    print(f"  Target: {len(target_images)} images")

    source_images, source_labels = [], []
    if args.source:
        source_images, source_labels = load_data(args.source, args.max_source)
        print(f"  Source: {len(source_images)} images")

    # Step 3: Setup LLM
    print("\n[3/4] Setting up LLM...")
    llm = create_llm_client(backend=args.llm, model=args.llm_model)
    if not llm:
        print("  WARNING: No LLM available. Using rule-based fallback.")
        print("  Set AZURE_OPENAI_API_KEY or ANTHROPIC_API_KEY for LLM reasoning.")

    # Step 4: Build TaskConfig (agent will auto-fill missing fields)
    config = TaskConfig(
        task_name=args.task or "auto_detected",
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        structures={"background": 0} | {f"class_{i}": i
                     for i in range(1, model_info["num_classes"])},
        image_size=256,
        max_iterations=args.max_iterations,
    )

    # Step 5: Run pipeline
    print("\n[4/4] Running adaptation pipeline...")
    result = run_pipeline(
        source_model=model,
        source_images=source_images,
        target_images=target_images,
        task_config=config,
        llm_client=llm,
        source_labels=source_labels,
        target_labels=target_labels,
        device=str(device),
        save_dir=args.output,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    delivery = result.get("delivery", {})
    print(f"  Final Dice: {delivery.get('final_dice', 0)*100:.1f}%")

    scaling = delivery.get("scaling_recommendation", {})
    if scaling.get("annotation_needed"):
        print(f"  Recommendation: annotate {scaling.get('recommended_n_volumes', '?')} volumes")
        est = scaling.get("estimated_performance", {})
        for n, v in est.items():
            print(f"    {n}: ~{v}%")

    if args.output:
        print(f"\n  Output saved to: {args.output}/")
        print(f"    adapted_model.pt — ready to deploy")
        print(f"    adaptation_report.json — full reasoning trace")


if __name__ == "__main__":
    main()
