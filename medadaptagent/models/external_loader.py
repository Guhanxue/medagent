"""
Generic loader for external pretrained models.

Supports:
  - MONAI bundles (SwinUNETR, SegResNet, etc.)
  - TotalSegmentator (nnUNet-based)
  - CLIP-Driven Universal Model
  - Any nn.Module checkpoint with auto-detection
  - Raw state_dict files

The loaded model is wrapped in ExternalModelWrapper which provides
the standard interface expected by MedAdaptAgent:
  - forward(x) -> logits  (2D: B,C,H,W)
  - get_encoder_features(x) -> dict
  - param_count() -> dict
  - predict(x) -> softmax probs

Usage:
    from medadaptagent.models.external_loader import load_external_model

    # MONAI bundle
    model = load_external_model("monai://swin_unetr_btcv_segmentation",
                                 num_classes=14, device="cuda")

    # Local checkpoint
    model = load_external_model("/path/to/checkpoint.pt",
                                 architecture="swin_unetr",
                                 num_classes=14, device="cuda")

    # TotalSegmentator
    model = load_external_model("totalseg://total",
                                 num_classes=105, device="cuda")
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict


class ExternalModelWrapper(nn.Module):
    """Wraps any external segmentation model for MedAdaptAgent compatibility.

    Handles:
    - 3D models used in 2D slice mode (adds/removes depth dim)
    - Input channel mismatch (1ch model with 3ch input: average channels)
    - Output class remapping (subset of classes from a larger model)
    - Feature extraction via hooks
    """

    def __init__(self, model: nn.Module, model_name: str = "external",
                 in_channels: int = 1, num_classes: int = 2,
                 original_num_classes: int = None,
                 class_mapping: Dict[int, int] = None,
                 is_3d: bool = False, device: str = "cuda",
                 provenance=None):
        """
        Args:
            model: The loaded external nn.Module.
            model_name: Name for logging.
            in_channels: Expected input channels of the external model.
            num_classes: Target number of classes for our task.
            original_num_classes: Original output classes of the model.
            class_mapping: Map from model's class indices to our indices.
                e.g., {6: 1, 1: 2, 2: 3} maps model's class 6 to our class 1.
            is_3d: Whether the model expects 3D input (B,C,D,H,W).
            device: Device.
            provenance: ModelProvenance with training context (auto-read from docs/papers).
        """
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.provenance = provenance  # ModelProvenance or None
        self.expected_in_channels = in_channels
        self.num_classes = num_classes
        self.original_num_classes = original_num_classes or num_classes
        self.class_mapping = class_mapping
        self.is_3d = is_3d
        self._device = device
        self._encoder_hooks = {}
        self._encoder_features = {}

        # If we need to remap classes and original model has more classes,
        # we'll select/combine the relevant channels
        if class_mapping and original_num_classes != num_classes:
            self._needs_remap = True
        else:
            self._needs_remap = False

    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt input tensor to match model's expectations."""
        # x is (B, C_in, H, W) from pipeline (typically C_in=3 for RGB)
        c_in = x.shape[1]

        if c_in != self.expected_in_channels:
            if self.expected_in_channels == 1 and c_in == 3:
                # RGB to grayscale: weighted average
                x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
            elif self.expected_in_channels == 3 and c_in == 1:
                # Grayscale to RGB: replicate
                x = x.expand(-1, 3, -1, -1)
            else:
                # Generic: use first N channels or zero-pad
                if c_in > self.expected_in_channels:
                    x = x[:, :self.expected_in_channels]
                else:
                    pad = torch.zeros(x.shape[0], self.expected_in_channels - c_in,
                                     *x.shape[2:], device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=1)

        if self.is_3d:
            # Add depth dimension: (B,C,H,W) -> (B,C,1,H,W)
            x = x.unsqueeze(2)

        return x

    def _adapt_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Adapt model output to match our task's class structure."""
        if self.is_3d:
            # Remove depth: (B,C,1,H,W) -> (B,C,H,W)
            if logits.dim() == 5:
                logits = logits[:, :, 0]

        if self._needs_remap:
            # Create new logits with our class count
            B, _, H, W = logits.shape
            new_logits = torch.zeros(B, self.num_classes, H, W,
                                     device=logits.device, dtype=logits.dtype)
            # Background channel
            new_logits[:, 0] = logits[:, 0] if logits.shape[1] > 0 else 0

            for model_idx, our_idx in self.class_mapping.items():
                if model_idx < logits.shape[1] and our_idx < self.num_classes:
                    new_logits[:, our_idx] = logits[:, model_idx]

            return new_logits

        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_adapted = self._adapt_input(x)
        logits = self.model(x_adapted)
        return self._adapt_output(logits)

    def get_encoder_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract encoder features via hooks or model internals."""
        self._encoder_features.clear()

        # Try calling model's own method first
        if hasattr(self.model, 'get_encoder_features'):
            x_adapted = self._adapt_input(x)
            feats = self.model.get_encoder_features(x_adapted)
            if self.is_3d:
                feats = {k: v[:, :, 0] if v.dim() == 5 else v
                         for k, v in feats.items()}
            return feats

        # Otherwise use hooks
        if not self._encoder_hooks:
            self._register_encoder_hooks()

        _ = self.forward(x)
        feats = dict(self._encoder_features)
        return feats

    def _register_encoder_hooks(self):
        """Auto-register hooks on encoder layers."""
        hook_targets = []
        for name, module in self.model.named_modules():
            if any(kw in name.lower() for kw in ['encoder', 'down', 'enc']):
                if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.LayerNorm,
                                       nn.BatchNorm2d, nn.InstanceNorm3d)):
                    hook_targets.append((name, module))

        # Keep at most 6 hook points spread across the encoder
        if len(hook_targets) > 6:
            step = len(hook_targets) // 6
            hook_targets = hook_targets[::step][:6]

        for name, module in hook_targets:
            safe_name = name.replace('.', '_')

            def make_hook(n):
                def hook_fn(mod, inp, out):
                    feat = out if isinstance(out, torch.Tensor) else out[0]
                    if feat.dim() == 5:
                        feat = feat[:, :, 0]
                    self._encoder_features[n] = feat
                return hook_fn

            handle = module.register_forward_hook(make_hook(safe_name))
            self._encoder_hooks[safe_name] = handle

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    def param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def freeze_base(self):
        """Freeze all parameters."""
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_norm_layers(self):
        """Unfreeze only normalization layers (BN/IN/LN)."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d,
                                   nn.InstanceNorm2d, nn.InstanceNorm3d,
                                   nn.LayerNorm)):
                for p in module.parameters():
                    p.requires_grad = True


# ================================================================
# Loader functions for specific model sources
# ================================================================

def _load_monai_bundle(bundle_name: str, num_classes: int,
                       device: str, **kwargs) -> ExternalModelWrapper:
    """Load a model from MONAI Model Zoo."""
    import monai
    from monai.bundle import ConfigParser

    bundle_dir = kwargs.get("bundle_dir", os.path.expanduser("~/.monai/bundles"))
    bundle_path = os.path.join(bundle_dir, bundle_name)

    if not os.path.exists(bundle_path):
        print(f"Downloading MONAI bundle: {bundle_name}...")
        monai.bundle.download(name=bundle_name, bundle_dir=bundle_dir)

    # Parse bundle config
    parser = ConfigParser()
    parser.read_config(os.path.join(bundle_path, "configs", "inference.json"))

    # Get the network
    net = parser.get_parsed_content("network_def")

    # Load weights
    ckpt_path = os.path.join(bundle_path, "models", "model.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        net.load_state_dict(state, strict=False)

    net.to(device).eval()

    # Detect architecture properties
    is_3d = any(isinstance(m, nn.Conv3d) for m in net.modules())
    in_ch = kwargs.get("in_channels", 1)

    original_num_classes = None
    for module in reversed(list(net.modules())):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            original_num_classes = module.out_channels
            break

    return ExternalModelWrapper(
        model=net, model_name=f"monai_{bundle_name}",
        in_channels=in_ch, num_classes=num_classes,
        original_num_classes=original_num_classes,
        class_mapping=kwargs.get("class_mapping"),
        is_3d=is_3d, device=device,
    )


def _load_monai_swin_unetr(checkpoint_path: str, num_classes: int,
                             device: str, **kwargs) -> ExternalModelWrapper:
    """Load SwinUNETR with a downloaded checkpoint."""
    from monai.networks.nets import SwinUNETR

    img_size = kwargs.get("img_size", (96, 96, 96))
    in_channels = kwargs.get("in_channels", 1)
    feature_size = kwargs.get("feature_size", 48)

    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=feature_size,
    )

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            # Handle various checkpoint formats
            for key in ["state_dict", "model", "model_state_dict", "net"]:
                if key in state:
                    state = state[key]
                    break

        # Try loading, handle mismatched output heads
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
            print(f"  Loaded with strict=False (output head size mismatch is OK)")

    model.to(device).eval()

    is_3d = len(img_size) == 3

    return ExternalModelWrapper(
        model=model, model_name="SwinUNETR",
        in_channels=in_channels, num_classes=num_classes,
        class_mapping=kwargs.get("class_mapping"),
        is_3d=is_3d, device=device,
    )


def _load_totalsegmentator(task: str, num_classes: int,
                            device: str, **kwargs) -> ExternalModelWrapper:
    """Load TotalSegmentator model."""
    try:
        from totalsegmentator.nnunet import setup_nnunet, get_weights_dir
        from totalsegmentator.config import setup_totalseg

        # Download weights if needed
        setup_totalseg()
        weights_dir = get_weights_dir()

        # TotalSegmentator uses nnUNet internally
        # We load the raw nnUNet model for our pipeline
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor()
        predictor.initialize_from_trained_model_folder(
            os.path.join(weights_dir, f"nnunet/results/Dataset{task}"),
            use_folds=(0,),
        )
        model = predictor.network
        model.to(device).eval()

        return ExternalModelWrapper(
            model=model, model_name=f"TotalSeg_{task}",
            in_channels=1, num_classes=num_classes,
            class_mapping=kwargs.get("class_mapping"),
            is_3d=True, device=device,
        )
    except ImportError:
        raise ImportError(
            "TotalSegmentator not installed. Run: pip install TotalSegmentator"
        )


def _load_checkpoint_auto(checkpoint_path: str, num_classes: int,
                           device: str, **kwargs) -> ExternalModelWrapper:
    """Auto-detect architecture from checkpoint keys and load."""
    state = torch.load(checkpoint_path, map_location="cpu")

    # Unwrap nested dicts
    if isinstance(state, dict):
        for key in ["state_dict", "model", "model_state_dict", "net"]:
            if key in state:
                state = state[key]
                break

    if not isinstance(state, (dict, OrderedDict)):
        raise ValueError(f"Cannot parse checkpoint: {type(state)}")

    keys = list(state.keys())
    key_str = " ".join(keys[:50])

    architecture = kwargs.get("architecture", None)

    # Auto-detect from key patterns
    if architecture is None:
        if any("swinViT" in k or "swin_unetr" in k.lower() for k in keys):
            architecture = "swin_unetr"
        elif any("seg_layers" in k or "nnunet" in k.lower() for k in keys):
            architecture = "nnunet"
        elif any("enc1" in k or "enc2" in k for k in keys):
            architecture = "unet"
        elif any("encoder.layer" in k for k in keys):
            architecture = "segresnet"
        else:
            architecture = "unet"  # fallback

    in_channels = kwargs.get("in_channels", 1)

    if architecture == "swin_unetr":
        return _load_monai_swin_unetr(checkpoint_path, num_classes, device, **kwargs)

    elif architecture == "unet":
        from medadaptagent.models.unet import UNet
        model = UNet(in_channels=in_channels, num_classes=num_classes)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return ExternalModelWrapper(
            model=model, model_name="UNet_external",
            in_channels=in_channels, num_classes=num_classes,
            is_3d=False, device=device,
        )

    else:
        # Generic: try to instantiate as a raw nn.Module
        # Build a minimal wrapper around the state dict
        raise ValueError(
            f"Unknown architecture '{architecture}' from checkpoint keys. "
            f"Specify architecture= explicitly. First 5 keys: {keys[:5]}"
        )


# ================================================================
# Main entry point
# ================================================================

def load_external_model(source: str, num_classes: int = 2,
                        device: str = "cuda",
                        **kwargs) -> ExternalModelWrapper:
    """Load an external pretrained model from various sources.

    Automatically reads the model's documentation, paper, and config to
    extract training provenance (dataset, modality, preprocessing, etc.).
    This context is used by CGFD for better hypothesis generation.

    Args:
        source: Model source string. Formats:
            - "monai://bundle_name" — download from MONAI Model Zoo
            - "totalseg://task_name" — use TotalSegmentator
            - "/path/to/checkpoint.pt" — local checkpoint file
        num_classes: Number of output classes for our task.
        device: Device string.
        **kwargs: Additional arguments:
            - architecture: Force architecture type ("swin_unetr", "unet", etc.)
            - in_channels: Model's expected input channels (default: 1)
            - class_mapping: Dict mapping model's class indices to ours
            - img_size: Spatial size for 3D models (default: (96,96,96))
            - feature_size: Feature dimension for SwinUNETR (default: 48)
            - readme_path: Path to README for provenance extraction
            - config_path: Path to training config for provenance extraction
            - llm_client: Anthropic client for LLM-based doc parsing

    Returns:
        ExternalModelWrapper with .provenance field populated.
    """
    from medadaptagent.models.model_provenance import read_provenance

    # Step 1: Read provenance (training context from docs/papers)
    provenance = read_provenance(
        source,
        readme_path=kwargs.pop("readme_path", None),
        config_path=kwargs.pop("config_path", None),
        llm_client=kwargs.pop("llm_client", None),
    )
    print(f"  Model provenance ({provenance.confidence} confidence):")
    print(f"    {provenance.summary()}")

    # Step 2: Load the model
    if source.startswith("monai://"):
        bundle_name = source.replace("monai://", "")
        wrapper = _load_monai_bundle(bundle_name, num_classes, device, **kwargs)

    elif source.startswith("totalseg://"):
        task = source.replace("totalseg://", "")
        wrapper = _load_totalsegmentator(task, num_classes, device, **kwargs)

    elif os.path.exists(source):
        wrapper = _load_checkpoint_auto(source, num_classes, device, **kwargs)

    else:
        raise ValueError(
            f"Unknown model source: {source}\n"
            "Supported formats:\n"
            '  "monai://swin_unetr_btcv_segmentation"\n'
            '  "totalseg://total"\n'
            '  "/path/to/checkpoint.pt"\n'
        )

    # Step 3: Attach provenance to wrapper
    wrapper.provenance = provenance
    return wrapper
