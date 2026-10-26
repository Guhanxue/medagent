"""
Swin UNETR wrapper for MedAdaptAgent domain adaptation pipeline.

Wraps the CLIP-Driven Universal Model (ICCV 2023), which uses:
- Swin UNETR backbone (from MONAI)
- CLIP text embeddings for organ-specific task conditioning
- Pre-trained on 14 CT datasets (32 organ/tumor classes)

For MedAdaptAgent, this wrapper:
1. Loads MONAI's SwinUNETR with pre-trained weights
2. Supports output head replacement for target task
3. Provides standard forward() interface for adaptation pipeline
4. LayerNorm architecture -> adapted via gradient updates (no BN reset)

Requires: pip install monai
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class SwinUNETRWrapper(nn.Module):
    """Wrapper for Swin UNETR / CLIP-Driven Universal Model.

    Two loading modes:
    1. MONAI SwinUNETR (standard): Uses monai.networks.nets.SwinUNETR
    2. CLIP-Driven Universal: Loads official checkpoint with key remapping

    Both use LayerNorm (transformer architecture), detected by our pipeline
    as 'transformer' type for norm adaptation.
    """

    def __init__(self, num_classes: int = 32, in_channels: int = 1,
                 img_size: Tuple[int, ...] = (96, 96, 96),
                 feature_size: int = 48, mode: str = "3d",
                 device: str = "cuda"):
        """
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels (1 for CT).
            img_size: Input spatial size (3D).
            feature_size: Swin UNETR feature dimension.
            mode: '3d' for volumetric, '2d' for slice-wise inference.
            device: Device string.
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size
        self.feature_size = feature_size
        self.mode = mode
        self.device_str = device
        self.model = None
        self.model_name = "SwinUNETR"
        self._use_sigmoid = False  # CLIP-Driven uses sigmoid, standard uses softmax

    def build_model(self):
        """Build MONAI SwinUNETR model."""
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError(
                "MONAI is required for SwinUNETR. Install: pip install monai")

        import inspect
        sig = inspect.signature(SwinUNETR.__init__)

        spatial_dims = 2 if self.mode == "2d" else 3

        if "spatial_dims" in sig.parameters:
            self.model = SwinUNETR(
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                feature_size=self.feature_size,
                spatial_dims=spatial_dims,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=False,
            )
        elif "img_size" in sig.parameters:
            img_size = self.img_size if spatial_dims == 3 else self.img_size[:2]
            self.model = SwinUNETR(
                img_size=img_size,
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                feature_size=self.feature_size,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=False,
            )
        return self

    def load_monai_pretrained(self, checkpoint_path: str):
        """Load standard MONAI SwinUNETR pre-trained weights."""
        if self.model is None:
            self.build_model()

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))

        # Remove 'module.' prefix
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"  SwinUNETR: {len(missing)} missing keys "
                  f"(expected if head differs)")
        return self

    def load_clip_universal(self, checkpoint_path: str,
                            word_embedding_path: str = None):
        """Load CLIP-Driven Universal Model weights.

        Official checkpoint uses 'net' key and needs key remapping.
        """
        if self.model is None:
            self.build_model()

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        load_dict = ckpt.get("net", ckpt.get("state_dict", ckpt))

        # Remap keys from CLIP-Driven format to MONAI SwinUNETR format
        store_dict = {}
        for key, value in load_dict.items():
            # Remove 'module.' prefix from DDP
            clean_key = key.replace("module.", "")

            # CLIP-Driven stores backbone params with 'backbone.' prefix
            if clean_key.startswith("backbone."):
                clean_key = clean_key[len("backbone."):]

            store_dict[clean_key] = value

        missing, unexpected = self.model.load_state_dict(
            store_dict, strict=False)
        if missing:
            print(f"  CLIP-Universal: {len(missing)} missing keys")
        if unexpected:
            print(f"  CLIP-Universal: {len(unexpected)} unexpected keys")

        self.model_name = "CLIP-Driven Universal Model"
        self._use_sigmoid = True  # CLIP-Driven uses per-channel sigmoid
        return self

    def replace_output_head(self, new_num_classes: int):
        """Replace output head for target task.

        MONAI SwinUNETR uses self.out as the final conv layer.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")

        ConvNd = nn.Conv2d if self.mode == "2d" else nn.Conv3d

        # MONAI SwinUNETR final layer
        if hasattr(self.model, 'out'):
            old_out = self.model.out
            if isinstance(old_out, nn.Sequential):
                # Find the last Conv layer
                last_conv = None
                for i, layer in enumerate(old_out):
                    if isinstance(layer, (nn.Conv3d, nn.Conv2d)):
                        last_conv = (i, layer)
                if last_conv is not None:
                    idx, conv = last_conv
                    new_conv = ConvNd(conv.in_channels, new_num_classes,
                                     conv.kernel_size, conv.stride,
                                     conv.padding)
                    nn.init.kaiming_normal_(new_conv.weight)
                    nn.init.zeros_(new_conv.bias)
                    old_out[idx] = new_conv
            elif isinstance(old_out, (nn.Conv3d, nn.Conv2d)):
                self.model.out = ConvNd(
                    old_out.in_channels, new_num_classes,
                    old_out.kernel_size, old_out.stride, old_out.padding)
                nn.init.kaiming_normal_(self.model.out.weight)
                nn.init.zeros_(self.model.out.bias)

        self.num_classes = new_num_classes
        self._use_sigmoid = False

    def forward(self, x):
        """Standard forward pass.

        For 2D mode (spatial_dims=2): expects (B, C, H, W).
        For 3D mode (spatial_dims=3): expects (B, C, D, H, W).
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model(x)

    def predict(self, x):
        """Predict with appropriate activation (sigmoid or softmax)."""
        logits = self.forward(x)
        if self._use_sigmoid:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=1)

    def sliding_window_inference(self, volume, roi_size=(96, 96, 96),
                                  overlap=0.5):
        """3D sliding window inference for full volumes.

        Args:
            volume: (B, C, D, H, W) input volume.
            roi_size: Patch size for inference.
            overlap: Overlap fraction between patches.
        """
        try:
            from monai.inferers import sliding_window_inference as swi
            return swi(volume, roi_size, 1, self.forward,
                       overlap=overlap, mode='gaussian')
        except ImportError:
            # Fallback: simple patch-based inference
            return self.forward(volume)

    def freeze_encoder(self):
        """Freeze Swin transformer encoder, keep decoder trainable."""
        if self.model is None:
            return
        if hasattr(self.model, 'swinViT'):
            for param in self.model.swinViT.parameters():
                param.requires_grad = False
        # Also freeze encoder conv stages if they exist
        for name, param in self.model.named_parameters():
            if 'encoder' in name and 'decoder' not in name:
                param.requires_grad = False

    def param_count(self) -> Dict[str, int]:
        if self.model is None:
            return {"total": 0, "trainable": 0, "frozen": 0}
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
