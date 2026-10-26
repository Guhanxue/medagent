"""
STU-Net wrapper for MedAdaptAgent domain adaptation pipeline.

STU-Net (Scalable and Transferable U-Net) is based on nnUNet v1 with:
- InstanceNorm3d normalization (key for our norm reset adaptation)
- Residual connections for stable gradient flow at scale
- Pre-trained on TotalSegmentator (104 classes, 3D CT)

Variants: STU-Net-S (14.6M), STU-Net-B (58.3M), STU-Net-L (440M), STU-Net-H (1.46B)

For MedAdaptAgent, this wrapper:
1. Loads pre-trained STU-Net weights
2. Replaces the output head for target task (e.g., brain segmentation)
3. Provides standard forward() interface for the adaptation pipeline
4. Supports 2D slice-wise or 3D volumetric inference
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class ConvDropoutNormNonlin(nn.Module):
    """nnUNet-style conv block: Conv -> InstanceNorm -> LeakyReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 spatial_dims=3):
        super().__init__()
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        Norm = nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
        self.conv = Conv(in_ch, out_ch, kernel_size, stride=stride,
                         padding=padding, bias=False)
        self.norm = Norm(out_ch, affine=True, track_running_stats=True)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with two ConvDropoutNormNonlin layers."""

    def __init__(self, in_ch, out_ch, stride=1, spatial_dims=3):
        super().__init__()
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.conv1 = ConvDropoutNormNonlin(in_ch, out_ch, stride=stride,
                                           spatial_dims=spatial_dims)
        self.conv2 = ConvDropoutNormNonlin(out_ch, out_ch,
                                           spatial_dims=spatial_dims)
        self.residual = (Conv(in_ch, out_ch, 1, stride=stride, bias=False)
                         if in_ch != out_ch or stride != 1 else nn.Identity())

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)


class STUNetLite(nn.Module):
    """Lightweight STU-Net implementation for MedAdaptAgent.

    Follows STU-Net-B architecture but can be scaled. This standalone
    implementation avoids requiring nnUNet v1 as a dependency.
    Supports both 2D (Conv2d) and 3D (Conv3d) modes.
    """

    def __init__(self, in_channels=1, num_classes=104, base_ch=32,
                 stages=(1, 2, 2, 2, 2), spatial_dims=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims

        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        ConvT = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        channels = [base_ch * (2 ** i) for i in range(len(stages))]

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = in_channels
        for i, (n_blocks, out_ch) in enumerate(zip(stages, channels)):
            blocks = nn.Sequential(*[
                ResBlock(ch if j == 0 else out_ch, out_ch,
                         spatial_dims=spatial_dims)
                for j in range(n_blocks)
            ])
            self.encoder_blocks.append(blocks)
            if i < len(stages) - 1:
                self.downsamples.append(
                    Conv(out_ch, out_ch, 2, stride=2, bias=False))
            ch = out_ch

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(stages) - 2, -1, -1):
            up_ch = channels[i + 1]
            skip_ch = channels[i]
            self.upsamples.append(
                ConvT(up_ch, skip_ch, 2, stride=2, bias=False))
            self.decoder_blocks.append(
                ResBlock(skip_ch * 2, skip_ch, spatial_dims=spatial_dims))

        # Output head
        self.seg_outputs = Conv(channels[0], num_classes, 1)

    def forward(self, x):
        interp_mode = 'bilinear' if self.spatial_dims == 2 else 'trilinear'

        # Encoder
        skips = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[i](x)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode=interp_mode,
                                  align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        return self.seg_outputs(x)

    def get_encoder_features(self, x):
        features = {}
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            features[f"enc{i}"] = x
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        features["bottleneck"] = x
        return features


class STUNetWrapper(nn.Module):
    """Wrapper for STU-Net that provides a standard interface for MedAdaptAgent.

    Handles:
    - Loading pre-trained STU-Net weights (official or our format)
    - Output head replacement for target task
    - 2D slice-wise inference mode (for 2D pipeline compatibility)
    - Standard forward() returning (B, C, H, W) or (B, C, D, H, W)
    """

    def __init__(self, num_classes: int = 104, in_channels: int = 1,
                 base_ch: int = 32, mode: str = "3d",
                 device: str = "cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.mode = mode
        self.device_str = device

        spatial_dims = 2 if mode == "2d" else 3
        self.model = STUNetLite(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_ch,
            spatial_dims=spatial_dims,
        )
        self.model_name = "STU-Net-B"

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """Load pre-trained weights.

        Handles both official nnUNet format and standard PyTorch checkpoints.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Official nnUNet format: state_dict is nested
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "network_weights" in ckpt:
            state_dict = ckpt["network_weights"]
        else:
            state_dict = ckpt

        # Remove 'module.' prefix from DDP training
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=strict)
        if missing and strict:
            print(f"Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"Note: {len(unexpected)} unexpected keys (pre-trained head)")
        return self

    def replace_output_head(self, new_num_classes: int):
        """Replace segmentation head for target task."""
        Conv = nn.Conv2d if self.mode == "2d" else nn.Conv3d
        old_in_ch = self.model.seg_outputs.in_channels
        self.model.seg_outputs = Conv(old_in_ch, new_num_classes, 1)
        self.num_classes = new_num_classes
        nn.init.kaiming_normal_(self.model.seg_outputs.weight)
        nn.init.zeros_(self.model.seg_outputs.bias)

    def forward(self, x):
        """Standard forward pass.

        For 2D mode (spatial_dims=2): expects (B, C, H, W).
        For 3D mode (spatial_dims=3): expects (B, C, D, H, W).
        """
        return self.model(x)

    def freeze_encoder(self):
        """Freeze encoder, keep decoder trainable."""
        for block in self.model.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        for ds in self.model.downsamples:
            for param in ds.parameters():
                param.requires_grad = False

    def param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
