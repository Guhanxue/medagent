"""
Vanilla nnU-Net 2D for the norm-agnostic pipeline test.

This is structurally identical to `medadaptagent.models.unet.UNet` except
for the normalization scheme — BatchNorm2d/ReLU is replaced with
InstanceNorm2d/LeakyReLU, matching nnU-Net v1/v2 conventions. No residual
connections, no deep supervision, no autoconfig — we want a clean "UNet
with IN instead of BN" baseline so the M1 norm-agnostic claim is testable
on a pure-vanilla nnU-Net architecture, separate from STU-Net (which adds
residual blocks).

nnU-Net conventions reflected:
  - Conv(bias=True) → InstanceNorm2d(affine=True, track_running_stats=True)
      → LeakyReLU(0.01)
  - track_running_stats=True so `bn_running_stats` capability is present
    (mirrors nnUNet which uses persistent InstanceNorm stats).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NNUNetConvBlock(nn.Module):
    """Double-conv block: Conv-IN-LeakyReLU, Conv-IN-LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=True),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=True),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class NNUNet2D(nn.Module):
    """Vanilla nnU-Net 2D. UNet topology, InstanceNorm + LeakyReLU.

    Shape contract: forward(x: [B, in_channels, H, W]) -> [B, num_classes, H, W].
    `base_ch=32` gives ~7M params (comparable to our UNet(base_ch=64)=7.8M
    and STU-Net(base_ch=32)=14.7M).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 3,
                 base_ch: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.base_ch = base_ch

        # Encoder
        self.enc1 = NNUNetConvBlock(in_channels, base_ch)
        self.enc2 = NNUNetConvBlock(base_ch, base_ch * 2)
        self.enc3 = NNUNetConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = NNUNetConvBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = NNUNetConvBlock(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = NNUNetConvBlock(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = NNUNetConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = NNUNetConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = NNUNetConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, num_classes, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
