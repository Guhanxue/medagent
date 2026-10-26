"""
Scale Adapter (~0.5M params).
Injected at multiple SAM2 encoder stages (2-3).
Multi-resolution LoRA for handling resolution/size mismatches across domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 16):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class ScaleAdapter(nn.Module):
    """Multi-resolution LoRA adapter for scale/resolution mismatch.

    Processes features at multiple scales via pooling + LoRA branches,
    then fuses multi-scale information back to original resolution.
    """

    def __init__(self, dim: int = 256, rank: int = 16,
                 scales: list = None):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.scales = scales
        self.dim = dim

        self.branches = nn.ModuleList()
        for s in scales:
            branch = nn.Sequential(
                LoRALayer(dim, dim, rank=rank),
                nn.GELU(),
            )
            self.branches.append(branch)

        self.fusion = nn.Sequential(
            nn.Linear(dim * len(scales), dim),
            nn.GELU(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W).
        Returns:
            Scale-adapted feature map (same shape).
        """
        B, C, H, W = x.shape
        multi_scale_features = []

        for s, branch in zip(self.scales, self.branches):
            if s > 1:
                # Downsample → process → upsample
                x_down = F.adaptive_avg_pool2d(x, (max(1, H // s), max(1, W // s)))
                x_down_perm = x_down.permute(0, 2, 3, 1)
                x_branch = branch(x_down_perm)
                x_branch = x_branch.permute(0, 3, 1, 2)
                x_branch = F.interpolate(x_branch, size=(H, W), mode="bilinear",
                                         align_corners=False)
            else:
                x_perm = x.permute(0, 2, 3, 1)
                x_branch = branch(x_perm).permute(0, 3, 1, 2)

            multi_scale_features.append(x_branch)

        # Fuse multi-scale features
        concat = torch.cat(multi_scale_features, dim=1)  # (B, C*num_scales, H, W)
        concat_perm = concat.permute(0, 2, 3, 1)  # (B, H, W, C*num_scales)
        fused = self.fusion(concat_perm).permute(0, 3, 1, 2)  # (B, C, H, W)

        return x + self.alpha * fused

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
