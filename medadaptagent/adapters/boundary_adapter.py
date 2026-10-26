"""
Boundary Adapter (~0.3M params).
Injected into SAM2 mask decoder.
Adds boundary-aware attention for sharper edge delineation.
Targets Type C failures (boundary errors).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAdapter(nn.Module):
    """Boundary-aware attention adapter for mask decoder.

    Uses a learnable edge detector and cross-attention mechanism
    to refine mask boundaries using image features.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.edge_detector = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, padding=1, groups=1),
        )
        self.boundary_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, mask_features: torch.Tensor,
                image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask_features: Decoder mask features (B, C, H, W).
            image_features: Encoder image features (B, C, H, W).
        Returns:
            Refined mask features (same shape).
        """
        B, C, H, W = mask_features.shape

        # Extract edge features from current mask prediction
        edge_map = self.edge_detector(mask_features.detach())

        # Reshape for attention: (B, C, H, W) → (B, H*W, C)
        edge_tokens = edge_map.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        img_tokens = image_features.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Boundary-focused cross-attention: edge queries attend to image keys
        edge_tokens = self.norm1(edge_tokens)
        img_tokens_normed = self.norm2(img_tokens)
        boundary_features, _ = self.boundary_attn(
            edge_tokens, img_tokens_normed, img_tokens_normed
        )

        # Gated residual connection
        g = self.gate(boundary_features)  # (B, HW, 1)
        refined = mask_features.flatten(2).permute(0, 2, 1) + g * boundary_features

        # Reshape back: (B, HW, C) → (B, C, H, W)
        refined = refined.permute(0, 2, 1).reshape(B, C, H, W)
        return refined

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
