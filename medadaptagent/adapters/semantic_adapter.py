"""
Semantic Adapter (~0.2M params).
Operates on SAM2 decoder features.
Text-guided prototype alignment using BiomedCLIP embeddings.
Targets Type A (miss) and Type D (class confusion) failures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAdapter(nn.Module):
    """Text-guided semantic prototype adapter.

    Learns class prototypes that are aligned with BiomedCLIP text embeddings
    for better class discrimination and reduced confusion.
    """

    def __init__(self, vis_dim: int = 256, text_dim: int = 512,
                 proj_dim: int = 128, n_classes: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.proj_dim = proj_dim

        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, proj_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.GELU(),
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, proj_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.text_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, vis_features: torch.Tensor,
                text_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            vis_features: Visual features from SAM2 decoder (B, C, H, W) or (B, N, C).
            text_features: BiomedCLIP text embeddings (n_classes, text_dim). Optional.
        Returns:
            Class similarity logits (B, n_classes, H, W) or (B, N, n_classes).
        """
        spatial = False
        if vis_features.ndim == 4:
            spatial = True
            B, C, H, W = vis_features.shape
            vis_features = vis_features.permute(0, 2, 3, 1).reshape(B * H * W, C)

        vis_proj = self.vis_proj(vis_features)
        vis_proj = F.normalize(vis_proj, dim=-1)

        # Combine learnable prototypes with text guidance
        if text_features is not None:
            text_proj = self.text_proj(text_features)
            text_proj = F.normalize(text_proj, dim=-1)
            w = torch.sigmoid(self.text_weight)
            prototypes = w * F.normalize(self.prototypes, dim=-1) + (1 - w) * text_proj
        else:
            prototypes = F.normalize(self.prototypes, dim=-1)

        # Cosine similarity → logits
        similarity = self.scale * (vis_proj @ prototypes.T)

        if spatial:
            similarity = similarity.reshape(B, H, W, self.n_classes)
            similarity = similarity.permute(0, 3, 1, 2)

        return similarity

    def get_prototype_loss(self, text_features: torch.Tensor) -> torch.Tensor:
        """Alignment loss between learnable prototypes and text embeddings."""
        text_proj = self.text_proj(text_features)
        text_proj = F.normalize(text_proj, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        alignment = (proto_norm * text_proj).sum(dim=-1).mean()
        return 1.0 - alignment  # Minimize to align

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
