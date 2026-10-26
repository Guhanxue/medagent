"""
Style Adapter (~0.1M params).
Injected into SAM2 encoder stage 1.
Modifies frequency-domain representation to handle appearance/contrast gaps.
Targets Type B failures (hallucination from style mismatch).
"""

import torch
import torch.nn as nn


class StyleAdapter(nn.Module):
    """Frequency-domain style adapter for domain appearance shift.

    Separates content (phase) from style (amplitude) in Fourier space,
    then learns a lightweight transform on amplitude to bridge domain gaps.
    """

    def __init__(self, dim: int = 256, bottleneck_ratio: int = 4):
        super().__init__()
        bottleneck = dim // bottleneck_ratio
        self.freq_transform = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )
        self.alpha = nn.Parameter(torch.zeros(1))  # Residual scaling, init 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W).
        Returns:
            Adapted feature map (same shape).
        """
        B, C, H, W = x.shape

        # Convert to frequency domain
        x_freq = torch.fft.rfft2(x, norm="ortho")
        amplitude = x_freq.abs()
        phase = x_freq.angle()

        # Adapt amplitude (style) through bottleneck
        # Reshape: (B, C, H, W//2+1) → (B, H, W//2+1, C) for Linear
        amp_perm = amplitude.permute(0, 2, 3, 1)
        amp_adapted = amp_perm + self.alpha * self.freq_transform(amp_perm)
        amp_adapted = amp_adapted.permute(0, 3, 1, 2)

        # Reconstruct with adapted amplitude, original phase (preserves content)
        x_freq_adapted = torch.polar(amp_adapted, phase)
        x_adapted = torch.fft.irfft2(x_freq_adapted, s=(H, W), norm="ortho")

        return x_adapted

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
