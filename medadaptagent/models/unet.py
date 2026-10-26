"""
Lightweight UNet for fundus segmentation (optic disc + cup).
Serves as the "source model" trained on REFUGE, then adapted to RIM-ONE.

This is deliberately simple — the novelty is in MedAdaptAgent's
diagnosis + adaptation, not in the base architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Standard UNet with 4 encoder/decoder stages.

    ~7.8M params with base_ch=64. Small enough to train quickly,
    large enough to segment fundus disc/cup well.

    Args:
        in_channels: Input channels (3 for RGB).
        num_classes: Output classes (3: background, disc, cup).
        base_ch: Base channel count (doubled at each stage).
    """

    def __init__(self, in_channels=3, num_classes=3, base_ch=64):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # Output
        self.out_conv = nn.Conv2d(base_ch, num_classes, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

    def get_encoder_features(self, x):
        """Extract multi-scale encoder features (for failure diagnosis)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "bottleneck": b}

    def predict(self, x):
        """Predict with softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict_with_uncertainty(self, x, n_augments=8):
        """Predict with TTA-based uncertainty estimation.

        Returns:
            pred: (B, C, H, W) mean softmax probabilities
            entropy: (B, 1, H, W) pixel-wise entropy
            variance: (B, C, H, W) prediction variance across augmentations
        """
        self.eval()
        preds = []

        with torch.no_grad():
            # Original
            preds.append(self.predict(x))

            # Augmented predictions
            for _ in range(n_augments - 1):
                # Random horizontal flip
                aug_x = x.flip(-1) if torch.rand(1) > 0.5 else x
                # Random small rotation via affine
                angle = (torch.rand(1).item() - 0.5) * 20  # +/- 10 degrees
                theta = torch.tensor([
                    [torch.cos(torch.tensor(angle * 3.14159 / 180)),
                     -torch.sin(torch.tensor(angle * 3.14159 / 180)), 0],
                    [torch.sin(torch.tensor(angle * 3.14159 / 180)),
                     torch.cos(torch.tensor(angle * 3.14159 / 180)), 0]
                ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)

                grid = F.affine_grid(theta, x.size(), align_corners=False)
                aug_x = F.grid_sample(x, grid, align_corners=False, mode='bilinear')
                aug_pred = self.predict(aug_x)

                # Inverse transform
                theta_inv = theta.clone()
                theta_inv[:, 0, 1] *= -1
                theta_inv[:, 1, 0] *= -1
                grid_inv = F.affine_grid(theta_inv, aug_pred.size(), align_corners=False)
                pred_back = F.grid_sample(aug_pred, grid_inv, align_corners=False,
                                          mode='bilinear')
                preds.append(pred_back)

        stacked = torch.stack(preds, dim=0)  # (N, B, C, H, W)
        mean_pred = stacked.mean(dim=0)
        variance = stacked.var(dim=0)

        # Pixel-wise entropy
        entropy = -(mean_pred * (mean_pred + 1e-8).log()).sum(dim=1, keepdim=True)

        return mean_pred, entropy, variance

    def freeze(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
