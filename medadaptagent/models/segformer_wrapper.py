"""
SegFormer wrapper for the MedAdaptAgent 2D-segmentation harness.

SegFormer (Xie et al., NeurIPS 2021) is the FIRST 2D *transformer* SEGMENTER
in this harness — every other 2D-seg source model is a CNN (UNet/BN,
STU-Net/IN, nnU-Net/IN). It pairs a hierarchical Mixformer (MiT) transformer
encoder (LayerNorm throughout) with a lightweight all-MLP decode head whose
only normalization is a single BatchNorm2d.

Why a wrapper is needed
-----------------------
The harness's 2D-seg contract (scripts/run_baseline_tta._to_tensor +
SegTask / _avg_dice) calls ``model(t)`` with ``t`` of shape (B, 3, H, W) and
expects per-pixel logits of shape ``(B, num_classes, H, W)`` so that
``argmax(dim=1)`` yields a full-resolution mask. HuggingFace's
``SegformerForSemanticSegmentation(pixel_values).logits`` returns logits at
H/4 x W/4 (e.g. 64x64 for a 256x256 input). This wrapper bilinearly upsamples
those logits back to the input H x W inside ``forward`` so SegFormer drops
into every existing seg tool (bn_reset / tent / sar / eata / fine-tune /
VLM scoring / Dice) with NO change to the harness, exactly like the UNet path.

Norm composition (mit-b0, measured)
-----------------------------------
    30x nn.LayerNorm   (MiT encoder: patch-embed + per-block before/after +
                        attention sequence-reduction + per-stage final LN)
     1x nn.BatchNorm2d (decode_head.batch_norm)
So bn_reset (running-stat recalibration) touches ONLY the single decode-head
BN -> it is near-structural-no-op compared to a UNet's ~18 BN layers. Tent /
SAR / EATA update norm-affine params, which here are dominated by the 62
LayerNorm weight+bias tensors; they ARE live but operate almost entirely on
LayerNorm, not BatchNorm. This is reported honestly by the baselines via the
n_affine_params / frac_changed diagnostics, and is the whole point of adding a
transformer (LN) segmenter next to the CNN (BN/IN) ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormer2D(nn.Module):
    """HuggingFace SegformerForSemanticSegmentation wrapped to return
    full-resolution ``(B, num_classes, H, W)`` logits, matching the UNet
    forward contract used everywhere in the 2D-seg harness.

    Args:
        num_classes: number of output classes (background + foreground).
        backbone:    MiT variant / HF repo id (default "nvidia/mit-b0").
        pretrained:  if True, load the pretrained MiT encoder from the HF hub
                     (decode head is always freshly initialized); if False,
                     build from config with random weights (no network).
        in_channels: input channels (3; SegFormer is RGB-native).
    """

    def __init__(self, num_classes=2, backbone="nvidia/mit-b0",
                 pretrained=True, in_channels=3):
        super().__init__()
        from transformers import (SegformerForSemanticSegmentation,
                                   SegformerConfig)
        self.num_classes = num_classes
        self.backbone = backbone
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                backbone, num_labels=num_classes,
                ignore_mismatched_sizes=True)
        else:
            cfg = SegformerConfig.from_pretrained(backbone) \
                if _hf_config_available(backbone) \
                else SegformerConfig()
            cfg.num_labels = num_classes
            self.net = SegformerForSemanticSegmentation(cfg)
        if in_channels != 3:
            raise ValueError("SegFormer2D expects 3-channel input "
                             "(RGB-native MiT patch embeddings).")

    def forward(self, x):
        # x: (B, 3, H, W). HF returns logits at (B, nc, H/4, W/4).
        logits = self.net(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits


def _hf_config_available(backbone):
    """True iff the backbone config can be loaded from the local HF cache
    (offline). Used only on the random-init path so a missing network never
    crashes the no-pretrain fallback."""
    try:
        from transformers import SegformerConfig
        SegformerConfig.from_pretrained(backbone)
        return True
    except Exception:
        return False
