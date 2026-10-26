"""
Advanced Test-Time Adaptation tools for SOTA performance.

Three key techniques from recent SFDA/TTA literature:

1. Fourier Domain Adaptation (FDA): Low-frequency spectrum swap to bridge
   appearance gap between source and target (Yang & Soatto, CVPR 2020).

2. EMA Mean Teacher: Exponential moving average of student model as teacher
   for generating stable pseudo-labels (CBMT, MICCAI 2023; Tarvainen 2017).

3. Prototype-based Pseudo-label Denoising: Per-class prototype filtering
   to remove noisy pseudo-labels (UP2D, 2025; DPL, MICCAI 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


# ============================================================
# 1. Fourier Domain Adaptation (FDA)
# ============================================================

def fda_source_to_target(source_img: np.ndarray,
                          target_img: np.ndarray,
                          beta: float = 0.01) -> np.ndarray:
    """Replace low-frequency spectrum of source with target's.

    This bridges the style/appearance gap while preserving source content.
    Based on FDA (Yang & Soatto, CVPR 2020).

    Args:
        source_img: (H, W, C) float32 [0,1] source image.
        target_img: (H, W, C) float32 [0,1] target image (style reference).
        beta: Fraction of low-frequency band to replace (0.01-0.09).
            Smaller = subtle style transfer, larger = aggressive.

    Returns:
        styled_img: (H, W, C) float32 [0,1] source content + target style.
    """
    # Resize target to source size if needed
    src_h, src_w = source_img.shape[:2]
    if target_img.shape[:2] != (src_h, src_w):
        from PIL import Image as PILImage
        tgt_pil = PILImage.fromarray((target_img * 255).astype(np.uint8))
        tgt_pil = tgt_pil.resize((src_w, src_h), PILImage.BILINEAR)
        target_img = np.array(tgt_pil).astype(np.float32) / 255.0

    result = np.zeros_like(source_img)

    for c in range(source_img.shape[2]):
        # FFT of both images
        src_fft = np.fft.fft2(source_img[:, :, c])
        tgt_fft = np.fft.fft2(target_img[:, :, c])

        # Shift zero frequency to center
        src_fft_shift = np.fft.fftshift(src_fft)
        tgt_fft_shift = np.fft.fftshift(tgt_fft)

        # Amplitude and phase
        src_amp = np.abs(src_fft_shift)
        src_phase = np.angle(src_fft_shift)
        tgt_amp = np.abs(tgt_fft_shift)

        # Create low-frequency mask
        h, w = src_amp.shape
        cy, cx = h // 2, w // 2
        b = int(min(h, w) * beta)
        if b < 1:
            b = 1

        # Replace source low-freq amplitude with target's
        mixed_amp = src_amp.copy()
        mixed_amp[cy - b:cy + b, cx - b:cx + b] = \
            tgt_amp[cy - b:cy + b, cx - b:cx + b]

        # Reconstruct with mixed amplitude + source phase
        mixed_fft = mixed_amp * np.exp(1j * src_phase)
        mixed_fft = np.fft.ifftshift(mixed_fft)
        result[:, :, c] = np.real(np.fft.ifft2(mixed_fft))

    return np.clip(result, 0, 1).astype(np.float32)


def fda_batch(source_images: List[np.ndarray],
              target_images: List[np.ndarray],
              beta: float = 0.01) -> List[np.ndarray]:
    """Apply FDA to a batch of source images using random target style references.

    Args:
        source_images: List of source images.
        target_images: List of target images (style pool).
        beta: Low-frequency replacement fraction.

    Returns:
        List of style-transferred source images.
    """
    styled = []
    for src in source_images:
        tgt = target_images[np.random.randint(len(target_images))]
        styled.append(fda_source_to_target(src, tgt, beta))
    return styled


# ============================================================
# 2. EMA Mean Teacher
# ============================================================

class MeanTeacher:
    """Exponential Moving Average (EMA) teacher for stable pseudo-labels.

    The teacher is a slowly-updated copy of the student model.
    Its predictions are more stable and less noisy than the student's,
    making them better pseudo-label sources.

    Based on Tarvainen & Valpola (NeurIPS 2017) and CBMT (MICCAI 2023).
    """

    def __init__(self, model: nn.Module, alpha: float = 0.999):
        """
        Args:
            model: Student model to track.
            alpha: EMA decay rate. Higher = slower teacher updates.
                0.999 is standard for most DA tasks.
        """
        self.teacher = deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.alpha = alpha

    @torch.no_grad()
    def update(self, student: nn.Module):
        """Update teacher parameters with EMA of student.

        Call this after each student gradient step.
        """
        for t_param, s_param in zip(self.teacher.parameters(),
                                     student.parameters()):
            t_param.data = self.alpha * t_param.data + \
                           (1 - self.alpha) * s_param.data

        # Also update buffer statistics (BN running mean/var)
        for t_buf, s_buf in zip(self.teacher.buffers(), student.buffers()):
            t_buf.data = self.alpha * t_buf.data + \
                         (1 - self.alpha) * s_buf.data

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get teacher predictions.

        Returns:
            logits: (B, C, H, W)
            pseudo_labels: (B, H, W) argmax labels
        """
        self.teacher.eval()
        logits = self.teacher(x)
        pseudo_labels = logits.argmax(dim=1)
        return logits, pseudo_labels

    @torch.no_grad()
    def get_reliable_pseudolabels(self, x: torch.Tensor,
                                   threshold: float = 0.9,
                                   class_balanced: bool = True,
                                   num_classes: int = 3
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate denoised pseudo-labels from teacher.

        Args:
            x: Input images (B, C, H, W).
            threshold: Global confidence threshold.
            class_balanced: Use per-class adaptive thresholds (CBMT-style).
            num_classes: Number of classes.

        Returns:
            pseudo_labels: (B, H, W) pseudo-labels (-1 for unreliable).
            reliable_mask: (B, H, W) bool mask of reliable pixels.
        """
        self.teacher.eval()
        logits = self.teacher(x)
        probs = F.softmax(logits, dim=1)
        max_probs, pred_labels = probs.max(dim=1)

        if class_balanced:
            # Per-class adaptive threshold (CBMT: use class-specific quantiles)
            reliable_mask = torch.zeros_like(pred_labels, dtype=torch.bool)
            for c in range(num_classes):
                class_mask = pred_labels == c
                if class_mask.sum() == 0:
                    continue
                class_probs = max_probs[class_mask]
                # Use the top p-th percentile as threshold (at least `threshold`)
                p = min(0.5, 1.0)  # take top 50% of each class
                class_thresh = max(
                    threshold * 0.8,  # minimum threshold
                    float(torch.quantile(class_probs, 1 - p))
                )
                reliable_mask |= (class_mask & (max_probs >= class_thresh))
        else:
            reliable_mask = max_probs >= threshold

        pseudo_labels = pred_labels.clone()
        pseudo_labels[~reliable_mask] = -1  # mark unreliable

        return pseudo_labels, reliable_mask


class MeanTeacherLoss(nn.Module):
    """Loss function for mean teacher training.

    Combines:
    - Cross-entropy with teacher pseudo-labels (reliable pixels only)
    - Consistency loss between student and teacher predictions
    """

    def __init__(self, consistency_weight: float = 1.0,
                 pseudo_label_weight: float = 1.0):
        super().__init__()
        self.w_con = consistency_weight
        self.w_pl = pseudo_label_weight

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                pseudo_labels: torch.Tensor,
                reliable_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            student_logits: (B, C, H, W) student predictions.
            teacher_logits: (B, C, H, W) teacher predictions (detached).
            pseudo_labels: (B, H, W) from teacher (-1 = ignore).
            reliable_mask: (B, H, W) bool mask.

        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}

        # 1. Pseudo-label cross-entropy (reliable pixels only)
        if reliable_mask.sum() > 0:
            pl_loss = F.cross_entropy(student_logits, pseudo_labels.clamp(min=0),
                                       reduction="none")
            pl_loss = (pl_loss * reliable_mask.float()).sum() / reliable_mask.sum()
        else:
            pl_loss = torch.tensor(0.0, device=student_logits.device)
        loss_dict["pseudo_label_teacher"] = pl_loss.item()

        # 2. Consistency: soft KL between student and teacher
        student_log_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
        con_loss = F.kl_div(student_log_probs, teacher_probs,
                             reduction="batchmean")
        loss_dict["teacher_consistency"] = con_loss.item()

        total = self.w_pl * pl_loss + self.w_con * con_loss
        loss_dict["total_teacher"] = total.item()
        return total, loss_dict


# ============================================================
# 3. Prototype-based Pseudo-label Denoising
# ============================================================

class PrototypeDenoiser:
    """Denoise pseudo-labels using per-class feature prototypes.

    Maintains running class prototypes from high-confidence predictions.
    Uses prototype similarity to filter out noisy pseudo-labels.

    Based on UP2D (2025) and DPL (MICCAI 2021).
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 momentum: float = 0.99,
                 similarity_threshold: float = 0.5):
        """
        Args:
            feature_dim: Dimension of feature vectors.
            num_classes: Number of segmentation classes.
            momentum: EMA momentum for prototype updates.
            similarity_threshold: Minimum cosine similarity to prototype.
        """
        self.num_classes = num_classes
        self.momentum = momentum
        self.sim_thresh = similarity_threshold
        # Initialize prototypes as None (will be set on first update)
        self.prototypes = None  # (num_classes, feature_dim)
        self.feature_dim = feature_dim
        self.initialized = False

    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor,
                           labels: torch.Tensor,
                           confidence: torch.Tensor = None,
                           conf_threshold: float = 0.9):
        """Update class prototypes from current batch.

        Args:
            features: (B, D, H, W) feature maps from model encoder.
            labels: (B, H, W) pseudo-labels or predictions.
            confidence: (B, H, W) prediction confidence. Optional.
            conf_threshold: Only use pixels above this confidence.
        """
        B, D, H, W = features.shape
        device = features.device

        if self.prototypes is None:
            self.prototypes = torch.zeros(self.num_classes, D, device=device)

        # Mask: only use confident pixels
        if confidence is not None:
            mask = confidence >= conf_threshold
        else:
            mask = torch.ones(B, H, W, dtype=torch.bool, device=device)

        new_prototypes = torch.zeros_like(self.prototypes)
        counts = torch.zeros(self.num_classes, device=device)

        for c in range(self.num_classes):
            class_mask = (labels == c) & mask
            if class_mask.sum() == 0:
                continue

            # Pool features for this class
            class_mask_exp = class_mask.unsqueeze(1).float()  # (B, 1, H, W)
            class_features = (features * class_mask_exp).sum(dim=[0, 2, 3])
            class_features = class_features / class_mask_exp.sum().clamp(min=1)
            new_prototypes[c] = F.normalize(class_features, dim=0)
            counts[c] = class_mask.sum()

        # EMA update
        if self.initialized:
            for c in range(self.num_classes):
                if counts[c] > 0:
                    self.prototypes[c] = (
                        self.momentum * self.prototypes[c] +
                        (1 - self.momentum) * new_prototypes[c]
                    )
                    self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)
        else:
            for c in range(self.num_classes):
                if counts[c] > 0:
                    self.prototypes[c] = new_prototypes[c]
            if counts.sum() > 0:
                self.initialized = True

    @torch.no_grad()
    def denoise(self, features: torch.Tensor,
                pseudo_labels: torch.Tensor,
                confidence: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denoise pseudo-labels by checking prototype consistency.

        Pixels whose features are far from their assigned class prototype
        are marked as unreliable.

        Args:
            features: (B, D, H, W) feature maps.
            pseudo_labels: (B, H, W) candidate pseudo-labels.
            confidence: (B, H, W) model confidence. Optional.

        Returns:
            denoised_labels: (B, H, W) cleaned pseudo-labels (-1 = removed).
            reliable_mask: (B, H, W) bool.
        """
        if not self.initialized:
            # No prototypes yet — fall back to confidence-based filtering
            if confidence is not None:
                reliable = confidence >= 0.9
            else:
                reliable = torch.ones_like(pseudo_labels, dtype=torch.bool)
            return pseudo_labels.clone(), reliable

        B, D, H, W = features.shape
        device = features.device

        # Normalize features
        feat_norm = F.normalize(features, dim=1)  # (B, D, H, W)

        # Compute similarity to assigned class prototype for each pixel
        proto_norm = F.normalize(self.prototypes, dim=1)  # (C, D)

        # Similarity map: (B, C, H, W)
        sim_map = torch.einsum("bdhw,cd->bchw", feat_norm, proto_norm)

        # For each pixel, get similarity to its assigned class
        assigned_sim = torch.zeros(B, H, W, device=device)
        for c in range(self.num_classes):
            mask = pseudo_labels == c
            if mask.any():
                assigned_sim[mask] = sim_map[:, c][mask]

        # Pixels with low similarity to assigned prototype are noisy
        reliable = assigned_sim >= self.sim_thresh

        # Optionally combine with confidence
        if confidence is not None:
            reliable = reliable & (confidence >= 0.7)

        denoised = pseudo_labels.clone()
        denoised[~reliable] = -1

        return denoised, reliable


# ============================================================
# 4. VLM Proxy Denoiser (ProDe-inspired)
# ============================================================

class VLMProxyDenoiser:
    """Denoise pseudo-labels using a Vision-Language Model as external proxy.

    Inspired by ProDe (ICLR 2025 Oral). Core idea: use BiomedCLIP's
    image-text similarity as a "second opinion" on whether a class
    should appear in a given image. If the VLM says class C is absent
    but the segmentation model predicts class C, those pseudo-labels
    are likely noise.

    Adapted for segmentation: operates at image level (class presence)
    and patch level (spatial localization), not pixel level.

    The agent decides when to use VLM denoising based on diagnosis:
    - halluc_ratio 20-50%: VLM can help disambiguate moderate noise
    - halluc_ratio > 70%: even VLM may not help (too much noise)
    - halluc_ratio < 20%: not needed (predictions already clean)
    """

    def __init__(self, class_descriptions: List[str],
                 device: str = "cuda",
                 presence_threshold: float = 0.15,
                 patch_size: int = 32):
        """
        Args:
            class_descriptions: Text description per class (index 0 = background).
                E.g., ["background", "optic disc", "optic cup"]
            device: CUDA device.
            presence_threshold: Minimum VLM similarity to consider class present.
            patch_size: Patch size for spatial localization.
        """
        self.class_descriptions = class_descriptions
        self.device = device
        self.presence_threshold = presence_threshold
        self.patch_size = patch_size
        self.vlm = None

    def load_vlm(self):
        """Lazy-load BiomedCLIP (only when VLM denoising is selected)."""
        if self.vlm is None:
            from medadaptagent.models.biomedclip_wrapper import BiomedCLIPWrapper
            self.vlm = BiomedCLIPWrapper(device=self.device)
            self.vlm.load()
        return self

    @torch.no_grad()
    def compute_class_presence(self, image: np.ndarray) -> Dict[int, float]:
        """Score each class's presence in image using VLM similarity.

        Returns:
            Dict mapping class_id -> similarity score (0 to 1).
        """
        self.load_vlm()
        # Encode image
        img_feat = self.vlm.encode_image(image)  # (1, D)
        # Encode class descriptions (skip background at index 0)
        fg_descs = self.class_descriptions[1:]
        if not fg_descs:
            return {}
        text_feats = self.vlm.encode_text(fg_descs)  # (C-1, D)
        # Cosine similarity
        sims = (img_feat @ text_feats.T).squeeze(0)  # (C-1,)
        scores = {}
        for i, sim in enumerate(sims.cpu().tolist()):
            scores[i + 1] = sim  # class_id starts at 1
        return scores

    @torch.no_grad()
    def denoise(self, images: List[np.ndarray],
                pseudo_labels: torch.Tensor,
                confidence: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denoise pseudo-labels using VLM class presence as proxy.

        For each image, if VLM says class C is absent (low similarity),
        suppress all pseudo-labels for class C in that image.

        Args:
            images: List of (H, W, 3) numpy images (original, not tensor).
            pseudo_labels: (B, H, W) candidate pseudo-labels.
            confidence: (B, H, W) optional confidence map.

        Returns:
            denoised_labels: (B, H, W) cleaned labels (-1 = removed).
            reliable_mask: (B, H, W) bool.
        """
        self.load_vlm()
        B = pseudo_labels.shape[0]
        denoised = pseudo_labels.clone()
        reliable = torch.ones_like(pseudo_labels, dtype=torch.bool)

        for b in range(min(B, len(images))):
            scores = self.compute_class_presence(images[b])
            for class_id, sim in scores.items():
                if sim < self.presence_threshold:
                    # VLM says this class is absent → suppress
                    class_mask = (pseudo_labels[b] == class_id)
                    if class_mask.any():
                        denoised[b][class_mask] = -1
                        reliable[b][class_mask] = False

        # Optionally combine with confidence
        if confidence is not None:
            low_conf = confidence < 0.7
            reliable = reliable & ~low_conf
            denoised[~reliable] = -1

        return denoised, reliable


# ============================================================
# 5. Feature Augmentation Regularization (SF(DA)²-inspired)
# ============================================================

class FeatureAugmentRegularizer(nn.Module):
    """Spectral feature perturbation for domain-invariant learning.

    Inspired by SF(DA)² (ICLR 2024). Core idea: during adaptation,
    perturb intermediate features in frequency space to encourage
    the model to be invariant to domain-specific style information
    encoded in low-frequency components.

    This acts as a regularizer during TTA: the model should produce
    consistent predictions under feature perturbation.

    The agent uses this when:
    - Feature distance between source/target is high (LDDP shows
      large gap in early layers)
    - Halluc ratio is moderate (features are partially useful)
    """

    def __init__(self, perturbation_strength: float = 0.1,
                 low_freq_ratio: float = 0.1):
        """
        Args:
            perturbation_strength: How much to perturb features (0-1).
            low_freq_ratio: Fraction of frequency spectrum to perturb.
        """
        super().__init__()
        self.strength = perturbation_strength
        self.freq_ratio = low_freq_ratio

    def perturb_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply spectral perturbation to feature maps.

        Perturbs the low-frequency amplitude of feature maps,
        which corresponds to domain-specific style information.

        Args:
            features: (B, D, H, W) feature tensor.

        Returns:
            Perturbed features (B, D, H, W).
        """
        B, D, H, W = features.shape

        # FFT along spatial dims
        f_fft = torch.fft.fft2(features, dim=(-2, -1))
        f_amp = torch.abs(f_fft)
        f_phase = torch.angle(f_fft)

        # Create low-frequency mask (center of spectrum)
        cy, cx = H // 2, W // 2
        ry = max(1, int(H * self.freq_ratio))
        rx = max(1, int(W * self.freq_ratio))

        # Random perturbation to low-frequency amplitude
        noise = torch.randn_like(f_amp) * self.strength
        mask = torch.zeros_like(f_amp)
        # fftshift: low freq is at center after shift
        f_amp_shifted = torch.fft.fftshift(f_amp, dim=(-2, -1))
        noise_shifted = torch.fft.fftshift(noise, dim=(-2, -1))
        mask_shifted = torch.zeros_like(mask)
        mask_shifted[:, :, cy - ry:cy + ry, cx - rx:cx + rx] = 1.0

        # Apply perturbation only to low-freq
        f_amp_perturbed = torch.fft.ifftshift(
            f_amp_shifted * (1 + noise_shifted * mask_shifted), dim=(-2, -1))

        # Reconstruct with perturbed amplitude + original phase
        f_perturbed = f_amp_perturbed * torch.exp(1j * f_phase)
        return torch.fft.ifft2(f_perturbed, dim=(-2, -1)).real

    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Compute feature augmentation consistency loss.

        Forward pass with original and perturbed features should
        produce similar predictions.

        Args:
            model: Segmentation model.
            x: (B, C, H, W) input images.

        Returns:
            Consistency loss scalar.
        """
        # Get features from intermediate layer via hook
        features = {}

        def hook_fn(module, input, output):
            features["feat"] = output

        # Hook deepest encoder
        target_module = None
        if hasattr(model, 'enc4'):
            target_module = model.enc4
        elif hasattr(model, 'bottleneck'):
            target_module = model.bottleneck

        if target_module is None:
            return torch.tensor(0.0, device=x.device)

        hook = target_module.register_forward_hook(hook_fn)

        # Original forward
        logits_orig = model(x)
        feat_orig = features.get("feat")

        hook.remove()

        if feat_orig is None:
            return torch.tensor(0.0, device=x.device)

        # Perturb features
        feat_perturbed = self.perturb_features(feat_orig.detach())

        # Re-run forward with perturbed features injected via hook
        def replace_hook_fn(module, input, output):
            return feat_perturbed

        hook2 = target_module.register_forward_hook(replace_hook_fn)
        logits_perturbed = model(x)
        hook2.remove()

        # KL divergence: perturbed predictions should match original
        # Original is the stop-gradient target (teacher signal)
        probs_orig = F.softmax(logits_orig.detach(), dim=1)
        log_probs_perturbed = F.log_softmax(logits_perturbed, dim=1)

        consistency_loss = F.kl_div(
            log_probs_perturbed, probs_orig,
            reduction='batchmean', log_target=False)

        return consistency_loss * self.strength


def extract_features_unet(model: nn.Module, x: torch.Tensor,
                           target_size: Tuple[int, int] = None
                           ) -> torch.Tensor:
    """Extract intermediate features from UNet for prototype computation.

    Uses the bottleneck features (deepest encoder output) upsampled to
    match prediction resolution.

    Args:
        model: UNet model.
        x: (B, C, H, W) input tensor.
        target_size: (H, W) to resize features to. If None, uses input size.

    Returns:
        features: (B, D, H, W) feature tensor.
    """
    # Hook into encoder bottleneck
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    # Register hook on the deepest encoder block
    hooks = []
    if hasattr(model, 'enc4'):
        hooks.append(model.enc4.register_forward_hook(hook_fn('bottleneck')))
    elif hasattr(model, 'encoder'):
        # Generic: hook last encoder layer
        encoder_children = list(model.encoder.children())
        if encoder_children:
            hooks.append(encoder_children[-1].register_forward_hook(
                hook_fn('bottleneck')))

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    if 'bottleneck' in features:
        feat = features['bottleneck']
        if target_size is not None:
            feat = F.interpolate(feat, size=target_size, mode='bilinear',
                                 align_corners=False)
        return feat
    else:
        # Fallback: use logits as features
        with torch.no_grad():
            logits = model(x)
        return logits
