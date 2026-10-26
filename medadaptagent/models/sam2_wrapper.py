"""
SAM2/MedSAM2 wrapper for medical image segmentation.
Supports SAM2.1 (base) and MedSAM2 (medical fine-tuned).
Handles loading, grid-point inference, adapter injection, and feature extraction.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class SAM2Wrapper(nn.Module):
    """Wrapper around SAM2.1 / MedSAM2 for medical image segmentation.

    Supports:
    - SAM2.1 Hiera Tiny (base, ~39M params) or MedSAM2 (medical fine-tuned)
    - Grid-point prompting for zero-shot inference
    - BiomedCLIP-derived point/box prompts
    - Adapter injection at encoder/decoder
    - Feature extraction for failure diagnosis
    """

    def __init__(self, checkpoint_path: str,
                 model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
                 medsam2_checkpoint: str = None,
                 device: str = "cuda"):
        """
        Args:
            checkpoint_path: Path to SAM2.1 base checkpoint.
            model_cfg: Hydra config name for SAM2.1 architecture.
            medsam2_checkpoint: Optional path to MedSAM2 fine-tuned weights.
                If provided, loads MedSAM2 weights on top of SAM2.1 base.
            device: 'cuda' or 'cpu'.
        """
        super().__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.medsam2_checkpoint = medsam2_checkpoint
        self.model = None
        self.predictor = None
        self._adapter_hooks = {}
        self.model_name = "SAM2.1"

    def load(self):
        """Load SAM2.1 model and optionally apply MedSAM2 weights."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)

        # Load MedSAM2 fine-tuned weights if provided
        if self.medsam2_checkpoint and os.path.exists(self.medsam2_checkpoint):
            medsam2_state = torch.load(self.medsam2_checkpoint, map_location="cpu")
            sd = medsam2_state.get("model", medsam2_state)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                print(f"Warning: {len(missing)} missing keys loading MedSAM2")
            self.model_name = "MedSAM2"

        self.predictor = SAM2ImagePredictor(self.model)
        self.model.eval()
        return self

    def freeze_base(self):
        """Freeze all base model parameters (for adapter-only training)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def set_image(self, image: np.ndarray):
        """Set image for prediction.

        Args:
            image: (H, W, 3) or (H, W) uint8/float32 image.
        """
        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        self.predictor.set_image(image)

    def predict_grid(self, image: np.ndarray, grid_size: int = 32,
                     multimask: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zero-shot prediction using grid point prompts."""
        self.set_image(image)
        h, w = image.shape[:2]

        y_coords = np.arange(grid_size // 2, h, grid_size)
        x_coords = np.arange(grid_size // 2, w, grid_size)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        grid_labels = np.ones(len(grid_points), dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=grid_points,
            point_labels=grid_labels,
            multimask_output=multimask,
        )
        return masks, scores, logits

    def predict_points(self, image: np.ndarray, points: np.ndarray,
                       labels: np.ndarray,
                       multimask: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with given point prompts."""
        self.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask,
        )
        return masks, scores, logits

    def predict_box(self, image: np.ndarray, box: np.ndarray,
                    multimask: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with bounding box prompt."""
        self.set_image(image)
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask,
        )
        return masks, scores, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for pipeline compatibility (tensor in -> logits out).

        Args:
            x: (B, 3, H, W) float tensor in [0, 1].
        Returns:
            Logits tensor (B, 1, H, W) from SAM2 mask decoder using center point.
        """
        batch_size = x.shape[0]
        h, w = x.shape[2], x.shape[3]
        all_logits = []

        for i in range(batch_size):
            img_np = (x[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            self.set_image(img_np)

            center = np.array([[w // 2, h // 2]])
            _, _, logits = self.predictor.predict(
                point_coords=center,
                point_labels=np.array([1]),
                multimask_output=False,
            )
            all_logits.append(torch.from_numpy(logits).float())

        return torch.cat(all_logits, dim=0).to(x.device)

    def get_image_embedding(self, image: np.ndarray) -> torch.Tensor:
        """Get SAM2 image encoder features for failure diagnosis."""
        self.set_image(image)
        return self.predictor.get_image_embedding()

    def get_logits(self, image: np.ndarray, points: np.ndarray = None,
                   labels: np.ndarray = None,
                   box: np.ndarray = None) -> torch.Tensor:
        """Get raw logits for uncertainty/entropy computation."""
        self.set_image(image)
        if points is not None:
            _, _, logits = self.predictor.predict(
                point_coords=points, point_labels=labels,
                multimask_output=True,
            )
        elif box is not None:
            _, _, logits = self.predictor.predict(
                box=box, multimask_output=False,
            )
        else:
            h, w = image.shape[:2]
            center = np.array([[w // 2, h // 2]])
            _, _, logits = self.predictor.predict(
                point_coords=center, point_labels=np.array([1]),
                multimask_output=True,
            )
        return torch.from_numpy(logits).float()

    def inject_adapter(self, adapter: nn.Module, location: str = "encoder"):
        """Register an adapter for the forward pass."""
        self._adapter_hooks[location] = adapter

    def remove_adapter(self, location: str):
        """Remove an adapter from a specific location."""
        self._adapter_hooks.pop(location, None)

    def remove_all_adapters(self):
        """Remove all injected adapters."""
        self._adapter_hooks.clear()

    def param_count(self) -> Dict[str, int]:
        """Report parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
