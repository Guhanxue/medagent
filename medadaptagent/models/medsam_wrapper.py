"""
MedSAM wrapper for medical image segmentation.
Uses BiomedCLIP-derived bounding box prompts for auto-prompting.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MedSAMWrapper(nn.Module):
    """Wrapper around MedSAM for medical image segmentation.

    MedSAM is SAM fine-tuned on 1.5M medical image-mask pairs.
    Best used with bounding box prompts (its training paradigm).
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load(self):
        """Load MedSAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not installed. "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        self.model = sam_model_registry["vit_b"](checkpoint=self.checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)
        return self

    def set_image(self, image: np.ndarray):
        """Set image for prediction."""
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        self.predictor.set_image(image)

    def predict_box(self, image: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mask from bounding box prompt.

        Args:
            image: (H, W, 3) image.
            box: [x1, y1, x2, y2] bounding box.

        Returns:
            mask: Binary mask (H, W).
            score: Confidence score.
        """
        self.set_image(image)
        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=False,
        )
        return masks[0], scores[0]

    def predict_boxes_batch(self, image: np.ndarray,
                            boxes: np.ndarray) -> list:
        """Predict masks for multiple boxes (one per class).

        Args:
            image: Input image.
            boxes: (N, 4) array of bounding boxes.

        Returns:
            List of (mask, score) tuples.
        """
        self.set_image(image)
        results = []
        for box in boxes:
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
            )
            results.append((masks[0], scores[0]))
        return results

    def predict_with_biomedclip_prompts(self, image: np.ndarray,
                                         biomedclip_boxes: list) -> np.ndarray:
        """Predict using BiomedCLIP-derived bounding boxes.

        Args:
            image: Input image (H, W, 3).
            biomedclip_boxes: List of boxes from BiomedCLIP localization,
                each dict with {'box': [x1,y1,x2,y2], 'class_id': int, 'score': float}.

        Returns:
            label_map: Integer label map (H, W) with class IDs.
        """
        h, w = image.shape[:2]
        label_map = np.zeros((h, w), dtype=np.int64)
        self.set_image(image)

        # Sort by score (highest first) — later predictions overwrite
        sorted_prompts = sorted(biomedclip_boxes, key=lambda x: x["score"])

        for prompt in sorted_prompts:
            box = np.array(prompt["box"])
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
            )
            mask = masks[0]
            class_id = prompt["class_id"]
            label_map[mask > 0] = class_id

        return label_map
