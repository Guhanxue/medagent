"""
BiomedCLIP wrapper for image-text encoding and auto-prompting.
Uses microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.
Provides:
- Image encoding for domain profiling
- Text encoding for class descriptions
- Heatmap-based localization → point/box prompts for SAM2/MedSAM
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class BiomedCLIPWrapper:
    """Wrapper for BiomedCLIP image-text model.

    Used for:
    1. Domain profiling (visual clustering of target images)
    2. Auto-prompting (text→heatmap→box/point prompts)
    3. Text-guided alignment in Semantic Adapter
    """

    def __init__(self, model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self):
        """Load BiomedCLIP model and processor."""
        self._use_open_clip = False
        try:
            import open_clip
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                "hf-hub:" + self.model_name
            )
            self.tokenizer = open_clip.get_tokenizer("hf-hub:" + self.model_name)
            self._use_open_clip = True
        except Exception:
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            self.model = AutoModel.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode a single image to feature vector.

        Args:
            image: (H, W, 3) uint8 or float32 image.
        Returns:
            Feature vector (1, D).
        """
        from PIL import Image as PILImage
        if image.dtype == np.float32 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        pil_img = PILImage.fromarray(image)
        if self._use_open_clip:
            img_tensor = self.processor(pil_img).unsqueeze(0).to(self.device)
            features = self.model.encode_image(img_tensor)
        else:
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0]  # CLS token
        features = F.normalize(features, dim=-1)
        return features

    @torch.no_grad()
    def encode_images_batch(self, images: List[np.ndarray],
                            batch_size: int = 16) -> torch.Tensor:
        """Encode a batch of images.

        Returns:
            Feature matrix (N, D).
        """
        from PIL import Image as PILImage
        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pil_batch = []
            for img in batch:
                if img.dtype == np.float32 and img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                pil_batch.append(PILImage.fromarray(img))

            if self._use_open_clip:
                tensors = torch.stack([self.processor(p) for p in pil_batch]).to(self.device)
                features = self.model.encode_image(tensors)
            else:
                inputs = self.processor(images=pil_batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0]
            features = F.normalize(features, dim=-1)
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text descriptions to feature vectors.

        Args:
            texts: List of text descriptions (e.g., class names).
        Returns:
            Feature matrix (N, D).
        """
        if self._use_open_clip:
            tokens = self.tokenizer(texts).to(self.device)
            features = self.model.encode_text(tokens)
        else:
            tokens = self.tokenizer(texts, return_tensors="pt", padding=True,
                                    truncation=True).to(self.device)
            outputs = self.model(**tokens)
            features = outputs.last_hidden_state[:, 0]
        features = F.normalize(features, dim=-1)
        return features

    @torch.no_grad()
    def compute_similarity_heatmap(self, image: np.ndarray,
                                    text: str,
                                    patch_size: int = 32) -> np.ndarray:
        """Compute patch-level similarity heatmap between image patches and text.

        Divides image into patches, encodes each, computes cosine similarity with text.

        Args:
            image: (H, W, 3) image.
            text: Text description.
            patch_size: Size of image patches.

        Returns:
            Heatmap (H_patches, W_patches) of similarity scores.
        """
        text_feat = self.encode_text([text])  # (1, D)
        h, w = image.shape[:2]
        h_patches = h // patch_size
        w_patches = w // patch_size

        heatmap = np.zeros((h_patches, w_patches))
        for i in range(h_patches):
            for j in range(w_patches):
                patch = image[
                    i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size,
                ]
                patch_feat = self.encode_image(patch)
                sim = (patch_feat @ text_feat.T).item()
                heatmap[i, j] = sim

        return heatmap

    def heatmap_to_box(self, heatmap: np.ndarray, patch_size: int = 32,
                       threshold: float = 0.5,
                       image_size: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """Convert similarity heatmap to bounding box prompt.

        Args:
            heatmap: (H_p, W_p) similarity scores.
            patch_size: Patch size used to create heatmap.
            threshold: Relative threshold (fraction of max).
            image_size: Original (H, W) for coordinate scaling.

        Returns:
            Box [x1, y1, x2, y2] in image coordinates, or None.
        """
        if heatmap.max() <= 0:
            return None

        mask = heatmap > (threshold * heatmap.max())
        if not mask.any():
            return None

        rows, cols = np.where(mask)
        y1 = rows.min() * patch_size
        y2 = (rows.max() + 1) * patch_size
        x1 = cols.min() * patch_size
        x2 = (cols.max() + 1) * patch_size

        if image_size is not None:
            y2 = min(y2, image_size[0])
            x2 = min(x2, image_size[1])

        # Add small margin
        margin = patch_size // 2
        y1 = max(0, y1 - margin)
        x1 = max(0, x1 - margin)
        if image_size:
            y2 = min(image_size[0], y2 + margin)
            x2 = min(image_size[1], x2 + margin)

        return np.array([x1, y1, x2, y2])

    def heatmap_to_points(self, heatmap: np.ndarray, patch_size: int = 32,
                          n_points: int = 5,
                          threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Convert similarity heatmap to point prompts.

        Returns top-N points as foreground prompts + low-scoring as background.

        Args:
            heatmap: Similarity heatmap.
            patch_size: Patch size.
            n_points: Number of foreground points.
            threshold: Threshold for foreground.

        Returns:
            points: (M, 2) array of (x, y) coordinates.
            labels: (M,) array of 1=foreground, 0=background.
        """
        flat_idx = np.argsort(heatmap.ravel())[::-1]
        h_p, w_p = heatmap.shape

        fg_points = []
        bg_points = []

        for idx in flat_idx:
            i, j = divmod(idx, w_p)
            y = i * patch_size + patch_size // 2
            x = j * patch_size + patch_size // 2

            if heatmap[i, j] > threshold * heatmap.max() and len(fg_points) < n_points:
                fg_points.append([x, y])
            elif len(bg_points) < 2:
                bg_points.append([x, y])

            if len(fg_points) >= n_points and len(bg_points) >= 2:
                break

        if not fg_points:
            # Fallback: use max point
            max_idx = np.argmax(heatmap)
            i, j = divmod(max_idx, w_p)
            fg_points.append([j * patch_size + patch_size // 2,
                             i * patch_size + patch_size // 2])

        all_points = fg_points + bg_points
        all_labels = [1] * len(fg_points) + [0] * len(bg_points)

        return np.array(all_points), np.array(all_labels)

    def auto_prompt_for_classes(self, image: np.ndarray,
                                 class_descriptions: List[str],
                                 patch_size: int = 32) -> List[Dict]:
        """Generate prompts (boxes) for multiple classes.

        Args:
            image: Input image.
            class_descriptions: Text description per class.
            patch_size: Patch size for heatmap.

        Returns:
            List of dicts with 'box', 'class_id', 'score', 'heatmap'.
        """
        h, w = image.shape[:2]
        prompts = []

        for class_id, desc in enumerate(class_descriptions, start=1):
            heatmap = self.compute_similarity_heatmap(image, desc, patch_size)
            box = self.heatmap_to_box(heatmap, patch_size, image_size=(h, w))
            if box is not None:
                prompts.append({
                    "box": box.tolist(),
                    "class_id": class_id,
                    "score": float(heatmap.max()),
                    "heatmap": heatmap,
                })

        return prompts
