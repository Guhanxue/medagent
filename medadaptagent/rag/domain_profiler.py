"""
Domain Profiler: Uses BiomedCLIP to characterize the target domain.
Performs visual clustering and generates a domain profile.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans


class DomainProfiler:
    """Profiles target domain using BiomedCLIP visual features.

    Creates a characterization of the target domain including:
    - Visual diversity (number of distinct visual patterns)
    - Feature statistics (mean, variance of embeddings)
    - Domain distance from known domains
    """

    def __init__(self, biomedclip=None, n_clusters: int = 5):
        """
        Args:
            biomedclip: BiomedCLIPWrapper instance.
            n_clusters: Number of visual clusters.
        """
        self.biomedclip = biomedclip
        self.n_clusters = n_clusters
        self.features = None
        self.cluster_labels = None
        self.cluster_centers = None

    def profile(self, images: List[np.ndarray]) -> Dict:
        """Generate domain profile from target images.

        Args:
            images: List of target domain images.

        Returns:
            Domain profile dict.
        """
        # Encode all images
        features = self.biomedclip.encode_images_batch(images)
        self.features = features.numpy()

        # Cluster visual features
        n_clusters = min(self.n_clusters, len(images))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.features)
        self.cluster_centers = kmeans.cluster_centers_

        # Compute statistics
        profile = {
            "n_images": len(images),
            "n_clusters": n_clusters,
            "cluster_sizes": [int((self.cluster_labels == i).sum()) for i in range(n_clusters)],
            "feature_mean": self.features.mean(axis=0).tolist(),
            "feature_std": self.features.std(axis=0).mean().item(),
            "intra_cluster_variance": float(self._intra_cluster_variance()),
            "inter_cluster_distance": float(self._inter_cluster_distance()),
        }

        # Get representative images per cluster
        profile["representative_indices"] = self._get_representatives()

        return profile

    def _intra_cluster_variance(self) -> float:
        """Average within-cluster variance."""
        variances = []
        for i in range(self.n_clusters):
            mask = self.cluster_labels == i
            if mask.sum() > 1:
                cluster_features = self.features[mask]
                variances.append(cluster_features.var(axis=0).mean())
        return np.mean(variances) if variances else 0.0

    def _inter_cluster_distance(self) -> float:
        """Average between-cluster distance."""
        distances = []
        for i in range(len(self.cluster_centers)):
            for j in range(i + 1, len(self.cluster_centers)):
                dist = np.linalg.norm(self.cluster_centers[i] - self.cluster_centers[j])
                distances.append(dist)
        return np.mean(distances) if distances else 0.0

    def _get_representatives(self) -> List[int]:
        """Get index of most representative image per cluster."""
        representatives = []
        for i in range(len(self.cluster_centers)):
            mask = self.cluster_labels == i
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            cluster_features = self.features[indices]
            dists = np.linalg.norm(cluster_features - self.cluster_centers[i], axis=1)
            best = indices[np.argmin(dists)]
            representatives.append(int(best))
        return representatives

    def compute_domain_distance(self, source_features: np.ndarray) -> Dict[str, float]:
        """Compute distance metrics between source and target domains.

        Args:
            source_features: (N_source, D) BiomedCLIP features of source images.

        Returns:
            Distance metrics dict.
        """
        if self.features is None:
            raise ValueError("Must call profile() first.")

        source_mean = source_features.mean(axis=0)
        target_mean = self.features.mean(axis=0)

        # L2 distance between means
        l2_dist = float(np.linalg.norm(source_mean - target_mean))

        # Cosine distance between means
        cos_sim = float(
            np.dot(source_mean, target_mean) /
            (np.linalg.norm(source_mean) * np.linalg.norm(target_mean))
        )

        # Maximum Mean Discrepancy (simplified: difference of means)
        mmd = l2_dist  # Simplified version

        return {
            "l2_distance": l2_dist,
            "cosine_similarity": cos_sim,
            "cosine_distance": 1.0 - cos_sim,
            "mmd": mmd,
        }
