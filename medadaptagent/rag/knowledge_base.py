"""
RAG-based Knowledge Base for domain adaptation guidance.

Retrieves and organizes knowledge from Semantic Scholar and PubMed
about specific domain shifts and adaptation strategies.
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KnowledgeEntry:
    """A single knowledge entry from literature."""
    paper_id: str
    title: str
    year: int
    domain_characteristics: str = ""
    known_gaps: str = ""
    adaptation_strategies: str = ""
    results: str = ""
    source: str = ""  # "semantic_scholar" or "pubmed"


@dataclass
class DomainProfile:
    """Synthesized profile of a target domain."""
    modality: str = ""
    body_region: str = ""
    shift_type: str = ""
    expected_challenges: List[str] = field(default_factory=list)
    recommended_strategies: List[str] = field(default_factory=list)
    anatomy_priors: Dict = field(default_factory=dict)
    class_descriptions: List[str] = field(default_factory=list)
    task_description: str = ""
    source_modality: str = ""
    target_modality: str = ""


class KnowledgeBase:
    """Knowledge base for domain adaptation literature.

    Can be built from Semantic Scholar API, PubMed, or loaded from
    pre-built JSON files (for offline use / reproducibility).
    """

    def __init__(self, save_dir: str = None):
        self.entries: List[KnowledgeEntry] = []
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def add_entry(self, entry: KnowledgeEntry):
        self.entries.append(entry)

    def search(self, query: str, top_k: int = 10) -> List[KnowledgeEntry]:
        """Simple keyword search over entries."""
        scored = []
        query_lower = query.lower()
        query_terms = query_lower.split()

        for entry in self.entries:
            text = f"{entry.title} {entry.domain_characteristics} {entry.known_gaps} {entry.adaptation_strategies}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

    def build_from_semantic_scholar(self, queries: List[str],
                                     limit_per_query: int = 20,
                                     year_min: int = 2023):
        """Build knowledge base from Semantic Scholar API."""
        try:
            import requests
        except ImportError:
            print("requests not installed. Skipping Semantic Scholar retrieval.")
            return

        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        for query in queries:
            params = {
                "query": query,
                "limit": limit_per_query,
                "year": f"{year_min}-",
                "fields": "title,year,abstract,tldr",
            }
            try:
                resp = requests.get(base_url, params=params, timeout=30)
                if resp.status_code != 200:
                    print(f"Semantic Scholar query failed ({resp.status_code}): {query}")
                    continue

                data = resp.json()
                for paper in data.get("data", []):
                    abstract = paper.get("abstract", "") or ""
                    tldr = paper.get("tldr", {}) or {}
                    tldr_text = tldr.get("text", "") if isinstance(tldr, dict) else ""

                    entry = KnowledgeEntry(
                        paper_id=paper.get("paperId", ""),
                        title=paper.get("title", ""),
                        year=paper.get("year", 0),
                        domain_characteristics=abstract[:500],
                        adaptation_strategies=tldr_text,
                        source="semantic_scholar",
                    )
                    self.add_entry(entry)

            except Exception as e:
                print(f"Error querying Semantic Scholar: {e}")

    def save(self, filename: str = "knowledge_base.json"):
        """Save knowledge base to JSON."""
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
        else:
            path = filename

        data = [vars(e) for e in self.entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filename: str = "knowledge_base.json"):
        """Load knowledge base from JSON."""
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
        else:
            path = filename

        if not os.path.exists(path):
            print(f"Knowledge base not found at {path}")
            return

        with open(path) as f:
            data = json.load(f)

        self.entries = [KnowledgeEntry(**d) for d in data]

    def __len__(self):
        return len(self.entries)

    def get_strategies_summary(self) -> str:
        """Summarize all adaptation strategies found in knowledge base."""
        strategies = set()
        for entry in self.entries:
            if entry.adaptation_strategies:
                strategies.add(entry.adaptation_strategies[:200])
        return "\n".join(f"- {s}" for s in list(strategies)[:20])


def build_domain_profile(config: dict) -> DomainProfile:
    """Build a DomainProfile from dataset config YAML."""
    dataset_cfg = config.get("dataset", {})
    adapt_cfg = config.get("adaptation", {})
    prompts = config.get("biomedclip_prompts", [])

    return DomainProfile(
        modality=dataset_cfg.get("target_modality", ""),
        body_region=dataset_cfg.get("body_region", ""),
        shift_type=dataset_cfg.get("shift_type", ""),
        expected_challenges=[],
        recommended_strategies=[],
        class_descriptions=prompts,
        task_description=dataset_cfg.get("description", ""),
        source_modality=dataset_cfg.get("source_modality", ""),
        target_modality=dataset_cfg.get("target_modality", ""),
    )


# =============================================
# Pre-built knowledge bases for the 3 datasets
# =============================================

CARDIAC_CT_MRI_KNOWLEDGE = [
    KnowledgeEntry(
        paper_id="sifa",
        title="Synergistic Image and Feature Adaptation (SIFA)",
        year=2019,
        domain_characteristics="Cardiac CT to MRI cross-modality adaptation",
        known_gaps="Large appearance gap between CT (HU values) and MRI (tissue contrast)",
        adaptation_strategies="Disentangle content/style in shared latent space; adversarial training",
        results="Dice 63-65% on MMWHS CT→MRI",
    ),
    KnowledgeEntry(
        paper_id="ddfp",
        title="Data-Dependent Frequency Prompts for SFDA",
        year=2025,
        domain_characteristics="Frequency-domain prompts for source-free adaptation",
        known_gaps="Source-free setting: no access to source data",
        adaptation_strategies="Learn frequency-domain prompts; adapt amplitude spectrum; entropy min",
        results="Dice 70-78% on cardiac cross-modality",
    ),
    KnowledgeEntry(
        paper_id="vpsfda",
        title="VP-SFDA: Visual Prompts for Source-Free DA",
        year=2024,
        domain_characteristics="Visual prompt tuning for source-free domain adaptation",
        known_gaps="Model fine-tuning without source data access",
        adaptation_strategies="Visual prompt tokens + entropy minimization + consistency",
        results="Dice 83.6% on MMWHS CT→MRI (current SFDA SOTA)",
    ),
]

BRAIN_CET1_HRT2_KNOWLEDGE = [
    KnowledgeEntry(
        paper_id="crossmoda_baseline",
        title="CrossMoDA Challenge: Source-only baseline",
        year=2021,
        domain_characteristics="ceT1 to hrT2 MRI cross-sequence adaptation for VS tumor",
        known_gaps="Contrast enhancement in ceT1 highlights tumor; hrT2 has different tissue contrast",
        adaptation_strategies="nnU-Net trained on ceT1 applied to hrT2",
        results="Source-only Dice ~42%",
    ),
    KnowledgeEntry(
        paper_id="crossmoda_winner",
        title="CrossMoDA Challenge Winner",
        year=2022,
        domain_characteristics="Image translation + segmentation for cross-sequence adaptation",
        known_gaps="Large gap between source-only and supervised performance",
        adaptation_strategies="CycleGAN-based image translation + self-training",
        results="Dice ~85-86% (challenge SOTA)",
    ),
]

FUNDUS_CROSS_DEVICE_KNOWLEDGE = [
    KnowledgeEntry(
        paper_id="fundus_uda",
        title="Cross-Device Fundus Adaptation",
        year=2023,
        domain_characteristics="Fundus images from different cameras/scanners",
        known_gaps="Color distribution shift, field of view differences, resolution changes",
        adaptation_strategies="Style transfer + boundary-aware pseudo-labeling",
        results="Dice ~80-82% for optic disc/cup segmentation",
    ),
]
