"""
Literature search tools for KnowledgeAgent.

When the local knowledge base has no matching entry, the agent
searches PubMed for relevant domain adaptation papers and uses
the LLM to synthesize findings into structured knowledge.

Flow:
  1. Agent generates search queries from TaskConfig
  2. PubMed API returns relevant paper abstracts
  3. LLM reads abstracts + reasons about imaging physics
  4. Outputs structured knowledge (same format as KB entries)

No API key needed for PubMed (NCBI E-utilities are free).
"""

import json
import os
import time
import urllib.request
import urllib.parse
from typing import Dict, List, Optional


def _opener():
    """Return a urllib opener that honors HTTPS_PROXY explicitly.

    Some environments (e.g., CHPC compute nodes that reach the
    institutional proxy only via an SSH local-forward tunnel)
    require an explicit ProxyHandler; relying on urllib's default
    env-based proxy resolution silently fails with "No route to
    host". Falls through to the default opener if no proxy env
    is set."""
    proxy = (os.environ.get("HTTPS_PROXY") or
              os.environ.get("https_proxy") or
              os.environ.get("HTTP_PROXY") or
              os.environ.get("http_proxy"))
    if not proxy:
        return urllib.request.build_opener()
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    return urllib.request.build_opener(handler)


# ================================================================
# PubMed Search
# ================================================================

def search_pubmed(query: str, max_results: int = 10) -> List[str]:
    """Search PubMed and return list of PMIDs."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    })
    url = f"{base}?{params}"
    try:
        resp = _opener().open(url, timeout=15)
        data = json.loads(resp.read())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"    PubMed search failed: {e}")
        return []


def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """Fetch title + abstract for a list of PMIDs."""
    if not pmids:
        return []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    })
    url = f"{base}?{params}"
    try:
        resp = _opener().open(url, timeout=30)
        xml_text = resp.read().decode("utf-8")
        return _parse_pubmed_xml(xml_text)
    except Exception as e:
        print(f"    PubMed fetch failed: {e}")
        return []


def _parse_pubmed_xml(xml_text: str) -> List[Dict]:
    """Parse PubMed XML to extract titles and abstracts."""
    import xml.etree.ElementTree as ET
    papers = []
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            pmid_el = article.find(".//PMID")
            year_el = article.find(".//PubDate/Year")

            title = title_el.text if title_el is not None and title_el.text else ""
            abstract = abstract_el.text if abstract_el is not None and abstract_el.text else ""
            pmid = pmid_el.text if pmid_el is not None else ""
            year = year_el.text if year_el is not None else ""

            if title:
                papers.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract[:1000],  # truncate long abstracts
                    "year": year,
                })
    except ET.ParseError:
        pass
    return papers


# ================================================================
# Query Generation
# ================================================================

def generate_search_queries(source_domain: str, target_domain: str,
                           task_name: str) -> List[str]:
    """Generate PubMed search queries from task description.

    Produces 3 queries: specific, moderate, broad.
    """
    queries = []

    # Extract key terms
    src_lower = source_domain.lower()
    tgt_lower = target_domain.lower()

    # Detect modalities
    modalities = []
    for mod in ["ct", "mri", "mrt", "fundus", "retinal", "cardiac",
                "brain", "prostate", "liver", "abdominal", "ultrasound"]:
        if mod in src_lower or mod in tgt_lower:
            modalities.append(mod)

    # Detect shift type
    shift_terms = []
    if any(m in src_lower for m in ["ct"]) and any(m in tgt_lower for m in ["mri", "mrt"]):
        shift_terms.extend(["cross-modality", "CT to MRI"])
    elif "fundus" in src_lower or "retinal" in src_lower:
        shift_terms.extend(["cross-device", "multi-site fundus"])
    elif "vendor" in tgt_lower or "scanner" in tgt_lower:
        shift_terms.extend(["multi-vendor", "scanner variation"])
    elif "site" in tgt_lower or "institution" in tgt_lower:
        shift_terms.extend(["multi-site", "cross-center"])

    # Query 1: Specific
    specific = f"domain adaptation segmentation {' '.join(modalities[:2])} {' '.join(shift_terms[:1])}"
    queries.append(specific)

    # Query 2: Method-focused
    method = f"test-time adaptation medical image segmentation {' '.join(modalities[:1])}"
    queries.append(method)

    # Query 3: Broad
    broad = f"source-free domain adaptation medical imaging {' '.join(shift_terms[:1])}"
    queries.append(broad)

    return queries


# ================================================================
# LLM Synthesis
# ================================================================

SYNTHESIS_PROMPT = """You are a medical imaging domain adaptation expert.

## Task
Adapt a segmentation model from:
  Source: {source_domain}
  Target: {target_domain}
  Task: {task_name}

## Relevant Literature (from PubMed)
{literature_text}

## Instructions
Based on these papers AND your own expertise about medical imaging physics:

1. Analyze the imaging physics difference between source and target
2. Identify what adaptation strategies work/fail for this shift type
3. Recommend specific adaptation parameters

Return JSON:
```json
{{
    "shift_types": ["list of shift types with physics explanations"],
    "intensity_mapping_type": "linear | non_linear | fundamentally_different",
    "standardization_gap": "none | mild | moderate | severe",
    "noise_model_change": "same | similar | different",
    "contrast_mechanism_change": "same | similar | different | fundamentally_different",
    "boundary_quality_change": "sharper | same | blurrier",
    "dominant_shift_factor": "the single most important factor",
    "what_will_fail": ["approaches known to fail"],
    "recommended_scope": "bn_only | bn+output | bn+decoder1",
    "recommended_mode": "per_image_tent | batch_multi_loss",
    "recommended_lr_range": "standard_1e3 | conservative_5e4 | very_conservative_3e4",
    "literature_evidence": ["key findings from the papers above"]
}}
```"""


def search_and_synthesize(source_domain: str, target_domain: str,
                          task_name: str, llm_client,
                          max_papers: int = 8) -> Optional[Dict]:
    """Full pipeline: search PubMed → fetch abstracts → LLM synthesizes.

    Returns structured knowledge dict (same format as KB entries),
    or None if search/synthesis fails.
    """
    print("    [Literature Search] Generating queries...")
    queries = generate_search_queries(source_domain, target_domain, task_name)

    # Search and deduplicate
    all_pmids = []
    seen = set()
    for q in queries:
        print(f"    [PubMed] Query: '{q}'")
        pmids = search_pubmed(q, max_results=5)
        for pid in pmids:
            if pid not in seen:
                all_pmids.append(pid)
                seen.add(pid)
        time.sleep(0.5)  # rate limit

    if not all_pmids:
        print("    [Literature Search] No papers found.")
        return None

    print(f"    [PubMed] Found {len(all_pmids)} unique papers, fetching abstracts...")
    papers = fetch_abstracts(all_pmids[:max_papers])
    if not papers:
        print("    [Literature Search] Failed to fetch abstracts.")
        return None

    print(f"    [Literature Search] Got {len(papers)} abstracts:")
    for p in papers[:5]:
        print(f"      - [{p['year']}] {p['title'][:80]}...")

    # Format for LLM
    lit_text = ""
    for i, p in enumerate(papers):
        lit_text += f"\n### Paper {i+1} (PMID: {p['pmid']}, {p['year']})\n"
        lit_text += f"**{p['title']}**\n"
        if p['abstract']:
            lit_text += f"{p['abstract'][:500]}\n"

    # LLM synthesis
    prompt = SYNTHESIS_PROMPT.format(
        source_domain=source_domain,
        target_domain=target_domain,
        task_name=task_name,
        literature_text=lit_text,
    )

    try:
        model_name = getattr(llm_client, '_model_name', 'claude-sonnet-4-20250514')
        print(f"    [LLM] Synthesizing knowledge with {model_name}...")
        response = llm_client.messages.create(
            model=model_name,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            knowledge = json.loads(text[start:end])
            knowledge["_source"] = "literature_search"
            knowledge["_papers"] = [
                {"pmid": p["pmid"], "title": p["title"], "year": p["year"]}
                for p in papers
            ]
            print(f"    [LLM] Synthesized knowledge: "
                  f"{knowledge.get('dominant_shift_factor', 'unknown')}")
            return knowledge
    except Exception as e:
        print(f"    [LLM] Synthesis failed: {e}")

    return None
