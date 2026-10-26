"""Phase S1 — Source Provenance Agent.

The agent that knows what the source model was trained on, where the
paper lives, what the documented augmentations / loss / failure modes
are. Runs ONCE per sandbox invocation on the login VM (which has
internet); caches the result as a sidecar JSON next to the model
checkpoint so subsequent runs and downstream agents pick it up.

The provenance object is passed into Agent 1's prompt as a
"Source provenance" block, turning A1 from a stats-summarizer of
target images into an informed analyst who can reason about the
target-vs-source shift in mechanism terms (e.g., "source was trained
on contrast-enhanced CT with HU clipped [-21, 189]; target is non-
contrast CT — explains the intensity shift Agent 1 measured").

No API keys required for v1. Uses public endpoints:
- HuggingFace Hub: api-inference.huggingface.co/api/models/{model_id}
- arxiv: export.arxiv.org/api/query
- GitHub raw content: raw.githubusercontent.com/{user}/{repo}/main/README.md
- MONAI bundles: the bundle's own metadata.json

All fetches are best-effort and graceful — if a source is unreachable
or the model is fully proprietary, we return a low-confidence provenance
object that still tells A1 what we DON'T know. Hallucination is
explicitly guarded against via an `evidence` field listing each fact's
source URL/path.

Schema:

    {
      "model_id": "...",
      "training_dataset": "...",
      "modality": "CT abdominal contrast-enhanced",
      "n_training_samples": 61,
      "preprocessing": "spacing 1.5x1.5x2, HU clip [-21, 189]",
      "augmentations": ["random_rot90", "scale", "gaussian_noise"],
      "loss": "DiceCELoss",
      "optimizer": "AdamW lr=1e-4 cosine",
      "documented_failure_modes": ["small spleens", "post-surgery"],
      "source_paper_url": "https://arxiv.org/abs/...",
      "source_repo_url": "https://github.com/...",
      "confidence": 0.85,
      "evidence": [
        {"fact": "training_dataset", "source": "HF model card"},
        ...
      ],
      "raw_sources": {
        "hf_model_card": "...",
        "arxiv_abstract": "...",
        "github_readme": "..."
      }
    }
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


HF_API = "https://huggingface.co/api/models"
ARXIV_API = "https://export.arxiv.org/api/query"
GITHUB_RAW = "https://raw.githubusercontent.com"


def _http_get(url: str, timeout: float = 15.0) -> Optional[str]:
    """Best-effort HTTP GET. Returns body text or None on error."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "MedAdaptAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [provenance] GET {url[:80]} failed: {type(e).__name__}: {e}")
        return None


# ============================================================
# Source 1: HuggingFace Hub
# ============================================================

def fetch_huggingface(model_id: str) -> Dict:
    """Fetch HF Hub model metadata + model card README."""
    out = {"source": "huggingface", "model_id": model_id, "raw": {}}
    meta_url = f"{HF_API}/{model_id}"
    meta = _http_get(meta_url)
    if not meta:
        return out
    try:
        out["raw"]["api_meta"] = json.loads(meta)
    except json.JSONDecodeError:
        return out
    # Fetch the README (model card)
    readme_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    readme = _http_get(readme_url)
    if readme:
        out["raw"]["model_card"] = readme[:20000]  # cap length
    return out


# ============================================================
# Source 2: arxiv
# ============================================================

_LAST_ARXIV_QUERY = [0.0]
ARXIV_MIN_INTERVAL = 3.5  # arxiv API guideline


def fetch_arxiv_by_url(url: str) -> Optional[Dict]:
    """Fetch an arxiv paper's metadata directly by URL.
    Accepts forms: https://arxiv.org/abs/{id}, /pdf/{id}, /abs/{id}vN.
    Returns {id, title, abstract, pdf_url} or None."""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w./-]+?)(?:v\d+)?(?:\.pdf)?$",
                   url.rstrip("/"))
    if not m:
        return None
    arxiv_id = m.group(1)
    elapsed = time.time() - _LAST_ARXIV_QUERY[0]
    if elapsed < ARXIV_MIN_INTERVAL:
        time.sleep(ARXIV_MIN_INTERVAL - elapsed)
    api_url = f"{ARXIV_API}?id_list={arxiv_id}"
    body = _http_get(api_url, timeout=20.0)
    _LAST_ARXIV_QUERY[0] = time.time()
    if not body:
        return None
    def _x(tag):
        mm = re.search(f"<{tag}>(.*?)</{tag}>", body, re.S)
        return re.sub(r"\s+", " ", mm.group(1)).strip() if mm else ""
    title = _x("title")
    summary = _x("summary")
    if not title:
        return None
    return {
        "id": f"https://arxiv.org/abs/{arxiv_id}",
        "title": title,
        "abstract": summary[:3000],
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def search_arxiv(query: str, max_results: int = 3) -> List[Dict]:
    """Search arxiv by title/abstract keywords. Returns list of
    {id, title, abstract, pdf_url} dicts. Throttled to one query per
    3.5s per arxiv's published guideline."""
    elapsed = time.time() - _LAST_ARXIV_QUERY[0]
    if elapsed < ARXIV_MIN_INTERVAL:
        time.sleep(ARXIV_MIN_INTERVAL - elapsed)
    q = urllib.parse.quote(query)
    url = (f"{ARXIV_API}?search_query=all:{q}"
           f"&start=0&max_results={max_results}")
    body = _http_get(url, timeout=20.0)
    _LAST_ARXIV_QUERY[0] = time.time()
    if not body:
        return []
    # Light XML parsing — arxiv returns Atom
    entries = []
    # Split entries roughly
    parts = body.split("<entry>")[1:]
    for p in parts:
        end = p.find("</entry>")
        if end < 0:
            continue
        entry_body = p[:end]
        def _extract(tag):
            m = re.search(f"<{tag}>(.*?)</{tag}>", entry_body, re.S)
            return m.group(1).strip() if m else ""
        id_url = _extract("id")
        title = re.sub(r"\s+", " ", _extract("title")).strip()
        summary = re.sub(r"\s+", " ", _extract("summary")).strip()
        if id_url and title:
            entries.append({
                "id": id_url,
                "title": title,
                "abstract": summary[:2000],
                "pdf_url": id_url.replace("/abs/", "/pdf/"),
            })
    return entries


# ============================================================
# Source 3: GitHub
# ============================================================

def fetch_github_readme(repo_url: str) -> Optional[str]:
    """Given a github.com/{user}/{repo} URL, fetch raw README."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/?#]+)", repo_url)
    if not m:
        return None
    user, repo = m.group(1), m.group(2).rstrip(".git")
    for branch in ("main", "master"):
        url = f"{GITHUB_RAW}/{user}/{repo}/{branch}/README.md"
        body = _http_get(url, timeout=10.0)
        if body:
            return body[:30000]
    return None


# ============================================================
# Source 4: MONAI bundle metadata
# ============================================================

def fetch_monai_bundle_meta(bundle_path: str) -> Optional[Dict]:
    """If `bundle_path` is a MONAI bundle directory, parse metadata."""
    if not os.path.isdir(bundle_path):
        # Maybe it's a ckpt path inside a bundle — try parent dirs
        cur = bundle_path
        for _ in range(3):
            cur = os.path.dirname(cur)
            cfg_dir = os.path.join(cur, "configs")
            if os.path.isdir(cfg_dir):
                bundle_path = cur
                break
        else:
            return None
    meta_path = os.path.join(bundle_path, "configs", "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================
# Synthesis: LLM combines sources -> structured provenance
# ============================================================

SYNTHESIS_PROMPT = """You are a research-engineering assistant. Synthesize
the following raw sources into a single structured JSON provenance
record describing what the SOURCE model was trained on, how, and
what its documented limitations are.

CRITICAL RULES:
1. Every non-null field MUST be grounded in one of the raw sources
   below. If a fact isn't in any source, leave the field null.
2. The "evidence" array MUST list one entry per non-null fact,
   citing which source it came from.
3. The "confidence" field is your assessment of overall provenance
   quality: 0.9+ when multiple sources agree, 0.5–0.8 when only one
   source provides a claim, <0.5 when sources conflict or are sparse.
4. DO NOT hallucinate. If you don't see "n_training_samples" anywhere,
   leave it null, do not guess.
5. Documented failure modes should be quoted near-verbatim from the
   source (e.g., "small lesions <5mm" from the model card).

Output schema (return ONLY this JSON object):

{{
  "model_id": "<best identifier>",
  "training_dataset": "<dataset name + size if known, else null>",
  "modality": "<e.g., 'CT abdominal contrast-enhanced'>",
  "n_training_samples": <int or null>,
  "preprocessing": "<spacing/intensity normalization recipe>",
  "augmentations": [<list of aug names>],
  "loss": "<loss function>",
  "optimizer": "<optimizer + lr + schedule>",
  "documented_failure_modes": [<list of strings>],
  "source_paper_url": "<url or null>",
  "source_repo_url": "<url or null>",
  "confidence": <float in [0, 1]>,
  "evidence": [
    {{"fact": "<field_name>", "source": "<which raw source>"}}
  ]
}}

=== RAW SOURCES ===

{sources}

=== END SOURCES ===

Return ONLY the JSON object."""


@dataclass
class ProvenanceResult:
    structured: Dict = field(default_factory=dict)
    raw_sources: Dict = field(default_factory=dict)
    fetched_at: float = 0.0
    n_sources: int = 0


def synthesize_provenance(model_hint: Dict, llm_client,
                            cache_path: Optional[str] = None) -> ProvenanceResult:
    """Top-level: fetch all sources for `model_hint`, synthesize via LLM.

    `model_hint` is a dict with any subset of:
        {"name", "hf_id", "paper_url", "repo_url", "ckpt_path",
         "search_terms"}

    If `cache_path` exists, return cached. Otherwise fetch + synthesize
    + save."""
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            print(f"  [provenance] cache hit: {cache_path}")
            return ProvenanceResult(
                structured=cached.get("structured", {}),
                raw_sources=cached.get("raw_sources", {}),
                fetched_at=cached.get("fetched_at", 0.0),
                n_sources=cached.get("n_sources", 0),
            )
        except Exception:
            pass

    raw: Dict[str, str] = {}

    # 1. HuggingFace
    hf_id = model_hint.get("hf_id")
    if hf_id:
        hf = fetch_huggingface(hf_id)
        if hf["raw"]:
            raw["huggingface"] = json.dumps(hf["raw"], indent=2)[:8000]

    # 2. GitHub README
    repo_url = model_hint.get("repo_url")
    if repo_url:
        readme = fetch_github_readme(repo_url)
        if readme:
            raw["github_readme"] = readme[:8000]

    # 3a. Direct arxiv URL (preferred — no rate-limit risk, exact)
    paper_url = model_hint.get("paper_url")
    arxiv_pulled = False
    if paper_url and "arxiv.org" in paper_url:
        ax = fetch_arxiv_by_url(paper_url)
        if ax:
            raw["arxiv"] = (f"Title: {ax['title']}\nID: {ax['id']}\n"
                              f"Abstract: {ax['abstract']}")[:10000]
            arxiv_pulled = True

    # 3b. arxiv search (fallback only if no URL hint or URL fetch failed)
    if not arxiv_pulled:
        search_terms = (model_hint.get("search_terms")
                         or model_hint.get("name"))
        if search_terms:
            hits = search_arxiv(search_terms, max_results=3)
            if hits:
                raw["arxiv"] = "\n---\n".join(
                    f"Title: {h['title']}\nID: {h['id']}\n"
                    f"Abstract: {h['abstract']}"
                    for h in hits[:3])[:10000]

    # 4. MONAI bundle metadata
    ckpt = model_hint.get("ckpt_path")
    if ckpt:
        bundle_meta = fetch_monai_bundle_meta(ckpt)
        if bundle_meta:
            raw["monai_bundle"] = json.dumps(
                bundle_meta, indent=2)[:8000]

    n_sources = len(raw)
    if n_sources == 0:
        # Nothing to synthesize — return a stub
        structured = {
            "model_id": model_hint.get("name", "unknown"),
            "training_dataset": None,
            "modality": None,
            "n_training_samples": None,
            "preprocessing": None,
            "augmentations": [],
            "loss": None,
            "optimizer": None,
            "documented_failure_modes": [],
            "source_paper_url": None,
            "source_repo_url": None,
            "confidence": 0.0,
            "evidence": [],
        }
        result = ProvenanceResult(structured=structured, raw_sources={},
                                       fetched_at=time.time(),
                                       n_sources=0)
        if cache_path:
            _save_cache(result, cache_path)
        return result

    print(f"  [provenance] fetched {n_sources} source(s): "
           f"{sorted(raw.keys())}")

    # 5. LLM synthesis
    sources_str = "\n\n".join(
        f"--- {name} ---\n{body}" for name, body in raw.items())
    prompt = SYNTHESIS_PROMPT.format(sources=sources_str)

    if llm_client is None:
        # No LLM — return raw sources with minimal structure
        structured = {
            "model_id": model_hint.get("name", "unknown"),
            "confidence": 0.3,
            "evidence": [{"fact": "raw_only",
                           "source": s} for s in raw.keys()],
        }
    else:
        text = llm_client.chat(prompt, max_tokens=1500) or ""
        structured = _parse_json_object(text)
        if not structured:
            structured = {
                "model_id": model_hint.get("name", "unknown"),
                "confidence": 0.2,
                "evidence": [],
            }

    result = ProvenanceResult(
        structured=structured, raw_sources=raw,
        fetched_at=time.time(), n_sources=n_sources)
    if cache_path:
        _save_cache(result, cache_path)
    return result


def _parse_json_object(text: str) -> Dict:
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
        try:
            return json.loads(cleaned)
        except Exception:
            return {}


def _save_cache(result: ProvenanceResult, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "w") as f:
            json.dump({
                "structured": result.structured,
                "raw_sources": result.raw_sources,
                "fetched_at": result.fetched_at,
                "n_sources": result.n_sources,
            }, f, indent=2)
        print(f"  [provenance] cached -> {path}")
    except Exception as e:
        print(f"  [provenance] cache save failed: {e}")


# ============================================================
# Prompt block: turn provenance into Agent 1/2 context
# ============================================================

def render_provenance_block(provenance: Dict) -> str:
    """Render provenance.structured as a prompt block for A1/A2."""
    if not provenance or provenance.get("confidence", 0) < 0.1:
        return ""
    lines = ["## Source provenance (web-retrieved, Phase S1)"]
    lines.append("")
    lines.append("The source model was trained as follows. Use this to "
                  "reason about target-vs-source shifts in mechanistic "
                  "terms (e.g., source HU clip vs target HU range).")
    lines.append("")
    keys = ["training_dataset", "modality", "n_training_samples",
             "preprocessing", "augmentations", "loss", "optimizer",
             "documented_failure_modes"]
    for k in keys:
        v = provenance.get(k)
        if v in (None, [], ""):
            continue
        if isinstance(v, list):
            lines.append(f"  - {k}: {', '.join(str(x) for x in v)}")
        else:
            lines.append(f"  - {k}: {v}")
    conf = provenance.get("confidence", 0.0)
    n_ev = len(provenance.get("evidence", []) or [])
    lines.append("")
    lines.append(f"  (provenance confidence: {conf:.2f}, "
                  f"grounded in {n_ev} cited fact(s))")
    return "\n".join(lines)
