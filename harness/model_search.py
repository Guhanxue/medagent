"""Read REAL model / dataset cards from the HuggingFace Hub so `recommend_alternative`
grounds its suggestions in models that ACTUALLY EXIST and are loadable — instead of the
LLM hallucinating a plausible-but-fake repo id.

NETWORKING NOTE (load-bearing): the CHPC compute node has no direct outbound. The
`huggingface_hub` client uses httpx, which does NOT pick up the HTTPS_PROXY env var set
by our tunnel boilerplate (it tries a direct connect -> "No route to host"). Only
`requests` with an EXPLICIT `proxies=` dict goes through the SSH tunnel. So this module
talks to the HF REST API directly via requests, never via huggingface_hub.
"""
import os
import requests

HF_API = "https://huggingface.co/api"

# task_type -> HF pipeline_tag (the Hub's coarse task taxonomy)
PIPELINE_FOR_TASK = {
    "segmentation": "image-segmentation",
    "seg": "image-segmentation",
    "seg_2d": "image-segmentation",
    "seg_3d": "image-segmentation",
    "classification": "image-classification",
    "cls": "image-classification",
    "cls_2d": "image-classification",
    "cls_3d": "image-classification",
    "detection": "object-detection",
}


def _proxies():
    p = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return {"https": p, "http": p} if p else None


def _get(url, params=None, timeout=30):
    return requests.get(url, params=params, proxies=_proxies(), timeout=timeout)


def search_models(keyword, pipeline_tag=None, n=5):
    """Hit the HF REST /models endpoint. `direction=-1` IS accepted by the raw REST API
    (it was only the python huggingface_hub wrapper that rejected the kwarg)."""
    params = {"search": keyword, "sort": "downloads", "direction": -1,
              "limit": n, "full": "true"}
    if pipeline_tag:
        params["pipeline_tag"] = pipeline_tag
    r = _get(f"{HF_API}/models", params)
    r.raise_for_status()
    return r.json()


def fetch_readme(repo_id, max_chars=3500):
    """Pull the model card (README.md) text — where training data / modality / usage live."""
    for branch in ("main", "master"):
        try:
            r = _get(f"https://huggingface.co/{repo_id}/raw/{branch}/README.md", timeout=20)
            if r.status_code == 200 and r.text.strip():
                return r.text[:max_chars]
        except Exception:
            pass
    return ""


def dataset_card(dataset_id, max_chars=1500):
    """Pull a training-dataset card (modality / size / classes) for one of cardData.datasets."""
    try:
        meta = _get(f"{HF_API}/datasets/{dataset_id}", timeout=20)
        info = meta.json() if meta.status_code == 200 else {}
    except Exception:
        info = {}
    txt = ""
    for branch in ("main", "master"):
        try:
            r = _get(f"https://huggingface.co/datasets/{dataset_id}/raw/{branch}/README.md", timeout=20)
            if r.status_code == 200 and r.text.strip():
                txt = r.text[:max_chars]
                break
        except Exception:
            pass
    cd = (info.get("cardData") or {})
    return {"dataset_id": dataset_id, "tags": (info.get("tags") or [])[:10],
            "task_categories": cd.get("task_categories"), "license": cd.get("license"),
            "card_excerpt": txt}


def read_model_cards(keyword, task_type=None, pipeline_tag=None, n=5,
                     with_readme=True, with_datasets=True):
    """Search the Hub and return REAL candidates with their model + training-dataset cards.

    Returns {"error": str|None, "query": {...}, "candidates": [ {repo_id, pipeline_tag,
    library, downloads, likes, license, datasets, tags, url, card_excerpt,
    dataset_cards:[...]} ]}.  `error` is set (and candidates=[]) on a network failure so the
    caller can fall back to LLM-knowledge with a clear caveat instead of crashing.
    """
    pt = pipeline_tag or (PIPELINE_FOR_TASK.get((task_type or "").lower()) if task_type else None)
    used_pt = pt
    try:
        models = search_models(keyword, pt, n)
        # The medical-imaging ecosystem on HF inconsistently sets pipeline_tag (many seg
        # models are tagged only 'pytorch'/'monai'/'nnunet', or no pipeline at all), so a
        # hard pipeline filter silently drops real candidates. Retry without it on empty.
        if not models and pt:
            models = search_models(keyword, None, n)
            used_pt = None
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:160]}",
                "query": {"keyword": keyword, "pipeline_tag": pt}, "candidates": []}
    out = []
    for m in models:
        cd = m.get("cardData") or {}
        dsets = cd.get("datasets")
        if isinstance(dsets, str):
            dsets = [dsets]
        entry = {
            "repo_id": m.get("id"),
            "pipeline_tag": m.get("pipeline_tag"),
            "library": m.get("library_name"),
            "downloads": m.get("downloads"),
            "likes": m.get("likes"),
            "license": cd.get("license"),
            "datasets": dsets,
            "tags": [t for t in (m.get("tags") or []) if not t.startswith("region:")][:14],
            "url": f"https://huggingface.co/{m.get('id')}",
        }
        if with_readme:
            entry["card_excerpt"] = fetch_readme(m.get("id"))
        if with_datasets and dsets:
            entry["dataset_cards"] = [dataset_card(d) for d in dsets[:2]]
        out.append(entry)
    return {"error": None,
            "query": {"keyword": keyword, "pipeline_tag": pt, "pipeline_used": used_pt},
            "candidates": out}


def read_model_cards_multi(queries, task_type=None, n_per=4, total=6,
                           with_readme=True, with_datasets=False):
    """Pool several keyword variants (the HF `search` param matches against repo ids, so a
    single keyword like 'liver MRI' can return nothing while 'liver'/'abdomen ct'/'monai'
    hit). Run each query, dedup by repo_id, keep the top `total` by downloads.

    Returns {"error": str|None, "queries": [...], "candidates": [...]}.  error is set only
    if EVERY query errored at the network layer (so the caller can fall back cleanly)."""
    # The HF `search` param matches against repo ids, so hyphenated / multi-word queries
    # ('liver-mri', 't2-spir') often return nothing while their plain tokens ('liver','mri')
    # hit. Expand each query into itself + its significant single tokens, dedup, order-stable.
    expanded, _seen_q = [], set()
    for q in queries:
        if not q or not q.strip():
            continue
        for v in [q] + q.lower().replace("-", " ").replace("_", " ").split():
            v = v.strip()
            if len(v) > 2 and v not in _seen_q:
                _seen_q.add(v); expanded.append(v)
    queries = expanded[:8]

    seen, pooled, errors, tried = set(), [], [], []
    for q in queries:
        if not q or not q.strip():
            continue
        tried.append(q)
        res = read_model_cards(q, task_type=task_type, n=n_per,
                               with_readme=with_readme, with_datasets=with_datasets)
        if res["error"]:
            errors.append(res["error"]); continue
        for c in res["candidates"]:
            rid = c.get("repo_id")
            if rid and rid not in seen:
                seen.add(rid); c["matched_query"] = q; pooled.append(c)
    if errors and not pooled and len(errors) == len(tried):
        return {"error": errors[0], "queries": tried, "candidates": []}
    pooled.sort(key=lambda c: (c.get("downloads") or 0), reverse=True)
    return {"error": None, "queries": tried, "candidates": pooled[:total]}


def _parse_json(txt):
    import json, re
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def recommend_alternative_models(llm, *, task_desc, task_type, metric_name,
                                 best_metric, usable_target, domain_shift=None,
                                 model_info=None, n=6):
    """Grounded smart-escalation. When training-free adaptation can't reach a usable
    metric on a fundamental shift (e.g. CT->MRI), suggest REAL replacement models +
    a fine-tune label estimate — NOT hallucinated repo names.

      step 1: LLM -> HF search-keyword variants for the TARGET modality+anatomy.
      step 2: read_model_cards_multi -> REAL repos w/ cards; LLM picks/justifies from them.

    Falls back to LLM-knowledge (clearly flagged `grounding='llm_knowledge_unverified'`)
    if the Hub is unreachable or returns nothing, so escalation never crashes."""
    shift = domain_shift or {}
    sc = shift.get("characterization", shift) if isinstance(shift, dict) else {}
    minfo = (model_info or {}).get("assessment", model_info) if isinstance(model_info, dict) else {}
    bm = "unknown" if best_metric is None else round(float(best_metric), 3)

    if llm is None:
        return {"grounding": "none", "note": "LLM unavailable for recommendation"}

    # ---- step 1: query variants -------------------------------------------------
    q_prompt = (
        f"You are choosing REPLACEMENT pretrained models for a FAILED medical-image "
        f"{task_type} deployment at a new site.\n"
        f"Task: {task_desc}\n"
        f"Measured shift: {sc.get('shift_type')} / {sc.get('severity')}; "
        f"{sc.get('key_differences')}\n"
        f"Source model: {minfo.get('model_family')}, likely trained on "
        f"{minfo.get('likely_training_domain')}\n"
        f"Best achievable {metric_name} after training-free adaptation + a few labels: "
        f"{bm} (usable bar {usable_target}) -> clinically unusable.\n"
        f"Give HuggingFace Hub SEARCH keywords to find models suited to the TARGET "
        f"modality+anatomy (NOT the source modality). The Hub matches against repo IDS, so "
        f"use SINGLE PLAIN WORDS — no hyphens, no underscores, no multi-word phrases. "
        f"GOOD: 'liver','mri','abdomen','fundus','retina','monai','nnunet'. "
        f"BAD: 'liver-mri','t2-spir','abdominal mri'. Respond ONLY JSON:\n"
        f'{{"task_type":"segmentation|classification|detection","queries":'
        f'["word1","word2","word3","word4","word5"],"target_summary":"target modality + anatomy + classes"}}')
    try:
        qj = _parse_json(llm.chat(q_prompt)) or {}
    except Exception as e:
        qj = {"_err": str(e)[:160]}
    queries = [q for q in (qj.get("queries") or []) if isinstance(q, str)][:5]
    tt = qj.get("task_type") or task_type
    target_summary = qj.get("target_summary", "")
    if not queries:                                   # last-ditch keyword from the shift
        queries = [w for w in [minfo.get("likely_training_domain"),
                               sc.get("shift_type"), task_type] if w][:3]

    # ---- step 2: real candidates from the Hub -----------------------------------
    pool = read_model_cards_multi(queries, task_type=tt, n_per=4, total=n,
                                  with_readme=True, with_datasets=False)
    cands = pool.get("candidates") or []
    if pool.get("error") or not cands:
        # Fallback: LLM knowledge, clearly flagged as UNVERIFIED.
        fb_prompt = (
            f"A medical {task_type} model fails at a new site ({task_desc}) and training-free "
            f"adaptation cannot reach usable {metric_name} (best ~{bm}, need >= {usable_target}). "
            f"The HuggingFace Hub could not be reached to verify repos. From your own knowledge, "
            f"respond ONLY JSON: {{\"verdict\":\"why unfixable training-free\","
            f"\"recommended_models\":[{{\"name\":...,\"source\":\"HF/GitHub/MONAI/TotalSegmentator\","
            f"\"why\":...}}],\"finetune_labels_estimate\":<int>,\"finetune_note\":...}}")
        try:
            fb = _parse_json(llm.chat(fb_prompt)) or {"raw": "parse_failed"}
        except Exception as e:
            fb = {"error": str(e)[:160]}
        fb["grounding"] = "llm_knowledge_unverified"
        fb["hf_status"] = pool.get("error") or "no_candidates_returned"
        fb["queries_tried"] = queries
        return fb

    # build a compact catalog of REAL repos for the LLM to choose from
    url_by_id = {}
    lines = []
    for i, c in enumerate(cands, 1):
        url_by_id[c["repo_id"]] = c["url"]
        excerpt = (c.get("card_excerpt") or "").replace("\n", " ")[:800]
        lines.append(
            f"[{i}] repo_id={c['repo_id']} | downloads={c.get('downloads')} | "
            f"license={c.get('license')} | tags={c.get('tags')} | datasets={c.get('datasets')}\n"
            f"    card: {excerpt}")
    catalog = "\n".join(lines)
    pick_prompt = (
        f"A medical {task_type} model failed at a new site and training-free adaptation "
        f"cannot reach usable {metric_name} (best ~{bm}, need >= {usable_target}). Below are "
        f"REAL pretrained models found on the HuggingFace Hub for the target "
        f"({target_summary or task_desc}). Pick the 2-3 BEST suited to the TARGET "
        f"modality+anatomy. Use ONLY repo_ids from this catalog — do NOT invent ids.\n\n"
        f"CATALOG:\n{catalog}\n\n"
        f"Respond ONLY JSON:\n"
        f'{{"verdict":"one sentence: why this shift is unfixable training-free",'
        f'"recommended_models":[{{"repo_id":"<from catalog>","why":"why it fits the target",'
        f'"training_data":"what it was trained on per its card","license":"...",'
        f'"how_to_load":"one line"}}],'
        f'"finetune_labels_estimate":<int labeled target images to fine-tune the EXISTING '
        f'model to ~usable>,"finetune_note":"one sentence"}}')
    try:
        rec = _parse_json(llm.chat(pick_prompt)) or {"raw": "parse_failed"}
    except Exception as e:
        rec = {"error": str(e)[:160]}
    # re-attach canonical URLs from the catalog (defend against hallucinated urls)
    for r in rec.get("recommended_models", []) if isinstance(rec, dict) else []:
        rid = r.get("repo_id")
        if rid in url_by_id:
            r["url"] = url_by_id[rid]
        else:
            r["_warning"] = "repo_id not in verified catalog"
    rec["grounding"] = "huggingface_verified"
    rec["queries_tried"] = pool.get("queries")
    rec["catalog_repo_ids"] = list(url_by_id.keys())
    return rec


if __name__ == "__main__":
    import json
    # Realistic recommend_alternative queries: single/short keyword + pipeline_tag filter.
    probes = [
        ("liver MRI", "segmentation"),
        ("abdomen organ", "segmentation"),
        ("nnunet", "segmentation"),
        ("diabetic retinopathy", "classification"),
        ("brain tumor mri", "segmentation"),
    ]
    for kw, tt in probes:
        print(f"\n{'='*64}\nQUERY keyword={kw!r} task={tt}")
        res = read_model_cards(kw, task_type=tt, n=4, with_readme=False, with_datasets=False)
        if res["error"]:
            print("  ERROR:", res["error"]); continue
        print(f"  pipeline_used={res['query']['pipeline_used']}")
        if not res["candidates"]:
            print("  (no models matched)"); continue
        for c in res["candidates"]:
            print(f"  - {c['repo_id']:<48} pl={c['pipeline_tag']} dl={c['downloads']} "
                  f"lic={c['license']} datasets={c['datasets']} tags={c['tags'][:5]}")
    # one full pull (with cards) to confirm model-card + dataset-card text comes back
    print(f"\n{'='*64}\nFULL CARD PULL (liver MRI / segmentation, n=2)")
    full = read_model_cards("liver MRI", task_type="segmentation", n=2,
                            with_readme=True, with_datasets=True)
    for c in full["candidates"]:
        print(f"\n  REPO {c['repo_id']}  (datasets={c['datasets']})")
        print("   card_excerpt[:240]:", (c.get("card_excerpt") or "")[:240].replace("\n", " "))
        for dc in c.get("dataset_cards", []):
            print(f"   DATASET {dc['dataset_id']} task={dc.get('task_categories')} "
                  f"excerpt:", (dc.get("card_excerpt") or "")[:140].replace("\n", " "))
