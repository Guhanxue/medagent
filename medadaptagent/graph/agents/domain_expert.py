"""
Agent 1: Domain Expert

Understands WHAT the domain shift is by combining:
  - Visual inspection of source/target images (multimodal LLM)
  - Knowledge base lookup for known shift types
  - PubMed literature search for novel shifts
  - Auto-filling TaskConfig fields from observations

The expert thinks like a radiologist: LOOK at the images first,
then confirm with knowledge and literature.
"""

import io
import base64
import json
import numpy as np
from typing import Dict, List, Optional

from medadaptagent.graph.vision_utils import (
    prepare_sample_images, build_multimodal_message,
)


# ---------------------------------------------------------------------------
# Per-channel quantitative + montage helpers (data-agnostic)
#
# Motivation: when multiple input channels are stacked into a single RGB
# tile, the LLM rescales the composite and stops seeing per-channel
# intensity drift. By (a) showing one grayscale montage PER channel and
# (b) injecting raw per-channel statistics directly in the prompt, the LLM
# can catch shifts that would otherwise be silently averaged away.
#
# These helpers make zero assumptions about modality, anatomy, or task.
# They operate on arbitrary numpy arrays of shape (H, W), (H, W, 1), or
# (H, W, C) for any C. Channel semantics (if any) are supplied by the
# caller via the `channel_names` argument to run_domain_expert.
# ---------------------------------------------------------------------------

def _to_hwc(img: np.ndarray) -> np.ndarray:
    """Coerce array to (H, W, C) with C >= 1."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1]:
        return np.transpose(arr, (1, 2, 0))
    return arr


def _channels_identical(images: List[np.ndarray], n_check: int = 5,
                        atol: float = 1e-6) -> bool:
    """Detect if all channels in the input images are identical.

    Common pattern: grayscale medical data (CT, MRI) replicated to 3 channels
    so it can be fed to RGB-input models. In this case the per-channel
    statistics are exactly the same, and presenting them three times to the
    LLM is just noise.

    Returns True if every checked image has all channels equal (within atol).
    Single-channel inputs return True trivially.
    """
    if not images:
        return False
    for img in images[:n_check]:
        a = _to_hwc(img)
        if a.shape[-1] <= 1:
            continue
        ch0 = a[..., 0].astype(np.float64)
        for c in range(1, a.shape[-1]):
            if not np.allclose(ch0, a[..., c].astype(np.float64), atol=atol):
                return False
    return True


def _compute_channel_stats(images: List[np.ndarray],
                           bg_threshold: float = 0.05) -> List[Dict]:
    """Per-channel intensity statistics across a list of images.

    Computes three views of each channel to disentangle different shift types:

    1. **raw**: stats over all pixels (the global view).
    2. **non_bg**: stats over pixels > `bg_threshold`. NOTE: this is NOT the
       segmentation foreground — we have no target mask. It is a HEURISTIC
       "non-air / non-padding pixel" view, useful for separating tissue
       intensity drift from background coverage shifts.
    3. **bg_fraction**: fraction of pixels at/below `bg_threshold`. A change
       in this signals a cropping / FOV / coverage shift, NOT an intensity
       shift, even though it would dominate the global mean and median.

    The non_bg view is only meaningful when the threshold actually splits
    the histogram (between ~5% and 95% of pixels on each side). Outside
    that range, the threshold is unreliable and the non_bg dict is omitted.

    Returns a list indexed by channel. Each entry has:
        raw:           {mean, std, p1, p50, p99}
        non_bg:        {mean, std, p50, p99}  (only if threshold is meaningful)
        non_bg_valid:  bool — whether non_bg view is reliable
        bg_fraction:   float in [0, 1]
        n_voxels:      int
    Plus top-level keys that mirror raw (back-compat).
    Channels with no data return {}.
    """
    if not images:
        return []
    samples = [_to_hwc(im) for im in images]
    n_channels = max(s.shape[-1] for s in samples)
    out: List[Dict] = []
    for c in range(n_channels):
        chunks = []
        for s in samples:
            if c < s.shape[-1]:
                v = s[..., c].astype(np.float64).ravel()
                v = v[np.isfinite(v)]
                if v.size:
                    chunks.append(v)
        if not chunks:
            out.append({})
            continue
        flat = np.concatenate(chunks)
        bg_mask = flat <= bg_threshold
        bg_fraction = float(bg_mask.mean())
        fg = flat[~bg_mask]

        raw = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "p1": float(np.percentile(flat, 1)),
            "p50": float(np.percentile(flat, 50)),
            "p99": float(np.percentile(flat, 99)),
        }

        # Only emit non_bg view if the threshold actually splits the data.
        # If <5% or >95% of pixels are background, the threshold is not
        # meaningful and we'd be reporting noise.
        non_bg_valid = (0.05 <= bg_fraction <= 0.95) and (fg.size > 0)
        if non_bg_valid:
            non_bg = {
                "mean": float(np.mean(fg)),
                "std": float(np.std(fg)),
                "p50": float(np.percentile(fg, 50)),
                "p99": float(np.percentile(fg, 99)),
            }
        else:
            non_bg = {}

        out.append({
            "raw": raw,
            "non_bg": non_bg,
            "non_bg_valid": non_bg_valid,
            "bg_fraction": bg_fraction,
            "n_voxels": int(flat.size),
            # Back-compat: top-level keys mirror the raw view
            "mean": raw["mean"],
            "std": raw["std"],
            "p1": raw["p1"],
            "p50": raw["p50"],
            "p99": raw["p99"],
        })
    return out


def _format_channel_stats_table(src_stats: List[Dict],
                                tgt_stats: List[Dict],
                                channel_names: Optional[List[str]] = None,
                                src_label: str = "SOURCE",
                                tgt_label: str = "TARGET") -> str:
    """Format per-channel stats as an LLM-friendly comparison table.

    Three views per channel, in order of interpretive importance:
      [FOV]    — background pixel fraction shift (cropping / coverage)
      [NON_BG] — heuristic non-air pixel stats (true intensity drift, when meaningful)
      [RAW]    — global stats (legacy view, confounded by both)
    """
    if not src_stats or not tgt_stats:
        return ""
    n = max(len(src_stats), len(tgt_stats))
    if channel_names is None or len(channel_names) < n:
        channel_names = [f"channel_{i}" for i in range(n)]
    lines = [
        f"Per-channel intensity statistics ({src_label} vs {tgt_label}, "
        f"computed across all provided images).",
        "",
        "READING GUIDE — three views are reported per channel:",
        "",
        "  [FOV] Background pixel fraction (pixels at/below 0.05 intensity).",
        "        Read this FIRST. A large change here (>5 percentage points)",
        "        means cropping / padding / field-of-view differs between",
        "        source and target. When FOV changed, the [RAW] mean and",
        "        median can drift dramatically WITHOUT any change in actual",
        "        tissue intensity. Do NOT call this a 'calibration shift'",
        "        unless [NON_BG] also drifts in the same direction.",
        "",
        "  [NON_BG] Stats over pixels brighter than 0.05. THIS IS NOT THE",
        "        SEGMENTATION FOREGROUND — we have no target mask. It is a",
        "        heuristic 'non-air pixel' filter. It is meaningful only",
        "        when 5-95% of pixels are above the threshold (otherwise",
        "        the threshold is unreliable and this view is omitted).",
        "        When valid, [NON_BG] is the BEST proxy for true tissue",
        "        intensity drift, because it is robust to FOV changes.",
        "",
        "  [RAW] Global stats over ALL pixels including background. This",
        "        is the legacy view. Drifts here can come from EITHER tissue",
        "        intensity changes OR background coverage changes. Use it",
        "        only after consulting [FOV] and [NON_BG] to attribute the",
        "        cause.",
        "",
        "EXPERT REASONING RULES — derive the dominant shift type yourself:",
        "  • [FOV] big AND [NON_BG] small  → cropping / FOV shift, NOT",
        "                                     calibration. Histogram matching",
        "                                     and BN reset will not help —",
        "                                     consider re-cropping or padding.",
        "  • [FOV] small AND [NON_BG] big   → real tissue intensity drift",
        "                                     (calibration / vendor / coil).",
        "                                     BN reset and histogram matching",
        "                                     are appropriate.",
        "  • [FOV] big AND [NON_BG] big     → mixed shift. Identify whether",
        "                                     intensity drift survives after",
        "                                     correcting for FOV.",
        "  • [FOV] small AND [NON_BG] small → genuinely mild shift. Source",
        "                                     model may already work well.",
        "  • [NON_BG] omitted               → threshold is degenerate (almost",
        "                                     all background or almost no",
        "                                     background). Reason from [RAW]",
        "                                     and [FOV] together, with caution.",
        "",
    ]

    def _row(metric: str, sv, tv, src_lbl, tgt_lbl):
        if sv is None or tv is None:
            return None
        if abs(sv) > 1e-6:
            rel = (tv - sv) / abs(sv) * 100.0
            rel_s = f"({rel:+.1f}%)"
        else:
            rel_s = ""
        return (f"    {metric:6s}  {src_lbl}={sv:>12.4f}  "
                f"{tgt_lbl}={tv:>12.4f}  {rel_s}")

    for i in range(n):
        s = src_stats[i] if i < len(src_stats) else {}
        t = tgt_stats[i] if i < len(tgt_stats) else {}
        if not s or not t:
            continue
        name = channel_names[i] if i < len(channel_names) else f"channel_{i}"
        lines.append(f"### {name} ###")

        # FOV — headline signal
        s_bg = s.get("bg_fraction")
        t_bg = t.get("bg_fraction")
        if s_bg is not None and t_bg is not None:
            delta = (t_bg - s_bg) * 100.0
            tag = ""
            abs_delta = abs(delta)
            if abs_delta > 15.0:
                tag = "  ← LARGE FOV SHIFT (cropping/padding likely changed)"
            elif abs_delta > 5.0:
                tag = "  ← moderate FOV shift"
            lines.append(
                f"  [FOV] background fraction: "
                f"{src_label}={s_bg:>5.1%}  {tgt_label}={t_bg:>5.1%}  "
                f"(delta={delta:+.1f}pp){tag}"
            )

        # NON_BG — heuristic tissue drift, only if meaningful for both
        s_non_bg_valid = s.get("non_bg_valid", False)
        t_non_bg_valid = t.get("non_bg_valid", False)
        sf = s.get("non_bg", {}) or {}
        tf = t.get("non_bg", {}) or {}
        if s_non_bg_valid and t_non_bg_valid and sf and tf:
            lines.append("  [NON_BG] heuristic non-air pixel stats (intensity > 0.05):")
            for metric in ["mean", "std", "p50", "p99"]:
                row = _row(metric, sf.get(metric), tf.get(metric),
                           src_label, tgt_label)
                if row:
                    lines.append(row)
        else:
            reason = []
            if not s_non_bg_valid:
                reason.append(f"{src_label} bg_frac={s_bg:.0%} outside 5-95%")
            if not t_non_bg_valid:
                reason.append(f"{tgt_label} bg_frac={t_bg:.0%} outside 5-95%")
            if reason:
                lines.append(f"  [NON_BG] OMITTED — threshold degenerate ({'; '.join(reason)})")

        # RAW — for completeness
        sr = s.get("raw", {}) or {}
        tr = t.get("raw", {}) or {}
        if not sr:
            sr = {k: s.get(k) for k in ["mean", "std", "p1", "p50", "p99"]}
            tr = {k: t.get(k) for k in ["mean", "std", "p1", "p50", "p99"]}
        if sr and tr:
            lines.append("  [RAW] global stats (all pixels, includes background):")
            for metric in ["mean", "std", "p1", "p50", "p99"]:
                row = _row(metric, sr.get(metric), tr.get(metric),
                           src_label, tgt_label)
                if row:
                    lines.append(row)
        lines.append("")
    return "\n".join(lines)


def _channel_montage_b64(images: List[np.ndarray],
                         channel_idx: int,
                         k: int = 5,
                         tile_size: int = 192) -> Optional[str]:
    """Pick K images, extract channel `channel_idx`, return a horizontal
    grayscale montage encoded as base64 PNG.

    Returns None if no valid tiles can be built.
    """
    if not images:
        return None
    from PIL import Image as PILImage
    indices = np.linspace(0, len(images) - 1, min(k, len(images)), dtype=int)
    tiles = []
    for i in indices:
        arr = _to_hwc(images[i])
        if channel_idx >= arr.shape[-1]:
            continue
        ch = arr[..., channel_idx].astype(np.float32)
        if ch.size == 0:
            continue
        lo = float(np.percentile(ch, 1))
        hi = float(np.percentile(ch, 99))
        if hi <= lo:
            ch = np.zeros_like(ch)
        else:
            ch = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        pil = PILImage.fromarray((ch * 255).astype(np.uint8))
        pil = pil.resize((tile_size, tile_size), PILImage.BILINEAR)
        tiles.append(np.array(pil, dtype=np.uint8))
    if not tiles:
        return None
    montage = np.concatenate(tiles, axis=1)
    montage_rgb = np.stack([montage] * 3, axis=-1)
    pil = PILImage.fromarray(montage_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


DOMAIN_EXPERT_PROMPT = """You are a radiologist analyzing domain shift between two medical imaging datasets.

You have THREE information sources:
1. **Visual**: the actual images (below)
2. **Knowledge base**: prior experiments on similar shifts (if available)
3. **Literature**: recent papers (if available)

Use ALL sources. When they agree, be confident. When they disagree, explain why.

## Your Analysis (think step by step)

### Step 1: IDENTIFY — What are these images?
Look at the images and determine:
- Imaging modality (CT, MRI, fundus, ultrasound, histopath, X-ray)
- Body region (brain, abdomen, cardiac, retinal, prostate, lung, breast)
- Imaging protocol/sequence (T1w, T2w, FLAIR, bSSFP, non-contrast CT, CTA, color fundus)

### Step 2: COMPARE — What is different between source and target?
Look carefully at both sets:
- Are the same anatomical structures visible?
- Do tissues have the same relative brightness/darkness?
- Is the difference in APPEARANCE (color, brightness) or CONTENT (what's bright vs dark)?

### Step 2.5: VERIFY with the quantitative table — REASON LIKE A DOMAIN EXPERT

A per-channel statistics table is provided in the context block. It has THREE
views per channel: [FOV], [NON_BG], and [RAW]. You MUST reason about what
each view tells you and synthesize them — do not just report numbers.

**Read the views in this order, with this expert reasoning:**

1. **[FOV] background pixel fraction** — read this FIRST.
   - If |delta| > 5pp: cropping / padding / field-of-view changed.
     RAW mean and median can drift dramatically with NO change in tissue intensity.
     This is NOT a calibration shift — do not call it one.
   - Common causes: different scanner FOV, different reconstruction matrix,
     different organ-of-interest crop, different image padding strategy.
   - Implication: BN reset and histogram matching will NOT fix this — they
     would map background pixels to background pixels, which is meaningless.
     The right response is to flag the FOV change and possibly recommend
     re-cropping or padding correction.

2. **[NON_BG] heuristic non-air-pixel stats** — this is the BEST proxy for
   true tissue intensity drift (when valid). It is NOT the segmentation
   foreground (we have no target mask) — it is just pixels above 0.05.
   - When [NON_BG] is OMITTED: the threshold is degenerate — almost all or
     almost no pixels are above it. Reason from [RAW] + [FOV] together,
     but be cautious about claiming intensity drift.
   - When [NON_BG] is reported and shows >20% drift in mean/p50/p99: this
     is a REAL tissue-level intensity / contrast shift.
   - When [NON_BG] is reported and shows <10% drift: tissue intensity is
     stable; any [RAW] drift is dominated by [FOV] change.

3. **[RAW] global stats** — confounded by both. Use only after attributing
   the cause via [FOV] and [NON_BG].

**Synthesis matrix — pick the dominant shift type from the four cases:**
| [FOV] delta | [NON_BG] drift | Conclusion |
|-------------|----------------|------------|
| big (>5pp)  | small (<10%)   | FOV / cropping shift — NOT calibration. Re-cropping needed. |
| big (>5pp)  | big (>20%)     | Mixed: FOV change + real tissue drift. Address both. |
| small       | big (>20%)     | True tissue calibration / contrast shift. BN/histogram appropriate. |
| small       | small          | Genuinely mild shift. Source model may already work. |

**Per-channel reasoning (true RGB only):**
For RGB inputs (e.g., color fundus), if different channels drift in OPPOSITE
directions or by very different amounts, this signals a colorimetric/white-balance
shift. Do NOT average them — list each affected channel in `dominant_shifts`.
For grayscale inputs (CT, MRI replicated to 3 channels), the table will report
a single 'grayscale' channel — reason as a single signal.

Use the per-channel grayscale montages below to visually confirm the table.

### Step 3: CLASSIFY — What level of shift is this?
Based on the PHYSICS of imaging:

**Level 1 — MILD (same physics, same protocol, different acquisition)**
The imaging PHYSICS and PROTOCOL are identical. Only the scanner/vendor/site/calibration differs.
Examples: T2w MRI on Siemens vs T2w MRI on GE. Fundus camera A vs camera B.
Key test: do tissues look the same RELATIVE to each other? If yes → Level 1.

**Level 2 — MODERATE (same physics, different protocol)**
Same imaging modality but different sequence/weighting/contrast agent.
The RELATIVE brightness of tissues CHANGES (e.g., CSF dark in T1 but bright in T2).
Examples: T1→T2, pre-contrast→post-contrast, different echo times.
Key test: do some tissues swap from bright to dark? If yes → Level 2.

**Level 3 — SEVERE (different physics)**
Fundamentally different imaging principles.
Examples: CT→MRI (X-ray vs magnetic resonance), MRI→ultrasound.
Key test: is the image formation physics completely different? If yes → Level 3.

### Step 4: ESTIMATE — Foreground priors
Based on the anatomy and task:
- Where should the target structure be in the image?
- How much of the image should it occupy? (percentage)
- What shape should it be?

### Step 5: WARN — Known pitfalls
Based on the shift level and imaging type, what typically goes wrong?

## Source Images (model was trained on these)
{source_image_section}

## Target Images (model will be deployed here)
{target_image_section}

## Knowledge & Literature
{context}

## Return your analysis as JSON:
```json
{{
    "modality_source": "<what you identified>",
    "modality_target": "<what you identified>",
    "protocol_source": "<imaging protocol/sequence>",
    "protocol_target": "<imaging protocol/sequence>",
    "shift_level": 1 | 2 | 3,
    "shift_level_reasoning": "<WHY this level, based on physics>",
    "visual_differences": ["<specific difference 1>", ...],
    "tissues_same_relative_brightness": true | false,
    "severity": "mild | moderate | severe",
    "contrast_mechanism_change": "same | similar | different | fundamentally_different",
    "dominant_shift_factor": "<single most important difference>",
    "foreground_priors": {{
        "expected_location": "<where>",
        "expected_area_fraction": [<min>, <max>],
        "expected_shape": "<shape>"
    }},
    "what_will_fail": ["<failure mode that this domain shift is likely to cause>", ...],
    "what_works_in_literature": ["<method with citation>", ...],
    "shift_types": ["<type>"],
    "dominant_shifts": [
        {{
            "channel": "<channel name or 'global'>",
            "shift_type": "intensity | contrast | resolution | fov | acquisition | anatomy",
            "description": "<one sentence with concrete numbers from the table when possible>",
            "severity": "low | medium | high"
        }}
    ],
    "calibration_concern": "calibration shift (intensity rescaling sufficient) | feature shift (model representations are wrong) | mixed",
    "evidence_sources": {{
        "visual": "<what you SAW>",
        "quantitative": "<which numbers from the per-channel table you used>",
        "knowledge_base": "<what KB says>",
        "literature": "<what papers say>"
    }}
}}
```"""


def run_domain_expert(source_images: List[np.ndarray],
                      target_images: List[np.ndarray],
                      llm_client,
                      task_name: str = "",
                      source_domain: str = "",
                      target_domain: str = "",
                      kb_knowledge: Dict = None,
                      n_samples: int = 3,
                      channel_names: Optional[List[str]] = None,
                      per_channel_view: bool = True) -> Dict:
    """Run the Domain Expert agent.

    Args:
        source_images: source domain images (H,W,C numpy)
        target_images: target domain images
        llm_client: UnifiedLLMClient with multimodal support
        task_name: optional task description
        source_domain: optional source domain description
        target_domain: optional target domain description
        kb_knowledge: optional knowledge base entry
        n_samples: number of sample images to show LLM
        channel_names: optional human-readable names for each input channel,
            in the same order as the C axis of the image arrays. Used to label
            the per-channel statistics table and the per-channel montages.
            Supplied by the caller (typically via TaskConfig.channel_names) so
            this agent stays modality-agnostic. If None, channels are labeled
            "channel_0", "channel_1", ...
        per_channel_view: if True (default), build one grayscale montage per
            channel so the LLM sees each modality independently. If False, fall
            back to the legacy combined RGB-tile presentation.

    Returns:
        Dict with domain analysis including shift characterization. In addition
        to the legacy fields, the dict contains:
            - per_channel_stats_source / per_channel_stats_target (raw stats)
            - dominant_shifts (LLM-extracted list)
            - calibration_concern (LLM-extracted enum)
    """
    print("\n" + "=" * 55)
    print("AGENT 1: Domain Expert (Visual + Knowledge + Literature)")
    print("=" * 55)

    # Step 1: Search literature for relevant domain adaptation papers
    literature_text = ""
    try:
        from medadaptagent.knowledge.literature_search import (
            generate_search_queries, search_pubmed, fetch_abstracts,
        )
        queries = generate_search_queries(
            source_domain or "", target_domain or "", task_name or "")
        all_pmids = []
        seen = set()
        for q in queries[:2]:  # limit to 2 queries to save time
            print(f"  [PubMed] Searching: '{q}'")
            pmids = search_pubmed(q, max_results=5)
            for pid in pmids:
                if pid not in seen:
                    all_pmids.append(pid)
                    seen.add(pid)

        if all_pmids:
            papers = fetch_abstracts(all_pmids[:6])
            print(f"  [PubMed] Found {len(papers)} relevant papers")
            for p in papers[:3]:
                print(f"    - [{p.get('year', '?')}] {p.get('title', '')[:70]}...")
            literature_text = "\n".join(
                f"[{p.get('year', '?')}] {p.get('title', '')}: {p.get('abstract', '')[:200]}"
                for p in papers[:5]
            )
        else:
            print("  [PubMed] No papers found")
    except Exception as e:
        print(f"  [PubMed] Search failed: {e}")

    # Step 2a: Compute per-channel statistics on the FULL image sets.
    # If all channels are identical (grayscale replicated to RGB, common
    # for CT/MRI fed to 3-channel-input models), collapse to a single
    # "grayscale" channel to avoid showing the same row 3 times.
    src_grayscale = _channels_identical(source_images)
    tgt_grayscale = _channels_identical(target_images)
    is_grayscale = src_grayscale and tgt_grayscale

    src_stats = _compute_channel_stats(source_images)
    tgt_stats = _compute_channel_stats(target_images)

    if is_grayscale:
        # Keep only ch0 — all channels are identical
        src_stats = src_stats[:1]
        tgt_stats = tgt_stats[:1]
        ch_names = ["grayscale"]
        n_channels = 1
        print("  [Channels] Grayscale-replicated input detected → reporting as 1 channel")
    else:
        n_src_ch = len(src_stats)
        n_tgt_ch = len(tgt_stats)
        n_channels = max(n_src_ch, n_tgt_ch)
        if channel_names is None:
            ch_names = [f"channel_{i}" for i in range(n_channels)]
        else:
            ch_names = list(channel_names) + [
                f"channel_{i}" for i in range(len(channel_names), n_channels)
            ]
    stats_table = _format_channel_stats_table(
        src_stats, tgt_stats, channel_names=ch_names,
        src_label="SOURCE", tgt_label="TARGET",
    )

    # Step 2b: Prepare images for the multimodal call.
    # Default: one grayscale montage per channel (preserves per-channel intensity).
    # Fallback: legacy combined-tile presentation.
    src_b64: List[str] = []
    tgt_b64: List[str] = []
    src_labels: List[str] = []
    tgt_labels: List[str] = []

    use_per_channel = bool(per_channel_view) and n_channels >= 1
    if use_per_channel:
        for c in range(n_channels):
            name = ch_names[c]
            src_m = _channel_montage_b64(source_images, c, k=n_samples)
            tgt_m = _channel_montage_b64(target_images, c, k=n_samples)
            if src_m is not None:
                src_b64.append(src_m)
                src_labels.append(f"SOURCE - {name} (montage of {n_samples} cases)")
            if tgt_m is not None:
                tgt_b64.append(tgt_m)
                tgt_labels.append(f"TARGET - {name} (montage of {n_samples} cases)")
        if not src_b64 or not tgt_b64:
            # Per-channel path produced nothing usable, fall back to legacy
            use_per_channel = False

    if not use_per_channel:
        src_b64 = prepare_sample_images(source_images, n=n_samples)
        tgt_b64 = prepare_sample_images(target_images, n=n_samples)
        src_labels = [f"Source Image {i+1}" for i in range(len(src_b64))]
        tgt_labels = [f"Target Image {i+1}" for i in range(len(tgt_b64))]

    # Step 3: Build context from ALL knowledge sources
    context_parts = []
    if task_name:
        context_parts.append(f"Task: {task_name}")
    if source_domain:
        context_parts.append(f"Source domain: {source_domain}")
    if target_domain:
        context_parts.append(f"Target domain: {target_domain}")
    if stats_table:
        context_parts.append(
            "## Per-Channel Quantitative Statistics (computed on the actual arrays):\n"
            + stats_table)
    if kb_knowledge:
        context_parts.append(f"## Local Knowledge Base (from previous experiments):\n{json.dumps(kb_knowledge, indent=2)[:500]}")
    if literature_text:
        context_parts.append(f"## Relevant Literature (from PubMed search):\n{literature_text}")
    if not context_parts:
        context_parts.append("No additional context provided. Reason from the images and your own medical imaging expertise.")
    context = "\n\n".join(context_parts)

    # Build multimodal prompt
    prompt_text = DOMAIN_EXPERT_PROMPT.format(
        source_image_section="(see images below)",
        target_image_section="(see images below)",
        context=context,
    )

    # Construct multimodal message
    all_images = src_b64 + tgt_b64
    all_labels = src_labels + tgt_labels
    content = build_multimodal_message(prompt_text, all_images, all_labels)

    # Call LLM with images
    view_mode = "per-channel montages" if use_per_channel else "combined tiles"
    print(f"  Showing {len(src_b64)} source + {len(tgt_b64)} target images to LLM ({view_mode})...")
    if stats_table:
        print(f"  Injected per-channel stats table for {n_channels} channel(s)")
    try:
        response = llm_client.cached_complete(
            messages=[{"role": "user", "content": content}],
            max_tokens=1500,
            temperature=0,
            seed=42,
        )
        text = response.choices[0].message.content or ""

        # Parse JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            analysis = json.loads(text[start:end])
        else:
            print(f"  Warning: could not parse JSON, using text analysis")
            analysis = {"raw_analysis": text, "severity": "moderate",
                        "contrast_mechanism_change": "unknown"}

    except Exception as e:
        print(f"  LLM failed ({e}), using fallback")
        analysis = _fallback_analysis(source_domain, target_domain)

    # Attach raw stats so downstream agents and the KB can use them.
    analysis["per_channel_stats_source"] = src_stats
    analysis["per_channel_stats_target"] = tgt_stats
    analysis["channel_names"] = ch_names

    # Print key findings
    print(f"  Source: {analysis.get('modality_source', '?')}")
    print(f"  Target: {analysis.get('modality_target', '?')}")
    print(f"  Severity: {analysis.get('severity', '?')}")
    print(f"  Contrast: {analysis.get('contrast_mechanism_change', '?')}")
    dom = analysis.get('dominant_shift_factor', '?')
    if isinstance(dom, str):
        print(f"  Dominant: {dom[:80]}")
    cal = analysis.get('calibration_concern')
    if cal:
        print(f"  Calibration concern: {cal}")
    dshifts = analysis.get('dominant_shifts')
    if isinstance(dshifts, list) and dshifts:
        print(f"  Dominant shifts ({len(dshifts)}):")
        for d in dshifts[:5]:
            if isinstance(d, dict):
                ch = d.get('channel', '?')
                st = d.get('shift_type', '?')
                sv = d.get('severity', '?')
                desc = str(d.get('description', ''))[:80]
                print(f"    [{str(sv).upper():6s}] {ch} - {st}: {desc}")
    if analysis.get("what_will_fail"):
        for wf in analysis["what_will_fail"][:2]:
            print(f"  Warning: {str(wf)[:80]}")

    return analysis


def _fallback_analysis(source_domain: str, target_domain: str) -> Dict:
    """Simple rule-based fallback when LLM is unavailable."""
    src = source_domain.lower()
    tgt = target_domain.lower()

    if ("ct" in src and "mri" in tgt) or ("mri" in src and "ct" in tgt):
        return {"severity": "severe", "contrast_mechanism_change": "fundamentally_different",
                "standardization_gap": "severe", "recommended_scope": "bn+output",
                "dominant_shift_factor": "cross-modality (CT vs MRI)"}
    elif "t1" in src and "t2" in tgt:
        return {"severity": "moderate", "contrast_mechanism_change": "different",
                "standardization_gap": "moderate", "recommended_scope": "bn+output",
                "dominant_shift_factor": "contrast reversal (T1 vs T2)"}
    else:
        return {"severity": "mild", "contrast_mechanism_change": "same",
                "standardization_gap": "mild", "recommended_scope": "bn_only",
                "dominant_shift_factor": "cross-site/device shift"}
