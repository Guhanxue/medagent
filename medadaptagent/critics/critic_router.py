"""Critic router — decide which label-free critic validates each structure.

WHY THIS EXISTS
---------------
MIFR's box-grid MedSAM reference is an ORGAN/REGION anatomy proxy. It ranks
adaptation candidates well when the segmentation target IS the salient
structure the reference recovers (liver, prostate, optic disc/cup, kidney,
cardiac chambers), and it ANTI-correlates with Dice when the target is a small
focal lesion inside a larger structure — BraTS cross-contrast matrix
(job 619695) gave within-ds tau(MIFR,Dice) = -0.060 because an over-segmenting
model overlaps whole-brain anatomy (high MIFR) yet has bad tumor Dice.

So the agentic system must KNOW what it is segmenting and route accordingly:

    salient organ / region  ->  MIFR critic   (+ VLM non-inferiority gate)
    small focal lesion       ->  VLM gate + safe-default abstain (no MIFR)

DESIGN PRINCIPLE (matches the house pattern: LLM categorical, rules numeric)
- The target is DECLARED by the user in FREE LANGUAGE, never guessed. An LLM
  normalizes that free text into a per-structure role + saliency (categorical).
- A numeric fg-fraction PROBE (fm_critic.mifr_score(..., return_probe=True))
  cross-checks the declaration against the actual masks, label-free, and flags
  contradictions back to the user instead of silently mis-routing.
- "label-free" is preserved: a one-time task DESCRIPTOR (what organ/lesion) is
  metadata, not per-image supervision.

Routing is PER STRUCTURE. For a multi-structure task the candidate is scored by
MIFR over only the MIFR-applicable classes; lesion classes go to the VLM/safe
branch. The adaptation METHOD is still chosen once for the whole model — per-
class routing changes how each candidate is SCORED, not which model is applied.
"""
from __future__ import annotations
import json
import re

# Open-vocabulary keyword priors for the rule fallback (LLM path overrides).
_SALIENT_WORDS = {
    "liver", "spleen", "kidney", "kidneys", "renal", "pancreas", "prostate",
    "bladder", "gallbladder", "stomach", "aorta", "vessel", "vena", "cava",
    "heart", "ventricle", "myocardium", "atrium", "lv", "rv", "myo",
    "lung", "lobe", "brain", "hippocampus", "gland", "adrenal", "uterus",
    "disc", "cup", "optic", "cornea", "retina", "vertebra", "femur", "muscle",
    "organ", "region", "parenchyma", "cortex", "chamber",
}
_LESION_WORDS = {
    "tumor", "tumour", "lesion", "mass", "nodule", "met", "mets", "metastasis",
    "metastases", "cancer", "carcinoma", "hcc", "polyp", "plaque", "infarct",
    "hemorrhage", "haemorrhage", "edema", "oedema", "enhancing", "necrosis",
    "abnormality", "focus", "foci", "spot", "fibroid",
}
# words that make a lesion big enough to behave like a salient structure
_LARGE_WORDS = {"large", "bulky", "dominant", "whole", "diffuse", "extensive",
                "replacing", "giant"}

# probe threshold: reference covers >~ this multiple of the scored mask => the
# reference is a much bigger structure than the target => MIFR mis-targeted.
COV_RATIO_INAPPLICABLE = 5.0
COV_RATIO_OK = 3.0


def _classify_token(name: str, context: str = "") -> dict:
    """Rule fallback: name (+ optional free-text context) -> role/salient."""
    txt = f"{name} {context}".lower()
    toks = set(re.findall(r"[a-z]+", txt))
    is_lesion = bool(toks & _LESION_WORDS)
    is_salient_word = bool(toks & _SALIENT_WORDS)
    is_large = bool(toks & _LARGE_WORDS)
    if is_lesion and not is_large:
        role, salient = "lesion", False
    elif is_lesion and is_large:
        role, salient = "large_mass", True  # dominant mass ~ salient structure
    elif is_salient_word:
        role, salient = "organ_region", True
    else:
        # unknown noun: default to organ/region but mark low-confidence so the
        # probe / user-confirm decides.
        role, salient = "unknown", True
    return {"name": name, "role": role, "salient": salient,
            "mifr_applicable": salient, "source": "rule",
            "confident": (is_lesion or is_salient_word)}


_LLM_PROMPT = """You normalize a free-text description of a medical image \
segmentation TARGET into structured per-structure metadata for routing a \
validation critic.

User description: "{desc}"
Class labels present (name -> id): {labels}

For EACH foreground structure return JSON with fields:
  name      : the structure label
  role      : one of organ | region | substructure | vessel | duct | \
large_mass | lesion | other  (open if none fit)
  salient   : true if it is a DOMINANT anatomical structure that a \
prompt-free foundation model would recover as a salient region; false if it \
is a SMALL focal target (e.g. tumor/nodule/lesion) occupying a tiny fraction \
inside a larger structure.

Rules: organs/regions/large masses/chambers -> salient true. \
small tumors/lesions/nodules/metastases -> salient false. \
A large/bulky/dominant mass -> salient true.
Return ONLY a JSON list, no prose."""


def parse_target_freetext(desc: str, labels: dict | None = None,
                          llm=None) -> list[dict]:
    """Parse a free-language target description into per-structure routing
    metadata. labels: {name: class_id} (background id 0 dropped). llm: an
    optional chat client with .complete(prompt)->str; falls back to the rule
    classifier (so the offline experiment runs without a key)."""
    labels = labels or {}
    fg = {k: v for k, v in labels.items() if int(v) != 0}
    if llm is not None:
        try:
            raw = llm.complete(_LLM_PROMPT.format(
                desc=desc, labels=json.dumps(fg)))
            m = re.search(r"\[.*\]", raw, re.S)
            parsed = json.loads(m.group(0) if m else raw)
            out = []
            for p in parsed:
                nm = p.get("name", "")
                sal = bool(p.get("salient", True))
                out.append({"name": nm, "role": p.get("role", "other"),
                            "salient": sal, "mifr_applicable": sal,
                            "class_id": fg.get(nm), "source": "llm",
                            "confident": True})
            if out:
                return out
        except Exception:
            pass  # fall through to rule classifier
    # rule fallback: classify each declared label name in the desc context
    names = list(fg) if fg else re.findall(r"[A-Za-z_][A-Za-z_ ]+", desc)
    out = []
    for nm in names:
        meta = _classify_token(nm, context=desc)
        meta["class_id"] = fg.get(nm)
        out.append(meta)
    return out


def cross_check_probe(meta: dict, cov_ratio: float) -> dict:
    """Reconcile a declared structure with the numeric fg-fraction probe.
    Returns meta augmented with probe verdict + a contradiction flag the
    caller surfaces to the user (no silent override of the declaration)."""
    meta = dict(meta)
    meta["cov_ratio"] = cov_ratio
    probe_says_focal = cov_ratio > COV_RATIO_INAPPLICABLE
    probe_says_salient = cov_ratio <= COV_RATIO_OK
    contradiction = None
    if meta["salient"] and probe_says_focal:
        contradiction = (f"declared SALIENT ('{meta['name']}') but masks look "
                         f"FOCAL (cov_ratio {cov_ratio:.1f}>{COV_RATIO_INAPPLICABLE}) "
                         f"- confirm target, or MIFR may mis-rank")
    elif (not meta["salient"]) and probe_says_salient:
        contradiction = (f"declared FOCAL ('{meta['name']}') but masks look "
                         f"SALIENT (cov_ratio {cov_ratio:.1f}<={COV_RATIO_OK}) "
                         f"- MIFR may in fact be usable here")
    meta["probe_focal"] = probe_says_focal
    meta["contradiction"] = contradiction
    return meta


def route(structures: list[dict]) -> dict:
    """Per-structure routing decision -> the scoring plan for the selector.

    Returns:
      use_mifr      : whether ANY class is MIFR-applicable (else pure VLM/safe)
      mifr_classes  : foreground class ids scored by MIFR
      lesion_classes: focal class ids routed to VLM + safe-default
      flags         : human-facing contradiction notes (from the probe)
    """
    mifr_classes, lesion_classes, flags = [], [], []
    for s in structures:
        cid = s.get("class_id")
        if s.get("contradiction"):
            flags.append(s["contradiction"])
        if s.get("mifr_applicable"):
            if cid is not None:
                mifr_classes.append(int(cid))
        else:
            if cid is not None:
                lesion_classes.append(int(cid))
    return {"use_mifr": len(mifr_classes) > 0,
            "mifr_classes": mifr_classes or None,  # None => all-fg legacy
            "lesion_classes": lesion_classes,
            "flags": flags}
