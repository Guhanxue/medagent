"""Free-sentence intake -> the three contracts.

The user describes what they want in plain language; the harness extracts the THREE CONTRACTS
every run needs, asks (at most once) for whatever is genuinely missing, and hands a validated
RunContract to the pipeline. Free text in, structure out -- the user never fills out a form.

  C1 MODEL   : what to deploy      (local ckpt path | hub id | repo url | registry name)
  C2 DATA    : where it must work  (directory | npz | manifest)  [+ optional task question]
  C3 BUDGET  : what may be spent   (max labels K, objective: max_performance |
                                    never_worse | min_annotation; compute cap optional)

Design rules (from the measured methodology):
  * M2 structure-enforces: extraction TARGETS a fixed schema and is VALIDATED deterministically
    (paths must exist, K must parse to an int >= 0, objective must be one of three). The LLM
    maps language onto the schema; it cannot invent fields or skip validation.
  * Ask, don't guess, for C1/C2: a wrong model path or data dir wastes a whole run. DEFAULT,
    don't ask, for C3 when absent: K=0 (label-free) + objective=never_worse -- the safe floor,
    stated back to the user in the confirmation.
"""
from dataclasses import dataclass, field, asdict
import json, os, re

OBJECTIVES = ("max_performance", "never_worse", "min_annotation")

@dataclass
class RunContract:
    model_ref: str = ""          # C1
    data_path: str = ""          # C2
    task_question: str = ""      # C2 optional ("cardiomegaly present vs absent")
    k_labels: int = 0             # C3
    objective: str = "never_worse"
    notes: str = ""
    missing: list = field(default_factory=list)
    def valid(self):
        return not self.missing

_SCHEMA_PROMPT = """Extract the deployment request into EXACTLY this JSON (no other text):
{"model_ref": "<path|hub-id|repo-url|registry-name|UNKNOWN>",
 "data_path": "<path|UNKNOWN>",
 "task_question": "<what the site wants predicted, verbatim-ish|UNKNOWN>",
 "k_labels": <int, 0 if they cannot label, UNKNOWN if unstated>,
 "objective": "max_performance|never_worse|min_annotation|UNKNOWN"}
Rules: do not invent paths. "a dozen"->12, "a handful"->6, "a few"->5. If they say they
cannot annotate / have no labels -> k_labels=0. If they emphasise safety ("must not be worse",
"cannot regress") -> objective=never_worse; if they emphasise best accuracy -> max_performance;
if they emphasise minimal clinician effort -> min_annotation.
User request: \"\"\"{text}\"\"\""""

def _clean_path(x):
    """Strip the punctuation that free text glues onto paths ('data/x,' 'data/x?' '(data/x)')."""
    return str(x or "").strip().strip("\"'()[]{}").rstrip(".,;:!?")

def _deterministic_validate(d, cwd="."):
    c = RunContract()
    mr = _clean_path(d.get("model_ref", ""))
    if mr and mr != "UNKNOWN":
        c.model_ref = mr
        looks_path = ("/" in mr and not mr.count("/") == 1) or mr.endswith((".pt", ".pth", ".ckpt"))
        if looks_path and not os.path.exists(os.path.join(cwd, mr)) and not os.path.exists(mr):
            c.missing.append(("model_ref", f"checkpoint path not found: {mr}"))
    else:
        c.missing.append(("model_ref", "which model should be deployed? (a checkpoint path, a hub id like org/name, or a registry task name)"))
    dp = _clean_path(d.get("data_path", ""))
    if dp and dp != "UNKNOWN":
        c.data_path = dp
        if not os.path.exists(os.path.join(cwd, dp)) and not os.path.exists(dp):
            c.missing.append(("data_path", f"data path not found: {dp}"))
    else:
        c.missing.append(("data_path", "where is the site's data? (a directory of images/volumes, or an npz)"))
    tq = str(d.get("task_question", "") or "")
    c.task_question = "" if tq == "UNKNOWN" else tq
    k = d.get("k_labels", "UNKNOWN")
    if k == "UNKNOWN" or k is None:
        c.k_labels = 0; c.notes += "k_labels unstated -> defaulted to 0 (label-free). "
    else:
        try: c.k_labels = max(0, int(k))
        except Exception:
            c.k_labels = 0; c.notes += f"could not parse k_labels={k!r} -> defaulted 0. "
    ob = str(d.get("objective", "") or "UNKNOWN")
    if ob in OBJECTIVES:
        c.objective = ob
    else:
        c.objective = "never_worse"; c.notes += "objective unstated -> defaulted to never_worse. "
    return c

def parse_request(text, llm=None, cwd="."):
    """Free text -> RunContract. With an llm client, extraction is LLM-mapped then
    deterministically validated; without one, a regex fallback handles the common phrasings."""
    d = {}
    if llm is not None:
        raw = llm.chat(_SCHEMA_PROMPT.replace("{text}", text), max_tokens=600)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("LLM intake returned no JSON object; request was not executed")
        d = json.loads(m.group(0))
        if not isinstance(d, dict) or not d:
            raise ValueError("LLM intake must return a nonempty JSON object")
    if not d:                                   # no-LLM fallback: conservative regexes
        pm = re.search(r"(\S+\.(?:pt|pth|ckpt))", text)
        hub = re.search(r"\b([\w.-]+/[\w.-]+)\b", text)
        dm = re.search(r"(\S*(?:/data/|data/|/dataset)\S*)", text)
        km = re.search(r"(\d+)\s*(?:cases|labels|annotations|examples)", text)
        d = {"model_ref": (pm and pm.group(1)) or (hub and hub.group(1)) or "UNKNOWN",
             "data_path": (dm and dm.group(1)) or "UNKNOWN",
             "task_question": "UNKNOWN",
             "k_labels": int(km.group(1)) if km else (0 if re.search(r"no label|cannot (?:label|annotate)|label-free", text, re.I) else "UNKNOWN"),
             "objective": ("never_worse" if re.search(r"not (?:be )?worse|regress|safe", text, re.I)
                           else "min_annotation" if re.search(r"minimal (?:clinician )?(?:effort|annotation|label)", text, re.I)
                           else "max_performance" if re.search(r"(?:best|maximum|highest) (?:accuracy|performance)", text, re.I)
                           else "UNKNOWN")}
    return _deterministic_validate(d, cwd=cwd)

def clarifying_question(contract):
    """ONE question covering everything genuinely missing (C1/C2 only; C3 defaults)."""
    need = [q for _, q in contract.missing]
    if not need: return None
    return "Before I start, I need: " + " AND ".join(need)

def confirmation(contract):
    c = contract
    return (f"Contract -- model: {c.model_ref} | data: {c.data_path} | "
            f"task: {c.task_question or '(from model card)'} | label budget: {c.k_labels} | "
            f"objective: {c.objective}. {c.notes}".strip())
