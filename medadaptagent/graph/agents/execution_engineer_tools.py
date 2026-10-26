"""Agent 4 v2 — Execution Engineer over the unified tool registry.

Takes a multi-candidate plan from Agent 3 v2:
    {"candidates": [{"tools": [...], "tool_kwargs": ..., ...}, ...],
     "skip_adaptation": bool}

Pipeline:
  1. For each candidate, deep-copy the source, apply the tool chain,
     and run a VLM "tournament vote" (current vs source) on a small
     anchor set. Each candidate gets a VLM score in [-1, +1].
  2. Pick the candidate with the highest score. Ties broken by fewer
     "broken" absolute votes. If best score < 0 (every candidate looked
     worse than source on tournament anchors), revert to source.
  3. Run the existing final-check gate on the winning candidate
     (catastrophic collapse + symmetric VLM second opinion on a
     DISJOINT anchor set, policy C: majority worse OR ≥2 broken).

The tournament uses K=3 anchors per candidate (cheap); the final check
uses K=5 disjoint anchors (the load-bearing gate). With 3 candidates,
total VLM cost is 3×3 + 5 = 14 calls per dataset.
"""
from __future__ import annotations
import copy
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from medadaptagent import tools as T
from medadaptagent.tools import composition as _composition
from medadaptagent.graph.agents.execution_engineer import (
    _run_final_check, _measure_quality, _vlm_iteration_vote,
    _build_vlm_task_ctx,
)
from PIL import Image as _PILImage


# Few-shot gate thresholds (#62). With K=5..20 labels the Dice estimate
# has noise on the order of 1-2pp; thresholds at 2pp are tight enough to
# catch real regressions while tolerating that noise. Larger K -> can
# tighten these.
FEWSHOT_REJECT_THR = -0.02   # delta below this -> force revert
FEWSHOT_ACCEPT_THR = +0.02   # delta above this -> force accept


def _fewshot_dice(model, target_images, fewshot_indices, fewshot_labels,
                    num_classes, device, size: int = 256, task=None,
                    metric: Optional[str] = None) -> float:
    """Compute the task metric on K labeled target samples. The chosen
    metric is task-configurable: `metric='auroc'` for binary cls (the
    right call for imbalanced data — accuracy is misleading when one
    class dominates). Defaults to the task adapter's own metric_name
    (dice for seg, accuracy for cls). Function name kept for back-compat."""
    if task is None:
        from medadaptagent.tasks import get_task
        task = get_task("segmentation")
    images = [target_images[i] for i in fewshot_indices]
    kw = dict(
        model=model, images=images, labels=fewshot_labels,
        num_classes=num_classes, device=device, size=size,
    )
    # task adapters that support metric= override accept the kwarg;
    # base implementation ignores it (back-compat).
    try:
        return task.avg_metric(metric=metric, **kw)
    except TypeError:
        return task.avg_metric(**kw)


def _decide_capabilities(model: nn.Module) -> List[str]:
    """Lightweight model-introspection in case the upstream caller didn't
    pass capabilities. Mirrors `norm_utils.model_capabilities` (which we
    don't import here to keep this module standalone)."""
    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                 for m in model.modules())
    has_affine = any(getattr(m, "affine", False) and getattr(m, "weight", None)
                     is not None for m in model.modules()
                     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                        nn.BatchNorm3d, nn.InstanceNorm1d,
                                        nn.InstanceNorm2d, nn.InstanceNorm3d,
                                        nn.LayerNorm, nn.GroupNorm)))
    caps = []
    if has_bn:
        caps.extend(["bn_running_stats", "tent_affine"])
    elif has_affine:
        caps.append("tent_affine")
    return caps


# Adapter holding the two pieces of state the existing gate consumes:
class _PipelineState:
    """Minimal stand-in for the orchestrator's state object so we can
    call `_run_final_check(ps, ...)` without dragging the full graph."""
    def __init__(self, adapted_model: nn.Module):
        self.adapted_model = adapted_model
        self.iteration_history: List[Dict] = []


def _apply_chain(source_model: nn.Module,
                   target_images: List[np.ndarray],
                   device: str,
                   num_classes: int,
                   tools_list: List[str],
                   tool_kwargs: Dict[str, Dict],
                   seed: int = 42,
                   fewshot_indices: Optional[List[int]] = None,
                   fewshot_labels: Optional[List[np.ndarray]] = None,
                   task=None,
                   composition: Optional[List[Dict]] = None,
                   capabilities: Optional[List[str]] = None) -> Dict:
    """Apply one candidate's pipeline to a fresh copy of source.

    Phase D4: the pipeline is expressed as an ordered list of *atoms*
    (one registered tool + bound kwargs each). When an explicit
    `composition` is supplied it is authoritative; otherwise the legacy
    `tools_list` + `tool_kwargs` pair is normalized into the same atom
    list, so plain candidates behave exactly as before. The ordered
    atoms are validated (illegal orderings dropped) and applied left-to-
    right over ONE ToolContext. Because each atom carries its own
    kwargs, the SAME tool may appear more than once with different
    kwargs — we overwrite `ctx.notes[tool]` immediately before each atom
    so repeats do not clobber one another.

    Re-seeds before adaptation so sequential variants in the tournament
    don't share RNG drift (otherwise variant 3 sees different torch RNG
    than variant 0, which silently shifts Tent's optimization trajectory).
    Returns dict with `model`, `applied`, `skipped`, `elapsed_s`."""
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    adapted = copy.deepcopy(source_model)
    ctx = T.ToolContext(
        source_model=source_model,
        current_model=adapted,
        target_images=target_images,
        device=device,
        num_classes=num_classes,
        fewshot_indices=fewshot_indices,
        fewshot_labels=fewshot_labels,
        task=task,
    )
    # Build the canonical ordered atom list from whichever shape the
    # candidate used, then validate orderings (illegal atoms dropped with
    # a logged reason). Length cap mirrors Agent 3's MAX_CHAIN_LEN.
    cand_shape = {"tools": list(tools_list or []),
                   "tool_kwargs": tool_kwargs or {}}
    if composition:
        cand_shape["composition"] = composition
    atoms = _composition.normalize_composition(cand_shape)
    vres = _composition.validate_composition(
        atoms, capabilities=capabilities, max_len=4)
    applied, skipped = [], []
    for w in vres.get("warnings", []):
        skipped.append({"name": "(validator)", "reason": w})
    t0 = time.time()
    for atom in vres["atoms"]:
        name = atom.tool
        if name not in T.TOOL_REGISTRY:
            skipped.append({"name": name, "reason": "unknown_tool"})
            continue
        kwargs = atom.kwargs or {}
        try:
            # Overwrite (not merge) this tool's notes so a repeated tool
            # with different kwargs does not inherit the prior atom's
            # values. apply_tool then re-merges these kwargs in.
            ctx.notes[name] = {}
            ctx = T.apply_tool(name, ctx, **kwargs)
            applied.append(name)
        except Exception as e:
            skipped.append({"name": name, "reason": str(e)[:200]})
    return {
        "model": ctx.current_model, "applied": applied,
        "skipped": skipped, "elapsed_s": round(time.time() - t0, 2),
    }


def _tournament_vote(candidates_state: List[Dict],
                       source_model: nn.Module,
                       target_images: List[np.ndarray],
                       device: str,
                       llm_client,
                       vlm_ctx: Dict,
                       k_anchors: int = 3,
                       fewshot_indices: Optional[List[int]] = None,
                       fewshot_labels: Optional[List[np.ndarray]] = None,
                       num_classes: int = 2,
                       task=None) -> List[Dict]:
    """Run a per-candidate VLM vote vs source on a SHARED anchor set
    sampled with linspace. Same anchors for all candidates so scores
    are comparable. Disjoint from the final-check anchors (we use a
    different linspace partition).

    Returns the input list mutated with `vlm_score`, `votes_rel`,
    `votes_abs`, `vlm_note` keys added.
    """
    if not target_images or k_anchors < 1 or len(candidates_state) == 0:
        for c in candidates_state:
            c["vlm_score"] = 0.0
            c["votes_rel"], c["votes_abs"] = [], []
            c["vlm_note"] = "skipped (no images)"
        return candidates_state
    n = len(target_images)
    # Use middle of each linspace bin so we don't overlap final-check
    # which uses linspace endpoints.
    base = np.linspace(0, n - 1, k_anchors + 2, dtype=int).tolist()
    anchors = base[1:-1] if k_anchors + 2 <= n else base[:k_anchors]
    anchor_imgs = [target_images[i] for i in anchors]

    for c in candidates_state:
        score, votes_rel, note, votes_abs = _vlm_iteration_vote(
            current_model=c["model"], source_model=source_model,
            anchor_imgs=anchor_imgs, device=device,
            llm_client=llm_client, task_ctx=vlm_ctx,
            k=len(anchor_imgs),
            task=task,
        )
        c["vlm_score"] = score if isinstance(score, (int, float)) else 0.0
        c["votes_rel"], c["votes_abs"] = votes_rel, votes_abs
        c["vlm_note"] = note or ""
        c["tournament_anchors"] = anchors
        # When K labels available, compute GT Dice on the labeled samples
        # for this variant. Used by _pick_winner as the primary signal.
        if fewshot_indices and fewshot_labels:
            c["fewshot_dice"] = _fewshot_dice(
                c["model"], target_images, fewshot_indices, fewshot_labels,
                num_classes, device, task=task)
        else:
            c["fewshot_dice"] = None
    return candidates_state


def _pick_winner(candidates_state: List[Dict]) -> Dict:
    """Pick best variant. When K labels are available (each variant
    has c["fewshot_dice"] set), fewshot_dice is the PRIMARY ranking
    signal — VLM votes are tie-breakers only. Without K labels, VLM
    votes are primary.

    Sort key (higher = better):
       1) fewshot_dice (when K labels available; else 0)
       2) vlm_score
       3) fewer 'broken' absolute votes
       4) SHORTER chain length (Occam)
       5) origin == 'agent3' over 'ablation'
       6) higher confidence

    If ALL signals say revert, the caller forces source revert."""
    def _key(c):
        n_broken = sum(1 for v in c.get("votes_abs", []) if v == "broken")
        applied = c.get("applied") or c["candidate"].get("tools", [])
        chain_len = len(applied)
        is_agent3 = 1 if c["candidate"].get("origin") == "agent3" else 0
        fs = c.get("fewshot_dice")
        # When fewshot_dice is None, all variants get the same value
        # so VLM dominates. When present, it dominates VLM.
        fs_key = fs if isinstance(fs, (int, float)) else 0.0
        return (fs_key, c["vlm_score"], -n_broken, -chain_len,
                is_agent3, c.get("confidence", 0.0))
    best = max(candidates_state, key=_key)
    return best


def _atoms_key(atoms):
    """Identity for a list of atoms: ordered (tool, sorted-kwargs) tuple.
    Two atoms of the same tool with different kwargs are DISTINCT, so a
    repeated-tool composition is not collapsed during dedup."""
    return tuple((a.tool, tuple(sorted((a.kwargs or {}).items())))
                 for a in atoms)


def _build_variants(candidates):
    """Expand Agent-3 candidates into the actual tournament pool. Each
    candidate is normalized to its canonical atom list (Phase D4): an
    explicit `composition` if present, else the legacy tools+tool_kwargs.
    A pipeline with >1 atom gets a singleton ablation per atom (pipeline
    ABC yields ABC, A, B, C), each preserving that atom's own kwargs.
    Dedupes by atom-identity (tool + kwargs) across candidates so we
    don't run the same singleton twice when two pipelines share an atom.
    Every emitted variant carries BOTH the legacy tools/tool_kwargs view
    AND a `composition` list so the executor has one code path."""
    variants = []
    seen = set()
    for cand in candidates:
        atoms = _composition.normalize_composition(cand)
        atoms = [a for a in atoms if a.tool in T.TOOL_REGISTRY]
        if not atoms:
            continue
        key = _atoms_key(atoms)
        if key not in seen:
            seen.add(key)
            tools, tkw = _composition.composition_to_legacy(atoms)
            variants.append({
                **cand,
                "tools": tools,
                "tool_kwargs": tkw,
                "composition": [a.to_dict() for a in atoms],
                "origin": "agent3", "parent": None,
            })
        if len(atoms) > 1:
            for a in atoms:
                akey = _atoms_key([a])
                if akey in seen:
                    continue
                seen.add(akey)
                a_tools, a_tkw = _composition.composition_to_legacy([a])
                variants.append({
                    "tools": a_tools,
                    "tool_kwargs": a_tkw,
                    "composition": [a.to_dict()],
                    "reason": (f"ablation: {a.tool} alone (from pipeline "
                                f"{_composition.render_composition(atoms)})"),
                    "stance": "ablation",
                    "confidence": float(cand.get("confidence", 0.5)) * 0.7,
                    "origin": "ablation",
                    "parent": [x.tool for x in atoms],
                })
    return variants


def run_execution_engineer_tools(plan: Dict,
                                    source_model: nn.Module,
                                    target_images: List[np.ndarray],
                                    task_config,
                                    knowledge: Dict,
                                    device: str = "cuda:0",
                                    llm_client=None,
                                    model_assessment: Optional[Dict] = None,
                                    num_classes: int = 2,
                                    vlm_final_k: int = 5,
                                    vlm_tournament_k: int = 3,
                                    fewshot_indices: Optional[List[int]] = None,
                                    fewshot_labels: Optional[List[np.ndarray]] = None,
                                    task=None,
                                    ) -> Dict:
    """Run the candidate tournament + final gate. Returns a result dict
    that mirrors the v1 execution_report fields so existing serialization
    keeps working."""
    print("\n" + "=" * 55)
    print("AGENT 4 v2: Tournament Dispatcher + Final Gate")
    print("=" * 55)

    candidates = list(plan.get("candidates") or [])
    skip = bool(plan.get("skip_adaptation", False)) or not candidates
    vlm_ctx = _build_vlm_task_ctx(knowledge or {}, model_assessment or {})

    if skip:
        print("  plan.skip_adaptation=True or empty candidates — no-op")
        ps = _PipelineState(copy.deepcopy(source_model))
        fc = _run_final_check(
            ps, source_model, target_images, device,
            llm_client, vlm_ctx, knowledge or {},
            model_assessment=model_assessment, vlm_final_k=vlm_final_k,
            task=task)
        return _wrap_result(ps, fc, [], None, [], skip=True)

    # 1. Expand candidates with chain ablations: each chain with >1 tool
    #    yields its singleton variants automatically. Lets the VLM
    #    tournament test "tool alone" against "preprocessing → tool" without
    #    Agent 3 needing to explicitly nominate the singleton.
    variants = _build_variants(candidates)
    # Auto-inject a supervised-gradient variant when K labels are
    # available and Agent 3 did not include any *_ss tool. The
    # tournament's GT-Dice ranking then picks it only if it actually
    # outperforms the agent's proposals on the K labeled samples.
    if fewshot_indices and fewshot_labels:
        has_ss = any(
            any("_ss" in t for t in (v.get("tools") or []))
            for v in variants)
        if not has_ss:
            variants.append({
                "tools": ["tent_ss"],
                "tool_kwargs": {"tent_ss": {}},
                "reason": (
                    "auto-injected: K labels available; supervised "
                    "gradient signal worth comparing against Agent 3's "
                    "pure-TTA proposals"),
                "stance": "auto_ss",
                "confidence": 0.5,
                "origin": "auto_ss_inject",
                "parent": None,
            })
            print(f"  auto-injected tent_ss (K labels available, "
                   f"agent3 had no *_ss tool)")
    print(f"\n  Tournament pool: {len(candidates)} agent3 candidates "
           f"-> {len(variants)} variants after ablation")
    states = []
    for i, cand in enumerate(variants):
        origin_note = (f" (from {cand['parent']})"
                        if cand.get("parent") else "")
        print(f"\n  [var {i}] stance={cand.get('stance','?')}  "
              f"tools={cand['tools']}{origin_note}")
        st = _apply_chain(
            source_model, target_images, device, num_classes,
            cand["tools"], cand.get("tool_kwargs", {}),
            fewshot_indices=fewshot_indices,
            fewshot_labels=fewshot_labels,
            task=task,
            composition=cand.get("composition"))
        st["candidate"] = cand
        st["confidence"] = cand.get("confidence", 0.5)
        states.append(st)
        print(f"    applied {st['applied']}  ({st['elapsed_s']:.1f}s)")
        for sk in st.get("skipped") or []:
            print(f"      [skip] {sk.get('name')}: "
                   f"{str(sk.get('reason'))[:160]}")

    # 2. Tournament — VLM votes each candidate vs source
    print(f"\n  Tournament: VLM voting on {vlm_tournament_k} anchors per "
           f"variant ({len(states)} variants)")
    states = _tournament_vote(
        states, source_model, target_images, device,
        llm_client, vlm_ctx, k_anchors=vlm_tournament_k,
        fewshot_indices=fewshot_indices,
        fewshot_labels=fewshot_labels,
        num_classes=num_classes,
        task=task)
    for i, s in enumerate(states):
        n_better = sum(1 for v in s["votes_rel"] if v == "better")
        n_worse = sum(1 for v in s["votes_rel"] if v == "worse")
        n_broken = sum(1 for v in s["votes_abs"] if v == "broken")
        print(f"    [var {i}] vlm_score={s['vlm_score']:+.2f}  "
              f"better={n_better}  worse={n_worse}  broken={n_broken}")

    # 3. Pick winner. If even the best is <= 0 (everyone tied at
    #    no-improvement OR clearly worse vs source), short-circuit
    #    to source instead of wasting the gate-then-revert dance.
    winner = _pick_winner(states)
    winner_idx = states.index(winner)
    print(f"\n  Tournament winner: var[{winner_idx}]  "
           f"tools={winner['applied']}  vlm_score={winner['vlm_score']:+.2f}")

    if winner["vlm_score"] <= 0:
        print(f"  No variant improved over source (best vlm_score "
               f"{winner['vlm_score']:+.2f}) — keeping source")
        ps = _PipelineState(copy.deepcopy(source_model))
        fc = _run_final_check(
            ps, source_model, target_images, device,
            llm_client, vlm_ctx, knowledge or {},
            model_assessment=model_assessment, vlm_final_k=vlm_final_k,
            task=task)
        return _wrap_result(ps, fc, [], None, states,
                              tournament_winner_idx=winner_idx,
                              tournament_forced_revert=True)

    # 4. Final gate on the winner
    ps = _PipelineState(winner["model"])
    fc = _run_final_check(
        ps, source_model, target_images, device,
        llm_client, vlm_ctx, knowledge or {},
        model_assessment=model_assessment, vlm_final_k=vlm_final_k,
        task=task)

    # 5. Few-shot override (#62). If user supplied K labels, compute
    #    Dice on source vs winner. Override gate verdict when few-shot
    #    delta is large enough to dominate VLM noise. Few-shot Dice is
    #    ground truth on a tiny held-out set; the gate's VLM signal is
    #    fuzzy. When they disagree, trust GT.
    fewshot_record = {}
    # Phase Q4: K=0 soft-VLM-gate override. When no labels are
    # available, the gate's binary accept/reject is the only signal —
    # and it's empirically too conservative (chaos K=0 v1: gate
    # rejected tent even when tent improves GT Dice 0.12 -> 0.39).
    #
    # The tournament-VLM score and final-gate-VLM verdict use the
    # same VLM with similar prompts; when they disagree the
    # disagreement IS information. We trust tournament VLM (a
    # comparative source-vs-candidate vote across multiple variants)
    # over the final gate (a single-shot judgment).
    #
    # Conditions for force-accept on K=0:
    #   (a) winner_vlm_score >= +0.33 (tournament saw real
    #       improvement, not noise — that's 2/3 of voters)
    #   (b) Agent 1 shift_level >= 2 (real distribution shift exists,
    #       so adaptation has plausible upside)
    #   (c) Agent 2 failure_type != "model_strong" (Agent 2 does not
    #       believe the source model is already fine on target —
    #       guards against mnms-style false-positive adaptation
    #       where source actually generalizes)
    # source_quality_score (model-on-source health) is NOT used:
    # a model can be 0.98 on source and 0.12 on target (chaos), so
    # source health doesn't predict target risk.
    k0_override_record = None
    if (not fewshot_indices) and fc.get("did_revert_in_final_check"):
        ma = model_assessment or {}
        ka = knowledge or {}
        ft = (ma.get("failure_type") or "").lower()
        shift = ka.get("shift_level")
        vlm = float(winner.get("vlm_score", 0.0))
        eligible = (
            vlm >= 0.33
            and shift is not None and int(shift) >= 2
            and ft not in ("model_strong", "near_source", "")
        )
        if eligible:
            print(f"    ✓ K=0 soft override: vlm={vlm:+.2f}, "
                   f"shift_level={shift}, failure_type={ft} — forcing "
                   f"accept (tournament VLM saw improvement; trusting "
                   f"it over final-gate VLM)")
            ps.adapted_model.load_state_dict(
                winner["model"].state_dict(), strict=False)
            fc["did_revert_in_final_check"] = False
            fc["final_verdict"] = "accept"
            fc["final_reason"] = (
                f"K=0 soft override (Phase Q4): tournament "
                f"vlm_score={vlm:+.2f} >= 0.33, shift_level={shift} "
                f">= 2, failure_type={ft} — accepted despite final-"
                f"gate reject because tournament VLM (comparative, "
                f"multi-variant) is a more reliable signal than the "
                f"final gate (single-shot judgment).")
            k0_override_record = {
                "vlm_score": vlm, "failure_type": ft,
                "shift_level": shift, "decision": "force_accept",
            }
        else:
            # Log near-miss for debugging
            print(f"    [K=0 soft gate] vlm={vlm:+.2f}, "
                   f"failure_type={ft}, shift={shift} — not eligible "
                   f"for force-accept (need vlm>=0.33, shift>=2, "
                   f"failure_type not in [model_strong, near_source])")

    if fewshot_indices and fewshot_labels:
        # Note: ps.adapted_model now holds source weights if gate reverted
        winner_model = winner["model"]
        src_dice = _fewshot_dice(source_model, target_images,
                                    fewshot_indices, fewshot_labels,
                                    num_classes, device, task=task)
        win_dice = _fewshot_dice(winner_model, target_images,
                                    fewshot_indices, fewshot_labels,
                                    num_classes, device, task=task)
        delta = win_dice - src_dice
        fewshot_record = {
            "n_labels": len(fewshot_indices),
            "indices": list(fewshot_indices),
            "src_dice": src_dice,
            "winner_dice": win_dice,
            "delta": delta,
            "override": None,
        }
        gate_revert = bool(fc.get("did_revert_in_final_check", False))
        if delta < FEWSHOT_REJECT_THR and not gate_revert:
            # Force revert. Gate said accept but few-shot proves regression.
            print(f"    ⛔ Few-shot override: delta={delta:+.4f} < "
                   f"{FEWSHOT_REJECT_THR} — forcing revert despite gate=accept")
            ps.adapted_model.load_state_dict(
                source_model.state_dict(), strict=False)
            fc["did_revert_in_final_check"] = True
            fc["final_verdict"] = "reject"
            fc["final_reason"] = (
                f"few-shot override: K={len(fewshot_indices)} labeled "
                f"samples show delta={delta:+.4f} (< {FEWSHOT_REJECT_THR}); "
                f"VLM gate had said accept")
            fewshot_record["override"] = "force_revert"
        elif delta > FEWSHOT_ACCEPT_THR and gate_revert:
            # Force accept. Gate reverted but few-shot proves real win.
            print(f"    ✓ Few-shot override: delta={delta:+.4f} > "
                   f"{FEWSHOT_ACCEPT_THR} — forcing accept despite gate=reject")
            ps.adapted_model.load_state_dict(
                winner_model.state_dict(), strict=False)
            fc["did_revert_in_final_check"] = False
            fc["final_verdict"] = "accept"
            fc["final_reason"] = (
                f"few-shot override: K={len(fewshot_indices)} labeled "
                f"samples show delta={delta:+.4f} (> {FEWSHOT_ACCEPT_THR}); "
                f"VLM gate had said reject")
            fewshot_record["override"] = "force_accept"
        else:
            fewshot_record["override"] = "no_change"
            print(f"    Few-shot delta={delta:+.4f}: keep gate decision")

    result = _wrap_result(ps, fc, winner["applied"], winner["candidate"],
                            states, tournament_winner_idx=winner_idx)
    if k0_override_record:
        result["execution_report"]["k0_soft_override"] = k0_override_record
    if fewshot_record:
        result["execution_report"]["fewshot"] = fewshot_record
    return result


def _wrap_result(ps, fc, applied: List[str],
                  winner_candidate: Optional[Dict],
                  states: List[Dict],
                  skip: bool = False,
                  tournament_winner_idx: Optional[int] = None,
                  tournament_forced_revert: bool = False) -> Dict:
    final_verdict = fc["final_verdict"]
    did_revert = bool(fc.get("did_revert_in_final_check", False)) \
                  or tournament_forced_revert
    # Strip non-serialisable model objects out of the candidate states
    # before stuffing them into execution_report.
    summarized_states = []
    for s in states:
        summarized_states.append({
            "tools": s["candidate"].get("tools", []),
            "composition": s["candidate"].get("composition"),
            "stance": s["candidate"].get("stance"),
            "origin": s["candidate"].get("origin"),
            "parent_chain": s["candidate"].get("parent"),
            "fewshot_dice": s.get("fewshot_dice"),
            "confidence": s.get("confidence"),
            "applied": s.get("applied", []),
            "skipped": s.get("skipped", []),
            "elapsed_s": s.get("elapsed_s"),
            "vlm_score": s.get("vlm_score"),
            "votes_rel": s.get("votes_rel", []),
            "votes_abs": s.get("votes_abs", []),
            "vlm_note": s.get("vlm_note", ""),
            "tournament_anchors": s.get("tournament_anchors"),
        })
    execution_report = {
        "status": "completed",
        "stopped_reason": (
            "tournament forced source revert" if tournament_forced_revert
            else (fc["final_reason"] if did_revert
                   else f"applied chain {applied} without regression")),
        "trajectory": [],
        "did_revert": did_revert,
        "final_verdict": final_verdict,
        "final_reason": fc["final_reason"],
        "final_votes": fc["final_votes"],
        "final_votes_absolute": fc.get("final_votes_absolute", []),
        "final_note": fc.get("final_note", ""),
        "quality_before_final": fc["quality_before_final"],
        "quality_after_final": fc["quality_after_final"],
        "tools_applied": applied,
        "tournament_winner_idx": tournament_winner_idx,
        "tournament_forced_revert": tournament_forced_revert,
        "candidates": summarized_states,
    }
    return {
        "skipped": skip,
        "did_revert": did_revert,
        "iteration_history": ps.iteration_history,
        "quality_scores": [],
        "execution_report": execution_report,
        "final_vlm_score": (winner_candidate or {}).get("confidence"),
        "final_quality_score": fc["quality_after_final"].get("consistency"),
        "final_method": "tournament_winner" if not skip else "skip",
        "elapsed_seconds": None,
        "plan_used": {
            "tools": applied,
            "winner": winner_candidate or {},
            "n_candidates": len(states),
            "skip_adaptation": skip,
        },
        "adapted_model": ps.adapted_model,
    }
