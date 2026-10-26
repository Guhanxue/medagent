"""MedAdaptAgent — Smart Agentic Deployment Harness (core).

A THIN, TASK-BLIND agent loop. It does NOT encode the order of work; the injected
policy (the LLM) decides what to do next and when it is done. The harness only:
  - holds run state + an auditable trace,
  - exposes injected TOOLS to the policy,
  - runs the policy's chosen tool, feeds the result back,
  - enforces four GUARDRAIL INVARIANTS the agent cannot cross:
        I1 SAFETY   ship() refuses a candidate failing task.safety_check
        I2 BUDGET   a tool call over budget is refused; exhaustion forces a safe terminal
        I3 HONESTY  every terminal carries a report w/ trust_level + confidence; a ship
                    claiming "label_verified" with zero labels used is rejected
        I4 TRACE    every step is recorded; the run always emits a decision trace

Everything domain/model/task-specific is INJECTED (policy, tools, task), so the core
has no torch / no LLM / no dataset dependency and is unit-testable on its own.
See SMART_AGENT_HARNESS_SPEC.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Protocol
import json


# ----------------------------------------------------------------------------- budget
@dataclass(frozen=True)
class Budget:
    """A multi-dimensional resource cap / spend (vector, all dims independent)."""
    compute_minutes: float = 0.0
    n_trials: int = 0
    llm_usd: float = 0.0
    k_labels: int = 0

    def __add__(self, o: "Budget") -> "Budget":
        return Budget(self.compute_minutes + o.compute_minutes, self.n_trials + o.n_trials,
                      self.llm_usd + o.llm_usd, self.k_labels + o.k_labels)

    def exceeds(self, cap: "Budget") -> bool:
        """True if ANY dimension is over the cap."""
        return (self.compute_minutes > cap.compute_minutes or self.n_trials > cap.n_trials
                or self.llm_usd > cap.llm_usd or self.k_labels > cap.k_labels)

    def as_dict(self) -> dict:
        return {"compute_minutes": self.compute_minutes, "n_trials": self.n_trials,
                "llm_usd": self.llm_usd, "k_labels": self.k_labels}


# ------------------------------------------------------------------------------- tool
@dataclass
class Tool:
    """A named capability the agent may invoke. fn(state, **args) -> result dict.
    cost is a flat Budget (or a callable args->Budget). terminal tools end the run."""
    name: str
    description: str
    fn: Callable[..., dict]
    cost: Budget = field(default_factory=Budget)
    terminal: bool = False

    def cost_of(self, args: dict) -> Budget:
        return self.cost(args) if callable(self.cost) else self.cost


# --------------------------------------------------------------------------- task port
@dataclass
class SafetyVerdict:
    safe: bool
    reason: str = ""


class TaskAdapter(Protocol):
    """The task BRAIN. The harness is generic; intelligence per task lives here.

    Crucially, each task assembles its OWN toolbox and its OWN failure analysis —
    a segmentation failure (over/under-seg, boundary) is diagnosed and fixed with
    different tools than a detection miss (localization/recall) or a regression
    drift (residual bias, uncertainty). A single global toolbox would not be smart.
    So:
      - toolbox()   -> the task-SPECIFIC tools, INCLUDING a task-specialized
                       `diagnose_*` failure-analysis tool and the adaptation
                       methods that actually apply to this task (e.g. tent/entropy
                       for seg/cls; box-NMS/conf-recal for detection; uncertainty
                       recalibration for regression — tent does NOT apply there).
      - safety_check -> the per-task "don't ship worse" gate (I1).
    The harness core only needs identity + safety_check + the assembled toolbox."""
    type: str   # classification | detection | segmentation | regression
    dims: str   # 2D | 3D
    metric_name: str

    def toolbox(self) -> list["Tool"]:
        """Task-specific tools (diagnosis + applicable adaptation + evaluate)."""
        ...

    def safety_check(self, candidate: Any, source: Any, evidence: dict) -> SafetyVerdict:
        ...


# -------------------------------------------------------------------------- run result
DECISIONS = ("ship", "keep_source", "escalate")


@dataclass
class Outcome:
    decision: str                 # ship | keep_source | escalate
    delivered: Any                # the model artifact shipped (or source kept)
    report: dict                  # clinician report (trust_level, confidence, summary, ...)
    trace: list                   # full auditable step log (I4)
    budget_spent: Budget
    steps: int
    evidence: dict = field(default_factory=dict)   # final accumulated evidence (shift, model, labels)


# ------------------------------------------------------------------------------- state
@dataclass
class HarnessState:
    goal: str
    task: TaskAdapter
    source_model: Any
    deployment: Any = None
    best_candidate: Any = None                 # arg-max-by-metric candidate so far
    last_candidate: Any = None                 # most recent candidate produced
    named_candidates: dict = field(default_factory=dict)   # method-name -> model, for ship-by-name
    best_metric: Optional[float] = None        # estimate or verified, per evidence
    evidence: dict = field(default_factory=dict)
    k_labels_used: int = 0
    budget_spent: Budget = field(default_factory=Budget)
    trace: list = field(default_factory=list)
    candidates_log: list = field(default_factory=list)   # comparison table for the policy
    step: int = 0
    feedback: Optional[str] = None             # guardrail feedback fed back to the policy
    prev_tool: Optional[str] = None            # for repeated-no-op (dithering) detection
    noop_repeats: int = 0

    def view(self, tools: dict, budget_cap: Budget) -> dict:
        """The serializable snapshot the policy (LLM) reasons over each turn."""
        remaining = Budget(
            budget_cap.compute_minutes - self.budget_spent.compute_minutes,
            budget_cap.n_trials - self.budget_spent.n_trials,
            budget_cap.llm_usd - self.budget_spent.llm_usd,
            budget_cap.k_labels - self.budget_spent.k_labels)
        return {
            "goal": self.goal,
            "task": {"type": self.task.type, "dims": self.task.dims,
                     "metric": self.task.metric_name},
            "step": self.step,
            "deployed": self.deployment is not None,
            "best_metric_estimate": self.best_metric,
            "k_labels_used": self.k_labels_used,
            "candidates_tried": self.candidates_log,
            "evidence": self.evidence,
            "budget_remaining": remaining.as_dict(),
            "tools": [{"name": t.name, "description": t.description,
                       "terminal": t.terminal} for t in tools.values()],
            "guardrail_feedback": self.feedback,
        }


# ----------------------------------------------------------------------------- harness
Policy = Callable[[dict], dict]   # view -> {"tool": name, "args": {...}}


class Harness:
    """The agent loop. Construct with injected (policy, tools, task); call run()."""

    def __init__(self, policy: Policy, tools: list[Tool], task: TaskAdapter,
                 budget_cap: Budget, max_steps: int = 50, no_progress_cap: int = 4,
                 finalize_on_no_progress: bool = False):
        self.policy = policy
        self.tools = {t.name: t for t in tools}
        self.task = task
        self.budget_cap = budget_cap
        self.max_steps = max_steps
        self.no_progress_cap = no_progress_cap
        self.finalize_on_no_progress = finalize_on_no_progress
        if "escalate" not in self.tools:
            raise ValueError("an 'escalate' terminal tool is required (safe fallback)")

    # -- I4: every step recorded --------------------------------------------------
    def _record(self, st: HarnessState, kind: str, **kw):
        st.trace.append({"step": st.step, "kind": kind, **kw})

    def _forced_escalate(self, st: HarnessState, reason: str) -> Outcome:
        self._record(st, "forced_terminal", decision="escalate", reason=reason)
        report = {"trust_level": "label_free_estimate", "confidence": 0.0,
                  "summary": f"Auto-escalated: {reason}",
                  "best_metric_estimate": st.best_metric}
        report = self._enrich("escalate", report, st, "source", None)
        return Outcome("escalate", st.source_model, report, st.trace,
                       st.budget_spent, st.step, evidence=dict(st.evidence))

    # -- I3: a terminal report must be honest ------------------------------------
    def _report_ok(self, decision: str, report: dict, st: HarnessState) -> tuple[bool, str]:
        if not isinstance(report, dict):
            return False, "terminal action requires a 'report' dict"
        tl = report.get("trust_level")
        if tl not in ("label_verified", "label_free_estimate"):
            return False, "report.trust_level must be label_verified | label_free_estimate"
        if not isinstance(report.get("confidence"), (int, float)):
            return False, "report.confidence (0..1) required"
        if decision == "ship" and tl == "label_verified" and st.k_labels_used == 0:
            return False, ("cannot claim label_verified with zero labels used — "
                           "request_labels first or report label_free_estimate")
        return True, ""

    @staticmethod
    def _disclaim(report: dict) -> dict:
        """Make a K=0 (label_free_estimate) ship's UNVERIFIED status explicit + machine-
        readable. Label-verified ships are left untouched. keep_source/escalate deliver the
        source itself, so they need no disclaimer."""
        if not isinstance(report, dict) or report.get("trust_level") != "label_free_estimate":
            return report
        report = dict(report)
        report["verified_not_worse_than_source"] = False
        report.setdefault("guarantee", "none")
        disc = ("UNVERIFIED (label-free estimate): NOT guaranteed to be >= the source model; "
                "provide a few (~6) labels to verify before clinical use.")
        summ = str(report.get("summary", ""))
        if "UNVERIFIED" not in summ:
            report["summary"] = (disc + " " + summ).strip()
        return report

    def _enrich(self, decision, report, st, chosen_name, chosen_metric):
        """Component F1: enrich the (already honesty-gated) report with the structured
        delivery report. Never raises -> falls back to the raw report."""
        try:
            try:
                from harness.delivery import build_report
            except ImportError:
                from delivery import build_report
            return build_report(decision=decision, task=self.task, state=st,
                                chosen_name=chosen_name, chosen_metric=chosen_metric,
                                base_report=report)
        except Exception:
            return report

    def run(self, goal: str, source_model: Any, deployment: Any = None) -> Outcome:
        st = HarnessState(goal=goal, task=self.task, source_model=source_model,
                          deployment=deployment)
        no_progress = 0
        finalizing = False
        finalization_attempts = 0

        while True:
            # hard backstops -> always a SAFE terminal, never a crash or infinite loop
            if st.step >= self.max_steps:
                return self._forced_escalate(st, f"max_steps ({self.max_steps}) reached")
            if st.budget_spent.exceeds(self.budget_cap):
                return self._forced_escalate(st, "budget exhausted")
            if no_progress >= self.no_progress_cap:
                if not self.finalize_on_no_progress:
                    return self._forced_escalate(st, f"no improvement in {no_progress} trials")
                if not finalizing:
                    finalizing = True
                    st.feedback = (
                        "The no-progress limit has ended experiment search. Make a terminal "
                        "decision from the existing candidate evidence: ship a candidate that "
                        "passes the release checks, keep_source, or escalate. No further "
                        "adaptation is available; the release criteria are unchanged.")
                    self._record(st, "search_closed", reason="no_progress_limit")
                if finalization_attempts >= 3:
                    return self._forced_escalate(st, "no valid terminal decision after search closed")
                finalization_attempts += 1

            st.step += 1
            available_tools = ({n: t for n, t in self.tools.items() if t.terminal}
                               if finalizing else self.tools)
            view = st.view(available_tools, self.budget_cap)
            action = self.policy(view)
            st.feedback = None  # consumed

            name = (action or {}).get("tool")
            args = (action or {}).get("args", {}) or {}
            tool = available_tools.get(name)
            if tool is None:
                st.feedback = f"unknown tool '{name}'; choose from {list(available_tools)}"
                self._record(st, "invalid_action", requested=name)
                continue

            # -- I2: budget guard (refuse over-budget calls) -------------------------
            projected = st.budget_spent + tool.cost_of(args)
            if projected.exceeds(self.budget_cap):
                st.feedback = (f"'{name}' would exceed budget "
                               f"({projected.as_dict()} > {self.budget_cap.as_dict()}); "
                               f"terminate (ship/keep_source/escalate)")
                self._record(st, "budget_refused", tool=name)
                continue

            # -- terminal tools (ship / keep_source / escalate) ----------------------
            if tool.terminal:
                report = args.get("report", {})
                # I3 honesty gate
                ok, why = self._report_ok(name, report, st)
                if not ok:
                    st.feedback = f"terminal '{name}' rejected (honesty): {why}"
                    self._record(st, "honesty_rejected", tool=name, why=why)
                    continue
                # I1 safety gate (only on ship)
                if name == "ship":
                    # resolve the candidate to ship: by NAME (the agent reasons over
                    # method names), else an explicit object, else the metric arg-max.
                    ca = args.get("candidate")
                    if isinstance(ca, str):
                        cand = st.named_candidates.get(ca, st.best_candidate)
                    elif ca is not None:
                        cand = ca
                    else:
                        cand = st.best_candidate
                    v = self.task.safety_check(cand, st.source_model, st.evidence)
                    if not v.safe:
                        st.feedback = (f"SHIP VETOED (safety): {v.reason}. "
                                       f"Pick keep_source, escalate, or adapt further.")
                        self._record(st, "safety_veto", reason=v.reason)
                        continue
                    report = self._disclaim(report)
                    chosen_name = next((n for n, m in (st.named_candidates or {}).items()
                                        if m is cand),
                                       "source" if cand is st.source_model else "adapted")
                    report = self._enrich("ship", report, st, chosen_name, st.best_metric)
                    st.budget_spent = projected
                    self._record(st, "terminal", decision="ship", report=report)
                    return Outcome("ship", cand, report, st.trace, st.budget_spent,
                                   st.step, evidence=dict(st.evidence))
                # keep_source / escalate
                report = self._enrich(name, report, st, "source", None)
                st.budget_spent = projected
                delivered = st.source_model
                self._record(st, "terminal", decision=name, report=report)
                return Outcome(name, delivered, report, st.trace, st.budget_spent,
                               st.step, evidence=dict(st.evidence))

            # -- non-terminal tool: execute, fold result into state ------------------
            try:
                result = tool.fn(st, **args) or {}
            except Exception as e:                       # a tool failing is not a crash
                st.feedback = f"tool '{name}' errored: {e}"
                self._record(st, "tool_error", tool=name, error=str(e))
                st.budget_spent = projected
                continue
            st.budget_spent = projected

            # standardized optional result keys the harness understands
            if "deployment" in result:
                st.deployment = result["deployment"]
            if "k_labels_used" in result:
                st.k_labels_used += int(result["k_labels_used"])
            if "evidence" in result:
                st.evidence.update(result["evidence"])
            # a SCORED candidate (adaptation attempt) updates the arg-max best;
            # diagnosis/evaluate (no candidate) do NOT touch best/no_progress.
            cand = result.get("candidate")
            if cand is not None:
                st.last_candidate = cand
                m = result.get("metric_estimate")
                ev = result.get("evidence", {})
                _nm = ev.get("last_method")
                if _nm:
                    st.named_candidates[str(_nm)] = cand   # enables ship-by-name
                rec = {"method": ev.get("last_method", name), "metric_estimate": m}
                for k in ("quality", "vlm_prefer"):
                    if k in ev:
                        rec[k] = ev[k]
                st.candidates_log.append(rec)
                if m is not None:
                    if st.best_metric is None or m > st.best_metric:
                        st.best_metric, st.best_candidate = m, cand
                        no_progress = 0
                    else:
                        no_progress += 1
            # loop hygiene (anti-dithering): a non-terminal tool that produces NO new
            # candidate and repeats the previous tool gives no new info -> nudge + count
            # toward no_progress so the backstop eventually forces a terminal action.
            if cand is None and name == st.prev_tool:
                st.noop_repeats += 1
                st.feedback = (f"You already called '{name}' and nothing changed — it "
                               f"returns no new information. Make progress now: run an "
                               f"adaptation, ship if quality is good, or escalate.")
                no_progress += 1
            else:
                st.noop_repeats = 0
            st.prev_tool = name
            self._record(st, "tool", tool=name, args=args,
                         result_keys=sorted(result.keys()),
                         budget_spent=st.budget_spent.as_dict())


# ------------------------------------------------------------------- assembly helper
def build_harness(policy: Policy, task: TaskAdapter, common_tools: list[Tool],
                  budget_cap: Budget, **kw) -> Harness:
    """Assemble the agent for a run: TASK-SPECIFIC tools (task.toolbox(): its own
    diagnosis + applicable adaptation methods) + the COMMON tools (load_and_deploy,
    request_labels, ship/keep_source/escalate, recall/note). The action space the
    agent reasons over is therefore DIFFERENT per task — that is where 'smart' lives.
    The harness loop + the four guardrails are identical across tasks."""
    task_tools = list(task.toolbox())
    names = {t.name for t in task_tools}
    tools = task_tools + [t for t in common_tools if t.name not in names]
    return Harness(policy, tools, task, budget_cap, **kw)
