"""Scaling-law + k-fold few-shot evaluation (design Component B).

Two outputs, kept SEPARATE on purpose:

1. THE GATE (safety/ship): a low-variance estimate of the ACTUAL shipped artifact — a
   k-fold CROSS-VALIDATION at the FULL revealed-K budget. Each fold fine-tunes on ~(k-1)/k
   of the K labels and evaluates on the rotating held-out 1/k. The mean reflects the real
   fine-tune procedure at K labels and is variance-reduced by rotating the held-out set —
   which is exactly what the single tiny-holdout estimate could not do (brats: one 4-slice
   holdout read 0.46, the rotating-fold mean recovers ~0.74). This is the ONLY value that
   gates delivery, and it is the value written to `labeled_dice` for the shipped candidate.

2. THE CURVE (advisory only): a small-k sweep (train on k' < K labels) fitted to a
   saturating curve → "annotate N → expect X ± e". NOT a gate; a deliverable for the
   researcher's annotation-budget decision.

Pure numpy + optional scipy; the model/metric/finetune are INJECTED so this file imports no
torch and no adapter (testable with synthetic callables).
"""
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import math
import numpy as np


# --------------------------------------------------------------------------- #
@dataclass
class CurvePoint:
    k: int
    mean: float
    std: float


@dataclass
class ScalingCurve:
    form: str                 # 'saturating_exp' | 'power' | 'linear' | 'flat' | 'insufficient'
    formula: str
    params: dict
    r2: float
    asymptote: float


@dataclass
class ScalingReport:
    metric_name: str
    best_mode: str                    # bare mode; candidate name = 'finetune_'+best_mode
    shipped_candidate: str
    shipped_eval_cv_mean: float       # full-K k-fold CV  <-- the GATE value
    shipped_eval_cv_std: float
    source_metric: float              # source CV mean on the SAME folds
    beats_source: bool                # shipped_eval_cv_mean - source >= margin (ONLY gate)
    margin: float
    points: List[CurvePoint]          # small-k sweep (advisory)
    curve: ScalingCurve
    needed_for_usable: Optional[int]
    plan: str
    paired_note: str                  # HONESTY annotation only — NOT a gate
    k_labels_used: int
    trust_level: str = "label_verified"

    def to_evidence(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "shipped_candidate": self.shipped_candidate,
            "shipped_eval_cv_mean": round(self.shipped_eval_cv_mean, 4),
            "shipped_eval_cv_std": round(self.shipped_eval_cv_std, 4),
            "source_metric": round(self.source_metric, 4),
            "beats_source": bool(self.beats_source),
            "margin": round(self.margin, 4),
            "measured": [{"k": p.k, "mean": round(p.mean, 4), "std": round(p.std, 4)}
                         for p in self.points],
            "fit": {"form": self.curve.form, "formula": self.curve.formula,
                    "params": {k: round(float(v), 5) for k, v in self.curve.params.items()},
                    "r2": round(self.curve.r2, 3), "asymptote": round(self.curve.asymptote, 4)},
            "needed_for_usable": self.needed_for_usable,
            "plan": self.plan,
            "paired_note": self.paired_note,
        }


# --------------------------------------------------------------------------- #
def default_schedule(K: int) -> List[int]:
    """Short geometric ladder of training sizes strictly below K (advisory curve)."""
    if K < 4:
        return [max(2, K - 1)] if K >= 3 else []
    sched, k = [], 2
    while k < K:
        sched.append(k)
        k = max(k + 1, int(round(k * 1.8)))
    return sorted(set(sched))


def _folds(K: int, n_folds: int, seed: int):
    n = max(2, min(n_folds, K))
    rng = np.random.default_rng(seed)
    order = rng.permutation(K)
    parts = np.array_split(order, n)
    out = []
    for i in range(n):
        val = list(parts[i])
        tr = [int(x) for j, p in enumerate(parts) if j != i for x in p]
        if tr and val:
            out.append(([int(x) for x in val], tr))   # (val_idx, train_idx)
    return out


def _mean_std(xs):
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    return float(a.mean()), float(a.std())


# --------------------------------------------------------------------------- #
def _fit_curve(points: List[CurvePoint]) -> tuple:
    """Fit metric-vs-#labels. Saturating-exp first, fallbacks power→linear→flat. Returns
    (ScalingCurve, predict(n)->metric, labels_to_reach(target)->Optional[int])."""
    xs = np.array([p.k for p in points], dtype=float)
    ys = np.array([p.mean for p in points], dtype=float)
    if len(xs) < 3:
        y = float(ys[-1]) if len(ys) else 0.0
        c = ScalingCurve("insufficient", f"y≈{y:.3f} (constant; <3 points)", {"y": y}, 0.0, y)
        return c, (lambda n: y), (lambda t: None)

    def r2(yhat):
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2)) or 1e-9
        return 1.0 - ss_res / ss_tot

    # saturating exponential y = a - b*exp(-c*x), a=asymptote in (0,1]
    try:
        from scipy.optimize import curve_fit
        def sat(x, a, b, c):
            return a - b * np.exp(-c * x)
        a0 = min(1.0, float(ys.max()) + 0.05)
        p0 = [a0, max(1e-3, a0 - float(ys.min())), 0.2]
        popt, _ = curve_fit(sat, xs, ys, p0=p0, maxfev=20000,
                            bounds=([0.0, 0.0, 1e-4], [1.05, 2.0, 5.0]))
        a, b, c = [float(v) for v in popt]
        yhat = sat(xs, a, b, c)
        rr = r2(yhat)
        if rr >= 0.5:
            predict = lambda n: float(a - b * math.exp(-c * max(0.0, n)))
            def reach(t):
                if t >= a:
                    return None                       # asymptote below target
                val = (a - t) / b
                if val <= 0:
                    return 1
                n = -math.log(val) / c
                return int(max(1, math.ceil(n)))
            cur = ScalingCurve("saturating_exp", f"y = {a:.3f} - {b:.3f}*exp(-{c:.3f}*n)",
                               {"a": a, "b": b, "c": c}, rr, a)
            return cur, predict, reach
    except Exception:
        pass

    # power fallback y = a * x^b (fit in log space), then linear, then flat
    try:
        mask = (xs > 0) & (ys > 0)
        if mask.sum() >= 3:
            lx, ly = np.log(xs[mask]), np.log(ys[mask])
            b, la = np.polyfit(lx, ly, 1)
            a = math.exp(la)
            yhat = a * np.power(xs, b)
            rr = r2(yhat)
            if rr >= 0.4:
                predict = lambda n: float(a * (max(1.0, n) ** b))
                def reach(t):
                    if a <= 0 or t <= 0:
                        return None
                    n = (t / a) ** (1.0 / b)
                    return int(max(1, math.ceil(n))) if math.isfinite(n) else None
                cur = ScalingCurve("power", f"y = {a:.3f}*n^{b:.3f}", {"a": a, "b": b}, rr, float("nan"))
                return cur, predict, reach
    except Exception:
        pass

    # linear y = m*x + c
    m, c0 = np.polyfit(xs, ys, 1)
    yhat = m * xs + c0
    rr = r2(yhat)
    if abs(m) > 1e-4 and rr >= 0.3:
        predict = lambda n: float(m * n + c0)
        def reach(t):
            if m <= 0:
                return None
            return int(max(1, math.ceil((t - c0) / m)))
        cur = ScalingCurve("linear", f"y = {m:.4f}*n + {c0:.3f}", {"m": m, "c": c0}, rr, float("nan"))
        return cur, predict, reach

    y = float(ys.mean())
    cur = ScalingCurve("flat", f"y≈{y:.3f} (no trend)", {"y": y}, max(0.0, rr), y)
    return cur, (lambda n: y), (lambda t: None)


# --------------------------------------------------------------------------- #
def run_scaling_eval(*, source_model, imgs, lbls, fit_metric: Callable, finetune_fn: Callable,
                     safety_margin_fn: Callable, modes=("head", "partial"),
                     k_schedule=None, n_folds=3, metric_name="Dice", usable_floor=None,
                     seed=0, max_finetunes=24,
                     pooled_score_fn: Optional[Callable] = None,
                     pooled_metric_fn: Optional[Callable] = None) -> ScalingReport:
    """fit_metric(model, eval_imgs, eval_lbls)->float ; finetune_fn(mode, tr_imgs, tr_lbls)->model.
    GATE = full-K k-fold CV of the best fine-tune mode vs source on the SAME folds.

    Two GATE flavours, switched on the metric:
      • per-fold mean (default, e.g. Dice): each fold yields ONE metric; the GATE is the
        rotating-fold MEAN. Fine for PER-SAMPLE metrics (Dice is per-image, meaningful at
        fold-size 1).
      • POOLED out-of-fold (when `pooled_score_fn`+`pooled_metric_fn` are supplied, e.g.
        AUROC): per-fold AUROC over 1–2 held-out samples is DEGENERATE (reads a fake 1.0 /
        is undefined). Instead collect each fold's PER-SAMPLE held-out scores, POOL all K
        out-of-fold predictions, and compute ONE metric over ALL K. Far less degenerate —
        won't read a fake 1.0 off a 2-sample holdout. This is the cls verification fix.
        `pooled_score_fn(model, eval_imgs)->per-sample scores (len==#eval)`;
        `pooled_metric_fn(scores, labels)->float`."""
    K = len(imgs)
    folds = _folds(K, n_folds, seed)
    pooled = pooled_score_fn is not None and pooled_metric_fn is not None
    def sub(idx):
        return [imgs[i] for i in idx], [lbls[i] for i in idx]

    # ---- POOLED out-of-fold GATE (AUROC-safe; one metric over ALL K held-out preds) ----
    def _pooled_oof(make_model):
        """Rotate folds; gather per-sample held-out scores into a single pooled vector
        indexed by the ORIGINAL sample id, then score ONCE over all K. `make_model(tr_idx)`
        returns the model to evaluate on that fold's held-out val (None == source: no train)."""
        scores = [None] * K
        ylab = np.array([0.0] * K, dtype=float)
        for v, tr in folds:
            m = make_model(tr)
            vi, _ = sub(v)
            s = np.asarray(pooled_score_fn(m, vi), dtype=float).reshape(-1)
            for j, idx in enumerate(v):
                scores[idx] = float(s[j]) if j < len(s) else None
                ylab[idx] = float(int(lbls[idx]))
        sv = [(scores[i], ylab[i]) for i in range(K) if scores[i] is not None]
        if not sv:
            return 0.0
        ss = np.array([a for a, _ in sv], dtype=float)
        yy = np.array([b for _, b in sv], dtype=float)
        return float(pooled_metric_fn(ss, yy))

    # source baseline (unbiased — source never trains on labels)
    if pooled:
        src_cv = _pooled_oof(lambda tr: source_model)
    else:
        src_cv, _ = _mean_std([fit_metric(source_model, *sub(v)) for v, _ in folds])

    # GATE: full-K k-fold CV per mode (each fold trains on ~(n-1)/n of K)
    n_ft = 0
    mode_cv = {}
    for mode in modes:
        if pooled:
            if n_ft + len(folds) > max_finetunes and mode_cv:
                break
            cache = {}
            def _mk(tr, _mode=mode):
                m = finetune_fn(_mode, *sub(tr)); cache.setdefault("n", 0); cache["n"] += 1
                return m
            mode_cv[mode] = (_pooled_oof(_mk), 0.0)   # pooled GATE: std N/A (single metric)
            n_ft += cache.get("n", len(folds))
        else:
            vals = []
            for v, tr in folds:
                if n_ft >= max_finetunes:
                    break
                m = finetune_fn(mode, *sub(tr)); n_ft += 1
                vals.append(fit_metric(m, *sub(v)))
            if vals:
                mode_cv[mode] = _mean_std(vals)
    if not mode_cv:                                   # nothing trained (budget) -> source-only
        mode_cv = {modes[0]: (src_cv, 0.0)}
    best_mode = max(mode_cv, key=lambda m: mode_cv[m][0])
    shipped_mean, shipped_std = mode_cv[best_mode]
    margin = float(safety_margin_fn(src_cv))
    beats = (shipped_mean - src_cv) >= margin

    # ADVISORY small-k curve (best_mode), bounded by max_finetunes
    points = []
    rng = np.random.default_rng(seed + 1)
    for k in (k_schedule or default_schedule(K)):
        kvals = []
        for v, tr in folds:
            if n_ft >= max_finetunes:
                break
            if k > len(tr):
                continue
            picks = list(rng.permutation(tr))[:k]
            m = finetune_fn(best_mode, *sub(picks)); n_ft += 1
            kvals.append(fit_metric(m, *sub(v)))
        if kvals:
            mu, sd = _mean_std(kvals)
            points.append(CurvePoint(k=k, mean=mu, std=sd))
    # anchor the curve with the full-K CV point
    full_k = int(round((len(folds[0][1])) if folds else K))
    points.append(CurvePoint(k=full_k, mean=shipped_mean, std=shipped_std))
    points = sorted({p.k: p for p in points}.values(), key=lambda p: p.k)

    curve, predict, reach = _fit_curve(points)
    needed = reach(float(usable_floor)) if usable_floor is not None else None

    # plan string (advisory) with error bars
    pm = f"{metric_name} {shipped_mean:.3f} ± {shipped_std:.3f}"
    plan = f"At K={K} labels: {pm} (k-fold CV)."
    if usable_floor is not None:
        if needed is None:
            plan += (f" Curve ({curve.form}) does not reach the usable floor {usable_floor:.2f} "
                     f"by extrapolation — a better-suited model is likely needed.")
        elif needed <= K:
            plan += f" Already at/above the usable floor {usable_floor:.2f}."
        else:
            plan += f" To reach usable {usable_floor:.2f}: annotate ~{needed} labels ({curve.form} fit)."

    # HONESTY annotation (NOT a gate): per-fold sign agreement
    signs = []
    paired_note = ""
    if pooled:
        # POOLED OOF gives ONE metric over all K (no per-fold std); flag tiny-K AUROC noise
        gain = shipped_mean - src_cv
        paired_note = (f"pooled out-of-fold {metric_name} over K={K} held-out preds "
                       f"(single estimate; no per-fold std). gain {gain:+.3f} vs source. "
                       + ("clears safety margin"
                          if gain >= margin else "within margin — keep source")
                       + f". NOTE: {metric_name} at K={K} is noisy even pooled; "
                         f"increase K for a tighter verification.")
    elif len(folds) >= 2 and best_mode in mode_cv:
        # cheap re-check: how many folds had the best-mode model beat source on that fold
        # (reuse is not exact here; annotate the spread instead)
        paired_note = (f"per-fold std {shipped_std:.3f}; "
                       + ("consistent gain" if shipped_mean - src_cv > 2 * (shipped_std + 1e-6)
                          else "gain within fold noise — treat as estimate"))

    return ScalingReport(
        metric_name=metric_name, best_mode=best_mode,
        shipped_candidate="finetune_" + best_mode,
        shipped_eval_cv_mean=shipped_mean, shipped_eval_cv_std=shipped_std,
        source_metric=src_cv, beats_source=beats, margin=margin,
        points=points, curve=curve, needed_for_usable=needed, plan=plan,
        paired_note=paired_note, k_labels_used=K)
