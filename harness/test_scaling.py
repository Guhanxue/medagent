"""Synthetic unit test for harness/scaling.py — no torch, no GPU. Proves the GATE uses the
full-K k-fold CV (not a small-k point), the curve recovers, labels_to_reach inverts, and a
no-learning scenario correctly reports beats_source=False."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from harness.scaling import (run_scaling_eval, default_schedule, _folds, _fit_curve, CurvePoint)

P=[]; F=[]
def chk(n,c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ")+n)

margin_fn = lambda sd: max(0.05, 0.10*float(sd or 0.0))

print("[1] schedule + folds")
chk("schedule bounded < K, geometric", default_schedule(12)==[2,4,7] and max(default_schedule(12))<12)
fs=_folds(12,3,0)
chk("3 folds, disjoint val cover, train+val=K", len(fs)==3 and all(len(v)+len(t)==12 for v,t in fs)
    and sorted([i for v,_ in fs for i in v])==list(range(12)))

print("[2] brats-like: source 0.28, partial saturates ~0.85 -> GATE (full-K CV) ~0.80, ships")
def ft(mode, tr_imgs, tr_lbls):  # 'model' = (mode, #labels trained on)
    return {"mode": mode, "k": len(tr_imgs)}
def metric(model, ev_imgs, ev_lbls):
    if model.get("source"): return 0.28
    asym = 0.85 if model["mode"]=="partial" else 0.55
    return float(asym*(1-np.exp(-0.35*model["k"])))
src={"source":True}; imgs=list(range(12)); lbls=list(range(12))
r=run_scaling_eval(source_model=src, imgs=imgs, lbls=lbls, fit_metric=metric, finetune_fn=ft,
                   safety_margin_fn=margin_fn, modes=("head","partial"), n_folds=3,
                   metric_name="Dice", usable_floor=0.70, seed=0)
print(f"   best_mode={r.best_mode} shipped_cv={r.shipped_eval_cv_mean:.3f}±{r.shipped_eval_cv_std:.3f} "
      f"source={r.source_metric:.3f} beats={r.beats_source} needed={r.needed_for_usable} curve={r.curve.form}")
chk("best_mode=partial (higher asymptote)", r.best_mode=="partial")
chk("GATE = full-K CV, ~0.80 (NOT a tiny-k 0.4)", 0.72 <= r.shipped_eval_cv_mean <= 0.86)
chk("beats_source True (0.80 >> 0.28 + margin)", r.beats_source is True)
chk("shipped_candidate name = finetune_partial", r.shipped_candidate=="finetune_partial")
chk("curve saturating, asymptote ~0.85", r.curve.form=="saturating_exp" and 0.78<=r.curve.asymptote<=0.92)
chk("needed_for_usable(0.70) ~ 5 labels", r.needed_for_usable is not None and 3<=r.needed_for_usable<=8)
chk("plan mentions annotate / usable", "annotate" in r.plan or "usable" in r.plan)
ev=r.to_evidence()
chk("evidence has needed_for_usable + fit.form (no slope_b/ci)",
    "needed_for_usable" in ev and ev["fit"]["form"]=="saturating_exp" and "slope_b" not in ev and "ci" not in ev)

print("[3] no-learning scenario: metric flat 0.30 regardless of k -> beats_source False")
def metric_flat(model, ev_imgs, ev_lbls):
    return 0.28 if model.get("source") else 0.30
r2=run_scaling_eval(source_model=src, imgs=imgs, lbls=lbls, fit_metric=metric_flat, finetune_fn=ft,
                    safety_margin_fn=margin_fn, modes=("head","partial"), n_folds=3, seed=0)
chk("flat: gain 0.02 < margin 0.05 -> beats_source False", r2.beats_source is False)

print("[4] curve inversion sanity")
pts=[CurvePoint(2,0.50,0.0),CurvePoint(5,0.72,0.0),CurvePoint(8,0.80,0.0),CurvePoint(11,0.83,0.0)]
cur,predict,reach=_fit_curve(pts)
chk("predict monotone increasing", predict(2) < predict(8) <= predict(40))
n=reach(0.78); chk("reach(0.78) in (5,12]", n is not None and 5 <= n <= 12)
chk("reach(target above asymptote) -> None", reach(0.99) is None)

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
