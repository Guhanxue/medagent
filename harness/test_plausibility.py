"""Unit test for the structural-plausibility pre-gate. Pure (stubs _stats_of via the s= arg)."""
import sys, os
_HD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HD))     # repo root -> harness.core resolves
import torch.nn as nn
from harness.seg_adapter import SegTask

P = []; F = []
def chk(n, c): (P if c else F).append(n); print(("  PASS " if c else "  FAIL ") + n)

# minimal SegTask (no data/model needed for the pure structural check)
t = SegTask(nn.Conv2d(3, 2, 1), [], [], 2, "cpu", fg_prior=0.10, task_desc="x")
t._src_stats = {"fg_frac": 0.10, "n_comp": 2, "entropy": 0.3}

print("[plausible] coherent mask passes regardless of accuracy")
ok, _ = t._structural_plausibility(None, s={"fg_frac": 0.09, "n_comp": 3})
chk("coherent low-area mask -> plausible (no quality veto)", ok)
ok, _ = t._structural_plausibility(None, s={"fg_frac": 0.10, "n_comp": 2})
chk("source-like mask -> plausible", ok)

print("[implausible] degenerate shapes vetoed")
ok, why = t._structural_plausibility(None, s={"fg_frac": 0.0, "n_comp": 0})
chk("empty/collapsed -> implausible", (not ok) and "empty" in why)
ok, why = t._structural_plausibility(None, s={"fg_frac": 0.9, "n_comp": 1})
chk("exploded foreground -> implausible", (not ok) and "exploded" in why)
ok, why = t._structural_plausibility(None, s={"fg_frac": 0.10, "n_comp": 80})
chk("scattered (80 vs source 2 comps) -> implausible", (not ok) and "fragmented" in why)

print("[relative-to-source] fragmentation bound scales with source, not a brewed constant")
t2 = SegTask(nn.Conv2d(3, 2, 1), [], [], 2, "cpu", fg_prior=0.10, task_desc="x")
t2._src_stats = {"fg_frac": 0.10, "n_comp": 30}     # a naturally multi-component source
ok, _ = t2._structural_plausibility(None, s={"fg_frac": 0.10, "n_comp": 80})
chk("80 comps OK when source already has 30 (relative bound)", ok)

print("[no _src_stats] fragmentation check skipped (no false veto)")
t3 = SegTask(nn.Conv2d(3, 2, 1), [], [], 2, "cpu", fg_prior=0.10, task_desc="x")
t3._src_stats = None
ok, _ = t3._structural_plausibility(None, s={"fg_frac": 0.10, "n_comp": 200})
chk("no source stats -> fragmentation not vetoed (fg still in band)", ok)

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
sys.exit(1 if F else 0)
