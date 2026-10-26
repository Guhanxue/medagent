#!/usr/bin/env python3
"""Recompute published summaries from saved metrics; no experiments are run."""
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent / "results"

def read(name):
    return json.loads((ROOT / name).read_text())

def main():
    seg = read("segmentation_budget.json")
    source = seg["source"]
    baseline = mean(source.values())
    print(f"Segmentation: {len(source)} transfers; source Dice {baseline:.3f}")
    for budget in (0, 1, 5, 20):
        values = seg["conditions"][f"medadapt_k{budget}"]
        if set(values) != set(source):
            raise ValueError("Segmentation source and adapted task sets differ")
        score = mean(values.values())
        harmed = sum(values[k] < source[k] - 1e-9 for k in source)
        print(f"  K={budget}: Dice {score:.3f}; harmed {harmed}/{len(source)}")
    cls = read("classification.json")
    draw = cls["reported_draw"]
    print(f"Classification reported draw: {mean(x['source'] for x in draw):.3f} -> "
          f"{mean(x['delivered'] for x in draw):.3f}; "
          f"harmed {sum(x['delivered'] < x['source'] - 1e-9 for x in draw)}/{len(draw)}")
    draws = cls["four_draw_selection"]["per_draw"]
    for name, field in (("audit_score_winner", "audit_winner_full"), ("medadaptagent", "shipped")):
        means = [mean(x[field] for x in d["choices"]) for d in draws]
        harmed = sum(x[field] < x["source"] - 1e-9 for d in draws for x in d["choices"])
        n = sum(len(d["choices"]) for d in draws)
        print(f"  {name}: {mean(means):.3f} +/- {pstdev(means):.3f}; harmed {harmed}/{n}")
    planner = read("planner_comparison.json")
    print("Planner comparison (saved aggregates, separate 11-transfer suite): "
          f"{planner['scripted_ladder']['mean']:.3f} -> {planner['harness_v4_safe']['mean']:.3f}")
    attempts = read("public_model_attempts.json")
    print(f"Public models: {sum(x['verified_package'] for x in attempts)}/{len(attempts)} verified packages")
    print("No training, inference, LLM calls, or new experiments were run.")

if __name__ == "__main__":
    main()
