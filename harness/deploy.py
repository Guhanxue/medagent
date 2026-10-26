"""Public free-text entry point for supported MedAdaptAgent deployments.

This module joins the already implemented intake/onboarding layer to the
adaptation harness.  It intentionally supports only the data/model formats
that the shared loaders can reconstruct without guessing preprocessing:

* 2-D segmentation checkpoints with NPZ target/source directories;
* 2-D classification checkpoints with NPZ target/source archives.

Recognized 3-D models are returned with an explicit typed status until their
intensity, spacing, orientation, and channel semantics are supplied by a
bundle-specific loader.  No target data are silently substituted for missing
source-reference data.
"""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

import numpy as np

from harness.core import Budget, Tool, build_harness
from harness.intake import RunContract
from harness.llm_policy import make_llm_policy
from harness.onboard import assess_request, probe_model


def _resolve(path: str | None, cwd: str) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        local = Path(cwd) / candidate
        if local.exists():
            candidate = local
    return str(candidate)


def _source_reference(model_info: dict[str, Any], supplied: str | None,
                      cwd: str) -> str | None:
    if supplied:
        return _resolve(supplied, cwd)
    metadata = model_info.get("meta") if isinstance(model_info.get("meta"), dict) else {}
    for key in ("source_data", "source_data_path", "src"):
        value = metadata.get(key)
        if value:
            return _resolve(str(value), cwd)
    return None


def _foreground_prior(labels: list[Any], fallback: float = 0.05) -> float:
    fractions = []
    for label in labels[:64]:
        if label is None:
            continue
        array = np.asarray(label)
        if array.size:
            fractions.append(float((array > 0).mean()))
    return float(np.mean(fractions)) if fractions else fallback


def _structures(model_info: dict[str, Any]) -> dict[str, int]:
    metadata = model_info.get("meta") if isinstance(model_info.get("meta"), dict) else {}
    value = metadata.get("structures")
    if isinstance(value, dict) and value:
        return {str(k): int(v) for k, v in value.items()}
    nc = int(model_info.get("n_outputs") or 0)
    if nc == 2:
        return {"background": 0, "foreground": 1}
    return {"background": 0, **{f"class_{i}": i for i in range(1, nc)}}


def _terminal_tools() -> list[Tool]:
    return [
        Tool("ship", "deliver the evidence-supported adapted model",
             lambda state, **kwargs: {}, terminal=True),
        Tool("keep_source", "deliver the unmodified source model",
             lambda state, **kwargs: {}, terminal=True),
        Tool("escalate", "stop automated deployment and return the evidence trace",
             lambda state, **kwargs: {}, terminal=True),
    ]


def _build_2d_task(contract: RunContract, model, model_info: dict[str, Any],
                   target_path: str, source_path: str, device: str, llm):
    task_type = model_info.get("task")
    nc = int(model_info["n_outputs"])
    metadata = model_info.get("meta") if isinstance(model_info.get("meta"), dict) else {}

    if task_type == "segmentation_2d":
        from harness.seg_adapter import SegTask
        from scripts.run_ablations import load_npz_dataset

        target_images, target_labels = load_npz_dataset(target_path, None)
        source_images, source_labels = load_npz_dataset(source_path, 64)
        if not target_images:
            raise ValueError("target dataset contains no loadable NPZ images")
        if not source_images:
            raise ValueError("source-reference dataset contains no loadable NPZ images")
        if contract.k_labels > 0 and not any(label is not None for label in target_labels):
            raise ValueError("the request permits target labels but the target dataset has none")

        prior = float(metadata.get("foreground_prior",
                                   _foreground_prior(source_labels)))
        task = SegTask(
            model, source_images, target_images, nc, device,
            fg_prior=prior, tlbls=target_labels,
            task_desc=contract.task_question or "new-site 2-D segmentation",
        )
        structures = _structures(model_info)
        if llm is not None and structures:
            from medadaptagent.graph.agents.execution_engineer import _build_vlm_task_ctx
            from medadaptagent.tasks import get_task

            ctx = _build_vlm_task_ctx(
                {"structures": structures, "foreground_priors": {}}, {})
            ctx["num_classes_hint"] = nc
            anchors = [target_images[int(i)] for i in np.linspace(
                0, len(target_images) - 1, min(6, len(target_images)), dtype=int)]
            task.set_vlm(llm, ctx, get_task("segmentation"), anchors)
        return task, structures

    if task_type == "classification":
        from harness.cls_adapter import ClsTask, load_cls_npz

        target_images, target_labels = load_cls_npz(target_path)
        source_images, _ = load_cls_npz(source_path)
        if target_labels is None and contract.k_labels > 0:
            raise ValueError("the request permits target labels but the target archive has none")
        task = ClsTask(
            model, source_images, target_images, nc, device,
            tlabs=target_labels, metric=str(metadata.get("metric", "auroc")),
            task_desc=contract.task_question or "new-site classification",
        )
        if llm is not None:
            task.set_llm(llm)
        return task, _structures(model_info)

    raise ValueError(f"unsupported bridge task type: {task_type!r}")


def deploy_request(text: str, *, llm, cwd: str = ".", device: str = "cuda:0",
                   source_data: str | None = None,
                   output_dir: str | None = None,
                   compute_minutes: int = 60, n_trials: int = 16,
                   max_steps: int = 44, no_progress_cap: int = 20) -> dict[str, Any]:
    """Analyze, adapt, verify, and package one supported free-text request.

    The returned dictionary always contains ``run_status`` and the onboarding
    report.  Non-ready requests return a typed status without starting an
    experiment.  A successful run also contains the terminal decision, trace,
    budget usage, report, and optional artifact manifest.
    """
    assessment = assess_request(text, llm=llm, cwd=cwd, device=device,
                                source_data=source_data)
    result: dict[str, Any] = {"run_status": "onboarding_incomplete",
                             "onboarding": assessment}
    if assessment.get("onboarding_status") != "ready":
        return result

    contract = RunContract(**{
        key: value for key, value in assessment["contract"].items()
        if key in RunContract.__dataclass_fields__
    })
    model_path = _resolve(contract.model_ref, cwd)
    target_path = _resolve(contract.data_path, cwd)
    model, model_info = probe_model(model_path, device=device)
    if model is None:
        result.update(run_status="model_not_loadable", model=model_info)
        return result
    if model_info.get("task") == "segmentation_3d":
        result.update(
            run_status="preprocessing_contract_required",
            model=model_info,
            message=("The 3-D model and target volumes were recognized, but the public "
                     "bridge will not guess intensity, spacing, orientation, or output-channel "
                     "semantics. Use a registered bundle loader."),
        )
        return result

    source_path = _source_reference(model_info, source_data, cwd)
    if not source_path or not os.path.exists(source_path):
        result.update(
            run_status="source_reference_required",
            model=model_info,
            message=("A source-reference dataset or source profile is required for the "
                     "available source-relative adjustment tools. Supply source_data or "
                     "include source_data_path in the model sidecar metadata."),
        )
        return result

    try:
        task, structures = _build_2d_task(
            contract, model, model_info, target_path, source_path, device, llm)
    except Exception as exc:
        result.update(run_status="task_construction_failed", model=model_info,
                      error=f"{type(exc).__name__}: {exc}")
        return result

    task.objective = contract.objective
    budget = Budget(compute_minutes=int(compute_minutes), n_trials=int(n_trials),
                    llm_usd=5.0, k_labels=int(contract.k_labels))
    policy = make_llm_policy(lambda prompt: llm.chat(prompt))
    harness = build_harness(policy, task, _terminal_tools(), budget,
                            max_steps=max_steps, no_progress_cap=no_progress_cap)
    previous_klab = os.environ.get("KLAB")
    previous_objective = os.environ.get("OBJECTIVE")
    os.environ["KLAB"] = str(contract.k_labels)
    os.environ["OBJECTIVE"] = contract.objective
    try:
        outcome = harness.run(goal=text, source_model=model)
    finally:
        if previous_klab is None:
            os.environ.pop("KLAB", None)
        else:
            os.environ["KLAB"] = previous_klab
        if previous_objective is None:
            os.environ.pop("OBJECTIVE", None)
        else:
            os.environ["OBJECTIVE"] = previous_objective

    artifact = None
    if output_dir and outcome.decision != "escalate" and outcome.delivered is not None:
        from harness.delivery import package_artifact

        config = {
            "name": Path(model_path).stem + "_new_site",
            "arch": model_info.get("arch"),
            "nc": int(model_info["n_outputs"]),
            "ckpt": model_path,
            "tgt": target_path,
            "src": source_path,
            "struct": structures,
        }
        manifest = package_artifact(
            out=outcome, task=task, cfg=config,
            dest_dir=_resolve(output_dir, cwd), project_root=str(Path(cwd).resolve()))
        artifact = {"directory": manifest.dir, **manifest.manifest}

    result.update(
        run_status="completed",
        contract=asdict(contract),
        decision=outcome.decision,
        delivered=outcome.report.get("delivered_model") if outcome.report else None,
        report=outcome.report,
        trace=outcome.trace,
        steps=outcome.steps,
        budget_spent=outcome.budget_spent.as_dict(),
        artifact=artifact,
    )
    return result
