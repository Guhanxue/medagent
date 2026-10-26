"""AUTO-ASSESSMENT: (model_ref, data_path) -> what the model is, what the data looks like,
what the likely shift is, and which toolbox repairs fit -- with reasons.

This is the step the user described: 'I have a model doing X task and want to use it on my
dataset here' -> the harness must work the rest out itself. Nothing here needs the user to
declare arch/nc/preprocessing; the checkpoint and the data are the ground truth.

  assess(model_ref, data_path) ->
    model   : task type, architecture (INFERRED from the state_dict), n_outputs,
              norm composition, head structure
    data    : n cases, resolution, channels, intensity profile
    shift   : source-vs-target statistics when the model ships with a source profile,
              else deployment statistics (entropy / collapse) as the shift proxy
    toolbox : each repair family -> applicable? why? label cost
"""
import glob
import json
import os
import numpy as np

# ---------- C1: model probing -------------------------------------------------

_ARCH_ALIASES = {
    "segformer_2d": "segformer", "stunet_lite_2d": "stunet",
    "nnunet2d": "nnunet", "unet2d": "unet",
    "segresnet3d": "monai_segresnet3d",
}
_HEAD_SUFFIXES = (
    "out_conv.weight", "seg_outputs.weight", "decode_head.classifier.weight",
    "conv_final.2.conv.weight", "out.conv.conv.weight", "classifier.weight",
    "head.weight", "fc.weight",
)


def _unwrap_checkpoint(raw):
    """Accept the checkpoint formats produced by this repository."""
    if not isinstance(raw, dict):
        return raw, {}
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(raw.get(key), dict):
            return raw[key], {k: v for k, v in raw.items() if k != key}
    return raw, {}


def _output_head(sd):
    candidates = []
    for order, (key, weight) in enumerate(sd.items()):
        if getattr(weight, "ndim", 0) not in (2, 4, 5):
            continue
        score = next((100 - i for i, suffix in enumerate(_HEAD_SUFFIXES)
                      if key.endswith(suffix)), 0)
        candidates.append((score, order, key, weight))
    if not candidates:
        return None, None
    _, _, key, weight = max(candidates, key=lambda item: (item[0], item[1]))
    return key, weight


def infer_arch_from_state_dict(sd, metadata=None):
    """Infer model family and output size from metadata, parameter names, and shapes."""
    metadata = metadata or {}
    family = None
    for key in ("arch", "model_type", "architecture"):
        hint = str(metadata.get(key, "")).lower().strip()
        hint = _ARCH_ALIASES.get(hint, hint)
        if hint:
            family = hint
            break
    joined = " ".join(sd)
    if not family:
        if "segformer" in joined or "decode_head.linear_fuse" in joined:
            family = "segformer"
        elif "convInit.conv.weight" in joined and "conv_final" in joined:
            family = "monai_segresnet3d"
        elif "encoder_blocks.0.0.conv1" in joined and "seg_outputs.weight" in joined:
            family = "stunet"
        elif "out_conv.weight" in joined and "enc1.conv.0.weight" in joined:
            first = sd.get("enc1.conv.0.weight")
            family = "nnunet" if first is not None and int(first.shape[0]) == 32 else "unet"
        elif "model.swinViT" in joined and "model.out.conv.conv.weight" in joined:
            family = "swin_unetr_2d"  # recognized; construction not yet in shared loader
        elif "layers.0.blocks" in joined and "patch_embed" in joined:
            family = "swin_tiny"
        elif "blocks.0.attn.qkv" in joined or ("cls_token" in joined and "patch_embed" in joined):
            family = "vit_small"
        elif "pooler" in joined and "encoder.layer." in joined:
            family = "rad_dino"
        elif "layer4" in joined and "conv1.weight" in joined:
            fc = sd.get("fc.weight")
            family = "resnet50" if fc is not None and fc.shape[1] == 2048 else "resnet18"
        else:
            family = "unknown"
    _, head = _output_head(sd)
    n_outputs = int(head.shape[0]) if head is not None else None
    feature_dim = int(head.shape[1]) if head is not None and head.ndim == 2 else None
    return family, n_outputs, feature_dim


def _task_from_head(head):
    return {2: "classification", 4: "segmentation_2d",
            5: "segmentation_3d"}.get(getattr(head, "ndim", 0), "unknown")


def probe_model(model_ref, device="cpu"):
    import torch
    info = {"ref": model_ref}
    if not (os.path.exists(model_ref) and model_ref.endswith((".pt", ".pth", ".ckpt"))):
        info.update(source="hub_or_registry", arch="(resolve via hub loader)",
                    task="unknown", onboarding_status="resolver_required")
        return None, info

    raw = torch.load(model_ref, map_location="cpu", weights_only=False)
    sd, metadata = _unwrap_checkpoint(raw)
    if not isinstance(sd, dict):
        info.update(source="local_checkpoint", arch="unknown", task="unknown",
                    onboarding_status="unsupported_checkpoint",
                    load_error="checkpoint does not contain a state dict")
        return None, info

    family, n_outputs, feature_dim = infer_arch_from_state_dict(sd, metadata)
    head_key, head = _output_head(sd)
    task = _task_from_head(head)
    info.update(source="local_checkpoint", arch=family, task=task,
                n_outputs=n_outputs, feature_dim=feature_dim,
                output_head=head_key,
                output_head_dims=int(head.ndim) if head is not None else None)

    safe_meta = {k: v for k, v in metadata.items()
                 if isinstance(v, (str, int, float, bool, list, tuple, dict))}
    sidecar = model_ref.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar) as handle:
                safe_meta.update(json.load(handle))
        except Exception as exc:
            info["meta_warning"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    if safe_meta:
        info["meta"] = safe_meta

    model = None
    if family == "unknown" or not n_outputs:
        info["onboarding_status"] = "unsupported_architecture"
    else:
        try:
            if task.startswith("segmentation"):
                from scripts._source_loader import load_source_model
                model = load_source_model(model_ref, n_outputs, device, arch=family)
            elif task == "classification":
                from harness.cls_adapter import load_cls_model
                model = load_cls_model(model_ref, family, n_outputs, device)
            else:
                raise ValueError(f"cannot determine task from output head {head_key!r}")
        except Exception as exc:
            info["onboarding_status"] = "recognized_but_not_loadable"
            info["load_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"

    if model is not None:
        import torch.nn as nn
        info["norms"] = {
            "BatchNorm": sum(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                             for m in model.modules()),
            "LayerNorm": sum(isinstance(m, nn.LayerNorm) for m in model.modules()),
            "InstanceNorm": sum(isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d,
                                                nn.InstanceNorm3d)) for m in model.modules()),
            "GroupNorm": sum(isinstance(m, nn.GroupNorm) for m in model.modules()),
        }
        info["has_linear_head"] = any(isinstance(m, nn.Linear) for m in model.modules())
        info["has_conv_head"] = any(
            isinstance(m, (nn.Conv2d, nn.Conv3d))
            and getattr(m, "out_channels", None) == n_outputs
            and tuple(getattr(m, "kernel_size", ())) in ((1, 1), (1, 1, 1))
            for m in model.modules()
        )
        info["onboarding_status"] = "model_ready"
    return model, info

# ---------- C2: data profiling ------------------------------------------------

def _channel_last(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[0] < image.shape[1]:
        image = image.transpose(1, 2, 0)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image.astype(np.float32, copy=False)


def _sample_values(array, limit=250_000):
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    if len(values) > limit:
        values = values[np.linspace(0, len(values) - 1, limit, dtype=int)]
    return values[np.isfinite(values)]


def _intensity_profile(arrays):
    values = [_sample_values(array) for array in arrays]
    values = [value for value in values if len(value)]
    if not values:
        return None
    values = np.concatenate(values)
    return {"mean": round(float(values.mean()), 2),
            "std": round(float(values.std()), 2),
            "p5": round(float(np.percentile(values, 5)), 1),
            "p95": round(float(np.percentile(values, 95)), 1)}


def _load_npz_image(path):
    archive = np.load(path, allow_pickle=True)
    for key in ("image", "images", "arr_0"):
        if key in archive.files:
            return archive[key], ("label" in archive.files or "labels" in archive.files)
    for key in archive.files:
        if key not in ("label", "labels"):
            return archive[key], ("label" in archive.files or "labels" in archive.files)
    raise ValueError(f"no image array in {path}")


def _image_files(path):
    patterns = ("*.npz", "*.npy", "*.nii", "*.nii.gz")
    return sorted(sum((glob.glob(os.path.join(path, pattern)) for pattern in patterns), []))


def _directory_layout(path):
    for child, label_child, layout in (
            ("images", "labels", "npz_split_dirs"),
            ("imagesTr", "labelsTr", "nifti_decathlon"),
            ("imagesTs", "labelsTs", "nifti_test_dirs")):
        image_dir = os.path.join(path, child)
        if os.path.isdir(image_dir):
            return _image_files(image_dir), os.path.isdir(os.path.join(path, label_child)), layout
    base = os.path.basename(os.path.normpath(path))
    if base in ("images", "imagesTr", "imagesTs"):
        label_base = {"images": "labels", "imagesTr": "labelsTr", "imagesTs": "labelsTs"}[base]
        return (_image_files(path),
                os.path.isdir(os.path.join(os.path.dirname(path), label_base)),
                "image_directory")
    return _image_files(path), False, "case_files"


def profile_data(data_path, max_n=256):
    path = data_path
    profile = {"path": data_path}
    if os.path.isdir(path):
        for candidate in ("tgt.npz", "src.npz"):
            bundled = os.path.join(path, candidate)
            if os.path.exists(bundled):
                path = bundled
                break

    if os.path.isfile(path) and path.endswith(".npz"):
        archive = np.load(path, allow_pickle=True)
        if "images" in archive.files:
            images = archive["images"]
            indices = np.linspace(0, len(images) - 1, min(max_n, len(images)), dtype=int)
            sampled = [images[int(i)].astype(np.float32) for i in indices]
            profile.update(layout="array_npz", n_cases=len(images),
                           shape=list(images.shape[1:]),
                           spatial_dims=3 if images.ndim >= 5 else 2,
                           intensity=_intensity_profile(sampled),
                           labels_present="labels" in archive.files)
            return sampled, profile
        image, has_label = _load_npz_image(path)
        image = _channel_last(image)
        profile.update(layout="single_npz", n_cases=1, shape=list(image.shape),
                       spatial_dims=2, intensity=_intensity_profile([image]),
                       labels_present=has_label)
        return [image], profile

    if os.path.isdir(path):
        files, sibling_labels, layout = _directory_layout(path)
        if not files:
            profile["error"] = "directory contains no NPZ/NPY/NIfTI images"
            return None, profile
        is_nifti = files[0].endswith((".nii", ".nii.gz"))
        count = min(max_n, len(files), 3 if is_nifti else 64)
        indices = np.linspace(0, len(files) - 1, count, dtype=int)
        arrays, embedded_labels, full_shape = [], False, None
        for index in indices:
            item = files[int(index)]
            if item.endswith(".npz"):
                image, has_label = _load_npz_image(item)
                arrays.append(_channel_last(image))
                embedded_labels = embedded_labels or has_label
            elif item.endswith(".npy"):
                arrays.append(_channel_last(np.load(item)))
            else:
                import nibabel as nib
                nii = nib.load(item)
                full_shape = full_shape or list(nii.shape)
                # Sample the proxy on a coarse grid; do not load whole clinical volumes.
                steps = tuple(max(1, int(np.ceil(size / 96))) for size in nii.shape)
                slices = tuple(slice(None, None, step) for step in steps)
                arrays.append(np.asarray(nii.dataobj[slices], dtype=np.float32))
        profile.update(layout=layout, n_cases=len(files),
                       shape=full_shape or list(arrays[0].shape),
                       spatial_dims=3 if is_nifti else 2,
                       intensity=_intensity_profile(arrays),
                       labels_present=bool(sibling_labels or embedded_labels),
                       sampled_cases=len(arrays),
                       file_type="nifti" if is_nifti else "numpy")
        return arrays, profile

    profile["error"] = "unsupported layout (expected NPZ/NPY/NIfTI file or directory)"
    return None, profile

# ---------- shift + toolbox fit -----------------------------------------------

def _classification_deployment(model, images, device):
    import torch
    from harness.cls_adapter import _cls_tensor
    probabilities = []
    with torch.no_grad():
        for image in images[:64]:
            logits = model(_cls_tensor(image, device))
            logits = logits.logits if hasattr(logits, "logits") else logits
            probabilities.append(torch.softmax(logits.float(), 1)[0].cpu().numpy())
    probabilities = np.asarray(probabilities)
    entropy = float((-probabilities * np.log(probabilities + 1e-9)).sum(1).mean())
    max_fraction = float(np.bincount(
        probabilities.argmax(1), minlength=probabilities.shape[1]).max() / len(probabilities))
    return {"mean_entropy": round(entropy, 3),
            "max_class_frac": round(max_fraction, 3),
            "reading": ("OUTPUT COLLAPSED to one class" if max_fraction > 0.9 else
                        "high-entropy predictions" if entropy > 0.6 else
                        "confident, distributed predictions")}


def _segmentation_2d_deployment(model, images, device):
    from harness.seg_adapter import _predict_stats
    stats = _predict_stats(model, images, device, n=min(24, len(images)))
    degenerate = stats["fg_frac"] < 1e-4 or stats["fg_frac"] > 0.95
    return {"mean_entropy": round(stats["entropy"], 3),
            "foreground_fraction": round(stats["fg_frac"], 4),
            "mean_components": round(stats["n_comp"], 2),
            "reading": ("DEGENERATE MASKS" if degenerate else
                        "non-degenerate masks; verify quality before release")}


def assess(model_ref, data_path, device="cpu", source_data=None):
    model, model_info = probe_model(model_ref, device)
    images, data_info = profile_data(data_path)
    report = {"model": model_info, "data": data_info}
    shift, source_images = {}, None

    if source_data and os.path.exists(source_data):
        source_images, source_profile = profile_data(source_data)
        shift["source_profile"] = source_profile.get("intensity")
    elif model_info.get("meta", {}).get("src"):
        candidate = model_info["meta"]["src"]
        if os.path.exists(candidate):
            source_images, source_profile = profile_data(candidate)
            shift["source_profile"] = source_profile.get("intensity")

    if source_images is not None and data_info.get("intensity") and shift.get("source_profile"):
        source_intensity, target_intensity = shift["source_profile"], data_info["intensity"]
        shift["axes"] = []
        if abs(source_intensity["mean"] - target_intensity["mean"]) > 12:
            shift["axes"].append("brightness/intensity")
        if abs(source_intensity["std"] - target_intensity["std"]) > 10:
            shift["axes"].append("contrast")
        if not shift["axes"]:
            shift["axes"] = ["no gross intensity shift (appearance/stain possible)"]

    task = model_info.get("task")
    expected_dims = {"segmentation_2d": 2, "segmentation_3d": 3}.get(task)
    if expected_dims and data_info.get("spatial_dims") != expected_dims:
        shift["deployment"] = {
            "status": "input_dimensionality_mismatch",
            "reading": (f"{task} checkpoint requires {expected_dims}-D inputs, but the "
                        f"data profiler found {data_info.get(spatial_dims)}-D inputs"),
        }
    elif model is not None and images:
        try:
            if task == "classification":
                shift["deployment"] = _classification_deployment(model, images, device)
            elif task == "segmentation_2d":
                shift["deployment"] = _segmentation_2d_deployment(model, images, device)
            elif task == "segmentation_3d":
                shift["deployment"] = {
                    "status": "preprocessing_contract_required",
                    "reading": ("3-D checkpoint and volumes recognized; recover the source "
                                "intensity/spacing contract before inference"),
                }
        except Exception as exc:
            shift["deployment"] = {"status": "probe_failed",
                                   "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
    report["shift"] = shift

    norms = model_info.get("norms", {})
    batch_norms = norms.get("BatchNorm", 0)
    task = model_info.get("task", "unknown")
    final_head = bool(model_info.get("has_linear_head") or model_info.get("has_conv_head"))
    fit = [
        {"family": "input-space repair (histogram/stain/gamma/fda)",
         "applicable": True, "labels": 0,
         "why": ("applicable after validating modality intensity semantics; targets " +
                 "/".join(shift.get("axes", ["appearance"])))},
        {"family": "bn_reset / tent", "applicable": batch_norms > 0, "labels": 0,
         "why": (f"{batch_norms} BatchNorm layers -> live" if batch_norms > 0 else
                 "0 BatchNorm layers -> STRUCTURAL NO-OP, excluded")},
        {"family": "threshold recalibration", "applicable": task == "classification",
         "labels": 2, "why": "classification operating-point repair"},
        {"family": "probe_head (final linear/1x1 conv)",
         "applicable": final_head and task in ("classification", "segmentation_2d"),
         "labels": "4-12",
         "why": ("recognized final head -> cheap frozen-feature repair available" if final_head
                 else "no supported final head found")},
        {"family": "finetune head/partial/lora", "applicable": model is not None,
         "labels": "12+", "why": "last resort; verify on held-out labels"},
    ]
    report["toolbox_fit"] = fit

    deployment_status = shift.get("deployment", {}).get("status")
    if (model is None or data_info.get("error") or
            deployment_status in ("probe_failed", "input_dimensionality_mismatch")):
        report["onboarding_status"] = "not_ready"
    elif (task == "segmentation_3d" and
          shift.get("deployment", {}).get("status") == "preprocessing_contract_required"):
        report["onboarding_status"] = "model_and_data_recognized_preprocessing_required"
    else:
        report["onboarding_status"] = "ready"
    return report


def assess_request(text, llm=None, cwd=".", device="cpu", source_data=None):
    """Free text -> validated contract -> task-agnostic automatic assessment."""
    from dataclasses import asdict
    from harness.intake import parse_request, clarifying_question, confirmation

    contract = parse_request(text, llm=llm, cwd=cwd)
    envelope = {"contract": asdict(contract), "confirmation": confirmation(contract)}
    if not contract.valid():
        envelope.update(onboarding_status="needs_clarification",
                        question=clarifying_question(contract))
        return envelope

    def resolved(path):
        candidate = os.path.join(cwd, path)
        return candidate if not os.path.isabs(path) and os.path.exists(candidate) else path

    report = assess(resolved(contract.model_ref), resolved(contract.data_path),
                    device=device,
                    source_data=resolved(source_data) if source_data else None)
    report.update(envelope)
    return report


def render(report):
    model, data, shift = report["model"], report["data"], report["shift"]
    lines = [
        "== AUTO-ASSESSMENT ==",
        f"STATUS: {report.get('onboarding_status')}",
        f"MODEL : {model.get('arch', '?')} | task {model.get('task', '?')} | outputs {model.get('n_outputs')} | norms {model.get('norms')} | head {model.get('output_head')}",
        f"DATA  : n={data.get('n_cases')} layout={data.get('layout')} shape={data.get('shape')} intensity={data.get('intensity')}",
        f"SHIFT : axes={shift.get('axes', '(no source profile)')} deployment={shift.get('deployment')}",
        "TOOLBOX FIT:",
    ]
    for family in report["toolbox_fit"]:
        mark = "+" if family["applicable"] else "x"
        lines.append(f"  [{mark}] {family['family']:48s} labels={family['labels']!s:5s} {family['why']}")
    return "\n".join(lines)
