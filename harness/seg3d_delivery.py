"""Explicit eager 3-D deployment packages, with unchanged Python inference.

Cloudpickle artifacts can execute Python during loading. Load only packages from
trusted authors; the SHA256 detects corruption, not malicious replacement.
The package contains the selected CPU model and a pure preprocessing function,
with numeric preprocessing configuration and imported helpers. Custom model attributes
remain part of the model and require inspection before sharing. The installed MedAdaptAgent,
PyTorch, and MONAI environment remains a runtime dependency.

CLI: python -m harness.seg3d_delivery PACKAGE IMAGE.nii.gz OUTPUT.npy --trusted
The default mask is on the preprocessed RAS grid. --original-grid nearest-resizes
it to the canonical RAS input shape using the experiment adapter's shape-ratio
rule. This is not an inverse affine transform into the original file orientation.
A JSON sidecar records the grid convention; .npy alone carries no spatial header.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import platform
import types
import sys
import dis

ARTIFACT_FORMAT = "python_eager_cloudpickle"
ARTIFACT_NAME = "model_and_preprocessing.pkl"
INFERENCE_SETTINGS = {
    "roi_size": [96, 96, 96],
    "sw_batch_size": 2,
    "overlap": 0.5,
    "mode": "gaussian",
    "prediction": "channel_argmax",
}
_METADATA_FIELDS = {
    "model_id", "selected_action", "architecture", "bundle", "bundle_version",
    "source_checkpoint_sha256", "code_revision", "preprocessing_name",
    "source_domain", "target_domain", "k_labels",
}


def _safe_preprocessing(fn):
    """Accept numeric configuration and imports, excluding data-bearing captures.

    This checks serialization captures; callers still need to trust the model and
    function code. Modules and importable helper functions are kept by reference.
    """
    if not isinstance(fn, types.FunctionType) or inspect.ismethod(fn):
        raise TypeError("Pass the plain preprocessing function, not a bound task method")

    def scalar_config(value):
        if value is None or isinstance(value, (bool, int, float)):
            return True
        if isinstance(value, (list, tuple)):
            return len(value) <= 32 and all(scalar_config(v) for v in value)
        return False

    for cell in fn.__closure__ or ():
        if not scalar_config(cell.cell_contents):
            raise ValueError("Preprocessing closure may capture only numeric configuration")
    if fn.__defaults__ and not scalar_config(fn.__defaults__):
        raise ValueError("Preprocessing defaults may contain only numeric configuration")
    if any(not scalar_config(v) for v in (fn.__kwdefaults__ or {}).values()):
        raise ValueError("Preprocessing keyword defaults may contain only numeric configuration")
    if fn.__annotations__:
        raise ValueError("Use a preprocessing function without serialized annotations")
    for name, value in fn.__dict__.items():
        if name not in {"_window", "_pixdim"}:
            raise ValueError("Unsupported preprocessing function attribute")
        if name == "_window" and value == "NormalizeIntensity(nonzero)+ScaleIntensity[-1,1]":
            continue
        if not scalar_config(value):
            raise ValueError("Preprocessing attributes may contain only numeric configuration")
    def global_names(code):
        names = {i.argval for i in dis.get_instructions(code) if i.opname == "LOAD_GLOBAL"}
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                names.update(global_names(value))
        return names

    for name in global_names(fn.__code__):
        if name not in fn.__globals__:
            continue  # built-in resolved at execution
        value = fn.__globals__[name]
        if isinstance(value, types.ModuleType):
            continue
        if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
            module = sys.modules.get(value.__module__)
            current = module
            for name in value.__qualname__.split("."):
                current = getattr(current, name, None)
            if module is not None and value.__module__ != "__main__" and current is value:
                continue
        raise ValueError("Preprocessing globals must be imported modules or importable helpers")


def _metadata(metadata):
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a provenance dictionary")
    if set(metadata) - _METADATA_FIELDS:
        raise ValueError("Unsupported metadata fields; only model provenance is accepted")
    if any(value is not None and not isinstance(value, (str, bool, int, float))
           for value in metadata.values()):
        raise ValueError("Provenance values must be JSON scalars")
    # Reject non-finite values and return an independent JSON-only copy.
    return json.loads(json.dumps(metadata, allow_nan=False))


def save_eager_package(model, to_tensor, output_dir, metadata):
    """Save a CPU copy without changing the selected model; return the manifest.

    `to_tensor` has signature (image, size, device), as used by
    TaskAdapter.set_source_preprocessing. Only public model provenance fields
    listed in _METADATA_FIELDS are accepted as metadata.
    """
    _safe_preprocessing(to_tensor)
    provenance = _metadata(metadata)
    import cloudpickle

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / ARTIFACT_NAME
    manifest_path = output_dir / "manifest.json"
    if artifact.exists() or manifest_path.exists():
        raise FileExistsError("An eager package already exists; choose a new output directory")
    cpu_model = copy.deepcopy(model).cpu().eval()
    # Do not serialize task, caller locals, metadata, data, or provider objects.
    payload = cloudpickle.dumps({"model": cpu_model, "to_tensor": to_tensor})
    manifest = {
        "schema_version": 1,
        "artifact_format": ARTIFACT_FORMAT,
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "inference": copy.deepcopy(INFERENCE_SETTINGS),
        "preprocessing_size": 96,
        "metadata": provenance,
        "python_version": platform.python_version(),
        "cloudpickle_version": cloudpickle.__version__,
        "output_geometry": {
            "default": "preprocessed canonical RAS grid",
            "original_grid_option": "canonical RAS input shape; nearest shape-ratio resize",
            "native_file_orientation_restored": False,
        },
    }
    # A manifest is written only after all artifact bytes are durable to this handle.
    with artifact.open("xb") as handle:
        handle.write(payload)
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def load_eager_package(directory, device="cpu", trusted=False):
    """Verify bytes before deserializing; return a fresh eval model and 3-D task."""
    if trusted is not True:
        raise PermissionError("Cloudpickle can execute code; trusted=True is required")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError("Unsupported eager package format")
    if manifest.get("artifact") != ARTIFACT_NAME:
        raise ValueError("Unexpected artifact filename")
    if manifest.get("inference") != INFERENCE_SETTINGS or manifest.get("preprocessing_size") != 96:
        raise ValueError("Package inference configuration differs from the supported 3-D pipeline")
    payload = (directory / ARTIFACT_NAME).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("artifact_sha256"):
        raise ValueError("Eager artifact SHA256 mismatch")
    import cloudpickle
    package = cloudpickle.loads(payload)
    if not isinstance(package, dict) or set(package) != {"model", "to_tensor"}:
        raise ValueError("Invalid eager package contents")
    _safe_preprocessing(package["to_tensor"])
    from medadaptagent.tasks.segmentation_3d import SegmentationTask3D
    model = package["model"].to(device).eval()
    task = SegmentationTask3D()
    task.set_source_preprocessing(
        manifest.get("metadata", {}).get("preprocessing_name", "delivered_preprocessing"),
        package["to_tensor"], 96,
    )
    return model, task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("image", type=Path, help="3-D .nii or .nii.gz input; canonicalized to RAS")
    parser.add_argument("output", type=Path, help="Output mask .npy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trusted", action="store_true", help="Trust the package's Python code")
    parser.add_argument("--original-grid", action="store_true",
                        help="Nearest-resize to canonical RAS input shape, not native file orientation")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npy":
        parser.error("output must end in .npy")
    model, task = load_eager_package(args.package, args.device, trusted=args.trusted)
    import nibabel as nib
    import numpy as np
    import torch
    from medadaptagent.calibration import Volume

    image = nib.as_closest_canonical(nib.load(str(args.image)))
    if len(image.shape) != 3:
        raise ValueError("A single 3-D volume is required")
    volume = Volume.from_array(image.get_fdata().astype(np.float32), image.affine.astype(np.float32))
    tensor = task.to_tensor(volume, args.device)
    with torch.no_grad():
        _, mask = task.forward_pred(model, tensor)
    preprocessed_shape = list(mask.shape)
    if args.original_grid and mask.shape != image.shape:
        from scipy import ndimage
        zoom = np.asarray(image.shape, dtype=np.float64) / np.asarray(mask.shape, dtype=np.float64)
        mask = ndimage.zoom(mask.astype(np.float32), zoom, order=0).astype(mask.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mask)
    geometry = {
        "mask_grid": "canonical_RAS_input_shape" if args.original_grid else "preprocessed_RAS",
        "shape": list(mask.shape),
        "preprocessed_shape": preprocessed_shape,
        "canonical_input_shape": list(image.shape),
        "canonical_input_affine": image.affine.tolist(),
        "native_file_orientation_restored": False,
        "resize": "nearest_shape_ratio" if args.original_grid else None,
        "note": "The canonical input affine describes the input, not the default preprocessed mask grid.",
    }
    args.output.with_suffix(".geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(f"Saved {geometry['mask_grid']} mask, shape {tuple(mask.shape)}")


if __name__ == "__main__":
    main()
