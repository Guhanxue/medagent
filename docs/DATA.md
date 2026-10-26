# Model and data inputs

The release does not distribute medical images or pretrained checkpoints. Supply a checkpoint and data you are authorized to use. Model and serialized inference files must come from a trusted source.

## 2D segmentation

A directory of per-case `.npz` files is supported:

```text
data/source/case001.npz
data/target/case001.npz
```

Each file contains `image` (numeric H×W, H×W×3, or 3×H×W array) and optionally `label` (integer H×W class mask). Alternatively, use separate `images/` and `labels/` subdirectories with matching filenames. Gray images are expanded to three channels by the current 2D loader. Preprocess and scale images consistently with the source checkpoint; the loader does not infer arbitrary acquisition conventions.

A UNet checkpoint can store its state dictionary under `model_state_dict`, `state_dict`, or `model`. The included UNet loader expects three input channels and base width 64. Other recognized families have their own fixed architecture expectations in `scripts/_source_loader.py`.

An optional `unet.meta.json` next to `unet.pt` can declare:

```json
{
  "arch": "unet",
  "source_data_path": "data/source",
  "structures": {"background": 0, "liver": 1}
}
```

Use `--source-data` to avoid relying on a relative sidecar path. Include the anatomy/task in the request. The source data provide reference statistics. Target labels are optional for zero-label runs; few-shot selection/adaptation requires corresponding labels. Verify label semantics and case independence before using a label budget. The software does not deduplicate patients for you.

## 2D classification

Provide one numeric `.npz` archive for each domain, with `images` (N×H×W×C or supported per-image arrays) and optional integer `labels` (N). Common aliases `x`/`y` are also accepted. Object arrays requiring pickle are rejected. Current image processing resizes and normalizes according to `medadaptagent/tasks/classification.py`; use checkpoints trained with compatible processing.

Add `arch` metadata to the checkpoint or sidecar for reliable architecture recognition (for example, `resnet18`), and use the expected output class count. AUROC requires an appropriate class distribution; undefined/nonfinite AUROC is reported as a failure, never replaced with accuracy.

## Delivered inference

Inside a successful delivery directory, import the generated `predict.py` and call `predict(images, device='cpu')`. Segmentation returns masks; classification returns class-probability vectors. The package must remain installed. Tested 2D artifacts reload their delivered weights without the original checkpoint. The public-model runner exports a separate 3D package; see [3D package inference](PUBLIC_MODELS.md#predict-with-a-delivered-package).

Preprocessing artifacts store their fitted inference functions in `inference_state.pkl`. Only load artifacts you created or trust; serialized Python state is executable. These functions can capture reference-image information. Keep artifacts private and out of Git. Restoration failures stop prediction rather than silently removing preprocessing.
