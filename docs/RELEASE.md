# Supported workflows

MedAdaptAgent provides a local browser demo, a command-line workflow for compatible 2D segmentation and classification models, and a reference runner for public 3D segmentation models.

## Choose a starting point

- **Try the software:** the [Tiny Lab](DEMO.md) runs on CPU with attributed public X-rays or synthetic shapes. It uses untrained models and a scripted policy to demonstrate decisions and model reload.
- **Adapt your model:** the [2D workflow](DATA.md) accepts your checkpoint, source reference images, target images, and resource budget.
- **Reproduce the paper's public-model experiments:** the [3D runner](PUBLIC_MODELS.md) supplies model-specific preprocessing, fixed annotation splits, and prediction checks after reload.
- **Use an open-weight LLM:** [local inference](LOCAL_LLM.md) connects to an OpenAI-compatible server on your machine.

## Delivered models

Successful 2D runs save the selected weights and fitted inference functions. The 3D runner saves the selected Python model graph and preprocessing in an explicit eager package, preserving dynamically constructed adapters. Each reported 3D package reproduces all ten test masks after reload. Use the documented loader with matching software versions and only load packages from trusted sources.

A run that ends with `escalate` delivers no model. Invalid actions, unavailable providers, undefined metrics, and preprocessing errors produce explicit diagnostics in the local output directory.

## Software checks

Run `python scripts/check_release.py` to exercise decisions, budgets, provider handling, preprocessing, package reload, and the demo. These offline checks use synthetic inputs, attributed public samples, saved result records, and HTTP fixtures. The GitHub workflow also builds the installable package on Python 3.10 and 3.12.

## Repository contents

The repository distributes source code, summarized experiment records, and the paper PDF. Datasets, model weights, credentials, and generated inference packages remain outside Git. Final result summaries include per-case measurements, dataset splits, model provenance, and prediction checks after reload.

The original source code is released under [Apache 2.0](../LICENSE). The bundled demo samples retain their CC BY 4.0 attribution. External models and datasets remain subject to their respective licenses.
