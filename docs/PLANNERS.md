# Comparing planners

Four planner configurations adapt the same public MONAI U-Net from CT to AMOS MRI. Each uses twenty annotations: thirteen scans for training and seven for selection. Dice is measured against expert annotations on ten separate test scans. The pretrained model scores 0.000 Dice.

| Planner | Selected adjustment | Test Dice | Exact reload masks |
| --- | --- | ---: | ---: |
| GPT-5.2 | LoRA | 0.230 | 10/10 |
| GPT-5.6 Luna | LoRA | 0.196 | 10/10 |
| Qwen3-VL-4B-Instruct | LoRA | 0.215 | 10/10 |
| Qwen3-VL-8B-Instruct | partial fine-tuning | 0.205 | 10/10 |

All four planners complete the workflow and deliver reproducible packages. GPT-5.2, Luna, and Qwen-4B choose LoRA; Qwen-8B chooses partial fine-tuning. The scores differ even when the adjustment family is the same, highlighting the role of the search and training recipe. GPT-5.2 achieves the highest test Dice on this transfer.

## What stays fixed

The checkpoint, preprocessing, annotation split, adaptation tools, four-candidate limit, selection rule, and random seed are identical. GPT-5.2 supplies image analysis and critique in every condition. There is one run per planner configuration. GPT-5.2 and Qwen use temperature zero and a 2,000-token response limit. Luna uses ChatGPT subscription access through Codex CLI 0.145.0, low reasoning effort, default sampling, and a fresh session for each decision. Responses above 2,000 output tokens are rejected after generation. Its CLI context and sampling differ from direct API serving; the result represents this stated planner configuration.

The public planners use official Qwen weights in bfloat16 with Transformers 4.57.1 on one NVIDIA H200 per run. The exact weight revisions, per-case results, and model-selection records are in [the result file](../results/llm_planner_comparison.json). The [Ollama setup example](LOCAL_LLM.md) uses a separately packaged version of the model; the measurements here use the stated Transformers configuration.

## Reproduce the task

Follow the [public-model data and runtime setup](PUBLIC_MODELS.md), then select a planner with `--planner-backend` and `--planner-model`. Point `MEDADAPT_LLM_BASE_URL` to the compatible server hosting that model. Keep the Azure analysis configuration in place. Use a fresh output directory for each planner.

```bash
python scripts/run_public_model_repair.py spleen_unet_amos_mri \
  --k-labels 20 --out-root experiments/planners/qwen4b \
  --planner-backend local --planner-model Qwen/Qwen3-VL-4B-Instruct
```

Omit the planner options for GPT-5.2, or select `Qwen/Qwen3-VL-8B-Instruct` for the larger public planner. The same model identifier must be served by the configured endpoint.

## Subscription-based Luna configuration

The Luna experiment passes each planner prompt unchanged to `codex exec -m gpt-5.6-luna` through stdin. It uses `--ignore-user-config --ephemeral --skip-git-repo-check`, a separate empty working directory, `--sandbox read-only`, `--disable shell_tool`, `project_doc_max_bytes=0`, and `model_reasoning_effort="low"`. A replacement instruction file contains: “Follow the user's instructions and return only the requested response. Do not use tools or external information.” Each call must complete with one text response and no tool events. The resulting JSON action returns to the same adaptation controller.

Codex keeps the user's sign-in credentials locally. The GPU experiment receives only planner responses. Subscription token usage is recorded separately from metered API calls for GPT-5.2 analysis and critique. The generic API runner connects directly to API-compatible providers; reproducing this subscription configuration requires the CLI callback described here.
