# Public-model deployment experiments

This runner adapts published MONAI segmentation models to a local target cohort, selects a model using a fixed annotation split, and checks the delivered predictor after reload.

The five model/target combinations are defined in `scripts/run_public_model_repair.py`. They cover the MONAI spleen U-Net, BTCV SwinUNETR, and whole-body SegResNet on non-contrast CT, MSD spleen CT, and AMOS MRI. The runner preserves each bundle's intensity and spacing preprocessing and extracts its spleen output.

## Prepare the inputs

Install MedAdaptAgent in an environment with CUDA-enabled PyTorch and MONAI. Obtain the public datasets and the model bundles under their respective access and license terms. Arrange paired NIfTI images and labels and the original bundle checkpoints at the paths listed in `MODELS` in the runner. `MEDADAPT_PROJECT_ROOT` can point to a separate directory containing those inputs and the installed source tree.

Each run uses 30 target volumes. Five-label runs reserve three scans for training and two for selection; twenty-label runs use thirteen and seven. The same ten separate test scans are used at both budgets. Test labels are evaluated after the terminal decision. Source-reference images and labels are also required by the configured loader. Use the published configuration to reproduce these experiments; a new dataset needs its own explicit loader and label mapping.

## Run a configuration

The seven-configuration public-model benchmark uses Azure-hosted GPT-5.2. The [four-planner comparison](PLANNERS.md) also evaluates GPT-5.6 Luna and two open-weight Qwen planners, with GPT-5.2 image analysis and critique shared across all four conditions. Set `AZURE_OPENAI_ENDPOINT`, `AZURE_API_VERSION`, and `AZURE_OPENAI_API_KEY` in your private environment. A cluster requiring an HTTP proxy also needs `HTTPS_PROXY`. Keep credentials outside repository files and shell history.

```bash
export MEDADAPT_API_LEDGER="$PWD/experiments/api_budget.json"
python scripts/run_public_model_repair.py wholebody_segresnet \
  --k-labels 5 --out-root experiments/public_models/run_01
python scripts/run_public_model_repair.py swinunetr_amos_mri \
  --k-labels 5 --out-root experiments/public_models/run_01
python scripts/run_public_model_repair.py spleen_unet_amos_mri \
  --k-labels 20 --out-root experiments/public_models/run_01
```

Each command needs one GPU. Use a fresh output root for retries. The shared API ledger refuses a request that would exceed its $10 conservative allowance. It reserves request costs before sending, records returned token counts, and retains the reservation for unsuccessful requests. Its accounting rates are deliberately above the provider's published list prices. GPU allocation time is controlled by your scheduler; the reference batch runs impose a 40-minute limit per configuration.

## Read the outputs

`terminal_decision.json` records the selected action and annotation split before export or test evaluation. `selected_state_dict.pt` preserves the chosen weights. A completed run adds `decision_chain.json`, per-case test scores, and `reload_verification.json`.

The delivered `eager_package/` contains the selected model and preprocessing function. The explicit Python format preserves dynamically constructed adapters. Verification compares the original and freshly loaded predictor on all ten test scans, requiring exact masks. `export_diagnostics.json` separately records repeat-inference stability and TorchScript conversion differences. A failed verification remains an unsuccessful attempt, with its diagnostics retained.

## Predict with a delivered package

Use the same Python, PyTorch, MONAI, and MedAdaptAgent versions as the producing run. Serialized Python packages can execute code, so load only packages you trust.

```bash
python -m harness.seg3d_delivery \
  experiments/public_models/run_01/wholebody_segresnet/eager_package \
  /path/to/scan.nii.gz /path/to/mask.npy --trusted --device cuda:0
```

The default mask uses the preprocessed canonical RAS grid. `--original-grid` nearest-resizes it to the canonical input shape using the experiment's shape-ratio rule. The accompanying `.geometry.json` states the coordinate convention. Native file orientation is not restored. Keep weights, data, policy logs, and prediction artifacts outside Git; the release distributes code and summarized evidence.

## Compare planners on one task

The planner can be changed independently of the image-analysis and critique model. For example, run the same U-Net-to-AMOS MRI task with a local planner:

```bash
export MEDADAPT_LLM_BASE_URL=http://127.0.0.1:11434/v1
python scripts/run_public_model_repair.py spleen_unet_amos_mri \
  --k-labels 20 --out-root experiments/planners/qwen4b \
  --planner-backend local --planner-model qwen3-vl:4b-instruct
```

Omitting the two planner options uses GPT-5.2 for all LLM roles. With these options, GPT-5.2 still supplies image analysis and critique, so the Azure configuration remains required. The data split, action library, prompt construction, and model-selection rules stay fixed. Each result records both model identities. Loopback planner requests incur no cloud API charge; cloud analysis remains subject to the shared ledger.
