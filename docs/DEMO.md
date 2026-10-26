# Welcome to the tiny lab 🌱

The browser demo is a quick way to see MedAdaptAgent’s software workflow with real public X-rays, without preparing your own dataset or connecting an LLM.

```sh
python -m pip install -e .
medadapt-demo
```

Open the printed local address if your browser does not open automatically. Stop the server with **Ctrl+C** in its terminal. For a different port, use `medadapt-demo --port 8766`.

## Take a little tour

1. **Choose your inputs.** Public X-rays from PneumoniaMNIST are selected by default. You can also switch back to synthetic shapes.
2. **Pick an example.** The three public examples are test indices 0, 1, and 2 from the official 128-pixel release. Six training images provide the source reference. Labels are displayed as metadata and never passed into zero-label adaptation.
3. **Move the brightness slider.** This changes the selected image's brightness. It is a simulated corruption, not evidence of a real hospital-to-hospital shift.
4. **Run the demo.** The real classification harness loads an untrained ResNet18, executes a scripted plan, tries BatchNorm recalibration, and retains the source. The synthetic option continues to use an untrained UNet.
5. **See what happened.** The public-data panels show the original X-ray, the modified image, and the actual classifier probabilities. The dataset label is marked separately as reference information. The timeline comes from the harness trace.
6. **Check the handoff.** The workflow saves and reloads a model package, then checks that its probabilities are identical. Success appears only after that check passes.
7. **Take a note home.** The JSON summary records the dataset, exact example index, attribution links, settings, timing, and reload result. Temporary model files are deleted after the run.

The classifier is untrained: its probabilities are not calibrated or diagnostically useful. The fixed source-retention decision illustrates software behavior. No training, new paper experiment, or clinical performance measurement is performed. To use the actual LLM planner, follow [local-model setup](LOCAL_LLM.md) with your own compatible model and data.

## Where the images come from

The demo bundles nine unchanged images selected from the official [PneumoniaMNIST / MedMNIST+ archive](https://doi.org/10.5281/zenodo.10519652). Yang et al. distribute this subset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), based on Kermany et al.'s chest X-ray dataset. [Full credits, selection indices and transformations](../medadaptagent/demo/samples/ATTRIBUTION.md) and [checksums](../medadaptagent/demo/samples/provenance.json) are included in the package.

Stored pixels are unchanged. The page enlarges them for display, and the classifier uses the repository's resizing/normalization. The slider's brightness transformation is applied only at runtime. Once installed, the demo works offline.

## Ready for your model?

The command builder at the bottom prepares a terminal command from the paths you enter. Those fields stay in the browser; they are not uploaded or executed by the demo server. Check [the data formats](DATA.md) before running your command.

## If something needs attention

- **Port already in use:** choose another `--port`.
- **Slow first run:** importing PyTorch and initializing the model can take longer the first time. The demo runs on CPU.
- **Run fails:** the page displays an error rather than a successful result. Check that installation completed, then run `python examples/smoke_workflow.py` in the terminal for diagnostics.
- **Remote machine:** the server listens only on `127.0.0.1`. Use an SSH tunnel to access it from your browser; do not expose this development server publicly.
