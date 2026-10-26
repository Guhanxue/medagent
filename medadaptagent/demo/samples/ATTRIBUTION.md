# Public X-ray example attribution

These nine images are a small subset of **PneumoniaMNIST, MedMNIST+ (128 × 128)**, from the official [Zenodo record, version 3.0](https://doi.org/10.5281/zenodo.10519652). The authors distribute PneumoniaMNIST under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). That dataset license applies to the included images independently of this repository's source-code licensing.

**Credit:** Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data (2023). [Paper](https://doi.org/10.1038/s41597-022-01721-8). Also see Yang, Shi, and Ni, *MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis*, ISBI (2021).

**Original image source:** Kermany et al., *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*. [Dataset](https://doi.org/10.17632/rscbjbr9sj.2).

**Selection and changes:** training indices 0–5 are reference images; test indices 0–2 are selectable examples (all indices zero-based). Stored image pixels and the three displayed test labels are unchanged from the official 128-pixel archive. At runtime the demo scales pixels to [0,1] and applies the selected simulated brightness transformation. Display previews are enlarged with nearest-neighbor interpolation. Classifier inputs follow the repository's 224-pixel resize and normalization. No new hospital/site labels are inferred. The test labels appear only as reference metadata, not as adaptation inputs.

`provenance.json` records exact source URLs, indices, original archive checksums and the bundled subset checksum. The demo uses an untrained classifier and a scripted policy; its probabilities are not diagnostic predictions or reported research results.
