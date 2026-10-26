#!/usr/bin/env python
"""
FALLBACK / also-run controlled acquisition shift on the LiTS thick-slice
TARGET sub-domain (n=47 from domain_split.json["target_eval"]).

Label-agnostic low-dose + reconstruction-kernel shift on IMAGES ONLY:
  1. additive Gaussian noise in HU  (sim low-dose photon starvation): sigma_hu
  2. mild Gaussian blur in-plane    (sim softer recon kernel):        blur_sigma

Applied AFTER RAS+Spacing resample, BEFORE intensity windowing, on the image
channel only (labels untouched). Everything else mirrors eval_gate.py exactly
so liver/tumor Dice is comparable to the clean-target reference
(clean target: liver 0.9263, tumor 0.5033 -- gate_results.json).

Sweep sigma_hu / blur_sigma to drive tumor Dice into the degraded-but-not-zero
band ~0.25-0.40 (a real recoverable gap). Report the params used.
"""

from pathlib import Path
import os, json, argparse
import numpy as np
import torch

from monai.networks.nets import SegResNet
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, AsDiscrete, MapTransform,
)
from scipy.ndimage import gaussian_filter

DATA = str(Path(os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[2]))) / 'data/3d/msd_liver/Task03_Liver')
IMG = os.path.join(DATA, "imagesTr")
LBL = os.path.join(DATA, "labelsTr")
SPLIT = str(Path(os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[2]))) / 'data/3d/msd_liver/domain_split.json')
CKPT = str(Path(os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[2]))) / 'checkpoints/segresnet_liver_source_best.pt')
PATCH = (96, 96, 96)
# clean-target reference from gate_results.json["target"]
CLEAN_LIVER = 0.9262732330788958
CLEAN_TUMOR = 0.5033009328106617


class LowDoseReconShiftd(MapTransform):
    """Add Gaussian HU noise + in-plane Gaussian blur to the image channel.

    image tensor is (C, H, W, D) post-EnsureChannelFirst+Spacing, still in HU
    (windowing happens after this in the pipeline). Blur is applied in the
    first two spatial axes (in-plane) to mimic a recon-kernel change; noise is
    isotropic HU. Deterministic per-call via a fixed seed offset by case index
    so the corruption is reproducible.
    """
    def __init__(self, keys, sigma_hu, blur_sigma, seed=0):
        super().__init__(keys)
        self.sigma_hu = float(sigma_hu)
        self.blur_sigma = float(blur_sigma)
        self.seed = seed

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            arr = np.asarray(d[k], dtype=np.float32)  # (C,H,W,D)
            rng = np.random.default_rng(self.seed)
            out = arr.copy()
            for c in range(out.shape[0]):
                vol = out[c]
                if self.blur_sigma > 0:
                    # in-plane blur only (axes 0,1), none through-slice (axis 2)
                    vol = gaussian_filter(vol, sigma=(self.blur_sigma,
                                                      self.blur_sigma, 0.0))
                if self.sigma_hu > 0:
                    vol = vol + rng.normal(0.0, self.sigma_hu, size=vol.shape
                                           ).astype(np.float32)
                out[c] = vol
            d[k] = torch.as_tensor(out)
        return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--sigma_hu", type=float, default=40.0)
    p.add_argument("--blur_sigma", type=float, default=0.7)
    p.add_argument("--out_json",
                   default=str(Path(os.environ.get("MEDADAPT_ROOT", str(Path(__file__).resolve().parents[2]))) / 'data/3d/msd_liver/lowdose_shift_eval.json'))
    a = p.parse_args()
    device = torch.device(a.device)

    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    nc = ck["num_classes"]
    inten = ck["intensity"]
    spacing = tuple(ck["spacing"])
    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=nc,
                      init_filters=ck["init_filters"],
                      blocks_down=tuple(ck["blocks_down"]),
                      blocks_up=tuple(ck["blocks_up"]),
                      dropout_prob=0.2).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"Loaded {a.ckpt}; sigma_hu={a.sigma_hu} blur_sigma={a.blur_sigma}")

    tfm = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing,
                 mode=("bilinear", "nearest")),
        # ---- controlled shift on image only, in HU, pre-windowing ----
        LowDoseReconShiftd(keys=["image"], sigma_hu=a.sigma_hu,
                           blur_sigma=a.blur_sigma, seed=1234),
        ScaleIntensityRanged(keys="image", a_min=inten["a_min"],
                             a_max=inten["a_max"], b_min=inten["b_min"],
                             b_max=inten["b_max"], clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    with open(SPLIT) as f:
        split = json.load(f)
    files = split["target_eval"]
    dicts = [{"image": os.path.join(IMG, f), "label": os.path.join(LBL, f)}
             for f in files]
    ds = Dataset(dicts, tfm)
    loader = DataLoader(ds, batch_size=1, num_workers=4)

    post_pred = AsDiscrete(argmax=True, to_onehot=nc)
    post_lbl = AsDiscrete(to_onehot=nc)

    per_liver, per_tumor = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            lbl = batch["label"].to(device)
            logits = sliding_window_inference(img, PATCH, 4, model, overlap=0.25)
            pred = post_pred(decollate_batch(logits)[0])[None]
            gt = post_lbl(decollate_batch(lbl)[0])[None]
            dm = DiceMetric(include_background=False, reduction="mean_batch")
            dm(y_pred=[pred[0]], y=[gt[0]])
            v = dm.aggregate().detach().cpu().numpy().tolist()
            per_liver.append(v[0]); per_tumor.append(v[1])

    liver_mean = float(np.mean(per_liver))
    tumor_mean = float(np.mean(per_tumor))
    print("\n========= LOW-DOSE+RECON SHIFT (LiTS target) =========")
    print(f"sigma_hu={a.sigma_hu}  blur_sigma={a.blur_sigma}  n={len(files)}")
    print(f"corrupted liver Dice : {liver_mean:.4f}  (clean {CLEAN_LIVER:.4f})")
    print(f"corrupted tumor Dice : {tumor_mean:.4f}  (clean {CLEAN_TUMOR:.4f})")
    print(f"tumor gap vs clean   : {CLEAN_TUMOR - tumor_mean:.4f}")
    print(f"tumor gap vs src ref : {0.5312618832462109 - tumor_mean:.4f}")

    out = {"sigma_hu": a.sigma_hu, "blur_sigma": a.blur_sigma,
           "n": len(files),
           "clean_liver": CLEAN_LIVER, "clean_tumor": CLEAN_TUMOR,
           "corrupted_liver_dice_mean": liver_mean,
           "corrupted_tumor_dice_mean": tumor_mean,
           "tumor_gap_vs_clean": CLEAN_TUMOR - tumor_mean,
           "tumor_gap_vs_source_ref": 0.5312618832462109 - tumor_mean,
           "liver_per_vol": [round(x, 4) for x in per_liver],
           "tumor_per_vol": [round(x, 4) for x in per_tumor]}
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {a.out_json}")


if __name__ == "__main__":
    main()
