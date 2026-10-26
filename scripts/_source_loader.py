"""
Shared source-model loader for the standalone runner scripts.

Dispatches on a DATASETS entry's `arch` field (defaulting to `"unet"` so
every existing entry keeps working without modification). Adding a new
model family is a purely additive change here — just extend the if-chain
below with the new architecture.

Used by: run_agent2_only, run_agent3_only, run_agent4_only,
         compare_source_vs_adapted.
"""

import os
import torch


def load_source_model(ckpt_rel: str, num_classes: int, device: str,
                        arch: str = "unet", project_root: str = None):
    """Load a source segmentation model from a checkpoint path.

    Args:
        ckpt_rel:    Path to the checkpoint, relative to `project_root` when
                     `project_root` is provided; otherwise absolute.
        num_classes: Number of output classes for the segmentation head.
        device:      "cuda:0", "cpu", etc.
        arch:        Architecture tag. Currently supported:
                       "unet"   — medadaptagent.models.unet.UNet (BatchNorm)
                       "stunet" — medadaptagent.models.stunet_wrapper.STUNetLite
                                  (InstanceNorm 2D, residual variant)
                       "nnunet" — medadaptagent.models.nnunet2d.NNUNet2D
                                  (InstanceNorm 2D, vanilla UNet topology,
                                   LeakyReLU, no residuals — nnU-Net v1/v2
                                   conventions)
                       "segformer" — medadaptagent.models.segformer_wrapper
                                  .SegFormer2D (MiT transformer encoder,
                                   LayerNorm; all-MLP decode head with a
                                   single BatchNorm2d — first 2D transformer
                                   SEGMENTER; HF logits upsampled to full res)
                     Defaults to "unet" so existing callers keep working.
        project_root: Optional; if provided, `ckpt_rel` is joined against it.

    Returns:
        The loaded model on `device`, in eval mode.
    """
    if arch == "stunet":
        from medadaptagent.models.stunet_wrapper import STUNetLite
        model = STUNetLite(in_channels=3, num_classes=num_classes,
                            base_ch=32, spatial_dims=2).to(device)
    elif arch == "nnunet":
        from medadaptagent.models.nnunet2d import NNUNet2D
        model = NNUNet2D(in_channels=3, num_classes=num_classes,
                          base_ch=32).to(device)
    elif arch == "unet":
        from medadaptagent.models.unet import UNet
        model = UNet(in_channels=3, num_classes=num_classes, base_ch=64).to(device)
    elif arch == "resnet18":
        from torchvision.models import resnet18
        import torch.nn as _nn
        model = resnet18(weights=None)
        model.fc = _nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
    elif arch == "resnet50":
        from torchvision.models import resnet50
        import torch.nn as _nn
        model = resnet50(weights=None)
        model.fc = _nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
    elif arch == "vit_small":
        import timm
        model = timm.create_model("vit_small_patch16_224",
                                    pretrained=False,
                                    num_classes=num_classes).to(device)
    elif arch == "segformer":
        # SegFormer (MiT transformer encoder + all-MLP decode head) — the
        # FIRST 2D *transformer* SEGMENTER in the harness. The wrapper
        # upsamples HF's H/4 logits back to full HxW so it satisfies the
        # exact UNet forward contract (B,nc,H,W). Norm composition is 30x
        # LayerNorm (encoder) + 1x BatchNorm2d (decode head): bn_reset hits
        # only the single decode-head BN, tent/sar/eata act on LN affine.
        import os as _os
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from ckpt
        from medadaptagent.models.segformer_wrapper import SegFormer2D
        model = SegFormer2D(num_classes=num_classes,
                            pretrained=False).to(device)
    elif arch == "monai_unet3d":
        # MONAI Bundle 3D UNet (BN, ResUNet variant). Standard config
        # for the spleen / liver / pancreas bundles; channels and
        # strides parameterized so multiple bundles share this arch.
        from monai.networks.nets import UNet
        # Defaults match spleen_ct_segmentation v0.6.1
        channels = [16, 32, 64, 128, 256]
        strides = [2, 2, 2, 2]
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=num_classes,
            channels=channels, strides=strides,
            num_res_units=2, norm="batch",
        ).to(device)
    elif arch == "monai_resnet3d" or arch == "monai_resnet18_3d":
        # MONAI 3D ResNet18 classifier (1-channel input, image-level cls).
        # Default config matches the PI-CAI csPCa training script:
        # spatial_dims=3, n_input_channels=1, num_classes from caller.
        from monai.networks.nets import resnet18 as monai_resnet18
        model = monai_resnet18(spatial_dims=3, n_input_channels=1,
                                  num_classes=num_classes).to(device)
    elif arch == "monai_segresnet3d":
        # MONAI 3D SegResNet (InstanceNorm/GroupNorm — no BatchNorm, so
        # bn_reset/tent are structural no-ops, consistent with the other
        # SegResNet 3D adapters). Arch hyperparameters (init_filters,
        # blocks_down, blocks_up) are read from the checkpoint dict when
        # present so a single tag reloads any SegResNet trained by
        # scripts/liver_tumor_codes/train_liver_segresnet3d.py.
        from monai.networks.nets import SegResNet
        _ckpt_path = (os.path.join(project_root, ckpt_rel)
                       if project_root else ckpt_rel)
        _ck = torch.load(_ckpt_path, map_location=device, weights_only=False)
        _init = _ck.get("init_filters", 16) if isinstance(_ck, dict) else 16
        _bd = tuple(_ck.get("blocks_down", [1, 2, 2, 4])) if isinstance(_ck, dict) else (1, 2, 2, 4)
        _bu = tuple(_ck.get("blocks_up", [1, 1, 1])) if isinstance(_ck, dict) else (1, 1, 1)
        model = SegResNet(
            spatial_dims=3, in_channels=1, out_channels=num_classes,
            init_filters=_init, blocks_down=_bd, blocks_up=_bu,
            dropout_prob=0.2,
        ).to(device)
        _state = _ck.get("model_state_dict", _ck) if isinstance(_ck, dict) else _ck
        model.load_state_dict(_state, strict=False)
        model.eval()
        return model
    elif arch == "monai_bundle":
        # Bundle-agnostic loader. Instantiates network_def directly from
        # the bundle's own inference config — works for UNet, SegResNet,
        # DiNTS, or any future arch published in the MONAI model zoo.
        # Optional sidecar field `class_extract` (list of channel indices)
        # wraps the model to expose only those classes from a larger output.
        ckpt_path = (os.path.join(project_root, ckpt_rel)
                      if project_root else ckpt_rel)
        return _load_monai_bundle(ckpt_path, num_classes, device)
    else:
        raise ValueError(f"Unknown arch '{arch}' — extend scripts/_source_loader.py")

    ckpt_path = (os.path.join(project_root, ckpt_rel)
                  if project_root else ckpt_rel)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = (ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
             if isinstance(ckpt, dict) else ckpt)
    incompat = model.load_state_dict(state, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint is incompatible with {arch}: "
            f"missing={incompat.missing_keys[:8]}, "
            f"unexpected={incompat.unexpected_keys[:8]}"
        )
    model.eval()
    return model


class _ClassExtractor(torch.nn.Module):
    """Wraps a model whose output has M channels, exposes only the K
    requested channels as the new output. The first channel is treated
    as background (all-other), built from per-voxel max of the unselected
    foreground channels — keeps softmax/argmax semantics sensible."""

    def __init__(self, base, fg_indices):
        super().__init__()
        self.base = base
        self.fg_indices = list(fg_indices)

    def forward(self, x):
        out = self.base(x)
        fg = out[:, self.fg_indices, ...]
        # background = max over the OTHER channels (not in fg_indices)
        keep = set(self.fg_indices)
        other = [i for i in range(out.shape[1]) if i not in keep]
        if other:
            bg = out[:, other, ...].max(dim=1, keepdim=True).values
        else:
            bg = -fg.max(dim=1, keepdim=True).values
        return torch.cat([bg, fg], dim=1)


def _load_monai_bundle(ckpt_path: str, num_classes: int, device: str):
    """Instantiate a model from a MONAI bundle directory.

    `ckpt_path` points at `<bundle>/models/model.pt`. We derive the bundle
    root, read its inference config, instantiate `network_def` via
    ConfigParser, load the state dict, and optionally wrap with a
    class-extractor if the bundle sidecar declares `class_extract`."""
    import json
    from monai.bundle import ConfigParser

    bundle_dir = os.path.dirname(os.path.dirname(ckpt_path))
    cfg_dir = os.path.join(bundle_dir, "configs")
    cfg_path = None
    for cfg_name in ("inference.json", "inference.yaml"):
        cand = os.path.join(cfg_dir, cfg_name)
        if os.path.exists(cand):
            cfg_path = cand
            break
    if cfg_path is None:
        raise FileNotFoundError(
            f"No inference.json/yaml under {cfg_dir} — "
            "bundle is not a standard MONAI bundle.")

    parser = ConfigParser()
    parser.read_config(cfg_path)
    parser["bundle_root"] = bundle_dir
    parser["device"] = device
    # Defang the bundle's own checkpointloader — we load weights below.
    if "load_pretrain" in parser.config:
        parser["load_pretrain"] = False

    # PyTorch 2.6+ defaults torch.load(weights_only=True), which
    # rejects NAS bundles (DiNTS) whose arch_ckpt pickles numpy
    # arrays. Temporarily monkeypatch torch.load to weights_only=False
    # while the parser resolves bundle configs — restored after.
    _orig_torch_load = torch.load
    def _safe_torch_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_torch_load(*a, **kw)
    torch.load = _safe_torch_load
    try:
        network = parser.get_parsed_content("network_def")
    finally:
        torch.load = _orig_torch_load
    network = network.to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict",
                       ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    network.load_state_dict(state, strict=False)
    network.eval()

    # Optional class-extract wrapper from sidecar
    sidecar_path = os.path.join(os.path.dirname(ckpt_path), "model.meta.json")
    if os.path.exists(sidecar_path):
        sidecar = json.load(open(sidecar_path))
        ce = sidecar.get("class_extract")
        if ce:
            fg_indices = ce if isinstance(ce, list) else ce.get("indices", [])
            if fg_indices:
                network = _ClassExtractor(network, fg_indices).to(device)
                network.eval()

    return network
