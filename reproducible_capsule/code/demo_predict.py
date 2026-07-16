"""Self-contained reproducible demonstration for LightningMedSeg3D.

The demo needs no GPU, no external datasets and no downloaded weights. It:

1. builds every one of the nine architectures through the public factory and
   verifies each produces an output of the expected (batch, class, spatial)
   shape; and
2. runs MONAI sliding-window inference with the 3D U-Net on a synthetic
   CT-sized volume and writes a NIfTI prediction plus a JSON summary to the
   results directory.

It is intended as the "reproducible run" of a Code Ocean capsule, so it writes
to ``/results`` when that Code Ocean mount exists and falls back to a local
``results/`` folder otherwise.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from lightning_medseg3d.models.factory import build_model

ARCHITECTURES = [
    "unet", "vnet", "resunet", "unetpp", "attention_unet",
    "unetr", "swin_unetr", "medformer", "segformer",
]
# 64**3 (not 32**3): SwinUNETR's default 5x downsampling (patch_size=2, 4
# stages) reduces a 32**3 input to an exact 1x1x1 bottleneck, which torch's
# InstanceNorm3d rejects ("Expected more than 1 spatial element").
ROI = (64, 64, 64)
NUM_CLASSES = 2


def results_dir() -> Path:
    """Code Ocean mounts ``/results``; fall back to a local folder elsewhere."""
    env = os.environ.get("RESULTS_DIR")
    if env:
        out = Path(env)
    elif Path("/results").is_dir():
        out = Path("/results")
    else:
        out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_all_architectures() -> list[dict]:
    """Build each architecture and check a forward pass returns the right shape."""
    x = torch.randn(1, 1, *ROI)
    report = []
    for arch in ARCHITECTURES:
        model = build_model(
            architecture=arch, in_channels=1, num_classes=NUM_CLASSES, roi_size=ROI
        ).eval()
        with torch.no_grad():
            out = model(x)
        if isinstance(out, (list, tuple)):  # some backbones return deep-supervision tuples
            out = out[-1]
        ok = out.shape[0] == 1 and out.shape[1] == NUM_CLASSES and tuple(out.shape[2:]) == ROI
        assert ok, f"{arch}: unexpected output shape {tuple(out.shape)}"
        report.append({"architecture": arch, "output_shape": list(out.shape), "ok": bool(ok)})
        print(f"[build] {arch:<15} -> {tuple(out.shape)}  OK")
    return report


def run_inference_demo(out_dir: Path) -> dict:
    """Sliding-window inference with the 3D U-Net on a synthetic volume."""
    model = build_model(
        architecture="unet", in_channels=1, num_classes=NUM_CLASSES, roi_size=ROI
    ).eval()
    volume = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        logits = sliding_window_inference(volume, ROI, 1, model, overlap=0.25)
    prediction = logits.argmax(dim=1)[0].cpu().numpy().astype("uint8")

    nib.save(nib.Nifti1Image(prediction, np.eye(4)), str(out_dir / "demo_prediction.nii.gz"))
    return {
        "architecture": "unet",
        "input_shape": list(volume.shape),
        "logits_shape": list(logits.shape),
        "prediction_shape": list(prediction.shape),
        "unique_labels": sorted(int(v) for v in np.unique(prediction)),
        "nifti_written": "demo_prediction.nii.gz",
    }


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    out_dir = results_dir()

    print("== LightningMedSeg3D reproducible demo (CPU) ==")
    build_report = build_all_architectures()
    inference = run_inference_demo(out_dir)
    print("[infer] unet sliding-window ->", inference)

    summary = {
        "device": "cpu",
        "num_architectures_built": len(build_report),
        "architectures": build_report,
        "inference_demo": inference,
    }
    (out_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"== Done. Outputs written to {out_dir} ==")


if __name__ == "__main__":
    main()
