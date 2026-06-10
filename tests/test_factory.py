"""Smoke tests for the model factory."""
from __future__ import annotations

import pytest
import torch

from lightning_medseg3d.models.factory import build_model


@pytest.mark.parametrize(
    "arch",
    [
        "unet",
        "vnet",
        "resunet",
        "unetpp",
        "attention_unet",
        "unetr",
        "swin_unetr",
        "medformer",
        "segformer",
    ],
)
def test_build_model_forward_shape(arch: str) -> None:
    model = build_model(architecture=arch, in_channels=1, num_classes=2, roi_size=(32, 32, 32))
    model.eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[-1]  # BasicUNetPlusPlus / MONAI return tuples
    assert out.shape[0] == 1
    assert out.shape[1] == 2
    assert out.shape[2:] == (32, 32, 32)
