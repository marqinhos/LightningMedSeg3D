"""Vendored 3D backbones from the *Medical Liver Segmentation Toolkit*.

These modules are faithful copies of the reference PyTorch implementations
used in the original toolkit
(https://github.com/Removirt/MedicalLiverSegmentationToolKit), preserving the
exact network topologies that produced the numbers reported in the paper:

* :class:`UNet`           -- original 3D U-Net (and Res-UNet via ``block='BasicBlock'``).
* :class:`VNet`           -- V-Net with residual sub-volume blocks.
* :class:`UNetPlusPlus`   -- nested U-Net++ with dense skip pathways.
* :class:`AttentionUNet`  -- U-Net with additive attention gates on the skips.
* :class:`UNETR`          -- ViT encoder + convolutional decoder (MONAI blocks).
* :class:`SwinUNETR`      -- Swin-Transformer encoder + U-Net decoder.
* :class:`MedFormer`      -- bidirectional multi-head attention U-shaped network.
* :class:`SegFormer3D`    -- hierarchical Mix-Transformer with an all-MLP head.

Only the model definitions are vendored; training/data utilities are not.
The factory in :mod:`lightning_medseg3d.models.factory` instantiates them
with the hyper-parameters from the toolkit's BTCV configuration files.
"""
from lightning_medseg3d.models.backbones.unet import UNet
from lightning_medseg3d.models.backbones.vnet import VNet
from lightning_medseg3d.models.backbones.unetpp import UNetPlusPlus
from lightning_medseg3d.models.backbones.attention_unet import AttentionUNet
from lightning_medseg3d.models.backbones.unetr import UNETR
from lightning_medseg3d.models.backbones.swin_unetr import SwinUNETR
from lightning_medseg3d.models.backbones.medformer import MedFormer
from lightning_medseg3d.models.backbones.segformer import SegFormer3D

__all__ = [
    "UNet",
    "VNet",
    "UNetPlusPlus",
    "AttentionUNet",
    "UNETR",
    "SwinUNETR",
    "MedFormer",
    "SegFormer3D",
]
